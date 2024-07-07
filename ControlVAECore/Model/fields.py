import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ControlVAECore.Model.embedder import get_embedder

from scipy.spatial import KDTree
from torch.utils.data.sampler import WeightedRandomSampler
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
# from ControlVAECore.Model.pointnet2_ssg_sem import PointNet2SemSegSSG

def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)

def update_quaternion(delta_angle, prev_quat):
    s1 = 0
    s2 = prev_quat[0]
    v2 = prev_quat[1:]
    v1 = delta_angle / 2
    new_v = s1 * v2 + s2 * v1 + torch.cross(v1, v2)
    new_s = s1 * s2 - torch.sum(v1 * v2)
    new_quat = torch.cat([new_s.unsqueeze(0), new_v], dim=0)
    return new_quat

# def euler_to_quaternion(yaw, pitch, roll):
def euler_to_quaternion(roll, pitch, yaw):
    qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
    qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
    qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    
    # qx = torch.sin()
    return [qw, qx, qy, qz]
    # return [qx, qy, qz, qw]
    
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1) # -1 for the quaternion matrix # 
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None: # input; input fn fine #
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)




class BendingNetworkActiveForceFieldForwardLagV18(nn.Module):
    def __init__(self, # self # 
                 d_in, # 
                 multires, # fileds #
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions,
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False,
                 nn_instances=1,
                 minn_dist_threshold=0.05,
                 ): # contact 
        # bending network active force field #
        super(BendingNetworkActiveForceFieldForwardLagV18, self).__init__()
        self.use_positionally_encoded_input = False
        self.input_ch = 3
        self.input_ch = 1
        d_in = self.input_ch
        self.output_ch = 3
        self.output_ch = 1
        self.bending_n_timesteps = bending_n_timesteps
        self.bending_latent_size = bending_latent_size
        self.use_rigidity_network = use_rigidity_network
        self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        self.rigidity_network_depth = rigidity_network_depth
        self.rigidity_use_latent = rigidity_use_latent

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64
        self.network_depth = 5
        self.contact_dist_thres = 0.1
        self.skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = True
        self.use_last_layer_bias = use_last_layer_bias
        
        self.static_friction_mu = 1.

        self.embed_fn_fine = None # embed fn and the embed fn #
        if multires > 0: 
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
        
        self.nn_uniformly_sampled_pts = 50000
        
        self.cur_window_size = 60 # get 
        self.bending_n_timesteps = self.cur_window_size + 10
        self.nn_patch_active_pts = 50
        self.nn_patch_active_pts = 1
        
        self.nn_instances = nn_instances
        
        self.contact_spring_rest_length = 2.
        
        # self.minn_dist_sampled_pts_passive_obj_thres = 0.05 # minn_dist_threshold ###
        self.minn_dist_sampled_pts_passive_obj_thres = minn_dist_threshold
        
        self.spring_contact_ks_values = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.spring_contact_ks_values.weight)
        self.spring_contact_ks_values.weight.data = self.spring_contact_ks_values.weight.data * 0.01
        
        self.spring_friction_ks_values = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.spring_friction_ks_values.weight)
        self.spring_friction_ks_values.weight.data = self.spring_friction_ks_values.weight.data * 0.001
        
        if self.nn_instances == 1:
            self.spring_ks_values = nn.Embedding(
                num_embeddings=5, embedding_dim=1
            )
            torch.nn.init.ones_(self.spring_ks_values.weight)
            self.spring_ks_values.weight.data = self.spring_ks_values.weight.data * 0.01
            self.spring_ks_values.weight.data[:, :] = 0.1395
        else:
            self.spring_ks_values = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=5, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_ks_values in self.spring_ks_values:
                torch.nn.init.ones_(cur_ks_values.weight)
                cur_ks_values.weight.data = cur_ks_values.weight.data * 0.01
        
        self.inertia_div_factor = nn.Embedding(
            num_embeddings=1, embedding_dim=1
        )
        torch.nn.init.ones_(self.inertia_div_factor.weight)
        # self.inertia_div_factor.weight.data[:, :] = 30.0
        self.inertia_div_factor.weight.data[:, :] = 20.0
        
        
        
        self.bending_latent = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        )
        
        self.bending_dir_latent = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        )
        # dist_k_a = self.distance_ks_val(torch.zeros((1,)).long().cuda()).view(1)
        # dist_k_b = self.distance_ks_val(torch.ones((1,)).long().cuda()).view(1) * 5# *#  0.1
        
        # distance
        self.distance_ks_val = nn.Embedding(
            num_embeddings=5, embedding_dim=1
        ) 
        torch.nn.init.ones_(self.distance_ks_val.weight) # distance_ks_val #
        # self.distance_ks_val.weight.data[0] = self.distance_ks_val.weight.data[0] * 0.6160 ## 
        # self.distance_ks_val.weight.data[1] = self.distance_ks_val.weight.data[1] * 4.0756 ## 
        
        self.ks_val = nn.Embedding(
            num_embeddings=2, embedding_dim=1
        )
        torch.nn.init.ones_(self.ks_val.weight)
        self.ks_val.weight.data = self.ks_val.weight.data * 0.2
        
        
        self.ks_friction_val = nn.Embedding(
            num_embeddings=2, embedding_dim=1
        )
        torch.nn.init.ones_(self.ks_friction_val.weight)
        self.ks_friction_val.weight.data = self.ks_friction_val.weight.data * 0.2
        
        
        ## [ \alpha, \beta ] ##
        if self.nn_instances == 1:
            self.ks_weights = nn.Embedding(
                num_embeddings=2, embedding_dim=1
            )
            torch.nn.init.ones_(self.ks_weights.weight) #
            self.ks_weights.weight.data[1] = self.ks_weights.weight.data[1] * (1. / (778 * 2))
        else:
            self.ks_weights = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=2, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_ks_weights in self.ks_weights:
                torch.nn.init.ones_(cur_ks_weights.weight) #
                cur_ks_weights.weight.data[1] = cur_ks_weights.weight.data[1] * (1. / (778 * 2))
        
        
        # sep_time_constant, sep_torque_time_constant, sep_damping_constant, sep_angular_damping_constant
        self.sep_time_constant = self.time_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_time_constant.weight) #
        
        self.sep_torque_time_constant = self.time_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_torque_time_constant.weight) #
        
        self.sep_damping_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_damping_constant.weight) # # # #
        # self.sep_damping_constant.weight.data = self.sep_damping_constant.weight.data * 0.9
        self.sep_damping_constant.weight.data = self.sep_damping_constant.weight.data * 0.2
        
        
        self.sep_angular_damping_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_angular_damping_constant.weight) # # # #
        # self.sep_angular_damping_constant.weight.data = self.sep_angular_damping_constant.weight.data * 0.9
        self.sep_angular_damping_constant.weight.data = self.sep_angular_damping_constant.weight.data * 0.2
        

    
        
        if self.nn_instances == 1:
            self.time_constant = nn.Embedding(
                num_embeddings=3, embedding_dim=1
            )
            torch.nn.init.ones_(self.time_constant.weight) #
        else:
            self.time_constant = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=3, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_time_constant in self.time_constant:
                torch.nn.init.ones_(cur_time_constant.weight) #
        
        if self.nn_instances == 1:
            self.damping_constant = nn.Embedding(
                num_embeddings=3, embedding_dim=1
            )
            torch.nn.init.ones_(self.damping_constant.weight) # # # #
            self.damping_constant.weight.data = self.damping_constant.weight.data * 0.9
        else:
            self.damping_constant = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=3, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_damping_constant in self.damping_constant:
                torch.nn.init.ones_(cur_damping_constant.weight) # # # #
                cur_damping_constant.weight.data = cur_damping_constant.weight.data * 0.9
        
        self.nn_actuators = 778 * 2 # vertices #
        self.nn_actuation_forces = self.nn_actuators * self.cur_window_size
        self.actuator_forces = nn.Embedding( # actuator's forces #
            num_embeddings=self.nn_actuation_forces + 10, embedding_dim=3
        )
        torch.nn.init.zeros_(self.actuator_forces.weight) # 
        
        
        
        # self.actuator_friction_forces = nn.Embedding( # actuator's forces #
        #     num_embeddings=self.nn_actuation_forces + 10, embedding_dim=3
        # )
        # torch.nn.init.zeros_(self.actuator_friction_forces.weight) # 
        
        
        if nn_instances == 1:
            self.actuator_friction_forces = nn.Embedding( # actuator's forces #
                num_embeddings=self.nn_actuation_forces + 10, embedding_dim=3
            )
            torch.nn.init.zeros_(self.actuator_friction_forces.weight) # 
        else:
            self.actuator_friction_forces = nn.ModuleList(
                [nn.Embedding( # actuator's forces #
                    num_embeddings=self.nn_actuation_forces + 10, embedding_dim=3
                ) for _ in range(self.nn_instances) ]
                )
            for cur_friction_force_net in self.actuator_friction_forces:
                torch.nn.init.zeros_(cur_friction_force_net.weight) # 
        
        # simulator #
        
        self.actuator_weights = nn.Embedding( # actuator's forces #
            num_embeddings=self.nn_actuation_forces + 10, embedding_dim=1
        )
        torch.nn.init.ones_(self.actuator_weights.weight) # 
        self.actuator_weights.weight.data = self.actuator_weights.weight.data * (1. / (778 * 2))
        
        
        ''' patch force network and the patch force scale network ''' 
        self.patch_force_network = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(3, self.hidden_dimensions), nn.ReLU()),
                nn.Sequential(nn.Linear(self.hidden_dimensions, self.hidden_dimensions)), # with maxpoll layers # 
                nn.Sequential(nn.Linear(self.hidden_dimensions * 2, self.hidden_dimensions), nn.ReLU()), # 
                nn.Sequential(nn.Linear(self.hidden_dimensions, 3)), # hidden_dimension x 1 -> the weights # 
            ]
        )
        with torch.no_grad():
            for i, layer in enumerate(self.patch_force_network[:]):
                for cc in layer:
                    if isinstance(cc, nn.Linear):
                        torch.nn.init.kaiming_uniform_(
                            cc.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        # if i == len(self.patch_force_network) - 1:
                        #     torch.nn.init.xavier_uniform_(cc.bias)
                        # else:
                        if i < len(self.patch_force_network) - 1:
                            torch.nn.init.zeros_(cc.bias)
                # torch.nn.init.zeros_(layer.bias)
        self.patch_force_scale_network = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(3, self.hidden_dimensions), nn.ReLU()),
                nn.Sequential(nn.Linear(self.hidden_dimensions, self.hidden_dimensions)), # with maxpoll layers # 
                nn.Sequential(nn.Linear(self.hidden_dimensions * 2, self.hidden_dimensions), nn.ReLU()), # 
                nn.Sequential(nn.Linear(self.hidden_dimensions, 1)), # hidden_dimension x 1 -> the weights # 
            ]
        )
        with torch.no_grad():
            for i, layer in enumerate(self.patch_force_scale_network[:]):
                for cc in layer:
                    if isinstance(cc, nn.Linear): ### ifthe lienar layer # # ## 
                        torch.nn.init.kaiming_uniform_(
                            cc.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        if i < len(self.patch_force_scale_network) - 1:
                            torch.nn.init.zeros_(cc.bias)
        ''' patch force network and the patch force scale network ''' 
        
        ''' the bending network '''
        # self.input_ch = 1
        self.network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)
            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0
                self.network[-1].bias.data += 0.2
        ''' the bending network '''

        self.dir_network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, 3)])

        with torch.no_grad():
            for i, layer in enumerate(self.dir_network[:]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

        self.friction_input_dim = 3 + 3 + 1 + 3 ### 
        self.friction_network = [
            nn.Linear(self.friction_input_dim, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, 3)
        ]
        for i_sub_net, sub_net in enumerate(self.friction_network):
            if isinstance(sub_net, nn.Linear):
                if i_sub_net < len(self.friction_network) - 1:
                    torch.nn.init.kaiming_uniform_(
                        sub_net.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(sub_net.bias)
                else:
                    torch.nn.init.zeros_(sub_net.weight)
                    torch.nn.init.zeros_(sub_net.bias)
        self.friction_network = nn.Sequential(
            *self.friction_network
        )
        
        
        self.contact_normal_force_network = [
            nn.Linear(self.friction_input_dim, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions), nn.ReLU(),
            nn.Linear(self.hidden_dimensions, 3)
        ]
        for i_sub_net, sub_net in enumerate(self.contact_normal_force_network):
            if isinstance(sub_net, nn.Linear):
                if i_sub_net < len(self.contact_normal_force_network) - 1:
                    torch.nn.init.kaiming_uniform_(
                        sub_net.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(sub_net.bias)
                else:
                    torch.nn.init.zeros_(sub_net.weight)
                    torch.nn.init.zeros_(sub_net.bias)
        self.contact_normal_force_network = nn.Sequential(
            *self.contact_normal_force_network
        )
        
        ## weighting_network for the network ##
        self.weighting_net_input_ch = 3
        self.weighting_network = nn.ModuleList(
            [nn.Linear(self.weighting_net_input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.weighting_net_input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, 1)])

        with torch.no_grad():
            for i, layer in enumerate(self.weighting_network[:]):
                if self.activation_function.__name__ == "sin": # periodict activation functions #
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    if i < len(self.weighting_network) - 1:
                        torch.nn.init.zeros_(layer.bias)
        
        # weighting model via the distance #
        # unormed_weight = k_a exp{-d * k_b} # weights # k_a; k_b #
        # distances # the kappa #
        self.weighting_model_ks = nn.Embedding( # k_a and k_b #
            num_embeddings=2, embedding_dim=1
        ) 
        torch.nn.init.ones_(self.weighting_model_ks.weight) 
        self.spring_rest_length = 2. # 
        self.spring_x_min = -2.
        self.spring_qd = nn.Embedding(
            num_embeddings=1, embedding_dim=1
        ) 
        torch.nn.init.ones_(self.spring_qd.weight) # q_d of the spring k_d model -- k_d = q_d / (x - self.spring_x_min) # 
        
        self.obj_inertia = nn.Embedding(
            num_embeddings=1, embedding_dim=3
        )
        torch.nn.init.ones_(self.obj_inertia.weight)
        
        self.optimizable_obj_mass = nn.Embedding(
            num_embeddings=1, embedding_dim=1
        )
        torch.nn.init.ones_(self.optimizable_obj_mass.weight)
        import math
        self.optimizable_obj_mass.weight.data *= math.sqrt(30)
        
        
        self.optimizable_spring_ks = nn.Embedding(
            num_embeddings=5, embedding_dim=1
        )
        torch.nn.init.ones_(self.optimizable_spring_ks.weight)
        # 400000000, 100000000 # optimizabale spring forces an the
        self.optimizable_spring_ks.weight.data[0, :] = math.sqrt(1.0)
        self.optimizable_spring_ks.weight.data[1, :] = math.sqrt(1.0)
        
        self.time_constants = nn.Embedding(
            num_embeddings=1000, embedding_dim=1
        )
        torch.nn.init.ones_(self.time_constants.weight)
        self.time_constants.weight.data[:, :] = 0.0005
        
        
        # spring_force = -k_d * \delta_x = -k_d * (x - self.spring_rest_length) #
        # 1) sample points from the active robot's mesh;
        # 2) calculate forces from sampled points to the action point;
        # 3) use the weight model to calculate weights for each sampled point;
        # 4) aggregate forces;
                
        self.timestep_to_vel = {}
        self.timestep_to_point_accs = {}
        # how to support frictions? # 
        ### TODO: initialize the t_to_total_def variable ### # tangential 
        self.timestep_to_total_def = {}
        
        self.timestep_to_input_pts = {}
        self.timestep_to_optimizable_offset = {} # record the optimizable offset #
        self.save_values = {}
        # ws_normed, defed_input_pts_sdf,  # 
        self.timestep_to_ws_normed = {}
        self.timestep_to_defed_input_pts_sdf = {}
        self.timestep_to_ori_input_pts = {}
        self.timestep_to_ori_input_pts_sdf = {}
        
        self.use_opt_rigid_translations = False # load utils and the loading .... ## 
        self.use_split_network = False
        
        self.timestep_to_prev_active_mesh_ori  = {}
        # timestep_to_prev_selected_active_mesh_ori, timestep_to_prev_selected_active_mesh # # active mesh # active mesh #
        self.timestep_to_prev_selected_active_mesh_ori = {}
        self.timestep_to_prev_selected_active_mesh = {}
        
        self.timestep_to_spring_forces = {}
        self.timestep_to_spring_forces_ori = {}
        
        # timestep_to_angular_vel, timestep_to_quaternion # 
        self.timestep_to_angular_vel = {}
        self.timestep_to_quaternion = {}
        self.timestep_to_torque = {}
        
        self.timestep_to_accum_acc = {}
        
        
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion
        self.timestep_to_optimizable_total_def = {}
        self.timestep_to_optimizable_quaternion = {}
        self.timestep_to_optimizable_rot_mtx = {}
        self.timestep_to_aggregation_weights = {}
        self.timestep_to_sampled_pts_to_passive_obj_dist = {}
        
        self.ts_to_contact_act_pts = {}
        self.ts_to_contact_passive_pts = {}

        self.time_quaternions = nn.Embedding(
            num_embeddings=60, embedding_dim=4
        )
        self.time_quaternions.weight.data[:, 0] = 1.
        self.time_quaternions.weight.data[:, 1] = 0.
        self.time_quaternions.weight.data[:, 2] = 0.
        self.time_quaternions.weight.data[:, 3] = 0.
        # torch.nn.init.ones_(self.time_quaternions.weight) #  # actuators
        
        self.time_translations = nn.Embedding( # tim
            num_embeddings=60, embedding_dim=3
        )
        torch.nn.init.zeros_(self.time_translations.weight) # 
        
        self.time_forces = nn.Embedding(
            num_embeddings=60, embedding_dim=3
        )
        torch.nn.init.zeros_(self.time_forces.weight) # 
        
        # self.time_velocities = nn.Embedding(
        #     num_embeddings=60, embedding_dim=3
        # )
        # torch.nn.init.zeros_(self.time_velocities.weight) # 
        self.time_torques = nn.Embedding(
            num_embeddings=60, embedding_dim=3
        )
        torch.nn.init.zeros_(self.time_torques.weight) # 
        
        self.obj_sdf_th = None
        self.obj_sdf_grad_th = None
        
        self.normal_plane_max_y = torch.tensor([0, 1., 0], dtype=torch.float32).cuda() ## 0, 1, 0
        self.normal_plane_min_y = torch.tensor([0, -1., 0.], dtype=torch.float32).cuda() # 
        
        self.normal_plane_max_x = torch.tensor([1, 0, 0], dtype=torch.float32).cuda() ## 0, 1, 0
        self.normal_plane_min_x = torch.tensor([-1, 0., 0.], dtype=torch.float32).cuda() # 
        
        self.normal_plane_max_z = torch.tensor([0, 0, 1.], dtype=torch.float32).cuda() ## 0, 1, 0
        self.normal_plane_min_z = torch.tensor([0, 0, -1.], dtype=torch.float32).cuda() # 
        
        ## set the initial passive object verts and normals ###
        ## the default scene is the box scene ##
        self.penetration_determining = "plane_primitives"
        self.canon_passive_obj_verts = None
        self.canon_passive_obj_normals = None
        self.train_residual_normal_forces = False
        self.only_residual = False
        
        # self.point_force_net = PointNet2SemSegSSG() ## load the point force network ##
        ##### point_type_embedding_net, point_force_net, point_force_net_out #####
        self.pts_type_emb_dim = 128
        self.point_type_embedding_net = nn.Embedding(
            num_embeddings=2, embedding_dim=self.pts_type_emb_dim, 
        )
        torch.nn.init.xavier_uniform_(self.point_type_embedding_net.weight)
        
        ### version 1 -> from point features to per-point forces -> to per-point forces and torques -> to object force and torque -> to object velocity and angular velocity ###
        self.point_force_net = nn.Sequential(
            nn.Linear(3 + self.pts_type_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 1024)
        )
        # torch.nn.init.xavier_uniform_(self.point_force_net[0].weight)
        # a max_pool #
        self.point_force_net_out = nn.Sequential(
            nn.Linear(1024 + 1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        torch.nn.init.zeros_(self.point_force_net_out[-1].weight)
        torch.nn.init.zeros_(self.point_force_net_out[-1].bias)
        
        ### to obj delta offset and delta rot angles directly ###
        # obj_kine_net_out_vel, obj_kine_net_out_anvel # 
        self.obj_kine_net_out_vel = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        torch.nn.init.zeros_(self.obj_kine_net_out_vel[-1].weight)
        torch.nn.init.zeros_(self.obj_kine_net_out_vel[-1].bias)
        
        self.obj_kine_net_out_anvel = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        torch.nn.init.zeros_(self.obj_kine_net_out_anvel[-1].weight)
        torch.nn.init.zeros_(self.obj_kine_net_out_anvel[-1].bias)
        
        self.pred_delta_offsets = False
        self.use_contact_region_pts = False
        self.use_optimizable_timecons = False
        self.nn_passive_to_act_pts_feat_dim = 3 + 3 + 3 + 3
        
        # passive_act_pts_feat_mapping_net, passive_pts_feat_mapping_net, passive_pts_force_mapping_net
        self.passive_act_pts_feat_mapping_net = nn.Sequential(
            nn.Linear(self.nn_passive_to_act_pts_feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512)
        )
        self.passive_pts_feat_mapping_net = nn.Sequential(
            nn.Linear(512 + 512, 512), nn.ReLU(),
            # nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), 
        )
        self.passive_pts_force_mapping_net = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3) ## to a 3-dim vector --- all the feats 
        )
        torch.nn.init.zeros_(self.passive_pts_force_mapping_net[-1].weight)
        torch.nn.init.zeros_(self.passive_pts_force_mapping_net[-1].bias)
        ## for each passive to act pts feat vec -> transform the feature into latent features via an MLP 
        ## pool the feature into a global feature vector; concatenate it with per-point feature 
        ## transform the concatenated feature into a latent feature using an MLP
        ## pool the feature and transform it into delta offsets and delta angular offsets


        ## set the initial passive object verts and normals ###
        ## the default scene is the box scene ##
        self.penetration_determining = "plane_primitives"
        self.canon_passive_obj_verts = None
        self.canon_passive_obj_normals = None
        self.train_residual_normal_forces = False
        
        self.lin_damping_coefs = nn.Embedding( # tim
            num_embeddings=150, embedding_dim=1
        )
        torch.nn.init.ones_(self.lin_damping_coefs.weight) # (1.0 - damping) * prev_ts_vel + cur_ts_delta_vel
        
        ## optimization method ##
        self.ang_damping_coefs = nn.Embedding( # tim
            num_embeddings=150, embedding_dim=1
        )
        torch.nn.init.ones_(self.ang_damping_coefs.weight)  # (1.0 - samping_coef) * prev_ts_vel + cur_ts_delta_vel ##
        
        # optimizable_spring_ks_normal, optimizable_spring_ks_friction #
        self.optimizable_spring_ks_normal = nn.Embedding(
            num_embeddings=200, embedding_dim=1
        )
        torch.nn.init.ones_(self.optimizable_spring_ks_normal.weight)
        # self.optimizable_spring_ks.weight.data[0, :] = math.sqrt(1.0)
        # self.optimizable_spring_ks.weight.data[1, :] = math.sqrt(1.0)
        
        self.optimizable_spring_ks_friction = nn.Embedding(
            num_embeddings=200, embedding_dim=1
        )
        torch.nn.init.ones_(self.optimizable_spring_ks_friction.weight)
        
        
        ### a manually set spring ks friction and the spring ks normals ### ### ## #
        

        
    def set_split_bending_network(self, ):
        self.use_split_network = True
        ##### split network single ##### ## 
        self.split_network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
            if i + 1 in self.skips
            else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
            for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
        )
        with torch.no_grad():
            for i, layer in enumerate(self.split_network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.split_network[-1].weight.data *= 0.0
            if self.use_last_layer_bias:
                self.split_network[-1].bias.data *= 0.0
                self.split_network[-1].bias.data += 0.2
        ##### split network single #####
        
        
        self.split_dir_network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
            if i + 1 in self.skips
            else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
            for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, 3)]
        )
        with torch.no_grad(): # no_grad()
            for i, layer in enumerate(self.split_dir_network[:]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays #
            # 
            # self.split_dir_network[-1].weight.data *= 0.0
            # if self.use_last_layer_bias:
            #     self.split_dir_network[-1].bias.data *= 0.0
        ##### split network single #####
        
        
        # ### 
        ## weighting_network for the network ##
        self.weighting_net_input_ch = 3
        self.split_weighting_network = nn.ModuleList(
            [nn.Linear(self.weighting_net_input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.weighting_net_input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, 1)])

        with torch.no_grad():
            for i, layer in enumerate(self.split_weighting_network[:]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    if i < len(self.split_weighting_network) - 1:
                        torch.nn.init.zeros_(layer.bias)
    
    def uniformly_sample_pts(self, tot_pts, nn_samples):
        tot_pts_prob = torch.ones_like(tot_pts[:, 0])
        tot_pts_prob = tot_pts_prob / torch.sum(tot_pts_prob)
        pts_dist = Categorical(tot_pts_prob)
        sampled_pts_idx = pts_dist.sample((nn_samples,))
        sampled_pts_idx = sampled_pts_idx.squeeze()
        sampled_pts = tot_pts[sampled_pts_idx]
        return sampled_pts
    
    
    def query_for_sdf(self, cur_pts, cur_frame_transformations):
        # 
        cur_frame_rotation, cur_frame_translation = cur_frame_transformations
        # cur_pts: nn_pts x 3 #
        # print(f"cur_pts: {cur_pts.size()}, cur_frame_translation: {cur_frame_translation.size()}, cur_frame_rotation: {cur_frame_rotation.size()}")
        ### transformed pts ###
        # cur_transformed_pts = torch.matmul(
        #     cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        # ).transpose(1, 0)
        # center_init_passive_obj_verts #
        cur_transformed_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0) - self.center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) + self.center_init_passive_obj_verts.unsqueeze(0)
        # v = (v - center) * scale #
        # sdf_space_center # 
        cur_transformed_pts_np = cur_transformed_pts.detach().cpu().numpy()
        cur_transformed_pts_np = (cur_transformed_pts_np - np.reshape(self.sdf_space_center, (1, 3))) * self.sdf_space_scale
        cur_transformed_pts_np = (cur_transformed_pts_np + 1.) / 2.
        cur_transformed_pts_xs = (cur_transformed_pts_np[:, 0] * self.sdf_res).astype(np.int32) # [x, y, z] of the transformed_pts_np # 
        cur_transformed_pts_ys = (cur_transformed_pts_np[:, 1] * self.sdf_res).astype(np.int32)
        cur_transformed_pts_zs = (cur_transformed_pts_np[:, 2] * self.sdf_res).astype(np.int32)
        
        cur_transformed_pts_xs = np.clip(cur_transformed_pts_xs, a_min=0, a_max=self.sdf_res - 1)
        cur_transformed_pts_ys = np.clip(cur_transformed_pts_ys, a_min=0, a_max=self.sdf_res - 1)
        cur_transformed_pts_zs = np.clip(cur_transformed_pts_zs, a_min=0, a_max=self.sdf_res - 1)
        
        
        if self.obj_sdf_th is None:
            self.obj_sdf_th = torch.from_numpy(self.obj_sdf).float().cuda()
        cur_transformed_pts_xs_th = torch.from_numpy(cur_transformed_pts_xs).long().cuda()
        cur_transformed_pts_ys_th = torch.from_numpy(cur_transformed_pts_ys).long().cuda()
        cur_transformed_pts_zs_th = torch.from_numpy(cur_transformed_pts_zs).long().cuda()
        
        cur_pts_sdf = batched_index_select(self.obj_sdf_th, cur_transformed_pts_xs_th, 0)
        # print(f"After selecting the x-axis: {cur_pts_sdf.size()}")
        cur_pts_sdf = batched_index_select(cur_pts_sdf, cur_transformed_pts_ys_th.unsqueeze(-1), 1).squeeze(1)
        # print(f"After selecting the y-axis: {cur_pts_sdf.size()}")
        cur_pts_sdf = batched_index_select(cur_pts_sdf, cur_transformed_pts_zs_th.unsqueeze(-1), 1).squeeze(1)
        # print(f"After selecting the z-axis: {cur_pts_sdf.size()}")
        
        if self.obj_sdf_grad is not None:
            if self.obj_sdf_grad_th is None:
                self.obj_sdf_grad_th = torch.from_numpy(self.obj_sdf_grad).float().cuda()
                self.obj_sdf_grad_th = self.obj_sdf_grad_th / torch.clamp(torch.norm(self.obj_sdf_grad_th, p=2, keepdim=True, dim=-1), min=1e-5)
            cur_pts_sdf_grad =  batched_index_select(self.obj_sdf_grad_th, cur_transformed_pts_xs_th, 0) # nn_pts x res x res x 3 
            cur_pts_sdf_grad =  batched_index_select(cur_pts_sdf_grad, cur_transformed_pts_ys_th.unsqueeze(-1), 1).squeeze(1)
            cur_pts_sdf_grad =  batched_index_select(cur_pts_sdf_grad, cur_transformed_pts_zs_th.unsqueeze(-1), 1).squeeze(1)
            # cur_pts_sdf_grad = cur_pts_sdf_grad / torch
        else:
            cur_pts_sdf_grad = None
            
        # cur_pts_sdf = self.obj_sdf[cur_transformed_pts_xs]
        # cur_pts_sdf = cur_pts_sdf[:, cur_transformed_pts_ys]
        # cur_pts_sdf = cur_pts_sdf[:, :, cur_transformed_pts_zs]
        # cur_pts_sdf = np.diagonal(cur_pts_sdf)
        # print(f"cur_pts_sdf: {cur_pts_sdf.shape}")
        # # gradient of sdf # 
        # # the contact force dierection should be the negative direction of the sdf gradient? #
        # # it seems true #
        # # get the cur_pts_sdf value #
        # cur_pts_sdf = torch.from_numpy(cur_pts_sdf).float().cuda()
        if cur_pts_sdf_grad is None:
            return cur_pts_sdf
        else:
            return cur_pts_sdf, cur_pts_sdf_grad # return the grad as the 
    
    
    def query_for_sdf_of_canon_obj(self, cur_pts, cur_frame_transformations):
        
        # 

        cur_frame_rotation, cur_frame_translation = cur_frame_transformations
        
        cur_transformed_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) 
        
        # cur_transformed_pts = torch.matmul(
        #     cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0) - self.center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)
        # ).transpose(1, 0) + self.center_init_passive_obj_verts.unsqueeze(0)
        # # v = (v - center) * scale #
        # sdf_space_center # 
        cur_transformed_pts_np = cur_transformed_pts.detach().cpu().numpy()
        # 
        cur_transformed_pts_np = (cur_transformed_pts_np - np.reshape(self.sdf_space_center, (1, 3))) * self.sdf_space_scale
        cur_transformed_pts_np = (cur_transformed_pts_np + 1.) / 2.
        cur_transformed_pts_xs = (cur_transformed_pts_np[:, 0] * self.sdf_res).astype(np.int32) # [x, y, z] of the transformed_pts_np # 
        cur_transformed_pts_ys = (cur_transformed_pts_np[:, 1] * self.sdf_res).astype(np.int32)
        cur_transformed_pts_zs = (cur_transformed_pts_np[:, 2] * self.sdf_res).astype(np.int32)
        
        cur_transformed_pts_xs = np.clip(cur_transformed_pts_xs, a_min=0, a_max=self.sdf_res - 1)
        cur_transformed_pts_ys = np.clip(cur_transformed_pts_ys, a_min=0, a_max=self.sdf_res - 1)
        cur_transformed_pts_zs = np.clip(cur_transformed_pts_zs, a_min=0, a_max=self.sdf_res - 1)
        
        
        if self.obj_sdf_th is None:
            self.obj_sdf_th = torch.from_numpy(self.obj_sdf).float().cuda()
        cur_transformed_pts_xs_th = torch.from_numpy(cur_transformed_pts_xs).long().cuda()
        cur_transformed_pts_ys_th = torch.from_numpy(cur_transformed_pts_ys).long().cuda()
        cur_transformed_pts_zs_th = torch.from_numpy(cur_transformed_pts_zs).long().cuda()
        
        cur_pts_sdf = batched_index_select(self.obj_sdf_th, cur_transformed_pts_xs_th, 0)
        # print(f"After selecting the x-axis: {cur_pts_sdf.size()}")
        cur_pts_sdf = batched_index_select(cur_pts_sdf, cur_transformed_pts_ys_th.unsqueeze(-1), 1).squeeze(1)
        # print(f"After selecting the y-axis: {cur_pts_sdf.size()}")
        cur_pts_sdf = batched_index_select(cur_pts_sdf, cur_transformed_pts_zs_th.unsqueeze(-1), 1).squeeze(1)
        # print(f"After selecting the z-axis: {cur_pts_sdf.size()}")
        
        if self.obj_sdf_grad is not None:
            if self.obj_sdf_grad_th is None:
                self.obj_sdf_grad_th = torch.from_numpy(self.obj_sdf_grad).float().cuda()
                self.obj_sdf_grad_th = self.obj_sdf_grad_th / torch.clamp(torch.norm(self.obj_sdf_grad_th, p=2, keepdim=True, dim=-1), min=1e-5)
            cur_pts_sdf_grad =  batched_index_select(self.obj_sdf_grad_th, cur_transformed_pts_xs_th, 0) # nn_pts x res x res x 3 
            cur_pts_sdf_grad =  batched_index_select(cur_pts_sdf_grad, cur_transformed_pts_ys_th.unsqueeze(-1), 1).squeeze(1)
            cur_pts_sdf_grad =  batched_index_select(cur_pts_sdf_grad, cur_transformed_pts_zs_th.unsqueeze(-1), 1).squeeze(1)
            # cur_pts_sdf_grad = cur_pts_sdf_grad / torch
        else:
            cur_pts_sdf_grad = None
            
        # cur_pts_sdf = self.obj_sdf[cur_transformed_pts_xs]
        # cur_pts_sdf = cur_pts_sdf[:, cur_transformed_pts_ys]
        # cur_pts_sdf = cur_pts_sdf[:, :, cur_transformed_pts_zs]
        # cur_pts_sdf = np.diagonal(cur_pts_sdf)
        # print(f"cur_pts_sdf: {cur_pts_sdf.shape}")
        # # the contact force dierection should be the negative direction of the sdf gradient? #
        # # get the cur_pts_sdf value # 
        # cur_pts_sdf = torch.from_numpy(cur_pts_sdf).float().cuda()
        if cur_pts_sdf_grad is None:
            return cur_pts_sdf
        else:
            return cur_pts_sdf, cur_pts_sdf_grad # return the grad as the 
    
    ## query for cotnacting 
    def query_for_contacting_ball_primitives(self, cur_pts, cur_frame_transformations):
        cur_frame_rotation, cur_frame_translation = cur_frame_transformations
        
        inv_transformed_queried_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0) - self.center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) + self.center_init_passive_obj_verts.unsqueeze(0)
        
        # center_verts, ball_r #
        center_verts = self.center_verts
        ball_r = self.ball_r
        dist_inv_transformed_pts_w_center_ball = torch.sum(
            (inv_transformed_queried_pts - center_verts.unsqueeze(0)) ** 2, dim=-1 ## 
        )
        
        penetration_indicators = dist_inv_transformed_pts_w_center_ball <= (ball_r ** 2)
        
        # maxx_dist_to_planes, projected_plane_pts_transformed, projected_plane_normals_transformed, projected_plane_pts, selected_plane_normals
        dir_center_to_ball = inv_transformed_queried_pts - center_verts.unsqueeze(0) ## nn_pts x 3 ##
        norm_center_to_ball = torch.norm(dir_center_to_ball, dim=-1, p=2, keepdim=True)
        dir_center_to_ball = dir_center_to_ball / torch.clamp(torch.norm(dir_center_to_ball, dim=-1, p=2, keepdim=True), min=1e-6)
        sd_dist = norm_center_to_ball - ball_r
        projected_ball_pts = center_verts.unsqueeze(0) + dir_center_to_ball * ball_r
        projected_ball_normals = dir_center_to_ball.clone()
        
        projected_ball_normals_transformed = torch.matmul(
            cur_frame_rotation, projected_ball_normals.contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous()
        projected_ball_pts_transformed = torch.matmul( ## center init passive obj verts 
            cur_frame_rotation, (projected_ball_pts - self.center_init_passive_obj_verts.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous() + cur_frame_translation.unsqueeze(0) + self.center_init_passive_obj_verts.unsqueeze(0)
        
        return penetration_indicators, sd_dist, projected_ball_pts_transformed, projected_ball_normals_transformed, projected_ball_pts, projected_ball_normals
    
    ## because of ## because of 
    def query_for_contacting_primitives(self, cur_pts, cur_frame_transformations):
        # cur_frame rotation -> 3 x 3 rtoations # translation -> 3 translations #
        cur_frame_rotation, cur_frame_translation = cur_frame_transformations
        # cur_pts: nn_pts x 3 #
        # print(f"cur_pts: {cur_pts.size()}, cur_frame_translation: {cur_frame_translation.size()}, cur_frame_rotation: {cur_frame_rotation.size()}")
        ### transformed pts ###
        # inv_transformed_queried_pts =  torch.matmul(
        #     cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        # ).transpose(1, 0)
        inv_transformed_queried_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0) - self.center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) + self.center_init_passive_obj_verts.unsqueeze(0)

        # maxximum # jut define the maxim # 
        # normal to six palnes -> 
        # normal to six planes -> 
        maxx_init_passive_mesh = self.maxx_init_passive_mesh
        minn_init_passive_mesh = self.minn_init_passive_mesh # 
        
        
        # max y-coordiante; min y-coordiante; max 
        dist_to_plane_max_y = torch.sum((inv_transformed_queried_pts - maxx_init_passive_mesh.unsqueeze(0)) * self.normal_plane_max_y.unsqueeze(0), dim=-1) ### signed distance to the upper s
        # maximum distnace? # 
        dist_to_plane_min_y = torch.sum((inv_transformed_queried_pts - minn_init_passive_mesh.unsqueeze(0)) * self.normal_plane_min_y.unsqueeze(0), dim=-1) ### signed distance to the lower surface #
        
        dist_to_plane_max_z = torch.sum((inv_transformed_queried_pts - maxx_init_passive_mesh.unsqueeze(0)) * self.normal_plane_max_z.unsqueeze(0), dim=-1) ### signed distance to the upper s
        # maximum distnace? # 
        dist_to_plane_min_z = torch.sum((inv_transformed_queried_pts - minn_init_passive_mesh.unsqueeze(0)) * self.normal_plane_min_z.unsqueeze(0), dim=-1) ### signed distance to the lower surface #
    
        dist_to_plane_max_x = torch.sum((inv_transformed_queried_pts - maxx_init_passive_mesh.unsqueeze(0)) * self.normal_plane_max_x.unsqueeze(0), dim=-1) ### signed distance to the upper s
        # maximum distnace? # 
        dist_to_plane_min_x = torch.sum((inv_transformed_queried_pts - minn_init_passive_mesh.unsqueeze(0)) * self.normal_plane_min_x.unsqueeze(0), dim=-1) ### signed distance to the lower surface #
        tot_dist_to_planes = torch.stack(
            [dist_to_plane_max_y, dist_to_plane_min_y, dist_to_plane_max_z, dist_to_plane_min_z, dist_to_plane_max_x, dist_to_plane_min_x], dim=-1
        )
        maxx_dist_to_planes, maxx_dist_to_planes_plane_idx = torch.max(tot_dist_to_planes, dim=-1) ### maxx dist to planes ## # nn_pts #
        
        # selected plane normals # selected plane normals # kinematics dirven mano? # 
        # a much more simplified setting> # # need frictio
        # contact points established and the contact information maintainacnce # 
        # test cases -> test such two relatively moving objects #
        # assume you have the correct forces --- how to opt them #
        # model the frictions # 
        tot_plane_normals = torch.stack(
            [self.normal_plane_max_y, self.normal_plane_min_y, self.normal_plane_max_z, self.normal_plane_min_z, self.normal_plane_max_x, self.normal_plane_min_x], dim=0 ### 6 x 3 -> plane normals #
        )
        # nearest plane points # # nearest plane points # # nearest plane points # # nearest plane points #
        # nearest_plane_points # 
        selected_plane_normals = tot_plane_normals[maxx_dist_to_planes_plane_idx ] ### nn_tot_pts x 3 ###
        projected_plane_pts = cur_pts - selected_plane_normals * maxx_dist_to_planes.unsqueeze(-1) ### nn_tot_pts x 3 ###
        projected_plane_pts_x = projected_plane_pts[:, 0]
        projected_plane_pts_y = projected_plane_pts[:, 1]
        projected_plane_pts_z = projected_plane_pts[:, 2]
        projected_plane_pts_x = torch.clamp(projected_plane_pts_x, min=minn_init_passive_mesh[0], max=maxx_init_passive_mesh[0])
        projected_plane_pts_y = torch.clamp(projected_plane_pts_y, min=minn_init_passive_mesh[1], max=maxx_init_passive_mesh[1])
        projected_plane_pts_z = torch.clamp(projected_plane_pts_z, min=minn_init_passive_mesh[2], max=maxx_init_passive_mesh[2])
        
        projected_plane_pts = torch.stack(
            [projected_plane_pts_x, projected_plane_pts_y, projected_plane_pts_z], dim=-1
        )
        
        # query #
        projected_plane_pts_transformed = torch.matmul(
            cur_frame_rotation, (projected_plane_pts - self.center_init_passive_obj_verts.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous() + cur_frame_translation.unsqueeze(0) + self.center_init_passive_obj_verts.unsqueeze(0)
        projected_plane_normals_transformed = torch.matmul(
            cur_frame_rotation, selected_plane_normals.contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous()
        
        # ### penetration indicator, signed distance, projected points onto the plane as the contact points ###
        return maxx_dist_to_planes <= 0, maxx_dist_to_planes, projected_plane_pts_transformed, projected_plane_normals_transformed, projected_plane_pts, selected_plane_normals
        
    
    # def forward(self, input_pts_ts, timestep_to_active_mesh, timestep_to_passive_mesh, passive_sdf_net, active_bending_net, active_sdf_net, details=None, special_loss_return=False, update_tot_def=True):
    def forward(self, input_pts_ts, timestep_to_active_mesh, timestep_to_passive_mesh, timestep_to_passive_mesh_normals, details=None, special_loss_return=False, update_tot_def=True, friction_forces=None, i_instance=0, reference_mano_pts=None, sampled_verts_idxes=None, fix_obj=False, contact_pairs_set=None):
        #### contact_pairs_set ####
        ### from input_pts to new pts ###
        # prev_pts_ts = input_pts_ts - 1 #
        ''' Kinematics rigid transformations only '''
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion # # 
        # self.timestep_to_optimizable_total_def[input_pts_ts + 1] = self.time_translations(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3) #
        # self.timestep_to_optimizable_quaternion[input_pts_ts + 1] = self.time_quaternions(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(4) #
        # cur_optimizable_rot_mtx = quaternion_to_matrix(self.timestep_to_optimizable_quaternion[input_pts_ts + 1]) #
        # self.timestep_to_optimizable_rot_mtx[input_pts_ts + 1] = cur_optimizable_rot_mtx #
        ''' Kinematics rigid transformations only '''
        
        nex_pts_ts = input_pts_ts + 1
        
        ''' Kinematics transformations from acc and torques '''
        # rigid_acc = self.time_forces(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3) #
        # torque = self.time_torques(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3) # TODO: note that inertial_matrix^{-1} real_torque #
        ''' Kinematics transformations from acc and torques '''

        # friction_qd = 0.1 # # 
        sampled_input_pts = timestep_to_active_mesh[input_pts_ts] # sampled points --> sampled points #
        ori_nns = sampled_input_pts.size(0)
        if sampled_verts_idxes is not None:
            sampled_input_pts = sampled_input_pts[sampled_verts_idxes]
        nn_sampled_input_pts = sampled_input_pts.size(0)
        
        if nex_pts_ts in timestep_to_active_mesh:
            ### disp_sampled_input_pts = nex_sampled_input_pts - sampled_input_pts ###
            nex_sampled_input_pts = timestep_to_active_mesh[nex_pts_ts].detach()
        else:
            nex_sampled_input_pts = timestep_to_active_mesh[input_pts_ts].detach()
        nex_sampled_input_pts = nex_sampled_input_pts[sampled_verts_idxes]
        
        
        # ws_normed = torch.ones((sampled_input_pts.size(0),), dtype=torch.float32).cuda()
        # ws_normed = ws_normed / float(sampled_input_pts.size(0))
        # m = Categorical(ws_normed)
        # nn_sampled_input_pts = 20000
        # sampled_input_pts_idx = m.sample(sample_shape=(nn_sampled_input_pts,))
        
        
        # sampled_input_pts_normals = #
        init_passive_obj_verts = timestep_to_passive_mesh[0]
        init_passive_obj_ns = timestep_to_passive_mesh_normals[0]
        center_init_passive_obj_verts = init_passive_obj_verts.mean(dim=0)
        
        # cur_passive_obj_rot, cur_passive_obj_trans # 
        cur_passive_obj_rot = quaternion_to_matrix(self.timestep_to_quaternion[input_pts_ts].detach())
        cur_passive_obj_trans = self.timestep_to_total_def[input_pts_ts].detach()
        
        # 
        
        cur_passive_obj_verts = torch.matmul(cur_passive_obj_rot, (init_passive_obj_verts - center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)).transpose(1, 0) + center_init_passive_obj_verts.squeeze(0) + cur_passive_obj_trans.unsqueeze(0) # 
        cur_passive_obj_ns = torch.matmul(cur_passive_obj_rot, init_passive_obj_ns.transpose(1, 0).contiguous()).transpose(1, 0).contiguous() ## transform the normals ##
        cur_passive_obj_ns = cur_passive_obj_ns / torch.clamp(torch.norm(cur_passive_obj_ns, dim=-1, keepdim=True), min=1e-8)
        cur_passive_obj_center = center_init_passive_obj_verts + cur_passive_obj_trans
        passive_center_point = cur_passive_obj_center
        
        # cur_active_mesh = timestep_to_active_mesh[input_pts_ts]
        # nex_active_mesh = timestep_to_active_mesh[input_pts_ts + 1]
        
        # ######## vel for frictions #########
        # vel_active_mesh = nex_active_mesh - cur_active_mesh ### the active mesh velocity
        # friction_k = self.ks_friction_val(torch.zeros((1,)).long().cuda()).view(1)
        # friction_force = vel_active_mesh * friction_k
        # forces = friction_force
        # ######## vel for frictions #########
        
        
        # ######## vel for frictions #########
        # vel_active_mesh = nex_active_mesh - cur_active_mesh # the active mesh velocity
        # if input_pts_ts > 0:
        #     vel_passive_mesh = self.timestep_to_vel[input_pts_ts - 1]
        # else:
        #     vel_passive_mesh = torch.zeros((3,), dtype=torch.float32).cuda() ### zeros ###
        # vel_active_mesh = vel_active_mesh - vel_passive_mesh.unsqueeze(0) ## nn_active_pts x 3 ## --> active pts ##
        
        # friction_k = self.ks_friction_val(torch.zeros((1,)).long().cuda()).view(1)
        # friction_force = vel_active_mesh * friction_k # 
        # forces = friction_force
        # ######## vel for frictions ######### # # maintain the contact / continuous contact -> patch contact
        # coantacts in previous timesteps -> ###

        # cur actuation #
        cur_actuation_embedding_st_idx = self.nn_actuators * input_pts_ts
        cur_actuation_embedding_ed_idx = self.nn_actuators * (input_pts_ts + 1)
        cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
        # ######### optimize the actuator forces directly #########
        # cur_actuation_forces = self.actuator_forces(cur_actuation_embedding_idxes) # actuation embedding idxes #
        # forces = cur_actuation_forces
        # ######### optimize the actuator forces directly #########
        
        if friction_forces is None:
            if self.nn_instances == 1:
                cur_actuation_friction_forces = self.actuator_friction_forces(cur_actuation_embedding_idxes)
            else:
                cur_actuation_friction_forces = self.actuator_friction_forces[i_instance](cur_actuation_embedding_idxes)
        else:
            if reference_mano_pts is not None:
                ref_mano_pts_nn = reference_mano_pts.size(0)
                cur_actuation_embedding_st_idx = ref_mano_pts_nn * input_pts_ts
                cur_actuation_embedding_ed_idx = ref_mano_pts_nn * (input_pts_ts + 1)
                cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
                cur_actuation_friction_forces = friction_forces(cur_actuation_embedding_idxes)
                
                # nn_ref_pts x 3 #
                # sampled_input_pts #
                # r = 0.01 #
                threshold_ball_r = 0.01
                dist_input_pts_to_reference_pts = torch.sum(
                    (sampled_input_pts.unsqueeze(1) - reference_mano_pts.unsqueeze(0)) ** 2, dim=-1
                )
                dist_input_pts_to_reference_pts = torch.sqrt(dist_input_pts_to_reference_pts)
                weights_input_to_reference = 0.5 - dist_input_pts_to_reference_pts
                weights_input_to_reference[weights_input_to_reference < 0] = 0
                weights_input_to_reference[dist_input_pts_to_reference_pts > threshold_ball_r] = 0
                
                minn_dist_input_pts_to_reference_pts, minn_idx_input_pts_to_reference_pts = torch.min(dist_input_pts_to_reference_pts, dim=-1)
                
                weights_input_to_reference[dist_input_pts_to_reference_pts == minn_dist_input_pts_to_reference_pts.unsqueeze(-1)] = 0.1 - dist_input_pts_to_reference_pts[dist_input_pts_to_reference_pts == minn_dist_input_pts_to_reference_pts.unsqueeze(-1)]
                
                weights_input_to_reference = weights_input_to_reference / torch.clamp(torch.sum(weights_input_to_reference, dim=-1, keepdim=True), min=1e-9)
                
                # cur_actuation_friction_forces = weights_input_to_reference.unsqueeze(-1) * cur_actuation_friction_forces.unsqueeze(0) # nn_input_pts x nn_ref_pts x 1 xxxx 1 x nn_ref_pts x 3 -> nn_input_pts x nn_ref_pts x 3
                # cur_actuation_friction_forces = cur_actuation_friction_forces.sum(dim=1)
                
                # cur_actuation_friction_forces * weights_input_to_reference.unsqueeze(-1)
                cur_actuation_friction_forces = batched_index_select(cur_actuation_friction_forces, minn_idx_input_pts_to_reference_pts, dim=0)
            else:
                # cur_actuation_embedding_st_idx = 365428 * input_pts_ts
                # cur_actuation_embedding_ed_idx = 365428 * (input_pts_ts + 1)
                if sampled_verts_idxes is not None:
                    cur_actuation_embedding_st_idx = ori_nns * input_pts_ts
                    cur_actuation_embedding_ed_idx = ori_nns * (input_pts_ts + 1)
                    cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
                    cur_actuation_friction_forces = friction_forces(cur_actuation_embedding_idxes)
                    cur_actuation_friction_forces = cur_actuation_friction_forces[sampled_verts_idxes]
                else:
                    cur_actuation_embedding_st_idx = nn_sampled_input_pts * input_pts_ts
                    cur_actuation_embedding_ed_idx = nn_sampled_input_pts * (input_pts_ts + 1)
                    cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
                    cur_actuation_friction_forces = friction_forces(cur_actuation_embedding_idxes)
        
        # nn instances # # nninstances # #
        if self.nn_instances == 1:
            ws_alpha = self.ks_weights(torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights(torch.ones((1,)).long().cuda()).view(1)
        else:
            ws_alpha = self.ks_weights[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        
        dist_sampled_pts_to_passive_obj = torch.sum( # nn_sampled_pts x nn_passive_pts 
            (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_sampled_pts_to_passive_obj, minn_idx_sampled_pts_to_passive_obj = torch.min(dist_sampled_pts_to_passive_obj, dim=-1)
        
        
        # use_penalty_based_friction, use_disp_based_friction # 
        # ### get the nearest object point to the in-active object ###
        # if contact_pairs_set is not None and self.use_penalty_based_friction and (not self.use_disp_based_friction):
        if contact_pairs_set is not None and self.use_penalty_based_friction and (not self.use_disp_based_friction): # contact pairs set # # contact pairs set ##
            # for each calculated contacts, calculate the current contact points reversed transformed to the contact local frame #
            # use the reversed transformed active point and the previous rest contact point position to calculate the contact friction force #
            # transform the force to the current contact frame #
            # x_h^{cur} - x_o^{cur} --- add the frictions for the hand 
            # add the friction force onto the object point # # contact point position -> nn_contacts x 3 #
            contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
            # contact active pos and contact passive pos # contact_active_pos; contact_passive_pos; #
            # contact_active_pos = sampled_input_pts[contact_active_idxes] # should not be inter_obj_pts... #
            # contact_passive_pos = cur_passive_obj_verts[contact_passive_idxes]
            # to the passive obje ###s
            minn_idx_sampled_pts_to_passive_obj[contact_active_idxes] = contact_passive_idxes
        
            dist_sampled_pts_to_passive_obj = torch.sum( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1
            )
            dist_sampled_pts_to_passive_obj = batched_index_select(dist_sampled_pts_to_passive_obj, minn_idx_sampled_pts_to_passive_obj.unsqueeze(1), dim=1).squeeze(1)
            # ### get the nearest object point to the in-active object ###
        
        
        
        # sampled_input_pts #
        # inter_obj_pts #
        # inter_obj_normals 
        
        # nn_sampledjpoints #
        # cur_passive_obj_ns # # inter obj normals # # 
        inter_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj]
        inter_obj_pts = cur_passive_obj_verts[minn_idx_sampled_pts_to_passive_obj]
        
        cur_passive_obj_verts_pts_idxes = torch.arange(0, cur_passive_obj_verts.size(0), dtype=torch.long).cuda() # 
        inter_passive_obj_pts_idxes = cur_passive_obj_verts_pts_idxes[minn_idx_sampled_pts_to_passive_obj]
        
        # inter_obj_normals #
        inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts.detach()
        dot_inter_obj_pts_to_sampled_pts_normals = torch.sum(inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1)
        
        # contact_pairs_set #
        
        ###### penetration penalty strategy v1 ######
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0
        # penetrating_depth =  -1 * torch.sum(inter_obj_pts_to_sampled_pts * inter_obj_normals.detach(), dim=-1)
        # penetrating_depth_penalty = penetrating_depth[penetrating_indicator].mean()
        # self.penetrating_depth_penalty = penetrating_depth_penalty
        # if torch.isnan(penetrating_depth_penalty): # get the penetration penalties #
        #     self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        ###### penetration penalty strategy v1 ######
        
        
        ###### penetration penalty strategy v2 ######
        # if input_pts_ts > 0:
        #     prev_active_obj = timestep_to_active_mesh[input_pts_ts - 1].detach()
        #     if sampled_verts_idxes is not None:
        #         prev_active_obj = prev_active_obj[sampled_verts_idxes]
        #     disp_prev_to_cur = sampled_input_pts - prev_active_obj
        #     disp_prev_to_cur = torch.norm(disp_prev_to_cur, dim=-1, p=2)
        #     penetrating_depth_penalty = disp_prev_to_cur[penetrating_indicator].mean()
        #     self.penetrating_depth_penalty = penetrating_depth_penalty
        #     if torch.isnan(penetrating_depth_penalty):
        #         self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        # else:
        #     self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        ###### penetration penalty strategy v2 ######
        #
        # jacobian matrix calculation?
        # jacobian 
        
        ###### penetration penalty strategy v3 ######
        # if input_pts_ts > 0:
        #     cur_rot = self.timestep_to_optimizable_rot_mtx[input_pts_ts].detach()
        #     cur_trans = self.timestep_to_total_def[input_pts_ts].detach()
        #     queried_sdf = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
        #     penetrating_indicator = queried_sdf < 0
        #     if sampled_verts_idxes is not None:
        #         prev_active_obj = prev_active_obj[sampled_verts_idxes]
        #     disp_prev_to_cur = sampled_input_pts - prev_active_obj
        #     disp_prev_to_cur = torch.norm(disp_prev_to_cur, dim=-1, p=2)
        #     penetrating_depth_penalty = disp_prev_to_cur[penetrating_indicator].mean()
        #     self.penetrating_depth_penalty = penetrating_depth_penalty
        # else:
        #     # cur_rot = torch.eye(3, dtype=torch.float32).cuda()
        #     # cur_trans = torch.zeros((3,), dtype=torch.float32).cuda()
        #     self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        ###### penetration penalty strategy v3 ######
        
        # ws_beta; 10 # # sum over the forces but not the weighted sum... # 
        ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha * 10) # ws_alpha #
        ####### sharp the weights #######
        
        # minn_dist_sampled_pts_passive_obj_thres = 0.05
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001 #
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001 # 
        minn_dist_sampled_pts_passive_obj_thres = self.minn_dist_sampled_pts_passive_obj_thres
        
        
       
        # # ws_unnormed = ws_normed_sampled
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # rigid_acc = torch.sum(forces * ws_normed.unsqueeze(-1), dim=0)
        
        #### using network weights ####
        # cur_act_weights = self.actuator_weights(cur_actuation_embedding_idxes).squeeze(-1)
        #### using network weights ####
        
        # penetrating #
        ### penetration strategy v4 ####
        
        if input_pts_ts > 0:
            cur_rot = self.timestep_to_optimizable_rot_mtx[input_pts_ts].detach()
            cur_trans = self.timestep_to_total_def[input_pts_ts].detach()
            queried_sdf = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
            penetrating_indicator = queried_sdf < 0
        else:
            penetrating_indicator = torch.zeros_like(dot_inter_obj_pts_to_sampled_pts_normals).bool()
            
        
        # if contact_pairs_set is not None and self.use_penalty_based_friction and (not self.use_disp_based_friction):
        #     penetrating
        
        ### nearest ####
        ''' decide forces via kinematics statistics '''
        ### nearest ####
        # rel_inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts # inter_obj_pts #
        # dot_rel_inter_obj_pts_normals = torch.sum(rel_inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1) ## nn_sampled_pts
        
        
        # cannot be adapted to this easily #
        # what's a better realization way? #
        
        
        # dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] = -1. * dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] #
        dist_sampled_pts_to_passive_obj[penetrating_indicator] = -1. * dist_sampled_pts_to_passive_obj[penetrating_indicator]
        # contact_spring_ka * | minn_spring_length - dist_sampled_pts_to_passive_obj |
        
        in_contact_indicator = dist_sampled_pts_to_passive_obj <= minn_dist_sampled_pts_passive_obj_thres
        
        # in_contact_indicator
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha )
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # cur_act_weights = ws_normed
        cur_act_weights = ws_unnormed
        
        
        # penetrating_indicator = 
        
        # penetrating 
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0 #
        self.penetrating_indicator = penetrating_indicator
        penetration_proj_ks = 0 - dot_inter_obj_pts_to_sampled_pts_normals
        ### penetratio nproj penalty ###
        penetration_proj_penalty = penetration_proj_ks * (-1 * torch.sum(inter_obj_pts_to_sampled_pts * inter_obj_normals.detach(), dim=-1))
        self.penetrating_depth_penalty = penetration_proj_penalty[penetrating_indicator].mean()
        if torch.isnan(self.penetrating_depth_penalty): # get the penetration penalties #
            self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        penetrating_points = sampled_input_pts[penetrating_indicator]
        penetration_proj_k_to_robot = 1.0 #  0.7
        # penetration_proj_k_to_robot = 0.01
        penetration_proj_k_to_robot = 0.0
        penetrating_forces = penetration_proj_ks.unsqueeze(-1) * inter_obj_normals.detach() * penetration_proj_k_to_robot
        penetrating_forces = penetrating_forces[penetrating_indicator]
        self.penetrating_forces = penetrating_forces #
        self.penetrating_points = penetrating_points #
        ### penetration strategy v4 #### # another mophology #
        
        # maintain the forces #
        
        # # contact_pairs_set # #
        
        # for contact pair in the contact_pair_set, get the contact pair -> the mesh index of the passive object and the active object #
        # the orientation of the contact frame #
        # original contact point position of the contact pair #
        # original orientation of the contact frame #
        ##### get previous contact information ######
        # for cur_contact_pair in contact_pairs_set:
        #     # cur_contact_pair = (contact point position, contact frame orientation) #
        #     # contact_point_positon -> should be the contact position transformed to the local contact frame #
        #     contact_point_positon, (contact_passive_idx, contact_active_idx),  contact_frame_pose = cur_contact_pair #
        #     # contact_point_positon of the contact pair #
        #     cur_active_pos = sampled_input_pts[contact_active_idx] # passive_position #
        #     # (original passive position - current passive position) * K_f = penalty based friction force # # # #
        #     cur_passive_pos = inter_obj_pts[contact_passive_idx] # active_position #
        #     # (the transformed passive position) #
        #     # 
        #     # # the continuous active and passive pos ##
        #     # # the continuous active and passive pos ##
        #     # the continuous active and passive pos ##
        #     contact_frame_orientation, contact_frame_translation = contact_frame_pose # # set the orientation and the contact frame translation
        #     # orientation, translation #
        #     cur_inv_transformed_active_pos = torch.matmul(
        #         contact_frame_orientation.contiguous().transpose(1, 0).contiguous(), (cur_active_pos - contact_frame_translation.unsqueeze(0)).transpose(1, 0)
        #     )
        
        
        
        # should be the contact penalty frictions added onto the passive object verts #
        # use the frictional force to mainatian the contact here #
        
        # maintain the contact and calculate the penetrating forces and points for each timestep and then use the displacemnet to calculate the penalty based friction forces #
        
        
        if self.nn_instances == 1: # spring ks values 
            # contact ks values # # if we set a fixed k value here #
            contact_spring_ka = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            contact_spring_kb = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 2).view(1,)
            contact_spring_kc = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            tangential_ks = self.spring_ks_values(torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        else:
            contact_spring_ka = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            contact_spring_kb = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 2).view(1,)
            contact_spring_kc = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            tangential_ks = self.spring_ks_values[i_instance](torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        
        
        
        ###### the contact force decided by the rest_length ###### # not very sure ... #
        # contact_force_d = contact_spring_ka * (self.contact_spring_rest_length - dist_sampled_pts_to_passive_obj) # + contact_spring_kb * (self.contact_spring_rest_length - dist_sampled_pts_to_passive_obj) ** 2 + contact_spring_kc * (self.contact_spring_rest_length - dist_sampled_pts_to_passive_obj) ** 3 # 
        ###### the contact force decided by the rest_length ######
 

        ##### the contact force decided by the theshold ###### # realted to the distance threshold and the HO distance #
        contact_force_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_sampled_pts_to_passive_obj) 
        ###### the contact force decided by the threshold ######
        
        ###### Get the tangential forces via optimizable forces  ###### # dot along the normals ## 
        cur_actuation_friction_forces_along_normals = torch.sum(cur_actuation_friction_forces * inter_obj_normals, dim=-1).unsqueeze(-1) * inter_obj_normals
        tangential_vel = cur_actuation_friction_forces - cur_actuation_friction_forces_along_normals
        ###### Get the tangential forces via optimizable forces  ######
        
        # cur actuation friction forces along normals #
        
        ###### Get the tangential forces via tangential velocities  ######
        # vel_sampled_pts_along_normals = torch.sum(vel_sampled_pts * inter_obj_normals, dim=-1).unsqueeze(-1) * inter_obj_normals
        # tangential_vel = vel_sampled_pts - vel_sampled_pts_along_normals
        ###### Get the tangential forces via tangential velocities  ######
        
        tangential_forces = tangential_vel * tangential_ks # tangential forces # 
        contact_force_d_scalar = contact_force_d.clone() # 
        contact_force_d = contact_force_d.unsqueeze(-1) * (-1. * inter_obj_normals)
        
        norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # nn_sampled_pts ##
        norm_along_normals_forces = torch.norm(contact_force_d, dim=-1, p=2) # nn_sampled_pts, nnsampledpts #
        penalty_friction_constraint = (norm_tangential_forces - self.static_friction_mu * norm_along_normals_forces) ** 2
        penalty_friction_constraint[norm_tangential_forces <= self.static_friction_mu * norm_along_normals_forces] = 0.
        penalty_friction_constraint = torch.mean(penalty_friction_constraint)
        self.penalty_friction_constraint = penalty_friction_constraint # penalty friction 
        contact_force_d_scalar = norm_along_normals_forces.clone()
        # penalty friction constraints #
        penalty_friction_tangential_forces = torch.zeros_like(contact_force_d)
        ''' Get the contact information that should be maintained''' 
        if contact_pairs_set is not None: # contact pairs set # # contact pairs set ##
            # for each calculated contacts, calculate the current contact points reversed transformed to the contact local frame #
            # use the reversed transformed active point and the previous rest contact point position to calculate the contact friction force #
            # transform the force to the current contact frame #
            # x_h^{cur} - x_o^{cur} --- add the frictions for the hand 
            # add the friction force onto the object point # # contact point position -> nn_contacts x 3 #
            contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
            # contact active pos and contact passive pos # contact_active_pos; contact_passive_pos; #
            contact_active_pos = sampled_input_pts[contact_active_idxes] # should not be inter_obj_pts... #
            contact_passive_pos = cur_passive_obj_verts[contact_passive_idxes]
            
            ''' Penalty based contact force v2 ''' 
            contact_frame_orientations, contact_frame_translations = contact_frame_pose
            transformed_prev_contact_active_pos = torch.matmul(
                contact_frame_orientations.contiguous(), contact_active_point_pts.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            transformed_prev_contact_point_position = torch.matmul(
                contact_frame_orientations.contiguous(), contact_point_position.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            diff_transformed_prev_contact_passive_to_active = transformed_prev_contact_active_pos - transformed_prev_contact_point_position
            # cur_contact_passive_pos_from_active = contact_passive_pos + diff_transformed_prev_contact_passive_to_active
            cur_contact_passive_pos_from_active = contact_active_pos - diff_transformed_prev_contact_passive_to_active
            
            friction_k = 1.0
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            # penalty_based_friction_forces = friction_k * (contact_active_pos - transformed_prev_contact_active_pos)
            
            # 
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            penalty_based_friction_forces = friction_k * (cur_contact_passive_pos_from_active - contact_passive_pos)
            ''' Penalty based contact force v2 '''
            
            ''' Penalty based contact force v1 ''' 
            ###### Contact frame orientations and translations ######
            # contact_frame_orientations, contact_frame_translations = contact_frame_pose # (nn_contacts x 3 x 3) # (nn_contacts x 3) #
            # # cur_passive_obj_verts #
            # inv_transformed_contact_active_pos = torch.matmul(
            #     contact_frame_orientations.contiguous().transpose(2, 1).contiguous(), (contact_active_pos - contact_frame_translations).contiguous().unsqueeze(-1)
            # ).squeeze(-1) # nn_contacts x 3 #
            # inv_transformed_contact_passive_pos = torch.matmul( # contact frame translations # ## nn_contacts x 3 ## # # 
            #     contact_frame_orientations.contiguous().transpose(2, 1).contiguous(), (contact_passive_pos - contact_frame_translations).contiguous().unsqueeze(-1)
            # ).squeeze(-1)
            # # inversely transformed cotnact active and passive pos #
            
            # # inv_transformed_contact_active_pos, inv_transformed_contact_passive_pos #
            # ### contact point position ### # 
            # ### use the passive point disp ###
            # # disp_active_pos = (inv_transformed_contact_active_pos - contact_point_position) # nn_contacts x 3 #
            # ### use the active point disp ###
            # # disp_active_pos = (inv_transformed_contact_active_pos - contact_active_point_pts)
            # disp_active_pos = (inv_transformed_contact_active_pos - contact_active_point_pts)
            # ### friction_k is equals to 1.0 ###
            # friction_k = 1.
            # # use the disp_active_pose as the penalty based friction forces # # nn_contacts x 3 #
            # penalty_based_friction_forces = disp_active_pos * friction_k
            
            # # get the penalty based friction forces #
            # penalty_based_friction_forces = torch.matmul(
            #     contact_frame_orientations.contiguous(), penalty_based_friction_forces.unsqueeze(-1)
            # ).contiguous().squeeze(-1).contiguous()
            ''' Penalty based contact force v1 ''' 
            
            #### strategy 1: implement the dynamic friction forces ####
            # dyn_friction_k = 1.0 # together with the friction_k #
            # # dyn_friction_k #
            # dyn_friction_force = dyn_friction_k * contact_force_d # nn_sampled_pts x 3 #
            # dyn_friction_force #
            # dyn_friction_force = #
            # tangential velocities # # tangential velocities #
            #### strategy 1: implement the dynamic friction forces ####
            
            #### strategy 2: do not use the dynamic friction forces ####
            # equalt to use a hard selector to screen the friction forces #
            #  
            # contact_force_d # # contact_force_d #
            
            valid_contact_force_d_scalar = contact_force_d_scalar[contact_active_idxes]
            
            
            # penalty_based_friction_forces #
            norm_penalty_based_friction_forces = torch.norm(penalty_based_friction_forces, dim=-1, p=2)
            # valid penalty friction forces # # valid contact force d scalar #
            valid_penalty_friction_forces_indicator = norm_penalty_based_friction_forces <= (valid_contact_force_d_scalar * self.static_friction_mu * 500)
            valid_penalty_friction_forces_indicator[:] = True


            summ_valid_penalty_friction_forces_indicator = torch.sum(valid_penalty_friction_forces_indicator.float())
            
            # print(f"summ_valid_penalty_friction_forces_indicator: {summ_valid_penalty_friction_forces_indicator}")
            # print(f"penalty_based_friction_forces: {penalty_based_friction_forces.size()}, summ_valid_penalty_friction_forces_indicator: {summ_valid_penalty_friction_forces_indicator}")
            # tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.005 #  * 1000.
            
            
            contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
        
            
            # penalty_friction_tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * contact_spring_kb
            
            penalty_friction_tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * contact_friction_spring_cur
             
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * contact_spring_kb
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000. # based friction 
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.02 
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.05 #
            
        else:
            contact_active_idxes = None
            self.contact_active_idxes = contact_active_idxes
            valid_penalty_friction_forces_indicator = None
        # tangential forces with inter obj normals # -> 
        dot_tangential_forces_with_inter_obj_normals = torch.sum(penalty_friction_tangential_forces * inter_obj_normals, dim=-1) ### nn_active_pts x # 
        penalty_friction_tangential_forces = penalty_friction_tangential_forces - dot_tangential_forces_with_inter_obj_normals.unsqueeze(-1) * inter_obj_normals
        tangential_forces_clone = tangential_forces.clone()
        # tangential_forces = torch.zeros_like(tangential_forces) ### 
        
        # if contact_active_idxes is not None:
        #     self.contact_active_idxes = contact_active_idxes
        #     self.valid_penalty_friction_forces_indicator = valid_penalty_friction_forces_indicator # 
        #     # print(f"here {summ_valid_penalty_friction_forces_indicator}")
        #     # tangential_forces[self.contact_active_idxes][self.valid_penalty_friction_forces_indicator] = tangential_forces_clone[self.contact_active_idxes][self.valid_penalty_friction_forces_indicator]
        #     contact_active_idxes_indicators = torch.ones((tangential_forces.size(0)), dtype=torch.float).cuda().bool()
        #     contact_active_idxes_indicators[:] = True
        #     contact_active_idxes_indicators[self.contact_active_idxes] = False
            
        #     tangential_forces[contact_active_idxes_indicators] = 0.
        
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # tangential forces #
        # maxx_norm_tangential, _ = torch.max(norm_tangential_forces, dim=-1)
        # minn_norm_tangential, _ = torch.min(norm_tangential_forces, dim=-1)
        # print(f"maxx_norm_tangential: {maxx_norm_tangential}, minn_norm_tangential: {minn_norm_tangential}")
        
        # two
        ### ## get new contacts ## ###
        tot_contact_point_position = []
        tot_contact_active_point_pts = []
        tot_contact_active_idxes = []
        tot_contact_passive_idxes = []
        tot_contact_frame_rotations = []
        tot_contact_frame_translations = []
        
        if torch.sum(in_contact_indicator.float()) > 0.5: # in contact indicator #
            cur_in_contact_passive_pts = inter_obj_pts[in_contact_indicator]
            cur_in_contact_passive_normals = inter_obj_normals[in_contact_indicator]
            cur_in_contact_active_pts = sampled_input_pts[in_contact_indicator] # in_contact_active_pts #
            
            # in contact active pts #
            # sampled input pts #
            # cur_passive_obj_rot, cur_passive_obj_trans #
            # cur_passive_obj_trans #
            # cur_in_contact_activE_pts #
            # in_contact_passive_pts #
            cur_contact_frame_rotations = cur_passive_obj_rot.unsqueeze(0).repeat(cur_in_contact_passive_pts.size(0), 1, 1).contiguous()
            cur_contact_frame_translations = cur_in_contact_passive_pts.clone() #
            #### contact farme active points ##### -> ##
            cur_contact_frame_active_pts = torch.matmul(
                cur_contact_frame_rotations.contiguous().transpose(1, 2).contiguous(), (cur_in_contact_active_pts - cur_contact_frame_translations).contiguous().unsqueeze(-1)
            ).squeeze(-1) ### cur_contact_frame_active_pts ###
            cur_contact_frame_passive_pts = torch.matmul(
                cur_contact_frame_rotations.contiguous().transpose(1, 2).contiguous(), (cur_in_contact_passive_pts - cur_contact_frame_translations).contiguous().unsqueeze(-1)
            ).squeeze(-1) ### cur_contact_frame_active_pts ###
            cur_in_contact_active_pts_all = torch.arange(0, sampled_input_pts.size(0)).long().cuda()
            cur_in_contact_active_pts_all = cur_in_contact_active_pts_all[in_contact_indicator]
            cur_inter_passive_obj_pts_idxes = inter_passive_obj_pts_idxes[in_contact_indicator]
            # contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose 
            # cur_contact_frame_pose = (cur_contact_frame_rotations, cur_contact_frame_translations)
            # contact_point_positions = cur_contact_frame_passive_pts #
            # contact_active_idxes, cotnact_passive_idxes #
            # contact_point_position = cur_contact_frame_passive_pts
            # contact_active_idxes = cur_in_contact_active_pts_all
            # contact_passive_idxes = cur_inter_passive_obj_pts_idxes
            tot_contact_active_point_pts.append(cur_contact_frame_active_pts)
            tot_contact_point_position.append(cur_contact_frame_passive_pts) # contact frame points
            tot_contact_active_idxes.append(cur_in_contact_active_pts_all) # active_pts_idxes 
            tot_contact_passive_idxes.append(cur_inter_passive_obj_pts_idxes) # passive_pts_idxes 
            tot_contact_frame_rotations.append(cur_contact_frame_rotations) # rotations 
            tot_contact_frame_translations.append(cur_contact_frame_translations) # translations 


        ## 
        # ####### if contact_pairs_set is not None and torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5: ########
        # if contact_pairs_set is not None and torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5:
        #     # contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
        #     prev_contact_active_point_pts = contact_active_point_pts[valid_penalty_friction_forces_indicator]
        #     prev_contact_point_position = contact_point_position[valid_penalty_friction_forces_indicator]
        #     prev_contact_active_idxes = contact_active_idxes[valid_penalty_friction_forces_indicator]
        #     prev_contact_passive_idxes = contact_passive_idxes[valid_penalty_friction_forces_indicator]
        #     prev_contact_frame_rotations = contact_frame_orientations[valid_penalty_friction_forces_indicator]
        #     prev_contact_frame_translations = contact_frame_translations[valid_penalty_friction_forces_indicator]
            
        #     tot_contact_active_point_pts.append(prev_contact_active_point_pts)
        #     tot_contact_point_position.append(prev_contact_point_position)
        #     tot_contact_active_idxes.append(prev_contact_active_idxes)
        #     tot_contact_passive_idxes.append(prev_contact_passive_idxes)
        #     tot_contact_frame_rotations.append(prev_contact_frame_rotations)
        #     tot_contact_frame_translations.append(prev_contact_frame_translations)
        ####### if contact_pairs_set is not None and torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5: ########
        
        
        
        if len(tot_contact_frame_rotations) > 0:
            upd_contact_active_point_pts = torch.cat(tot_contact_active_point_pts, dim=0)
            upd_contact_point_position = torch.cat(tot_contact_point_position, dim=0)
            upd_contact_active_idxes = torch.cat(tot_contact_active_idxes, dim=0)
            upd_contact_passive_idxes = torch.cat(tot_contact_passive_idxes, dim=0)
            upd_contact_frame_rotations = torch.cat(tot_contact_frame_rotations, dim=0)
            upd_contact_frame_translations = torch.cat(tot_contact_frame_translations, dim=0)
            upd_contact_pairs_information = [upd_contact_active_point_pts, upd_contact_point_position, (upd_contact_active_idxes, upd_contact_passive_idxes), (upd_contact_frame_rotations, upd_contact_frame_translations)]
        else:
            upd_contact_pairs_information = None
            
        
        
        # # previus 
        if self.use_penalty_based_friction and self.use_disp_based_friction:
            disp_friction_tangential_forces = nex_sampled_input_pts - sampled_input_pts
            
            contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
            
            disp_friction_tangential_forces = disp_friction_tangential_forces * contact_friction_spring_cur
            disp_friction_tangential_forces_dot_normals = torch.sum(
                disp_friction_tangential_forces * inter_obj_normals, dim=-1
            )
            disp_friction_tangential_forces = disp_friction_tangential_forces - disp_friction_tangential_forces_dot_normals.unsqueeze(-1) * inter_obj_normals
            
            penalty_friction_tangential_forces = disp_friction_tangential_forces
        
        
        # # tangential forces # 
        # tangential_forces = tangential_forces * mult_weights.unsqueeze(-1) # #
        ### strict cosntraints ###
        if self.use_penalty_based_friction:
            forces = penalty_friction_tangential_forces + contact_force_d # tantential forces and contact force d #
        else:
            # print(f"not using use_penalty_based_friction...")
            tangential_forces_norm = torch.sum(tangential_forces ** 2, dim=-1)
            pos_tangential_forces = tangential_forces[tangential_forces_norm > 1e-5]
            # print(pos_tangential_forces)
            forces = tangential_forces + contact_force_d # tantential forces and contact force d #
        # forces = penalty_friction_tangential_forces + contact_force_d # tantential forces and contact force d #
        ''' decide forces via kinematics statistics '''

        ''' Decompose forces and calculate penalty froces ''' # 
        # penalty_dot_forces_normals, penalty_friction_constraint # # contraints # # 
        # # get the forces -> decompose forces # 
        dot_forces_normals = torch.sum(inter_obj_normals * forces, dim=-1) ### nn_sampled_pts ###
        # forces_along_normals = dot_forces_normals.unsqueeze(-1) * inter_obj_normals ## the forces along the normal direction ##
        # tangential_forces = forces - forces_along_normals # tangential forces # # tangential forces ### tangential forces ##
        # penalty_friction_tangential_forces = force - 
        
        
        #### penalty_friction_tangential_forces, tangential_forces ####
        self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
        self.tangential_forces = tangential_forces
    
        
        penalty_dot_forces_normals = dot_forces_normals ** 2
        penalty_dot_forces_normals[dot_forces_normals <= 0] = 0 # 1) must in the negative direction of the object normal #
        penalty_dot_forces_normals = torch.mean(penalty_dot_forces_normals) # 1) must # 2) must #
        self.penalty_dot_forces_normals = penalty_dot_forces_normals #
        
        
        rigid_acc = torch.sum(forces * cur_act_weights.unsqueeze(-1), dim=0) # rigid acc #
        
        ###### sampled input pts to center #######
        center_point_to_sampled_pts = sampled_input_pts - passive_center_point.unsqueeze(0)
        ###### sampled input pts to center #######
        
        ###### nearest passive object point to center #######
        # cur_passive_obj_verts_exp = cur_passive_obj_verts.unsqueeze(0).repeat(sampled_input_pts.size(0), 1, 1).contiguous() ###
        # cur_passive_obj_verts = batched_index_select(values=cur_passive_obj_verts_exp, indices=minn_idx_sampled_pts_to_passive_obj.unsqueeze(1), dim=1)
        # cur_passive_obj_verts = cur_passive_obj_verts.squeeze(1) # squeeze(1) #
        
        # center_point_to_sampled_pts = cur_passive_obj_verts -  passive_center_point.unsqueeze(0) #
        ###### nearest passive object point to center #######
        
        sampled_pts_torque = torch.cross(center_point_to_sampled_pts, forces, dim=-1) 
        # torque = torch.sum(
        #     sampled_pts_torque * ws_normed.unsqueeze(-1), dim=0
        # )
        torque = torch.sum(
            sampled_pts_torque * cur_act_weights.unsqueeze(-1), dim=0
        )
        
        
        
        if self.nn_instances == 1:
            time_cons = self.time_constant(torch.zeros((1,)).long().cuda()).view(1)
            time_cons_2 = self.time_constant(torch.zeros((1,)).long().cuda() + 2).view(1)
            time_cons_rot = self.time_constant(torch.ones((1,)).long().cuda()).view(1)
            damping_cons = self.damping_constant(torch.zeros((1,)).long().cuda()).view(1)
            damping_cons_rot = self.damping_constant(torch.ones((1,)).long().cuda()).view(1)
        else:
            time_cons = self.time_constant[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            time_cons_2 = self.time_constant[i_instance](torch.zeros((1,)).long().cuda() + 2).view(1)
            time_cons_rot = self.time_constant[i_instance](torch.ones((1,)).long().cuda()).view(1)
            damping_cons = self.damping_constant[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            damping_cons_rot = self.damping_constant[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        
        k_acc_to_vel = time_cons
        k_vel_to_offset = time_cons_2
        delta_vel = rigid_acc * k_acc_to_vel
        if input_pts_ts == 0:
            cur_vel = delta_vel
        else:
            cur_vel = delta_vel + self.timestep_to_vel[input_pts_ts - 1].detach() * damping_cons
        self.timestep_to_vel[input_pts_ts] = cur_vel.detach()
        
        cur_offset = k_vel_to_offset * cur_vel
        cur_rigid_def = self.timestep_to_total_def[input_pts_ts].detach()
        
        
        delta_angular_vel = torque * time_cons_rot
        if input_pts_ts == 0:
            cur_angular_vel = delta_angular_vel
        else:
            cur_angular_vel = delta_angular_vel + self.timestep_to_angular_vel[input_pts_ts - 1].detach() * damping_cons_rot ### (3,)
        cur_delta_angle = cur_angular_vel * time_cons_rot # \delta_t w^1 / 2 
        
        prev_quaternion = self.timestep_to_quaternion[input_pts_ts].detach() # 
        cur_quaternion = prev_quaternion + update_quaternion(cur_delta_angle, prev_quaternion)
        
        
        cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
        prev_rot_mtx = quaternion_to_matrix(prev_quaternion)
        
        
        
        # cur_delta_rot_mtx = torch.matmul(cur_optimizable_rot_mtx, prev_rot_mtx.transpose(1, 0))
        
        # cur_delta_quaternion = euler_to_quaternion(cur_delta_angle[0], cur_delta_angle[1], cur_delta_angle[2]) ### delta_quaternion ###
        # cur_delta_quaternion = torch.stack(cur_delta_quaternion, dim=0) ## (4,) quaternion ##
        
        # cur_quaternion = prev_quaternion + cur_delta_quaternion ### (4,)
        
        # cur_delta_rot_mtx = quaternion_to_matrix(cur_delta_quaternion) ## (4,) -> (3, 3)
        
        # print(f"input_pts_ts {input_pts_ts},, prev_quaternion { prev_quaternion}")
        
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_rigid_def.unsqueeze(0), cur_delta_rot_mtx.detach()).squeeze(0)
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        cur_upd_rigid_def = cur_offset.detach() + cur_rigid_def
        # curupd
        # if update_tot_def:
        
        
        
        # cur_optimizable_total_def = cur_offset + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        # cur_optimizable_total_def = cur_offset + torch.matmul(cur_delta_rot_mtx, cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        cur_optimizable_total_def = cur_offset + cur_rigid_def
        # cur_optimizable_quaternion = prev_quaternion.detach() + cur_delta_quaternion 
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternio
        # 
        
        if not fix_obj:
            self.timestep_to_total_def[nex_pts_ts] = cur_upd_rigid_def
            self.timestep_to_optimizable_total_def[nex_pts_ts] = cur_optimizable_total_def
            self.timestep_to_optimizable_quaternion[nex_pts_ts] = cur_quaternion
            
            cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
            self.timestep_to_optimizable_rot_mtx[nex_pts_ts] = cur_optimizable_rot_mtx
            # new_pts = raw_input_pts - cur_offset.unsqueeze(0)
            
            self.timestep_to_angular_vel[input_pts_ts] = cur_angular_vel.detach()
            self.timestep_to_quaternion[nex_pts_ts] = cur_quaternion.detach()
            # self.timestep_to_optimizable_total_def[input_pts_ts + 1] = self.time_translations(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3)
        
        
        self.timestep_to_input_pts[input_pts_ts] = sampled_input_pts.detach()
        self.timestep_to_point_accs[input_pts_ts] = forces.detach()
        self.timestep_to_aggregation_weights[input_pts_ts] = cur_act_weights.detach()
        self.timestep_to_sampled_pts_to_passive_obj_dist[input_pts_ts] = dist_sampled_pts_to_passive_obj.detach()
        self.save_values = {
            'timestep_to_point_accs': {cur_ts: self.timestep_to_point_accs[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_point_accs}, 
            # 'timestep_to_vel': {cur_ts: self.timestep_to_vel[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_vel},
            'timestep_to_input_pts': {cur_ts: self.timestep_to_input_pts[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_input_pts},
            'timestep_to_aggregation_weights': {cur_ts: self.timestep_to_aggregation_weights[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_aggregation_weights},
            'timestep_to_sampled_pts_to_passive_obj_dist': {cur_ts: self.timestep_to_sampled_pts_to_passive_obj_dist[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_sampled_pts_to_passive_obj_dist},
            # 'timestep_to_ws_normed': {cur_ts: self.timestep_to_ws_normed[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_ws_normed},
            # 'timestep_to_defed_input_pts_sdf': {cur_ts: self.timestep_to_defed_input_pts_sdf[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_defed_input_pts_sdf},
        }
        
        return upd_contact_pairs_information


    ### forward; #### # 
    def forward2(self, input_pts_ts, timestep_to_active_mesh, timestep_to_passive_mesh, timestep_to_passive_mesh_normals, details=None, special_loss_return=False, update_tot_def=True, friction_forces=None, i_instance=0, reference_mano_pts=None, sampled_verts_idxes=None, fix_obj=False, contact_pairs_set=None, pts_frictional_forces=None):

        
        nex_pts_ts = input_pts_ts + 1
        
        # sampled_input_pts 
        sampled_input_pts = timestep_to_active_mesh[input_pts_ts]
        # ori_nns = sampled_input_pts.size(0)
        if sampled_verts_idxes is not None:
            sampled_input_pts = sampled_input_pts[sampled_verts_idxes]
        # nn_sampled_input_pts = sampled_input_pts.size(0)
        
        if nex_pts_ts in timestep_to_active_mesh:
            ### disp_sampled_input_pts = nex_sampled_input_pts - sampled_input_pts ###
            nex_sampled_input_pts = timestep_to_active_mesh[nex_pts_ts].detach()
        else:
            nex_sampled_input_pts = timestep_to_active_mesh[input_pts_ts].detach() ## 
        
        if sampled_verts_idxes is not None:
            nex_sampled_input_pts = nex_sampled_input_pts[sampled_verts_idxes] ## 
        
        disp_act_pts_cur_to_nex = nex_sampled_input_pts - sampled_input_pts ## act pts cur to nex ## ## act pts cur to nex ##
        # disp_act_pts_cur_to_nex = disp_act_pts_cur_to_nex / torch.clamp(torch.norm(disp_act_pts_cur_to_nex, p=2, keepdim=True, dim=-1), min=1e-5)
        
        ### 
        norm_disp_act_pts = torch.clamp(torch.norm(disp_act_pts_cur_to_nex, p=2, keepdim=True), min=1e-5)
        
        disp_act_pts_cur_to_nex = disp_act_pts_cur_to_nex / norm_disp_act_pts
        real_norm = torch.clamp(torch.norm(disp_act_pts_cur_to_nex, p=2, keepdim=True, dim=-1), min=1e-5)
        real_norm = torch.mean(real_norm)
        # print(norm_disp_act_pts, real_norm)
        
        if self.canon_passive_obj_verts is None:
            ## center init passsive obj verts ##
            init_passive_obj_verts = timestep_to_passive_mesh[0] # 
            init_passive_obj_ns = timestep_to_passive_mesh_normals[0]
            center_init_passive_obj_verts = init_passive_obj_verts.mean(dim=0)
            self.center_init_passive_obj_verts = center_init_passive_obj_verts.clone()
        else:
            init_passive_obj_verts = self.canon_passive_obj_verts
            init_passive_obj_ns = self.canon_passive_obj_normals
            
            # center_init_passive_obj_verts = init_passive_obj_verts.mean(dim=0)
            # self.center_init_passive_obj_verts = center_init_passive_obj_verts.clone()
            
            # direction of the normal direction has been changed #
            # contact region and multiple contact points ##
            center_init_passive_obj_verts = torch.zeros((3, ), dtype=torch.float32).cuda()
            self.center_init_passive_obj_verts = center_init_passive_obj_verts.clone()
        
        
        
        # cur_passive_obj_rot, cur_passive_obj_trans # ## quaternion to matrix -- quaternion for 
        cur_passive_obj_rot = quaternion_to_matrix(self.timestep_to_quaternion[input_pts_ts].detach())
        cur_passive_obj_trans = self.timestep_to_total_def[input_pts_ts].detach()
        
        # sampled_input_pts, cur_passive_obj_verts, cur_passive_obj_ns #
        ''' Transform the passive object verts and normals ''' 
        cur_passive_obj_verts = torch.matmul(cur_passive_obj_rot, (init_passive_obj_verts - center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)).transpose(1, 0) + center_init_passive_obj_verts.squeeze(0) + cur_passive_obj_trans.unsqueeze(0)
        
        ## passive obj 
        cur_passive_obj_ns = torch.matmul(cur_passive_obj_rot, init_passive_obj_ns.transpose(1, 0).contiguous()).transpose(1, 0).contiguous()
        ### passvie obj ns ###
        cur_passive_obj_ns = cur_passive_obj_ns / torch.clamp(torch.norm(cur_passive_obj_ns, dim=-1, keepdim=True), min=1e-8)
        cur_passive_obj_center = center_init_passive_obj_verts + cur_passive_obj_trans
        passive_center_point = cur_passive_obj_center # passive obj center #
        
        self.cur_passive_obj_ns = cur_passive_obj_ns
        self.cur_passive_obj_verts = cur_passive_obj_verts
        
        
        # nn instances # # # cur passive obj ns ##
        # if self.nn_instances == 1:
        #     ws_alpha = self.ks_weights(torch.zeros((1,)).long().cuda()).view(1)
        #     ws_beta = self.ks_weights(torch.ones((1,)).long().cuda()).view(1)
        # else:
        #     ws_alpha = self.ks_weights[i_instance](torch.zeros((1,)).long().cuda()).view(1)
        #     ws_beta = self.ks_weights[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        
        if self.use_sqrt_dist: # use sqrt distance #
            dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
            )
        else:
            dist_sampled_pts_to_passive_obj = torch.sum( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1
            )
        #### use sqrt distances ####
        
        # ### add the sqrt for calculate the l2 distance ###
        # dist_sampled_pts_to_passive_obj = torch.sqrt(dist_sampled_pts_to_passive_obj) ### 
        
        
        # dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
        #     (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
        # )
        
        ''' distance between sampled pts and the passive object '''
        ## get the object vert idx with the 
        dist_sampled_pts_to_passive_obj, minn_idx_sampled_pts_to_passive_obj = torch.min(dist_sampled_pts_to_passive_obj, dim=-1) # get the minn idx sampled pts to passive obj ##
        
        
        ''' calculate the apssvie objects normals '''
        # inter obj normals at the current frame #
        # inter_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj]
        
        
        
        # disp_act_pts_cur_to_nex = disp_act_pts_cur_to_nex[mini]
        
        ### use obj normals as the direction ###
        # inter_obj_normals = -1 * inter_obj_normals.detach().clone()
        ### use the active points displacement directions as the direction ### # the normal 
        
        inter_obj_normals = -1 * disp_act_pts_cur_to_nex.detach().clone()
        
        # penetration_determining # ## parameters -> optimizable ##3
        inter_obj_pts = cur_passive_obj_verts[minn_idx_sampled_pts_to_passive_obj]
        
        
        # cur_passive_obj_verts_pts_idxes = torch.arange(0, cur_passive_obj_verts.size(0), dtype=torch.long).cuda() # 
        # inter_passive_obj_pts_idxes = cur_passive_obj_verts_pts_idxes[minn_idx_sampled_pts_to_passive_obj]
        
        # inter_obj_normals #
        # inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts.detach() # sampled p
        # dot_inter_obj_pts_to_sampled_pts_normals = torch.sum(inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1)
        
        ###### penetration penalty strategy v1 ######
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0
        # penetrating_depth =  -1 * torch.sum(inter_obj_pts_to_sampled_pts * inter_obj_normals.detach(), dim=-1)
        # penetrating_depth_penalty = penetrating_depth[penetrating_indicator].mean()
        # self.penetrating_depth_penalty = penetrating_depth_penalty
        # if torch.isnan(penetrating_depth_penalty): # get the penetration penalties #
        #     self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        ###### penetration penalty strategy v1 ######
        
        # ws_beta; 10 # # sum over the forces but not the weighted sum... # 
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha * 10) # ws_alpha #
        ####### sharp the weights #######
        
        # minn_dist_sampled_pts_passive_obj_thres = 0.05
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001 # m
        minn_dist_sampled_pts_passive_obj_thres = self.minn_dist_sampled_pts_passive_obj_thres
        
        
       
        # # ws_unnormed = ws_normed_sampled
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9) e
        # rigid_acc = torch.sum(forces * ws_normed.unsqueeze(-1), dim=0)
        
        #### using network weights ####
        # cur_act_weights = self.actuator_weights(cur_actuation_embedding_idxes).squeeze(-1)
        #### using network weights ####
        
        # penetrating #
        ### penetration strategy v4 #### ## threshold of the sampled pts ##
        
        if input_pts_ts > 0 or (input_pts_ts == 0 and input_pts_ts in self.timestep_to_total_def):
            cur_rot = self.timestep_to_optimizable_rot_mtx[input_pts_ts].detach()
            cur_trans = self.timestep_to_total_def[input_pts_ts].detach()
            # obj_sdf_grad
            if self.penetration_determining == "sdf_of_canon": ### queried sdf? ###
                if self.obj_sdf_grad is None:
                    queried_sdf = self.query_for_sdf_of_canon_obj(sampled_input_pts, (cur_rot, cur_trans))
                else:
                    queried_sdf, queried_sdf_grad = self.query_for_sdf_of_canon_obj(sampled_input_pts, (cur_rot, cur_trans))
            else:
                if self.obj_sdf_grad is None:
                    queried_sdf = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
                else:
                    queried_sdf, queried_sdf_grad = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
                    # inter_obj_normals = -1.0 * queried_sdf_grad
                    # inter_obj_normals = queried_sdf_grad
                    # inter_obj_normals = torch.matmul( # 3 x 3 xxxx 3 x N -> 3 x N 
                    #     cur_rot, inter_obj_normals.contiguous().transpose(1, 0).contiguous()
                    # ).contiguous().transpose(1, 0).contiguous()
            # elif self.penetration_determining == ## timestep to total def ##
            penetrating_indicator = queried_sdf < 0
        else:
            cur_rot = torch.eye(n=3, dtype=torch.float32).cuda()
            cur_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            if self.penetration_determining == "sdf_of_canon":
                if self.obj_sdf_grad is None:
                    queried_sdf = self.query_for_sdf_of_canon_obj(sampled_input_pts, (cur_rot, cur_trans))
                else:
                    queried_sdf, queried_sdf_grad = self.query_for_sdf_of_canon_obj(sampled_input_pts, (cur_rot, cur_trans))
            else:
                if self.obj_sdf_grad is None:
                    queried_sdf = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
                else:
                    queried_sdf, queried_sdf_grad = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
                    # inter_obj_normals = -1.0 * queried_sdf_grad
                    # inter_obj_normals =  queried_sdf_grad
            penetrating_indicator = queried_sdf < 0

        ### nearest ####
        ''' decide forces via kinematics statistics '''
        ### nearest ####
        # rel_inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts # inter_obj_pts #
        # dot_rel_inter_obj_pts_normals = torch.sum(rel_inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1) ## nn_sampled_pts
        
        penetrating_indicator_mult_factor = torch.ones_like(penetrating_indicator).float()
        penetrating_indicator_mult_factor[penetrating_indicator] = -1. ## penetration indicator ##
        
        
        
        dist_sampled_pts_to_passive_obj = dist_sampled_pts_to_passive_obj * penetrating_indicator_mult_factor
        # contact_spring_ka * | minn_spring_length - dist_sampled_pts_to_passive_obj |
        
        # use contact
        if self.use_contact_dist_as_sdf: # 
            queried_sdf = dist_sampled_pts_to_passive_obj # queried sdf # # queried sdf #
        
        # 
        in_contact_indicator_robot_to_obj = queried_sdf <= self.minn_dist_threshold_robot_to_obj ### queried_sdf <= minn_dist_threshold_robot_to_obj
        
        zero_level_incontact_indicator_robot_to_obj = queried_sdf <= 0.0
        
        
        ## minn_dist_sampled_pts_passive_obj_thres  #  ## in contct indicator ##
        in_contact_indicator = dist_sampled_pts_to_passive_obj <= minn_dist_sampled_pts_passive_obj_thres
        
        
        # ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        # ws_unnormed = torch.ones_like(ws_unnormed)
        ws_unnormed = torch.ones_like(dist_sampled_pts_to_passive_obj)
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha )
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # cur_act_weights = ws_normed
        cur_act_weights = ws_unnormed
        
        
        # minimized motions #
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0 #
        # self.penetrating_indicator = penetrating_indicator #  # 
        self.penetrating_indicator = in_contact_indicator_robot_to_obj
        cur_inter_obj_normals = inter_obj_normals.clone().detach()
        
        ### 
        if self.penetration_determining == "plane_primitives":  # optimize the ruels for ball case? # 
            in_contact_indicator_robot_to_obj, queried_sdf, inter_obj_pts, inter_obj_normals, canon_inter_obj_pts, canon_inter_obj_normals = self.query_for_contacting_primitives(sampled_input_pts, (cur_rot, cur_trans))
            
            self.penetrating_indicator = in_contact_indicator_robot_to_obj
        
            cur_inter_obj_normals = inter_obj_normals.clone().detach()
        elif self.penetration_determining == "ball_primitives":
            in_contact_indicator_robot_to_obj, queried_sdf, inter_obj_pts, inter_obj_normals, canon_inter_obj_pts, canon_inter_obj_normals = self.query_for_contacting_ball_primitives(sampled_input_pts, (cur_rot, cur_trans))
            self.penetrating_indicator = in_contact_indicator_robot_to_obj
            cur_inter_obj_normals = inter_obj_normals.clone().detach()
        else:
            # inter_obj_pts
            canon_inter_obj_pts = torch.matmul(
                cur_passive_obj_rot.contiguous().transpose(1, 0).contiguous(), (inter_obj_pts - cur_passive_obj_trans.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
            ).contiguous().transpose(1, 0).contiguous() ## 
            canon_inter_obj_normals = torch.matmul( # passive obj rot ## # R^T n --> R R^T n --the current inter obj normals ##
                cur_passive_obj_rot.contiguous().transpose(1, 0).contiguous(), inter_obj_normals.contiguous().transpose(1, 0).contiguous()
            ).contiguous().transpose(1, 0).contiguous() ## -> inter obj normals 
        
        
        ##### penetration depth penalty loss calculation: strategy 2 #####
        penetration_proj_ks = self.minn_dist_threshold_robot_to_obj - queried_sdf
        penetration_proj_pos = sampled_input_pts + penetration_proj_ks.unsqueeze(-1) * inter_obj_normals ## nn_sampled_pts x 3 ##
        dot_pos_to_proj_with_normal = torch.sum(
            (penetration_proj_pos.detach() - sampled_input_pts) * inter_obj_normals, dim=-1 ### nn_sampled_pts
        )
        # self.penetrating_depth_penalty = dot_pos_to_proj_with_normal[in_contact_indicator_robot_to_obj].mean()
        self.penetrating_depth_penalty = dot_pos_to_proj_with_normal[zero_level_incontact_indicator_robot_to_obj].mean()
        ##### penetration depth penalty loss calculation: strategy 2 #####
        
        
        ##### penetration depth penalty loss calculation: strategy 1 #####
        # self.penetrating_depth_penalty = (self.minn_dist_threshold_robot_to_obj - queried_sdf[in_contact_indicator_robot_to_obj]).mean()
        ##### penetration depth penalty loss calculation: strategy 1 #####
        
        
        ### penetration strategy v4 #### # another mophology #
        # sampled_input_pts, cur_passive_obj_verts, cur_passive_obj_ns #
        
        # dist_sampled_input_pts_to_passive_pts = torch.sum(
        #     (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1 ## nn_act_pts x nn_passive_pts
        # )
        # dist_sampled_input_pts_to_passive_pts = torch.sqrt(dist_sampled_input_pts_to_passive_pts)
        # dist_sampled_input_pts_to_passive_pts, _ = torch.min(dist_sampled_input_pts_to_passive_pts, dim=-1) ## nn_act_pts ##

        
        
        if self.nn_instances == 1: # spring ks values 
            # contact ks values # # if we set a fixed k value here #
            contact_spring_ka = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            contact_spring_kb = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 1).view(1,)
            # contact_spring_kc = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            # tangential_ks = self.spring_ks_values(torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        else:
            contact_spring_ka = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            # contact_spring_kb = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 2).view(1,)
            # contact_spring_kc = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            # tangential_ks = self.spring_ks_values[i_instance](torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        
        
        # optimizable_spring_ks #
        # opt_penetration_proj_k_to_robot = self.optimizable_spring_ks(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
        # opt_penetration_proj_k_to_robot_friction = self.optimizable_spring_ks(torch.zeros((1,), dtype=torch.long).cuda() + 1).view(1,)
        
        if self.optm_alltime_ks:
            opt_penetration_proj_k_to_robot = self.optimizable_spring_ks_normal(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
            opt_penetration_proj_k_to_robot_friction = self.optimizable_spring_ks_friction(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
        else:
            # optimizable_spring_ks #
            opt_penetration_proj_k_to_robot = self.optimizable_spring_ks(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            opt_penetration_proj_k_to_robot_friction = self.optimizable_spring_ks(torch.zeros((1,), dtype=torch.long).cuda() + 1).view(1,)
        
        
        # self.penetration_proj_k_to_robot = opt_penetration_proj_k_to_robot ** 2
        # self.penetration_proj_k_to_robot_friction = opt_penetration_proj_k_to_robot_friction ** 2
        
        ## penetration proj k to robot ##
        penetration_proj_k_to_robot = self.penetration_proj_k_to_robot * opt_penetration_proj_k_to_robot ** 2
        penetration_proj_k_to_robot_friction = self.penetration_proj_k_to_robot_friction * opt_penetration_proj_k_to_robot_friction ** 2
        
        # penetration_proj_k_to_robot = self.penetration_proj_k_to_robot
        # penetration_proj_k_to_robot = opt_penetration_proj_k_to_robot
        
        if self.use_split_params: ## 
            contact_spring_ka = self.spring_contact_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
        
        if self.use_sqr_spring_stiffness:
            contact_spring_ka = contact_spring_ka ** 2
            
        if self.train_residual_friction:
            contact_spring_ka = 0.1907073 ** 2
            contact_spring_kb = 0.00131699
            
        if self.use_base_sim_mpc:
            ###### set spring_ka, spring_kb, penetration_spring_ks to 1 for the future use ######
            contact_spring_ka = 0.1907073 ** 2
            contact_spring_kb = 0.00131699
            
            # penetration_proj_k_to_robot = (self.penetration_proj_k_to_robot ** 2 ) / 100.0
            # penetration_proj_k_to_robot_friction = (self.penetration_proj_k_to_robot_friction ** 2)  / 100.0
            
            penetration_proj_k_to_robot = (self.penetration_proj_k_to_robot ** 2 ) / 200.0
            penetration_proj_k_to_robot_friction = (self.penetration_proj_k_to_robot_friction ** 2)  / 200.0
            
        
        if self.use_same_contact_spring_k:
            # contact_spring_ka_ori = contact_spring_ka.clone()
            
            ''' Equal forces '''
            contact_spring_ka = penetration_proj_k_to_robot * contact_spring_ka # equal forc stiffjess
            
            contact_spring_kb = contact_spring_kb * penetration_proj_k_to_robot
            
            penetration_proj_k_to_robot = contact_spring_ka 
            
            ''' N-Equal forces '''
            # contact_spring_ka = 30. * contact_spring_ka ## change the contact spring k ##
            # penetration_proj_k_to_robot = penetration_proj_k_to_robot * contact_spring_ka_ori ### change the projection coeff to the robot
            
            
            ''' N-Equal forces '''
            # contact_spring_ka = penetration_proj_k_to_robot * contact_spring_ka ## contact spring ka ##
            # penetration_proj_k_to_robot = 30. * contact_spring_ka_ori
        else:
            contact_spring_ka = penetration_proj_k_to_robot * contact_spring_ka #
            # contact_spring_kb = contact_spring_kb * self.penetration_proj_k_to_robot_friction #proje k to robot friction 
            contact_spring_kb = contact_spring_kb * penetration_proj_k_to_robot_friction 
            penetration_proj_k_to_robot = contact_spring_ka 
        
        
        
                
        if self.use_ambient_contact_net and( not self.use_base_sim_mpc):
            ###### Passive obj pts features ######
            # if inter_obj_pts.size(0) == 0:
            #     inter_obj_pts = cur_passive_obj_verts[]
            
            # nn_ambient_max = 100
            nn_ambient_max = 300
            
            in_contact_inter_obj_pts = inter_obj_pts[in_contact_indicator_robot_to_obj]
            
            is_contact = True
            if in_contact_inter_obj_pts.size(0) == 0:
                in_contact_inter_obj_pts = inter_obj_pts[:1,]
                is_contact = False
            
            
            if in_contact_inter_obj_pts.size(0) > nn_ambient_max:
                rnd_sampled_in_contact_obj_pts_idxes = torch.randperm(in_contact_inter_obj_pts.size(0))[:nn_ambient_max]
                in_contact_inter_obj_pts = in_contact_inter_obj_pts[rnd_sampled_in_contact_obj_pts_idxes]
            else:
                rnd_sampled_in_contact_obj_pts_idxes = torch.arange(0, in_contact_inter_obj_pts.size(0), dtype=torch.long).cuda()
            
            dist_sampled_input_pts_to_in_contact_obj_pts = torch.sum(
                (sampled_input_pts.unsqueeze(1) - in_contact_inter_obj_pts.unsqueeze(0)) ** 2, dim=-1
            )
            dist_sampled_input_pts_to_in_contact_obj_pts = torch.sqrt(dist_sampled_input_pts_to_in_contact_obj_pts)
            
            # print(f"dist_sampled_input_pts_to_in_contact_obj_pts: {dist_sampled_input_pts_to_in_contact_obj_pts.size()}")
            
            dist_sampled_input_pts_to_in_contact_obj_pts, _  = torch.min(dist_sampled_input_pts_to_in_contact_obj_pts, dim=-1)
            filtered_sampled_input_pts_mask = dist_sampled_input_pts_to_in_contact_obj_pts < 0.01
            
            filtered_sampled_input_pts = sampled_input_pts[filtered_sampled_input_pts_mask]
            filtered_nex_sampled_input_pts = nex_sampled_input_pts[filtered_sampled_input_pts_mask]
            if filtered_nex_sampled_input_pts.size(0) > nn_ambient_max:
                rnd_sampled_input_pts_idxes = torch.randperm(filtered_nex_sampled_input_pts.size(0))[:nn_ambient_max]
                filtered_sampled_input_pts = filtered_sampled_input_pts[rnd_sampled_input_pts_idxes]
                filtered_nex_sampled_input_pts = filtered_nex_sampled_input_pts[rnd_sampled_input_pts_idxes]
            elif filtered_sampled_input_pts.size(0) == 0:
                filtered_sampled_input_pts = sampled_input_pts[:1]
                filtered_nex_sampled_input_pts = nex_sampled_input_pts[:1]
            
            ## distp passive obj ##
            disp_passive_obj_pts_to_act_pts = in_contact_inter_obj_pts.unsqueeze(1) - filtered_sampled_input_pts.unsqueeze(0) ## nn_passive_pts x nn_active_pts x 3 ##
            disp_passive_obj_pts_to_nex_act_pts = in_contact_inter_obj_pts.unsqueeze(1) - filtered_nex_sampled_input_pts.unsqueeze(0) ## nn_passive_pts x nn_active_pts x 3 ##
            
            # print(f'disp_passive_obj_pts_to_act_pts: {disp_passive_obj_pts_to_act_pts.size()}, disp_passive_obj_pts_to_nex_act_pts: {disp_passive_obj_pts_to_nex_act_pts.size()}')
            
            # v_p = v_o + w x r # 
            passive_obj_pts_to_obj_center = in_contact_inter_obj_pts - passive_center_point.unsqueeze(0 ) ## nn_passive_pts x 3 #
            cur_fr_angular_vel = self.timestep_to_angular_vel[input_pts_ts].detach()
            passive_obj_pts_vel = torch.cross(cur_fr_angular_vel.unsqueeze(0).repeat(in_contact_inter_obj_pts.size(0), 1).contiguous(), passive_obj_pts_to_obj_center, dim=-1) + self.timestep_to_vel[input_pts_ts].detach().unsqueeze(0) ## nn_passive_pts x 3 #
            # passive_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj] ## nn_passive_pts x 3 #
            
            if not is_contact:
                passive_obj_normals = cur_passive_obj_ns[:1]
            else:
                passive_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj][in_contact_indicator_robot_to_obj][rnd_sampled_in_contact_obj_pts_idxes]
            
            nn_passive_to_act_pts_feat_dim = self.nn_passive_to_act_pts_feat_dim
            passive_obj_to_act_obj_pts_feats = torch.cat(
                [disp_passive_obj_pts_to_act_pts, disp_passive_obj_pts_to_nex_act_pts, passive_obj_pts_vel.unsqueeze(1).repeat(1, filtered_sampled_input_pts.size(0), 1).contiguous(), passive_obj_normals.unsqueeze(1).repeat(1, filtered_sampled_input_pts.size(0), 1).contiguous()], dim=-1 #### per passive to active pts feat dim = 3 + 3 + 3 + 3 ## passive pts feats dim 
            )
            # passive_act_pts_feat_mapping_net, passive_pts_feat_mapping_net, passive_pts_force_mapping_net
            passive_obj_to_act_obj_pts_feats_exp = passive_obj_to_act_obj_pts_feats.contiguous().view(passive_obj_to_act_obj_pts_feats.size(0) * passive_obj_to_act_obj_pts_feats.size(1), -1).contiguous()
            passive_obj_to_act_obj_pts_feats = self.passive_act_pts_feat_mapping_net(passive_obj_to_act_obj_pts_feats_exp) # 
            passive_obj_to_act_obj_pts_feats = passive_obj_to_act_obj_pts_feats.contiguous().view(in_contact_inter_obj_pts.size(0), filtered_sampled_input_pts.size(0), -1).contiguous() ## nn_passive_pts x nn_act_pts x feat_dim #
            passive_obj_to_act_obj_pts_feats_pooled, _ = torch.max(passive_obj_to_act_obj_pts_feats, dim=1, keepdim=True)
            passive_obj_to_act_obj_pts_feats_concat = torch.cat(
                [passive_obj_to_act_obj_pts_feats, passive_obj_to_act_obj_pts_feats_pooled.repeat(1, filtered_sampled_input_pts.size(0), 1).contiguous()], dim=-1
            )
            passive_obj_to_act_obj_pts_feats_concat = self.passive_pts_feat_mapping_net(passive_obj_to_act_obj_pts_feats_concat) ## passive pts mapping net ## 
            passive_obj_to_act_obj_pts_feats_concat, _ = torch.max(passive_obj_to_act_obj_pts_feats_concat, dim=1) ## nn_passive_pts x feat_dim #
            
            ## use_ambient_contact_net, passive_obj_pts_force, cur_passive_obj_verts, passive_center_point # -> calculate the force and torques here ##
            # use_ambient_contact_net, disp_passive_obj_pts_to_act_pts, # 
            ## passive pts force mapping net ##
            passive_obj_pts_force = self.passive_pts_force_mapping_net(passive_obj_to_act_obj_pts_feats_concat) ## nn_passive_pts x 3 #
            passive_obj_pts_force = passive_obj_pts_force * contact_spring_kb 
            ###### Passive obj pts features ######
        
            
        
        
        if torch.isnan(self.penetrating_depth_penalty): #
            self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        
        # penetrating_points = sampled_input_pts[penetrating_indicator] # robot to obj # # 
        penetrating_points = sampled_input_pts[in_contact_indicator_robot_to_obj] # 
        # penetration_proj_k_to_robot = 1.0 
        # penetration_proj_k_to_robot = 0.01
        
        # penetration_proj_k_to_robot = 0.0
        # proj_force = dist * normal * penetration_k # # 
        penetrating_forces = penetration_proj_ks.unsqueeze(-1) * cur_inter_obj_normals * penetration_proj_k_to_robot
        # penetrating_forces = penetrating_forces[penetrating_indicator]
        penetrating_forces = penetrating_forces[in_contact_indicator_robot_to_obj]
        self.penetrating_forces = penetrating_forces # forces 
        self.penetrating_points = penetrating_points # penetrating points ## # incontact indicator toothers ##
        
        
        # contact psring ka ## cotnact 
        
        ##### the contact force decided by the theshold ###### # realted to the distance threshold and the HO distance #
        contact_force_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_sampled_pts_to_passive_obj) 
        ###### the contact force decided by the threshold ######
        
        
        # contac force d  contact spring ka * penetration depth # 
        contact_force_d = contact_force_d.unsqueeze(-1) * (-1. * inter_obj_normals)
        
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # nn_sampled_pts ##
        # norm_along_normals_forces = torch.norm(contact_force_d, dim=-1, p=2) # nn_sampled_pts, nnsampledpts #
        # penalty_friction_constraint = (norm_tangential_forces - self.static_friction_mu * norm_along_normals_forces) ** 2
        # penalty_friction_constraint[norm_tangential_forces <= self.static_friction_mu * norm_along_normals_forces] = 0.
        # penalty_friction_constraint = torch.mean(penalty_friction_constraint) # friction 
        self.penalty_friction_constraint = torch.zeros((1,), dtype=torch.float32).cuda().mean() # penalty friction 
        # contact_force_d_scalar = norm_along_normals_forces.clone()
        
        # friction models #
        # penalty friction constraints #
        penalty_friction_tangential_forces = torch.zeros_like(contact_force_d)
        
        
        
        # rotation  and translatiosn #
        cur_fr_rot = cur_passive_obj_rot # passive obj rot #
        cur_fr_trans = cur_passive_obj_trans #
        
        tot_contact_active_pts = []
        tot_contact_passive_pts = []
        tot_contact_active_idxes = []
        # tot_contact_passive_idxes = [] # # 
        tot_canon_contact_passive_normals = []
        tot_canon_contact_passive_pts = []
        tot_contact_passive_normals = [] # tot contact passive pts; tot cotnact passive normals #
        tot_contact_frictions = []
        tot_residual_normal_forces = []
        
        if contact_pairs_set is not None:
            # contact_active_pts = contact_pairs_set['contact_active_pts']
            # contact_passive_pts = contact_pairs_set['contact_passive_pts']
            contact_active_idxes = contact_pairs_set['contact_active_idxes']
            # contact_passive_idxes = contact_pairs_set # # app 
            
            # contact active idxes #
            # nn_contact_pts x 3 -> as the cotnact passvie normals #
            canon_contact_passive_normals = contact_pairs_set['canon_contact_passive_normals']
            canon_contact_passive_pts = contact_pairs_set['canon_contact_passive_pts']
            cur_fr_contact_passive_normals = torch.matmul( ## penetration normals ##
                cur_fr_rot, canon_contact_passive_normals.contiguous().transpose(1, 0).contiguous()
            ).contiguous().transpose(1, 0).contiguous() # tranformed normals  # frame passive normals #
            
            # not irrelevant at all #
            cur_fr_contact_act_pts = sampled_input_pts[contact_active_idxes]
            # cur_fr_contact_passive_pts = canon_contact_passive_pts
            # 
            cur_fr_contact_passive_pts = torch.matmul(
                cur_fr_rot, (canon_contact_passive_pts - self.center_init_passive_obj_verts.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
            ).contiguous().transpose(1, 0).contiguous()  + cur_fr_trans.unsqueeze(0) + self.center_init_passive_obj_verts.unsqueeze(0) ## passive pts
            
            nex_fr_contact_act_pts = nex_sampled_input_pts[contact_active_idxes]
            
            # cur_fr_contact_passive_to_act = cur_fr_contact_act_pts - cur_fr_contact_passive_pts # 
            
            cur_fr_contact_passive_to_act = nex_fr_contact_act_pts - cur_fr_contact_passive_pts
            
            dot_rel_disp_with_passive_normals = torch.sum(
                cur_fr_contact_passive_to_act * cur_fr_contact_passive_normals, dim=-1
            )
            cur_friction_forces = cur_fr_contact_passive_to_act - dot_rel_disp_with_passive_normals.unsqueeze(-1) * cur_fr_contact_passive_normals
            
            ## cur frame cotnct act 
            cur_cur_fr_contact_passive_to_act = cur_fr_contact_act_pts - cur_fr_contact_passive_pts
            cur_cur_penetration_depth = torch.sum(
                cur_cur_fr_contact_passive_to_act * cur_fr_contact_passive_normals, dim=-1
            )
            
            
            # if self.train_residual_friction:
            ''' add residual fictions '''
            # 3 + 3 + 3 + 3 ### active points's current relative position, active point's offset, penetration depth, normal direction
            friction_net_in_feats = torch.cat(
                [cur_cur_fr_contact_passive_to_act, cur_fr_contact_passive_to_act, cur_cur_penetration_depth.unsqueeze(-1), cur_fr_contact_passive_normals], dim=-1
            )
            residual_frictions = self.friction_network(friction_net_in_feats)
            
            residual_frictions_dot_w_normals = torch.sum(
                residual_frictions * cur_fr_contact_passive_normals, dim=-1
            )
            residual_frictions = residual_frictions - residual_frictions_dot_w_normals.unsqueeze(-1) * cur_fr_contact_passive_normals
            
            if self.only_residual:
                cur_friction_forces = residual_frictions
            else:
                cur_friction_forces = cur_friction_forces + residual_frictions
            ''' add residual fictions '''
                
                
            ''' Remaining penetration indicator determining -- strategy 1 '''
            # remaining_contact_indicators = cur_cur_penetration_depth <= 0.0 ## dot relative passive to active with passive normals ##
            ''' Remaining penetration indicator determining -- strategy 2 '''
            remaining_contact_indicators = cur_cur_penetration_depth <= self.minn_dist_threshold_robot_to_obj
            
            remaining_contact_act_idxes = contact_active_idxes[remaining_contact_indicators]
            
            
            if self.train_residual_normal_forces:
                # contact_normal_force_network
                contact_normal_forces_in_feats = torch.cat(
                    [cur_cur_fr_contact_passive_to_act, cur_fr_contact_passive_to_act, cur_cur_penetration_depth.unsqueeze(-1), cur_fr_contact_passive_normals], dim=-1
                )
                residual_normal_forces = self.contact_normal_force_network(contact_normal_forces_in_feats)
                residual_normal_forces_dot_w_normals = torch.sum(
                    residual_normal_forces * cur_fr_contact_passive_normals, dim=-1
                )
                residual_normal_forces = residual_normal_forces_dot_w_normals.unsqueeze(-1) * cur_fr_contact_passive_normals
                tot_residual_normal_forces.append(residual_normal_forces[remaining_contact_indicators])
            
            
            # cur_rel_passive_to_active = cur_fr_contact_act_pts - cur_fr_contact_passive_pts
            # dot_rel_disp_w_obj_normals = torch.sum(
            #     cur_rel_passive_to_active * cur_fr_contact_passive_normals, dim=-1
            # )
            # cur_friction_forces = cur_rel_passive_to_active - dot_rel_disp_w_obj_normals.unsqueeze(-1) * cur_fr_contact_passive_normals
            
            
            # if the dot < 0 -> still in contact ## rremaning contacts ##
            # if the dot > 0 -. not in contact and can use the points to establish new conatcts --- # maitnian the contacts # 
            # remaining_contact_indicators = dot_rel_disp_with_passive_normals <= 0.0 ##
            
            
            # remaining contact act idxes #
            
            if torch.sum(remaining_contact_indicators.float()).item() > 0.5:
                # contact_active_pts, contact_passive_pts, ## remaining cotnact indicators ##
                tot_contact_passive_normals.append(cur_fr_contact_passive_normals[remaining_contact_indicators])
                tot_contact_passive_pts.append(cur_fr_contact_passive_pts[remaining_contact_indicators]) ## 
                tot_contact_active_pts.append(cur_fr_contact_act_pts[remaining_contact_indicators]) ## contact act pts 
                
                tot_contact_active_idxes.append(contact_active_idxes[remaining_contact_indicators])
                # tot_contact_passive_idxes.append(contact_passive_idxes[remaining_contact_indicators]) # # passive idxes # 
                tot_contact_frictions.append(cur_friction_forces[remaining_contact_indicators])
                tot_canon_contact_passive_pts.append(canon_contact_passive_pts[remaining_contact_indicators])
                tot_canon_contact_passive_normals.append(canon_contact_passive_normals[remaining_contact_indicators])
                
        else:
            remaining_contact_act_idxes = torch.empty((0,), dtype=torch.long).cuda() ## remaining contact act idxes ##
        
        # remaining idxes #
        new_in_contact_indicator_robot_to_obj = in_contact_indicator_robot_to_obj.clone()
        new_in_contact_indicator_robot_to_obj[remaining_contact_act_idxes] = False
        
        tot_active_pts_idxes = torch.arange(0, sampled_input_pts.size(0), dtype=torch.long).cuda() 
    
        
        if torch.sum(new_in_contact_indicator_robot_to_obj.float()).item() > 0.5:
            # 
            # in_contact_indicator_robot_to_obj, queried_sdf, inter_obj_pts, inter_obj_normals, canon_inter_obj_pts, canon_inter_obj_normals 
            new_contact_active_pts = sampled_input_pts[new_in_contact_indicator_robot_to_obj]
            new_canon_contact_passive_pts = canon_inter_obj_pts[new_in_contact_indicator_robot_to_obj]
            new_canon_contact_passive_normals = canon_inter_obj_normals[new_in_contact_indicator_robot_to_obj] ## obj normals ##
            new_contact_active_idxes = tot_active_pts_idxes[new_in_contact_indicator_robot_to_obj]
            
            new_cur_fr_contact_passive_normals = torch.matmul( 
                cur_fr_rot, new_canon_contact_passive_normals.contiguous().transpose(1, 0).contiguous() ## transform for the new passive obj normals 
            ).contiguous().transpose(1, 0).contiguous() # 
            
            # new cur fr contact passive pts # 
            new_cur_fr_contact_passive_pts = torch.matmul(
                cur_fr_rot, (new_canon_contact_passive_pts - self.center_init_passive_obj_verts.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
            ).contiguous().transpose(1, 0).contiguous()  + cur_fr_trans.unsqueeze(0) + self.center_init_passive_obj_verts.unsqueeze(0) ## passive pts
            
            
            
            new_nex_fr_contact_active_pts = nex_sampled_input_pts[new_in_contact_indicator_robot_to_obj]
            
            
            new_cur_fr_contact_passive_to_act = new_nex_fr_contact_active_pts - new_cur_fr_contact_passive_pts
            
            dot_rel_disp_with_passive_normals = torch.sum( # new cur contact passive normals #
                new_cur_fr_contact_passive_to_act * new_cur_fr_contact_passive_normals, dim=-1
            )
            new_frictions = new_cur_fr_contact_passive_to_act - dot_rel_disp_with_passive_normals.unsqueeze(-1) * new_cur_fr_contact_passive_normals
            
            
            # if self.train_residual_friction:
            ''' add residual fictions '''
            new_cur_cur_fr_contact_passive_to_act = new_contact_active_pts - new_cur_fr_contact_passive_pts
            new_cur_cur_penetration_depth = torch.sum(
                new_cur_cur_fr_contact_passive_to_act * new_cur_fr_contact_passive_normals, dim=-1
            )
            # 3 + 3 + 3 + 3 ### active points's current relative position, active point's offset, penetration depth, normal direction
            new_friction_net_in_feats = torch.cat(
                [new_cur_cur_fr_contact_passive_to_act, new_cur_fr_contact_passive_to_act, new_cur_cur_penetration_depth.unsqueeze(-1), new_cur_fr_contact_passive_normals], dim=-1
            )
            new_residual_frictions = self.friction_network(new_friction_net_in_feats)
            
            new_residual_frictions_dot_w_normals = torch.sum(
                new_residual_frictions * new_cur_fr_contact_passive_normals, dim=-1
            )
            new_residual_frictions = new_residual_frictions - new_residual_frictions_dot_w_normals.unsqueeze(-1) * new_cur_fr_contact_passive_normals
            # new_frictions = new_frictions + new_residual_frictions
            
            if self.only_residual:
                new_frictions = new_residual_frictions
            else:
                new_frictions = new_frictions + new_residual_frictions
            ''' add residual fictions '''
            
            if self.train_residual_normal_forces:
                contact_normal_forces_in_feats = torch.cat(
                    [new_cur_cur_fr_contact_passive_to_act, new_cur_fr_contact_passive_to_act, new_cur_cur_penetration_depth.unsqueeze(-1), new_cur_fr_contact_passive_normals],  dim=-1
                )
                new_residual_normal_forces = self.contact_normal_force_network(contact_normal_forces_in_feats)
                new_residual_normal_forces_dot_w_normals = torch.sum(
                    new_residual_normal_forces * new_cur_fr_contact_passive_normals, dim=-1
                ) # dot with normals ##
                ## normal forces dot with normals ## # dot w 
                
                new_residual_normal_forces = new_residual_normal_forces_dot_w_normals.unsqueeze(-1) * new_cur_fr_contact_passive_normals # normal 
                tot_residual_normal_forces.append(new_residual_normal_forces)
                
            
            # new_frictions = torch.zeros_like(new_cur_fr_contact_passive_pts)
            tot_contact_passive_normals.append(new_cur_fr_contact_passive_normals)
            tot_contact_passive_pts.append(new_cur_fr_contact_passive_pts)
            tot_contact_active_pts.append(new_contact_active_pts)
            tot_contact_active_idxes.append(new_contact_active_idxes)
            tot_canon_contact_passive_pts.append(new_canon_contact_passive_pts)
            tot_canon_contact_passive_normals.append(new_canon_contact_passive_normals)
            tot_contact_frictions.append(new_frictions)
        
        
        # contact points nearest to the contact area ## --- nearest to the active contact points #
        # nn_ambient_contact_active_pts = 500
        # nn_ambient_contact_passive_pts = 500
        
        
        if len(tot_contact_passive_normals) > 0:
            # forces ? # not hard to compute ... #
            # passive normals; passive pts #
            tot_contact_passive_normals = torch.cat(
                tot_contact_passive_normals, dim=0
            )
            tot_contact_passive_pts = torch.cat(tot_contact_passive_pts, dim=0)
            tot_contact_active_pts = torch.cat(tot_contact_active_pts, dim=0)
            tot_contact_active_idxes = torch.cat(tot_contact_active_idxes, dim=0)
            tot_canon_contact_passive_pts = torch.cat(tot_canon_contact_passive_pts, dim=0)
            tot_canon_contact_passive_normals = torch.cat(tot_canon_contact_passive_normals, dim=0)
            tot_contact_frictions = torch.cat(tot_contact_frictions, dim=0)
            if self.train_residual_normal_forces: ## the 
                tot_residual_normal_forces = torch.cat(tot_residual_normal_forces, dim=0)
            
            contact_passive_to_active = tot_contact_active_pts - tot_contact_passive_pts
            # dot relative passive to active with the passive normals # ## relative
            
            # this depth should be adjusted according to minn_dist_threshold_robot_to_obj ##
            dot_rel_passive_to_active_with_normals = torch.sum(
                contact_passive_to_active * tot_contact_passive_normals, dim=-1 ### dot with the passive normals ##
            )
            # Adjust the penetration depth used for contact force computing using the distance threshold #
            dot_rel_passive_to_active_with_normals = dot_rel_passive_to_active_with_normals - self.minn_dist_threshold_robot_to_obj
            # dot with the passive normals ## dot with passive normals ## ## passive normals ##
            ### penetration depth * the passive obj normals ### # dot value and with 
            contact_forces_along_normals = dot_rel_passive_to_active_with_normals.unsqueeze(-1) * tot_contact_passive_normals * contact_spring_ka # dot wiht relative # negative normal directions #
            
            if self.train_residual_normal_forces:
                tot_residual_normal_forces = tot_residual_normal_forces * contact_spring_ka
                if self.only_residual: # forces along normals ---- residual normal forces # # residual normal forces #
                    # print(f"only residual normal forces")
                    contact_forces_along_normals = tot_residual_normal_forces
                else:
                    contact_forces_along_normals = contact_forces_along_normals + tot_residual_normal_forces
            
            # return the contact pairs and return the contact dicts # 
            # return the contact pairs and the contact dicts # 
            # having got the contact pairs -> contact dicts # 
            # having got the contact pairs -> contact dicts # ## contact spring kb ##
            
            
            # if not self.only_residual: # tot contact friction ##
            tot_contact_frictions = tot_contact_frictions * contact_spring_kb # unlash the potential of networks 
            
            
            # active points nearest to the contact area #
            
            
            
            ######## NOTE: use the network to produce point forces #########
            if self.use_contact_network:
                # contact_passvie_and local pts #
                ##### point_type_embedding_net, point_force_net, point_force_net_out #####
                act_pts_embeddings_single = self.point_type_embedding_net(torch.zeros((1,), dtype=torch.long).cuda()).view(1, -1)
                act_pts_embeddings = act_pts_embeddings_single.repeat(tot_contact_active_pts.size(0), 1)
                passive_pts_embeddings_single = self.point_type_embedding_net(torch.zeros((1,), dtype=torch.long).cuda() + 1).view(1, -1)
                passive_pts_embeddings = passive_pts_embeddings_single.repeat(tot_contact_passive_pts.size(0), 1)
                
                
                if not self.use_contact_region_pts:
                    self.ts_to_contact_act_pts[input_pts_ts] = tot_contact_active_pts.detach().cpu().numpy()
                    self.ts_to_contact_passive_pts[input_pts_ts] = tot_contact_passive_pts.detach().cpu().numpy()
                    
                    nex_fr_contact_active_pts = nex_sampled_input_pts[tot_contact_active_idxes]
                    disp_act_contact_pts = nex_fr_contact_active_pts - tot_contact_active_pts
                    
                    ### active pts feats ###
                    act_pts_feat = torch.cat(
                        [tot_contact_active_pts, disp_act_contact_pts, act_pts_embeddings[:, 3:]], dim=-1
                    )
                    ### passive pts feats ###
                    passive_pts_feat = torch.cat(
                        [tot_contact_passive_pts, torch.zeros_like(tot_contact_passive_pts), passive_pts_embeddings[:, 3:]], dim=-1
                    )
                    
                    # act_pts_feat = torch.cat(
                    #     [tot_contact_active_pts, act_pts_embeddings], dim=-1
                    # )
                    # passive_pts_feat = torch.cat(
                    #     [tot_contact_passive_pts, passive_pts_embeddings], dim=-1
                    # )
                else:
                    dist_sampled_input_pts_to_contact_act_pts = torch.sum(
                        sampled_input_pts.unsqueeze(1) - tot_contact_active_pts.unsqueeze(0), dim=-1 ## nn_sampled_pts x nn_contact_pts
                    )
                    dist_sampled_input_pts_to_contact_act_pts, _ = torch.min(dist_sampled_input_pts_to_contact_act_pts, dim=-1)
                    near_contact_act_pts = dist_sampled_input_pts_to_contact_act_pts < 0.01
                    dist_passive_obj_pts_to_contact_passive_pts = torch.sum(
                        cur_passive_obj_verts.unsqueeze(1) - tot_contact_passive_pts.unsqueeze(0), dim=-1 ## nn_passive_obj_pts x nn_contact_pts
                    )
                    dist_passive_obj_pts_to_contact_passive_pts, _ = torch.min(dist_passive_obj_pts_to_contact_passive_pts, dim=-1) ## dist
                    near_contact_passive_pts = dist_passive_obj_pts_to_contact_passive_pts < 0.02
                    
                    selected_cur_contact_pts = sampled_input_pts[near_contact_act_pts]
                    selected_nex_contact_pts = nex_sampled_input_pts[near_contact_act_pts]
                    selected_cur_contact_passive_pts = cur_passive_obj_verts[near_contact_passive_pts]
                    disp_act_contact_pts = selected_nex_contact_pts - selected_cur_contact_pts
                    selected_act_pts_embeddings = act_pts_embeddings_single.repeat(selected_cur_contact_pts.size(0), 1)
                    selected_passive_pts_embeddings = passive_pts_embeddings_single.repeat(selected_cur_contact_passive_pts.size(0), 1)
                    
                    act_pts_feat = torch.cat(
                        [selected_cur_contact_pts, disp_act_contact_pts, selected_act_pts_embeddings[..., 3:]], dim=-1
                    )
                    passive_pts_feat = torch.cat(
                        [selected_cur_contact_passive_pts, torch.zeros_like(selected_cur_contact_passive_pts), selected_passive_pts_embeddings[..., 3:]], dim=-1
                    )
                    
                    self.ts_to_contact_act_pts[input_pts_ts] = selected_cur_contact_pts.detach().cpu().numpy()
                    self.ts_to_contact_passive_pts[input_pts_ts] = selected_cur_contact_passive_pts.detach().cpu().numpy()
                
                # for a 
                # print(f"act_pts_feat: {act_pts_feat.size()}, passive_pts_feat: {passive_pts_feat.size()}") ## nn of the passive and active pts and feats 
                tot_pts_feat = torch.cat(
                    [act_pts_feat, passive_pts_feat], dim=0
                )
                tot_pts_upd_feat = self.point_force_net(tot_pts_feat)
                glb_pts_feat, _ = torch.max(tot_pts_upd_feat, dim=0, keepdim=True)
                
                if self.pred_delta_offsets:
                    # linear_offset, angular_offset
                    # glb_feat: 1 x dim # 
                    # obj_kine_net_out_vel, obj_kine_net_out_anvel # 
                    linear_offset = self.obj_kine_net_out_vel(glb_pts_feat) ## 1 x 3 
                    linear_offset = linear_offset.squeeze(0)
                    angular_offset = self.obj_kine_net_out_anvel(glb_pts_feat)
                    angular_offset = angular_offset.squeeze(0)
                else:
                    
                    expanded_glb_pts_feat = glb_pts_feat.repeat(tot_contact_passive_pts.size(0), 1)
                    tot_pts_upd_feat = torch.cat(
                        [tot_pts_upd_feat[act_pts_embeddings.size(0): ], expanded_glb_pts_feat], dim=-1 ### tot_pts_upd_feat ---- nn_pts x tot_feat_dim
                    )
                    passive_pts_force = self.point_force_net_out(tot_pts_upd_feat)
                    
                    tot_contact_frictions = passive_pts_force
                    contact_forces_along_normals = torch.zeros_like(contact_forces_along_normals)
                    tot_contact_frictions = tot_contact_frictions * contact_spring_kb
            ######## NOTE: use the network to produce point forces ######### ##
            
            
            if pts_frictional_forces is not None:
                tot_contact_frictions = pts_frictional_forces[tot_contact_active_idxes] 
            
            # contac_forces_along_normals 
            upd_contact_pairs_information = {
                'contact_active_idxes': tot_contact_active_idxes.clone().detach(),
                'canon_contact_passive_normals': tot_canon_contact_passive_normals.clone().detach(),
                'canon_contact_passive_pts': tot_canon_contact_passive_pts.clone().detach(),
                'contact_passive_pts': tot_contact_passive_pts.clone().detach(),
            }
        else:
            upd_contact_pairs_information = None
            linear_offset = torch.zeros((3,), dtype=torch.float32).cuda()
            angular_offset = torch.zeros((3,), dtype=torch.float32).cuda()



        ''' average acitve points weights '''
        if torch.sum(cur_act_weights).item() > 0.5: ## sum cur_act_weights ##
            cur_act_weights = cur_act_weights / torch.sum(cur_act_weights)
        
        
        # norm_penalty_friction_tangential_forces = torch.norm(penalty_friction_tangential_forces, dim=-1, p=2)
        # maxx_norm_penalty_friction_tangential_forces, _ = torch.max(norm_penalty_friction_tangential_forces, dim=-1)
        # minn_norm_penalty_friction_tangential_forces, _ = torch.min(norm_penalty_friction_tangential_forces, dim=-1)
        # print(f"maxx_norm_penalty_friction_tangential_forces: {maxx_norm_penalty_friction_tangential_forces}, minn_norm_penalty_friction_tangential_forces: {minn_norm_penalty_friction_tangential_forces}")
        
        # tangetntial forces --- dot with normals #
        if not self.use_pre_proj_frictions: # inter obj normals # # if ue proj frictions # 
            dot_tangential_forces_with_inter_obj_normals = torch.sum(penalty_friction_tangential_forces * inter_obj_normals, dim=-1) ### nn_active_pts x # 
            penalty_friction_tangential_forces = penalty_friction_tangential_forces - dot_tangential_forces_with_inter_obj_normals.unsqueeze(-1) * inter_obj_normals
            
        # penalty_friction_tangential_forces = torch.zeros_like(penalty_friction_tangential_forces)
        penalty_friction_tangential_forces = tot_contact_frictions

        
        
        
        if upd_contact_pairs_information is not None:
            contact_force_d = contact_forces_along_normals # forces along normals #
            # contact forces along normals # 
            self.contact_force_d = contact_force_d
            
            # penalty_friction_tangential_forces = torch.zeros_like(contact_force_d)
            
            #### penalty_frictiontangential_forces, tangential_forces ####
            # self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
            self.tangential_forces = penalty_friction_tangential_forces
            
            self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
            self.contact_force_d = contact_force_d
            self.penalty_based_friction_forces = penalty_friction_tangential_forces
            
            self.tot_contact_passive_normals = tot_contact_passive_normals
            # penalty dot forces normals #
            ''' Penalty dot forces normals '''
            # penalty_dot_forces_normals = dot_forces_normals ** 2 # must in the negative direction of the object normal #
            # penalty_dot_forces_normals[dot_forces_normals <= 0] = 0 # 1) must in the negative direction of the object normal #
            # penalty_dot_forces_normals = torch.mean(penalty_dot_forces_normals) # 1) must # 2) must # # 
            # self.penalty_dot_forces_normals = penalty_dot_forces_normals #

            forces = self.contact_force_d + self.penalty_friction_tangential_forces
            
            center_point_to_contact_pts = tot_contact_passive_pts - passive_center_point.unsqueeze(0)
            # cneter point to contact pts # 
            # cneter point to contact pts #
            torque = torch.cross(center_point_to_contact_pts, forces)
            
            # ## use_ambient_contact_net, passive_obj_pts_force, cur_passive_obj_verts, passive_center_point # -> calculate the force and torques here ##
            if self.use_ambient_contact_net:
                if is_contact:
                    center_point_to_contact_pts = in_contact_inter_obj_pts - passive_center_point.unsqueeze(0) ## unsqueeze (0) # 
                    passive_obj_pts_torque = torch.cross(center_point_to_contact_pts, passive_obj_pts_force, dim=-1)
                    torque = torch.mean(passive_obj_pts_torque, dim=0) # torque  
                    forces = torch.mean(passive_obj_pts_force, dim=0) # forces #
                else:
                    torque = torch.zeros((3,), dtype=torch.float32).cuda()
                    forces = torch.zeros((3,), dtype=torch.float32).cuda()
                # print(f"using ambient contact net with torque: {torque}, forces: {forces}")
            else: ## torque ## 
                torque = torch.mean(torque, dim=0)
                forces = torch.mean(forces, dim=0) ## 
        else:
            # self.contact_force_d = torch.zeros((3,), dtype=torch.float32).cuda() ## contact force d ##
            torque = torch.zeros((3,), dtype=torch.float32).cuda()
            forces = torch.zeros((3,), dtype=torch.float32).cuda()
            self.contact_force_d = torch.zeros((1, 3), dtype=torch.float32).cuda()
            self.penalty_friction_tangential_forces =  torch.zeros((1, 3), dtype=torch.float32).cuda()
            self.penalty_based_friction_forces = torch.zeros((1, 3), dtype=torch.float32).cuda()
            
            self.tot_contact_passive_normals = torch.zeros((1, 3), dtype=torch.float32).cuda()
        
        
        
        
        ''' Forces and rigid acss: Strategy and version 1 '''
 
        
        
        
        if self.nn_instances == 1:
            time_cons = self.time_constant(torch.zeros((1,)).long().cuda()).view(1)
            time_cons_2 = self.time_constant(torch.zeros((1,)).long().cuda() + 2).view(1)
            time_cons_rot = self.time_constant(torch.ones((1,)).long().cuda()).view(1)
            damping_cons = self.damping_constant(torch.zeros((1,)).long().cuda()).view(1)
            damping_cons_rot = self.damping_constant(torch.ones((1,)).long().cuda()).view(1)
        else:
            time_cons = self.time_constant[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            time_cons_2 = self.time_constant[i_instance](torch.zeros((1,)).long().cuda() + 2).view(1)
            time_cons_rot = self.time_constant[i_instance](torch.ones((1,)).long().cuda()).view(1)
            damping_cons = self.damping_constant[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            damping_cons_rot = self.damping_constant[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        ## 
        if self.use_split_params: ## ## friction network should be trained? ## 
            # sep_time_constant, sep_torque_time_constant, sep_damping_constant, sep_angular_damping_constant
            time_cons = self.sep_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            time_cons_2 = self.sep_torque_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            # damping_cons = self.sep_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            # damping_cons_2 = self.sep_angular_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
        
        # time_cons = 0.05
        # time_cons_2 = 0.05
        # time_cons_rot = 0.05
        
        
        time_cons = 0.005
        time_cons_2 = 0.005
        time_cons_rot = 0.005
        
        
        time_cons = 0.0005
        time_cons_2 = 0.0005
        time_cons_rot = 0.0005
        
        # time_cons = 0.00005
        # time_cons_2 = 0.00005
        # time_cons_rot = 0.00005
        
        # time_cons = 0.0005
        # time_cons_2 = 0.0005
        # time_cons_rot = 0.0005
        
        
        ## not a good ##
        # time_cons = 0.005
        # time_cons_2 = 0.005
        # time_cons_rot = 0.005
        
        if self.use_optimizable_timecons:
            time_cons = self.time_constants(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1)
            time_cons_2 = time_cons
            time_cons_rot = self.time_constants(torch.zeros((1,), dtype=torch.long).cuda() + self.time_constants.weight.size(0) - 1 - input_pts_ts).view(1)
        
        
        
        # obj_mass = self.obj_mass
        
        obj_mass_value = self.optimizable_obj_mass(torch.zeros((1,), dtype=torch.long).cuda()).view(1)
        
        obj_mass_value = obj_mass_value ** 2
        
        rigid_acc = forces / obj_mass_value # 
        
        # damping_coef = 5e2
        
        # damping_coef = 0.0
        # damping_coef_angular = 0.0
        
        
        # # small clip with not very noticiable # # 
        
        
        # if self.use_optimizable_params: ## use optimizable params ##
        damping_coef = self.sep_damping_constant(torch.zeros((1,), dtype=torch.long).cuda()).view(1)
        damping_coef_angular = self.sep_angular_damping_constant(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
        
        damping_coef = damping_coef ** 2
        damping_coef_angular = damping_coef_angular ** 2 ## sue the sampiing coef angular and dampoing coef here ##
        
        
        if self.use_damping_params_vel:
            damping_coef_lin_vel = self.lin_damping_coefs(torch.zeros((1,), dtype=torch.long).cuda()).view(1)
            damping_coef_ang_vel = self.ang_damping_coefs(torch.zeros((1,), dtype=torch.long).cuda()).view(1)
            damping_coef_lin_vel = damping_coef_lin_vel ** 2
            damping_coef_ang_vel = damping_coef_ang_vel ** 2
        else:
            damping_coef_lin_vel = 1.0
            damping_coef_ang_vel = self.ang_vel_damping
            
        
        
        if input_pts_ts > 0:
            # the sampoing for the rigid acc here ? #
            rigid_acc = rigid_acc - damping_coef * self.timestep_to_vel[input_pts_ts - 1].detach() ## dam
        
        
        #F the sampoing for the rigid acc here ? #
        # rigid_acc = # 
        # rigid acc = forces #
        
        k_acc_to_vel = time_cons
        k_vel_to_offset = time_cons_2
        delta_vel = rigid_acc * k_acc_to_vel
        if input_pts_ts == 0:
            cur_vel = delta_vel
        else:
            ##### TMP ######
            # cur_vel = delta_vel #
            cur_vel = delta_vel + (1.0 - damping_coef_lin_vel) * self.timestep_to_vel[input_pts_ts - 1].detach() 
        self.timestep_to_vel[input_pts_ts] = cur_vel.detach()
         
        ### calculate linear object offset ##
        cur_offset = k_vel_to_offset * cur_vel 
        
        
        cur_rigid_def = self.timestep_to_total_def[input_pts_ts].detach() # timestep 
        
        
        # cur_inertia_div_factor = self.inertia_div_factor(torch.zeros((1,), dtype=torch.long).cuda()).view(1)
        
        ## inv_inertia ##
        # cur inv inertia is a large value? # # bug free? #  ### divide the inv_inertia using the factor 20.0 #
        cur_inv_inertia = torch.matmul(torch.matmul(cur_passive_obj_rot, self.I_inv_ref), cur_passive_obj_rot.transpose(1, 0))  / float(20.)
        # cur_inv_inertia = torch.matmul(torch.matmul(cur_passive_obj_rot, self.I_inv_ref), cur_passive_obj_rot.transpose(1, 0))  / float(10.) ## 
        # cur_inv_inertia = torch.matmul(torch.matmul(cur_passive_obj_rot, self.I_inv_ref), cur_passive_obj_rot.transpose(1, 0)) / float(cur_inertia_div_factor) ## 
        
        cur_inv_inertia = torch.eye(n=3, dtype=torch.float32).cuda() # three values for the inertia? # 
        
        obj_inertia_value = self.obj_inertia(torch.zeros((1,), dtype=torch.long).cuda()).view(3,)
        obj_inertia_value = obj_inertia_value ** 2
        # cur_inv_inertia = torch.diag(obj_inertia_value)
        cur_inv_inertia = cur_inv_inertia * obj_inertia_value.unsqueeze(0) ## 3 x 3 matrix ## ### the inertia values ## 
        cur_inv_inertia = torch.matmul(torch.matmul(cur_passive_obj_rot, cur_inv_inertia), cur_passive_obj_rot.transpose(1, 0)) 
        
        ### calculate object torque ###
        torque = torch.matmul(cur_inv_inertia, torque.unsqueeze(-1)).contiguous().squeeze(-1)  ### get the torque of the object ###
        #
        # 
        if input_pts_ts > 0: # torque # 
            torque = torque - damping_coef_angular * self.timestep_to_angular_vel[input_pts_ts - 1].detach()
        delta_angular_vel = torque * time_cons_rot
        
        # print(f"torque: {torque}") #
        
        if input_pts_ts == 0:
            cur_angular_vel = delta_angular_vel
        else:
            ##### TMP ######
            # cur_angular_vel = delta_angular_vel # 
            # cur_angular_vel = delta_angular_vel + self.timestep_to_angular_vel[input_pts_ts - 1].detach()
            cur_angular_vel = delta_angular_vel + (1.0 - damping_coef_ang_vel) * (self.timestep_to_angular_vel[input_pts_ts - 1].detach()) # damping coef ###
        
        
        cur_delta_angle = cur_angular_vel * time_cons_rot # \delta_t w^1 / 2 # / 2 # # \delta_t w^1 #
        
        
        if self.use_contact_network and self.pred_delta_offsets:
            # linear_offset, angular_offset
            cur_offset = linear_offset
            cur_delta_angle = angular_offset
        
        # prev ## 
        prev_quaternion = self.timestep_to_quaternion[input_pts_ts].detach() # input pts ts # 
        cur_quaternion = prev_quaternion + update_quaternion(cur_delta_angle, prev_quaternion)
        
        ## curqueternion = prev_quaternion 
        # cur_quaternion = prev_quaternion + update_quaternion(cur_delta_angle, prev_quaternion) #
        cur_quaternion = cur_quaternion / torch.norm(cur_quaternion, p=2, dim=-1, keepdim=True)
        # obj_mass # 
        
        
        cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
        prev_rot_mtx = quaternion_to_matrix(prev_quaternion)
        
        # cur_  3 no frictions/ # # # 
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_rigid_def.unsqueeze(0), cur_delta_rot_mtx.detach()).squeeze(0)
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        cur_upd_rigid_def = cur_offset.detach() + cur_rigid_def
        
        ## quaternion to matrix and 
        # cur_optimizable_total_def = cur_offset + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        # cur_optimizable_total_def = cur_offset + torch.matmul(cur_delta_rot_mtx, cur_rigid_def.unsqueeze(-1)).squeeze(-1) #
        cur_optimizable_total_def = cur_offset + cur_rigid_def
        # cur_optimizable_quaternion = prev_quaternion.detach() + cur_delta_quaternion #
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion # timestep # timestep #

        
        self.upd_rigid_acc = rigid_acc.clone()
        self.upd_rigid_def = cur_upd_rigid_def.clone()
        self.upd_optimizable_total_def = cur_optimizable_total_def.clone()
        self.upd_quaternion = cur_quaternion.clone()
        self.upd_rot_mtx = cur_optimizable_rot_mtx.clone()
        self.upd_angular_vel = cur_angular_vel.clone()
        self.upd_forces = forces.clone() ## ##
        
        ## 
        self.timestep_to_accum_acc[input_pts_ts] = rigid_acc.detach().clone()
        
        if not fix_obj:
            if input_pts_ts == 0 and input_pts_ts not in self.timestep_to_optimizable_total_def:
                self.timestep_to_total_def[input_pts_ts] = torch.zeros_like(cur_upd_rigid_def)
                self.timestep_to_optimizable_total_def[input_pts_ts] = torch.zeros_like(cur_optimizable_total_def)
                self.timestep_to_optimizable_quaternion[input_pts_ts] = torch.tensor([1., 0., 0., 0.],dtype=torch.float32).cuda()
                self.timestep_to_quaternion[input_pts_ts] = torch.tensor([1., 0., 0., 0.],dtype=torch.float32).cuda()
                self.timestep_to_angular_vel[input_pts_ts] = torch.zeros_like(cur_angular_vel).detach()
            self.timestep_to_total_def[nex_pts_ts] = cur_upd_rigid_def
            self.timestep_to_optimizable_total_def[nex_pts_ts] = cur_optimizable_total_def
            self.timestep_to_optimizable_quaternion[nex_pts_ts] = cur_quaternion
            self.timestep_to_quaternion[nex_pts_ts] = cur_quaternion.detach()
            
            cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
            self.timestep_to_optimizable_rot_mtx[nex_pts_ts] = cur_optimizable_rot_mtx
            # new_pts = raw_input_pts - cur_offset.unsqueeze(0)
            self.timestep_to_angular_vel[input_pts_ts] = cur_angular_vel.detach()
            
            ### 
            # self.timestep_to_optimizable_total_def[input_pts_ts + 1] = self.time_translations(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3)
        
        
        self.timestep_to_input_pts[input_pts_ts] = sampled_input_pts.detach()
        self.timestep_to_point_accs[input_pts_ts] = forces.detach()
        self.timestep_to_aggregation_weights[input_pts_ts] = cur_act_weights.detach()
        self.timestep_to_sampled_pts_to_passive_obj_dist[input_pts_ts] = dist_sampled_pts_to_passive_obj.detach()
        self.save_values = {
            'timestep_to_point_accs': {cur_ts: self.timestep_to_point_accs[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_point_accs}, 
            # 'timestep_to_vel': {cur_ts: self.timestep_to_vel[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_vel},
            'timestep_to_input_pts': {cur_ts: self.timestep_to_input_pts[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_input_pts},
            'timestep_to_aggregation_weights': {cur_ts: self.timestep_to_aggregation_weights[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_aggregation_weights},
            'timestep_to_sampled_pts_to_passive_obj_dist': {cur_ts: self.timestep_to_sampled_pts_to_passive_obj_dist[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_sampled_pts_to_passive_obj_dist}, # quaternion 
            # 'timestep_to_ws_normed': {cur_ts: self.timestep_to_ws_normed[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_ws_normed},
            # 'timestep_to_defed_input_pts_sdf': {cur_ts: self.timestep_to_defed_input_pts_sdf[cur_ts].cpu().numpy() for cur_ts in self.timestep_to_defed_input_pts_sdf},
        }
        
        return upd_contact_pairs_information
    
    def update_timestep_to_quantities(self, input_pts_ts, upd_quat, upd_trans):
        nex_pts_ts = input_pts_ts + 1
        self.timestep_to_total_def[nex_pts_ts] = upd_trans # .detach().clone().detach()
        self.timestep_to_optimizable_total_def[nex_pts_ts] = upd_trans # .detach().clone().detach()
        self.timestep_to_optimizable_quaternion[nex_pts_ts] = upd_quat # .detach().clone().detach()
        self.timestep_to_quaternion[nex_pts_ts] = upd_quat # .detach().clone().detach()
        
        # self.timestep_to_optimizable_rot_mtx[nex_pts_ts] = quaternion_to_matrix(upd_quat.detach().clone()).clone().detach()
        self.timestep_to_optimizable_rot_mtx[nex_pts_ts] = quaternion_to_matrix(upd_quat) # .clone().detach() # the upd quat #
        
        

    def reset_timestep_to_quantities(self, input_pts_ts):
        nex_pts_ts = input_pts_ts + 1
        self.timestep_to_accum_acc[input_pts_ts] = self.upd_rigid_acc.detach()
        self.timestep_to_total_def[nex_pts_ts] = self.upd_rigid_def
        self.timestep_to_optimizable_total_def[nex_pts_ts] = self.upd_optimizable_total_def
        self.timestep_to_optimizable_quaternion[nex_pts_ts] = self.upd_quaternion
        self.timestep_to_quaternion[nex_pts_ts] = self.upd_quaternion.detach()
        
        # cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
        self.timestep_to_optimizable_rot_mtx[nex_pts_ts] = self.upd_rot_mtx
        # new_pts = raw_input_pts - cur_offset.unsqueeze(0)
        
        self.timestep_to_angular_vel[input_pts_ts] = self.upd_angular_vel.detach()
        self.timestep_to_point_accs[input_pts_ts] = self.upd_forces.detach()
         





class BendingNetworkForward(nn.Module):
    def __init__(self,
                 d_in,
                 multires, # fileds #
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions, # hidden dimensions #
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False):

        super(BendingNetworkForward, self).__init__()
        self.use_positionally_encoded_input = False
        self.input_ch = 3
        self.output_ch = 3
        self.bending_n_timesteps = bending_n_timesteps
        self.bending_latent_size = bending_latent_size
        self.use_rigidity_network = use_rigidity_network
        self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        self.rigidity_network_depth = rigidity_network_depth
        self.rigidity_use_latent = rigidity_use_latent

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64
        self.network_depth = 5
        self.skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = False
        self.use_last_layer_bias = use_last_layer_bias

        self.embed_fn_fine = None # embed fn and the embed fn #
        if multires > 0: 
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
        
        ## timestep t ## # 
        
        # the bending latent # # passive and the active object at the current state #
        # extract meshes for the passive and the active object atthe current
        # step 1 -> get active object's points correspondences
        # step 2 -> get passive object's points in correspondences
        # step 3 -> for t-1, get the deformation for each point at t in the active robot mesh and average them as the averaged motion 
        # step 4 -> sample points from the passive mesh at t-1 and calculate their forces using the active robot's actions, signed distances, and the parameter K #
        # step 5 -> aggregate the translation motion (the most simple translation models) and use that as the deformation direction for the passive object at time t
        # step 6 -> an additional rigidity mask should be optimized for the passive object #
            
        # self.bending_latent = nn.Parameter(
        #     torch.zeros((self.bending_n_timesteps, self.bending_hi))
        # )
        self.bending_latent = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        )
        
        self.ks_val = nn.Embedding(
            num_embeddings=1, embedding_dim=1
        )
        torch.nn.init.ones_(self.ks_val.weight)
        self.ks_val.weight.data = self.ks_val.weight.data * 0.2
        
        # ks_val #
        self.network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0

        if self.use_rigidity_network:
            self.rigidity_activation_function = F.relu  # F.relu, torch.sin
            self.rigidity_skips = []  # do not include 0 and do not include depth-1 # 
            use_last_layer_bias = True
            self.rigidity_tanh = nn.Tanh()

            if self.rigidity_use_latent:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch + self.bending_latent_size, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])
            else:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.rigidity_network[:-1]):
                    if self.rigidity_activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                        # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.rigidity_activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights
                self.rigidity_network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.rigidity_network[-1].bias.data *= 0.0
                    
        self.rigid_translations = torch.tensor(
            [[ 0.0000,  0.0000,  0.0000],
            [-0.0008,  0.0040,  0.0159],
            [-0.0566,  0.0099,  0.0173]], dtype=torch.float32
        ).to(self.device)
        self.use_opt_rigid_translations = False # load utils and the loading .... ## 
        self.use_split_network = False
        
        self.timestep_to_forward_deform = {}
        
    def set_rigid_translations_optimizable(self, n_ts):
        if n_ts == 3:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0008,  0.0040,  0.0159],
                [-0.0566,  0.0099,  0.0173]], dtype=torch.float32, requires_grad=True
            ).to(self.device)
        elif n_ts == 5:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0097,  0.0305,  0.0342],
                [-0.1211,  0.1123,  0.0565],
                [-0.2700,  0.1271,  0.0412],
                [-0.3081,  0.1174,  0.0529]], dtype=torch.float32, requires_grad=False
            ).to(self.device)
        # self.rigid_translations.requires_grad = True
        # self.rigid_translations.requires_grad_ = True
        # self.rigid_translations = nn.Parameter(
        #     self.rigid_translations, requires_grad=True
        # )
        
    def set_split_bending_network(self, ):
        self.use_split_network = True
        # self.split_network = nn.ModuleList(
        #     [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)])
        ##### split network full #####
        # self.split_network = nn.ModuleList(
        #     [nn.ModuleList(
        #         [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #         [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #         if i + 1 in self.skips
        #         else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #         for i in range(self.network_depth - 2)] +
        #         [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
        #     ) for _ in range(self.bending_n_timesteps - 5)]
        # )
        # for i_split in range(len(self.split_network)):
        #     with torch.no_grad():
        #         for i, layer in enumerate(self.split_network[i_split][:-1]):
        #             if self.activation_function.__name__ == "sin":
        #                 # SIREN ( Implicit Neural Representations with Periodic Activation Functions
        #                 # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
        #                 if type(layer) == nn.Linear:
        #                     a = (
        #                         1.0 / layer.in_features
        #                         if i == 0
        #                         else np.sqrt(6.0 / layer.in_features)
        #                     )
        #                     layer.weight.uniform_(-a, a)
        #             elif self.activation_function.__name__ == "relu":
        #                 torch.nn.init.kaiming_uniform_(
        #                     layer.weight, a=0, mode="fan_in", nonlinearity="relu"
        #                 )
        #                 torch.nn.init.zeros_(layer.bias)

        #         # initialize final layer to zero weights to start out with straight rays
        #         self.split_network[i_split][-1].weight.data *= 0.0
        #         if self.use_last_layer_bias:
        #             self.split_network[i_split][-1].bias.data *= 0.0
        ##### split network full #####
        
        ##### split network single #####
        self.split_network = nn.ModuleList(
                [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
                [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
                if i + 1 in self.skips
                else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
                for i in range(self.network_depth - 2)] +
                [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
            )
        with torch.no_grad():
            for i, layer in enumerate(self.split_network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.split_network[-1].weight.data *= 0.0
            if self.use_last_layer_bias:
                self.split_network[-1].bias.data *= 0.0
        ##### split network single #####
        
    # the bending latent # # passive and the active object at the current state #
    # extract meshes for the passive and the active object atthe current
    # step 1 -> get active object's points correspondences
    # step 2 -> get passive object's points in correspondences
    # step 3 -> for t-1, get the deformation for each point at t in the active robot mesh and average them as the averaged motion 
    # step 4 -> sample points from the passive mesh at t-1 and calculate their forces using the active robot's actions, signed distances, and the parameter K #
    # step 5 -> aggregate the translation motion (the most simple translation models) and use that as the deformation direction for the passive object at time t
    # step 6 -> an additional rigidity mask should be optimized for the passive object #
    
    def forward(self, input_pts, input_pts_ts, timestep_to_mesh, timestep_to_passive_mesh, bending_net, bending_net_passive, act_sdf_net, details=None, special_loss_return=False): # special loss return # 
        
        # query the bending_net for the active obj's deformation flow #
        # query the bending_net_passive for the passive obj's deformation flow #
        # input_pts_ts should goes from zero to maxx_ts - 1 #
        nex_pts_ts = input_pts_ts + 1
        nex_timestep_mesh = timestep_to_mesh[nex_pts_ts]
        
        active_mesh_deformation = bending_net(nex_timestep_mesh, nex_pts_ts) ## get the nex_pts_ts's bending direction ##
        active_mesh_deformation = torch.mean(active_mesh_deformation, dim=0) ### (3, ) deformation direction ##
        # sdf value ? # 
        
        ks_val = self.ks_val(torch.zeros((1,)).long().to(self.device))
        
        self.ks_val_vals = ks_val.detach().cpu().numpy().item()
        
        # passive mesh # 
        cur_timestep_mesh = timestep_to_passive_mesh[input_pts_ts]
        passive_mesh_sdf_value = act_sdf_net.sdf(cur_timestep_mesh) # nn_passive_pts -> the shape of the passive pts's sdf values #
        maxx_dist = 2.
        fs = (-1. * torch.sin(passive_mesh_sdf_value / maxx_dist * float(np.pi) / 2.) + 1) * active_mesh_deformation.unsqueeze(0) * ks_val #### 
        fs = torch.mean(fs, dim=0) ## (3,) ##
        # fs = fs * -1.
        
        self.timestep_to_forward_deform[input_pts_ts] = fs.detach().cpu().numpy() # 
        
        
        raw_input_pts = input_pts[:, :3]  # positional encoding includes raw 3D coordinates as first three entries # 
        # print(f)
        if self.embed_fn_fine is not None: # embed fn #
            input_pts = self.embed_fn_fine(input_pts)

        
        if special_loss_return and details is None: # details is None #
            details = {}
            
        expanded_input_pts_ts = torch.zeros((input_pts.size(0)), dtype=torch.long).to(self.device)
        expanded_input_pts_ts = expanded_input_pts_ts + nex_pts_ts
        input_latents = self.bending_latent(expanded_input_pts_ts)

        # # print(f"input_pts: {input_pts.size()}, input_latents: {input_latents.size()}, raw_input_pts: {raw_input_pts.size()}")
        # input_latents = input_latent.expand(input_pts.size()[0], -1)
        # x = torch.cat([input_pts, input_latents], -1)  # input pts with bending latents # 
        
        x = fs.unsqueeze(0).repeat(input_pts.size(0), 1).contiguous() # 
        # x = self.rigid_translations(expanded_input_pts_ts) # .repeat(input_pts.size(0), 1).contiguous()

        unmasked_offsets = x
        if details is not None:
            details["unmasked_offsets"] = unmasked_offsets # get the unmasked offsets # 

        if self.use_rigidity_network: # bending network? rigidity network... # # bending network and the bending network # #
            if self.rigidity_use_latent:
                x = torch.cat([input_pts, input_latents], -1)
            else:
                x = input_pts

            for i, layer in enumerate(self.rigidity_network):
                x = layer(x)
                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    x *= 30.0
                if i != len(self.rigidity_network) - 1:
                    x = self.rigidity_activation_function(x)
                if i in self.rigidity_skips:
                    x = torch.cat([input_pts, x], -1)

            rigidity_mask = (self.rigidity_tanh(x) + 1) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection # rigidity 
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection
            # if input_pts_ts >= 5:
            #     avg_offsets_abs = torch.mean(torch.abs(unmasked_offsets), dim=0)
            #     print(f"input_ts: {input_pts_ts}, offset_avg: {avg_offsets_abs}")


        if special_loss_return:
            return details
        else:
            return new_points
        
        
        



class BendingNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions, # hidden dimensions #
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False):

        super(BendingNetwork, self).__init__()
        self.use_positionally_encoded_input = False
        self.input_ch = 3
        self.output_ch = 3
        self.bending_n_timesteps = bending_n_timesteps
        self.bending_latent_size = bending_latent_size
        self.use_rigidity_network = use_rigidity_network
        self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        self.rigidity_network_depth = rigidity_network_depth
        self.rigidity_use_latent = rigidity_use_latent

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64
        self.network_depth = 5
        self.skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = False
        self.use_last_layer_bias = use_last_layer_bias

        self.embed_fn_fine = None # embed fn and the embed fn # 
        if multires > 0: 
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            
        # self.bending_latent = nn.Parameter(
        #     torch.zeros((self.bending_n_timesteps, self.bending_hi))
        # )
        self.bending_latent = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        )
        

        self.network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0

        if self.use_rigidity_network:
            self.rigidity_activation_function = F.relu  # F.relu, torch.sin
            self.rigidity_skips = []  # do not include 0 and do not include depth-1 # 
            use_last_layer_bias = True
            self.rigidity_tanh = nn.Tanh()

            if self.rigidity_use_latent:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch + self.bending_latent_size, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])
            else:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.rigidity_network[:-1]):
                    if self.rigidity_activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                        # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.rigidity_activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights
                self.rigidity_network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.rigidity_network[-1].bias.data *= 0.0
                    
        self.rigid_translations = torch.tensor(
            [[ 0.0000,  0.0000,  0.0000],
            [-0.0008,  0.0040,  0.0159],
            [-0.0566,  0.0099,  0.0173]], dtype=torch.float32
        ).to(self.device)
        self.use_opt_rigid_translations = False # load utils and the loading .... ## 
        self.use_split_network = False
        
    def set_rigid_translations_optimizable(self, n_ts):
        if n_ts == 3:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0008,  0.0040,  0.0159],
                [-0.0566,  0.0099,  0.0173]], dtype=torch.float32, requires_grad=True
            ).to(self.device)
        elif n_ts == 5:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0097,  0.0305,  0.0342],
                [-0.1211,  0.1123,  0.0565],
                [-0.2700,  0.1271,  0.0412],
                [-0.3081,  0.1174,  0.0529]], dtype=torch.float32, requires_grad=False
            ).to(self.device)
        # self.rigid_translations.requires_grad = True
        # self.rigid_translations.requires_grad_ = True
        # self.rigid_translations = nn.Parameter(
        #     self.rigid_translations, requires_grad=True
        # )
        
    def set_split_bending_network(self, ):
        self.use_split_network = True
        # self.split_network = nn.ModuleList(
        #     [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)])
        ##### split network full #####
        # self.split_network = nn.ModuleList(
        #     [nn.ModuleList(
        #         [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #         [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #         if i + 1 in self.skips
        #         else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #         for i in range(self.network_depth - 2)] +
        #         [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
        #     ) for _ in range(self.bending_n_timesteps - 5)]
        # )
        # for i_split in range(len(self.split_network)):
        #     with torch.no_grad():
        #         for i, layer in enumerate(self.split_network[i_split][:-1]):
        #             if self.activation_function.__name__ == "sin":
        #                 # SIREN ( Implicit Neural Representations with Periodic Activation Functions
        #                 # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
        #                 if type(layer) == nn.Linear:
        #                     a = (
        #                         1.0 / layer.in_features
        #                         if i == 0
        #                         else np.sqrt(6.0 / layer.in_features)
        #                     )
        #                     layer.weight.uniform_(-a, a)
        #             elif self.activation_function.__name__ == "relu":
        #                 torch.nn.init.kaiming_uniform_(
        #                     layer.weight, a=0, mode="fan_in", nonlinearity="relu"
        #                 )
        #                 torch.nn.init.zeros_(layer.bias)

        #         # initialize final layer to zero weights to start out with straight rays
        #         self.split_network[i_split][-1].weight.data *= 0.0
        #         if self.use_last_layer_bias:
        #             self.split_network[i_split][-1].bias.data *= 0.0
        ##### split network full #####
        
        ##### split network single #####
        self.split_network = nn.ModuleList(
                [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
                [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
                if i + 1 in self.skips
                else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
                for i in range(self.network_depth - 2)] +
                [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
            )
        with torch.no_grad():
            for i, layer in enumerate(self.split_network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.split_network[-1].weight.data *= 0.0
            if self.use_last_layer_bias:
                self.split_network[-1].bias.data *= 0.0
        ##### split network single #####
        

    def forward(self, input_pts, input_pts_ts, details=None, special_loss_return=False): # special loss return # 
        raw_input_pts = input_pts[:, :3]  # positional encoding includes raw 3D coordinates as first three entries # 
        # print(f)
        if self.embed_fn_fine is not None: # embed fn #
            input_pts = self.embed_fn_fine(input_pts)

        if special_loss_return and details is None: # details is None #
            details = {}
            
        expanded_input_pts_ts = torch.zeros((input_pts.size(0)), dtype=torch.long).to(self.device)
        expanded_input_pts_ts = expanded_input_pts_ts + input_pts_ts
        input_latents = self.bending_latent(expanded_input_pts_ts)

        # print(f"input_pts: {input_pts.size()}, input_latents: {input_latents.size()}, raw_input_pts: {raw_input_pts.size()}")
        # input_latents = input_latent.expand(input_pts.size()[0], -1)
        x = torch.cat([input_pts, input_latents], -1)  # input pts with bending latents # 
        
        if (not self.use_split_network) or (self.use_split_network and input_pts_ts < 5):
            cur_network = self.network
        else:
            cur_network = self.split_network 
            # cur_network = self.split_network[input_pts_ts - 5]

        # use no grad # 
        # n_timesteps
        # if self.use_split_network and input_pts_ts < 5:
        # if self.use_split_network and input_pts_ts < self.n_timesteps - 1:
        #     with torch.no_grad():
        #         for i, layer in enumerate(cur_network):
        #             x = layer(x)
        #             # SIREN
        #             if self.activation_function.__name__ == "sin" and i == 0:
        #                 x *= 30.0
        #             if i != len(self.network) - 1:
        #                 x = self.activation_function(x)
        #             if i in self.skips:
        #                 x = torch.cat([input_pts, x], -1)
        # else:
        #     for i, layer in enumerate(cur_network):
        #         x = layer(x)
        #         # SIREN
        #         if self.activation_function.__name__ == "sin" and i == 0:
        #             x *= 30.0
        #         if i != len(self.network) - 1:
        #             x = self.activation_function(x)
        #         if i in self.skips:
        #             x = torch.cat([input_pts, x], -1)
        
        ''' use the single split network without no_grad setting '''
        for i, layer in enumerate(cur_network):
            x = layer(x)
            # SIREN
            if self.activation_function.__name__ == "sin" and i == 0:
                x *= 30.0
            if i != len(self.network) - 1:
                x = self.activation_function(x)
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
        ''' use the single split network without no_grad setting '''
        
        # if self.use_split_network:
        #     with torch.no_grad():
        #         for i, layer in enumerate(cur_network):
        #             x = layer(x)
        #             # SIREN
        #             if self.activation_function.__name__ == "sin" and i == 0:
        #                 x *= 30.0
        #             if i != len(self.network) - 1:
        #                 x = self.activation_function(x)
        #             if i in self.skips:
        #                 x = torch.cat([input_pts, x], -1)
        # else:
        #     for i, layer in enumerate(cur_network):
        #         x = layer(x)
        #         # SIREN
        #         if self.activation_function.__name__ == "sin" and i == 0:
        #             x *= 30.0
        #         if i != len(self.network) - 1:
        #             x = self.activation_function(x)
        #         if i in self.skips:
        #             x = torch.cat([input_pts, x], -1)

        unmasked_offsets = x
        if details is not None:
            details["unmasked_offsets"] = unmasked_offsets

        if self.use_rigidity_network: # bending network? rigidity network... # # bending network and the bending network # #
            if self.rigidity_use_latent:
                x = torch.cat([input_pts, input_latents], -1)
            else:
                x = input_pts

            for i, layer in enumerate(self.rigidity_network):
                x = layer(x)
                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    x *= 30.0
                if i != len(self.rigidity_network) - 1:
                    x = self.rigidity_activation_function(x)
                if i in self.rigidity_skips:
                    x = torch.cat([input_pts, x], -1)

            rigidity_mask = (self.rigidity_tanh(x) + 1) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection # rigidity 
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection
            # if input_pts_ts >= 5:
            #     avg_offsets_abs = torch.mean(torch.abs(unmasked_offsets), dim=0)
            #     print(f"input_ts: {input_pts_ts}, offset_avg: {avg_offsets_abs}")
            

        if self.use_opt_rigid_translations:
            def_points = self.rigid_translations.unsqueeze(0).repeat(raw_input_pts.size(0), 1, 1)
            def_points = batched_index_select(def_points, indices=expanded_input_pts_ts.unsqueeze(-1), dim=1).squeeze(1)
            if len(def_points.size()) > 2:
                def_points = def_points.squeeze(1)
            minn_raw_input_pts, _ = torch.min(raw_input_pts, dim=0)
            maxx_raw_input_pts, _ = torch.max(raw_input_pts, dim=0)
            # print(f"minn_raw_input_pts: {minn_raw_input_pts}, maxx_raw_input_pts: {maxx_raw_input_pts}")
            # get the 3D points tracking # optimize for the 3D points #
            # 
            new_points = raw_input_pts - def_points

        if special_loss_return:  # used for compute_divergence_loss() # long term motion and the tracking # long term 
            return details
        else:
            return new_points


# model 1 
# use rigid translations #
# use the rigidity network #
class BendingNetworkRigidTrans(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions, # hidden dimensions #
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False):

        super(BendingNetworkRigidTrans, self).__init__()
        self.use_positionally_encoded_input = False
        self.input_ch = 3
        self.output_ch = 3
        self.bending_n_timesteps = bending_n_timesteps
        self.bending_latent_size = bending_latent_size
        self.use_rigidity_network = use_rigidity_network
        self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        self.rigidity_network_depth = rigidity_network_depth
        self.rigidity_use_latent = rigidity_use_latent

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64
        self.network_depth = 5
        self.skips = []  # do not include 0 and do not include depth-1
        use_last_layer_bias = False
        self.use_last_layer_bias = use_last_layer_bias

        self.embed_fn_fine = None # embed fn and the embed fn # 
        if multires > 0: 
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            
        # self.bending_latent = nn.Parameter(
        #     torch.zeros((self.bending_n_timesteps, self.bending_hi))
        # )
        self.bending_latent = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        )
        
        self.rigid_translations = nn.Embedding(
            num_embeddings=self.bending_n_timesteps, embedding_dim=3 # get the 
        )
        # self.rigid_translations.weight.
        torch.nn.init.zeros_(self.rigid_translations.weight)
        

        self.network = nn.ModuleList(
            [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
            [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
             if i + 1 in self.skips
             else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
             for i in range(self.network_depth - 2)] +
            [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # initialize weights
        with torch.no_grad():
            for i, layer in enumerate(self.network[:-1]):
                if self.activation_function.__name__ == "sin":
                    # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                    # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                    if type(layer) == nn.Linear:
                        a = (
                            1.0 / layer.in_features
                            if i == 0
                            else np.sqrt(6.0 / layer.in_features)
                        )
                        layer.weight.uniform_(-a, a)
                elif self.activation_function.__name__ == "relu":
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    torch.nn.init.zeros_(layer.bias)

            # initialize final layer to zero weights to start out with straight rays
            self.network[-1].weight.data *= 0.0
            if use_last_layer_bias:
                self.network[-1].bias.data *= 0.0

        if self.use_rigidity_network:
            self.rigidity_activation_function = F.relu  # F.relu, torch.sin
            self.rigidity_skips = []  # do not include 0 and do not include depth-1 # 
            use_last_layer_bias = True
            self.rigidity_tanh = nn.Tanh()

            if self.rigidity_use_latent:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch + self.bending_latent_size, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])
            else:
                self.rigidity_network = nn.ModuleList(
                    [nn.Linear(self.input_ch, self.rigidity_hidden_dimensions)] +
                    [nn.Linear(self.input_ch + self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     if i + 1 in self.rigidity_skips
                     else nn.Linear(self.rigidity_hidden_dimensions, self.rigidity_hidden_dimensions)
                     for i in range(self.rigidity_network_depth - 2)] +
                    [nn.Linear(self.rigidity_hidden_dimensions, 1, bias=use_last_layer_bias)])

            # initialize weights
            with torch.no_grad():
                for i, layer in enumerate(self.rigidity_network[:-1]):
                    if self.rigidity_activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                        # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.rigidity_activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights
                self.rigidity_network[-1].weight.data *= 0.0
                if use_last_layer_bias:
                    self.rigidity_network[-1].bias.data *= 0.0
                    
        # self.rigid_translations = torch.tensor(
        #     [[ 0.0000,  0.0000,  0.0000],
        #     [-0.0008,  0.0040,  0.0159],
        #     [-0.0566,  0.0099,  0.0173]], dtype=torch.float32
        # ).to(self.device)
        # self.use_opt_rigid_translations = False # load utils and the loading .... ## 
        self.use_split_network = False
        
    def set_rigid_translations_optimizable(self, n_ts):
        if n_ts == 3:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0008,  0.0040,  0.0159],
                [-0.0566,  0.0099,  0.0173]], dtype=torch.float32, requires_grad=True
            ).to(self.device)
        elif n_ts == 5:
            self.rigid_translations = torch.tensor(
                [[ 0.0000,  0.0000,  0.0000],
                [-0.0097,  0.0305,  0.0342],
                [-0.1211,  0.1123,  0.0565],
                [-0.2700,  0.1271,  0.0412],
                [-0.3081,  0.1174,  0.0529]], dtype=torch.float32, requires_grad=False
            ).to(self.device)
        # self.rigid_translations.requires_grad = True
        # self.rigid_translations.requires_grad_ = True
        # self.rigid_translations = nn.Parameter(
        #     self.rigid_translations, requires_grad=True
        # )
        
    def set_split_bending_network(self, ):
        self.use_split_network = True
        # self.split_network = nn.ModuleList(
        #     [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)])
        self.split_network = nn.ModuleList(
            [nn.ModuleList(
                [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
                [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
                if i + 1 in self.skips
                else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
                for i in range(self.network_depth - 2)] +
                [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
            ) for _ in range(self.bending_n_timesteps - 5)]
        )
        for i_split in range(len(self.split_network)):
            with torch.no_grad():
                for i, layer in enumerate(self.split_network[i_split][:-1]):
                    if self.activation_function.__name__ == "sin":
                        # SIREN ( Implicit Neural Representations with Periodic Activation Functions
                        # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
                        if type(layer) == nn.Linear:
                            a = (
                                1.0 / layer.in_features
                                if i == 0
                                else np.sqrt(6.0 / layer.in_features)
                            )
                            layer.weight.uniform_(-a, a)
                    elif self.activation_function.__name__ == "relu":
                        torch.nn.init.kaiming_uniform_(
                            layer.weight, a=0, mode="fan_in", nonlinearity="relu"
                        )
                        torch.nn.init.zeros_(layer.bias)

                # initialize final layer to zero weights to start out with straight rays
                self.split_network[i_split][-1].weight.data *= 0.0
                if self.use_last_layer_bias:
                    self.split_network[i_split][-1].bias.data *= 0.0

    def forward(self, input_pts, input_pts_ts, details=None, special_loss_return=False): # special loss return # 
        raw_input_pts = input_pts[:, :3]  # positional encoding includes raw 3D coordinates as first three entries # 
        # print(f)
        if self.embed_fn_fine is not None: # embed fn #
            input_pts = self.embed_fn_fine(input_pts)

        if special_loss_return and details is None: # details is None #
            details = {}
            
        expanded_input_pts_ts = torch.zeros((input_pts.size(0)), dtype=torch.long).to(self.device)
        expanded_input_pts_ts = expanded_input_pts_ts + input_pts_ts
        # input_latents = self.bending_latent(expanded_input_pts_ts)

        # # print(f"input_pts: {input_pts.size()}, input_latents: {input_latents.size()}, raw_input_pts: {raw_input_pts.size()}")
        # # input_latents = input_latent.expand(input_pts.size()[0], -1)
        # x = torch.cat([input_pts, input_latents], -1)  # input pts with bending latents # 
        
        # if (not self.use_split_network) or (self.use_split_network and input_pts_ts < 5):
        #     cur_network = self.network
        # else:
        #     # cur_network = self.split_network 
        #     cur_network = self.split_network[input_pts_ts - 5]

        # # n_timesteps
        # # if self.use_split_network and input_pts_ts < 5:
        # if self.use_split_network and input_pts_ts < self.n_timesteps - 1:
        #     with torch.no_grad():
        #         for i, layer in enumerate(cur_network):
        #             x = layer(x)
        #             # SIREN
        #             if self.activation_function.__name__ == "sin" and i == 0:
        #                 x *= 30.0
        #             if i != len(self.network) - 1:
        #                 x = self.activation_function(x)
        #             if i in self.skips:
        #                 x = torch.cat([input_pts, x], -1)
        # else:
        #     for i, layer in enumerate(cur_network):
        #         x = layer(x)
        #         # SIREN
        #         if self.activation_function.__name__ == "sin" and i == 0:
        #             x *= 30.0
        #         if i != len(self.network) - 1:
        #             x = self.activation_function(x)
        #         if i in self.skips:
        #             x = torch.cat([input_pts, x], -1)

        x = self.rigid_translations(expanded_input_pts_ts) # .repeat(input_pts.size(0), 1).contiguous()

        unmasked_offsets = x
        if details is not None:
            details["unmasked_offsets"] = unmasked_offsets

        if self.use_rigidity_network: # bending network? rigidity network... # # bending network and the bending network # #
            if self.rigidity_use_latent:
                x = torch.cat([input_pts, input_latents], -1)
            else:
                x = input_pts

            for i, layer in enumerate(self.rigidity_network):
                x = layer(x)
                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    x *= 30.0
                if i != len(self.rigidity_network) - 1:
                    x = self.rigidity_activation_function(x)
                if i in self.rigidity_skips:
                    x = torch.cat([input_pts, x], -1)

            rigidity_mask = (self.rigidity_tanh(x) + 1) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection # rigidity 
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection
            # if input_pts_ts >= 5:
            #     avg_offsets_abs = torch.mean(torch.abs(unmasked_offsets), dim=0)
            #     print(f"input_ts: {input_pts_ts}, offset_avg: {avg_offsets_abs}")
            

        # if self.use_opt_rigid_translations:
        #     def_points = self.rigid_translations.unsqueeze(0).repeat(raw_input_pts.size(0), 1, 1)
        #     def_points = batched_index_select(def_points, indices=expanded_input_pts_ts.unsqueeze(-1), dim=1).squeeze(1)
        #     if len(def_points.size()) > 2:
        #         def_points = def_points.squeeze(1)
        #     minn_raw_input_pts, _ = torch.min(raw_input_pts, dim=0)
        #     maxx_raw_input_pts, _ = torch.max(raw_input_pts, dim=0)
        #     # print(f"minn_raw_input_pts: {minn_raw_input_pts}, maxx_raw_input_pts: {maxx_raw_input_pts}")
        #     # get the 3D points tracking # optimize for the 3D points #
        #     # 
        #     new_points = raw_input_pts - def_points

        if special_loss_return:  # used for compute_divergence_loss() # long term motion and the tracking # long term 
            return details
        else:
            return new_points


    def forward_delta(self, input_pts, input_pts_ts, details=None, special_loss_return=False): # special loss return # 
        raw_input_pts = input_pts[:, :3]  # positional encoding includes raw 3D coordinates as first three entries # 
        # print(f)
        if self.embed_fn_fine is not None: # embed fn #
            input_pts = self.embed_fn_fine(input_pts)

        if special_loss_return and details is None: # details is None #
            details = {}
            
        expanded_input_pts_ts = torch.zeros((input_pts.size(0)), dtype=torch.long).to(self.device)
        expanded_input_pts_ts = expanded_input_pts_ts + input_pts_ts
        # input_latents = self.bending_latent(expanded_input_pts_ts)

        # # print(f"input_pts: {input_pts.size()}, input_latents: {input_latents.size()}, raw_input_pts: {raw_input_pts.size()}")
        # # input_latents = input_latent.expand(input_pts.size()[0], -1)
        # x = torch.cat([input_pts, input_latents], -1)  # input pts with bending latents # 
        
        # if (not self.use_split_network) or (self.use_split_network and input_pts_ts < 5):
        #     cur_network = self.network
        # else:
        #     # cur_network = self.split_network 
        #     cur_network = self.split_network[input_pts_ts - 5]

        # # n_timesteps
        # # if self.use_split_network and input_pts_ts < 5:
        # if self.use_split_network and input_pts_ts < self.n_timesteps - 1:
        #     with torch.no_grad():
        #         for i, layer in enumerate(cur_network):
        #             x = layer(x)
        #             # SIREN
        #             if self.activation_function.__name__ == "sin" and i == 0:
        #                 x *= 30.0
        #             if i != len(self.network) - 1:
        #                 x = self.activation_function(x)
        #             if i in self.skips:
        #                 x = torch.cat([input_pts, x], -1)
        # else:
        #     for i, layer in enumerate(cur_network):
        #         x = layer(x)
        #         # SIREN
        #         if self.activation_function.__name__ == "sin" and i == 0:
        #             x *= 30.0
        #         if i != len(self.network) - 1:
        #             x = self.activation_function(x)
        #         if i in self.skips:
        #             x = torch.cat([input_pts, x], -1)

        if input_pts_ts > 0:
            ## x_{0} = x_t + off_t
            ## x_0 = x_{t-1} + off_t
            ## 0 = x_t - x_{t-1} + off_t - off_{t-1} --> x_{t-1} = x_t + off_t - off_{t-1}
            off_t = self.rigid_translations(expanded_input_pts_ts)
            prev_expanded_input_pts_ts = expanded_input_pts_ts - 1
            prev_off_t = self.rigid_translations(prev_expanded_input_pts_ts)
            x = off_t - prev_off_t
        else:
            off_t = self.rigid_translations(expanded_input_pts_ts)
            x = off_t 

        # x = self.rigid_translations(expanded_input_pts_ts) # .repeat(input_pts.size(0), 1).contiguous()

        unmasked_offsets = x
        if details is not None:
            details["unmasked_offsets"] = unmasked_offsets

        if self.use_rigidity_network: # bending network? rigidity network... # # bending network and the bending network # #
            if self.rigidity_use_latent:
                x = torch.cat([input_pts, input_latents], -1)
            else:
                x = input_pts

            for i, layer in enumerate(self.rigidity_network):
                x = layer(x)
                # SIREN
                if self.rigidity_activation_function.__name__ == "sin" and i == 0:
                    x *= 30.0
                if i != len(self.rigidity_network) - 1:
                    x = self.rigidity_activation_function(x)
                if i in self.rigidity_skips:
                    x = torch.cat([input_pts, x], -1)

            rigidity_mask = (self.rigidity_tanh(x) + 1) / 2  # close to 1 for nonrigid, close to 0 for rigid

            if self.rigidity_test_time_cutoff is not None:
                rigidity_mask[rigidity_mask <= self.rigidity_test_time_cutoff] = 0.0

        if self.use_rigidity_network:
            masked_offsets = rigidity_mask * unmasked_offsets
            if self.test_time_scaling is not None:
                masked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + masked_offsets  # skip connection # rigidity 
            if details is not None:
                details["rigidity_mask"] = rigidity_mask
                details["masked_offsets"] = masked_offsets
        else:
            if self.test_time_scaling is not None:
                unmasked_offsets *= self.test_time_scaling
            new_points = raw_input_pts + unmasked_offsets  # skip connection
            # if input_pts_ts >= 5:
            #     avg_offsets_abs = torch.mean(torch.abs(unmasked_offsets), dim=0)
            #     print(f"input_ts: {input_pts_ts}, offset_avg: {avg_offsets_abs}")
            

        # if self.use_opt_rigid_translations:
        #     def_points = self.rigid_translations.unsqueeze(0).repeat(raw_input_pts.size(0), 1, 1)
        #     def_points = batched_index_select(def_points, indices=expanded_input_pts_ts.unsqueeze(-1), dim=1).squeeze(1)
        #     if len(def_points.size()) > 2:
        #         def_points = def_points.squeeze(1)
        #     minn_raw_input_pts, _ = torch.min(raw_input_pts, dim=0)
        #     maxx_raw_input_pts, _ = torch.max(raw_input_pts, dim=0)
        #     # print(f"minn_raw_input_pts: {minn_raw_input_pts}, maxx_raw_input_pts: {maxx_raw_input_pts}")
        #     # get the 3D points tracking # optimize for the 3D points #
        #     # 
        #     new_points = raw_input_pts - def_points

        if special_loss_return:  # used for compute_divergence_loss() # long term motion and the tracking # long term 
            return details
        else:
            return new_points
