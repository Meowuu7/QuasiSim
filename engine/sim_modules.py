import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

# from scipy.spatial import KDTree
# from torch.utils.data.sampler import WeightedRandomSampler
# from torch.distributions.categorical import Categorical
# from torch.distributions.uniform import Uniform
from engine.math_utils import batched_index_select, update_quaternion, euler_to_quaternion, quaternion_to_matrix



class BendingNetworkActiveForceFieldForwardLagV15(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                #  bending_latent_size,
                 nn_instances=1,
                 minn_dist_threshold=0.05,
                 ):
        # bending network active force field #
        super(BendingNetworkActiveForceFieldForwardLagV15, self).__init__()
        self.use_positionally_encoded_input = False
        # self.input_ch = 3
        self.input_ch = 1
        d_in = self.input_ch
        # self.output_ch = 3
        self.output_ch = 1
        # self.bending_n_timesteps = bending_n_timesteps
        # self.bending_latent_size = bending_latent_size
        # self.use_rigidity_network = use_rigidity_network
        # self.rigidity_hidden_dimensions = rigidity_hidden_dimensions
        # self.rigidity_network_depth = rigidity_network_depth
        # self.rigidity_use_latent = rigidity_use_latent

        # simple scene editing. set to None during training.
        self.rigidity_test_time_cutoff = None
        self.test_time_scaling = None
        self.activation_function = F.relu  # F.relu, torch.sin
        self.hidden_dimensions = 64
        self.network_depth = 5
        self.contact_dist_thres = 0.1
        self.skips = []
        use_last_layer_bias = True
        self.use_last_layer_bias = use_last_layer_bias
        
        self.static_friction_mu = 1.

        self.embed_fn_fine = None
        if multires > 0: 
            embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
        
        self.nn_uniformly_sampled_pts = 50000
        
        self.cur_window_size = 60
        # self.bending_n_timesteps = self.cur_window_size + 10
        self.nn_patch_active_pts = 50
        self.nn_patch_active_pts = 1
        
        self.nn_instances = nn_instances
        
        self.contact_spring_rest_length = 2.
        
        # self.minn_dist_sampled_pts_passive_obj_thres = 0.05 # minn_dist_threshold ###
        self.minn_dist_sampled_pts_passive_obj_thres = minn_dist_threshold
        
        
        if self.nn_instances == 1:
            self.spring_ks_values = nn.Embedding(
                num_embeddings=5, embedding_dim=1
            )
            torch.nn.init.ones_(self.spring_ks_values.weight)
            self.spring_ks_values.weight.data = self.spring_ks_values.weight.data * 0.01
        else:
            self.spring_ks_values = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=5, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_ks_values in self.spring_ks_values:
                torch.nn.init.ones_(cur_ks_values.weight)
                cur_ks_values.weight.data = cur_ks_values.weight.data * 0.01
        
        
        # self.bending_latent = nn.Embedding(
        #     num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        # )
        
        # self.bending_dir_latent = nn.Embedding(
        #     num_embeddings=self.bending_n_timesteps, embedding_dim=self.bending_latent_size
        # )
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
        
        
        
        self.actuator_weights = nn.Embedding( # actuator's forces #
            num_embeddings=self.nn_actuation_forces + 10, embedding_dim=1
        )
        torch.nn.init.ones_(self.actuator_weights.weight) # 
        self.actuator_weights.weight.data = self.actuator_weights.weight.data * (1. / (778 * 2))
        
        
        ''' patch force network and the patch force scale network ''' 
        # self.patch_force_network = nn.ModuleList(
        #     [
        #         nn.Sequential(nn.Linear(3, self.hidden_dimensions), nn.ReLU()),
        #         nn.Sequential(nn.Linear(self.hidden_dimensions, self.hidden_dimensions)), # with maxpoll layers # 
        #         nn.Sequential(nn.Linear(self.hidden_dimensions * 2, self.hidden_dimensions), nn.ReLU()), # 
        #         nn.Sequential(nn.Linear(self.hidden_dimensions, 3)), # hidden_dimension x 1 -> the weights # 
        #     ]
        # )
        
        # with torch.no_grad():
        #     for i, layer in enumerate(self.patch_force_network[:]):
        #         for cc in layer:
        #             if isinstance(cc, nn.Linear):
        #                 torch.nn.init.kaiming_uniform_(
        #                     cc.weight, a=0, mode="fan_in", nonlinearity="relu"
        #                 )
        #                 if i < len(self.patch_force_network) - 1:
        #                     torch.nn.init.zeros_(cc.bias)
        #         # torch.nn.init.zeros_(layer.bias)
        
        # self.patch_force_scale_network = nn.ModuleList(
        #     [
        #         nn.Sequential(nn.Linear(3, self.hidden_dimensions), nn.ReLU()),
        #         nn.Sequential(nn.Linear(self.hidden_dimensions, self.hidden_dimensions)), # with maxpoll layers # 
        #         nn.Sequential(nn.Linear(self.hidden_dimensions * 2, self.hidden_dimensions), nn.ReLU()), # 
        #         nn.Sequential(nn.Linear(self.hidden_dimensions, 1)), # hidden_dimension x 1 -> the weights # 
        #     ]
        # )
        
        # with torch.no_grad():
        #     for i, layer in enumerate(self.patch_force_scale_network[:]):
        #         for cc in layer:
        #             if isinstance(cc, nn.Linear): ### ifthe lienar layer # # ## 
        #                 torch.nn.init.kaiming_uniform_(
        #                     cc.weight, a=0, mode="fan_in", nonlinearity="relu"
        #                 )
        #                 if i < len(self.patch_force_scale_network) - 1:
        #                     torch.nn.init.zeros_(cc.bias)
        ''' patch force network and the patch force scale network ''' 
        
        # self.input_ch = 1
        # self.network = nn.ModuleList(
        #     [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, self.output_ch, bias=use_last_layer_bias)])

        # # initialize weights
        # with torch.no_grad():
        #     for i, layer in enumerate(self.network[:-1]):
        #         if self.activation_function.__name__ == "sin":
        #             # SIREN ( Implicit Neural Representations with Periodic Activation Functions
        #             # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
        #             if type(layer) == nn.Linear:
        #                 a = (
        #                     1.0 / layer.in_features
        #                     if i == 0
        #                     else np.sqrt(6.0 / layer.in_features)
        #                 )
        #                 layer.weight.uniform_(-a, a)
        #         elif self.activation_function.__name__ == "relu":
        #             torch.nn.init.kaiming_uniform_(
        #                 layer.weight, a=0, mode="fan_in", nonlinearity="relu"
        #             )
        #             torch.nn.init.zeros_(layer.bias)
                    
        #     # initialize final layer to zero weights to start out with straight rays
        #     self.network[-1].weight.data *= 0.0
        #     if use_last_layer_bias:
        #         self.network[-1].bias.data *= 0.0
        #         self.network[-1].bias.data += 0.2

        # self.dir_network = nn.ModuleList(
        #     [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, 3)])

        # with torch.no_grad():
        #     for i, layer in enumerate(self.dir_network[:]):
        #         if self.activation_function.__name__ == "sin":
        #             # SIREN ( Implicit Neural Representations with Periodic Activation Functions
        #             # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
        #             if type(layer) == nn.Linear:
        #                 a = (
        #                     1.0 / layer.in_features
        #                     if i == 0
        #                     else np.sqrt(6.0 / layer.in_features)
        #                 )
        #                 layer.weight.uniform_(-a, a)
        #         elif self.activation_function.__name__ == "relu":
        #             torch.nn.init.kaiming_uniform_(
        #                 layer.weight, a=0, mode="fan_in", nonlinearity="relu"
        #             )
        #             torch.nn.init.zeros_(layer.bias)
                    
        ## weighting_network for the network ##
        # self.weighting_net_input_ch = 3
        # self.weighting_network = nn.ModuleList(
        #     [nn.Linear(self.weighting_net_input_ch + self.bending_latent_size, self.hidden_dimensions)] +
        #     [nn.Linear(self.weighting_net_input_ch + self.hidden_dimensions, self.hidden_dimensions)
        #      if i + 1 in self.skips
        #      else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
        #      for i in range(self.network_depth - 2)] +
        #     [nn.Linear(self.hidden_dimensions, 1)])

        # with torch.no_grad():
        #     for i, layer in enumerate(self.weighting_network[:]):
        #         if self.activation_function.__name__ == "sin": # periodict activation functions #
        #             # SIREN ( Implicit Neural Representations with Periodic Activation Functions
        #             # https://arxiv.org/pdf/2006.09661.pdf Sec. 3.2)
        #             if type(layer) == nn.Linear:
        #                 a = (
        #                     1.0 / layer.in_features
        #                     if i == 0
        #                     else np.sqrt(6.0 / layer.in_features)
        #                 )
        #                 layer.weight.uniform_(-a, a)
        #         elif self.activation_function.__name__ == "relu":
        #             torch.nn.init.kaiming_uniform_(
        #                 layer.weight, a=0, mode="fan_in", nonlinearity="relu"
        #             )
        #             if i < len(self.weighting_network) - 1:
        #                 torch.nn.init.zeros_(layer.bias)
        
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
        # timestep_to_prev_selected_active_mesh_ori, timestep_to_prev_selected_active_mesh # 
        self.timestep_to_prev_selected_active_mesh_ori = {}
        self.timestep_to_prev_selected_active_mesh = {}
        
        self.timestep_to_spring_forces = {}
        self.timestep_to_spring_forces_ori = {}
        
        # timestep_to_angular_vel, timestep_to_quaternion # 
        self.timestep_to_angular_vel = {}
        self.timestep_to_quaternion = {}
        self.timestep_to_torque = {}
        
        
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion
        self.timestep_to_optimizable_total_def = {}
        self.timestep_to_optimizable_quaternion = {}
        self.timestep_to_optimizable_rot_mtx = {}
        self.timestep_to_aggregation_weights = {}
        self.timestep_to_sampled_pts_to_passive_obj_dist = {}

        self.time_quaternions = nn.Embedding(
            num_embeddings=60, embedding_dim=4
        )
        self.time_quaternions.weight.data[:, 0] = 1.
        self.time_quaternions.weight.data[:, 1] = 0.
        self.time_quaternions.weight.data[:, 2] = 0.
        self.time_quaternions.weight.data[:, 3] = 0.
        # torch.nn.init.ones_(self.time_quaternions.weight) # 
        
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


        
    # def set_split_bending_network(self, ):
    #     self.use_split_network = True
    #     ##### split network single ##### ## 
    #     self.split_network = nn.ModuleList(
    #         [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
    #         [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
    #         if i + 1 in self.skips
    #         else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
    #         for i in range(self.network_depth - 2)] +
    #         [nn.Linear(self.hidden_dimensions, self.output_ch, bias=self.use_last_layer_bias)]
    #     )
    #     with torch.no_grad():
    #         for i, layer in enumerate(self.split_network[:-1]):
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
    #         self.split_network[-1].weight.data *= 0.0
    #         if self.use_last_layer_bias:
    #             self.split_network[-1].bias.data *= 0.0
    #             self.split_network[-1].bias.data += 0.2
    #     ##### split network single #####
        
        
    #     self.split_dir_network = nn.ModuleList(
    #         [nn.Linear(self.input_ch + self.bending_latent_size, self.hidden_dimensions)] +
    #         [nn.Linear(self.input_ch + self.hidden_dimensions, self.hidden_dimensions)
    #         if i + 1 in self.skips
    #         else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
    #         for i in range(self.network_depth - 2)] +
    #         [nn.Linear(self.hidden_dimensions, 3)]
    #     )
    #     with torch.no_grad(): # no_grad()
    #         for i, layer in enumerate(self.split_dir_network[:]):
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

    #         # initialize final layer to zero weights to start out with straight rays #
    #         # 
    #         # self.split_dir_network[-1].weight.data *= 0.0
    #         # if self.use_last_layer_bias:
    #         #     self.split_dir_network[-1].bias.data *= 0.0
    #     ##### split network single #####
        
        
    #     # ### 
    #     ## weighting_network for the network ##
    #     self.weighting_net_input_ch = 3
    #     self.split_weighting_network = nn.ModuleList(
    #         [nn.Linear(self.weighting_net_input_ch + self.bending_latent_size, self.hidden_dimensions)] +
    #         [nn.Linear(self.weighting_net_input_ch + self.hidden_dimensions, self.hidden_dimensions)
    #          if i + 1 in self.skips
    #          else nn.Linear(self.hidden_dimensions, self.hidden_dimensions)
    #          for i in range(self.network_depth - 2)] +
    #         [nn.Linear(self.hidden_dimensions, 1)])

    #     with torch.no_grad():
    #         for i, layer in enumerate(self.split_weighting_network[:]):
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
    #                 if i < len(self.split_weighting_network) - 1:
    #                     torch.nn.init.zeros_(layer.bias)
    
    # def uniformly_sample_pts(self, tot_pts, nn_samples):
    #     tot_pts_prob = torch.ones_like(tot_pts[:, 0])
    #     tot_pts_prob = tot_pts_prob / torch.sum(tot_pts_prob)
    #     pts_dist = Categorical(tot_pts_prob)
    #     sampled_pts_idx = pts_dist.sample((nn_samples,))
    #     sampled_pts_idx = sampled_pts_idx.squeeze()
    #     sampled_pts = tot_pts[sampled_pts_idx]
    #     return sampled_pts
    
    
    def query_for_sdf(self, cur_pts, cur_frame_transformations):
        # 
        cur_frame_rotation, cur_frame_translation = cur_frame_transformations
        # cur_pts: nn_pts x 3 #
        # print(f"cur_pts: {cur_pts.size()}, cur_frame_translation: {cur_frame_translation.size()}, cur_frame_rotation: {cur_frame_rotation.size()}")
        cur_transformed_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0)
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
        return cur_pts_sdf # # cur_pts_sdf # 
    
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

        # friction_qd = 0.1 #
        sampled_input_pts = timestep_to_active_mesh[input_pts_ts] # sampled points --> sampled points #
        ori_nns = sampled_input_pts.size(0)
        if sampled_verts_idxes is not None:
            sampled_input_pts = sampled_input_pts[sampled_verts_idxes]
        nn_sampled_input_pts = sampled_input_pts.size(0)
        
        
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
        
        ''' friction forces from optimized frictions '''
        if friction_forces is None:
            ###### get the friction forces #####
            if self.nn_instances == 1:
                cur_actuation_friction_forces = self.actuator_friction_forces(cur_actuation_embedding_idxes)
            else:
                cur_actuation_friction_forces = self.actuator_friction_forces[i_instance](cur_actuation_embedding_idxes)
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
        
        
        # sampled_input_pts #
        # inter_obj_pts #
        # inter_obj_normals 
        
        # nn_sampledjpoints #
        # cur_passive_obj_ns # # inter obj normals # # 
        inter_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj]
        inter_obj_pts = cur_passive_obj_verts[minn_idx_sampled_pts_to_passive_obj]
        
        cur_passive_obj_verts_pts_idxes = torch.arange(0, cur_passive_obj_verts.size(0), dtype=torch.long).cuda() # 
        inter_passive_obj_pts_idxes = cur_passive_obj_verts_pts_idxes[minn_idx_sampled_pts_to_passive_obj]
        
        # the contact point # 
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
        
        # ws_beta; 10 #
        ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha * 10) # ws_alpha #
        ####### sharp the weights #######
        
        # minn_dist_sampled_pts_passive_obj_thres = 0.05
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001
        minn_dist_sampled_pts_passive_obj_thres = self.minn_dist_sampled_pts_passive_obj_thres
        
        
        # # ws_unnormed = ws_normed_sampled
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # rigid_acc = torch.sum(forces * ws_normed.unsqueeze(-1), dim=0)
        
        #### using network weights ####
        # cur_act_weights = self.actuator_weights(cur_actuation_embedding_idxes).squeeze(-1)
        #### using network weights ####
        
        ### nearest ####
        ''' decide forces via kinematics statistics '''
        ### neares
        rel_inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts # inter_obj_pts #
        dot_rel_inter_obj_pts_normals = torch.sum(rel_inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1) ## nn_sampled_pts
        dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] = -1. * dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0]
        # contact_spring_ka * | minn_spring_length - dist_sampled_pts_to_passive_obj |
        
        
        in_contact_indicator = dist_sampled_pts_to_passive_obj <= minn_dist_sampled_pts_passive_obj_thres
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha )
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # cur_act_weights = ws_normed
        cur_act_weights = ws_unnormed
        
        # penetrating #
        ### penetration strategy v4 ####
        
        if input_pts_ts > 0:
            cur_rot = self.timestep_to_optimizable_rot_mtx[input_pts_ts].detach()
            cur_trans = self.timestep_to_total_def[input_pts_ts].detach()
            queried_sdf = self.query_for_sdf(sampled_input_pts, (cur_rot, cur_trans))
            penetrating_indicator = queried_sdf < 0
        else:
            penetrating_indicator = torch.zeros_like(dot_inter_obj_pts_to_sampled_pts_normals).bool()
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
        
        
        
        ###### the contact force decided by the rest_length ######
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
            cur_contact_passive_pos_from_active = contact_passive_pos + diff_transformed_prev_contact_passive_to_active
            
            friction_k = 1.0
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            # penalty_based_friction_forces = friction_k * (contact_active_pos - transformed_prev_contact_active_pos)
            
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            penalty_based_friction_forces = friction_k * (cur_contact_passive_pos_from_active - contact_passive_pos)
            ''' Penalty based contact force v2 '''
            
            
            valid_contact_force_d_scalar = contact_force_d_scalar[contact_active_idxes]
            
            
            # penalty_based_friction_forces #
            norm_penalty_based_friction_forces = torch.norm(penalty_based_friction_forces, dim=-1, p=2)
            # valid penalty friction forces # # valid contact force d scalar #
            valid_penalty_friction_forces_indicator = norm_penalty_based_friction_forces <= (valid_contact_force_d_scalar * self.static_friction_mu)
            # valid_penalty_friction_forces_indicator[:] = True

            # norm and the penalty friction forces
            # sampled_input_pts
            # inter_obj_pts
            # inter_obj_normals
            # in_contact_indicator
            # contact passive pose # contac frame translation #
            
            
            # if torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5: # penalty based friction forces #
            summ_valid_penalty_friction_forces_indicator = torch.sum(valid_penalty_friction_forces_indicator.float())
            
            
            # print(f"penalty_based_friction_forces: {penalty_based_friction_forces.size()}, summ_valid_penalty_friction_forces_indicator: {summ_valid_penalty_friction_forces_indicator}")
            # tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.005 #  * 1000.
            
            # contact_spring_kb
            # tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * contact_spring_kb
            
            penalty_friction_tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * contact_spring_kb
             
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * contact_spring_kb
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000. # based friction 
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.02 
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.05 #
        # tangential forces with inter obj normals # -> 
        dot_tangential_forces_with_inter_obj_normals = torch.sum(penalty_friction_tangential_forces * inter_obj_normals, dim=-1) ### nn_active_pts x 
        penalty_friction_tangential_forces = penalty_friction_tangential_forces - dot_tangential_forces_with_inter_obj_normals.unsqueeze(-1) * inter_obj_normals
        
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2)
        # maxx_norm_tangential, _ = torch.max(norm_tangential_forces, dim=-1)
        # minn_norm_tangential, _ = torch.min(norm_tangential_forces, dim=-1)
        # print(f"maxx_norm_tangential: {maxx_norm_tangential}, minn_norm_tangential: {minn_norm_tangential}")
        
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
            cur_in_contact_active_pts_all = torch.arange(0, sampled_input_pts.size(0)).long().cuda() ## in_contact_active_pts_all ###
            cur_in_contact_active_pts_all = cur_in_contact_active_pts_all[in_contact_indicator] # activejpoints #
            
            cur_inter_passive_obj_pts_idxes = inter_passive_obj_pts_idxes[in_contact_indicator] # inter_passive_obj_pts_idxes #
            # contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose 
            # cur_contact_frame_pose = (cur_contact_frame_rotations, cur_contact_frame_translations ) 
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
        ####### if contact_pairs_set is not None and torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5: ########
        if contact_pairs_set is not None and torch.sum(valid_penalty_friction_forces_indicator.float()) > 0.5:
            # contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
            prev_contact_active_point_pts = contact_active_point_pts[valid_penalty_friction_forces_indicator]
            prev_contact_point_position = contact_point_position[valid_penalty_friction_forces_indicator]
            prev_contact_active_idxes = contact_active_idxes[valid_penalty_friction_forces_indicator]
            prev_contact_passive_idxes = contact_passive_idxes[valid_penalty_friction_forces_indicator]
            prev_contact_frame_rotations = contact_frame_orientations[valid_penalty_friction_forces_indicator]
            prev_contact_frame_translations = contact_frame_translations[valid_penalty_friction_forces_indicator]
            
            tot_contact_active_point_pts.append(prev_contact_active_point_pts)
            tot_contact_point_position.append(prev_contact_point_position)
            tot_contact_active_idxes.append(prev_contact_active_idxes)
            tot_contact_passive_idxes.append(prev_contact_passive_idxes)
            tot_contact_frame_rotations.append(prev_contact_frame_rotations)
            tot_contact_frame_translations.append(prev_contact_frame_translations)
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
            
        
        
        
        # new contacts? #
        
        
        
        # if the scale of the penalty based friction forces is larger than contact force * static_friction_mu; then change it to the dynamic friction force #
        
        ##### deal with the contact issues #####
        # find contact pairs with dynamic frictions and screen them out from the contact pair #
        # find new contact pairs #
        # related to tangential velocity # # related to the tangential velocity #
        
        # initially the penetrations should be neglected and gradually it should be considered to generate contact forces #
        # the gradient of the sdf field? # 
        ### strict cosntraints ###
        # mult_weights = torch.ones_like(norm_along_normals_forces).detach()
        # hard_selector = norm_tangential_forces > self.static_friction_mu * norm_along_normals_forces
        # hard_selector = hard_selector.detach()
        # mult_weights[hard_selector] = self.static_friction_mu * norm_along_normals_forces.detach()[hard_selector] / norm_tangential_forces.detach()[hard_selector]
        # ### change to the strict constraint ###
        # # tangential_forces[norm_tangential_forces > self.static_friction_mu * norm_along_normals_forces] = tangential_forces[norm_tangential_forces > self.static_friction_mu * norm_along_normals_forces] / norm_tangential_forces[norm_tangential_forces > self.static_friction_mu * norm_along_normals_forces].unsqueeze(-1) * self.static_friction_mu * norm_along_normals_forces[norm_tangential_forces > self.static_friction_mu * norm_along_normals_forces].unsqueeze(-1)
        # ### change to the strict constraint ###
        
        # # tangential forces # #
        # tangential_forces = tangential_forces * mult_weights.unsqueeze(-1)
        ### strict cosntraints ###
        
        ## penalty friction tangential forces ##
        # forces = tangential_forces + contact_force_d # 
        # tantential forces and contact force #
        forces = penalty_friction_tangential_forces + contact_force_d
        ''' decide forces via kinematics statistics '''

        ''' Decompose forces and calculate penalty froces ''' # 
        # penalty_dot_forces_normals, penalty_friction_constraint # # contraints # # 
        # # get the forces -> decompose forces # 
        dot_forces_normals = torch.sum(inter_obj_normals * forces, dim=-1) ### nn_sampled_pts ###
        forces_along_normals = dot_forces_normals.unsqueeze(-1) * inter_obj_normals ## the forces along the normal direction ##
        tangential_forces = forces - forces_along_normals # tangential forces # # tangential forces ### tangential forces ##
        # penalty_friction_tangential_forces = force - 
        
        
        #### penalty_friction_tangential_forces, tangential_forces ####
        self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
        self.tangential_forces = tangential_forces
    
        
        penalty_dot_forces_normals = dot_forces_normals ** 2
        penalty_dot_forces_normals[dot_forces_normals <= 0] = 0 # 1) must in the negative direction of the object normal #
        penalty_dot_forces_normals = torch.mean(penalty_dot_forces_normals) # 1) must # 2) must #
        self.penalty_dot_forces_normals = penalty_dot_forces_normals #
        
        
        rigid_acc = torch.sum(forces * cur_act_weights.unsqueeze(-1), dim=0) # rigid acc #
        rigid_acc = rigid_acc + self.gravity_acc * self.gravity_dir # gravity acc #
        
        
        #  = rigid_acc + 
        
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



# 
class BendingNetworkActiveForceFieldForwardLagV16(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions,
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False,
                 nn_instances=1,
                 minn_dist_threshold=0.05,
                 ): 
        # bending network active force field #
        super(BendingNetworkActiveForceFieldForwardLagV16, self).__init__()
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

        # simple scene editing. set to None during training. #
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
        
        self.cur_window_size = 60
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
        else:
            self.spring_ks_values = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=5, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_ks_values in self.spring_ks_values:
                torch.nn.init.ones_(cur_ks_values.weight)
                cur_ks_values.weight.data = cur_ks_values.weight.data * 0.01
        
        
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
        self.sep_damping_constant.weight.data = self.sep_damping_constant.weight.data * 0.9
        
        
        self.sep_angular_damping_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_angular_damping_constant.weight) # # # #
        self.sep_angular_damping_constant.weight.data = self.sep_angular_damping_constant.weight.data * 0.9
    
        
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
        torch.nn.init.zeros_(self.actuator_forces.weight)
        
        
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
        # timestep_to_prev_selected_active_mesh_ori, timestep_to_prev_selected_active_mesh # 
        self.timestep_to_prev_selected_active_mesh_ori = {}
        self.timestep_to_prev_selected_active_mesh = {}
        
        self.timestep_to_spring_forces = {}
        self.timestep_to_spring_forces_ori = {}
        
        # timestep_to_angular_vel, timestep_to_quaternion # 
        self.timestep_to_angular_vel = {}
        self.timestep_to_quaternion = {}
        self.timestep_to_torque = {}
        
        
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion
        self.timestep_to_optimizable_total_def = {}
        self.timestep_to_optimizable_quaternion = {}
        self.timestep_to_optimizable_rot_mtx = {}
        self.timestep_to_aggregation_weights = {}
        self.timestep_to_sampled_pts_to_passive_obj_dist = {}

        self.time_quaternions = nn.Embedding(
            num_embeddings=60, embedding_dim=4
        )
        self.time_quaternions.weight.data[:, 0] = 1.
        self.time_quaternions.weight.data[:, 1] = 0.
        self.time_quaternions.weight.data[:, 2] = 0.
        self.time_quaternions.weight.data[:, 3] = 0.
        # torch.nn.init.ones_(self.time_quaternions.weight) # 
        
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
        cur_transformed_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0)
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
        return cur_pts_sdf # # cur_pts_sdf # 
    
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
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001
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


    ### forward; 
    def forward2(self, input_pts_ts, timestep_to_active_mesh, timestep_to_passive_mesh, timestep_to_passive_mesh_normals, details=None, special_loss_return=False, update_tot_def=True, friction_forces=None, i_instance=0, reference_mano_pts=None, sampled_verts_idxes=None, fix_obj=False, contact_pairs_set=None, ):
        #### contact_pairs_set ####

        nex_pts_ts = input_pts_ts + 1

        ''' Kinematics transformations from acc and torques '''
        # rigid_acc = self.time_forces(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3)
        # torque = self.time_torques(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(3)
        ''' Kinematics transformations from acc and torques '''

        sampled_input_pts = timestep_to_active_mesh[input_pts_ts] # sampled points --> sampled points #
        # ori_nns = sampled_input_pts.size(0)
        if sampled_verts_idxes is not None:
            sampled_input_pts = sampled_input_pts[sampled_verts_idxes]
        # nn_sampled_input_pts = sampled_input_pts.size(0)
        
        if nex_pts_ts in timestep_to_active_mesh: ##
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
        
        
        init_passive_obj_verts = timestep_to_passive_mesh[0]
        init_passive_obj_ns = timestep_to_passive_mesh_normals[0]
        center_init_passive_obj_verts = init_passive_obj_verts.mean(dim=0)
        
        # cur_passive_obj_rot, cur_passive_obj_trans # 
        cur_passive_obj_rot = quaternion_to_matrix(self.timestep_to_quaternion[input_pts_ts].detach())
        cur_passive_obj_trans = self.timestep_to_total_def[input_pts_ts].detach()
        
        
        cur_passive_obj_verts = torch.matmul(cur_passive_obj_rot, (init_passive_obj_verts - center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)).transpose(1, 0) + center_init_passive_obj_verts.squeeze(0) + cur_passive_obj_trans.unsqueeze(0) # 
        cur_passive_obj_ns = torch.matmul(cur_passive_obj_rot, init_passive_obj_ns.transpose(1, 0).contiguous()).transpose(1, 0).contiguous()
        cur_passive_obj_ns = cur_passive_obj_ns / torch.clamp(torch.norm(cur_passive_obj_ns, dim=-1, keepdim=True), min=1e-8)
        cur_passive_obj_center = center_init_passive_obj_verts + cur_passive_obj_trans
        passive_center_point = cur_passive_obj_center # obj_center #
        
        # cur_active_mesh = timestep_to_active_mesh[input_pts_ts] #
        # nex_active_mesh = timestep_to_active_mesh[input_pts_ts + 1] #
        
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
        
        # # assue the passive object cannot move at all # #


        cur_actuation_embedding_st_idx = self.nn_actuators * input_pts_ts
        cur_actuation_embedding_ed_idx = self.nn_actuators * (input_pts_ts + 1)
        cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
        # ######### optimize the actuator forces directly #########
        # cur_actuation_forces = self.actuator_forces(cur_actuation_embedding_idxes) # actuation embedding idxes #
        # forces = cur_actuation_forces
        # ######### optimize the actuator forces directly #########


        if self.nn_instances == 1:
            ws_alpha = self.ks_weights(torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights(torch.ones((1,)).long().cuda()).view(1)
        else:
            ws_alpha = self.ks_weights[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        
        if self.use_sqrt_dist:
            dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
            )
        else:
            dist_sampled_pts_to_passive_obj = torch.sum( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1
            )
        
        # ### add the sqrt for calculate the l2 distance ###
        # dist_sampled_pts_to_passive_obj = torch.sqrt(dist_sampled_pts_to_passive_obj) ### #
        # ### add the sqrt for ### #
        
        
        ## dist dampled ## ## dist sampled ## # 
        # dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
        #     (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
        # )
        
        dist_sampled_pts_to_passive_obj, minn_idx_sampled_pts_to_passive_obj = torch.min(dist_sampled_pts_to_passive_obj, dim=-1)
        
        
        
        # inte robj normals at the current frame # # 
        inter_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj]
        inter_obj_pts = cur_passive_obj_verts[minn_idx_sampled_pts_to_passive_obj]
        
        # ### passive obj verts pts idxes ### # 
        cur_passive_obj_verts_pts_idxes = torch.arange(0, cur_passive_obj_verts.size(0), dtype=torch.long).cuda() # 
        inter_passive_obj_pts_idxes = cur_passive_obj_verts_pts_idxes[minn_idx_sampled_pts_to_passive_obj]
        
        # inter_obj_normals #
        inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts.detach()
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
        ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha * 10) # ws_alpha #
        ####### sharp the weights #######
        
        # minn_dist_sampled_pts_passive_obj_thres = 0.05
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001
        minn_dist_sampled_pts_passive_obj_thres = self.minn_dist_sampled_pts_passive_obj_thres
        
        
       
        # # ws_unnormed = ws_normed_sampled
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9) e
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
            # penetrating_indicator = torch.zeros_like(dot_inter_obj_pts_to_sampled_pts_normals).bool()
            penetrating_indicator = torch.zeros((sampled_input_pts.size(0),), dtype=torch.bool).cuda().bool()
        
        # if contact_pairs_set is not None and self.use_penalty_based_friction and (not self.use_disp_based_friction):
        #     penetrating
        
        ''' decide forces via kinematics statistics '''
        # rel_inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts # inter_obj_pts #
        # dot_rel_inter_obj_pts_normals = torch.sum(rel_inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1) ## nn_sampled_pts
        
        penetrating_indicator_mult_factor = torch.ones_like(penetrating_indicator).float()
        penetrating_indicator_mult_factor[penetrating_indicator] = -1.
        
        
        # dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] = -1. * dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] #
        # dist_sampled_pts_to_passive_obj[penetrating_indicator] = -1. * dist_sampled_pts_to_passive_obj[penetrating_indicator]
        # dist_sampled_pts_to_passive_obj[penetrating_indicator] = -1. * dist_sampled_pts_to_passive_obj[penetrating_indicator]
        dist_sampled_pts_to_passive_obj = dist_sampled_pts_to_passive_obj * penetrating_indicator_mult_factor
        # contact_spring_ka * | minn_spring_length - dist_sampled_pts_to_passive_obj | # think that it should be the 
        
        ## minn_dist_sampled_pts_passive_obj_thres 
        in_contact_indicator = dist_sampled_pts_to_passive_obj <= minn_dist_sampled_pts_passive_obj_thres
        
        
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        ws_unnormed = torch.ones_like(ws_unnormed)
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha )
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # cur_act_weights = ws_normed
        cur_act_weights = ws_unnormed # 
        
        
        
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0 #
        self.penetrating_indicator = penetrating_indicator
        cur_inter_obj_normals = inter_obj_normals.clone().detach()
        penetration_proj_ks = 0 - torch.sum(inter_obj_pts_to_sampled_pts * cur_inter_obj_normals, dim=-1)
        ### penetratio nproj penalty ###
        # inter_obj_pts_to_sampled_pts #
        
        penetration_proj_penalty = penetration_proj_ks * (-1 * torch.sum(inter_obj_pts_to_sampled_pts * cur_inter_obj_normals, dim=-1))
        self.penetrating_depth_penalty = penetration_proj_penalty[penetrating_indicator].mean()
        if torch.isnan(self.penetrating_depth_penalty): # get the penetration penalties #
            self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        penetrating_points = sampled_input_pts[penetrating_indicator]
        # penetration_proj_k_to_robot = 1.0 
        # penetration_proj_k_to_robot = 0.01
        penetration_proj_k_to_robot = self.penetration_proj_k_to_robot
        # penetration_proj_k_to_robot = 0.0
        penetrating_forces = penetration_proj_ks.unsqueeze(-1) * cur_inter_obj_normals * penetration_proj_k_to_robot
        penetrating_forces = penetrating_forces[penetrating_indicator]
        self.penetrating_forces = penetrating_forces #
        self.penetrating_points = penetrating_points #
        ### penetration strategy v4 #### # another mophology #
        
        
        
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
        
        
        contact_spring_ka = self.spring_contact_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
        
        
        ##### the contact force decided by the theshold ###### # realted to the distance threshold and the HO distance #
        contact_force_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_sampled_pts_to_passive_obj) 
        ###### the contact force decided by the threshold ######
        
        
        # 
        contact_force_d = contact_force_d.unsqueeze(-1) * (-1. * inter_obj_normals)
        
        # norm along normals forces # 
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # nn_sampled_pts ##
        norm_along_normals_forces = torch.norm(contact_force_d, dim=-1, p=2) # nn_sampled_pts, nnsampledpts #
        # penalty_friction_constraint = (norm_tangential_forces - self.static_friction_mu * norm_along_normals_forces) ** 2
        # penalty_friction_constraint[norm_tangential_forces <= self.static_friction_mu * norm_along_normals_forces] = 0.
        # penalty_friction_constraint = torch.mean(penalty_friction_constraint) #
        self.penalty_friction_constraint = torch.zeros((1,), dtype=torch.float32).cuda().mean() # penalty friction #
        # contact_force_d_scalar = norm_along_normals_forces.clone() # contact forces #
        contact_force_d_scalar = norm_along_normals_forces.clone() #
        
        # use the test cases to demonstrate the correctness #
        # penalty friction constraints #
        penalty_friction_tangential_forces = torch.zeros_like(contact_force_d)
        
        
        ''' Get the contact information that should be maintained'''
        if contact_pairs_set is not None:
            ### active point pts, point position ###
            contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
            # contact active pos and contact passive pos # contact_active_pos; contact_passive_pos # # # 
            contact_active_pos = sampled_input_pts[contact_active_idxes]
            contact_passive_pos = cur_passive_obj_verts[contact_passive_idxes]
            
            ''' Penalty based contact force v2 ''' 
            contact_frame_orientations, contact_frame_translations = contact_frame_pose
            transformed_prev_contact_active_pos = torch.matmul(
                contact_frame_orientations.contiguous(), contact_active_point_pts.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            transformed_prev_contact_point_position = torch.matmul(
                contact_frame_orientations.contiguous(), contact_point_position.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            # transformed prev contact active pose #
            diff_transformed_prev_contact_passive_to_active = transformed_prev_contact_active_pos - transformed_prev_contact_point_position
            ### 
            # cur_contact_passive_pos_from_active = contact_passive_pos + diff_transformed_prev_contact_passive_to_active
            # cur_contact_passive_pos_from_active = contact_active_pos - diff_transformed_prev_contact_passive_to_active
            
            # friction_k = 1.0
            # friction_k = 0.01
            # friction_k = 0.001
            # friction_k = 0.001
            friction_k = 1.0
            ##### not a very accurate formulation.... #####
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            # penalty_based_friction_forces = friction_k * (contact_active_pos - transformed_prev_contact_active_pos)
            
            
            # cur_passive_obj_ns[contact_passive_idxes]
            cur_inter_passive_obj_ns = cur_passive_obj_ns[contact_passive_idxes]
            cur_contact_passive_to_active_pos = contact_active_pos - contact_passive_pos
            dot_cur_contact_passive_to_active_pos_with_ns = torch.sum( ### dot produce between passive to active and the passive ns ###
                cur_inter_passive_obj_ns * cur_contact_passive_to_active_pos, dim=-1
            )
            cur_contact_passive_to_active_pos = cur_contact_passive_to_active_pos - dot_cur_contact_passive_to_active_pos_with_ns.unsqueeze(-1) * cur_inter_passive_obj_ns
            
            # contact passive posefrom active ## 
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            penalty_based_friction_forces = friction_k * cur_contact_passive_to_active_pos
            # penalty_based_friction_forces = friction_k * (cur_contact_passive_pos_from_active - contact_passive_pos)
            
            
            # a good way to optiize the actions? #
            dist_contact_active_pos_to_passive_pose = torch.sum(
                (contact_active_pos - contact_passive_pos) ** 2, dim=-1
            )
            dist_contact_active_pos_to_passive_pose = torch.sqrt(dist_contact_active_pos_to_passive_pose)
            
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.1 # how many contacts to keep #
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.2
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.3
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.5
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 1.0
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 1000.0 # 
            ### contact maintaning dist thres ### # 
            remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= self.contact_maintaining_dist_thres
            ''' Penalty based contact force v2 '''
            
            ### hwo to produce the cotact force and how to produce the frictional forces #
            ### optimized 
            # tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.005 #  * 1000.
            
            # ### spring_friction_ks_values ### #
            # spring_friction_ks_values #
            
            ''' update contact_force_d '''
            if self.use_sqrt_dist:
                dist_cur_active_to_passive = torch.norm(
                    (contact_active_pos - contact_passive_pos), dim=-1, p=2
                )
            else:
                dist_cur_active_to_passive = torch.sum(
                    (contact_active_pos - contact_passive_pos) ** 2, dim=-1
                )
            ''' update contact force d '''
            
            ''' #### Get contact normal forces for active points in contact ''' 
            ## dist cur active to passive ##
            penetrating_indicator_mult_factor = torch.ones_like(penetrating_indicator).float().detach()
            penetrating_indicator_mult_factor[penetrating_indicator] = -1.
            cur_penetrating_indicator_mult_factor = penetrating_indicator_mult_factor[contact_active_idxes]
            dist_cur_active_to_passive = dist_cur_active_to_passive * cur_penetrating_indicator_mult_factor
            #### dist cur active to passive #### ## passive obj to active obj ##
            # dist_cur_active_to_passive[penetrating_indicator[contact_active_idxes]] = -1. * dist_cur_active_to_passive[penetrating_indicator[contact_active_idxes]
            # dist -- contact_d # # spring_ka -> force scales # # spring_ka -> force scales ##
            cur_contact_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_cur_active_to_passive) 
            # contact_force_d_scalar = contact_force_d.clone() #
            cur_contact_d = cur_contact_d.unsqueeze(-1) * (-1. * cur_passive_obj_ns[contact_passive_idxes])
            contact_active_force_d_scalar = torch.norm(cur_contact_d, dim=-1, p=2)
            '''  #### Get contact normal forces for active points in contact ''' 
            
            
            ### use the dynamical model to update the spring firction ks ### # 
            ### TODO: how to check the correctness of the switching between the static friction and the dynamic friction ###
            contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
            # use the relative scale of the friction force and thejcoantact force to decide the remaining contact indicator #
            ##### contact active penalty based friction forces -> spring_k * relative displacement ##### # penalty based friction forces #
            contact_active_penalty_based_friction_forces = penalty_based_friction_forces * contact_friction_spring_cur
            contact_active_penalty_based_friction_forces_norm = torch.norm(contact_active_penalty_based_friction_forces, p=2, dim=-1)
            # contact_active_penalty_based_friction_forces # # forces norm #### 
            # #### contact_force_d_scalar_ #### #
            # contact_active_force_d_scalar = contact_force_d_scalar[contact_active_idxes] ### contact active force d scalar ###
            #### # contact_friction_static_mu #
            # contact_friction_static_mu = 10.
            contact_friction_static_mu = 1. # the friction mu #
            remaining_contact_indicator = contact_active_penalty_based_friction_forces_norm <= (contact_friction_static_mu * contact_active_force_d_scalar)
            
            ### not_remaining_contacts ###
            not_remaining_contacts = contact_active_penalty_based_friction_forces_norm > (contact_friction_static_mu * contact_active_force_d_scalar)
            
            ## contact active penalty based friction forces ##
            #### #### contact #### #### # contact activ penalty based friction forces #
            contact_active_penalty_based_friction_forces_dir = contact_active_penalty_based_friction_forces / torch.clamp(contact_active_penalty_based_friction_forces_norm.unsqueeze(-1), min=1e-5)
            dyn_contact_active_penalty_based_friction_forces = contact_active_penalty_based_friction_forces_dir * (contact_friction_static_mu * contact_active_force_d_scalar).unsqueeze(-1)
            # contact_active_penalty_based_friction_forces[not_remaining_contacts] = dyn_contact_active_penalty_based_friction_forces[not_remaining_contacts] # correctnesss #
            ##### ocntact active penatl based friction forces #####
            not_remaining_contacts_mask = torch.zeros((contact_active_penalty_based_friction_forces.size(0)), dtype=torch.float32).cuda()
            not_remaining_contacts_mask[not_remaining_contacts] = 1
            
            contact_active_penalty_based_friction_forces = contact_active_penalty_based_friction_forces * (1. - not_remaining_contacts_mask).unsqueeze(-1) + dyn_contact_active_penalty_based_friction_forces * not_remaining_contacts_mask.unsqueeze(-1)
            ### TODO: how to check the correctness of the switching between the static friction and the dynamic friction ###
            
            
            # # penalty_friction_tangential_forces[contact_active_idxes] = penalty_based_friction_forces * contact_friction_spring_cur # * 0.1
            # penalty_friction_tangential_forces[contact_active_idxes] = contact_active_penalty_based_friction_forces ## 0.1 ##
            
            # contact_active_idxes_mask = torch.zeros((penalty_friction_tangential_forces.size(0)), dtype=torch.float32).cuda()
            # contact_active_idxes_mask[contact_active_idxes] = 1 ##### ### contact_active_idxes ### ##### ## 
            # contact_active_idxes_mask = contact_active_idxes_mask.unsqueeze(-1).repeat(1, 3).contiguous()
            # ### or I think that the penalty 
            # penalty_friction_tangential_forces = torch.where( # 
            #     contact_active_idxes_mask > 0.5, penalty_friction_tangential_forces, torch.zeros_like(penalty_friction_tangential_forces)
            # )
            expanded_contact_active_idxes = contact_active_idxes.unsqueeze(-1).contiguous().repeat(1, 3).contiguous()
            penalty_friction_tangential_forces = torch.scatter(penalty_friction_tangential_forces, dim=0, index=expanded_contact_active_idxes, src=contact_active_penalty_based_friction_forces)
            
            ####### penalty friction tangential forces #######
            
            
            
            # ### add the sqrt for calculate the l2 distance ###
            # dist_cur_active_to_passive = torch.sqrt(dist_cur_active_to_passive)
            
            # dist_cur_active_to_passive = torch.norm(
            #     (contact_active_pos - contact_passive_pos), dim=-1, p=2
            # )
            
            
            # inter_obj_normals[contact_active_idxes] = cur_passive_obj_ns[contact_passive_idxes]
            
            # contact_force_d[contact_active_idxes] = cur_contact_d
            
            inter_obj_normals = torch.scatter(
                inter_obj_normals, dim=0, index=expanded_contact_active_idxes, src=cur_passive_obj_ns
            )
            
            contact_force_d = torch.scatter(
                contact_force_d, dim=0, index=expanded_contact_active_idxes, src=cur_contact_d
            )
            
            ''' update contact_force_d '''
            
            cur_act_weights[contact_active_idxes] = 1.
            ws_unnormed[contact_active_idxes] = 1.
        else:
            contact_active_idxes = None
            self.contact_active_idxes = contact_active_idxes
            valid_penalty_friction_forces_indicator = None
            penalty_based_friction_forces = None
        # tangential forces with inter obj normals # -> ####
        if torch.sum(cur_act_weights).item() > 0.5:
            cur_act_weights = cur_act_weights / torch.sum(cur_act_weights)
        
        # penalty based # 
        
        # norm_penalty_friction_tangential_forces = torch.norm(penalty_friction_tangential_forces, dim=-1, p=2)
        # maxx_norm_penalty_friction_tangential_forces, _ = torch.max(norm_penalty_friction_tangential_forces, dim=-1)
        # minn_norm_penalty_friction_tangential_forces, _ = torch.min(norm_penalty_friction_tangential_forces, dim=-1)
        # print(f"maxx_norm_penalty_friction_tangential_forces: {maxx_norm_penalty_friction_tangential_forces}, minn_norm_penalty_friction_tangential_forces: {minn_norm_penalty_friction_tangential_forces}")
        
        
        ####### penalty_friction_tangential_forces #######
        
        # tangetntial forces --- dot with normals #
        # dot_tangential_forces_with_inter_obj_normals = torch.sum(penalty_friction_tangential_forces * inter_obj_normals, dim=-1) ### nn_active_pts x # 
        # penalty_friction_tangential_forces = penalty_friction_tangential_forces - dot_tangential_forces_with_inter_obj_normals.unsqueeze(-1) * inter_obj_normals
        
        ####### penalty_friction_tangential_forces #######
        
        
        
        # self.penetrating_forces = penetrating_forces #
        # self.penetrating_points = penetrating_points #
        
        ### TODO: is that stable? ###
        ####### update the forces to the active object #######
        ### next: add the negative frictional forces to the penetrating forces ### 
        # penetrating_friction_to_active_mult_factor = 1.0
        # penetrating_friction_to_active_mult_factor = 0.001
        # penetrating_penalty_friction_forces_to_active = -1. * penalty_friction_tangential_forces ### get the tangential forces ###
        # penetrating_penalty_friction_forces_to_active = penetrating_penalty_friction_forces_to_active[penetrating_indicator]
        # self.penetrating_forces = self.penetrating_forces + penetrating_penalty_friction_forces_to_active * penetrating_friction_to_active_mult_factor
        # ## add the negative frictional forces to the repulsion forces to the active object ##
        # penetrating_forces = penetrating_forces + penetrating_penalty_friction_forces_to_active * penetrating_friction_to_active_mult_factor
        ####### penetrating forces #######
        ### TODO: is that stable? ###
        
        
        ####### penalty_based_friction_forces #######
        # norm_penalty_friction_tangential_forces = torch.norm(penalty_friction_tangential_forces, dim=-1, p=2)
        # # valid penalty friction forces # # valid contact force d scalar #
        # maxx_norm_penalty_friction_tangential_forces, _ = torch.max(norm_penalty_friction_tangential_forces, dim=-1)
        # minn_norm_penalty_friction_tangential_forces, _ = torch.min(norm_penalty_friction_tangential_forces, dim=-1)
        # print(f"[After proj.] maxx_norm_penalty_friction_tangential_forces: {maxx_norm_penalty_friction_tangential_forces}, minn_norm_penalty_friction_tangential_forces: {minn_norm_penalty_friction_tangential_forces}")
        
        
        # tangential_forces_clone = tangential_forces.clone()
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
        
        ### ## get new contacts ## ###
        tot_contact_point_position = []
        tot_contact_active_point_pts = []
        tot_contact_active_idxes = []
        tot_contact_passive_idxes = []
        tot_contact_frame_rotations = []
        tot_contact_frame_translations = []
        
        
        if contact_pairs_set is not None: # contact
            if torch.sum(remaining_contact_indicator.float()) > 0.5:
                # contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
                remaining_contact_active_point_pts = contact_active_point_pts[remaining_contact_indicator]
                remaining_contact_point_position = contact_point_position[remaining_contact_indicator]
                remaining_contact_active_idxes = contact_active_idxes[remaining_contact_indicator]
                remaining_contact_passive_idxes = contact_passive_idxes[remaining_contact_indicator]
                remaining_contact_frame_rotations = contact_frame_orientations[remaining_contact_indicator]
                remaining_contact_frame_translations = contact_frame_translations[remaining_contact_indicator]
                tot_contact_point_position.append(remaining_contact_point_position)
                tot_contact_active_point_pts.append(remaining_contact_active_point_pts)
                tot_contact_active_idxes.append(remaining_contact_active_idxes)
                tot_contact_passive_idxes.append(remaining_contact_passive_idxes)
                tot_contact_frame_rotations.append(remaining_contact_frame_rotations)
                tot_contact_frame_translations.append(remaining_contact_frame_translations)
                
                # remaining_contact_active_idxes
                in_contact_indicator[remaining_contact_active_idxes] = False
        
        
        
        if torch.sum(in_contact_indicator.float()) > 0.5: # in contact indicator #
            cur_in_contact_passive_pts = inter_obj_pts[in_contact_indicator]
            # cur_in_contact_passive_normals = inter_obj_normals[in_contact_indicator] ##
            cur_in_contact_active_pts = sampled_input_pts[in_contact_indicator] # in_contact_active_pts ##
            
            
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
        self.tangential_forces = penalty_friction_tangential_forces
        
        self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
        self.contact_force_d = contact_force_d
        self.penalty_based_friction_forces = penalty_based_friction_forces
    
        
        penalty_dot_forces_normals = dot_forces_normals ** 2
        penalty_dot_forces_normals[dot_forces_normals <= 0] = 0 # 1) must in the negative direction of the object normal #
        penalty_dot_forces_normals = torch.mean(penalty_dot_forces_normals) # 1) must # 2) must #
        self.penalty_dot_forces_normals = penalty_dot_forces_normals #
        
        
        #
        rigid_acc = torch.sum(forces * cur_act_weights.unsqueeze(-1), dim=0) # rigid acc #
        
        ###### sampled input pts to center #######
        if contact_pairs_set is not None:
            inter_obj_pts[contact_active_idxes] = cur_passive_obj_verts[contact_passive_idxes]
        
        # center_point_to_sampled_pts = sampled_input_pts - passive_center_point.unsqueeze(0)
        
        center_point_to_sampled_pts = inter_obj_pts - passive_center_point.unsqueeze(0)
        
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
        
        
        # sep_time_constant, sep_torque_time_constant, sep_damping_constant, sep_angular_damping_constant
        time_cons = self.sep_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
        time_cons_2 = self.sep_torque_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
        damping_cons = self.sep_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
        damping_cons_2 = self.sep_angular_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
        
        
        k_acc_to_vel = time_cons
        k_vel_to_offset = time_cons_2
        delta_vel = rigid_acc * k_acc_to_vel
        if input_pts_ts == 0:
            cur_vel = delta_vel
        else:
            ##### TMP ######
            # cur_vel = delta_vel
            cur_vel = delta_vel + self.timestep_to_vel[input_pts_ts - 1].detach() * damping_cons
        self.timestep_to_vel[input_pts_ts] = cur_vel.detach()
         
        cur_offset = k_vel_to_offset * cur_vel 
        cur_rigid_def = self.timestep_to_total_def[input_pts_ts].detach() # 
        
        
        delta_angular_vel = torque * time_cons_rot
        if input_pts_ts == 0:
            cur_angular_vel = delta_angular_vel
        else:
            ##### TMP ######
            # cur_angular_vel = delta_angular_vel
            cur_angular_vel = delta_angular_vel + self.timestep_to_angular_vel[input_pts_ts - 1].detach() * damping_cons_rot ### (3,)
        cur_delta_angle = cur_angular_vel * time_cons_rot # \delta_t w^1 / 2 
        
        prev_quaternion = self.timestep_to_quaternion[input_pts_ts].detach() # 
        cur_quaternion = prev_quaternion + update_quaternion(cur_delta_angle, prev_quaternion)
        
        
        cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
        prev_rot_mtx = quaternion_to_matrix(prev_quaternion)
        
        # cur_
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_rigid_def.unsqueeze(0), cur_delta_rot_mtx.detach()).squeeze(0)
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        cur_upd_rigid_def = cur_offset.detach() + cur_rigid_def # update the current rigid def using the offset and the cur_rigid_def ## # 
        # curupd
        # if update_tot_def: # update rigid def #
        
        
        
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




# 
class BendingNetworkActiveForceFieldForwardLagV17(nn.Module):
    def __init__(self,
                 d_in,
                 multires,
                 bending_n_timesteps,
                 bending_latent_size,
                 rigidity_hidden_dimensions,
                 rigidity_network_depth,
                 rigidity_use_latent=False,
                 use_rigidity_network=False,
                 nn_instances=1,
                 minn_dist_threshold=0.05,
                 ): 
        # bending network active force field #
        super(BendingNetworkActiveForceFieldForwardLagV17, self).__init__()
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

        # simple scene editing. set to None during training. #
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
        
        self.cur_window_size = 60
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
        else:
            self.spring_ks_values = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=5, embedding_dim=1) for _ in range(self.nn_instances)
                ]
            )
            for cur_ks_values in self.spring_ks_values:
                torch.nn.init.ones_(cur_ks_values.weight)
                cur_ks_values.weight.data = cur_ks_values.weight.data * 0.01
        
        
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
        self.sep_damping_constant.weight.data = self.sep_damping_constant.weight.data * 0.9
        
        
        self.sep_angular_damping_constant = nn.Embedding(
            num_embeddings=64, embedding_dim=1
        )
        torch.nn.init.ones_(self.sep_angular_damping_constant.weight) # # # #
        self.sep_angular_damping_constant.weight.data = self.sep_angular_damping_constant.weight.data * 0.9
    
        
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
        torch.nn.init.zeros_(self.actuator_forces.weight)
        
        
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
        # timestep_to_prev_selected_active_mesh_ori, timestep_to_prev_selected_active_mesh # 
        self.timestep_to_prev_selected_active_mesh_ori = {}
        self.timestep_to_prev_selected_active_mesh = {}
        
        self.timestep_to_spring_forces = {}
        self.timestep_to_spring_forces_ori = {}
        
        # timestep_to_angular_vel, timestep_to_quaternion # 
        self.timestep_to_angular_vel = {}
        self.timestep_to_quaternion = {}
        self.timestep_to_torque = {}
        
        
        # timestep_to_optimizable_total_def, timestep_to_optimizable_quaternion
        self.timestep_to_optimizable_total_def = {}
        self.timestep_to_optimizable_quaternion = {}
        self.timestep_to_optimizable_rot_mtx = {}
        self.timestep_to_aggregation_weights = {}
        self.timestep_to_sampled_pts_to_passive_obj_dist = {}

        self.time_quaternions = nn.Embedding(
            num_embeddings=60, embedding_dim=4
        )
        self.time_quaternions.weight.data[:, 0] = 1.
        self.time_quaternions.weight.data[:, 1] = 0.
        self.time_quaternions.weight.data[:, 2] = 0.
        self.time_quaternions.weight.data[:, 3] = 0.
        # torch.nn.init.ones_(self.time_quaternions.weight) # 
        
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
        cur_transformed_pts = torch.matmul(
            cur_frame_rotation.contiguous().transpose(1, 0).contiguous(), (cur_pts - cur_frame_translation.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0)
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
        return cur_pts_sdf # # cur_pts_sdf # 
    
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
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001
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


    ### forward; 
    def forward2(self, input_pts_ts, timestep_to_active_mesh, timestep_to_passive_mesh, timestep_to_passive_mesh_normals, details=None, special_loss_return=False, update_tot_def=True, friction_forces=None, i_instance=0, reference_mano_pts=None, sampled_verts_idxes=None, fix_obj=False, contact_pairs_set=None, ):
        
        nex_pts_ts = input_pts_ts + 1


        sampled_input_pts = timestep_to_active_mesh[input_pts_ts]
        # ori_nns = sampled_input_pts.size(0)
        if sampled_verts_idxes is not None:
            sampled_input_pts = sampled_input_pts[sampled_verts_idxes]
        # nn_sampled_input_pts = sampled_input_pts.size(0)
        
        if nex_pts_ts in timestep_to_active_mesh: ##
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
        
        
        init_passive_obj_verts = timestep_to_passive_mesh[0]
        init_passive_obj_ns = timestep_to_passive_mesh_normals[0]
        center_init_passive_obj_verts = init_passive_obj_verts.mean(dim=0)
        
        # cur_passive_obj_rot, cur_passive_obj_trans # 
        cur_passive_obj_rot = quaternion_to_matrix(self.timestep_to_quaternion[input_pts_ts].detach())
        cur_passive_obj_trans = self.timestep_to_total_def[input_pts_ts].detach()
        
        
        cur_passive_obj_verts = torch.matmul(cur_passive_obj_rot, (init_passive_obj_verts - center_init_passive_obj_verts.unsqueeze(0)).transpose(1, 0)).transpose(1, 0) + center_init_passive_obj_verts.squeeze(0) + cur_passive_obj_trans.unsqueeze(0) # 
        cur_passive_obj_ns = torch.matmul(cur_passive_obj_rot, init_passive_obj_ns.transpose(1, 0).contiguous()).transpose(1, 0).contiguous()
        cur_passive_obj_ns = cur_passive_obj_ns / torch.clamp(torch.norm(cur_passive_obj_ns, dim=-1, keepdim=True), min=1e-8)
        cur_passive_obj_center = center_init_passive_obj_verts + cur_passive_obj_trans
        passive_center_point = cur_passive_obj_center # obj_center #
        
        # cur_active_mesh = timestep_to_active_mesh[input_pts_ts] #
        # nex_active_mesh = timestep_to_active_mesh[input_pts_ts + 1] #
        
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
        
        # # assue the passive object cannot move at all # #


        # cur_actuation_embedding_st_idx = self.nn_actuators * input_pts_ts
        # cur_actuation_embedding_ed_idx = self.nn_actuators * (input_pts_ts + 1)
        # cur_actuation_embedding_idxes = torch.tensor([idxx for idxx in range(cur_actuation_embedding_st_idx, cur_actuation_embedding_ed_idx)], dtype=torch.long).cuda()
        # ######### optimize the actuator forces directly #########
        # cur_actuation_forces = self.actuator_forces(cur_actuation_embedding_idxes) # actuation embedding idxes #
        # forces = cur_actuation_forces
        # ######### optimize the actuator forces directly #########


        if self.nn_instances == 1:
            ws_alpha = self.ks_weights(torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights(torch.ones((1,)).long().cuda()).view(1)
        else:
            ws_alpha = self.ks_weights[i_instance](torch.zeros((1,)).long().cuda()).view(1)
            ws_beta = self.ks_weights[i_instance](torch.ones((1,)).long().cuda()).view(1)
        
        
        if self.use_sqrt_dist:
            dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
            )
        else:
            dist_sampled_pts_to_passive_obj = torch.sum( # nn_sampled_pts x nn_passive_pts 
                (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)) ** 2, dim=-1
            )
        
        # ### add the sqrt for calculate the l2 distance ###
        # dist_sampled_pts_to_passive_obj = torch.sqrt(dist_sampled_pts_to_passive_obj) ### #
        # ### add the sqrt for ### #
        
        
        ## dist dampled ## ## dist sampled ## # 
        # dist_sampled_pts_to_passive_obj = torch.norm( # nn_sampled_pts x nn_passive_pts 
        #     (sampled_input_pts.unsqueeze(1) - cur_passive_obj_verts.unsqueeze(0)), dim=-1, p=2
        # )
        
        dist_sampled_pts_to_passive_obj, minn_idx_sampled_pts_to_passive_obj = torch.min(dist_sampled_pts_to_passive_obj, dim=-1)
        
        
        
        # inte robj normals at the current frame # # 
        inter_obj_normals = cur_passive_obj_ns[minn_idx_sampled_pts_to_passive_obj]
        inter_obj_pts = cur_passive_obj_verts[minn_idx_sampled_pts_to_passive_obj]
        
        # ### passive obj verts pts idxes ### # 
        cur_passive_obj_verts_pts_idxes = torch.arange(0, cur_passive_obj_verts.size(0), dtype=torch.long).cuda() # 
        inter_passive_obj_pts_idxes = cur_passive_obj_verts_pts_idxes[minn_idx_sampled_pts_to_passive_obj]
        
        # inter_obj_normals #
        inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts.detach()
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
        ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha * 10) # ws_alpha #
        ####### sharp the weights #######
        
        # minn_dist_sampled_pts_passive_obj_thres = 0.05
        # # minn_dist_sampled_pts_passive_obj_thres = 0.001
        # minn_dist_sampled_pts_passive_obj_thres = 0.0001
        minn_dist_sampled_pts_passive_obj_thres = self.minn_dist_sampled_pts_passive_obj_thres
        
        
       
        # # ws_unnormed = ws_normed_sampled
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9) e
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
            # penetrating_indicator = torch.zeros_like(dot_inter_obj_pts_to_sampled_pts_normals).bool()
            penetrating_indicator = torch.zeros((sampled_input_pts.size(0),), dtype=torch.bool).cuda().bool()
        
        # if contact_pairs_set is not None and self.use_penalty_based_friction and (not self.use_disp_based_friction):
        #     penetrating
        
        ''' decide forces via kinematics statistics '''
        # rel_inter_obj_pts_to_sampled_pts = sampled_input_pts - inter_obj_pts # inter_obj_pts #
        # dot_rel_inter_obj_pts_normals = torch.sum(rel_inter_obj_pts_to_sampled_pts * inter_obj_normals, dim=-1) ## nn_sampled_pts
        
        penetrating_indicator_mult_factor = torch.ones_like(penetrating_indicator).float()
        penetrating_indicator_mult_factor[penetrating_indicator] = -1.
        
        
        # dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] = -1. * dist_sampled_pts_to_passive_obj[dot_rel_inter_obj_pts_normals < 0] #
        # dist_sampled_pts_to_passive_obj[penetrating_indicator] = -1. * dist_sampled_pts_to_passive_obj[penetrating_indicator]
        # dist_sampled_pts_to_passive_obj[penetrating_indicator] = -1. * dist_sampled_pts_to_passive_obj[penetrating_indicator]
        dist_sampled_pts_to_passive_obj = dist_sampled_pts_to_passive_obj * penetrating_indicator_mult_factor
        # contact_spring_ka * | minn_spring_length - dist_sampled_pts_to_passive_obj | # think that it should be the 
        
        ## minn_dist_sampled_pts_passive_obj_thres 
        in_contact_indicator = dist_sampled_pts_to_passive_obj <= minn_dist_sampled_pts_passive_obj_thres
        
        
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        ws_unnormed = torch.ones_like(ws_unnormed)
        # distance > minn distance #
        ws_unnormed[dist_sampled_pts_to_passive_obj > minn_dist_sampled_pts_passive_obj_thres] = 0
        
        # ws_unnormed = ws_beta * torch.exp(-1. * dist_sampled_pts_to_passive_obj * ws_alpha )
        # ws_normed = ws_unnormed / torch.clamp(torch.sum(ws_unnormed), min=1e-9)
        # cur_act_weights = ws_normed
        cur_act_weights = ws_unnormed
        
        
        # penetrating_indicator = dot_inter_obj_pts_to_sampled_pts_normals < 0 #
        self.penetrating_indicator = penetrating_indicator
        cur_inter_obj_normals = inter_obj_normals.clone().detach()
        penetration_proj_ks = 0 - torch.sum(inter_obj_pts_to_sampled_pts * cur_inter_obj_normals, dim=-1)
        ### penetratio nproj penalty ###
        # inter_obj_pts_to_sampled_pts #
        
        penetration_proj_penalty = penetration_proj_ks * (-1 * torch.sum(inter_obj_pts_to_sampled_pts * cur_inter_obj_normals, dim=-1))
        self.penetrating_depth_penalty = penetration_proj_penalty[penetrating_indicator].mean()
        if torch.isnan(self.penetrating_depth_penalty): # get the penetration penalties #
            self.penetrating_depth_penalty = torch.tensor(0., dtype=torch.float32).cuda()
        penetrating_points = sampled_input_pts[penetrating_indicator]
        # penetration_proj_k_to_robot = 1.0 
        # penetration_proj_k_to_robot = 0.01
        penetration_proj_k_to_robot = self.penetration_proj_k_to_robot
        # penetration_proj_k_to_robot = 0.0
        penetrating_forces = penetration_proj_ks.unsqueeze(-1) * cur_inter_obj_normals * penetration_proj_k_to_robot
        penetrating_forces = penetrating_forces[penetrating_indicator]
        self.penetrating_forces = penetrating_forces #
        self.penetrating_points = penetrating_points #
        ### penetration strategy v4 #### # another mophology #
        
        
        
        if self.nn_instances == 1: # spring ks values 
            # contact ks values # # if we set a fixed k value here #
            contact_spring_ka = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            # contact_spring_kb = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 2).view(1,)
            # contact_spring_kc = self.spring_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            # tangential_ks = self.spring_ks_values(torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        else:
            contact_spring_ka = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda()).view(1,)
            # contact_spring_kb = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 2).view(1,)
            # contact_spring_kc = self.spring_ks_values[i_instance](torch.zeros((1,), dtype=torch.long).cuda() + 3).view(1,)
            
            # tangential_ks = self.spring_ks_values[i_instance](torch.ones((1,), dtype=torch.long).cuda()).view(1,)
        
        if self.use_split_params:
            contact_spring_ka = self.spring_contact_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
        
        
        ##### the contact force decided by the theshold ###### # realted to the distance threshold and the HO distance #
        contact_spring_ka = contact_spring_ka ** 2
        contact_force_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_sampled_pts_to_passive_obj) 
        ###### the contact force decided by the threshold ######
        
        
        contact_force_d = contact_force_d.unsqueeze(-1) * (-1. * inter_obj_normals)
        
        # norm along normals forces # 
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # nn_sampled_pts ##
        norm_along_normals_forces = torch.norm(contact_force_d, dim=-1, p=2) # nn_sampled_pts, nnsampledpts #
        # penalty_friction_constraint = (norm_tangential_forces - self.static_friction_mu * norm_along_normals_forces) ** 2
        # penalty_friction_constraint[norm_tangential_forces <= self.static_friction_mu * norm_along_normals_forces] = 0.
        # penalty_friction_constraint = torch.mean(penalty_friction_constraint) #
        self.penalty_friction_constraint = torch.zeros((1,), dtype=torch.float32).cuda().mean() # penalty friction #
        # contact_force_d_scalar = norm_along_normals_forces.clone() # contact forces #
        contact_force_d_scalar = norm_along_normals_forces.clone() #
        
        # use the test cases to demonstrate the correctness #
        # penalty friction constraints #
        penalty_friction_tangential_forces = torch.zeros_like(contact_force_d)
        
        
        ''' Get the contact information that should be maintained'''
        if contact_pairs_set is not None:
            ### active point pts, point position ###
            contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
            # contact active pos and contact passive pos # contact_active_pos; contact_passive_pos # # # 
            contact_active_pos = sampled_input_pts[contact_active_idxes]
            contact_passive_pos = cur_passive_obj_verts[contact_passive_idxes]
            
            ''' Penalty based contact force v2 ''' 
            contact_frame_orientations, contact_frame_translations = contact_frame_pose
            transformed_prev_contact_active_pos = torch.matmul(
                contact_frame_orientations.contiguous(), contact_active_point_pts.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            transformed_prev_contact_point_position = torch.matmul(
                contact_frame_orientations.contiguous(), contact_point_position.unsqueeze(-1)
            ).squeeze(-1) + contact_frame_translations
            # transformed prev contact active pose #
            diff_transformed_prev_contact_passive_to_active = transformed_prev_contact_active_pos - transformed_prev_contact_point_position
            ### 
            # cur_contact_passive_pos_from_active = contact_passive_pos + diff_transformed_prev_contact_passive_to_active
            # cur_contact_passive_pos_from_active = contact_active_pos - diff_transformed_prev_contact_passive_to_active
            
            # friction_k = 1.0
            # friction_k = 0.01
            # friction_k = 0.001
            # friction_k = 0.001
            friction_k = 1.0 #
            ##### not a very accurate formulation.... ##### # tracking loss # # # tracking loss # # 
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            # penalty_based_friction_forces = friction_k * (contact_active_pos - transformed_prev_contact_active_pos)
            
            
            # cur_passive_obj_ns[contact_passive_idxes]
            cur_inter_passive_obj_ns = cur_passive_obj_ns[contact_passive_idxes]
            cur_contact_passive_to_active_pos = contact_active_pos - contact_passive_pos
            dot_cur_contact_passive_to_active_pos_with_ns = torch.sum( ### dot produce between passive to active and the passive ns ###
                cur_inter_passive_obj_ns * cur_contact_passive_to_active_pos, dim=-1
            )
            cur_contact_passive_to_active_pos = cur_contact_passive_to_active_pos - dot_cur_contact_passive_to_active_pos_with_ns.unsqueeze(-1) * cur_inter_passive_obj_ns
            
            # contact passive posefrom active ## 
            # penalty_based_friction_forces = friction_k * (contact_active_pos - contact_passive_pos)
            penalty_based_friction_forces = friction_k * cur_contact_passive_to_active_pos
            # penalty_based_friction_forces = friction_k * (cur_contact_passive_pos_from_active - contact_passive_pos)
            
            
            # a good way to optiize the actions? #
            dist_contact_active_pos_to_passive_pose = torch.sum(
                (contact_active_pos - contact_passive_pos) ** 2, dim=-1
            )
            dist_contact_active_pos_to_passive_pose = torch.sqrt(dist_contact_active_pos_to_passive_pose)
            
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.1
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 0.2
            # remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= 1000.0 # 
            ### contact maintaning dist thres ### # 
            remaining_contact_indicator = dist_contact_active_pos_to_passive_pose <= self.contact_maintaining_dist_thres
            ''' Penalty based contact force v2 '''
            
            ### hwo to produce the cotact force and how to produce the frictional forces #
            ### optimized 
            # tangential_forces[contact_active_idxes][valid_penalty_friction_forces_indicator] = penalty_based_friction_forces[valid_penalty_friction_forces_indicator] * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.01 #  * 1000.
            # tangential_forces[contact_active_idxes] = penalty_based_friction_forces * 0.005 #  * 1000.
            
            # ### spring_friction_ks_values ### #
            # spring_friction_ks_values #
            
            ''' update contact_force_d '''
            if self.use_sqrt_dist:
                dist_cur_active_to_passive = torch.norm(
                    (contact_active_pos - contact_passive_pos), dim=-1, p=2
                )
            else:
                dist_cur_active_to_passive = torch.sum(
                    (contact_active_pos - contact_passive_pos) ** 2, dim=-1
                )
            ''' update contact_force_d '''
            
            ''' #### Get contact normal forces for active points in contact #### ''' 
            ## dist cur active to passive ##
            penetrating_indicator_mult_factor = torch.ones_like(penetrating_indicator).float().detach()
            penetrating_indicator_mult_factor[penetrating_indicator] = -1.
            cur_penetrating_indicator_mult_factor = penetrating_indicator_mult_factor[contact_active_idxes]
            dist_cur_active_to_passive = dist_cur_active_to_passive * cur_penetrating_indicator_mult_factor
            #### dist cur active to passive #### ## passive obj to active obj ##
            # dist_cur_active_to_passive[penetrating_indicator[contact_active_idxes]] = -1. * dist_cur_active_to_passive[penetrating_indicator[contact_active_idxes]] #
            # dist -- contact_d # # spring_ka -> force scales # # spring_ka -> force scales # #
            cur_contact_d = contact_spring_ka * (self.minn_dist_sampled_pts_passive_obj_thres - dist_cur_active_to_passive) 
            # contact_force_d_scalar = contact_force_d.clone() #
            cur_contact_d = cur_contact_d.unsqueeze(-1) * (-1. * cur_passive_obj_ns[contact_passive_idxes])
            contact_active_force_d_scalar = torch.norm(cur_contact_d, dim=-1, p=2)
            '''  #### Get contact normal forces for active points in contact #### ''' 
            
            
            ### use the dynamical model to update the spring firction ks ###
            
            # contact friction spring cur #
            if self.use_split_params:
                contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
            else:
                contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda()).view(1,) # time-varying?
            
            # use the relative scale of the friction force and thejcoantact force to decide the remaining contact indicator #
            ##### contact active penalty based friction forces -> spring_k * relative displacement ##### # penalty based friction forces #
            contact_friction_spring_cur = contact_friction_spring_cur ** 2
            contact_active_penalty_based_friction_forces = penalty_based_friction_forces * contact_friction_spring_cur
            
            ### TODO: how to check the correctness of the switching between the static friction and the dynamic friction ###
            if self.use_static_mus:
                ''' Use the relative scale between the static_mu * contact force and the penalty friction to decide dynamic and static frictions '''
                contact_active_penalty_based_friction_forces_norm = torch.norm(contact_active_penalty_based_friction_forces, p=2, dim=-1)
                contact_friction_static_mu = 10.
                contact_friction_static_mu = 1.
                contact_friction_static_mu = self.contact_friction_static_mu
                ## remaining contact indicator ##
                remaining_contact_indicator = contact_active_penalty_based_friction_forces_norm <= (contact_friction_static_mu * contact_active_force_d_scalar)
                
                ### not_remaining_contacts ###
                not_remaining_contacts = contact_active_penalty_based_friction_forces_norm > (contact_friction_static_mu * contact_active_force_d_scalar)
                
                ## contact active penalty based friction forces ##
                contact_active_penalty_based_friction_forces_dir = contact_active_penalty_based_friction_forces / torch.clamp(contact_active_penalty_based_friction_forces_norm.unsqueeze(-1), min=1e-5)
                dyn_contact_active_penalty_based_friction_forces = contact_active_penalty_based_friction_forces_dir * (contact_friction_static_mu * contact_active_force_d_scalar).unsqueeze(-1)
                
                if self.debug:
                    avg_contact_active_penalty_based_friction_forces_norm = torch.mean(contact_active_penalty_based_friction_forces_norm)
                    avg_contact_active_force_d_scalar = torch.mean(contact_active_force_d_scalar)
                    print(f"avg_contact_active_force_d_scalar: {avg_contact_active_force_d_scalar}, avg_contact_active_penalty_based_friction_forces_norm: {avg_contact_active_penalty_based_friction_forces_norm}, contact_friction_static_mu: {contact_friction_static_mu}")
                    print(f"contact_active_force_d_scalar: {contact_active_force_d_scalar[:10]}, contact_active_penalty_based_friction_forces_norm: {contact_active_penalty_based_friction_forces_norm[:10]}")
                    nn_remaining_contact = torch.sum(remaining_contact_indicator).item()
                    print(f"nn_remaining_contact / tot_contacts: {nn_remaining_contact} / {contact_active_penalty_based_friction_forces.size(0)}")
                
                ### reamining contact based frictions ### remaining cotnacts and not remaining contacts #
                not_remaining_contacts_mask = torch.zeros((contact_active_penalty_based_friction_forces.size(0)), dtype=torch.float32).cuda()
                not_remaining_contacts_mask[not_remaining_contacts] = 1
                ''' Use the relative scale between the static_mu * contact force and the penalty friction to decide dynamic and static frictions '''
                
                ''' Update penalty based friction forces '''
                contact_active_penalty_based_friction_forces = contact_active_penalty_based_friction_forces * (1. - not_remaining_contacts_mask).unsqueeze(-1) + dyn_contact_active_penalty_based_friction_forces * not_remaining_contacts_mask.unsqueeze(-1)
                ''' Update penalty based friction forces '''
            ### TODO: how to check the correctness of the switching between the static friction and the dynamic friction ###
            
            
            # # penalty_friction_tangential_forces[contact_active_idxes] = penalty_based_friction_forces * contact_friction_spring_cur # * 0.1
            # penalty_friction_tangential_forces[contact_active_idxes] = contact_active_penalty_based_friction_forces ## 0.1 ##
            
            # contact_active_idxes_mask = torch.zeros((penalty_friction_tangential_forces.size(0)), dtype=torch.float32).cuda()
            # contact_active_idxes_mask[contact_active_idxes] = 1 ##### ### contact_active_idxes ### ##### ## 
            # contact_active_idxes_mask = contact_active_idxes_mask.unsqueeze(-1).repeat(1, 3).contiguous()
            # ### or I think that the penalty 
            # penalty_friction_tangential_forces = torch.where( # 
            #     contact_active_idxes_mask > 0.5, penalty_friction_tangential_forces, torch.zeros_like(penalty_friction_tangential_forces)
            # )
            expanded_contact_active_idxes = contact_active_idxes.unsqueeze(-1).contiguous().repeat(1, 3).contiguous()
            penalty_friction_tangential_forces = torch.scatter(penalty_friction_tangential_forces, dim=0, index=expanded_contact_active_idxes, src=contact_active_penalty_based_friction_forces)
            
            
            # ### add the sqrt for calculate the l2 distance ###
            # dist_cur_active_to_passive = torch.sqrt(dist_cur_active_to_passive)
            
            # dist_cur_active_to_passive = torch.norm(
            #     (contact_active_pos - contact_passive_pos), dim=-1, p=2
            # )
            
            
            # inter_obj_normals[contact_active_idxes] = cur_passive_obj_ns[contact_passive_idxes]
            # contact_force_d[contact_active_idxes] = cur_contact_d
            
            ''' Update inter_obj_normals and contact_force_d '''
            inter_obj_normals = torch.scatter( # 
                inter_obj_normals, dim=0, index=expanded_contact_active_idxes, src=cur_passive_obj_ns[contact_passive_idxes]
            )
            contact_force_d = torch.scatter(
                contact_force_d, dim=0, index=expanded_contact_active_idxes, src=cur_contact_d
            )
            ''' Update inter_obj_normals and contact_force_d '''
            
            
            cur_act_weights[contact_active_idxes] = 1.
            ws_unnormed[contact_active_idxes] = 1.
        else:
            contact_active_idxes = None
            self.contact_active_idxes = contact_active_idxes
            valid_penalty_friction_forces_indicator = None
            penalty_based_friction_forces = None
        # tangential forces with inter obj normals # -> ####
        if torch.sum(cur_act_weights).item() > 0.5:
            cur_act_weights = cur_act_weights / torch.sum(cur_act_weights)
        
        
        # norm_penalty_friction_tangential_forces = torch.norm(penalty_friction_tangential_forces, dim=-1, p=2)
        # maxx_norm_penalty_friction_tangential_forces, _ = torch.max(norm_penalty_friction_tangential_forces, dim=-1)
        # minn_norm_penalty_friction_tangential_forces, _ = torch.min(norm_penalty_friction_tangential_forces, dim=-1)
        # print(f"maxx_norm_penalty_friction_tangential_forces: {maxx_norm_penalty_friction_tangential_forces}, minn_norm_penalty_friction_tangential_forces: {minn_norm_penalty_friction_tangential_forces}")
        
        
        ####### penalty_friction_tangential_forces #######
        # tangetntial forces --- dot with normals #
        # dot_tangential_forces_with_inter_obj_normals = torch.sum(penalty_friction_tangential_forces * inter_obj_normals, dim=-1) ### nn_active_pts x # 
        # penalty_friction_tangential_forces = penalty_friction_tangential_forces - dot_tangential_forces_with_inter_obj_normals.unsqueeze(-1) * inter_obj_normals
        ####### penalty_friction_tangential_forces #######
        
        
        
        # self.penetrating_forces = penetrating_forces #
        # self.penetrating_points = penetrating_points #
        
        ### TODO: is that stable? ###
        ####### update the forces to the active object #######
        ### next: add the negative frictional forces to the penetrating forces ###
        # penetrating_friction_to_active_mult_factor = 1.0
        # penetrating_friction_to_active_mult_factor = 0.001
        # penetrating_penalty_friction_forces_to_active = -1. * penalty_friction_tangential_forces ### get the tangential forces ###
        # penetrating_penalty_friction_forces_to_active = penetrating_penalty_friction_forces_to_active[penetrating_indicator]
        # self.penetrating_forces = self.penetrating_forces + penetrating_penalty_friction_forces_to_active * penetrating_friction_to_active_mult_factor
        # ## add the negative frictional forces to the repulsion forces to the active object ##
        # penetrating_forces = penetrating_forces + penetrating_penalty_friction_forces_to_active * penetrating_friction_to_active_mult_factor
        ####### penetrating forces #######
        ### TODO: is that stable? ###
        
        
        ####### penalty_based_friction_forces #######
        # norm_penalty_friction_tangential_forces = torch.norm(penalty_friction_tangential_forces, dim=-1, p=2)
        # # valid penalty friction forces # # valid contact force d scalar #
        # maxx_norm_penalty_friction_tangential_forces, _ = torch.max(norm_penalty_friction_tangential_forces, dim=-1)
        # minn_norm_penalty_friction_tangential_forces, _ = torch.min(norm_penalty_friction_tangential_forces, dim=-1)
        # print(f"[After proj.] maxx_norm_penalty_friction_tangential_forces: {maxx_norm_penalty_friction_tangential_forces}, minn_norm_penalty_friction_tangential_forces: {minn_norm_penalty_friction_tangential_forces}")
        
        
        # norm_tangential_forces = torch.norm(tangential_forces, dim=-1, p=2) # tangential forces #
        # maxx_norm_tangential, _ = torch.max(norm_tangential_forces, dim=-1)
        # minn_norm_tangential, _ = torch.min(norm_tangential_forces, dim=-1)
        # print(f"maxx_norm_tangential: {maxx_norm_tangential}, minn_norm_tangential: {minn_norm_tangential}")
        
        ### ## get new contacts ## ###
        tot_contact_point_position = []
        tot_contact_active_point_pts = []
        tot_contact_active_idxes = []
        tot_contact_passive_idxes = []
        tot_contact_frame_rotations = []
        tot_contact_frame_translations = []
        
        
        if contact_pairs_set is not None:
            if torch.sum(remaining_contact_indicator.float()) > 0.5:
                # contact_active_point_pts, contact_point_position, (contact_active_idxes, contact_passive_idxes), contact_frame_pose = contact_pairs_set
                remaining_contact_active_point_pts = contact_active_point_pts[remaining_contact_indicator]
                remaining_contact_point_position = contact_point_position[remaining_contact_indicator]
                remaining_contact_active_idxes = contact_active_idxes[remaining_contact_indicator]
                remaining_contact_passive_idxes = contact_passive_idxes[remaining_contact_indicator]
                remaining_contact_frame_rotations = contact_frame_orientations[remaining_contact_indicator]
                remaining_contact_frame_translations = contact_frame_translations[remaining_contact_indicator]
                tot_contact_point_position.append(remaining_contact_point_position)
                tot_contact_active_point_pts.append(remaining_contact_active_point_pts)
                tot_contact_active_idxes.append(remaining_contact_active_idxes)
                tot_contact_passive_idxes.append(remaining_contact_passive_idxes)
                tot_contact_frame_rotations.append(remaining_contact_frame_rotations)
                tot_contact_frame_translations.append(remaining_contact_frame_translations)
                
                # remaining_contact_active_idxes
                in_contact_indicator[remaining_contact_active_idxes] = False
        
        
        
        if torch.sum(in_contact_indicator.float()) > 0.5: # in contact indicator #
            cur_in_contact_passive_pts = inter_obj_pts[in_contact_indicator]
            # cur_in_contact_passive_normals = inter_obj_normals[in_contact_indicator] ##
            cur_in_contact_active_pts = sampled_input_pts[in_contact_indicator] # in_contact_active_pts ##
            
            
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
            
        
        
        if self.use_penalty_based_friction and self.use_disp_based_friction:
            disp_friction_tangential_forces = nex_sampled_input_pts - sampled_input_pts
            
            if self.use_split_params:
                contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda() + input_pts_ts).view(1,)
            else:
                contact_friction_spring_cur = self.spring_friction_ks_values(torch.zeros((1,), dtype=torch.long).cuda()).view(1,) # time-varying?
            
            disp_friction_tangential_forces = disp_friction_tangential_forces * contact_friction_spring_cur
            disp_friction_tangential_forces_dot_normals = torch.sum(
                disp_friction_tangential_forces * inter_obj_normals, dim=-1
            )
            disp_friction_tangential_forces = disp_friction_tangential_forces - disp_friction_tangential_forces_dot_normals.unsqueeze(-1) * inter_obj_normals
            
            penalty_friction_tangential_forces = disp_friction_tangential_forces
        
        
        ### strict cosntraints ###
        if self.use_penalty_based_friction:
            forces = penalty_friction_tangential_forces + contact_force_d # tantential forces and contact force d #
        else:
            # print(f"not using use_penalty_based_friction...")
            tangential_forces_norm = torch.sum(tangential_forces ** 2, dim=-1)
            pos_tangential_forces = tangential_forces[tangential_forces_norm > 1e-5]
            forces = tangential_forces + contact_force_d # tantential forces and contact force d #
        # forces = penalty_friction_tangential_forces + contact_force_d # tantential forces and contact force d #
        ''' decide forces via kinematics statistics '''

        ''' Decompose forces and calculate penalty froces ''' # 
        # penalty_dot_forces_normals, penalty_friction_constraint # # contraints # # 
        # # get the forces -> decompose forces # 
        dot_forces_normals = torch.sum(inter_obj_normals * forces, dim=-1) ### nn_sampled_pts ###
        # forces_along_normals = dot_forces_normals.unsqueeze(-1) * inter_obj_normals ## the forces along the normal direction ##
        # tangential_forces = forces - forces_along_normals # tangential forces # # tangential forces ### tangential forces ##
        
        #### penalty_friction_tangential_forces, tangential_forces ####
        self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
        self.tangential_forces = penalty_friction_tangential_forces
        
        self.penalty_friction_tangential_forces = penalty_friction_tangential_forces
        self.contact_force_d = contact_force_d
        self.penalty_based_friction_forces = penalty_based_friction_forces
    
        
        penalty_dot_forces_normals = dot_forces_normals ** 2
        penalty_dot_forces_normals[dot_forces_normals <= 0] = 0 # 1) must in the negative direction of the object normal #
        penalty_dot_forces_normals = torch.mean(penalty_dot_forces_normals) # 1) must # 2) must #
        self.penalty_dot_forces_normals = penalty_dot_forces_normals #
        
        
        rigid_acc = torch.sum(forces * cur_act_weights.unsqueeze(-1), dim=0)
        ###### sampled input pts to center #######
        if contact_pairs_set is not None:
            inter_obj_pts[contact_active_idxes] = cur_passive_obj_verts[contact_passive_idxes]
        
        # center_point_to_sampled_pts = sampled_input_pts - passive_center_point.unsqueeze(0)
        center_point_to_sampled_pts = inter_obj_pts - passive_center_point.unsqueeze(0)
        
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
        
        if self.use_split_params:
            # sep_time_constant, sep_torque_time_constant, sep_damping_constant, sep_angular_damping_constant
            time_cons = self.sep_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            time_cons_2 = self.sep_torque_time_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            damping_cons = self.sep_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            damping_cons_2 = self.sep_angular_damping_constant(torch.zeros((1,)).long().cuda() + input_pts_ts).view(1)
            
        
        
        
        k_acc_to_vel = time_cons
        k_vel_to_offset = time_cons_2
        delta_vel = rigid_acc * k_acc_to_vel
        if input_pts_ts == 0:
            cur_vel = delta_vel
        else:
            ##### TMP ######
            # cur_vel = delta_vel
            ##### TMP ######
            cur_vel = delta_vel + self.timestep_to_vel[input_pts_ts - 1].detach() * damping_cons
        self.timestep_to_vel[input_pts_ts] = cur_vel.detach() # ts to the velocity # 
         
        cur_offset = k_vel_to_offset * cur_vel 
        cur_rigid_def = self.timestep_to_total_def[input_pts_ts].detach() # ts to total def #
        
        
        delta_angular_vel = torque * time_cons_rot
        if input_pts_ts == 0:
            cur_angular_vel = delta_angular_vel
        else:
            ##### TMP ######
            # cur_angular_vel = delta_angular_vel
            cur_angular_vel = delta_angular_vel + self.timestep_to_angular_vel[input_pts_ts - 1].detach() * damping_cons_rot ### (3,)
        cur_delta_angle = cur_angular_vel * time_cons_rot # \delta_t w^1 / 2 
        
        prev_quaternion = self.timestep_to_quaternion[input_pts_ts].detach() # 
        cur_quaternion = prev_quaternion + update_quaternion(cur_delta_angle, prev_quaternion)
        
        
        cur_optimizable_rot_mtx = quaternion_to_matrix(cur_quaternion)
        prev_rot_mtx = quaternion_to_matrix(prev_quaternion)
        
        # cur_
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_rigid_def.unsqueeze(0), cur_delta_rot_mtx.detach()).squeeze(0)
        # cur_upd_rigid_def = cur_offset.detach() + torch.matmul(cur_delta_rot_mtx.detach(), cur_rigid_def.unsqueeze(-1)).squeeze(-1)
        cur_upd_rigid_def = cur_offset.detach() + cur_rigid_def # update the current rigid def using the offset and the cur_rigid_def ## # 
        # curupd
        # if update_tot_def: # update rigid def #
        
        
        
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
