'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''
from torch import nn
import torch
import ControlVAECore.Utils.pytorch_utils as ptu
# import ControlVAECore.Utils.diff_quat as DiffRotation
# from ControlVAECore.Utils.motion_utils import *
import trimesh
# import open3d as o3d
from pyhocon import ConfigFactory
import ControlVAECore.Model.fields as fields
import ControlVAECore.Model.dyn_model_act_v2 as dyn_model_act_mano
from scipy.spatial.transform import Rotation as R


def update_quaternion_batched(delta_angle, prev_quat):
    
    s1 = 0
    s2 = prev_quat[:, 0] # s2 
    v2 = prev_quat[:, 1:] # v2 
    v1 = delta_angle / 2
    new_v = s1 * v2 + s2.unsqueeze(-1) * v1 + torch.cross(v1, v2, dim=-1)
    new_s = s1 * s2 - torch.sum(v1 * v2, dim=-1) ## nb # 
    new_quat = torch.cat([new_s.unsqueeze(-1), new_v], dim=-1)
    return new_quat

## update quaternion batched ##
# @torch.jit.script
def integrate_obs_vel(states, delta):
    shape_ori = states.shape

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"

    if len(delta.shape) == 2:
        delta = delta[None]
    assert len(delta.shape) == 3, "acc shape error"
    
    assert delta.shape[-1] == 6
    assert states.shape[-1] == 13

    batch_size, num_body, _ = states.shape

    
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)

    delta = delta.view(-1, 6)
    local_delta = delta[..., 0:3]
    local_ang_delta = delta[..., 3:6]

    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    true_delta = quat_apply(root_rot,local_delta)
    true_ang_delta = quat_apply(root_rot, local_ang_delta)

    next_vel = vel + true_delta
    next_avel = avel + true_ang_delta
    return next_vel, next_avel


def integrate_state(states, delta, dt):
    pos = states[..., 0:3].contiguous().view(-1, 3)
    rot = states[..., 3:7].contiguous().view(-1, 4)
    
    next_vel, next_avel = integrate_obs_vel(states, delta)
    next_pos = pos + next_vel * dt
    next_rot = quat_integrate(rot, next_avel, dt)
    
    batch_size, num_body, _ = states.shape
    
    next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                                 next_rot.view(batch_size, num_body, 4),
                                 next_vel.view(batch_size, num_body, 3),
                                 next_avel.view(batch_size, num_body, 3)
                                 ], dim=-1)

    return next_state.view(states.shape)


# def split_state(self, state):
#     mano_trans = state[:, : self.bullet_mano_num_trans_state]
#     mano_rot = state[:, self.bullet_mano_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
#     mano_states = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]
    
#     obj_rot = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state]
#     obj_trans = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state + self.obj_num_trans_state]
#     return mano_trans, mano_rot, mano_states, obj_rot, obj_trans

# def split_delta(self, delta):
#     delta_mano_trans = delta[:, :self.bullet_mano_num_trans_state] ## determined by the rot state ##
#     delta_mano_rot = delta[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state]
#     delta_mano_states = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints] ## delta mano  states ##
    
#     delta_obj_rot = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act]
#     delta_obj_trans = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act + self.obj_num_trans_act]
#     ## delta obj trans delta obj rot ##
#     return delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_rot, delta_obj_trans

## TODO: set world model weights by world_model_weight_XXX ##


def split_state(state):
    nn_finger_states = state.size(-1) - (3 + 3) - (3 + 4)
    mano_trans = state[:, : 3]
    mano_rot = state[:, 3: 3 + 3]
    mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    obj_rot = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 4]
    obj_trans = state[:, 6 + nn_finger_states + 4: 6 + nn_finger_states + 4 + 3]
    return mano_trans, mano_rot, mano_states, obj_rot, obj_trans
    
## simpleworldmodel ## 
class SimpleWorldModel(nn.Module):
    def __init__(self, ob_size, ac_size, delta_size, dt, statistics, **kargs) -> None:
        super(SimpleWorldModel, self).__init__()
        
        
        self.wana = kargs['wana']
        
        if self.wana:
            self.input_dim = ob_size + ob_size + ac_size
        else:
            self.input_dim = ob_size + ac_size
        
        ##### mlp #####
        self.mlp = ptu.build_mlp(
            input_dim = self.input_dim, # rotvec to 6d vector # delta size ## ac size # 
            output_dim= delta_size,
            hidden_layer_num=kargs['world_model_hidden_layer_num'],
            hidden_layer_size=kargs['world_model_hidden_layer_size'],
            activation=kargs['world_model_activation']
        )
        
        ## observation size = ##
        
        ## bullet_mano_num_joints, bullet_mano_finger_num_joints, bullet_mano_num_rot_state, bullet_mano_num_trans_state ###
        ## obj_num_rot_state, obj_num_rot_act, obj_num_trans_state, obj_num_trans_act ##
        self.bullet_mano_num_joints = kargs['bullet_mano_num_joints']
        self.bullet_mano_finger_num_joints = kargs['bullet_mano_finger_num_joints']
        self.bullet_mano_num_rot_state = 3
        self.bullet_mano_num_trans_state = 3
        self.obj_num_rot_state = 4
        self.obj_num_rot_act = 3
        self.obj_num_trans_state = 3
        self.obj_num_trans_act = 3

        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad= False)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False)
        # delta mean; delta std ##
        self.delta_mean = nn.Parameter(ptu.from_numpy(statistics['delta_mean']), requires_grad= False)
        self.delta_std = nn.Parameter(ptu.from_numpy(statistics['delta_std']), requires_grad= False) ## delta states and delta manea # and delta states ##
        
        ## dt ##
        self.dt = dt
        
        self.weight = {}
        for key,value in kargs.items():
            if 'world_model_weight' in key:
                self.weight[key.replace('world_model_weight_','')] = value
        for k in ['mano_trans', 'mano_rot', 'mano_states', 'obj_rot', 'obj_trans']: # 
            self.weight[f'weight_{k}'] = 1.0
        
        
        case = "test_wm_ana"
        conf_path = kargs['conf_path']
        self.nn_instances = 1
        self.conf_path = conf_path
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        ## conf and parse string ##  ## CASE NA<e ##
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        # self.base_exp_dir = self.conf['general.base_exp_dir']
        # self.base_exp_dir = self.base_exp_dir + f"_reverse_value_totviews_tag_{self.conf['general.tag']}"
        # os.makedirs(self.base_exp_dir, exist_ok=True)
        self.wnorm = kargs['wnorm']
        
        if 'model.minn_dist_threshold' in self.conf:
            self.minn_dist_threshold = self.conf['model.minn_dist_threshold']
        else:
            self.minn_dist_threshold = 0.05
        if 'model.use_sqrt_dist' in self.conf:
            self.use_sqrt_dist = self.conf['model.use_sqrt_dist']
        else:
            self.use_sqrt_dist = False
        if 'model.contact_maintaining_dist_thres' in self.conf:
            self.contact_maintaining_dist_thres = float(self.conf['model.contact_maintaining_dist_thres'])
        else:
            self.contact_maintaining_dist_thres = 0.1
        if 'model.penetration_proj_k_to_robot' in self.conf:
            self.penetration_proj_k_to_robot = float(self.conf['model.penetration_proj_k_to_robot'])
        else:
            self.penetration_proj_k_to_robot = 0.0
        if 'model.use_split_params' in self.conf:
            self.use_split_params = self.conf['model.use_split_params']
        else:
            self.use_split_params = False
        if 'model.use_sqr_spring_stiffness' in self.conf:
            self.use_sqr_spring_stiffness = self.conf['model.use_sqr_spring_stiffness']
        else:
            self.use_sqr_spring_stiffness = False
        if 'model.use_pre_proj_frictions' in self.conf:
            self.use_pre_proj_frictions = self.conf['model.use_pre_proj_frictions']
        else:
            self.use_pre_proj_frictions = False
        if 'model.use_static_mus' in self.conf:
            self.use_static_mus = self.conf['model.use_static_mus']
        else:
            self.use_static_mus = False
        if 'model.contact_friction_static_mu' in self.conf:
            self.contact_friction_static_mu = self.conf['model.contact_friction_static_mu']
        else:
            self.contact_friction_static_mu = 1.0
        if 'model.debug' in self.conf:
            self.debug = self.conf['model.debug']
        else:
            self.debug  = False
        if 'model.use_sdf_as_contact_dist' in self.conf:
            self.use_sdf_as_contact_dist = self.conf['model.use_sdf_as_contact_dist']
        else:
            self.use_sdf_as_contact_dist = False
        if 'model.use_contact_dist_as_sdf' in self.conf:
            self.use_contact_dist_as_sdf = self.conf['model.use_contact_dist_as_sdf']
        else:
            self.use_contact_dist_as_sdf = False
        if 'model.minn_dist_threshold_robot_to_obj' in self.conf:
            self.minn_dist_threshold_robot_to_obj = float(self.conf['model.minn_dist_threshold_robot_to_obj'])
        else:
            self.minn_dist_threshold_robot_to_obj = 0.0
        if 'model.use_same_contact_spring_k' in self.conf:
            self.use_same_contact_spring_k = self.conf['model.use_same_contact_spring_k']
        else:
            self.use_same_contact_spring_k = False
        if 'model.obj_mass' in self.conf:
            self.obj_mass = float(self.conf['model.obj_mass'])
        else:
            self.obj_mass = 100.0
        if 'model.contact_maintaining_dist_thres' in self.conf:
            self.contact_maintaining_dist_thres = float(self.conf['model.contact_maintaining_dist_thres'])
        else:
            self.contact_maintaining_dist_thres = 0.1
        
        if 'model.train_residual_friction' in self.conf:
            self.train_residual_friction = self.conf['model.train_residual_friction']
        else:
            self.train_residual_friction = False
        
        ## world model 
        
        ## load the conf here ##
        self.ana_sim = fields.BendingNetworkActiveForceFieldForwardLagV18(**self.conf['model.bending_network'], nn_instances=self.nn_instances, minn_dist_threshold=self.minn_dist_threshold).to(self.device)
        
        self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh()
        self.ana_sim.sdf_space_center = self.sdf_space_center
        self.ana_sim.sdf_space_scale = self.sdf_space_scale
        self.ana_sim.obj_sdf = self.obj_sdf # 
        self.ana_sim.sdf_res = self.sdf_res # 
        
        self.calculate_obj_inertia()
        
        ##### load the dyn model #####
        if 'model.ckpt_fn' in self.conf and len(self.conf['model.ckpt_fn']) > 0:
            cur_ckpt_fn = self.conf['model.ckpt_fn']
            if not os.path.exists(cur_ckpt_fn):
                cur_ckpt_fn = "/data2/xueyi/ckpt_191000.pth"
            self.load_checkpoint_via_fn(cur_ckpt_fn)
        
        model_path_mano = self.conf['model.mano_sim_model_path'] ## model path mano ## ## 
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano, dev=self.device )
        self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### ## ##
        self.mult_after_center = 3.
        self.cur_ts = 0
        
        self.calculate_collision_geometry_bounding_boxes()
        
        self.n_timesteps = self.conf['model.n_timesteps'] #
        for i_time_idx in range(self.n_timesteps):
            ## ana_sim ##
            self.ana_sim.timestep_to_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_point_accs[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_total_def[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_angular_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_quaternion[i_time_idx] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_torque[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)

        
        
        self.ana_sim.use_penalty_based_friction = True
        self.ana_sim.use_disp_based_friction = False
        self.ana_sim.use_sqrt_dist = self.use_sqrt_dist
        self.ana_sim.contact_maintaining_dist_thres = self.contact_maintaining_dist_thres
        self.ana_sim.penetration_proj_k_to_robot = self.penetration_proj_k_to_robot
        self.ana_sim.use_split_params = self.use_split_params
        self.ana_sim.use_sqr_spring_stiffness = self.use_sqr_spring_stiffness
        self.ana_sim.use_pre_proj_frictions = self.use_pre_proj_frictions
        self.ana_sim.use_static_mus = self.use_static_mus
        self.ana_sim.contact_friction_static_mu = self.contact_friction_static_mu
        self.ana_sim.debug = self.debug
        self.ana_sim.obj_sdf_grad = self.obj_sdf_grad ## set obj_sdf #
        self.ana_sim.use_sdf_as_contact_dist = self.use_sdf_as_contact_dist
        self.ana_sim.use_contact_dist_as_sdf = self.use_contact_dist_as_sdf
        self.ana_sim.minn_dist_threshold_robot_to_obj = self.minn_dist_threshold_robot_to_obj
        self.ana_sim.use_same_contact_spring_k = self.use_same_contact_spring_k
        self.ana_sim.I_ref = self.I_ref
        self.ana_sim.I_inv_ref = self.I_inv_ref
        self.ana_sim.obj_mass = self.obj_mass
        
        # self.maxx_init_passive_mesh, self.minn_init_passive_mesh
        self.ana_sim.maxx_init_passive_mesh = self.maxx_init_passive_mesh
        self.ana_sim.minn_init_passive_mesh = self.minn_init_passive_mesh # ### init maximum passive meshe #
        self.ana_sim.train_residual_friction = self.train_residual_friction
        self.ana_sim.n_timesteps = self.n_timesteps
        
        ### set the initial contact pairs set ###
        self.contact_pairs_set = None
        
        # self.gt_data_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/bullet_hoi_data.npy"
        self.gt_data_fn = kargs['sv_gt_refereces_fn']
        self.gt_data = np.load(self.gt_data_fn, allow_pickle=True).item()
        
        self.mano_glb_rot = self.gt_data['mano_glb_rot'] ## mano glb rot  # in the quaternion 
        self.mano_glb_rot = self.mano_glb_rot / np.clip(np.sqrt(np.sum(self.mano_glb_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        
        # new_mano_glb_rot = []
        # for i_fr in range(self.mano_glb_rot.shape[0]):
        #     cur_glb_rot_quat = self.mano_glb_rot[i_fr]
        #     cur_glb_rot_quat = cur_glb_rot_quat[[1, 2, 3, 0]]
        #     cur_glb_rot_struct = R.from_quat(cur_glb_rot_quat)
        #     cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
        #     cur_glb_rot_mtx = cur_glb_rot_mtx.T
        #     cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx)
        #     cur_glb_rot_quat = cur_glb_rot_struct.as_quat()
        #     cur_glb_rot_quat = cur_glb_rot_quat[[3, 0, 1, 2]]
        #     new_mano_glb_rot.append(cur_glb_rot_quat)
        # new_mano_glb_rot = np.stack(new_mano_glb_rot, axis=0)
        # self.mano_glb_rot = new_mano_glb_rot
        
        self.mano_glb_trans = self.gt_data['mano_glb_trans'] #
        ## mano rot ## ## reset the state ---> 
        self.mano_states = self.gt_data['mano_states'][:, :self.bullet_mano_finger_num_joints]
        
        self.mano_states[:, 56 - 6] = 3.0
        self.mano_states[:, 58 - 6] = 0.9
        
        self.obj_rot = self.gt_data['obj_rot'] # in the quaternion #
        if self.obj_rot.shape[1] == 3:
            tot_obj_rot_quat = []
            for i_fr in range(self.obj_rot.shape[0]):
                
                cur_obj_rot_vec = self.obj_rot[i_fr]
                cur_obj_rot_struct = R.from_rotvec(cur_obj_rot_vec)
                
                cur_obj_rot_mtx = cur_obj_rot_struct.as_matrix()
                cur_obj_rot_mtx = cur_obj_rot_mtx.T
                cur_obj_rot_struct = R.from_matrix(cur_obj_rot_mtx)
                
                
                cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
                
                # cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
                cur_obj_rot_quat = cur_obj_rot_quat[[3, 0, 1, 2]] ## as 3, 0, 1, 2
                tot_obj_rot_quat.append(cur_obj_rot_quat)
            tot_obj_rot_quat = np.stack(tot_obj_rot_quat, axis=0)
            self.obj_rot = tot_obj_rot_quat
        else:
            self.obj_rot = self.obj_rot / np.clip(np.sqrt(np.sum(self.obj_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        self.obj_trans = self.gt_data['obj_trans'] # # 
        
        self.timestep_to_active_mesh = {}
        # 
        # # set the base rotation and translation ## # 
        sampled_verts_idxes_fn = "/home/xueyi/diffsim/NeuS/robo_sampled_verts_idxes.npy"
        sampled_verts_idxes = np.load(sampled_verts_idxes_fn)
        sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().to(self.device)
        self.sampled_verts_idxes = sampled_verts_idxes # sampled verts idxes
        
        self.ts_to_mano_bullet_states = {}
        self.ts_to_mano_bullet_glb_rot = {}
        self.ts_to_mano_bullet_glb_trans = {}
    
    ## reset the state and step the action #
    
    def calculate_obj_inertia(self, ):
        cur_init_passive_mesh_verts = self.timestep_to_passive_mesh[0].clone()
        cur_init_passive_mesh_center = torch.mean(cur_init_passive_mesh_verts, dim=0)
        cur_init_passive_mesh_verts = cur_init_passive_mesh_verts - cur_init_passive_mesh_center
        # per_vert_mass=  cur_init_passive_mesh_verts.size(0) / self.obj_mass
        per_vert_mass = self.obj_mass / cur_init_passive_mesh_verts.size(0) 
        
        I_ref = torch.zeros((3, 3), dtype=torch.float32).to(self.device)
        for i_v in range(cur_init_passive_mesh_verts.size(0)):
            cur_vert = cur_init_passive_mesh_verts[i_v]
            cur_r = cur_vert # - cur_init_passive_mesh_center
            cur_v_inertia = per_vert_mass * (torch.sum(cur_r * cur_r) - torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0)))
            # cur_v_inertia = torch.cross(cur_r, cur_r) * per_vert_mass3 # # 
            I_ref += cur_v_inertia

        self.I_inv_ref = torch.linalg.inv(I_ref)
        self.I_ref = I_ref
    
    def load_checkpoint_via_fn(self, checkpoint_fn):
        print(f"Loading from {checkpoint_fn}")
        checkpoint = torch.load(checkpoint_fn, map_location=self.device, )
        
        self.ana_sim.load_state_dict(checkpoint['dyn_model'], strict=False)
        
        # logging.info(f"checkpoint with sdf_net and bending_net loaded from {checkpoint_fn}")
        # # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # # self.iter_step = checkpoint['iter_step']
        # logging.info('End')
        
    # rreset the state and step the action #
    def reset(self, state, target_ts):
        ## no actions if using no analytical sim ##
        self.contact_pairs_set = None
        # reset to the tans rot trans rot # close 
        mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state.unsqueeze(0))
        
    
        init_mano_states = mano_states.squeeze(0)
        init_mano_rot = mano_rot.squeeze(0)
        init_mano_trans =mano_trans.squeeze(0)
        
        # mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state)
        
        ## get init obj rot ##
        ## reset state and get observations ##
        init_obj_rot = obj_rot.squeeze(0) # the obj rot is in the xyzw format
        init_obj_rot = init_obj_rot[[3, 0, 1, 2]]
        init_obj_trans = obj_trans.squeeze(0)
        
        
        if self.wana:
            self.ana_sim.timestep_to_total_def[target_ts] = init_obj_trans.clone()
            self.ana_sim.timestep_to_optimizable_quaternion[target_ts] = init_obj_rot.clone()
            cur_obj_rot_mtx = fields.quaternion_to_matrix(init_obj_rot)
            self.ana_sim.timestep_to_optimizable_rot_mtx[target_ts] = cur_obj_rot_mtx.clone()
            # self.ana_sim.timestep_to_total_def[target_ts] = init_obj_trans.clone()
            self.ana_sim.timestep_to_quaternion[target_ts] = init_obj_rot.clone() # init obj rot vec and the rot mtx ## 
            self.ana_sim.timestep_to_angular_vel[max(0, target_ts - 1)] = torch.zeros_like(init_obj_trans)
            self.ana_sim.timestep_to_vel[max(0, target_ts - 1)] = torch.zeros_like(init_obj_trans) ## initjvel; angular vel # ## init vel and init angular vel
            
            init_active_mesh_pts=  self.forward_kinematics(init_mano_trans, init_mano_rot, init_mano_states)
            self.timestep_to_active_mesh[target_ts] = init_active_mesh_pts
            
            
            ## the target_ts + 1's object rot and trans would be calculated here ##
            self.contact_pairs_set = self.ana_sim.forward2( input_pts_ts=target_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
        
        self.cur_ts = target_ts
        
        # init_mano_rot_struct = R.from_quat(init_mano_rot.detach().cpu().numpy()[[1, 2, 3, 0]])
        # init_mano_rot_vec = init_mano_rot_struct.as_rotvec()
        # init_mano_rot_vec = torch.from_numpy(init_mano_rot_vec).float().to(self.device)
        init_mano_rot_vec = init_mano_rot
        self.ts_to_mano_bullet_glb_rot[self.cur_ts] = init_mano_rot_vec
        self.ts_to_mano_bullet_glb_trans[self.cur_ts] = init_mano_trans
        self.ts_to_mano_bullet_states[self.cur_ts] = init_mano_states ## init mano states 
        
        
        
    ## def ## ## def ## ## 
    ## filed observation of the  observation ## ## filed observation ##
    def normalize_obs(self, observation):
        # return observation
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        if self.wnorm:
            observation = ptu.normalize(observation, self.obs_mean, self.obs_std)
        else:
            observation = observation
        return observation
    # 
    ### laod active passive timestp 
    def load_active_passive_timestep_to_mesh(self,): ## 
        # sv_fn = "/data2/datasets/sim/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy" # load 
        sv_fn = "/data2/xueyi/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy"
        if not os.path.exists(sv_fn):
            sv_fn = "/data/xueyi/arctic/processed_sv_dicts/box_grab_01_extracted_dict.npy"
        active_passive_sv_dict = np.load(sv_fn, allow_pickle=True).item() # 

        obj_verts = active_passive_sv_dict['obj_verts'] # object orientation #
        obj_faces = active_passive_sv_dict['obj_faces']
        
        init_obj_verts = obj_verts[0]
        
        # the solution of
        ### get the boundary information of the SDF grid ###
        # init_obj_verts # 
        mesh_scale = 0.8
        bbmin = init_obj_verts.min(0) #
        bbmax = init_obj_verts.max(0) #
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
        # vertices = (vertices - center) * scale # (vertices - center) * scale # #
        
        ### load sdfs information ###
        self.sdf_space_center = center
        self.sdf_space_scale = scale
        sdf_sv_fn = "/data/xueyi/diffsim/NeuS/init_box_mesh.npy"
        if not os.path.exists(sdf_sv_fn):
            sdf_sv_fn = "/home/xueyi/diffsim/NeuS/init_box_mesh.npy"
        self.obj_sdf = np.load(sdf_sv_fn, allow_pickle=True)
        self.sdf_res = self.obj_sdf.shape[0]
        print(f"obj_sdf loaded from {sdf_sv_fn} with shape {self.obj_sdf.shape}")
        
        sdf_grad_sv_fn = "/home/xueyi/diffsim/NeuS/init_box_mesh_sdf_grad.npy"
        if os.path.exists(sdf_grad_sv_fn):
            self.obj_sdf_grad = np.load(sdf_grad_sv_fn, allow_pickle=True)
            print(f"obj_sdf_grad loaded from {sdf_grad_sv_fn} with shape {self.obj_sdf_grad.shape}")
        else:
            self.obj_sdf_grad = None
                
        init_trimesh = trimesh.Trimesh(vertices=init_obj_verts, faces=obj_faces)
        mesh_exported_path = 'init_box_mesh.ply'
        init_trimesh.export('init_box_mesh.ply')
        
        # o3d_mesh = o3d.io.read_triangle_mesh(mesh_exported_path)
        # o3d_mesh.compute_vertex_normals()
        # init_verts_normals = o3d_mesh.vertex_normals
        # init_verts_normals = np.array(init_verts_normals, dtype=np.float32)
        # init_verts_normals = torch.from_numpy(init_verts_normals).float().to(self.device)
        
        
        
        rhand_verts = active_passive_sv_dict['rhand_verts']
        lhand_verts = active_passive_sv_dict['lhand_verts']
        hand_verts = np.concatenate([rhand_verts, lhand_verts], axis=1)
        obj_verts = torch.from_numpy(obj_verts).float().to(self.device)
        
        
        init_verts_normals = torch.zeros_like(obj_verts[0])
        
        hand_verts = torch.from_numpy(hand_verts).float().to(self.device)
        # rhand_verts = ## 
        # self.hand_faces, self.obj_faces # # rhand verts and lhand verts #
        hand_faces = active_passive_sv_dict['hand_faces']
        
        self.rhand_verts = torch.from_numpy(rhand_verts).float().to(self.device)
        self.hand_faces = torch.from_numpy(hand_faces).long().to(self.device)
        self.obj_faces = torch.tensor(obj_faces).long().to(self.device)
        self.obj_normals = {0: init_verts_normals}
        
        # rhand verts and hand faces and obj vertices # 
        dist_rhand_verts_to_obj_vertts = torch.sum(
            (self.rhand_verts.unsqueeze(2) - obj_verts.unsqueeze(1)) ** 2, dim=-1
        ) ## nn_hand_verts x nn_obj_verts ## 
        dist_rhand_verts_to_obj_vertts, _ = torch.min(dist_rhand_verts_to_obj_vertts, dim=-1) ### nn_hand_verts ##
        self.ts_to_contact_pts = {}
        thresold_dist = 0.001
        for ts in range(self.rhand_verts.size(0)):
            cur_contact_pts_threshold_indicator = dist_rhand_verts_to_obj_vertts[ts] <= thresold_dist
            if torch.sum(cur_contact_pts_threshold_indicator.float()).item() > 0.1:
                self.ts_to_contact_pts[ts] = self.rhand_verts[ts][cur_contact_pts_threshold_indicator]
        
        print(f"obj_verts: {obj_verts.size()}, hand_verts: {hand_verts.size()}, rhand_verts: {rhand_verts.shape}, lhand_verts: {lhand_verts.shape}, hand_faces: {self.hand_faces.size()}, obj_faces: {self.obj_faces.size()}")
        return obj_verts, hand_verts, self.obj_normals
    
    def calculate_collision_geometry_bounding_boxes(self, ):    
        init_passive_mesh = self.timestep_to_passive_mesh[0]
        maxx_init_passive_mesh, _ = torch.max(init_passive_mesh, dim=0) ## (3, )
        minn_init_passive_mesh, _ = torch.min(init_passive_mesh, dim=0) ## (3, )
        self.maxx_init_passive_mesh = maxx_init_passive_mesh
        self.minn_init_passive_mesh = minn_init_passive_mesh # 



    @staticmethod
    def integrate_state(states, delta, dt): ## 
        # needs pytorch >= 11.0 #
        
        pos = states[..., 0:3].contiguous().view(-1, 3)
        rot = states[..., 3:7].contiguous().view(-1, 4)
        batch_size, num_body, _ = states.shape

        ## ##
        vel = states[..., 7:10].contiguous().view(-1, 3)
        avel = states[..., 10:13].contiguous().view(-1, 3)

        root_rot = states[:, 0, 3:7].contiguous().view(-1,1,4)        
        delta = delta.contiguous().view(batch_size, -1, 3)
        
        true_delta = broadcast_quat_apply(root_rot, delta).contiguous().view(batch_size, -1, 6) # true delta #
        true_delta, true_ang_delta = true_delta[...,:3].contiguous().view(-1,3), true_delta[...,3:].contiguous().view(-1,3)
        
        next_vel = vel + true_delta
        next_avel = avel + true_ang_delta
        
        next_pos = pos + next_vel * dt
        next_rot = quat_integrate(rot, next_avel, dt)
        next_state = torch.cat([next_pos.contiguous().view(batch_size, num_body, 3),
                                    next_rot.contiguous().view(batch_size, num_body, 4),
                                    next_vel.contiguous().view(batch_size, num_body, 3),
                                    next_avel.contiguous().view(batch_size, num_body, 3)
                                    ], dim=-1)
        return next_state.contiguous().view(states.shape)
    
    
    
    def loss(self, pred, tar):
        pred_mano_trans, pred_mano_rot, pred_mano_states, pred_obj_trans,  pred_obj_rot = self.split_state(pred)
        tar_mano_trans, tar_mano_rot, tar_mano_states, tar_obj_trans, tar_obj_rot = self.split_state(tar)
        
        if torch.any(torch.isnan(tar_mano_trans))  or torch.any(torch.isnan(tar_mano_rot)) or torch.any(torch.isnan(tar_mano_states)) or torch.any(torch.isnan(tar_obj_rot)) or torch.any(torch.isnan(tar_obj_trans)):
            print(f"HAs nan values in target values ")
            
        
        if torch.any(torch.isnan(pred_mano_trans))  or torch.any(torch.isnan(pred_mano_rot)) or torch.any(torch.isnan(pred_mano_states)) or torch.any(torch.isnan(pred_obj_rot)) or torch.any(torch.isnan(pred_obj_trans)):
            print(f"HAs nan values in predicted values ")
        
        
        # if torch.any(torch.isnan(pred_mano_trans)):
        #     print(f"None value in pred_mano_trans!")
        # if torch.any(torch.isnan(pred_mano_rot)):
        #     print(f"None value in pred_mano_rot!")
        # if torch.any(torch.isnan(pred_mano_states)):
        #     print(f"None value in pred_mano_states!")
        # if torch.any(torch.isnan(pred_obj_rot)):
        #     print(f"None value in pred_obj_rot!")
        # if torch.any(torch.isnan(pred_obj_trans)):
        #     print(f"None value in pred_obj_trans!")
        
        
        weight_mano_trans, weight_mano_rot, weight_mano_states, weight_obj_rot, weight_obj_trans = self.weight['weight_mano_trans'], self.weight['weight_mano_rot'], self.weight['weight_mano_states'], self.weight['weight_obj_rot'], self.weight['weight_obj_trans']
        
        mano_trans_loss = weight_mano_trans * torch.mean(torch.norm(pred_mano_trans - tar_mano_trans, p=1, dim=-1))
        mano_rot_loss = weight_mano_rot * torch.mean(torch.norm(pred_mano_rot - tar_mano_rot, p=1, dim=-1))
        mano_states_loss = weight_mano_states * torch.mean(torch.norm(pred_mano_states - tar_mano_states, p=1, dim=-1))
        obj_rot_loss = weight_obj_rot * torch.mean(torch.norm(pred_obj_rot - tar_obj_rot, p=1, dim=-1))
        obj_trans_loss = weight_obj_trans * torch.mean(torch.norm(pred_obj_trans - tar_obj_trans, p=1, dim=-1))
        
        return mano_trans_loss, mano_rot_loss, mano_states_loss, obj_rot_loss, obj_trans_loss
        
        pred_pos, pred_rot, pred_vel, pred_avel = decompose_state(pred)
        tar_pos, tar_rot, tar_vel, tar_avel = decompose_state(tar)

        weight_pos, weight_vel, weight_rot, weight_avel = self.weight[
            "pos"], self.weight["vel"], self.weight["rot"], self.weight["avel"]

        batch_size = tar_pos.shape[0]

        pos_loss = weight_pos * \
            torch.mean(torch.norm(pred_pos - tar_pos, p=1, dim=-1))
        vel_loss = weight_vel * \
            torch.mean(torch.norm(pred_vel - tar_vel, p=1, dim=-1))
        avel_loss = weight_avel * \
            torch.mean(torch.norm(pred_avel - tar_avel, p=1, dim=-1))

        # special for rotation
        pred_rot_inv = quat_inv(pred_rot)
        tar_rot = DiffRotation.flip_quat_by_w(tar_rot.view(-1,4))
        pred_rot_inv = DiffRotation.flip_quat_by_w(pred_rot_inv.view(-1, 4))
        dot_pred_tar = torch.norm( DiffRotation.quat_to_rotvec( DiffRotation.quat_multiply(tar_rot,
                                              pred_rot_inv) ), p =2, dim=-1)
        rot_loss = weight_rot * \
            torch.mean(torch.abs(dot_pred_tar))
        return pos_loss, rot_loss, self.dt * vel_loss, self.dt * avel_loss
    
    def split_state(self, state): ## 
        mano_trans = state[:, : self.bullet_mano_num_trans_state]
        mano_rot = state[:, self.bullet_mano_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
        mano_states = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]
        
        obj_trans = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state]
        obj_rot = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state + self.obj_num_trans_state]
        return mano_trans, mano_rot, mano_states, obj_trans, obj_rot
    
    ## split
    def split_delta(self, delta):
        delta_mano_trans = delta[:, :self.bullet_mano_num_trans_state] ## determined by the rot state ##
        delta_mano_rot = delta[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state]
        delta_mano_states = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints] ## delta mano  states ##
        
        delta_obj_trans = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act]
        delta_obj_rot = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act + self.obj_num_trans_act]
        ## delta obj trans delta obj rot ## ## split detla and jget the rot act and the transact e
        return delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot #gedeta mano rot # 
    
    
    def split_action(self, action):
        act_trans = action[:, :self.bullet_mano_num_trans_state]
        act_rot = action[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
        act_states = action[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]

        ## delta obj trans delta obj rot ## ## split detla and jget the rot act and the transact e
        return act_trans, act_rot, act_states
    
        
    def integrate_states_hoi(self, nex_mano_trans, nex_mano_rot, nex_mano_states, nex_obj_rot, nex_obj_trans):
        mano_states = torch.cat( # get states #
            [nex_mano_trans, nex_mano_rot, nex_mano_states], dim=-1
        )
        obj_states = torch.cat(
            [nex_obj_trans, nex_obj_rot], dim=-1

        )
        states = torch.cat(
            [mano_states, obj_states], dim=-1
        )
        return states 
    
    def forward_kinematics(self, mano_trans, mano_rot, mano_states):
        mano_rot_np = mano_rot.detach().cpu().numpy()
        mano_rot_struct = R.from_rotvec(mano_rot_np)
        rot_mtx = mano_rot_struct.as_matrix()
        rot_mtx = torch.from_numpy(rot_mtx).float().to(self.device)
        # rot_mtx = quat_to_rotmat(mano_rot)
        self.mano_agent.set_init_states_target_value(mano_states)
        cur_visual_pts = self.mano_agent.get_init_state_visual_pts()
        cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + mano_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        return cur_visual_pts
        
    ## 
    def forward(self, state, action, **obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                # observation = state2ob(state) # 
                observation = state # normalize obs # 
                # observation1 = state2obfast(state) # 
            n_observation = self.normalize_obs(observation)
        
        n_observation_ori = n_observation.clone()
        
        ### observations ## state ## ## states and 
        mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state)
        
        if self.wana:
        
            act_trans, act_rot, act_states = self.split_action(action=action)
            
            ## ## mano trans ## ## ## cur_ts # wana ###
            # mano_trans = self.ts_to_mano_bullet_glb_trans[self.cur_ts]
            # mano_rot = self.ts_to_mano_bullet_glb_rot[self.cur_ts]
            # mano_states = self.ts_to_mano_bullet_states[self.cur_ts] # generative simulation ? #
            
            cur_mano_trans = mano_trans.squeeze(0)  + act_trans.squeeze(0) ## get the mnao trans here ##
            cur_mano_rot = mano_rot.squeeze(0)  + act_rot.squeeze(0) ## ge mnao rot ##
            cur_mano_states = mano_states.squeeze(0)  + act_states.squeeze(0) ## ## ## 
            
            # cur_mano_trans = cur_mano_trans # .squeeze(0)
            # cur_mano_rot = cur_mano_rot # .squeeze(0)
            # cur_mano_states = cur_mano_states.squeeze(0)
            # large models 
            
            
            ## get hte active mesh ## forward kinematics ##
            cur_visual_pts = self.forward_kinematics(cur_mano_trans, cur_mano_rot, cur_mano_states)
            self.timestep_to_active_mesh[self.cur_ts + 1] = cur_visual_pts # visual pts # 
            
            ## forward the active mesh and get the contact pairs set ##
            self.contact_pairs_set = self.ana_sim.forward2( input_pts_ts=self.cur_ts + 1, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
            
            # ## get the object data ## 
            ## total def -> total def ##
            cur_obj_trans = self.ana_sim.timestep_to_optimizable_total_def[self.cur_ts + 2] # (3,) #
            cur_obj_quat = self.ana_sim.timestep_to_optimizable_quaternion[self.cur_ts + 2] ## to the quat ##
            
            ## object state ##
            cur_obj_state = torch.cat(
                [cur_obj_trans, cur_obj_quat[[1,2,3,0]]], dim=-1 ## obj_state
            )
            ## mano state ## mano states ## 
            cur_mano_state = torch.cat( 
                [cur_mano_trans, cur_mano_rot, cur_mano_states], dim=-1 ## mano states
            )
            state = torch.cat(
                [cur_mano_state, cur_obj_state], dim=-1
            )
            observation = state
            n_observation = self.normalize_obs(observation).unsqueeze(0)
            
            
            obj_rot = cur_obj_quat.unsqueeze(0)
            obj_trans = cur_obj_trans.unsqueeze(0)
            
        
        
            if torch.any(torch.isnan(n_observation)):
                print(f"Has NaN value in n_observation!!")
                
            if torch.any(torch.isnan(action)):
                print(f"Has NaN value in action!!")
            
            ## observtion with actions ## ## observation with actions ## ## actions -> delta pd targets here ## use that as the delta targets to predict the delta chagn ebetween the current stepped observation and the next real observation ## ##  and 
            n_delta = self.mlp( torch.cat([n_observation_ori, n_observation, action], dim = -1) )
            
            
            mano_trans = cur_mano_trans.unsqueeze(0)
            mano_rot = cur_mano_rot.unsqueeze(0)
            mano_states =cur_mano_states.unsqueeze(0)
            
            obj_rot = obj_rot[:, [1, 2, 3, 0]]
            
        else:
            n_delta = self.mlp( torch.cat([n_observation, action], dim = -1) )


        # delta = n_delta
        
        if self.wnorm:
            delta = ptu.unnormalize(n_delta, self.delta_mean[:-1], self.delta_std[:-1]) 
        else:
            delta = n_delta
        
        if torch.any(torch.isnan(n_delta)):
            print(f"Has NaN value in n_delta!!")
        # if torch.any(torch.isnan(delta)):
        #     print(f"Has NaN value in delta!!")
        
        ## bullet_mano_num_joints, bullet_mano_finger_nu
        # ## mano trans mano root, ## m_joints, bullet_mano_num_rot_state, bullet_mano_num_trans_state ###
        ## obj_num_rot_state, obj_num_rot_act, obj_num_trans_state, obj_num_trans_act ##
        ## just predict them as delta states here ##
        #### delta mano rot ##
        # mano_trans, mano_rot, mano_states, obj_rot, obj_trans = self.split_state(state)
        
        # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_rot, delta_obj_trans = self.split_delta(delta)
        delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot = self.split_delta(delta)
        
        ## self.wana # 
        ### aggreagate them together ##
        nex_mano_trans = mano_trans + delta_mano_trans
        nex_mano_rot=  mano_rot + delta_mano_rot
        nex_mano_states = mano_states + delta_mano_states
        
        obj_rot  = obj_rot[:, [3,0,1,2]] # obj
        nex_obj_rot = obj_rot + update_quaternion_batched(delta_obj_rot, obj_rot)
        ## get obj rot ##
        nex_obj_rot = nex_obj_rot / torch.clamp(torch.norm(nex_obj_rot, dim=-1, keepdim=True), min=1e-5)
        
        nex_obj_rot = nex_obj_rot[:, [1, 2, 3, 0]]
        
        if torch.any(torch.isnan(nex_obj_rot)):
            print(f"Has NaN value in nex_obj_rot (after norm)!!")
        
        nex_obj_trans = obj_trans + delta_obj_trans
        
        ### get the nex t state ## ## states and the cne x oj rot 3# 
        state = self.integrate_states_hoi(
            nex_mano_trans, nex_mano_rot, nex_mano_states, nex_obj_rot, nex_obj_trans
        )
        
        return state
        
