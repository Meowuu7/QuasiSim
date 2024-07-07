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
import numpy as np
import os
import trimesh
# import open3d as o3d
from pyhocon import ConfigFactory
import ControlVAECore.Model.fields as fields
import ControlVAECore.Model.dyn_model_act_v2 as dyn_model_act_mano
from scipy.spatial.transform import Rotation as R
from ControlVAECore.Model.pointnet2_cls_msg import get_model_feature_extraction
import ControlVAECore.Model.pointnet2_utils as pointnet2_utils


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
    def __init__(self, ob_size, ac_size, delta_size, dt, statistics, env, **kargs) -> None:
        super(SimpleWorldModel, self).__init__()
        
        
        self.wana = kargs['wana']
        self.traj_opt = kargs['traj_opt']
        
        self.wnorm_obs = kargs['wnorm_obs']
        
        self.point_features = kargs['point_features']
        
        self.train_res_contact_forces = kargs['train_res_contact_forces'] # res contact forces --- True #
        self.reset_mano_states = kargs['reset_mano_states'] # 
        
        self.only_residual = kargs['only_residual'] if 'only_residual' in kargs else False
        
        self.use_mano_delta_states = kargs['use_mano_delta_states']
        self.pred_delta_offsets = kargs['pred_delta_offsets']
        self.use_contact_region_pts = kargs['use_contact_region_pts']
        
        self.use_contact_network = kargs['use_contact_network']
        
        self.use_optimizable_timecons = kargs['use_optimizable_timecons']
        
        if 'use_ambient_contact_net' in kargs:
            self.use_ambient_contact_net = kargs['use_ambient_contact_net']
        else:
            self.use_ambient_contact_net = False
        
        if self.wana:
            self.input_dim = ob_size + ob_size + ac_size
        else:
            self.input_dim = ob_size + ac_size
        
        if self.point_features:
            self.input_dim = 1024
        
        
        ##### mlp #####
        self.mlp = ptu.build_mlp( # imlement the point net as the encoder and the trainable layer ##
            input_dim = self.input_dim, # rotvec to 6d vector # delta size ## ac size ##
            output_dim= delta_size,
            hidden_layer_num=kargs['world_model_hidden_layer_num'],
            hidden_layer_size=kargs['world_model_hidden_layer_size'],
            activation=kargs['world_model_activation'],
            init_to_zero=True if self.wana else False
        )
        
        self.rot_mlp = ptu.build_mlp( # rot mlp #
            input_dim = self.input_dim, # rotvec to 6d vector # delta size ## ac size ##
            output_dim= delta_size,
            hidden_layer_num=kargs['world_model_hidden_layer_num'],
            hidden_layer_size=kargs['world_model_hidden_layer_size'],
            activation=kargs['world_model_activation'],
            init_to_zero=True if self.wana else False
        )
        
        ## observation size = ##
        # self.obs_feature_model 
        
        if self.point_features: # point features ##
            self.pnpp_feat_model = get_model_feature_extraction(normal_channel=False)
        
        self.point_scale_to_feat = 5.0
        
        # simple world model 
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
        conf_path = kargs['conf_path'] ## conf path ##
        self.nn_instances = 1
        self.conf_path = conf_path
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu') 
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        ## conf and parse string ##  ## CASE NAME ##
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        # self.base_exp_dir = self.conf['general.base_exp_dir']
        # self.base_exp_dir = self.base_exp_dir + f"_reverse_value_totviews_tag_{self.conf['general.tag']}"
        # os.makedirs(self.base_exp_dir, exist_ok=True) # 
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
            
        if 'penetration_proj_k_to_robot' in kargs:
            self.penetration_proj_k_to_robot = kargs['penetration_proj_k_to_robot']
            print(f"Using penetration_proj_k_to_robot from kargs: {self.penetration_proj_k_to_robot}")
            
        if 'model.penetration_proj_k_to_robot_friction' in self.conf:
            self.penetration_proj_k_to_robot_friction = float(self.conf['model.penetration_proj_k_to_robot_friction'])
        else:
            self.penetration_proj_k_to_robot_friction = self.penetration_proj_k_to_robot
        
        if 'penetration_proj_k_to_robot_friction' in kargs:
            self.penetration_proj_k_to_robot_friction = kargs['penetration_proj_k_to_robot_friction']
                
        
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
            
        if 'model.use_optimizable_params' in self.conf:  # use optimi
            self.use_optimizable_params = self.conf['model.use_optimizable_params']
        else:
            self.use_optimizable_params = False
             
        if 'model.penetration_determining' in self.conf:
            self.penetration_determining = self.conf['model.penetration_determining']
        else:
            self.penetration_determining = "sdf_of_canon"
        
        # if 'model.use_ambient_contact_net' in self.conf:
        #     self.use_ambient_contact_net = self.conf['model.use_ambient_contact_net']
        # else:
        #     self.use_ambient_contact_net = False
        
        
        ## gt_data_fn ## bullet hoi data ##
        # self.gt_data_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/bullet_hoi_data.npy" #
        self.gt_data_fn = kargs['sv_gt_refereces_fn']
        
        
        if "singlerighthand" in self.gt_data_fn:
            self.arctic_singlerighthand = True
        else:
            self.arctic_singlerighthand = False
        
        
        ## world model 
        ## 
        ## load the conf here ## ## bendingnetwork ## 
        self.ana_sim = fields.BendingNetworkActiveForceFieldForwardLagV18(**self.conf['model.bending_network'], nn_instances=self.nn_instances, minn_dist_threshold=self.minn_dist_threshold).to(self.device)
        
        ## timestep to passive mesh ##
        # self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh()
        
        # self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_v3()
        
        # load_active_passive_timestep_to_mesh_twohands_arctic
        self.timestep_to_passive_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_twohands_arctic()
        
        self.timestep_to_active_mesh = {}
        
        
        if self.conf['model.penetration_determining'] == "ball_primitives":
            self.center_verts, self.ball_r = self.get_ball_primitives()
                
        ## init ball primitives ##
        self.canon_passive_obj_verts = None
        ## add ana_sim obj_verts centers, normals, sdf space cneter, sdf space scale ##
        self.ana_sim.canon_passive_obj_verts = self.obj_verts
        self.ana_sim.canon_passive_obj_normals = self.obj_normals
        
        self.canon_passive_obj_verts = self.obj_verts
        
        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        self.obj_sdf_fn = self.conf['model.obj_sdf_fn'] 
        
        self.ana_sim.sdf_space_center = self.sdf_space_center
        self.ana_sim.sdf_space_scale = self.sdf_space_scale
        
        self.obj_sdf = np.load(self.obj_sdf_fn, allow_pickle=True)
        self.sdf_res = self.obj_sdf.shape[0]
        self.ana_sim.obj_sdf = self.obj_sdf
        self.ana_sim.sdf_res = self.sdf_res # get objsdf
        
        self.penetration_determining = self.conf['model.penetration_determining']
        if self.penetration_determining == "sdf_of_canon":
            self.ana_sim.penetration_determining = self.penetration_determining
        elif self.penetration_determining == "plane_primitives":
            print(f"setting the penetration determining method to plane_primitives with maxx_xyz: {self.maxx_init_passive_mesh}, minn_xyz: {self.minn_init_passive_mesh}")
            self.ana_sim.penetration_determining = "plane_primitives" #
        elif self.conf['model.penetration_determining'] == 'ball_primitives':
            print(f"Setting the penetration determining method to ball_primitives with ball_r: {self.ball_r}, center: {self.center_verts}")
            self.ana_sim.penetration_determining = "ball_primitives" #
            self.ana_sim.center_verts = self.center_verts
            self.ana_sim.ball_r = self.ball_r ## get the ball primitives here? ##
        else:
            raise NotImplementedError(f"penetration determining method {self.conf['model.penetration_determining']} not implemented")
                
        ### mnao urdf fn ###
        mano_hand_mean_meshcoll_urdf_fn = kargs['mano_urdf_fn']
        
        hand_model_name = mano_hand_mean_meshcoll_urdf_fn.split("/")[-1].split(".")[0]
        if "mano" in hand_model_name:
            self.hand_type = "mano"
        elif "shadow" in hand_model_name:
            self.hand_type = "shadow"
        else:
            raise NotImplementedError(f"Unknown hand type: {hand_model_name}")
        
        
        ##### load the dyn model #####
        if 'model.ckpt_fn' in self.conf and len(self.conf['model.ckpt_fn']) > 0:
            cur_ckpt_fn = self.conf['model.ckpt_fn']
            if not os.path.exists(cur_ckpt_fn):
                cur_ckpt_fn = "/data2/xueyi/ckpt_191000.pth"
            self.load_checkpoint_via_fn(cur_ckpt_fn)
        
        # model_path_mano = self.conf['model.sim_model_path'] ## model path mano ## ## 
        # model_path_mano = mano_hand_mean_meshcoll_urdf_fn
        # model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
        
        # model_path_mano = self.conf['model.sim_model_path']
        
        # export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"

        # export left_mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left/shadowhand_left_new_scaled.urdf"
        
        model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
        
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
        self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
        self.mult_after_center = 1.0
        self.cur_ts = 0
        self.counter = 0
        
        
        # left_model_path_mano = self.conf['model.sim_model_path_left']
        left_model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left/shadowhand_left_new_scaled_nnrot.urdf"
        mano_agent_left = dyn_model_act_mano.RobotAgent(xml_fn=left_model_path_mano)
        self.mano_agent_left  = mano_agent_left #

        
        
        if self.maxx_init_passive_mesh is None and self.minn_init_passive_mesh is None:
            self.calculate_collision_geometry_bounding_boxes()
        # self.calculate_collision_geometry_bounding_boxes()
        
        self.n_timesteps = self.conf['model.n_timesteps'] # number of simulation timestep ##
        for i_time_idx in range(self.n_timesteps):
            ## ana_sim ## ## ana_sim ##
            self.ana_sim.timestep_to_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_point_accs[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_total_def[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_angular_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_quaternion[i_time_idx] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).to(self.device)
            self.ana_sim.timestep_to_torque[i_time_idx] = torch.zeros((3,), dtype=torch.float32).to(self.device)
        ## total def ##
        # self.ana_sim.timestep_to_total_def[0] = 
        # self.ana_sim.timestep_to_quaternion[0] = self.tot_obj_quat[0]
        # self.ana_sim.timestep_to_optimizable_offset[0] = self.object_transl[0].detach()
        # self.ana_sim.timestep_to_optimizable_quaternion[0] = self.tot_obj_quat[0].detach()
        # self.ana_sim.timestep_to_optimizable_rot_mtx[0] = self.tot_reversed_obj_rot_mtx[0].detach()
        # self.ana_sim.timestep_to_optimizable_total_def[0] = self.object_transl[0].detach()
            
        self.calculate_obj_inertia() ## compute # computeobj inertia ##
        
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
        self.ana_sim.obj_sdf_grad = None ## set obj_sdf #
        self.ana_sim.use_sdf_as_contact_dist = self.use_sdf_as_contact_dist
        self.ana_sim.use_contact_dist_as_sdf = self.use_contact_dist_as_sdf
        self.ana_sim.minn_dist_threshold_robot_to_obj = self.minn_dist_threshold_robot_to_obj
        self.ana_sim.use_same_contact_spring_k = self.use_same_contact_spring_k
        self.ana_sim.I_ref = self.I_ref
        self.ana_sim.I_inv_ref = self.I_inv_ref
        self.ana_sim.obj_mass = self.obj_mass
        self.ana_sim.use_optimizable_timecons = self.use_optimizable_timecons
        
        #### Config the ana sim ####
        # self.maxx_init_passive_mesh, self.minn_init_passive_mesh #
        self.ana_sim.maxx_init_passive_mesh = self.maxx_init_passive_mesh
        self.ana_sim.minn_init_passive_mesh = self.minn_init_passive_mesh # ### init maximum passive meshe ### #
        self.ana_sim.train_residual_friction = self.train_residual_friction
        self.ana_sim.n_timesteps = self.n_timesteps
        self.ana_sim.train_res_contact_forces = self.train_res_contact_forces
        self.ana_sim.use_optimizable_params = self.use_optimizable_params ## optimizable params
        self.ana_sim.penetration_proj_k_to_robot_friction = self.penetration_proj_k_to_robot_friction
        self.ana_sim.train_residual_normal_forces = True
        print(f"setting only_residual to {self.only_residual}")
        self.ana_sim.only_residual = self.only_residual # only use only_residual #
        self.ana_sim.pred_delta_offsets = self.pred_delta_offsets ## use delta offset or not ##
        ### ## get the ana_sim ## ###
        self.ana_sim.use_contact_region_pts = self.use_contact_region_pts
        self.ana_sim.use_contact_network = self.use_contact_network
        self.ana_sim.use_ambient_contact_net = self.use_ambient_contact_net

        self.ts_to_transformed_obj_pts = {}

        # self.timestep to states; 
        # self.timesttep to states; #
        
        # ts_to_mano_glb_rot, ts_to_mano_glb_trans, ts_to_mano_states # 
        self.ts_to_mano_glb_rot = {}
        self.ts_to_mano_glb_trans = {}
        self.ts_to_mano_states = {}
        
        self.ts_to_transformed_obj_pts = {}
        
        ## 
        ## 200 ts points ## 
        ### set the initial contact pairs set ###
        self.contact_pairs_set = None
        
        
        
        
        
        
        print(f"sv_gt_refereces_fn: {self.gt_data_fn}")
        self.gt_data = np.load(self.gt_data_fn, allow_pickle=True).item()
        
        ## self.mano_glb_rot, self.mano_glb_trans, self.mano_states ##
        self.mano_glb_rot = self.gt_data['mano_glb_rot'] ## mano glb rot  # in the quaternion 
        self.mano_glb_rot = self.mano_glb_rot / np.clip(np.sqrt(np.sum(self.mano_glb_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        
        ## mano glb trans ##
        self.mano_glb_trans = self.gt_data['mano_glb_trans'] # 
        ## mano rot ## ## reset the state --->  ## ## 
        
        if self.hand_type == "mano":
            self.mano_states = self.gt_data['mano_states'][:, :]
        elif self.hand_type == "shadow":
            self.mano_states = self.gt_data['mano_states'][:, 2 :]
        # self.mano_states = self.gt_data['mano_states'][:, :]
        
        
        
        
        if self.reset_mano_states:
            # self.mano_states[:, 56 - 6] = 3.0
            # self.mano_states[:, 58 - 6] = 0.9
            if '30' in self.gt_data_fn: # gt data fn #
                print(f"Resetting mano states...")
                self.mano_states[:, 56 - 6] = 3.0
                self.mano_states[:, 58 - 6] = 0.9
            elif '20' in self.gt_data_fn:
                print(f"Resetting mano states...")
                self.mano_states[:, 56 - 6] = -0.2
                self.mano_states[:, 55 - 6] = 0.0
                self.mano_states[:, 57 - 6] = 0.0
            elif '25' in self.gt_data_fn: ## jin 
                self.mano_states[:, 58 - 6] = 0.0
                self.mano_states[:, 59 - 6] = 0.0
                self.mano_states[:, 60 - 6] = 0.0
        
        
        ### mano glb rot and glb states ###
        ##### mano glb rot and mano glb states #####
        # left_mano_glb_rot, left_mano_glb_trans, left_mano_states #
        self.left_mano_glb_rot = self.gt_data['left_mano_glb_rot']
        self.left_mano_glb_rot = self.left_mano_glb_rot / np.clip(np.sqrt(np.sum(self.left_mano_glb_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        self.left_mano_glb_trans = self.gt_data['left_mano_glb_trans']
        if self.hand_type == "mano":
            self.left_mano_states = self.gt_data['left_mano_states'][:, :self.bullet_mano_finger_num_joints]
        elif self.hand_type == "shadow":
            self.left_mano_states = self.gt_data['left_mano_states'][:, 2 : 2 + self.bullet_mano_finger_num_joints]
        
        
        
        # self.obj_rot = self.gt_data['obj_rot'] # in the quaternion #
        self.obj_rot = self.gt_data['optimized_quat']
        # ### from object rot vectors to object quaternions ###
        # if self.obj_rot.shape[1] == 3:
        #     tot_obj_rot_quat = []
        #     for i_fr in range(self.obj_rot.shape[0]):
                
        #         cur_obj_rot_vec = self.obj_rot[i_fr]
        #         cur_obj_rot_struct = R.from_rotvec(cur_obj_rot_vec)
                
        #         cur_obj_rot_mtx = cur_obj_rot_struct.as_matrix()
        #         cur_obj_rot_mtx = cur_obj_rot_mtx.T
        #         cur_obj_rot_struct = R.from_matrix(cur_obj_rot_mtx)
                
                
        #         cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
                
        #         # cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
        #         cur_obj_rot_quat = cur_obj_rot_quat[[3, 0, 1, 2]] ## as 3, 0, 1, 2
                
        #         if cur_obj_rot_quat[0] < 0:
        #             cur_obj_rot_quat = -1.0 * cur_obj_rot_quat
                
                
        #         cur_obj_rot_quat = env.obj_rot[i_fr]
        #         tot_obj_rot_quat.append(cur_obj_rot_quat) ## obj rot quat 
        #     tot_obj_rot_quat = np.stack(tot_obj_rot_quat, axis=0)
        #     self.obj_rot = tot_obj_rot_quat
        # else:
        #     self.obj_rot = self.obj_rot / np.clip(np.sqrt(np.sum(self.obj_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
            
        self.obj_rot = env.obj_rot.copy()
            
        for i_ts in range(self.obj_rot.shape[0]):
            print(f"ts: {i_ts}, obj_quat: {self.obj_rot[i_ts]}")
        # self.obj_trans = self.gt_data['obj_trans'] # # 
        self.obj_trans = env.obj_trans.copy()
        
        self.canon_obj_fn = kargs['canon_obj_fn']
        ## caonical obj fn ## 
        # /data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes/camera.obj # 
        self.canon_obj_mesh = trimesh.load(self.canon_obj_fn, force='mesh')
        self.canon_obj_vertices = torch.from_numpy(self.canon_obj_mesh.vertices).float().to(self.device)
        self.canon_obj_faces = torch.from_numpy(self.canon_obj_mesh.faces).long().to(self.device)
        
        # hand_visual_feats # 
        
        self.obj_verts_fps_idx = pointnet2_utils.farthest_point_sampling(self.canon_obj_vertices.unsqueeze(0), 1000)
        
        
        
        self.timestep_to_active_mesh = {}
        # 
        # # set the base rotation and translation ## # 
        # sampled_verts_idxes_fn = "/home/xueyi/diffsim/NeuS/robo_sampled_verts_idxes.npy"
        # sampled_verts_idxes = np.load(sampled_verts_idxes_fn)
        # sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().to(self.device)
        # self.sampled_verts_idxes = sampled_verts_idxes # sampled verts idxes
        
        self.mano_glb_rot = torch.from_numpy(self.mano_glb_rot).float().to(self.device)
        self.mano_glb_trans = torch.from_numpy(self.mano_glb_trans).float().to(self.device)
        self.mano_states = torch.from_numpy(self.mano_states).float().to(self.device)
        
        self.left_mano_glb_rot = torch.from_numpy(self.left_mano_glb_rot).float().to(self.device)
        self.left_mano_glb_trans = torch.from_numpy(self.left_mano_glb_trans).float().to(self.device)
        self.left_mano_states = torch.from_numpy(self.left_mano_states).float().to(self.device)
        
        self.obj_rot = torch.from_numpy(self.obj_rot).float().to(self.device)
        self.obj_trans = torch.from_numpy(self.obj_trans).float().to(self.device)
        
        ## mano glb rot ##
        self.bullet_mano_num_joints = self.mano_glb_rot.shape[-1] + self.mano_glb_trans.shape[-1] + self.mano_states.shape[-1] # 
        self.bullet_mano_finger_num_joints = self.mano_states.shape[-1]
        # self.ts_to_mano_bullet_states = {}
        # self.ts_to_mano_bullet_glb_rot = {}
        # self.ts_to_mano_bullet_glb_trans = {}
        self.dt = 0.0005
        
        self.frame_length = min(self.mano_glb_rot.shape[0], self.obj_rot.shape[0]) - 10
        # self.frame_length = 32
        self.done = np.zeros((self.frame_length,), dtype=np.int32)
        self.done[-1] = 1
        
        self.max_length = self.frame_length ## frmae lenght ##
        
        self.val = np.zeros(self.frame_length)    
        
    # # nromalizeization and the encoder ##
    
    
    def load_active_passive_timestep_to_mesh_twohands_arctic(self, ):
        # train_dyn_mano_model_states ## rhand 
        # sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
        # /data1/xueyi/GRAB_extracted_test/train/20_sv_dict_real_obj.obj # data1
        # import utils.utils as utils
        # from manopth.manolayer import ManoLayer
        
        self.start_idx = 20 # start_idx 
        self.window_size = 60
        start_idx = self.start_idx
        window_size = self.window_size
        
        # if 'model.kinematic_mano_gt_sv_fn' in self.conf:
        sv_fn = self.conf['model.kinematic_mano_gt_sv_fn']
        
        gt_data_folder = "/".join(sv_fn.split("/")[:-1]) ## 
        gt_data_fn_name = sv_fn.split("/")[-1].split(".")[0]
        arctic_processed_data_sv_folder = "/home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj"
        gt_data_canon_obj_sv_fn = f"{arctic_processed_data_sv_folder}/{gt_data_fn_name}_canon_obj.obj"
            
        print(f"Loading data from {sv_fn}")
        
        sv_dict = np.load(sv_fn, allow_pickle=True).item()
        
        
        object_global_orient = sv_dict["obj_rot"][start_idx: start_idx + window_size] # num_frames x 3 
        object_transl = sv_dict["obj_trans"][start_idx: start_idx + window_size] # num_frames x 3
        obj_pcs = sv_dict["verts.object"][start_idx: start_idx + window_size]
        
        # obj_pcs = sv_dict['object_pc']
        obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
        
        
        obj_vertex_normals = torch.zeros_like(obj_pcs)
        self.obj_normals = obj_vertex_normals
        
        # /data/xueyi/sim/arctic_processed_data/processed_seqs/s01/espressomachine_use_01.npy
        
        # obj_vertex_normals = sv_dict['obj_vertex_normals']
        # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
        # self.obj_normals = obj_vertex_normals
        
        # object_global_orient = sv_dict['object_global_orient'] # glboal orient 
        # object_transl = sv_dict['object_transl']
        
        
        obj_faces = sv_dict['f'][0]
        obj_faces = torch.from_numpy(obj_faces).long().cuda()
        self.obj_faces = obj_faces
        
        # obj_verts = sv_dict['verts.object']
        # minn_verts = np.min(obj_verts, axis=0)
        # maxx_verts = np.max(obj_verts, axis=0)
        # extent = maxx_verts - minn_verts
        # # center_ori = (maxx_verts + minn_verts) / 2
        # # scale_ori = np.sqrt(np.sum(extent ** 2))
        # obj_verts = torch.from_numpy(obj_verts).float().cuda()
        
        
        # obj_sv_path = "/data3/datasets/xueyi/arctic/arctic_data/data/meta/object_vtemplates"
        # obj_name = sv_fn.split("/")[-1].split("_")[0]
        # obj_mesh_fn = os.path.join(obj_sv_path, obj_name, "mesh.obj")
        # print(f"loading from {obj_mesh_fn}")
        # # template_obj_vs, template_obj_fs = trimesh.load(obj_mesh_fn, force='mesh')
        # template_obj_vs, template_obj_fs = utils.read_obj_file_ours(obj_mesh_fn, sub_one=True)
        
        
        
        
        # self.obj_verts = obj_verts
        init_obj_verts = obj_pcs[0]
        init_obj_rot_vec = object_global_orient[0]
        init_obj_transl = object_transl[0]
        
        init_obj_transl = torch.from_numpy(init_obj_transl).float().cuda()
        init_rot_struct = R.from_rotvec(init_obj_rot_vec)
        
        init_glb_rot_mtx = init_rot_struct.as_matrix()
        init_glb_rot_mtx = torch.from_numpy(init_glb_rot_mtx).float().cuda()
        # ## reverse the global rotation matrix ##
        init_glb_rot_mtx_reversed = init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        # nn_obj_verts x  3 ##
        ##### 
        # canon_obj_verts = torch.matmul(
        #     (init_obj_verts - init_obj_transl.unsqueeze(0)), init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        # )
        canon_obj_verts = torch.matmul(
            init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous()

        ## get canon obj verts ##
        
        canon_obj_verts = obj_pcs[0].clone()
        self.obj_verts = canon_obj_verts.clone()
        obj_verts = canon_obj_verts.clone()
        
        
        #### save canonical obj mesh ####
        print(f"canon_obj_verts: {canon_obj_verts.size()}, obj_faces: {obj_faces.size()}")
        # canon_obj_mesh = trimesh.Trimesh(vertices=canon_obj_verts.detach().cpu().numpy(), faces=obj_faces.detach().cpu().numpy())   
        # canon_obj_mesh.export(gt_data_canon_obj_sv_fn)
        # print(f"canonical obj exported to {gt_data_canon_obj_sv_fn}")
        #### save canonical obj mesh ####
        
        
        
        # # glb_rot xx obj_verts + obj_trans = cur_obj_verts 
        # canon_obj_verts = torch.matmul(
        #     init_glb_rot_mtx.transpose(1, 0).contiguous(), self.obj_verts[0] - init_obj_transl.unsqueeze(0)
        # )
        
        
        # obj_verts = torch.from_numpy(template_obj_vs).float().cuda()
        
        self.obj_verts = obj_verts.clone()
        
        mesh_scale = 0.8
        bbmin, _ = obj_verts.min(0) #
        bbmax, _ = obj_verts.max(0) #
        
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
        # vertices = (vertices - center) * scale # (vertices - center) * scale # #
        
        self.sdf_space_center = center
        self.sdf_space_scale = scale
        # sdf_sv_fn = self.sdf_sv_fn
        # self.obj_sdf = np.load(sdf_sv_fn, allow_pickle=True)
        # self.sdf_res = self.obj_sdf.shape[0]
        
        
        # init_obj_pcs = obj_pcs[0].detach().cpu().numpy()
        # init_glb_rot = object_global_orient[0]
        # init_glb_trans = object_transl[0]
        # init_glb_rot_struct = R.from_rotvec(init_glb_rot)
        # init_glb_rot_mtx = init_glb_rot_struct.as_matrix()
        # self.obj_verts = np.matmul((init_obj_pcs - init_glb_trans[None]), init_glb_rot_mtx.T)
        # obj_verts = self.obj_verts
        # minn_verts = np.min(obj_verts, axis=0)
        # maxx_verts = np.max(obj_verts, axis=0)
        # extent = maxx_verts - minn_verts
        # scale_cur = np.sqrt(np.sum(extent ** 2))
        
        # center_cur= (minn_verts + maxx_verts) / 2
        
        # obj_verts = (sv_dict['obj_verts'] - center_ori[None]) / scale_ori * scale_cur + center_cur[None]
        
        # obj_verts = torch.from_numpy(obj_verts).float().cuda()
        # self.obj_verts = obj_verts
        
        # sv_fn_obj_fn = sv_fn[:-4] + "_real_obj.obj"
        # scaled_obj = trimesh.Trimesh(vertices=self.obj_verts.detach().cpu().numpy(), faces=self.obj_faces.detach().cpu().numpy(), vertex_normals=self.obj_normals.detach().cpu().numpy())
        # scaled_obj.export(sv_fn_obj_fn)
        # print(f"Scaled obj saved to {scaled_obj}")
        
        
        
        tot_reversed_obj_rot_mtx = []
        tot_obj_quat = [] ## rotation matrix 
        
        re_transformed_obj_verts = []
        
        # transformed_obj_verts = []
        for i_fr in range(object_global_orient.shape[0]):
            cur_glb_rot = object_global_orient[i_fr]
            cur_transl = object_transl[i_fr]
            cur_transl = torch.from_numpy(cur_transl).float().cuda()
            cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
            cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
            cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
            
            # transformed verts ## canon_verts x R + t = transformed_verts #
            # (transformed_verts - t) x R^T = canon_verts #
            # cur_transformed_verts = torch.matmul(
            #     self.obj_verts, cur_glb_rot_mtx
            # ) + cur_transl.unsqueeze(0)
            
            cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
            tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
            
            cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
            cur_obj_quat = cur_glb_rot_struct.as_quat()
            cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
            cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
            tot_obj_quat.append(cur_obj_quat)
            
            cur_re_transformed_obj_verts = torch.matmul(
                cur_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), self.obj_verts.transpose(1, 0)
            ).transpose(1, 0) + cur_transl.unsqueeze(0)
            re_transformed_obj_verts.append(cur_re_transformed_obj_verts)
            
            # cur_re_transformed_obj_verts = torch.matmul(
            #     cur_glb_rot_mtx, self.obj_verts.transpose(1, 0)
            # ).transpose(1, 0) + cur_transl.unsqueeze(0)
            # re_transformed_obj_verts.append(cur_re_transformed_obj_verts)

            # center_obj_verts = torch.mean(self.obj_verts, dim=0, keepdim=True)
            # cur_transformed_verts = torch.matmul(
            #     (self.obj_verts - center_obj_verts), cur_glb_rot_mtx
            # ) + cur_transl.unsqueeze(0) + center_obj_verts
            
            # cur_transformed_verts = torch.matmul(
            #     cur_glb_rot_mtx, self.obj_verts.transpose(1, 0)
            # ).contiguous().transpose(1, 0).contiguous() + cur_transl.unsqueeze(0)
            # transformed_obj_verts.append(self.obj_)
        # transformed_obj_verts = torch.stack(transformed_obj_verts, dim=0)
        
        transformed_obj_verts = obj_pcs.clone()
        
        
        
        # rhand_verts = sv_dict['rhand_verts']
        # rhand_verts = torch.from_numpy(rhand_verts).float().cuda()
        # self.rhand_verts = rhand_verts ## rhand verts ## 
        
        
        self.mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
        # self.rgt_mano_layer = ManoLayer(
        #     flat_hand_mean=False,
        #     side='right',
        #     mano_root=self.mano_path,
        #     ncomps=45,
        #     use_pca=False,
        # ).cuda()
        
        # self.lft_mano_layer = ManoLayer(
        #     flat_hand_mean=False,
        #     side='left',
        #     mano_root=self.mano_path,
        #     ncomps=45,
        #     use_pca=False,
        # ).cuda()
        
        
        # ##### rhand parameters #####
        # rhand_global_orient_gt, rhand_pose_gt = sv_dict["rot_r"], sv_dict["pose_r"]
        # # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        # rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        # rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        # rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        # rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        # rhand_transl, rhand_betas = sv_dict["trans_r"], sv_dict["shape_r"][0]
        # rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        # rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        # rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        # rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float().cuda()
        # rhand_pose_var = torch.from_numpy(rhand_pose_gt).float().cuda()
        # rhand_beta_var = torch.from_numpy(rhand_betas).float().cuda()
        # rhand_transl_var = torch.from_numpy(rhand_transl).float().cuda()
        # # R.from_rotvec(obj_rot).as_matrix()
        # ##### rhand parameters #####
        
        
        # ##### lhand parameters #####
        # lhand_global_orient_gt, lhand_pose_gt = sv_dict["rot_l"], sv_dict["pose_l"]
        # # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        # lhand_global_orient_gt = lhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        # lhand_pose_gt = lhand_pose_gt[start_idx: start_idx + self.window_size]
        
        # lhand_global_orient_gt = lhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        # lhand_pose_gt = lhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        # lhand_transl, lhand_betas = sv_dict["trans_l"], sv_dict["shape_l"][0]
        # lhand_transl, lhand_betas = lhand_transl[start_idx: start_idx + self.window_size], lhand_betas
        
        # # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        # lhand_transl = lhand_transl.reshape(self.window_size, -1).astype(np.float32)
        # lhand_betas = lhand_betas.reshape(-1).astype(np.float32)
        
        # lhand_global_orient_var = torch.from_numpy(lhand_global_orient_gt).float().cuda()
        # lhand_pose_var = torch.from_numpy(lhand_pose_gt).float().cuda()
        # lhand_beta_var = torch.from_numpy(lhand_betas).float().cuda()
        # lhand_transl_var = torch.from_numpy(lhand_transl).float().cuda() # self.window_size x 3
        # # R.from_rotvec(obj_rot).as_matrix()
        # ##### lhand parameters #####
        
    
        
        # rhand_verts, rhand_joints = self.rgt_mano_layer(
        #     torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
        #     rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        # )
        # ### rhand_joints: for joints ###
        # rhand_verts = rhand_verts * 0.001
        # rhand_joints = rhand_joints * 0.001
        
        
        # lhand_verts, lhand_joints = self.lft_mano_layer(
        #     torch.cat([lhand_global_orient_var, lhand_pose_var], dim=-1),
        #     lhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), lhand_transl_var
        # )
        # ### rhand_joints: for joints ###
        # lhand_verts = lhand_verts * 0.001
        # lhand_joints = lhand_joints * 0.001
        
        
        ### lhand and the rhand ###
        # # rhand_verts, lhand_verts #
        # self.rhand_verts = rhand_verts
        # self.lhand_verts = lhand_verts 
        
        
        
        if '30_sv_dict' in sv_fn:
            bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
            obj_selected_verts = self.obj_verts[bbox_selected_verts_idxes]
        else:
            obj_selected_verts = self.obj_verts.clone()
        
        maxx_init_passive_mesh, _ = torch.max(obj_selected_verts, dim=0)
        minn_init_passive_mesh, _ = torch.min(obj_selected_verts, dim=0)
        self.maxx_init_passive_mesh = maxx_init_passive_mesh
        self.minn_init_passive_mesh = minn_init_passive_mesh
        
        
        init_obj_verts = obj_verts # [0] # cannnot rotate it at all # frictional forces in the pybullet? # 
        
        mesh_scale = 0.8
        bbmin, _ = init_obj_verts.min(0) #
        bbmax, _ = init_obj_verts.max(0) #
        print(f"bbmin: {bbmin}, bbmax: {bbmax}")
        center = (bbmin + bbmax) * 0.5
        
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
        # vertices = (vertices - center) * scale # (vertices - center) * scale # #
        
        self.sdf_space_center = center.detach().cpu().numpy()
        self.sdf_space_scale = scale.detach().cpu().numpy()
        # sdf_sv_fn = "/data/xueyi/diffsim/NeuS/init_box_mesh.npy"
        # if not os.path.exists(sdf_sv_fn):
        #     sdf_sv_fn = "/home/xueyi/diffsim/NeuS/init_box_mesh.npy"
        # self.obj_sdf = np.load(sdf_sv_fn, allow_pickle=True)
        # self.sdf_res = self.obj_sdf.shape[0]
        # print(f"obj_sdf loaded from {sdf_sv_fn} with shape {self.obj_sdf.shape}")
        
        # re_transformed_obj_verts = torch.stack(re_transformed_obj_verts, dim=0)
        # self.re_transformed_obj_verts = re_transformed_obj_verts
        
        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        tot_obj_quat = torch.stack(tot_obj_quat, dim=0) ## tot obj quat ##
        tot_reversed_obj_rot_mtx = torch.stack(tot_reversed_obj_rot_mtx, dim=0)
        self.tot_obj_quat = tot_obj_quat # obj quat #
        
        self.tot_obj_quat[0, 0] = 1.
        self.tot_obj_quat[0, 1] = 0.
        self.tot_obj_quat[0, 2] = 0.
        self.tot_obj_quat[0, 3] = 0.
        
        self.tot_reversed_obj_rot_mtx = tot_reversed_obj_rot_mtx
        
        self.tot_reversed_obj_rot_mtx[0] = torch.eye(3, dtype=torch.float32).cuda()
        ## should save self.object_global_orient and self.object_transl ##
        # object_global_orient, object_transl #
        self.object_global_orient = torch.from_numpy(object_global_orient).float().cuda()
        self.object_transl = torch.from_numpy(object_transl).float().cuda()
        
        self.object_transl[0, :] = self.object_transl[0, :] * 0.0
        return transformed_obj_verts, self.obj_normals
    
    
    
    def load_active_passive_timestep_to_mesh_v3(self, ):
        # train_dyn_mano_model_states ##
        sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
        if 'model.kinematic_mano_gt_sv_fn' in self.conf:
            sv_fn = self.conf['model.kinematic_mano_gt_sv_fn']
        print(f"sv_fn: {sv_fn}")
        sv_dict = np.load(sv_fn, allow_pickle=True).item() # get sv_dict ## 
        
        rhand_verts = sv_dict['rhand_verts']
        rhand_verts = torch.from_numpy(rhand_verts).float().cuda()
        self.rhand_verts = rhand_verts ## rhand verts ## 
        
        obj_faces = sv_dict['obj_faces']
        obj_faces = torch.from_numpy(obj_faces).long().cuda()
        self.obj_faces = obj_faces
        
        obj_verts = sv_dict['obj_verts']
        obj_verts = torch.from_numpy(obj_verts).float().cuda()
        self.obj_verts = obj_verts
        
        if 'model.scaled_obj_mesh_fn' in self.conf:
            self.scaled_obj_mesh_fn = self.conf['model.scaled_obj_mesh_fn']
            print(f"scaled_obj_mesh_fn: {self.scaled_obj_mesh_fn}")
            self.scaled_obj_mesh = trimesh.load(self.scaled_obj_mesh_fn, force='mesh')
            obj_verts = self.scaled_obj_mesh.vertices
            obj_faces = self.scaled_obj_mesh.faces
            obj_verts = torch.from_numpy(obj_verts).float().cuda()
            obj_faces = torch.from_numpy(obj_faces).long().cuda()
            self.obj_verts = obj_verts
            self.obj_faces = obj_faces
        
        if '30_sv_dict' in sv_fn:
            bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
            obj_selected_verts = self.obj_verts[bbox_selected_verts_idxes]
        else:
            obj_selected_verts = self.obj_verts.clone()
        
        
        # bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
        # obj_selected_verts = self.obj_verts[bbox_selected_verts_idxes]
        maxx_init_passive_mesh, _ = torch.max(obj_selected_verts, dim=0)
        minn_init_passive_mesh, _ = torch.min(obj_selected_verts, dim=0)
        self.maxx_init_passive_mesh = maxx_init_passive_mesh
        self.minn_init_passive_mesh = minn_init_passive_mesh
        
        
        init_obj_verts = obj_verts # [0]
        
        mesh_scale = 0.8
        bbmin, _ = init_obj_verts.min(0) #
        bbmax, _ = init_obj_verts.max(0) #
        print(f"bbmin: {bbmin}, bbmax: {bbmax}")
        center = (bbmin + bbmax) * 0.5
        
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
        # vertices = (vertices - center) * scale # (vertices - center) * scale # #
        
        self.sdf_space_center = center.detach().cpu().numpy()
        self.sdf_space_scale = scale.detach().cpu().numpy()


        obj_vertex_normals = sv_dict['obj_vertex_normals']
        obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
        self.obj_normals = obj_vertex_normals
        
        object_global_orient = sv_dict['object_global_orient'] # glboal orient # obj gl
        object_transl = sv_dict['object_transl']
        
        tot_reversed_obj_rot_mtx = []
        tot_obj_quat = [] ## rotation matrix 
        
        
        transformed_obj_verts = []
        for i_fr in range(object_global_orient.shape[0]):
            cur_glb_rot = object_global_orient[i_fr]
            cur_transl = object_transl[i_fr]
            cur_transl = torch.from_numpy(cur_transl).float().cuda()
            cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
            cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
            cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
            
            cur_transformed_verts = torch.matmul(
                self.obj_verts, cur_glb_rot_mtx
            ) + cur_transl.unsqueeze(0)
            
            cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
            tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
            
            cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
            cur_obj_quat = cur_glb_rot_struct.as_quat()
            cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
            cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
            tot_obj_quat.append(cur_obj_quat)

            # center_obj_verts = torch.mean(self.obj_verts, dim=0, keepdim=True)
            # cur_transformed_verts = torch.matmul(
            #     (self.obj_verts - center_obj_verts), cur_glb_rot_mtx
            # ) + cur_transl.unsqueeze(0) + center_obj_verts
            
            # cur_transformed_verts = torch.matmul(
            #     cur_glb_rot_mtx, self.obj_verts.transpose(1, 0)
            # ).contiguous().transpose(1, 0).contiguous() + cur_transl.unsqueeze(0)
            transformed_obj_verts.append(cur_transformed_verts)
        transformed_obj_verts = torch.stack(transformed_obj_verts, dim=0)
        
        
        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        tot_obj_quat = torch.stack(tot_obj_quat, dim=0)
        # reversed obj rot mtx #
        tot_reversed_obj_rot_mtx = torch.stack(tot_reversed_obj_rot_mtx, dim=0)
        self.tot_obj_quat = tot_obj_quat
        self.tot_reversed_obj_rot_mtx = tot_reversed_obj_rot_mtx
        
        ## should save self.object_global_orient and self.object_transl ##
        # object_global_orient, object_transl #
        self.object_global_orient = torch.from_numpy(object_global_orient).float().cuda()
        self.object_transl = torch.from_numpy(object_transl).float().cuda()
        return transformed_obj_verts, rhand_verts, self.obj_normals
    
    
    ## reset the state and step the action #
    def calculate_obj_inertia(self, ): ## ob inertia 
        ## clone  ## obj inertia ##
        if self.canon_passive_obj_verts is None:
            cur_init_passive_mesh_verts = self.timestep_to_passive_mesh[0].clone()
        else:
            cur_init_passive_mesh_verts = self.canon_passive_obj_verts.clone() ## canon passive obj verts ## # obj verts # 
        
        cur_init_passive_mesh_center = torch.mean(cur_init_passive_mesh_verts, dim=0)
        ## 
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
        
        
        
    # rreset the state and step the action #
    def reset(self, state, target_ts):
        ## no actions if using no analytical sim ##
        self.contact_pairs_set = None
        # reset to the tans rot trans rot # close 
        # mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state.unsqueeze(0))
        
        ### cur_ts ### target ts ## counter ##
        self.cur_ts = target_ts
        self.counter = target_ts
        if self.wana:
            init_mano_trans = self.mano_glb_trans[target_ts]
            init_mano_rot = self.mano_glb_rot[target_ts]
            init_mano_states = self.mano_states[target_ts]
            
            init_left_mano_trans = self.left_mano_glb_trans[target_ts]
            init_left_mano_rot = self.left_mano_glb_rot[target_ts]
            init_left_mano_states = self.left_mano_states[target_ts]
            
            # # trans, rot, states ### trans rot states ###
            if state is not None:
                init_obj_rot = state[-4:]
                init_obj_rot = init_obj_rot[[3, 0, 1, 2]]
                init_obj_trans = state[-7: -4]
            else:
                ## or the ##
                init_obj_rot = self.obj_rot[target_ts]
                init_obj_trans = self.obj_trans[target_ts]
            
            ## or the ##
            # init_obj_rot = self.obj_rot[target_ts]
            # init_obj_trans = self.obj_trans[target_ts]
        
            ### strategy 2 --- set the init states via the given state ### # 
            # # state should not be none #
            # # sampled mano states #
            # # trans | rot | mano_states | obj_trans | obj_rot #
            if state is not None:
                init_mano_trans = state[: 6]
                init_mano_rot = state[6: 6 + 6]
                init_mano_states = state[6 + 6: -7]
                
                init_mano_trans, init_left_mano_trans = init_mano_trans[:3], init_mano_trans[3:]
                init_mano_rot, init_left_mano_rot = init_mano_rot[:3], init_mano_rot[3:]
                init_mano_states, init_left_mano_states = init_mano_states[:init_mano_states.shape[0] // 2], init_mano_states[init_mano_states.shape[0] // 2: ]
                
                init_obj_trans = state[-7: -4]
                init_obj_rot = state[-4:]
                init_obj_rot = init_obj_rot[[3, 0, 1, 2]]
                init_mano_rot = self.euler_state_to_quat_wxyz(init_mano_rot) # w x y z rot quaternion #
                init_left_mano_rot = self.euler_state_to_quat_wxyz(init_left_mano_rot)
                # print(f"init_mano_rot: {init_mano_rot}")
                
            print(f"state: {state.size()}, init_mano_trans: {init_mano_trans.size()}, init_mano_rot: {init_mano_rot.size()}, init_mano_states: {init_mano_states.size()}")
            
            # #### set mano glb rot, trans, and states ####
            # # ts_to_mano_glb_rot, ts_to_mano_glb_trans, ts_to_mano_states # 
            # self.ts_to_mano_glb_rot[self.cur_ts] = init_mano_rot.detach()
            # self.ts_to_mano_glb_trans[self.cur_ts] = init_mano_trans.detach()
            # self.ts_to_mano_states[self.cur_ts] = init_mano_states.detach()
            ### strategy 2 --- set the init states via the given state ###
            
            ## total def init trans ##
            
            self.ana_sim.timestep_to_total_def[target_ts] = init_obj_trans.clone()
            self.ana_sim.timestep_to_optimizable_quaternion[target_ts] = init_obj_rot.clone()
            cur_obj_rot_mtx = fields.quaternion_to_matrix(init_obj_rot) # quat to matrix trans #
            self.ana_sim.timestep_to_optimizable_rot_mtx[target_ts] = cur_obj_rot_mtx.clone() # ## obj mtx ##
            # self.ana_sim.timestep_to_total_def[target_ts] = init_obj_trans.clone()
            self.ana_sim.timestep_to_quaternion[target_ts] = init_obj_rot.clone() # init obj rot vec and the rot mtx ## 
            self.ana_sim.timestep_to_angular_vel[max(0, target_ts - 1)] = torch.zeros_like(init_obj_trans)
            self.ana_sim.timestep_to_vel[max(0, target_ts - 1)] = torch.zeros_like(init_obj_trans) ## initjvel; angular 
            
            
            init_active_mesh_pts = self.forward_kinematics_v2(init_mano_trans, init_mano_rot, init_mano_states, self.mano_agent)
            init_active_left_mesh_pts = self.forward_kinematics_v2(init_left_mano_trans, init_left_mano_rot, init_left_mano_states, self.mano_agent_left)
            
            # ts_to_transformed_obj_pts, timestep_to_active_mesh #
            self.timestep_to_active_mesh[target_ts] = torch.cat([init_active_mesh_pts, init_active_left_mesh_pts], dim=0)
            
            ## 
            cur_transformed_passive_pts = self.forward_kinematics_obj(init_obj_rot.detach(), init_obj_trans.detach()).detach()
            
            self.ts_to_transformed_obj_pts[target_ts] = cur_transformed_passive_pts.squeeze(0)
            ## network issue -> perhaps still with some network issue ##
            ## the target_ts + 1's object rot and trans would be calculated here ##
            self.contact_pairs_set = self.ana_sim.forward2( input_pts_ts=target_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
    
    ## filed observation of the  observation ## ## filed observation ## ## filed observation ##
    def normalize_obs(self, observation):
        # return observation
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        if self.wnorm or self.wnorm_obs:
            observation = ptu.normalize(observation, self.obs_mean, self.obs_std)
        else:
            observation = observation
        return observation
    # 
    
    ''' Not in use currently '''
    ### laod active passive timestp 
    def load_active_passive_timestep_to_mesh(self,): ## # load active passive timestep to mesh ##
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


    def get_ball_primitives(self, ):
        # obj_verts # 
        ## get the maximum outer ball #
        maxx_verts, _ = torch.max(self.obj_verts, dim=0)
        minn_verts, _ = torch.min(self.obj_verts, dim=0) # 
        center_verts = (maxx_verts + minn_verts) / 2.
        extent_verts = (maxx_verts - minn_verts)
        ball_d = max(extent_verts[0].item(), max(extent_verts[1].item(), extent_verts[2].item()))
        ball_r = ball_d / 2.
        return center_verts, ball_r


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
        # pred_mano_trans, pred_mano_rot, pred_mano_states, pred_obj_trans,  pred_obj_rot = self.split_state(pred)
        # tar_mano_trans, tar_mano_rot, tar_mano_states, tar_obj_trans, tar_obj_rot = self.split_state(tar)

        pred_mano_trans, pred_left_mano_trans, pred_mano_rot, pred_left_mano_rot, pred_mano_states, pred_left_mano_states, pred_obj_trans, pred_obj_rot  = self.split_state(pred)
        tar_mano_trans, tar_left_mano_trans, tar_mano_rot, tar_left_mano_rot, tar_mano_states, tar_left_mano_states, tar_obj_trans, tar_obj_rot = self.split_state(tar)
        
        if torch.any(torch.isnan(tar_mano_trans))  or torch.any(torch.isnan(tar_mano_rot)) or torch.any(torch.isnan(tar_mano_states)) or torch.any(torch.isnan(tar_obj_rot)) or torch.any(torch.isnan(tar_obj_trans)):
            print(f"HAs nan values in target values ")
            
        
        if torch.any(torch.isnan(pred_mano_trans))  or torch.any(torch.isnan(pred_mano_rot)) or torch.any(torch.isnan(pred_mano_states)) or torch.any(torch.isnan(pred_obj_rot)) or torch.any(torch.isnan(pred_obj_trans)):
            print(f"HAs nan values in predicted values ")
        
        # print(f"pred_mano_trans: {pred_mano_trans.size()}, pred_mano_rot: {pred_mano_rot.size()}, pred_mano_states: {pred_mano_states.size()}, pred_obj_trans: {pred_obj_trans.size()}, pred_obj_rot: {pred_obj_rot.size()}")
        # print(f"tar_mano_trans: {tar_mano_trans.size()}, tar_mano_rot: {tar_mano_rot.size()}, tar_mano_states: {tar_mano_states.size()}, tar_obj_trans: {tar_obj_trans.size()}, tar_obj_rot: {tar_obj_rot.size()}")
        
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
        
        mano_trans_loss = weight_mano_trans * (torch.mean(torch.norm(pred_mano_trans - tar_mano_trans, p=1, dim=-1)) + torch.mean(torch.norm(pred_left_mano_trans - tar_left_mano_trans, p=1, dim=-1)))
        mano_rot_loss = weight_mano_rot * (torch.mean(torch.norm(pred_mano_rot - tar_mano_rot, p=1, dim=-1)) + torch.mean(torch.norm(pred_left_mano_rot - tar_left_mano_rot, p=1, dim=-1)))
        
        # print(f"pred: {pred.size()}, tar: {tar.size()}, pred_mano_states: {pred_mano_states.size()}, tar_mano_states: {tar_mano_states.size()}, pred_left_mano_states: {pred_left_mano_states.size()}, tar_left_mano_states: {tar_left_mano_states.size()}")
        
        mano_states_loss = weight_mano_states * (torch.mean(torch.norm(pred_mano_states - tar_mano_states, p=1, dim=-1)) + (torch.mean(torch.norm(pred_left_mano_states - tar_left_mano_states, p=1, dim=-1))))
        # obj_rot_loss = weight_obj_rot * torch.mean(torch.norm(pred_obj_rot - tar_obj_rot, p=1, dim=-1))
        
        cos_half_angle = torch.sum(pred_obj_rot * tar_obj_rot, dim=-1)
        # print(f"pred_obj_rot: {pred_obj_rot}, tar_obj_rot: {tar_obj_rot}, cos_half_angle: {cos_half_angle}")
        obj_rot_loss = weight_obj_rot * (1. - cos_half_angle) ## negative half angle of the angle ## # negative cos 
        # obj_trans_loss = weight_obj_trans * torch.norm(delta_obj_trans, p=1, dim=-1)
        # print(f"o")
    
        
        obj_trans_loss = weight_obj_trans * torch.mean(torch.norm(pred_obj_trans - tar_obj_trans, p=1, dim=-1))
        
        # print(f"pred_obj_rot: {pred_obj_rot}, tar_obj_rot: {tar_obj_rot},pred_obj_trans: {pred_obj_trans}, tar_obj_trans: {tar_obj_trans}")
        
        return mano_trans_loss, mano_rot_loss, mano_states_loss, obj_rot_loss, obj_trans_loss
        
        # pred_pos, pred_rot, pred_vel, pred_avel = decompose_state(pred)
        # tar_pos, tar_rot, tar_vel, tar_avel = decompose_state(tar)

        # weight_pos, weight_vel, weight_rot, weight_avel = self.weight[
        #     "pos"], self.weight["vel"], self.weight["rot"], self.weight["avel"]

        # batch_size = tar_pos.shape[0]

        # pos_loss = weight_pos * \
        #     torch.mean(torch.norm(pred_pos - tar_pos, p=1, dim=-1))
        # vel_loss = weight_vel * \
        #     torch.mean(torch.norm(pred_vel - tar_vel, p=1, dim=-1))
        # avel_loss = weight_avel * \
        #     torch.mean(torch.norm(pred_avel - tar_avel, p=1, dim=-1))

        # # special for rotation
        # pred_rot_inv = quat_inv(pred_rot)
        # tar_rot = DiffRotation.flip_quat_by_w(tar_rot.view(-1,4))
        # pred_rot_inv = DiffRotation.flip_quat_by_w(pred_rot_inv.view(-1, 4))
        # dot_pred_tar = torch.norm( DiffRotation.quat_to_rotvec( DiffRotation.quat_multiply(tar_rot,
        #                                       pred_rot_inv) ), p =2, dim=-1)
        # rot_loss = weight_rot * \
        #     torch.mean(torch.abs(dot_pred_tar))
        # return pos_loss, rot_loss, self.dt * vel_loss, self.dt * avel_loss
    
    def split_state(self, state): ## 
        # mano_trans = state[:, : self.bullet_mano_num_trans_state]
        # mano_rot = state[:, self.bullet_mano_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
        # mano_states = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]
        
        # obj_trans = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state]
        # obj_rot = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state + self.obj_num_trans_state]
        
        trans_dim = 3
        rot_dim = 3
        
        tot_mano_trans = state[:, : trans_dim * 2]
        tot_mano_rot = state[:, trans_dim * 2: (trans_dim + rot_dim) * 2]
        tot_mano_states = state[:, (trans_dim + rot_dim) * 2: -7]
        
        mano_trans, left_mano_trans = tot_mano_trans[:, :trans_dim], tot_mano_trans[:, trans_dim:]
        mano_rot, left_mano_rot = tot_mano_rot[:,:rot_dim], tot_mano_rot[:,rot_dim: ]
        mano_states, left_mano_states = tot_mano_states[:,: tot_mano_states.shape[-1] // 2], tot_mano_states[:,tot_mano_states.shape[-1] // 2: ]
        
        
        # mano_trans = state[:, :3]
        # mano_rot = state[:, 3:6]
        # mano_states = state[:, 6:-7]
        obj_trans = state[:, -7:-4]
        obj_rot = state[:, -4:]
        
        return mano_trans, left_mano_trans, mano_rot, left_mano_rot, mano_states, left_mano_states, obj_trans, obj_rot    
        
        return mano_trans, mano_rot, mano_states,  obj_trans, obj_rot
    
    
    
    
    ## split
    def split_delta(self, delta):
        
        delta_mano_trans = delta[:, :3]
        delta_mano_rot = delta[:, 3:6]
        delta_mano_states = delta[:, 6:-6]
        delta_obj_trans = delta[:, -6:-3]
        delta_obj_rot = delta[:, -3:]
        
        
        # 
        # delta_mano_trans = delta[:, :self.bullet_mano_num_trans_state] ## determined by the rot state ##
        # delta_mano_rot = delta[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state]
        # delta_mano_states = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints] ## delta mano  states ##
        
        # delta_obj_trans = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act]
        # delta_obj_rot = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act + self.obj_num_trans_act]
        # ## delta obj trans delta obj rot ## ## split detla and jget the rot act and the transact e
        
        # print(f"delta_mano_trans: {delta_mano_trans.size()}, delta_mano_rot: {delta_mano_rot.size()}, delta_mano_states: {delta_mano_states.size()}, delta_obj_trans: {delta_obj_trans.size()}, delta_obj_rot: {delta_obj_rot.size()}")
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
    
    def forward_kinematics_v2(self,mano_trans, mano_rot, mano_states, dyn_model=None):
        # mano_rot_np = mano_rot.detach().cpu().numpy()[[1, 2, 3, 0]] ## xyz w; obj rot loss = 15+? #
        # mano_rot_struct = R.from_quat(mano_rot_np)
        # rot_mtx = mano_rot_struct.as_matrix()
        # rot_mtx = torch.from_numpy(rot_mtx).float().to(self.device)
        
        dyn_model = dyn_model if dyn_model is not None else self.mano_agent
        
        rot_mtx = fields.quaternion_to_matrix(mano_rot)
        ## some problems ##
        dyn_model.set_init_states_target_value(mano_states)
        cur_visual_pts = dyn_model.get_init_state_visual_pts()
        cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + mano_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        return cur_visual_pts
        
    def quat_wxyz_to_euler_state(self, quat_wxyz):
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        quat_xyzw = quat_xyzw.detach().cpu().numpy()
        quat_xyzw_struct = R.from_quat(quat_xyzw)
        euler = quat_xyzw_struct.as_euler('zyx', degrees=False)
        euler = [euler[2], euler[1], euler[0]]
        euler = np.array(euler, dtype=np.float32)
        euler = torch.from_numpy(euler).float().to(self.device)
        # euler = torch.from_numpy(euler).float().to(self.device)
        return euler
    
    def euler_state_to_quat_wxyz(self, euler_state):
        euler_state_np = euler_state.detach().cpu().numpy()
        euler_state_np = euler_state_np[[2, 1, 0]]
        euler_rot_struct = R.from_euler('zyx', euler_state_np, degrees=False)
        quat_xyzw = euler_rot_struct.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        quat_wxyz = torch.tensor(quat_wxyz, dtype=torch.float32).cuda()
        return quat_wxyz
        
        
    # def euler_state_to-q
    
    # encode the observation ##
    
    ## # encode
    ## # cur_visual_pts # 
    
    def forward_kinematics_obj(self, obj_rot, obj_trans):
        ## object quaternion 
        if len(obj_rot.size()) == 1:
            obj_rot = obj_rot.unsqueeze(0)
        obj_rot_mtx = []
        for cur_bz in range(obj_rot.size(0)):
            cur_obj_quat = obj_rot[cur_bz]
            obj_rot_mtx.append(fields.quaternion_to_matrix(cur_obj_quat))
        obj_rot_mtx = torch.stack(obj_rot_mtx, dim=0)
        # obj_rot_mtx = fields.quaternion_to_matrix(obj_rot)
        canon_obj_verts = self.canon_obj_vertices
        # ojb_rot_mtx: 1
        # cnon_obj_verts: nn_verts x 3 # ## nn_verts x 3 ## --- bsz x 3 x 3 xxxxx 1 x 3 x inn_verts
        transformed_obj_verts = torch.matmul(
            obj_rot_mtx, canon_obj_verts.contiguous().transpose(1, 0).contiguous().unsqueeze(0) ## bszx 3 x nn_verts
        ).contiguous().transpose(2, 1).contiguous() # 
        return transformed_obj_verts ## nn_bsz x nn_verts x 3 # ## # nn_verts x 3 ## obj ##
    
    def get_point_features(self, hand_visual_feats, obj_rot, obj_trans):
        obj_transformed_verts = self.forward_kinematics_obj(obj_rot, obj_trans)
        obj_transformed_verts = obj_transformed_verts[:, self.obj_verts_fps_idx, :]
        
        # # 
        if len(hand_visual_feats.size()) == 2:
            hand_visual_feats = hand_visual_feats.unsqueeze(0)
        
        tot_visual_feats = torch.cat([hand_visual_feats, obj_transformed_verts], dim=1) #
        tot_visual_feats = tot_visual_feats * self.point_scale_to_feat
        tot_visual_feats = tot_visual_feats.contiguous().transpose(2, 1).contiguous()
        visual_feats = self.pnpp_feat_model(tot_visual_feats)
        return visual_feats

    
    # or jsut normalize the observations? # ## observations ##
    ## forward ## 
    def forward(self, state, action, **obs_info): # get the next state objs # # a sharp point here ## ## to imitate the kientmaics retargeted result ##
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                # observation = state2ob(state) # 
                observation = state # normalize obs # 
                # observation1 = state2obfast(state) # 
            n_observation = self.normalize_obs(observation) # normalized obs #
        
        # n_observation_ori = n_observation.clone()
        
        ### observations ## state ## ## states ##
        # #
        # mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state)
        mano_trans, left_mano_trans, mano_rot, left_mano_rot, mano_states, left_mano_states, obj_trans, obj_rot = self.split_state(state)
        
        if self.wana:
            if self.traj_opt:
                rot_dim = 4
                trans_dim = 3
                
                tot_act_trans = action[..., :trans_dim * 2]
                tot_act_rot = action[..., trans_dim * 2: (trans_dim + rot_dim) * 2]
                tot_act_states = action[..., (trans_dim + rot_dim) * 2: ]
                
                act_trans, left_act_trans = tot_act_trans[..., :3], tot_act_trans[..., 3:]
                act_rot, left_act_rot = tot_act_rot[..., :rot_dim], tot_act_rot[..., rot_dim:]
                act_states, left_act_states = tot_act_states[..., : tot_act_states.size(-1) // 2], tot_act_states[..., tot_act_states.size(-1) // 2: ]
                
                # act_trans = action[..., :3]
                # act_rot = action[..., 3:7]
                # act_states = action[..., 7:]
                
                ## 
                cur_mano_trans = act_trans.squeeze(0)      
                cur_mano_rot = act_rot.squeeze(0)          
                cur_mano_states = act_states.squeeze(0)   
                
                cur_left_mano_trans = left_act_trans.squeeze(0) 
                cur_left_mano_rot = left_act_rot.squeeze(0)
                cur_left_mano_states = left_act_states.squeeze(0)
                
                
                
                # if self.use_mano_delta_states: # split
                #     cur_mano_states = cur_mano_states + self.ts_to_mano_states[self.cur_ts]
                
                #### set mano glb rot, trans, and states ####
                # set mano glb rot , trans, and states ##
                # ts_to_mano_glb_rot, ts_to_mano_glb_trans, ts_to_mano_states # 
                self.ts_to_mano_glb_rot[self.cur_ts + 1] = cur_mano_rot.detach()
                self.ts_to_mano_glb_trans[self.cur_ts + 1] = cur_mano_trans.detach()
                self.ts_to_mano_states[self.cur_ts + 1] = cur_mano_states.detach()
                ## split states as the initial states and use actions as delta states for the subsequent timesteps ##
                ## traj collection -> ## state to A ##
                
                # ## mano states ## #
                # trajectory; trajectory #
                cur_visual_pts = self.forward_kinematics_v2(cur_mano_trans, cur_mano_rot, cur_mano_states)
                
                cur_left_visual_pts = self.forward_kinematics_v2(cur_left_mano_trans, cur_left_mano_rot, cur_left_mano_states, self.mano_agent_left)
                
                cur_mano_rot = self.quat_wxyz_to_euler_state(cur_mano_rot) # cur mano rot ##
                cur_mano_states = cur_mano_states[:mano_states.size(-1)] # mano states # 
                
                cur_left_mano_rot = self.quat_wxyz_to_euler_state(cur_left_mano_rot)
                cur_left_mano_states = cur_left_mano_states[:left_mano_states.size(-1)]
                
                
                cur_visual_pts = torch.cat(
                    [cur_visual_pts, cur_left_visual_pts], dim=0
                )
                
                ###### cur_mano_trans, cur_mano_rot, cur_mano_states ######
                cur_mano_trans = torch.cat(
                    [cur_mano_trans, cur_left_mano_trans], dim=-1
                )
                cur_mano_rot = torch.cat(
                    [cur_mano_rot, cur_left_mano_rot], dim=-1
                )
                cur_mano_states = torch.cat(
                    [cur_mano_states, cur_left_mano_states], dim=-1
                )
                
            else:
                act_trans, act_rot, act_states = self.split_action(action=action)
                
                ## ## mano trans ##
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
                
                ## cur mano states ##
                ## get hte active mesh ## forward kinematics ##
                cur_visual_pts = self.forward_kinematics(cur_mano_trans, cur_mano_rot, cur_mano_states)
            
            self.timestep_to_active_mesh[self.cur_ts + 1] = cur_visual_pts # visual pts # 
            
            
            ## use the analytical sim as the skeleton for the model ##
            ## forward the active mesh and get the contact pairs set ##
            self.contact_pairs_set = self.ana_sim.forward2( input_pts_ts=self.cur_ts + 1, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
            
            # if self.train_res_contact_forces
            
            
            # transformed_obj_ts_idx = self.cur_ts + 1
            transformed_obj_ts_idx = self.cur_ts + 2
            # ## get the object data ## 
            ## total def -> total def ## ## optimizatle ##
            # cur_obj_trans = self.ana_sim.timestep_to_optimizable_total_def[self.cur_ts + 2] # (3,) #
            # cur_obj_quat = self.ana_sim.timestep_to_optimizable_quaternion[self.cur_ts + 2] ## to the quat ## ## to quat ##
            
            cur_obj_trans = self.ana_sim.timestep_to_optimizable_total_def[transformed_obj_ts_idx] # (3,) # ## jts + 1 -> tras and the rob
            cur_obj_quat = self.ana_sim.timestep_to_optimizable_quaternion[transformed_obj_ts_idx] ## to the quat ## ## to quat ##
            # print(f"cur_ts: {self.cur_ts}, obj_quat: {cur_obj_quat}")
            ## object state ##
            cur_obj_state = torch.cat(
                [cur_obj_trans, cur_obj_quat[[1,2,3,0]]], dim=-1 ## obj_state
            )
            ## mano state ## mano states ## 
            cur_mano_state = torch.cat(  ## cur mano states ##
                [cur_mano_trans, cur_mano_rot, cur_mano_states], dim=-1 ## mano states
            )
            state = torch.cat( # 
                [cur_mano_state, cur_obj_state], dim=-1
            )
            observation = state
            n_observation = self.normalize_obs(observation) # .unsqueeze(0)
            
            ## obj quat and obj trans ##
            obj_rot = cur_obj_quat.unsqueeze(0)
            obj_trans = cur_obj_trans.unsqueeze(0)
            new_obj_rot = obj_rot[:, [1, 2, 3, 0]] # 
        
            ## get the point features ##
            if torch.any(torch.isnan(n_observation)):
                print(f"Has NaN value in n_observation!!")
                
            if torch.any(torch.isnan(action)):
                print(f"Has NaN value in action!!")
            
            
            # manipulation ##  # use the blender for 
            ## ts to obj transfromed jpts #### transformed
            transformed_obj_pts = self.forward_kinematics_obj(cur_obj_quat, cur_obj_trans).detach()
            
            ### to the transformed obj pts ###
            self.ts_to_transformed_obj_pts[self.cur_ts + 1] = transformed_obj_pts.squeeze(0)
            
            
            
            observation_ana = n_observation.clone()
            self.observation_ana = observation_ana
            
            mano_trans = cur_mano_trans.unsqueeze(0)
            mano_rot = cur_mano_rot.unsqueeze(0)
            mano_states =cur_mano_states.unsqueeze(0)
            
            
            
        else:
            n_delta = self.mlp( torch.cat([n_observation, action], dim = -1) )


        # delta = n_delta
        
        # if self.wnorm:
        #     delta = ptu.unnormalize(n_delta, self.delta_mean[:-1], self.delta_std[:-1]) 
        # else:
        #     delta = n_delta
        
        # if torch.any(torch.isnan(n_delta)):
        #     print(f"Has NaN value in n_delta!!")
        
        # if torch.any(torch.isnan(delta)):
        #     print(f"Has NaN value in delta!!")
        
        ## bullet_mano_num_joints, bullet_mano_finger_nu
        # ## mano trans mano root, ## m_joints, bullet_mano_num_rot_state, bullet_mano_num_trans_state ###
        ## obj_num_rot_state, obj_num_rot_act, obj_num_trans_state, obj_num_trans_act ##
        ## just predict them as delta states here ##
        #### delta mano rot ##
        # mano_trans, mano_rot, mano_states, obj_rot, obj_trans = self.split_state(state)
        
        # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_rot, delta_obj_trans = self.split_delta(delta)
        # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot = self.split_delta(delta)
        
        # print(f'[train wm] delta_obj_rot: {delta_obj_rot}, delta_obj_trans: {delta_obj_trans}')
        ## self.wana # 
        ### aggreagate them together ##
        # nex_mano_trans = mano_trans + delta_mano_trans
        # nex_mano_rot=  mano_rot + delta_mano_rot
        
        # if delta_mano_states.size(-1) < mano_states.size(-1):
        #     delta_mano_states = torch.cat([delta_mano_states, torch.zeros_like(mano_states[..., delta_mano_states.size(-1):])], dim=-1)
        
        # nex_mano_states = mano_states + delta_mano_states
        
        nex_mano_states = mano_states
        nex_mano_rot = mano_rot
        nex_mano_trans = mano_trans
        
        ## t
        if self.train_res_contact_forces:
            nex_obj_rot = new_obj_rot # xyz w # new obj rot ## 
            nex_obj_trans = obj_trans
        else:
            obj_rot  = obj_rot[:, [3,0,1,2]] # obj
            # print(f"delta_obj_rot: {delta_obj_rot.size()}, obj_rot: {obj_rot.size()}")
            nex_obj_rot = obj_rot + update_quaternion_batched(delta_obj_rot, obj_rot)
            ## get obj rot ##
            nex_obj_rot = nex_obj_rot / torch.clamp(torch.norm(nex_obj_rot, dim=-1, keepdim=True), min=1e-5)
            nex_obj_rot = nex_obj_rot[:, [1, 2, 3, 0]]
            if torch.any(torch.isnan(nex_obj_rot)):
                print(f"Has NaN value in nex_obj_rot (after norm)!!")
            nex_obj_trans = obj_trans + delta_obj_trans
        
        # 
        
        # print(f"cur_ts: {self.cur_ts}, nex_obj_rot: {nex_obj_rot[0].tolist()}, nex_obj_trans: {nex_obj_trans[0].tolist()}")
        ### get the nex t state ## ## states and the cne x oj rot 3# 
        state = self.integrate_states_hoi(
            nex_mano_trans, nex_mano_rot, nex_mano_states, nex_obj_rot, nex_obj_trans
        )
        
        
        ## cur ts ##
        self.cur_ts = self.cur_ts + 1
        self.counter = self.counter + 1
        
        
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        return state
        
