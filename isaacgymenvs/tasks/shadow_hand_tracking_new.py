# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import os.path as osp
import random

from pyparsing import And
import torch

from utils.torch_jit_utils import *
# from utils.data_info import plane2euler
# from tasks.hand_base.base_task import BaseTask
from isaacgymenvs.tasks.vec_task import VecTask as BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

from scipy.spatial.transform import Rotation as R




class ShadowHandTracking(BaseTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, exp_logging_dir=None): # 

        self.cfg = cfg
        # self.test = self.cfg['task']['test']
        # self.sim_params = sim_params
        # self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.test = self.cfg['env']['test']
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        
        
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.w_obj_ornt = self.cfg["env"]["w_obj_ornt"]
        self.w_obj_vels = self.cfg["env"]["w_obj_vels"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)
        self.rl_device = rl_device
        self.exp_logging_dir = exp_logging_dir
        
        if self.exp_logging_dir is None:
            self.exp_logging_dir = self.cfg['env']['exp_logging_dir']
        
        self.object_name = self.cfg["env"]["object_name"]

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.mocap_sv_info_fn = self.cfg["env"]["mocap_sv_info_fn"]
        
        self.use_kinematics_bias = self.cfg['env']['use_kinematics_bias'] ## env and the kinematics bias #
        self.kinematics_only = self.cfg['env']['kinematics_only']
        self.use_kinematics_bias_wdelta = self.cfg['env']['use_kinematics_bias_wdelta']
        # get the control frequency #
        
        self.use_canonical_state = self.cfg['env']['use_canonical_state']
        self.separate_stages = self.cfg['env']['separate_stages']
        self.use_unified_canonical_state = self.cfg['env']['use_unified_canonical_state']
        
        self.rigid_obj_density = self.cfg['env']['rigid_obj_density']
        # self.density = self.cfg["env"]["rigid_obj_density"]
        self.use_fingertips = self.cfg["env"]["use_fingertips"]
        self.glb_trans_vel_scale = self.cfg["env"]["glb_trans_vel_scale"]
        self.glb_rot_vel_scale = self.cfg["env"]["glb_rot_vel_scale"] # get the rot vel scale #
        self.tight_obs = self.cfg["env"]["tight_obs"]
        # hand_pose_guidance_glb_trans_coef, hand_pose_guidance_glb_rot_coef, hand_pose_guidance_fingerpose_coef
        self.hand_pose_guidance_glb_trans_coef = self.cfg["env"]["hand_pose_guidance_glb_trans_coef"]
        self.hand_pose_guidance_glb_rot_coef = self.cfg["env"]["hand_pose_guidance_glb_rot_coef"]
        self.hand_pose_guidance_fingerpose_coef = self.cfg["env"]["hand_pose_guidance_fingerpose_coef"]
        self.rew_finger_obj_dist_coef = self.cfg["env"]["rew_finger_obj_dist_coef"]
        self.rew_delta_hand_pose_coef = self.cfg["env"]["rew_delta_hand_pose_coef"]
        


        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        self.nn_hand_dof = 22 + 6
        
        
        
        self.shadow_hand_dof_speed_scale_list = [self.glb_trans_vel_scale] * 3 + [self.glb_rot_vel_scale] * 3 + [self.shadow_hand_dof_speed_scale] * (self.nn_hand_dof - 6)
        self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, device=self.rl_device, dtype=torch.float) # speed scale tsr #
        
        
        self.up_axis = 'z'
        # 'palm': 'palm_link',
        # 'thumb': 'link_15_tip',
        # 'index': 'link_3_tip', # link 3 tip
        # 'middle': 'link_7_tip', # link 3 tip
        # 'ring': 'link_11_tip' # link 3 tip
        # self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        # if self.use_fingertips:
        #     self.fingertips = ["link_15_tip", "link_3_tip", "link_7_tip", "link_11_tip"]
        # else:
        #     self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        # self.hand_center = ["palm_link"]
        # self.num_fingertips = len(self.fingertips) 
        
        
        self._load_mocap_info()
        
        self.max_episode_length = min(self.max_episode_length, self.hand_qs.shape[0] - 1)
        
        
        
        
        self.use_vel_obs = False
        self.fingertip_obs = True
        # self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        # if self.asymmetric_obs:
        #     num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        
        # if self.tight_obs: # with next state references and with the current delta targets #
        #     self.cfg['env']['numObservations'] -= 7
        
        # print(f"obs_type: {self.obs_type}, num_obs: {self.cfg['env']['numObservations']}")
        
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = self.nn_hand_dof #  
        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # super().__init__(cfg=self.cfg, enable_camera_sensors=False)
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        print(f"num_shadow_hand_dofs: {self.num_shadow_hand_dofs}")
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        #     dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        #     self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
        #                                             self.num_shadow_hand_dofs + self.num_object_dofs)
        #     self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        # self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        
        # dof_state_tensor[:, : self.num_shadow_hand_dofs, 0] = self.shadow_hand_default_dof_pos
        
        
        print(f"[Debug] shadow_hand_default_dof_pos: {self.shadow_hand_default_dof_pos}")
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).view(-1).contiguous()
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print(f"[Debug] num_dofs: {self.num_dofs}")
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device) # consecutive successes #
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device) # apply force #
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0
        
        self.ts_to_hand_obj_states = {}
        
        self.ref_ts =  0
        self.reset_nn = 0
    
    
    def _calculate_obj_vels(self, obj_trans, obj_rot_quat):
        mocap_freq = 120
        mocap_dt = float(1) / float(mocap_freq)
        obj_lin_vels = []
        obj_ang_vels = []
        for i_fr in range(obj_trans.shape[0]):
            nex_fr = i_fr + 1
            if nex_fr < obj_trans.shape[0]:
                cur_fr_trans = obj_trans[i_fr]
                nex_fr_trans = obj_trans[nex_fr]
                obj_lin_vel = (nex_fr_trans - cur_fr_trans) / float(mocap_dt)
                obj_lin_vels.append(obj_lin_vel)

                cur_fr_rot_euler = R.from_quat(obj_rot_quat[i_fr]).as_euler('xyz', degrees=False)
                nex_fr_rot_euler = R.from_quat(obj_rot_quat[nex_fr]).as_euler('xyz', degrees=False)
                obj_rot_vel = (nex_fr_rot_euler - cur_fr_rot_euler) / float(mocap_dt)
                obj_ang_vels.append(obj_rot_vel)
        obj_lin_vels.append(obj_lin_vels[-1])
        obj_ang_vels.append(obj_ang_vels[-1])
        obj_lin_vels = np.stack(obj_lin_vels, axis=0)
        obj_ang_vels = np.stack(obj_ang_vels, axis=0)
        return obj_lin_vels, obj_ang_vels
            
    
    def _load_mocap_info(self,):
        print(f"==> Loading mocap reference information from {self.mocap_sv_info_fn}")
        save_info = np.load(self.mocap_sv_info_fn, allow_pickle=True).item()
        hand_qs = save_info['robot_delta_states_weights_np'] # n_ts x nn_q
        hand_qs = hand_qs[: , : self.nn_hand_dof]
        goal_obj_trans = save_info['object_transl'][:] # 
        goal_obj_rot_quat = save_info['object_rot_quat'][: ] # obj trans #
        self.hand_qs = hand_qs
        self.goal_obj_trans = goal_obj_trans
        self.goal_obj_rot_quat = goal_obj_rot_quat  # 
        print(f"==> Info loaded with hand_qs: {hand_qs.shape}, goal_obj_trans: {goal_obj_trans.shape}, goal_obj_rot_quat: {goal_obj_rot_quat.shape}")
        
        
        
        
        self.hand_qs_th = torch.from_numpy(hand_qs[:, :]).float().to(self.rl_device)
        self.goal_obj_trans_th = torch.from_numpy(goal_obj_trans).float().to(self.rl_device)
        self.goal_obj_rot_quat_th = torch.from_numpy(goal_obj_rot_quat).float().to(self.rl_device)
        
        self.hand_qs = self.hand_qs[:, :]
        
        
        goal_obj_lin_vels, goal_obj_ang_vels = self._calculate_obj_vels(goal_obj_trans, goal_obj_rot_quat)
        self.goal_obj_lin_vels_th = torch.from_numpy(goal_obj_lin_vels).float().to(self.rl_device)
        self.goal_obj_ang_vels_th = torch.from_numpy(goal_obj_ang_vels).float().to(self.rl_device)
        

    def create_sim(self):
        self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row): # create envs #
        object_scale_dict = self.cfg['env']['object_code_dict']
        
        if len(self.object_name) > 0: #
            object_scale_dict = { f'sem/{self.object_name}' : [1.0] }
        
        
        self.object_code_list = list(object_scale_dict.keys())
        all_scales = set()
        for object_scales in object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales): # # 
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)
        self.repose_z = self.cfg['env']['repose_z']
        self.repose_z = False

        self.grasp_data = {}
        assets_path = '../assets'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        print(f"[Debug] object_code_list: {self.object_code_list}")
        
        
        for object_code in self.object_code_list: # 
            data_per_object = {}
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            for num in data_num_list: # qpos, scale, target hand rot, target hand pos
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                qpos = data_dict['qpos'] # .item()
                scale_inverse = data_dict['scale'] # .item()
                scale = round(1 / scale_inverse, 2)
                print(f"[Debug] scale: {scale}")
                # assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.from_numpy(qpos).float().to(self.device)
                target_hand_rot_xyz = torch.zeros((3, ), device=self.device) 
                target_hand_pos = torch.zeros((3, ), device=self.device)
                
                
                target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                # target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                # plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                # translation, euler = plane2euler(plane, axes='sxyz')  # object
                # object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                # object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)
                # ## TODO: inspct into the common setting of init_z in the original pipeline #
                
                # object_euler_xy = torch.tensor([0, 0], dtype=torch.float, device=self.device)
                # # object_init_z = torch.tensor([0.049], dtype=torch.float, device=self.device)
                # object_init_z = torch.tensor([0.0], dtype=torch.float, device=self.device)
                
                
                # self.init_hand_q = data_dict['qpos_init']
                self.init_hand_q = self.hand_qs[0, :]
                self.init_hand_q = torch.from_numpy(self.init_hand_q).float().to(self.device)
                
                
                init_obj_tarns = self.goal_obj_trans[0, :]
                init_obj_quat = self.goal_obj_rot_quat[0, :]
                
                
                self.obj_trans_init = torch.from_numpy(init_obj_tarns).float().to(self.device)
                self.obj_ornt_init = torch.from_numpy(init_obj_quat).float().to(self.device)
                
                obj_ornt_init_euler_xyz = R.from_quat(init_obj_quat).as_euler('xyz', degrees=False) # not th degress # 
                object_euler_xy = torch.tensor([obj_ornt_init_euler_xyz[0], obj_ornt_init_euler_xyz[1]], dtype=torch.float, device=self.device) # 
                object_init_z = torch.tensor([init_obj_tarns[2]], dtype=torch.float, device=self.device) ## ge the devie # 
                
                print(f"[Debug] object_init_z: {object_init_z}, object_euler_xy: {object_euler_xy}")
                
                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                    data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                    data_per_object[scale]['object_init_z'].append(object_init_z)
                else:
                    data_per_object[scale] = {}
                    data_per_object[scale]['target_qpos'] = [target_qpos]
                    data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                    data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                    data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                    data_per_object[scale]['object_init_z'] = [object_init_z]
            self.grasp_data[object_code] = data_per_object

        ### NOTE: not a fly hand here ###
        # target pos is all joints ? # 
        self.goal_cond = self.cfg["env"]["goal_cond"] 
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        # add the env test mode # 
        
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        
        # asset_root = "../../assets" # 
        # asset_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc"
        asset_root = "../assets"
        
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        # if "asset" in self.cfg["env"]: # asset and the env #
        #     asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root) # asset and the env #
        #     shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file) # asset file name #

        print(f"[Debug] shadow_hand_asset_file: {shadow_hand_asset_file}")
        
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False # fixed base link #
        asset_options.fix_base_link =  True # False
        # asset_options.collapse_fixed_joints = True
        if self.use_fingertips:
            asset_options.collapse_fixed_joints = False
        else:
            asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        ## TODO: angulear damping and linear damping settings in the pybullet? 
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        ## TODO: default dof drive mode? -- NONE ??
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        ### TODO: what's the difference between tendons and regular actuators? 
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        # self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        asset_rigid_body_names = [self.gym.get_asset_rigid_body_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_bodies)]
        print("asset_rigid_body_names: ", asset_rigid_body_names)
        
        
        
        # tendon set up
        # TODO: tendon set up? #
        limit_stiffness = 30
        t_damping = 0.1
        # relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        # tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        # for i in range(self.num_shadow_hand_tendons):
        #     for rt in relevant_tendons:
        #         if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
        #             tendon_props[i].limit_stiffness = limit_stiffness
        #             tendon_props[i].damping = t_damping

        # self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        # TODO: tendon set up? #

        # actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        # self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

 
        # set shadow_hand dof properties # 
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            ### HACK ###
            print(f"i: {i}")
            # if i > 5:
                
            #     self.shadow_hand_dof_lower_limits.append(0.0)
            # else:
            #     self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        
        
        
        
        
        print(f"[Debug] shadow_hand_dof_lower_limits: {self.shadow_hand_dof_lower_limits}")
        print(f"[Debug] shadow_hand_dof_upper_limits: {self.shadow_hand_dof_upper_limits}")

        shadow_hand_joint_dict = self.gym.get_asset_joint_dict(shadow_hand_asset)
        for key in shadow_hand_joint_dict:
            val = shadow_hand_joint_dict[key]
            print(f"key: {key} val: {val}")

        # visual feature
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012', # get
            0.15: '015',
            1: '1',
        } # 

        object_scale_idx_pairs = []
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64), device=self.device)
        
        for object_id in range(len(self.object_code_list)):
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                
                object_scale_idx_pairs.append([object_id, scale_id])
                self.visual_feat_data[object_id][scale_id] = torch.zeros((64, ), device=self.device, dtype=torch.float)
                
                # ##### original code #####
                # if scale in self.grasp_data[object_code]:
                #     object_scale_idx_pairs.append([object_id, scale_id])
                # else:
                #     print(f'prior not found: {object_code}/{scale}')
                # file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                # with open(file_dir, 'rb') as f:
                #     feat = np.load(f)
                # self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)     
                # ##### original code #####   

        object_asset_dict = {}
        goal_asset_dict = {}

        ## TODO: decompose the object and add into the mesh_data_scaled #
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            # object_asset_options.density = 500
            object_asset_options.density = self.rigid_obj_density
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            
            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    print(f"scaled_object_asset_file: {scaled_object_asset_file}")
                    scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file,
                                                              object_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset

                    if object_asset is None:
                        object_asset = scaled_object_asset
            
            assert object_asset is not None
            object_asset_options.disable_gravity = True    
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset) # asset dof properties #
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            # 
            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            print(f"[Debug] object_dof_lower_limits: {self.object_dof_lower_limits}")
            print(f"[Debug] object_dof_upper_limits: {self.object_dof_upper_limits}")


        # create table asset
        # table_dims = gymapi.Vec3(1, 1, 0.6)
        table_dims = gymapi.Vec3(1, 1, 0.00001)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform() # 
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # gymapi.Vec3(0.1, 0.1, 0.65)
        # shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)

        object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        obj_init_x = self.obj_trans_init[0].item() # 
        obj_init_y = self.obj_trans_init[1].item() # 
        obj_init_z = self.obj_trans_init[2].item() # 
        object_start_pose.p = gymapi.Vec3(obj_init_x, obj_init_y, obj_init_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0) # from euler zyx #
        # pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
        object_start_pose.r = gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())
        
        

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.r = gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())

        goal_start_pose.p.z -= 0.0 # goal start pose # # goal pose #

        table_pose = gymapi.Transform()
        # table_pose.p = gymapi.Vec3(0.0, 0.0, -0.5 * table_dims.z)
        table_pose.p = gymapi.Vec3(0.0, 0.0, -1.0 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size # num shadowh hand bodies # 
        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ## #
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1  ##

        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        
        # ## TODO: change it to the alegro body name #
        # body_names = {
        #     # 'wrist': 'robot0:wrist',
        #     'palm': 'palm_link',
        #     'thumb': 'link_15',
        #     'index': 'link_3',
        #     'middle': 'link_7',
        #     'ring': 'link_11',
        #     # 'little': 'robot0:lfdistal'
        # }
        
        # if self.use_fingertips:
        #     body_names = {
        #         # 'wrist': 'robot0:wrist',
        #         'palm': 'palm_link',
        #         'thumb': 'link_15_tip',
        #         'index': 'link_3_tip',
        #         'middle': 'link_7_tip',
        #         'ring': 'link_11_tip',
        #         # 'little': 'robot0:lfdistal'
        #     }
        # else:
        #     body_names = {
        #         # 'wrist': 'robot0:wrist',
        #         'palm': 'palm_link',
        #         'thumb': 'link_15',
        #         'index': 'link_3',
        #         'middle': 'link_7',
        #         'ring': 'link_11',
        #         # 'little': 'robot0:lfdistal'
        #     }
        
        # print(f"[Debug] fingertips handles: {self.fingertip_handles}")
        # self.hand_body_idx_dict = {}
        # for name, body_name in body_names.items():
        #     self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)
        #     print(f"[Debug] body_name: {name}, idx: {self.hand_body_idx_dict[name]}")

        # # create fingertip force sensors, if needed
        # if self.obs_type == "full_state" or self.asymmetric_obs: # 
        #     sensor_pose = gymapi.Transform()
        #     for ft_handle in self.fingertip_handles:
        #         self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)



        self.object_scale_buf = {}

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)


            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)


            if self.obs_type == "full_state" or self.asymmetric_obs: 
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)


            id = int(i / self.num_envs * len(self.object_code_list))
            object_code = self.object_code_list[id]
            available_scale = []
            for scale_id in self.object_scale_id_list[id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    available_scale.append(scale)
                else:
                    print(f'prior not found: {object_code}/{scale}')
            scale = available_scale[i % len(available_scale)]
            scale_id = self.scale2id[scale]
            self.object_scale_buf[i] = scale
            self.object_id_buf[i] = id

            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            #### NOTE: we have disabled table here ##
            # table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            # table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            # self.table_indices.append(table_idx)

            # set friction
            # table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            # table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            # self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # shadow hand default dof pos #
        self.shadow_hand_default_dof_pos = self.init_hand_q
        
        if self.use_canonical_state:
            # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
            # self.shadow_hand_default_dof_pos[2] = 0.2
            if self.use_unified_canonical_state:
                self.shadow_hand_default_dof_pos = torch.zeros_like(self.shadow_hand_default_dof_pos)
                self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
                self.shadow_hand_default_dof_pos[2] = 0.2
                self.shadow_hand_default_dof_pos[1] = 0.0
            else:
                self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
                self.shadow_hand_default_dof_pos[2] = 0.2
                self.shadow_hand_default_dof_pos[1] = -0.07
        
        ###### ######
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # the best grasping strategy  # 
        
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        # self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)



    def compute_observations(self):
        ### TODO: about the link body pose and the object global pose? ###
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # object pos # 
        # object pose #
        # shadow_hand_dof_pos # 
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        # self. #
        self.object_handle_pos = self.object_pos  ## + quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos # + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        # object linvel # object angvel # object linvel #
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]



        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        # idx = self.hand_body_idx_dict['little']
        # self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        # self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        # # self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3] # goal state #
        self.goal_rot = self.goal_states[:, 3:7]
        
        self.goal_lifting_pos = self.goal_states[:, 0:3] # goal lifting 
        
        maxx_progress = torch.max(self.progress_buf)
        minn_progress = torch.min(self.progress_buf)
        # print(f"maxx_progress: {maxx_progress}, minn_progress: {minn_progress}, goal_obj_trans_th: {self.goal_obj_trans_th.size()}")
        
        # goal obj trans # 
        cur_goal_pos = self.goal_obj_trans_th[self.progress_buf]
        cur_goal_rot = self.goal_obj_rot_quat_th[self.progress_buf]
        
        
        self.goal_pos_ref = cur_goal_pos
        self.goal_rot_ref = cur_goal_rot 
        self.goal_pose_ref = torch.cat(
            [self.goal_pos_ref, self.goal_rot_ref], dim=-1
        )
        
        if self.separate_stages:
            obj_lowest_z_less_than_thres = (self.object_pos[:, 2] < 0.19)
            self.goal_pos = torch.where(
                obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 3), self.goal_pos, self.goal_pos_ref
            )
            self.goal_rot = torch.where(
                obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 4), self.goal_rot, self.goal_rot_ref
            )
            self.goal_pose = torch.cat(
                [self.goal_pos, self.goal_rot], dim=-1
            )
        else:
            self.goal_pose = self.goal_pose_ref
            self.goal_rot = self.goal_rot_ref
            self.goal_pos  = self.goal_pos_ref # get the goal rot, pos, and pose #
        
        
        
        
        # goal_linvel, goal_angvel # # goal angvel #
        cur_goal_lin_vels = self.goal_obj_lin_vels_th[self.progress_buf]
        cur_goal_ang_vels = self.goal_obj_ang_vels_th[self.progress_buf]
        self.goal_linvel = cur_goal_lin_vels
        self.goal_angvel = cur_goal_ang_vels
        self.goal_vels = torch.cat(
            [self.goal_linvel, self.goal_angvel], dim=-1
        )

        # fingertip state #
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

        # ### HACK : original implementation ####
        # self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        # self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        # self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        # ### HACK : original implementation ####
        
        ### HACK ###
        self.delta_target_hand_pos = torch.zeros((3,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.rel_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.delta_target_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ### HACK ###
        

        ### current target hand pose, and the difference from the reference hand pos ###
        cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        
        ### next progress buffer ###
        nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        
        
        # shadow_hand_dof_pos, object_pos #
        # self.shadow_hand_dof_pos # 
        # self.object_pose #
        self.object_pose_np = self.object_pose.detach().cpu().numpy()
        self.shadow_hand_dof_pos_np = self.shadow_hand_dof_pos.detach().cpu().numpy()
        self.ts_to_hand_obj_states[self.ref_ts] = {
            'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
            'object_pose': self.object_pose_np
        }
        
        # self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos
        # comptue full state #
        self.compute_full_state()

        if self.asymmetric_obs: # asymmetric obs #
            self.compute_full_state(True)

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec): # p
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()

        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0:66
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        
        if self.obs_type == "full_state" or asymm_obs:
            self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        
            fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        else:
            fingertip_obs_start = 2 * self.num_shadow_hand_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(self.num_fingertips):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 66:131: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 131:161: ft sensors: do not need repose
        if self.obs_type == "full_state" or asymm_obs:
        #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques]
        # else
            self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :num_ft_force_torques]

            hand_pose_start = fingertip_obs_start + num_ft_states + num_ft_force_torques #  95
        else:
            hand_pose_start = fingertip_obs_start + num_ft_states
        # 161:167: hand_pose
        ### Global hand pose ###
        
        
        if self.tight_obs:
            # right_hand_rot
            self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
            euler_xyz = get_euler_xyz(self.unpose_quat(self.right_hand_rot))
        else:
            self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
            euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)
            
        # Actions #
        action_obs_start = hand_pose_start + 6
        # 167:191: action #
        try:
            aux = self.actions[:, :self.num_shadow_hand_dofs]
        except: # using the
            aux = torch.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=torch.float32, device=self.device)
        aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_shadow_hand_dofs] = aux

        obj_obs_start = action_obs_start + self.num_shadow_hand_dofs  # 144
        # 191:207 object_pose, goal_pos
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

        # + 6 + nn_dof (action) + 16 (obj) + 7 + nn_dof (goal) + 64
         # 207:236 goal # obj obs start # 
        hand_goal_start = obj_obs_start + 16
        
        if self.tight_obs:
            self.obs_buf[:, hand_goal_start: hand_goal_start +  self.num_shadow_hand_dofs] = self.delta_qpos
        else:
            self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
            self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
            # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos
            self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.delta_qpos

            hand_goal_start = hand_goal_start + 7

        
        # prue state wref #
        if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
            cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
            self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs]
            
        elif self.obs_type == 'pure_state_wref':
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
        elif not self.obs_type == 'pure_state':
            
            # 236: visual feature 
            visual_feat_start = hand_goal_start + self.num_shadow_hand_dofs #  29
            
            # 236: 300: visual feature # # obs buf #
            self.obs_buf[:, visual_feat_start: visual_feat_start + 64] = 0.1 * self.visual_feat_buf
            self.obs_buf[:, visual_feat_start + 64: 300] = 0.0
        # use relative controls #
        return

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        rand_length = torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        rand_angle = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor # 
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids=None, goal_env_ids=None):
        
        self.ref_ts = 0
        
        if env_ids is None : 
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        ### NOTE: we have disabled the random_prior setting 
        # if self.random_prior:
        #     for env_id in env_ids:
        #         i = env_id.item()
        #         object_code = self.object_code_list[self.object_id_buf[i]]
        #         scale = self.object_scale_buf[i]

        #         data = self.grasp_data[object_code][scale] # data for one object one scale # reset target pose #
        #         buf = data['object_euler_xy']
        #         prior_idx = random.randint(0, len(buf) - 1)
        #         # prior_idx = 0 ## use only one data

        #         self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
        #         self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
        #         self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
        #         self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
        #         self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]
        ### NOTE: we have disabled the random_prior setting 

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos # env_ids #

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        # shadow_hand_default_dof_pos #
        # self.dof_state[:, : self.]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        
        self.prev_delta_targets[env_ids, :] = 0
        self.cur_delta_targets[env_ids, :] = 0

        hand_indices = self.hand_indices[env_ids].to(torch.int32) # hand indices #
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        # all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##
        
        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], ]).to(torch.int32))  ##

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]


        ## NOTE: we disable the object random rotations here ##
        # theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        # #reset obejct with all data:
        # new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        # prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # # coordinate transform according to theta(object)/ prior_rot_z(hand)
        # self.z_theta[env_ids] = prior_rot_z
        # prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)


        ## NOTE: we disable the object random rotations here ##
        # self.hand_orientations[hand_indices.to(torch.long), :] = prior_rot_quat
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # reset object # #
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        ## NOTE: we disable the object random rotations here #
        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        # 
        # all indices #
        # all_indices = torch.unique(torch.cat([all_hand_indices, # 
        #                                       self.object_indices[env_ids],
        #                                       self.goal_object_indices[env_ids], # to torch.int32 #
        #                                       self.table_indices[env_ids], ]).to(torch.int32))
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids], ]).to(torch.int32))
        # state tensor indexed #
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            ## TODO: waht's the role of random-time here ##
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        
        # after that, we can set states #
        # if self.cfg.test and self.reset_nn % 1000 == 0:
        # if self.reset_nn % 1000 == 0:
        #     print(f"reset_nn: {self.reset_nn}")
        #     logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
        #     logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
        #     np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
        #     print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")
        
        self.reset_nn += 1
        
        self.compute_observations()
        
        
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def try_save(self, ):
        # if self.reset_nn % 1000 == 0:
        if self.reset_nn % 1 == 0:
            print(f"reset_nn: {self.reset_nn}")
            logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
            np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
            print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API 
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset() #
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        # use_kinematics_bias # 
        
        # self.get_pose_quat() # 
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6]) # 
        
        self.actions = actions.clone().to(self.device)


        if self.use_kinematics_bias_wdelta:
            # print(f"self.use_kinematics_bias_wdelta: {self.use_kinematics_bias_wdelta}")
            # prev_delta_targets # 
            increased_progress_buf = self.progress_buf + 1
            increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
            # get the kinematicsof the increaesd progres buf as the kinematics bias # 
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # hand_qs_th
            ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
            # prev_detlat_targets # 
            delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
            self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
            self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
            if self.kinematics_only:
                cur_targets = ctl_kinematics_bias
            else:
                cur_targets = ctl_kinematics_bias + self.cur_delta_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            
        # if we use the kinematics motion as the bis # 
        elif self.use_kinematics_bias:
            increased_progress_buf = self.progress_buf + 1
            increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
            # get the kinematicsof the increaesd progres buf as the kinematics bias # 
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # hand_qs_th
            ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
            # ctl kinematics bias #
            if self.kinematics_only:
                targets = ctl_kinematics_bias
            else:
                # targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale * self.dt * self.actions 
                #### from actions to targets ####
                targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            # kinematics_only # targets #
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        elif self.use_relative_control: # use relative control #
            # 
            # targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            # shadow_hand_dof_speed_scale_tsr
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # targets = self.prev_targets #
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            ### TODO: check whether if it is because we use the shadow hand dof pos to set the dof state tensor, so we need to set the dof state tensor here ###
            # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        # prev targets an the current jarets #
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.ref_ts += 1

        self.compute_observations()
        # self.compute_reward(self.actions)
        
        if self.test: # test the test setting #
            if self.ref_ts >= self.max_episode_length - 3: # try save #
                self.try_save()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])



#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot