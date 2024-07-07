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

import pickle
import numpy as np
import torch
# from ..Utils.motion_dataset import MotionDataSet
# from scipy.spatial.transform import Rotation
# try:
#     from VclSimuBackend import SetInitSeed
# except:
#     from ModifyODE import SetInitSeed
# from ..Utils.motion_utils import character_state, state2ob, state_to_BodyInfoState
from ..Utils.index_counter import index_counter


# import VclSimuBackend
# try:
#     from VclSimuBackend.Common.MathHelper import MathHelper
#     from VclSimuBackend.ODESim.Saver import CharacterToBVH
#     from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
#     from VclSimuBackend.ODESim.PDControler import DampedPDControler
# except ImportError:
#     MathHelper = VclSimuBackend.Common.MathHelper
#     CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
#     JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
#     DampedPDControler = VclSimuBackend.ODESim.PDController.DampedPDControler


import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from utils import pytorch_utils as ptu
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC
# import gym
# from gym import spaces

# 1) env #
# 2) train the env 
##### the number of observations ; and actions -> used to construct the trainer world model 
##### the ## observations #####

# track env #
class VCLODETrackEnv():
    """A tracking environment, also performs as a environment base... because we need a step function.
    it contains:
        a ODEScene: physics environment which contains two characters, one for simulation another for kinematic
        a MotionDataset: contains target states and observations of reference motion
        a counter: represents the number of current target pose
        a step_cnt: represents the trajectory length, will be set to zero when env is reset
    """
    def __init__(self, **kargs) -> None:
        super(VCLODETrackEnv, self).__init__()
        
        self.object_reset(**kargs)
    
    def object_reset(self, **kargs):
        if 'seed' in kargs:
            self.seed(kargs['seed']) # using global_seed    
        
        
        
        # self.min_length = kargs['env_min_length']
        # self.max_length = kargs['env_max_length']
        self.err_threshod = kargs['env_err_threshod']
        self.err_length = kargs['env_err_length']
        # functional element
        # self.stable_pd = DampedPDControler(self.sim_character)
        # name_list = self.sim_character.body_info.get_name_list()
        # self.head_idx = name_list.index('head')
        self.balance = not kargs['env_no_balance']
        self.use_com_height = kargs['env_use_com_height']
        self.recompute_velocity = kargs['env_recompute_velocity']
        self.random_count = kargs['env_random_count']
        self.wnorm = kargs['wnorm']
        
        self.reset_mano_states = kargs['reset_mano_states']
        
        self.traj_opt = kargs['traj_opt']
        self.use_mano_delta_states = kargs['use_mano_delta_states']
        
        self.use_preset_inertia = kargs['use_preset_inertia']
        self.angular_damping_coef = kargs['angular_damping_coef']
        self.damping_coef = kargs['damping_coef']
        ## rnadom count ##
        # reset -> #
        
        # self.p_init = 0
        
        ## adapt to a new object -> rescale the mano hand ##
        ## object urdf ##
        ## restart strategy robot hand init rotations and transformations ##
        
        ## save the rferecne data # 
        
        
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        ## scaled scaled wcollision ##
        ### scaled scaled wcollision ### # # urdf_fn # # ##
        # mano_hand_mean_meshcoll_urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled.urdf"
        mano_hand_mean_meshcoll_urdf_fn = kargs['mano_urdf_fn']
        left_hand_urdf_fn = kargs['left_mano_urdf_fn']
        
        ## mao hand mena meshcoll urdf fn ##
        hand_model_name = mano_hand_mean_meshcoll_urdf_fn.split("/")[-1].split(".")[0]
        if "mano" in hand_model_name:
            self.hand_type = "mano"
        elif "shadow" in hand_model_name:
            self.hand_type = "shadow"
        else:
            raise NotImplementedError(f"Unknown hand type: {hand_model_name}")
        
        
        self.mano_obj_urdf_fn = mano_hand_mean_meshcoll_urdf_fn     
        mano_hand_idx = p.loadURDF(self.mano_obj_urdf_fn,  [0, 0, 0], useFixedBase=1) ### fixed bae ###
        self.mano_hand_idx = mano_hand_idx # get the mano hand idx # ## mano hand idx ##
        
        self.left_hand_urdf_fn = left_hand_urdf_fn
        left_hand_idx = p.loadURDF(self.left_hand_urdf_fn, [0, 0, 0], useFixedBase=1)   
        self.left_mano_hand_idx = left_hand_idx
        
        
        # -0.0348542
        # minn_z = -0.0348542
        plane_z = -0.045 # set a lower plane for the cube #
        if 'plane_z' in kargs:
            plane_z = kargs['plane_z']  # get plane_z #
        self.plane_z = plane_z
        print(f"Loading plane with z: {plane_z}")
        p.loadURDF("plane.urdf", [0, 0, self.plane_z], useFixedBase=True)
    
        p.loadURDF("table/table.urdf", [0, 0, 0.34], useFixedBase=True)
    
    
        
        ## urdf for the box ##
        # ## load urdf for box ##
        # box_urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/redmax_box_test_3_wcollision.urdf"
        box_urdf_fn = kargs['obj_urdf_fn'] # obj urdf fn #
        
        self.box_urdf_fn = box_urdf_fn
        print(f"box_urdf_fn; {box_urdf_fn}")
        box_idx = p.loadURDF(box_urdf_fn,  [0, 0, 0], useFixedBase=0) # # 
        self.box_idx = box_idx
        
        
        
        ''''''' hand number fingers '''''''
        bullet_mano_num_joints = p.getNumJoints(mano_hand_idx) # ## get the manojoints ##
        self.bullet_mano_num_joints = bullet_mano_num_joints
        
        st_finger_joints_tates = 6
        self.st_finger_joints_tates = st_finger_joints_tates
        
        self.bullet_mano_finger_num_joints = self.bullet_mano_num_joints - st_finger_joints_tates
        
        self.observatin_size = 3 + 3 + self.bullet_mano_finger_num_joints  + 3 + 4 ## 3 + 3 + 15 + 3 + 4 = 28 ##
        
        
        
        gravity_scale = 0.
        p.setGravity(0, 0, gravity_scale) ## gravity scale ##
        # p.setRealTimeSimulation(1) ##
        
        # self.real_bullet_nn_substeps = 100
        # self.bullet_nn_substeps = 100
        
        
        ''''''' Basic simulation parameters '''''''
        nn_substeps = kargs['bullet_nn_substeps']
        self.bullet_nn_substeps = nn_substeps
        self.real_bullet_nn_substeps = nn_substeps
        
        # bullet_nn_substeps = self.bullet_nn_substeps
        # ours_sim_cons_rot = 0.0005 / float(bullet_nn_substeps)
        # ours_sim_cons_rot = 0.0005 / float(10)
        # ours_sim_cons_rot = 0.0005 / float(self.bullet_nn_substeps)
        ours_sim_cons_rot = 0.0005 / float(10000)
        ours_sim_cons_rot = 0.0005 / float(1000)
        ours_sim_cons_rot = 0.0005 / float(100)
        # ours_sim_cons_rot = 0.0005 / float(200)
        # ours_sim_cons_rot = 0.0005 / float(1000)
        
        ours_sim_cons_rot = 0.0005 ## 0.05 -> ? 
        
        ours_sim_cons_rot = 0.00005 ## 0.05 -> ? 
        
        ours_sim_cons_rot = 1.0 / float(240) ## use a smaller dt? ## # default dt ##
        
        ours_sim_cons_rot = 0.0005 / float(100)
        self.dt = ours_sim_cons_rot ## dt 
        # self.real_bullet_nn_substeps = 100 ## 100 as the substeps ##
        # self.real_bullet_nn_substeps = 10 #
        
        ### set the timestep ##
        p.setTimeStep(self.dt) ## orus sim cons rot ##
        
        ### change friction 
        firction_coef = 10.0
        box_friction_coef = 10.0 
        
        # damping_coef = 100.0
        
        self.firction_coef = firction_coef
        self.box_friction_coef = box_friction_coef
        # self.damping_coef = damping_coef
        # self.angular_damping_coef = 1.0
        # self.angular_damping_coef = 1.0
        # self.angular_damping_coef = 

        ## timesteppoing ###
        p.changeDynamics(self.mano_hand_idx, -1, lateralFriction=self.firction_coef)
        # # p.changeDynamics(mano_hand_idx, -1, spinningFriction=firction_coef)
        p.changeDynamics(self.mano_hand_idx, -1, rollingFriction=self.firction_coef)
        
        p.changeDynamics(self.left_mano_hand_idx, -1, lateralFriction=self.firction_coef)
        # # p.changeDynamics(mano_hand_idx, -1, spinningFriction=firction_coef)
        p.changeDynamics(self.left_mano_hand_idx, -1, rollingFriction=self.firction_coef)
        for i_link in range(0, self.bullet_mano_num_joints): ## get bullet mano substates##
            p.changeDynamics(self.mano_hand_idx, i_link, lateralFriction=self.firction_coef)
            # p.changeDynamics(mano_hand_idx, i_link, spinningFriction=firction_coef)
            p.changeDynamics(self.mano_hand_idx, i_link, rollingFriction=self.firction_coef)
            
            p.changeDynamics(self.left_mano_hand_idx, i_link, lateralFriction=self.firction_coef)
            p.changeDynamics(self.left_mano_hand_idx, i_link, rollingFriction=self.firction_coef)
        p.changeDynamics(self.box_idx, -1, lateralFriction=self.box_friction_coef)
        # # p.changeDynamics(box_idx, -1, spinningFriction=box_friction_coef)
        p.changeDynamics(self.box_idx, -1, rollingFriction=self.box_friction_coef)
        
        p.changeDynamics(self.box_idx, -1, linearDamping=self.damping_coef) # 
        # p.changeDynamics(self.box_idx, -1, angularDamping=0.4) # 
        ## damping and damping coefs ##
        p.changeDynamics(self.box_idx, -1, angularDamping=self.angular_damping_coef) #  ## angular damping  ## a good value for ball
        # p.changeDynamics(box_idx, -1, angularDamping=0.3) # 
        # p.changeDynamics(box_idx, -1, angularDamping=2.0) 
        
        
        
        ''''''' Load GT data '''''''
        ## TODO: move this one to an argument ##
        # self.gt_data_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/bullet_hoi_data.npy"
        self.gt_data_fn = kargs['sv_gt_refereces_fn'] # data fn ##
        self.gt_data = np.load(self.gt_data_fn, allow_pickle=True).item()
        
        ##### mano glb rot #####
        self.mano_glb_rot = self.gt_data['mano_glb_rot']
        self.mano_glb_rot = self.mano_glb_rot / np.clip(np.sqrt(np.sum(self.mano_glb_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        
        
        
        self.mano_glb_trans = self.gt_data['mano_glb_trans'] # 
        ## mano rot ## ## reset the state ---> # 
        if self.hand_type == "mano":
            self.mano_states = self.gt_data['mano_states'][:, :self.bullet_mano_finger_num_joints]
        elif self.hand_type == "shadow":
            self.mano_states = self.gt_data['mano_states'][:, 2 : 2 + self.bullet_mano_finger_num_joints]
        print(f"[sample mano states] {self.mano_states[1]}")
        
        print(f"mano_glb_rot: {self.mano_glb_rot.shape}, mano_glb_trans: {self.mano_glb_trans.shape}, mano_states: {self.mano_states.shape}")
            
        
        if self.reset_mano_states:
            if '30' in self.gt_data_fn:
                print(f"Resetting mano states...")
                self.mano_states[:, 56 - 6] = 3.0
                self.mano_states[:, 58 - 6] = 0.9
            elif '20' in self.gt_data_fn:
                print(f"Resetting mano states...")
                self.mano_states[:, 56 - 6] = -0.2
                self.mano_states[:, 55 - 6] = 0.0
                self.mano_states[:, 57 - 6] = 0.0
            elif '25' in self.gt_data_fn:
                
                self.mano_states[:, 58 - 6] = 0.0
                self.mano_states[:, 59 - 6] = 0.0
                self.mano_states[:, 60 - 6] = 0.0
        
        ##### mano glb rot and mano glb states #####
        # left_mano_glb_rot, left_mano_glb_trans, left_mano_states #
        self.left_mano_glb_rot = self.gt_data['left_mano_glb_rot'] # left mano glb rot #
        self.left_mano_glb_rot = self.left_mano_glb_rot / np.clip(np.sqrt(np.sum(self.left_mano_glb_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
        self.left_mano_glb_trans = self.gt_data['left_mano_glb_trans']
        if self.hand_type == "mano":
            self.left_mano_states = self.gt_data['left_mano_states'][:, :self.bullet_mano_finger_num_joints]
        elif self.hand_type == "shadow":
            self.left_mano_states = self.gt_data['left_mano_states'][:, 2 : 2 + self.bullet_mano_finger_num_joints]
        
        
        # self.obj_rot = self.gt_data['obj_rot'] # in the quaternion #
        # self.obj_trans = self.gt_data['obj_trans'] # # rot and trans # # 
        
        self.obj_rot = self.gt_data['optimized_quat'] # in the quaternion #
        self.obj_trans = self.gt_data['optimized_trans'] # # rot and trans # # 
        
        
        print(f"obj_rot: {self.obj_rot.shape}, obj_trans: {self.obj_trans.shape}")
        
        if self.obj_rot.shape[1] == 3:
            tot_obj_rot_quat = []
            for i_fr in range(self.obj_rot.shape[0]):
                
                cur_obj_rot_vec = self.obj_rot[i_fr]
                cur_obj_rot_struct = R.from_rotvec(cur_obj_rot_vec)
                
                ## curobj rot mtx##
                cur_obj_rot_mtx = cur_obj_rot_struct.as_matrix()
                cur_obj_rot_mtx = cur_obj_rot_mtx.T
                cur_obj_rot_struct = R.from_matrix(cur_obj_rot_mtx)
                cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
                
                # cur_obj_rot_quat = cur_obj_rot_struct.as_quat() # 
                cur_obj_rot_quat = cur_obj_rot_quat[[3, 0, 1, 2]] ## as 3, 0, 1, 2 # 
                
                
                p.resetBasePositionAndOrientation(box_idx, posObj=self.obj_trans[i_fr].tolist(), ornObj=cur_obj_rot_quat[[1, 2, 3, 0]])
                obj_state_info = p.getBasePositionAndOrientation(box_idx)
                env_obj_pos = obj_state_info[0]
                env_obj_rot = np.array(obj_state_info[1])
                env_obj_rot = env_obj_rot[[3, 0, 1, 2]]
                
                # print(f"i_fr: {i_fr}, cur_obj_rot_quat: {cur_obj_rot_quat}, env_obj_rot: {env_obj_rot}")
                
                tot_obj_rot_quat.append(env_obj_rot)
                
            tot_obj_rot_quat = np.stack(tot_obj_rot_quat, axis=0)
            
            self.obj_rot = tot_obj_rot_quat # obj rot quat # # # 
        else:
            self.obj_rot = self.obj_rot / np.clip(np.sqrt(np.sum(self.obj_rot**2, axis=-1, keepdims=True)), a_min=1e-5, a_max=None)
            ## i_fr ##
            tot_obj_quat = []
            for i_fr in range(self.obj_rot.shape[0]):
                cur_obj_quat = self.obj_rot[i_fr]
                cur_xyzw_obj_quat = cur_obj_quat[[1, 2, 3, 0]]
                p.resetBasePositionAndOrientation(box_idx, posObj=[0,0,0], ornObj=cur_xyzw_obj_quat)
                obj_state_info = p.getBasePositionAndOrientation(box_idx)
                # env_obj_pos = obj_state_info[0]
                env_obj_rot = np.array(obj_state_info[1])
                env_obj_rot = env_obj_rot[[3, 0, 1, 2]]
                tot_obj_quat.append(env_obj_rot) # w x y z format
            tot_obj_quat = np.stack(tot_obj_quat, axis=0) ## stacked obj quat
            self.obj_rot = tot_obj_quat ### obj_rot and quat 
            
        
        
        ## initialization ##
        init_obj_rot_quat = self.obj_rot[0][[1, 2, 3, 0]]
        init_obj_trans = self.obj_trans[0]
        
        ## init the mano rot and trans ##
        p.resetBasePositionAndOrientation(box_idx, posObj=init_obj_trans.tolist(), ornObj=init_obj_rot_quat.tolist())
        
        ## p reset ase position and orientation ##
        
        
        
        # self.frame_length = min(self.mano_glb_rot.shape[0], self.obj_rot.shape[0]) - 11
        # 
        self.frame_length = min(self.mano_glb_rot.shape[0], self.obj_rot.shape[0]) - 2
        # self.frame_length = 3
        self.done = np.zeros((self.frame_length,), dtype=np.int32)
        self.done[-1] = 1
        
        self.max_length = self.frame_length ## frmae lenght ##
        
        # 
        self.calculate_obs_statistics() # ## caclulate obs statistics ## # 
        
        
        self.init_index = index_counter.calculate_feasible_index(self.done, 10)
        
        print(f"init_index: {self.init_index}")
        self.val = np.zeros(self.frame_length)    
        self.update_p()
        
        return 
    
    
    def calculate_obs_statistics(self,): ## obs statistics ## 
        # obs # delta # 
        tot_gt_delta = []
        tot_gt_obs = [] # framlength 
        denormalized_glb_actions = []
        for i_ts in range(self.frame_length):
            cur_gt_val = self.get_target(frame_idx=i_ts)
            tot_gt_obs.append(cur_gt_val)
            if i_ts > 0:
                prev_gt_obs = tot_gt_obs[-2]
                cur_gt_obs = tot_gt_obs[-1]
                delta_gt_obs = cur_gt_obs - prev_gt_obs
                tot_gt_delta.append(delta_gt_obs)
                #
                cur_denorm_glb_action = delta_gt_obs[:6]
                denormalized_glb_actions.append(cur_denorm_glb_action)
                
            # cur_obs = self._get_obs()
            # tot_gt_obs.append(cur_obs)
            # self.step_counter(random = False)
        tot_gt_obs = np.stack(tot_gt_obs, axis=0)
        obs_mean = np.mean(tot_gt_obs, axis=0)
        obs_std = np.std(tot_gt_obs, axis=0)
        self.obs_mean = obs_mean
        self.obs_std = obs_std## get obs mean and obs std ## # get std get mean ##
        
        tot_gt_delta = np.stack(tot_gt_delta, axis=0)
        delta_mean = np.mean(tot_gt_delta, axis=0)
        delta_std = np.std(tot_gt_delta, axis=0)
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        
        
        self.denormalized_glb_actions = np.stack(denormalized_glb_actions, axis=0) # denormalized actions #
        self.normalized_glb_actions = (self.denormalized_glb_actions - self.delta_mean[:6][None]) / (self.delta_std[:6][None] + 1e-8)
        # denormalized actions = 
        
        print(f"obs_mean: {self.obs_mean}, obs_std: {self.obs_std}")
        
        
        
    ### TODO: calculate statistics ### # tot_gt_obs ## #  tot gt obs ##
    def reset_passive_obj(self):
        
        p.removeBody(self.box_idx)
        if self.use_preset_inertia:
            box_idx = p.loadURDF(self.box_urdf_fn,  [0, 0, 0], useFixedBase=0, flags=p.URDF_USE_INERTIA_FROM_FILE) #
        else:
            box_idx = p.loadURDF(self.box_urdf_fn,  [0, 0, 0], useFixedBase=0) #
        self.box_idx = box_idx
        p.changeDynamics(self.box_idx, -1, lateralFriction=self.box_friction_coef)
        # # p.changeDynamics(box_idx, -1, spinningFriction=box_friction_coef)
        p.changeDynamics(self.box_idx, -1, rollingFriction=self.box_friction_coef)
        
        p.changeDynamics(self.box_idx, -1, linearDamping=self.damping_coef)
        # p.changeDynamics(self.box_idx, -1, angularDamping=0.4)
        p.changeDynamics(self.box_idx, -1, angularDamping=self.angular_damping_coef) 
    
    def reset_simulation(self,):
        p.resetSimulation()
        mano_hand_idx = p.loadURDF(self.mano_obj_urdf_fn,  [0, 0, 0], useFixedBase=1) ### fixed bae ###
        self.mano_hand_idx = mano_hand_idx # get the mano hand idx # ## mano hand idx ##
        
        
        box_idx = p.loadURDF(self.box_urdf_fn,  [0, 0, 0], useFixedBase=0) #
        self.box_idx = box_idx
        
        
        p.loadURDF("plane.urdf", [0, 0, self.plane_z], useFixedBase=True)
    
        gravity_scale = 0.
        p.setGravity(0, 0, gravity_scale) ## gravity scale ##
        # p.setRealTimeSimulation(1) ##
        
        ### set the timestep ##
        p.setTimeStep(self.dt) ## orus sim cons rot ##
        
        ## timesteppoing ###
        p.changeDynamics(self.mano_hand_idx, -1, lateralFriction=self.firction_coef)
        # # p.changeDynamics(mano_hand_idx, -1, spinningFriction=firction_coef)
        p.changeDynamics(self.mano_hand_idx, -1, rollingFriction=self.firction_coef)
        for i_link in range(0, self.bullet_mano_num_joints):
            p.changeDynamics(self.mano_hand_idx, i_link, lateralFriction=self.firction_coef)
            p.changeDynamics(self.mano_hand_idx, i_link, rollingFriction=self.firction_coef)
        p.changeDynamics(self.box_idx, -1, lateralFriction=self.box_friction_coef)
        p.changeDynamics(self.box_idx, -1, rollingFriction=self.box_friction_coef)
        
        p.changeDynamics(self.box_idx, -1, linearDamping=self.damping_coef) # 
        p.changeDynamics(self.box_idx, -1, angularDamping=0.4) # 
        p.changeDynamics(self.box_idx, -1, angularDamping=1.0) #
        
        
        
    
    def reset_state(self, **kargs):
        
        self.reset_passive_obj()
        # self.reset_simulation()
        
        ##  real bullet ##  ## reset the joint state ##
        # bullet_nn_substeps = self.real_bullet_nn_substeps
        # ours_sim_cons_rot = 0.0005 / float(bullet_nn_substeps)
        
        # reset state ##
        ''' right mano '''
        init_mano_state = self.mano_states[self.counter] # mano states
        init_mano_rot = self.mano_glb_rot[self.counter]
        # init_mano_rot = init_mano_rot # /torch.clamp( torch.norm(init_mano_rot, p=2, dim=-1), min=1e-5)
        init_mano_trans = self.mano_glb_trans[self.counter]
        
        # init_mano_rot = init_mano_rot # .detach().cpu().numpy()
        # init_mano_trans = init_mano_trans # .detach().cpu().numpy()
        # init_mano_state = init_mano_state # .detach().cpu().numpy() ### mano states ##
        
        ### rest mano states ## 
        revolute_joint_idx = 0 # init mano sat
        for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
            p.resetJointState(self.mano_hand_idx, cur_joint_idx, init_mano_state[revolute_joint_idx].item(), targetVelocity=0)
            revolute_joint_idx = revolute_joint_idx + 1
        
        ## 
        init_mano_rot = init_mano_rot[[1, 2, 3, 0]]
        init_mano_rot_struct = R.from_quat(init_mano_rot)
        
        init_mano_rot_vec = init_mano_rot_struct.as_euler('zyx', degrees=False)
        init_mano_rot_vec = [init_mano_rot_vec[2], init_mano_rot_vec[1], init_mano_rot_vec[0]]
        
        ## 
        # init_mano_rot_vec = init_mano_rot_struct.as_rotvec()
        
        for trans_joint_idx in range(3):
            cur_trans_joint_state = init_mano_trans[trans_joint_idx].item()
            p.resetJointState(self.mano_hand_idx, trans_joint_idx, cur_trans_joint_state, targetVelocity=0)
        
        for rot_joint_idx in range(3, 6):
            cur_rot_joint_state = init_mano_rot_vec[rot_joint_idx - 3].item()
            p.resetJointState(self.mano_hand_idx, rot_joint_idx, cur_rot_joint_state, targetVelocity=0)
        
        
        
        ''' left mano '''
        init_left_mano_state = self.left_mano_states[self.counter]
        init_left_mano_rot = self.left_mano_glb_rot[self.counter]
        init_left_mano_trans = self.left_mano_glb_trans[self.counter]
        
        revolute_joint_idx = 0
        for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
            p.resetJointState(self.left_mano_hand_idx, cur_joint_idx, init_left_mano_state[revolute_joint_idx].item(), targetVelocity=0)
            revolute_joint_idx = revolute_joint_idx + 1
        
        init_left_mano_rot = init_left_mano_rot[[1, 2, 3, 0]]
        init_left_mano_rot_struct = R.from_quat(init_left_mano_rot)
        init_left_mano_rot_vec = init_left_mano_rot_struct.as_euler('zyx', degrees=False)
        init_left_mano_rot_vec = [init_left_mano_rot_vec[2], init_left_mano_rot_vec[1], init_left_mano_rot_vec[0]]
        
        for trans_joint_idx in range(3):
            cur_trans_joint_state = init_left_mano_trans[trans_joint_idx].item()
            p.resetJointState(self.left_mano_hand_idx, trans_joint_idx, cur_trans_joint_state, targetVelocity=0)
        
        for rot_joint_idx in range(3, 6):
            cur_rot_joint_state = init_left_mano_rot_vec[rot_joint_idx - 3].item()
            p.resetJointState(self.left_mano_hand_idx, rot_joint_idx, cur_rot_joint_state, targetVelocity=0)
        
        
        
        
        init_obj_trans = self.obj_trans[self.counter]
        init_obj_rot = self.obj_rot[self.counter] 
        init_obj_rot = init_obj_rot[[1, 2, 3, 0]]  
        
        # print(f"Resetting box position to frame {self.counter} with {init_obj_trans}, rot: {init_obj_rot}")
        # based velocity -> need ar larger k value here ? #
        ### reset box states ### ## init obj rot ##
        # p.resetBasePositionAndOrientation(box_idx, posObj=[0, 0, 0,], ornObj=[0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.box_idx, posObj=init_obj_trans.tolist(), ornObj=init_obj_rot.tolist())
        p.resetBaseVelocity(self.box_idx, [0, 0, 0], [0, 0, 0])
        # for i_bullet_substep in range(bullet_nn_substeps):
        
        #     p.stepSimulation() # #
        # p.stepSimulation() 
        
        return

    def update_p(self):
        self.p = 1/ self.val.clip(min = 0.01) #
        self.p_init = self.p[self.init_index] #
        self.p_init /= np.sum(self.p_init)
    
        
    @property
    def stastics(self):
        # TODO: what does this used for? # 
        ## 
        
        # mean = torch.zeros((self.observatin_size, ), dtype=torch.float32).numpy()
        # std = torch.ones((self.observatin_size, ), dtype=torch.float32).numpy()
        
        ## mean std ##
        # delta_mean = torch.zeros((self.observatin_size - 1, ), dtype=torch.float32).numpy()
        # delta_std = torch.ones((self.observatin_size - 1, ), dtype=torch.float32).numpy()
        
        # stastics = { # statistics ##
            # 'obs_mean': mean, 'obs_std': std, 'delta_mean': delta_mean.copy(), 'delta_std': delta_std.copy() 
        # }
        stastics = {
            'obs_mean': self.obs_mean, 'obs_std': self.obs_std, 'delta_mean': self.delta_mean.copy(), 'delta_std': self.delta_std.copy() 
        }
        return stastics
        # return self.motion_data.stastics # 
    
    @property
    def sim_character(self):
        ## TODO: what's this used for #
        return self.scene.characters[0]

    @property
    def ref_character(self): # ref 
        ## TODO: what's this? ##
        return self.scene.characters[1]
    
    def get_bvh_saver(self): # 
        ## TODO: what's this # 
        bvh_saver = VclSimuBackend.ODESim.CharacterTOBVH(self.sim_character, self.fps)
        bvh_saver.bvh_hierarchy_no_root()
        return bvh_saver

    @staticmethod
    def seed( seed):
        # set init seeds ##
        SetInitSeed(seed)
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--env_contact_type", type=int, default=0, help="contact type, 0 for LCP and 1 for maxforce")
        arg_parser.add_argument("--env_close_self_collision", default=False, help="flag for closing self collision", action = 'store_true')
        arg_parser.add_argument("--env_min_length", type=int, default=26, help="episode won't terminate if length is less than this")
        arg_parser.add_argument("--env_max_length", type=int, default=512, help="episode will terminate if length reach this")
        arg_parser.add_argument("--env_err_threshod", type = float, default = 0.5, help="height error threshod between simulated and tracking character")
        arg_parser.add_argument("--env_err_length", type = int, default = 20, help="episode will terminate if error accumulated ")
        arg_parser.add_argument("--env_scene_fname", type = str, default = "character_scene.pickle", help="pickle file for scene")
        arg_parser.add_argument("--env_fps", type = int, default = 20, help="fps of control policy")
        arg_parser.add_argument("--env_substep", type = int, default = 6, help="substeps for simulation")
        arg_parser.add_argument("--env_no_balance", default = False, help="whether to use distribution balance when choose initial state", action = 'store_true')
        arg_parser.add_argument("--env_use_com_height", default = False, help="if true, calculate com in terminate condition, else use head height", action = 'store_true')
        arg_parser.add_argument("--motion_dataset", type = str, default = None, help="path to motion dataset")
        arg_parser.add_argument("--env_recompute_velocity", default = True, help = "whether to resample velocity")
        arg_parser.add_argument("--env_random_count", type=int, default=96, help = "target will be random switched for every 96 frame")
        return arg_parser
    
    def update_and_get_target(self):
        self.step_cur_frame()
        return 
    
    @staticmethod
    def isnotfinite(arr):
        res = np.isfinite(arr)
        return not np.all(res) # not all res is finite -> has infinite values 

    ## calculate done ##
    def cal_done(self, state, obs):
        # TODO: ours' done ?#  how to des # 
        # TODO: but just how tp judge whether the 
        
        #### the result has infinite values or has observation value larger than 50 -> invalid results and sthereturncode is 2 #
        if self.isnotfinite(state): ## calculate done ##
            return 2
        
        if np.any( np.abs(obs) > 50):
            return 2
        
        ### 
        if self.counter >= self.frame_length - 2:
            return 1
        
        # if self.step_cnt >= self.max_length:
        #     return 1
        return 0
        
        height = state[...,self.head_idx,1]
        target_height = self.motion_data.state[self.counter][self.head_idx,1]
        # height/ # 
        if abs(height - target_height) > self.err_threshod: # done cnt += 1 -> # 
            self.done_cnt +=1
        else:
            self.done_cnt = max(0, self.done_cnt - 1)
        
        #### the result has infinite values or has observation value larger than 50 -> invalid results and sthereturncode is 2 #
        if self.isnotfinite(state): ## # isnofinite ## # 
            return 2
        
        if np.any( np.abs(obs) > 50):
            return 2
        
        if self.step_cnt >= self.min_length:
            if self.done_cnt >= self.err_length:
                return 2
            if self.step_cnt >= self.max_length: # maximum length # done cnt # # 
                return 1
        return 0
    
    def update_val(self, done, rwd, frame_num):
        tmp_val = self.val / 2
        last_i = 0
        print(f"done: {done.shape}, rwd: {rwd.shape}, frame_num: {frame_num.shape}, tmp_val: {tmp_val.shape}")
        for i in range(frame_num.shape[0]):
            if done[i] !=0:
                tmp_val[frame_num[i]] = rwd[i] if done[i] == 1 else 0
                for j in reversed(range(last_i, i)):
                    tmp_val[frame_num[j]] = 0.95*tmp_val[frame_num[j+1]] + rwd[j]
                last_i = i
        self.val = 0.9 * self.val + 0.1 * tmp_val
        return


    def get_info(self):
        return {
            'frame_num': self.counter
            # 'frame_num': self.frame_length - 1
        }
        
    def _get_mano_target(self, frame_idx=-1): # get 
        cur_countdr = self.counter if frame_idx == -1 else frame_idx
        rot =self.mano_glb_rot[cur_countdr]
        
        rot = rot[[1, 2, 3, 0]]
        rot_struct = R.from_quat(rot)
        
        rot_vec = rot_struct.as_euler('zyx', degrees=False)
        rot_vec = [rot_vec[2], rot_vec[1], rot_vec[0]]
        
        # rot_vec = rot_struct.as_rotvec()
        
        
        trans = self.mano_glb_trans[cur_countdr]
        states = self.mano_states[cur_countdr]
        # mano_target = np.concatenate([rot_vec, trans, states], axis=-1)
        mano_target = np.concatenate([trans, rot_vec, states], axis=-1)
        return mano_target
    
    def _get_left_mano_target(self, frame_idx=-1):
        cur_countdr = self.counter if frame_idx == -1 else frame_idx
        left_rot = self.left_mano_glb_rot[cur_countdr]
        left_rot = left_rot[[1, 2, 3, 0]]
        left_rot_struct = R.from_quat(left_rot)
        
        left_rot_vec = left_rot_struct.as_euler('zyx', degrees=False)
        left_rot_vec = [left_rot_vec[2], left_rot_vec[1], left_rot_vec[0]]
        
        left_trans = self.left_mano_glb_trans[cur_countdr]
        left_states = self.left_mano_states[cur_countdr]
        
        left_mano_target = np.concatenate([left_trans, left_rot_vec, left_states], axis=-1)
        return left_mano_target
    
    
    def _get_obj_target(self, frame_idx=-1):
        cur_countdr = self.counter if frame_idx == -1 else frame_idx
        rot = self.obj_rot[cur_countdr]
        trans = self.obj_trans[cur_countdr] ## obj_rot 
        rot = rot[[1, 2, 3, 0]]
        # obj_target = np.concatenate([rot, trans], axis=-1)
        obj_target = np.concatenate([trans, rot], axis=-1) ## obj target ##
        return obj_target 
    
    def get_target(self, frame_idx=-1): # 
        # target # 
        ## use self.counter ##
        mano_target = self._get_mano_target(frame_idx=frame_idx)
        left_mano_target = self._get_left_mano_target(frame_idx=frame_idx)
        obj_target = self._get_obj_target(frame_idx=frame_idx)
        # target = np.concatenate([mano_target, obj_target], axis=-1) # get the mano and the obj targets # ## get the mano and obj target ## 
        
        trans_dim = 3
        rot_dim = 3
        mano_trans = np.concatenate([mano_target[:3], left_mano_target[:trans_dim]], axis=-1)
        mano_rot = np.concatenate([mano_target[trans_dim: trans_dim + rot_dim], left_mano_target[trans_dim: trans_dim + rot_dim]], axis=-1)
        mano_states = np.concatenate([mano_target[trans_dim + rot_dim: ], left_mano_target[trans_dim + rot_dim: ]], axis=-1)
        
        target = np.concatenate([mano_trans, mano_rot, mano_states, obj_target], axis=-1)
        # target = np.concatenate([mano_target, left_mano_target, obj_target], axis=-1) 
        return target
        # return self.motion_data.observation[self.counter]
    
    ## self.m
    @staticmethod
    def load_character_state(character, state):
        # TODO: what's this # # state to body infos
        ### TODO: reset state via mano states and object states ##
        character.load(state_to_BodyInfoState(state))
        aabb = character.get_aabb() # prevent under the floor....
        if aabb[2]<0:
            character.move_character_by_delta(np.array([0,-aabb[2]+1e-8,0]))
        
    
    ## when to rset the en 
    def step_counter(self, random = False): # random
        self.counter += 1
        if self.done[self.counter] or random:    
            self.counter = index_counter.random_select(self.init_index, p = self.p_init)
            ## set to counterst 
            # self.counter = 30
            
            
    ### reset the env ### ## reset the env ##
    def reset(self, frame = -1, set_state = True ):
        """reset enviroment

        Args:
            frame (int, optional): if not -1, reset character to this frame, target to next frame. Defaults to -1.
            set_state (bool, optional): if false, enviroment will only reset target, not character. Defaults to True.
        """
        self.counter = frame
        if frame == -1: # set the
            self.step_counter(random=True) # step counter # 
            print(f'reset to {self.counter}')
        self.reset_state() 
        self.state = self._get_obs() ## statet as the observation ##
        self.observation = self.state
        # if set_state: # set state # loadre set # 
        #     self.load_character_state(self.sim_character, self.motion_data.state[self.counter])     
           
        # ## TODO: currently we assum observations are the same as states ##
        # self.state = character_state(self.sim_character)
        # self.observation = state2ob(torch.from_numpy(self.state)).numpy()
        
        ## reste state ##
        info = self.get_info()
        
        self.step_counter(random = False) # random = false -> increase the counter ##
        self.step_cnt = 0
        self.done_cnt = 0
        return {
            'state': self.state,
            'observation': self.observation,
            'target': self.get_target(),
            'frame_num': self.counter # eval 
        }, info

    def after_step(self, **kargs):  # -> get target step the counter ## 
        """usually used to update target...
        """
        ### I thnkit is not successfully proceeded ###
        # increase the step counter ### #  # step # counter counts the step #
        # self.step_counter(random = (self.step_cnt % self.random_count)==0 and self.step_cnt!=0) # increaseth e
        # self.target = self.get_target()  # get target #
        # self.step_cnt += 1 # step # the end of the timestep ? 
        
        ## the target state ##
        ## the target state ##
        ## equal to the maximum length - 1 -> 
        ## start frame -> end frame ##
        ## start frame -> end frame ##
        ## step to te next step ##
        
        # is_rnd_start = self.counter == self.max_length - 1
        ##  ## mano glb rot mtx ##
        self.step_counter(random=self.counter >= self.max_length - 1) # random = false -> increase the counter ##
        self.target = self.get_target() # step the step and get the target #
        ## the target state is at the end timestep -> so need random start to proceed ##
        need_rnd_start = self.counter == self.max_length - 1
        return need_rnd_start
        
        
        
    def _get_mano_obs(self, ):
        target_sim_act_joint_info = p.getJointStates(self.mano_hand_idx, range(p.getNumJoints(self.mano_hand_idx)))
        target_sim_act_joint_info = [target_sim_act_joint_info[ii][0] for ii in range(len(target_sim_act_joint_info))]
        target_sim_act_joint_info = np.array(target_sim_act_joint_info, dtype=np.float32)
        
        left_target_sim_act_joint_info = p.getJointStates(self.left_mano_hand_idx, range(p.getNumJoints(self.left_mano_hand_idx)))
        left_target_sim_act_joint_info = [left_target_sim_act_joint_info[ii][0] for ii in range(len(left_target_sim_act_joint_info))]
        left_target_sim_act_joint_info = np.array(left_target_sim_act_joint_info, dtype=np.float32)
        
        target_sim_act_joint_info = np.concatenate([target_sim_act_joint_info, left_target_sim_act_joint_info], axis=-1)
        
        return target_sim_act_joint_info
        
    
    #  
    def _get_obs(self,):
        target_sim_act_joint_info = p.getJointStates(self.mano_hand_idx, range(p.getNumJoints(self.mano_hand_idx)))
        target_sim_act_joint_info = [target_sim_act_joint_info[ii][0] for ii in range(len(target_sim_act_joint_info))]
        target_sim_act_joint_info = np.array(target_sim_act_joint_info, dtype=np.float32)
        
        left_target_sim_act_joint_info = p.getJointStates(self.left_mano_hand_idx, range(p.getNumJoints(self.left_mano_hand_idx)))
        left_target_sim_act_joint_info = [left_target_sim_act_joint_info[ii][0] for ii in range(len(left_target_sim_act_joint_info))]
        left_target_sim_act_joint_info = np.array(left_target_sim_act_joint_info, dtype=np.float32)
        
        hand_trans_state = np.concatenate(
            [target_sim_act_joint_info[:3], left_target_sim_act_joint_info[:3]], axis=-1
        )
        hand_rot_state = np.concatenate(
            [target_sim_act_joint_info[3: 6], left_target_sim_act_joint_info[3: 6]], axis=-1
        )
        hand_finger_state = np.concatenate(
            [target_sim_act_joint_info[6: ], left_target_sim_act_joint_info[6: ]], axis=-1
        )
        hand_state = np.concatenate(
            [hand_trans_state, hand_rot_state, hand_finger_state], axis=-1
        )
        
        
        ### active object joint info #### # get base position ##
        box_info = p.getBasePositionAndOrientation(self.box_idx)
        box_trans = box_info[0] ## box trans
        box_ornt = box_info[1] ## 
        box_trans = np.array(box_trans, dtype=np.float32)
        box_ornt = np.array(box_ornt, dtype=np.float32) # 
        
        box_vel_info = p.getBaseVelocity(self.box_idx)
        # linear_vel = box_vel_info[0]
        # angular_vel = box_vel_info[1]
        # print(f"Got box trans: {box_trans}, box_ornt: {box_ornt}, linear_vel: {linear_vel}, angular_vel: {angular_vel}, frame: {self.counter}")
        # get the linear / angular box velocity ##
        ## get the linear / angular box velocities ##
        
        
        
        box_states = np.concatenate([box_trans, box_ornt], axis=-1) ## box states ##
        # sim_states = np.concatenate([target_sim_act_joint_info, box_states], axis=-1) ## sim_states ##
        sim_states = np.concatenate([hand_state, box_states], axis=-1) ## sim_states ##
        return sim_states
        
    # step core #
    # step 
    def step_core(self, action, using_yield = False, **kargs):
        
        ## step core ##
        # not all actions but the actions for #
        # # for every actions # 
        # would not remove the actions? #
        # add actions to mano params #
        
        ## aggragate actions here ##
        
        ## sim states ##
        sim_states = self._get_obs() ## get states ##
        self.state = sim_states ## get the observation ## g
        self.observation = sim_states
        cur_state = self._get_mano_obs()
        cur_state = torch.from_numpy(cur_state).float()
        
        
        
        # action = self.unnormalize_actions(action=action) # delta mean and delta std # actions  #
        
        # mano_target_state = self._get_mano_target(frame_idx=self.counter + 1) ##
        # action = np.concatenate([mano_target_state[:6] - sim_states[:6], action[6:]], axis=-1)
        
        # glb_delta_trans = action[:3]
        # glb_delta_rot = action[3:6]
        # glb_delta_state = action[6: self.bullet_mano_finger_num_joints + 6]
        
        # # cur_state = self._get_mano_target()
        # cur_glb_trans = cur_state[:3]
        # cur_glb_rot_vec = cur_state[3:6]
        # cur_states = cur_state[6:  6 + self.bullet_mano_finger_num_joints]
        
        # glb_tar_trans = cur_glb_trans + glb_delta_trans
        # glb_tar_rot = cur_glb_rot_vec + glb_delta_rot 
        # glb_tar_states = cur_states + glb_delta_state
        
        if not self.traj_opt: ## traj opt ##
            action = self.unnormalize_actions(action=action)
            
            mano_target_state = self._get_mano_target(frame_idx=self.counter + 1) ##
            action = np.concatenate([mano_target_state[:6] - sim_states[:6], action[6:]], axis=-1)
            
            ### policy sampling 
            glb_delta_trans = action[:3]
            glb_delta_rot = action[3:6]
            glb_delta_state = action[6: self.bullet_mano_finger_num_joints + 6]
            
            cur_glb_trans = cur_state[:3]
            cur_glb_rot_vec = cur_state[3:6]
            cur_states = cur_state[6:  6 + self.bullet_mano_finger_num_joints]
            
            glb_tar_trans = cur_glb_trans + glb_delta_trans
            glb_tar_rot = cur_glb_rot_vec + glb_delta_rot 
            glb_tar_states = cur_states + glb_delta_state
        else:
            glb_tar_trans = action[:3]
            glb_tar_rot = action[3:7]
            glb_tar_states = action[7: self.bullet_mano_finger_num_joints + 7]
            
            glb_tar_rot = glb_tar_rot[[1, 2, 3, 0]]
            glb_tar_rot_struct = R.from_quat(glb_tar_rot)
            
            glb_tar_rot_vec = glb_tar_rot_struct.as_euler('zyx', degrees=False)
            glb_tar_rot = [glb_tar_rot_vec[2], glb_tar_rot_vec[1], glb_tar_rot_vec[0]]
            glb_tar_rot = np.array(glb_tar_rot, dtype=np.float32)
        
        
        
        ## glb target trans ##
        # glb_tar_trans = action[:3]
        # glb_tar_rot = action[3:6]
        # glb_tar_states = action[6:self.bullet_mano_finger_num_joints + 6]
        
        ##set contorls and updatestates ####
        revolute_joint_idx = 0 # set revoluete joit states  # 
        for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
            p.setJointMotorControl2(self.mano_hand_idx, cur_joint_idx, p.POSITION_CONTROL, targetPosition=glb_tar_states[revolute_joint_idx].item(), force=1e20)
            revolute_joint_idx = revolute_joint_idx + 1
        
        for trans_joint_idx in range(3):
            cur_trans_joint_state = glb_tar_trans[trans_joint_idx].item()
            p.setJointMotorControl2(self.mano_hand_idx, trans_joint_idx, p.POSITION_CONTROL, targetPosition=cur_trans_joint_state, force=1e20)
        
        for rot_joint_idx in range(3, 6):
            cur_rot_joint_state = glb_tar_rot[rot_joint_idx - 3].item()
            p.setJointMotorControl2(self.mano_hand_idx, rot_joint_idx, p.POSITION_CONTROL, targetPosition=cur_rot_joint_state, force=1e20)

        ## using yeild ? ##
        for i_step in range(self.real_bullet_nn_substeps):
            p.stepSimulation()
            
        
        # revolute_joint_idx = 0 # 
        # for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
        #     # p.setJointMotorControl2(self.mano_hand_idx, cur_joint_idx, p.POSITION_CONTROL, targetPosition=glb_tar_states[revolute_joint_idx].item(), force=1e20) ##  
        #     p.resetJointState(self.mano_hand_idx, cur_joint_idx, glb_tar_states[revolute_joint_idx].item())
        #     revolute_joint_idx = revolute_joint_idx + 1
        
        # for trans_joint_idx in range(3): # position # rot joint idx # 
        #     cur_trans_joint_state = glb_tar_trans[trans_joint_idx].item()
        #     # p.setJointMotorControl2(self.mano_hand_idx, trans_joint_idx, p.POSITION_CONTROL, targetPosition=cur_trans_joint_state, force=1e20)
        #     p.resetJointState(self.mano_hand_idx, trans_joint_idx, cur_trans_joint_state)
        
        # for rot_joint_idx in range(3, 6): ## rot joint idx - 3 ##
        #     cur_rot_joint_state = glb_tar_rot[rot_joint_idx - 3].item()
        #     # p.setJointMotorControl2(self.mano_hand_idx, rot_joint_idx, p.POSITION_CONTROL, targetPosition=cur_rot_joint_state, force=1e20)
        #     p.resetJointState(self.mano_hand_idx, rot_joint_idx, cur_rot_joint_state)
        
        # TODO: cal_done, get_info, after_step
        # action = Rotation.from_rotvec(action.reshape(-1,3)).as_quat()
        # action = MathHelper.flip_quat_by_w(action)
        # world model 

        # for i in range(self.substep):
        #     self.stable_pd.add_torques_by_quat(action)
        #     if 'force' in kargs:
        #         self.add_force(kargs['force'])
        #     self.scene.damped_simulate(1)

        #     if using_yield:
        #         yield self.sim_character.save()
        
        #### the observation contains the current state, target state, and the observation ####
        
        # 
        # character state #
        # self.state = character_state(self.sim_character, self.state if self.recompute_velocity else None, self.dt)
        # # state to observation # # state to observations ##
        # self.observation = state2ob(torch.from_numpy(self.state)).numpy() # observation #  # reward # 
        
        sim_states = self._get_obs() ## get states ##
        self.state = sim_states
        self.observation = sim_states
        
        reward = 0 
        ##  get info here ##   ## cal done ##
        done = self.cal_done(self.state, self.observation)
        info = self.get_info()
        need_rnd_start = self.after_step() # target 
        if need_rnd_start:
            # reset the state ##
            self.reset(frame = -1, set_state = True) ## need reset; need reset the state ##
        
        observation = {
            'state': self.state,
            'target': self.target,
            'observation': self.observation,
            'frame_num': self.counter # 
        }
        # yieldj # 
        # if not using_yield: # for convenient, so that we do not have to capture exception
        #     yield observation, reward, done, info ### observation j
        # else:
        #     return observation, reward, done, info  
        return observation, reward, done, info  
        
    def unnormalize_actions(self, action):
        if self.wnorm:
            action_dim = action.shape[-1]
            n_action = (action * self.delta_std[:action_dim]) + self.delta_mean[:action_dim]
        else:
            n_action = action
        return n_action
    
    # normalize?
    def step(self, action, **kargs): # step #
        # step_generator = self.step_core(action, **kargs)
        step_generator = self.step_core_new_wcontrol(action, **kargs)
        return next(step_generator)
    
    def step_core_new_wcontrol(self, action, using_yield = False, **kargs):
        
        ## get current observation? ##
        ### 
        sim_states = self._get_obs() ## get states ##
        
        self.state = sim_states ## get the observation ## g
        self.observation = sim_states
        
        # cur_state = self._get_mano_target()
        cur_state = self._get_mano_obs() # get observation 3
        # cur_state = self._get_mano_target() # use mano targets 
        cur_state = torch.from_numpy(cur_state).float()
        
        
        if not self.traj_opt:
            action = self.unnormalize_actions(action=action)
            
            # action = self.unnormalize_actions
            
            mano_target_state = self._get_mano_target(frame_idx=self.counter + 1) ##
            action = np.concatenate([mano_target_state[:6] - sim_states[:6], action[6:]], axis=-1)
            
            # policy sampling #
            ### policy sampling ## policy sampling ## ## ##  ## mano target ## ## mano target ## 
            ### get the flashlight ##
            glb_delta_trans = action[:3]
            glb_delta_rot = action[3:6]
            glb_delta_state = action[6: self.bullet_mano_finger_num_joints + 6]
            
            
            
            cur_glb_trans = cur_state[:3]
            cur_glb_rot_vec = cur_state[3:6]
            cur_states = cur_state[6:  6 + self.bullet_mano_finger_num_joints]
            
            glb_tar_trans = cur_glb_trans + glb_delta_trans
            glb_tar_rot = cur_glb_rot_vec + glb_delta_rot 
            glb_tar_states = cur_states + glb_delta_state
        else:
            trans_dim = 3
            rot_dim = 4
            # state_dim = 
            glb_tar_trans_tot = action[:trans_dim * 2]
            glb_tar_rot_tot = action[trans_dim * 2: (trans_dim + rot_dim) * 2]
            glb_tar_states_tot = action[(trans_dim + rot_dim) * 2: ]
            
            rgt_tar_trans, lft_tar_trans = glb_tar_trans_tot[: glb_tar_trans_tot.shape[0] // 2], glb_tar_trans_tot[glb_tar_trans_tot.shape[0] // 2:]
            rgt_tar_rot, lft_tar_rot = glb_tar_rot_tot[: glb_tar_rot_tot.shape[0] // 2], glb_tar_rot_tot[glb_tar_rot_tot.shape[0] // 2:]
            rgt_tar_states, lft_tar_states = glb_tar_states_tot[: glb_tar_states_tot.shape[0] // 2], glb_tar_states_tot[glb_tar_states_tot.shape[0] // 2:]
            
            rgt_tar_rot = rgt_tar_rot[[1, 2, 3, 0]]
            rgt_tar_rot_struct = R.from_quat(rgt_tar_rot)
            
            rgt_tar_rot_vec = rgt_tar_rot_struct.as_euler('zyx', degrees=False)
            rgt_tar_rot_vec = [rgt_tar_rot_vec[2], rgt_tar_rot_vec[1], rgt_tar_rot_vec[0]]
            rgt_tar_rot_vec = np.array(rgt_tar_rot_vec, dtype=np.float32)
            
            lft_tar_rot = lft_tar_rot[[1, 2, 3, 0]]
            lft_tar_rot_struct = R.from_quat(lft_tar_rot)
            
            lft_tar_rot_vec = lft_tar_rot_struct.as_euler('zyx', degrees=False)
            lft_tar_rot_vec = [lft_tar_rot_vec[2], lft_tar_rot_vec[1], lft_tar_rot_vec[0]]
            lft_tar_rot_vec = np.array(lft_tar_rot_vec, dtype=np.float32)
            
            
            # glb_tar_trans = action[:3]
            # glb_tar_rot = action[3:7]
            # glb_tar_states = action[7: self.bullet_mano_finger_num_joints + 7]
            
            # # use_mano_delta_states
            # if self.use_mano_delta_states:
            #     cur_finger_states = cur_state[6: 6 + self.bullet_mano_finger_num_joints]
            #     glb_tar_states = cur_finger_states + glb_tar_states
            
            # glb_tar_rot = glb_tar_rot[[1, 2, 3, 0]]
            # glb_tar_rot_struct = R.from_quat(glb_tar_rot)
            
            # glb_tar_rot_vec = glb_tar_rot_struct.as_euler('zyx', degrees=False)
            # glb_tar_rot = [glb_tar_rot_vec[2], glb_tar_rot_vec[1], glb_tar_rot_vec[0]]
            # glb_tar_rot = np.array(glb_tar_rot, dtype=np.float32)

            # print(f"mano glb_rot: {glb_tar_rot}, mano glb_trans: {glb_tar_trans}")

        ## lft tar rot vec ##
        
        revolute_joint_idx = 0 # set revoluete joit states  # 
        for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
            p.setJointMotorControl2(self.mano_hand_idx, cur_joint_idx, p.POSITION_CONTROL, targetPosition=rgt_tar_states[revolute_joint_idx].item(), force=1e20)
            
            # p.resetJointState(self.mano_hand_idx, cur_joint_idx, glb_tar_states[revolute_joint_idx].item())
            revolute_joint_idx = revolute_joint_idx + 1
        
        for trans_joint_idx in range(3): # position # rot joint idx # 
            cur_trans_joint_state = rgt_tar_trans[trans_joint_idx].item()
            p.setJointMotorControl2(self.mano_hand_idx, trans_joint_idx, p.POSITION_CONTROL, targetPosition=cur_trans_joint_state, force=1e20)
            
            # p.resetJointState(self.mano_hand_idx, trans_joint_idx, cur_trans_joint_state)
        
        for rot_joint_idx in range(3, 6): ## rot joint idx - 3 ##
            cur_rot_joint_state = rgt_tar_rot_vec[rot_joint_idx - 3].item()
            p.setJointMotorControl2(self.mano_hand_idx, rot_joint_idx, p.POSITION_CONTROL, targetPosition=cur_rot_joint_state, force=1e20)
            # p.resetJointState(self.mano_hand_idx, rot_joint_idx, cur_rot_joint_state)

        revolute_joint_idx = 0 # set revoluete joit states  # 
        for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
            p.setJointMotorControl2(self.left_mano_hand_idx, cur_joint_idx, p.POSITION_CONTROL, targetPosition=lft_tar_states[revolute_joint_idx].item(), force=1e20)
            # p.resetJointState(self.mano_hand_idx, cur_joint_idx, glb_tar_states[revolute_joint_idx].item())
            revolute_joint_idx = revolute_joint_idx + 1
        
        for trans_joint_idx in range(3): # position # rot joint idx # 
            cur_trans_joint_state = lft_tar_trans[trans_joint_idx].item()
            p.setJointMotorControl2(self.left_mano_hand_idx, trans_joint_idx, p.POSITION_CONTROL, targetPosition=cur_trans_joint_state, force=1e20)
            
            # p.resetJointState(self.mano_hand_idx, trans_joint_idx, cur_trans_joint_state)
        
        for rot_joint_idx in range(3, 6): ## rot joint idx - 3 ##
            cur_rot_joint_state = lft_tar_rot_vec[rot_joint_idx - 3].item()
            p.setJointMotorControl2(self.left_mano_hand_idx, rot_joint_idx, p.POSITION_CONTROL, targetPosition=cur_rot_joint_state, force=1e20)
            # p.resetJointState(self.mano_hand_idx, rot_joint_idx, cur_rot_joint_state)

        ## using yeild ? ##
        for i_step in range(self.real_bullet_nn_substeps):
            p.stepSimulation()
        
        ## pd control ## ## step pd control ##
        cur_step_pd_control = np.concatenate(
            [glb_tar_trans_tot, glb_tar_rot_tot, glb_tar_states_tot], axis=-1
        )
        
        
        ### recover the original state ###
        # revolute_joint_idx = 0 # set revoluete joit states  # 
        # for cur_joint_idx in range(self.st_finger_joints_tates, self.bullet_mano_num_joints):
        #     # p.setJointMotorControl2(self.mano_hand_idx, cur_joint_idx, p.POSITION_CONTROL, targetPosition=glb_tar_states[revolute_joint_idx].item(), force=1e20)
            
        #     p.resetJointState(self.mano_hand_idx, cur_joint_idx, glb_tar_states[revolute_joint_idx].item())
        #     revolute_joint_idx = revolute_joint_idx + 1
        
        # for trans_joint_idx in range(3): # position # rot joint idx # 
        #     cur_trans_joint_state = glb_tar_trans[trans_joint_idx].item()
        #     # p.setJointMotorControl2(self.mano_hand_idx, trans_joint_idx, p.POSITION_CONTROL, targetPosition=cur_trans_joint_state, force=1e20)
            
        #     p.resetJointState(self.mano_hand_idx, trans_joint_idx, cur_trans_joint_state)
        
        # for rot_joint_idx in range(3, 6): ## rot joint idx - 3 ##
        #     cur_rot_joint_state = glb_tar_rot[rot_joint_idx - 3].item()
        #     # p.setJointMotorControl2(self.mano_hand_idx, rot_joint_idx, p.POSITION_CONTROL, targetPosition=cur_rot_joint_state, force=1e20)
        #     p.resetJointState(self.mano_hand_idx, rot_joint_idx, cur_rot_joint_state)
        ### recover the original state ###
        
        
        # p.stepSimulation() #
        # cur_step_pd_control = cur_state.detach().cpu().numpy() #
        
        sim_states = self._get_obs() ## get states #### get obs ##
        self.state = sim_states ### get states and observations ##
        self.observation = sim_states
        
        reward = 0 
        ## ## 
        done = self.cal_done(self.state, self.observation)
        info = self.get_info()
        need_rnd_start  = self.after_step()
        if need_rnd_start:
            self.reset(frame = -1, set_state = True)
        
        
        observation = { # observations 
            'state': self.state,
            'target': self.target,
            'observation': self.observation,
            'frame_num': self.counter # 
        }
        # yield # # frame num #
        # if not using_yield: # for convenient, so that we do not have to capture exception
        #     yield observation, cur_step_pd_control, reward, done, info ### observation j
        # else:
        return observation, cur_step_pd_control, reward, done, info  
        
    