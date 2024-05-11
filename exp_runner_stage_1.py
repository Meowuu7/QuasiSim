import os
import time
import logging
import argparse
import numpy as np
# import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = None
    pass
from shutil import copyfile
# from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, BendingNetwork
import models.data_utils_torch as data_utils
# import models.dyn_model_utils as dyn_utils
import torch.nn as nn
# import models.renderer_def_multi_objs as render_utils
import models.fields as fields

from torch.distributions.categorical import Categorical

try:
    import redmax_py as redmax
except:
    pass
import open3d as o3d
import models.dyn_model_act as dyn_model_act
import models.dyn_model_act_v2 as dyn_model_act_mano
from scipy.spatial.transform import Rotation as R
import traceback

import pickle as pkl



class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')


        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        
        
            
        
        self.base_exp_dir = self.conf['general.base_exp_dir']
        
        # local_exp_dir = "/data2/xueyi/quasisim/exp/"
        local_exp_dir = "/data/xueyi/quasisim/exp"
        if os.path.exists(local_exp_dir):
            self.base_exp_dir = local_exp_dir
        
        
        print(f"self.base_exp_dir:", self.base_exp_dir)
        self.base_exp_dir = self.base_exp_dir + f"_reverse_value_totviews_tag_{self.conf['general.tag']}"
        os.makedirs(self.base_exp_dir, exist_ok=True)
        # self.dataset = Dataset(self.conf['dataset'])
        
        
        self.n_timesteps = self.conf['model.n_timesteps']

        
        self.iter_step = 0

        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        
        self.use_bending_network = True
        # use_split_network
        self.use_selector = True


        # Weights # 
        self.igr_weight = self.conf.get_float('train.igr_weight') 
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue # 
        self.mode = mode
        self.model_list = []
        self.writer = None


        self.bending_latent_size = self.conf['model.bending_network']['bending_latent_size']


        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        
        # self.use_bending_network = self.conf['model.use_bending_network']
        # # bending network size #
        # if self.use_bending_network:  # add the bendingnetwork #
        self.bending_network = BendingNetwork(**self.conf['model.bending_network']).to(self.device)
        
        
        self.use_split_network = self.conf.get_bool('model.use_split_network', False)
        if self.use_split_network:
            self.bending_network.set_split_bending_network()
        self.bending_network.n_timesteps = self.n_timesteps
        
        
        self.extract_delta_mesh = self.conf['model.extract_delta_mesh']
        
        
        
        # self.use_passive_nets = self.conf['model.use_passive_nets']
        if 'model.bending_net_type' in self.conf:
            self.bending_net_type = self.conf['model.bending_net_type']
        else:
            self.bending_net_type = "pts_def"
        
        if 'model.train_multi_seqs' in self.conf and self.conf['model.train_multi_seqs']:
            self.rhand_verts, self.hand_faces, self.obj_faces, self.obj_normals, self.ts_to_contact_pts, self.hand_verts = self.load_active_passive_timestep_to_mesh_multi_seqs()
            self.train_multi_seqs = True
            self.nn_instances = len(self.rhand_verts)
        else:
            self.train_multi_seqs = False
            self.nn_instances = 1
            
        if 'model.minn_dist_threshold' in self.conf:
            self.minn_dist_threshold = self.conf['model.minn_dist_threshold']
        else:
            self.minn_dist_threshold = 0.05
            
        if 'model.optimize_with_intermediates' in self.conf:
            self.optimize_with_intermediates = self.conf['model.optimize_with_intermediates']
        else:
            self.optimize_with_intermediates = False
            
        if 'model.no_friction_constraint' in self.conf:
            self.no_friction_constraint = self.conf['model.no_friction_constraint']
        else:
            self.no_friction_constraint = False
            
        if 'model.optimize_active_object' in self.conf:
            self.optimize_active_object = self.conf['model.optimize_active_object']
        else:
            self.optimize_active_object = False
            
        if 'model.optimize_glb_transformations' in self.conf:
            self.optimize_glb_transformations = self.conf['model.optimize_glb_transformations']
        else:
            self.optimize_glb_transformations = False
            
        if 'model.with_finger_tracking_loss' in self.conf:
            self.with_finger_tracking_loss = self.conf['model.with_finger_tracking_loss']
        else:
            self.with_finger_tracking_loss = True


        if 'model.finger_cd_loss' in self.conf:
            self.finger_cd_loss_coef = self.conf['model.finger_cd_loss']
        else:
            self.finger_cd_loss_coef = 0.

        if 'model.finger_tracking_loss' in self.conf:
            self.finger_tracking_loss_coef = self.conf['model.finger_tracking_loss']
        else:
            self.finger_tracking_loss_coef = 0.
        
        if 'model.tracking_loss' in self.conf:
            self.tracking_loss_coef = self.conf['model.tracking_loss']
        else:
            self.tracking_loss_coef = 0.
            
        if 'model.penetrating_depth_penalty' in self.conf:
            self.penetrating_depth_penalty_coef = self.conf['model.penetrating_depth_penalty']
        else:
            self.penetrating_depth_penalty_coef = 0.
            
            
        if 'model.ragged_dist' in self.conf:
            self.ragged_dist_coef = self.conf['model.ragged_dist']
        else:
            self.ragged_dist_coef = 1.


        if 'mode.load_only_glb' in self.conf:
            self.load_only_glb = self.conf['model.load_only_glb']
        else:
            self.load_only_glb = False
            
        # optimize_rules; optimize_robot # 
        if 'model.optimize_robot' in self.conf:
            self.optimize_robot = self.conf['model.optimize_robot']
        else:
            self.optimize_robot = True
            
        if 'model.optimize_rules' in self.conf:
            self.optimize_rules = self.conf['model.optimize_rules']
        else:
            self.optimize_rules = False
            
            
        if 'model.optimize_expanded_pts' in self.conf:
            self.optimize_expanded_pts = self.conf['model.optimize_expanded_pts']
        else:
            self.optimize_expanded_pts = True
            
        if 'model.optimize_expanded_ragged_pts' in self.conf:
            self.optimize_expanded_ragged_pts = self.conf['model.optimize_expanded_ragged_pts']
        else:
            self.optimize_expanded_ragged_pts = False
        
        if 'model.add_delta_state_constraints' in self.conf:
            self.add_delta_state_constraints = self.conf['model.add_delta_state_constraints']
        else:
            self.add_delta_state_constraints = True
            
        # 
        if 'model.train_actions_with_states' in self.conf:
            self.train_actions_with_states = self.conf['model.train_actions_with_states']
        else:
            self.train_actions_with_states = False
            
            
        if 'model.train_with_forces_to_active' in self.conf:
            self.train_with_forces_to_active = self.conf['model.train_with_forces_to_active']
        else:
            self.train_with_forces_to_active = False
            
            
        if 'model.loss_weight_diff_states' in self.conf:
            self.loss_weight_diff_states = self.conf['model.loss_weight_diff_states']
        else:
            self.loss_weight_diff_states = 1.
            
        if 'model.loss_tangential_diff_coef' in self.conf:
            self.loss_tangential_diff_coef = float(self.conf['model.loss_tangential_diff_coef'])
        else:
            self.loss_tangential_diff_coef = 1.
            
        if 'model.use_penalty_based_friction' in self.conf:
            self.use_penalty_based_friction = self.conf['model.use_penalty_based_friction']
        else:
            self.use_penalty_based_friction = False
            
        if 'model.use_disp_based_friction' in self.conf:
            self.use_disp_based_friction = self.conf['model.use_disp_based_friction']
        else:
            self.use_disp_based_friction = False 
            
        if 'model.use_sqrt_dist' in self.conf:
            self.use_sqrt_dist = self.conf['model.use_sqrt_dist']
        else:
            self.use_sqrt_dist = False
            
        if 'model.reg_loss_coef' in self.conf:
            self.reg_loss_coef = float(self.conf['model.reg_loss_coef'])
        else:
            self.reg_loss_coef = 0.
            
        if 'model.contact_maintaining_dist_thres' in self.conf:
            self.contact_maintaining_dist_thres = float(self.conf['model.contact_maintaining_dist_thres'])
        else:
            self.contact_maintaining_dist_thres = 0.1
            
        if 'model.penetration_proj_k_to_robot' in self.conf:
            self.penetration_proj_k_to_robot = float(self.conf['model.penetration_proj_k_to_robot'])
        else:
            self.penetration_proj_k_to_robot = 0.0
            
        if 'model.use_mano_inputs' in self.conf:
            self.use_mano_inputs = self.conf['model.use_mano_inputs']
        else:
            self.use_mano_inputs = False
            
        
        if 'model.use_split_params' in self.conf:
            self.use_split_params = self.conf['model.use_split_params']
        else:
            self.use_split_params = False
            
        
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
        
        if 'model.robot_actions_diff_coef' in self.conf:
            self.robot_actions_diff_coef = self.conf['model.robot_actions_diff_coef']
        else:
            self.robot_actions_diff_coef = 0.1
        
        if 'model.use_sdf_as_contact_dist' in self.conf:
            self.use_sdf_as_contact_dist = self.conf['model.use_sdf_as_contact_dist']
        else:
            self.use_sdf_as_contact_dist = False
        
        if 'model.use_contact_dist_as_sdf' in self.conf:
            self.use_contact_dist_as_sdf = self.conf['model.use_contact_dist_as_sdf']
        else:
            self.use_contact_dist_as_sdf = False
        
        if 'model.use_same_contact_spring_k' in self.conf:
            self.use_same_contact_spring_k = self.conf['model.use_same_contact_spring_k']
        else:
            self.use_same_contact_spring_k = False
        
        if 'model.minn_dist_threshold_robot_to_obj' in self.conf:
            self.minn_dist_threshold_robot_to_obj = float(self.conf['model.minn_dist_threshold_robot_to_obj'])
        else:
            self.minn_dist_threshold_robot_to_obj = 0.0
            
        if 'model.obj_mass' in self.conf:
            self.obj_mass = float(self.conf['model.obj_mass'])
        else:
            self.obj_mass = 100.0
            
        if 'model.diff_hand_tracking_coef' in self.conf:
            self.diff_hand_tracking_coef = float(self.conf['model.diff_hand_tracking_coef'])
        else: # 
            self.diff_hand_tracking_coef = 0.0
            
        if 'model.use_mano_hand_for_test' in self.conf:
            self.use_mano_hand_for_test = self.conf['model.use_mano_hand_for_test']
        else:
            self.use_mano_hand_for_test = False
        
        if 'model.train_residual_friction' in self.conf:
            self.train_residual_friction = self.conf['model.train_residual_friction']
        else:
            self.train_residual_friction = False
            
        if 'model.use_LBFGS' in self.conf:
            self.use_LBFGS = self.conf['model.use_LBFGS']
        else:
            self.use_LBFGS = False
            
        if 'model.use_optimizable_params' in self.conf:
            self.use_optimizable_params = self.conf['model.use_optimizable_params']
        else:
            self.use_optimizable_params = False
             
        if 'model.penetration_determining' in self.conf:
            self.penetration_determining = self.conf['model.penetration_determining']
        else:
            self.penetration_determining = "sdf_of_canon"
            
        if 'model.sdf_sv_fn' in self.conf:
            self.sdf_sv_fn = self.conf['model.sdf_sv_fn']
        else:
            self.sdf_sv_fn = None
            
        if 'model.loss_scale_coef' in self.conf:
            self.loss_scale_coef = float(self.conf['model.loss_scale_coef'])
        else:
            self.loss_scale_coef = 1.0
        
        if 'model.penetration_proj_k_to_robot_friction' in self.conf:
            self.penetration_proj_k_to_robot_friction = float(self.conf['model.penetration_proj_k_to_robot_friction'])
        else:
            self.penetration_proj_k_to_robot_friction = self.penetration_proj_k_to_robot
            
        if 'model.retar_only_glb' in self.conf:
            self.retar_only_glb = self.conf['model.retar_only_glb']
        else:
            self.retar_only_glb = False
            
        if 'model.optim_sim_model_params_from_mano' in self.conf:
            self.optim_sim_model_params_from_mano = self.conf['model.optim_sim_model_params_from_mano']
        else:
            self.optim_sim_model_params_from_mano = False
            
            
        # opt_robo_states, opt_robo_glb_trans, opt_robo_glb_rot #
        if 'model.opt_robo_states' in self.conf:
            self.opt_robo_states = self.conf['model.opt_robo_states']
        else:
            self.opt_robo_states = True
            
        if 'model.opt_robo_glb_trans' in self.conf:
            self.opt_robo_glb_trans = self.conf['model.opt_robo_glb_trans']
        else:
            self.opt_robo_glb_trans = False
            
        if 'model.opt_robo_glb_rot' in self.conf:
            self.opt_robo_glb_rot = self.conf['model.opt_robo_glb_rot'] 
        else:
            self.opt_robo_glb_rot = False
            
        # motion_reg_loss_coef
        
        if 'model.motion_reg_loss_coef' in self.conf:
            self.motion_reg_loss_coef = self.conf['model.motion_reg_loss_coef'] 
        else:
            self.motion_reg_loss_coef = 1.0
            
        if 'model.drive_robot' in self.conf:
            self.drive_robot = self.conf['model.drive_robot']
        else:
            self.drive_robot = 'states'
            
        if 'model.use_scaled_urdf' in self.conf:
            self.use_scaled_urdf =self.conf['model.use_scaled_urdf']
        else:
            self.use_scaled_urdf = False
            
        if 'model.window_size' in self.conf:
            self.window_size = self.conf['model.window_size']
        else:
            self.window_size = 60
            
        if 'model.use_taco' in self.conf:
            self.use_taco = self.conf['model.use_taco']
        else:
            self.use_taco = False
            
        if 'model.ang_vel_damping' in self.conf:
            self.ang_vel_damping = float(self.conf['model.ang_vel_damping'])
        else:
            self.ang_vel_damping = 0.0
            
        if 'model.drive_glb_delta' in self.conf:
            self.drive_glb_delta = self.conf['model.drive_glb_delta']
        else:
            self.drive_glb_delta = False
            
        if 'model.fix_obj' in self.conf:
            self.fix_obj = self.conf['model.fix_obj']
        else:
            self.fix_obj = False
            
        if 'model.diff_reg_coef' in self.conf:
            self.diff_reg_coef = self.conf['model.diff_reg_coef']
        else:
            self.diff_reg_coef = 0.01
            
        if 'model.use_damping_params_vel' in self.conf:
            self.use_damping_params_vel = self.conf['model.use_damping_params_vel']
        else:
            self.use_damping_params_vel = False
        
        if 'train.ckpt_sv_freq' in self.conf:
            self.ckpt_sv_freq = int(self.conf['train.ckpt_sv_freq'])
        else:
            self.ckpt_sv_freq = 100
        
        
        if 'model.optm_alltime_ks' in self.conf:
            self.optm_alltime_ks = self.conf['model.optm_alltime_ks']
        else:
            self.optm_alltime_ks = False
            
        if 'model.retar_dense_corres' in self.conf:
            self.retar_dense_corres = self.conf['model.retar_dense_corres']
        else:
            self.retar_dense_corres = False
        
        
        if 'model.retar_delta_glb_trans' in self.conf:
            self.retar_delta_glb_trans = self.conf['model.retar_delta_glb_trans']
        else:
            self.retar_delta_glb_trans = False
            
        if 'model.use_multi_stages' in self.conf:
            self.use_multi_stages = self.conf['model.use_multi_stages']
        else:
            self.use_multi_stages = False
            
        if 'model.seq_start_idx' in self.conf:
            self.seq_start_idx = self.conf['model.seq_start_idx']
        else:
            self.seq_start_idx = 40
            
        if 'model.obj_sdf_fn' in self.conf:
            self.obj_sdf_fn = self.conf['model.obj_sdf_fn']
        else:
            self.obj_sdf_fn = ""
        
        if 'model.kinematic_mano_gt_sv_fn' in self.conf:
            self.kinematic_mano_gt_sv_fn = self.conf['model.kinematic_mano_gt_sv_fn']
        else:
            self.kinematic_mano_gt_sv_fn = ""
        
        if 'model.scaled_obj_mesh_fn' in self.conf:
            self.scaled_obj_mesh_fn = self.conf['model.scaled_obj_mesh_fn']
        else:
            self.scaled_obj_mesh_fn = ""
        
        if 'model.ckpt_fn' in self.conf:
            self.ckpt_fn = self.conf['model.ckpt_fn']
        else:
            self.ckpt_fn = ""
        
        if 'model.load_optimized_init_transformations' in self.conf:
            self.load_optimized_init_transformations = self.conf['model.load_optimized_init_transformations']
        else:
            self.load_optimized_init_transformations = ""
            
        if 'model.optimize_dyn_actions' in self.conf:
            self.optimize_dyn_actions = self.conf['model.optimize_dyn_actions']
        else:
            self.optimize_dyn_actions = False
            
        if 'model.load_optimized_obj_transformations' in self.conf:
            self.load_optimized_obj_transformations = self.conf['model.load_optimized_obj_transformations']
        else:
            self.load_optimized_obj_transformations = None
            
        if 'model.train_pointset_acts_via_deltas' in self.conf:
            self.train_pointset_acts_via_deltas = self.conf['model.train_pointset_acts_via_deltas']
        else:
            self.train_pointset_acts_via_deltas = False
            
            
        if 'model.drive_pointset' in self.conf:
            self.drive_pointset = self.conf['model.drive_pointset']
        else:
            self.drive_pointset = "states"
            
        if 'model.optimize_anchored_pts' in self.conf:
            self.optimize_anchored_pts = self.conf['model.optimize_anchored_pts']
        else:
            self.optimize_anchored_pts = True
            
        if 'model.optimize_pointset_motion_only' in self.conf:
            self.optimize_pointset_motion_only = self.conf['model.optimize_pointset_motion_only']
        else:
            self.optimize_pointset_motion_only = True
            
        # print(f"optimize_dyn_actions: {self.optimize_dyn_actions}")
        
        
        #### get pointset parameters ###
        if 'model.pointset_expand_factor' in self.conf:
            self.pointset_expand_factor = self.conf['model.pointset_expand_factor']
        else:
            self.pointset_expand_factor = 0.1
        
        if 'model.pointset_nn_expand_pts' in self.conf:
            self.pointset_nn_expand_pts = self.conf['model.pointset_nn_expand_pts']
        else:
            self.pointset_nn_expand_pts = 10
        
        print(f"[Settings] Setting pointset_expand_factor to {self.pointset_expand_factor}, pointset_nn_expand_pts to {self.pointset_nn_expand_pts}")
        

            
        if 'dataset.obj_idx' in self.conf:
            print(f"dataset.obj_idx:", self.conf['dataset.obj_idx'])
            self.obj_idx = self.conf['dataset.obj_idx']
            
            # ###### only for the grab dataset only currently ########
            # GRAB_data_root = "/data1/xueyi/GRAB_extracted_test/train"
            # # /data/xueyi/GRAB/GRAB_extracted_test/train/102_obj.npy
            # if not os.path.exists(GRAB_data_root):
            #     GRAB_data_root = "/data/xueyi/GRAB/GRAB_extracted_test/train"
                
                
            GRAB_data_root = "data/grab"
            GRAB_data_root = os.path.join(GRAB_data_root, f"{self.obj_idx}")
            
            
            
            self.obj_sdf_fn = os.path.join(GRAB_data_root, f"{self.obj_idx}_obj.npy")
            self.kinematic_mano_gt_sv_fn =  os.path.join(GRAB_data_root, f"{self.obj_idx}_sv_dict.npy")
            self.scaled_obj_mesh_fn = os.path.join(GRAB_data_root, f"{self.obj_idx}_obj.obj")
            # self.ckpt_fn =  self.conf['model.ckpt_fn']
            # self.load_optimized_init_transformations =  self.conf['model.load_optimized_init_transformations']
            
            print(f"obj_sdf_fn:", self.obj_sdf_fn)
            print(f"kinematic_mano_gt_sv_fn:", self.kinematic_mano_gt_sv_fn)
            print(f"scaled_obj_mesh_fn:", self.scaled_obj_mesh_fn)
            
        
            
            
        self.minn_init_passive_mesh = None
        self.maxx_init_passive_mesh = None
        
        
        self.mano_nn_substeps = 1
        
        self.canon_passive_obj_verts = None
        self.canon_passive_obj_normals = None

        if self.bending_net_type == "active_force_field_v18":
            self.other_bending_network = fields.BendingNetworkActiveForceFieldForwardLagV18(**self.conf['model.bending_network'], nn_instances=self.nn_instances, minn_dist_threshold=self.minn_dist_threshold).to(self.device)
            
            
            if mode in ["train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_grab", "train_point_set", "train_sparse_retar", "train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab", "train_point_set_retar", "train_point_set_retar_pts", "train_finger_kinematics_retargeting_arctic_twohands", "train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_arctic_twohands", "train_real_robot_actions_from_mano_model_rules_shadowhand", "train_redmax_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab", "train_dyn_mano_model", "train_dyn_mano_model_wreact"]:
                
                
                if mode in ['train_finger_kinematics_retargeting_arctic_twohands', 'train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_arctic_twohands']:
                    self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_twohands_arctic()
                else:
                    if self.use_taco:
                        self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_v3_taco()
                    else:
                        self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_v3()
                
                if self.conf['model.penetration_determining'] == "ball_primitives":
                    self.center_verts, self.ball_r = self.get_ball_primitives()
                
                
                self.other_bending_network.canon_passive_obj_verts = self.obj_verts
                self.other_bending_network.canon_passive_obj_normals = self.obj_normals
                
                self.canon_passive_obj_verts = self.obj_verts
                self.canon_passive_obj_normals = self.obj_normals

                
                # tot_obj_quat, tot_reversed_obj_rot_mtx #
                ''' Load passive object's SDF '''
                self.obj_sdf_fn = self.obj_sdf_fn  ## load passive object's sdf
                self.other_bending_network.sdf_space_center = self.sdf_space_center
                self.other_bending_network.sdf_space_scale = self.sdf_space_scale
                self.obj_sdf = np.load(self.obj_sdf_fn, allow_pickle=True)
                self.sdf_res = self.obj_sdf.shape[0]
                self.other_bending_network.obj_sdf = self.obj_sdf
                self.other_bending_network.sdf_res = self.sdf_res
                
                
                if self.conf['model.penetration_determining'] == "sdf_of_canon":
                    print(f"Setting the penetration determining method to sdf_of_canon")
                    self.other_bending_network.penetration_determining = "sdf_of_canon" 
                elif self.conf['model.penetration_determining'] == 'plane_primitives':
                    print(f"setting the penetration determining method to plane_primitives with maxx_xyz: {self.maxx_init_passive_mesh}, minn_xyz: {self.minn_init_passive_mesh}")
                    self.other_bending_network.penetration_determining = "plane_primitives" #
                elif self.conf['model.penetration_determining'] == 'ball_primitives':
                    print(f"Setting the penetration determining method to ball_primitives with ball_r: {self.ball_r}, center: {self.center_verts}")
                    self.other_bending_network.penetration_determining = "ball_primitives" #
                    self.other_bending_network.center_verts = self.center_verts
                    self.other_bending_network.ball_r = self.ball_r ## get the ball primitives here? ##
                else:
                    raise NotImplementedError(f"penetration determining method {self.conf['model.penetration_determining']} not implemented")
                
                
            elif mode in ["train_dyn_mano_model", "train_dyn_mano_model_wreact"]:
                self.load_active_passive_timestep_to_mesh()
                self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh_v3()
                self.obj_sdf_grad = None
            else:
                
                if not self.train_multi_seqs:
                    self.timestep_to_passive_mesh, self.timestep_to_active_mesh, self.timestep_to_passive_mesh_normals = self.load_active_passive_timestep_to_mesh()
                    
                    
                    self.other_bending_network.sdf_space_center = self.sdf_space_center
                    self.other_bending_network.sdf_space_scale = self.sdf_space_scale
                    self.other_bending_network.obj_sdf = self.obj_sdf # 
                    self.other_bending_network.sdf_res = self.sdf_res #
        else:
            raise ValueError(f"Unrecognized bending net type: {self.bending_net_type}")


        if self.maxx_init_passive_mesh is None and self.minn_init_passive_mesh is None:
            self.calculate_collision_geometry_bounding_boxes()
            
            ###### initialize the dyn model ######
        for i_time_idx in range(self.n_timesteps):
            self.other_bending_network.timestep_to_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).cuda()
            self.other_bending_network.timestep_to_point_accs[i_time_idx] = torch.zeros((3,), dtype=torch.float32).cuda()
            self.other_bending_network.timestep_to_total_def[i_time_idx] = torch.zeros((3,), dtype=torch.float32).cuda()
            self.other_bending_network.timestep_to_angular_vel[i_time_idx] = torch.zeros((3,), dtype=torch.float32).cuda()
            self.other_bending_network.timestep_to_quaternion[i_time_idx] = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            self.other_bending_network.timestep_to_torque[i_time_idx] = torch.zeros((3,), dtype=torch.float32).cuda()
            

        ### set initial transformations ###
        if mode in ["train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_grab", "train_point_set", "train_point_set_retar", "train_point_set_retar_pts", "train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab", "train_finger_kinematics_retargeting_arctic_twohands", "train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_arctic_twohands", "train_real_robot_actions_from_mano_model_rules_shadowhand", "train_redmax_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab", "train_dyn_mano_model_wreact"] and self.bending_net_type == "active_force_field_v18":
            self.other_bending_network.timestep_to_total_def[0] = self.object_transl[0]
            self.other_bending_network.timestep_to_quaternion[0] = self.tot_obj_quat[0]
            self.other_bending_network.timestep_to_optimizable_offset[0] = self.object_transl[0].detach()
            self.other_bending_network.timestep_to_optimizable_quaternion[0] = self.tot_obj_quat[0].detach()
            self.other_bending_network.timestep_to_optimizable_rot_mtx[0] = self.tot_reversed_obj_rot_mtx[0].detach()
            self.other_bending_network.timestep_to_optimizable_total_def[0] = self.object_transl[0].detach()
            
            if self.fix_obj:
                print(f"fix_obj = True")
                for i_fr in range(self.object_transl.size(0)):
                    self.other_bending_network.timestep_to_total_def[i_fr] = self.object_transl[i_fr]
                    self.other_bending_network.timestep_to_quaternion[i_fr] = self.tot_obj_quat[i_fr]
                    self.other_bending_network.timestep_to_optimizable_offset[i_fr] = self.object_transl[i_fr].detach()
                    self.other_bending_network.timestep_to_optimizable_quaternion[i_fr] = self.tot_obj_quat[i_fr].detach()
                    self.other_bending_network.timestep_to_optimizable_rot_mtx[i_fr] = self.tot_reversed_obj_rot_mtx[i_fr].detach()
                    self.other_bending_network.timestep_to_optimizable_total_def[i_fr] = self.object_transl[i_fr].detach()
            


        self.calculate_obj_inertia()

        self.other_bending_network.use_penalty_based_friction = self.use_penalty_based_friction
        self.other_bending_network.use_disp_based_friction = self.use_disp_based_friction
        self.other_bending_network.use_sqrt_dist = self.use_sqrt_dist
        self.other_bending_network.contact_maintaining_dist_thres = self.contact_maintaining_dist_thres
        self.other_bending_network.penetration_proj_k_to_robot = self.penetration_proj_k_to_robot
        self.other_bending_network.use_split_params = self.use_split_params
        # self.other_bending_network.use_split_params = self.use_split_params
        self.other_bending_network.use_sqr_spring_stiffness = self.use_sqr_spring_stiffness
        self.other_bending_network.use_pre_proj_frictions = self.use_pre_proj_frictions
        self.other_bending_network.use_static_mus = self.use_static_mus
        self.other_bending_network.contact_friction_static_mu = self.contact_friction_static_mu
        self.other_bending_network.debug = self.debug
        self.obj_sdf_grad = None
        self.other_bending_network.obj_sdf_grad = self.obj_sdf_grad ## set obj_sdf #
        self.other_bending_network.use_sdf_as_contact_dist = self.use_sdf_as_contact_dist
        self.other_bending_network.use_contact_dist_as_sdf = self.use_contact_dist_as_sdf
        self.other_bending_network.minn_dist_threshold_robot_to_obj = self.minn_dist_threshold_robot_to_obj
        self.other_bending_network.use_same_contact_spring_k = self.use_same_contact_spring_k
        self.other_bending_network.I_ref = self.I_ref
        self.other_bending_network.I_inv_ref = self.I_inv_ref
        self.other_bending_network.obj_mass = self.obj_mass
        
        # self.maxx_init_passive_mesh, self.minn_init_passive_mesh
        self.other_bending_network.maxx_init_passive_mesh = self.maxx_init_passive_mesh
        self.other_bending_network.minn_init_passive_mesh = self.minn_init_passive_mesh # ### init maximum passive meshe #
        self.other_bending_network.train_residual_friction = self.train_residual_friction
        ### use optimizable params ###
        self.other_bending_network.use_optimizable_params = self.use_optimizable_params
        self.other_bending_network.penetration_proj_k_to_robot_friction = self.penetration_proj_k_to_robot_friction
        self.other_bending_network.ang_vel_damping = self.ang_vel_damping
        self.other_bending_network.use_damping_params_vel = self.use_damping_params_vel ## use_damping_params
        self.other_bending_network.optm_alltime_ks = self.optm_alltime_ks
        
        # self.ts_to_mesh_offset = self.load_calcu_timestep_to_passive_mesh_offset()
        # self.ts_to_mesh_offset_for_opt = self.load_calcu_timestep_to_passive_mesh_offset()


        params_to_train += list(self.other_bending_network.parameters()) # 
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        
        if len(self.ckpt_fn) > 0:
            cur_ckpt_fn = self.ckpt_fn
            self.load_checkpoint_via_fn(cur_ckpt_fn)
            # if self.train_multi_seqs:
            #     damping_coefs = self.other_bending_network.damping_constant[0].weight.data
            #     spring_ks_values = self.other_bending_network.spring_ks_values[0].weight.data
            # else:
            damping_coefs = self.other_bending_network.damping_constant.weight.data
            spring_ks_values = self.other_bending_network.spring_ks_values.weight.data
            print(f"loaded ckpt has damping_coefs: {damping_coefs}, and spring ks values: {spring_ks_values}")
            try:
                friction_spring_ks = self.other_bending_network.spring_friction_ks_values.weight.data
                print(f"friction_spring_ks:")
                print(friction_spring_ks)
                
                obj_inertia_val = self.other_bending_network.obj_inertia.weight.data
                optimizable_obj_mass = self.other_bending_network.optimizable_obj_mass.weight.data
                print(f"obj_inertia_val: {obj_inertia_val ** 2}, optimizable_obj_mass: {optimizable_obj_mass ** 2}")
            except:
                pass
            
            time_constant = self.other_bending_network.time_constant.weight.data 
            print(f"time_constant: {time_constant}")
        
        ''' set gravity '''
        ### gravity ### #
        self.gravity_acc = 9.8
        self.gravity_dir = torch.tensor([0., 0., -1]).float().cuda()
        self.passive_obj_mass = 1.

        if not self.bending_net_type == "active_force_field_v13":
            #### init passive mesh center and I_ref # # 
            self.init_passive_mesh_center, self.I_ref = self.calculate_passive_mesh_center_intertia()
            self.inv_I_ref = torch.linalg.inv(self.I_ref)
            self.other_bending_network.passive_obj_inertia = self.I_ref
            self.other_bending_network.passive_obj_inertia_inv = self.inv_I_ref

    def get_robohand_type_from_conf_fn(self, conf_model_fn):
        if "redmax" in conf_model_fn:
            hand_type = "redmax_hand"
        elif "shadow" in conf_model_fn:
            hand_type = "shadow_hand"
        else:
            raise ValueError(f"Cannot identify robot hand type from the conf_model file: {conf_model_fn}")
        return hand_type
        
    def calculate_passive_mesh_center_intertia(self, ): # passive 
        # self.timestep_to_passive_mesh # ## passvie mesh center ### 
        init_passive_mesh = self.timestep_to_passive_mesh[0] ### nn_mesh_pts x 3 ###
        init_passive_mesh_center = torch.mean(init_passive_mesh, dim=0) ### init_center ###
        per_vert_mass = self.passive_obj_mass / float(init_passive_mesh.size(0))
        # (center to the vertex)
        # assume the mass is uniformly distributed across all vertices ##
        I_ref = torch.zeros((3, 3), dtype=torch.float32).cuda()
        for i_v in range(init_passive_mesh.size(0)):
            cur_vert = init_passive_mesh[i_v]
            cur_r = cur_vert - init_passive_mesh_center
            dot_r_r = torch.sum(cur_r * cur_r)
            cur_eye_mtx = torch.eye(3, dtype=torch.float32).cuda()
            r_mult_rT = torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0))
            I_ref += (dot_r_r * cur_eye_mtx - r_mult_rT) * per_vert_mass
        return init_passive_mesh_center, I_ref

    def calculate_obj_inertia(self, ):
        if self.canon_passive_obj_verts is None:
            cur_init_passive_mesh_verts = self.timestep_to_passive_mesh[0].clone()
        else:
            cur_init_passive_mesh_verts = self.canon_passive_obj_verts.clone()
        cur_init_passive_mesh_center = torch.mean(cur_init_passive_mesh_verts, dim=0)
        cur_init_passive_mesh_verts = cur_init_passive_mesh_verts - cur_init_passive_mesh_center
        # per_vert_mass=  cur_init_passive_mesh_verts.size(0) / self.obj_mass
        per_vert_mass = self.obj_mass / cur_init_passive_mesh_verts.size(0) 
        ## 
        print(f"[Calculating obj inertia] per_vert_mass: {per_vert_mass}")
        I_ref = torch.zeros((3, 3), dtype=torch.float32).cuda() ## caclulate I_ref; I_inv_ref; ##
        for i_v in range(cur_init_passive_mesh_verts.size(0)):
            cur_vert = cur_init_passive_mesh_verts[i_v]
            cur_r = cur_vert # - cur_init_passive_mesh_center
            cur_v_inertia = per_vert_mass * (torch.sum(cur_r * cur_r) - torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0)))
            # cur_v_inertia = torch.cross(cur_r, cur_r) * per_vert_mass3 # # 
            I_ref += cur_v_inertia

        print(f"In calculating inertia")
        print(I_ref)
        self.I_inv_ref = torch.linalg.inv(I_ref)
        
        self.I_ref = I_ref
    
    


    # the collison geometry should be able to locate the contact points #
    def calculate_collision_geometry_bounding_boxes(self, ):    
        # #
        # nearest ppont ? 
        init_passive_mesh = self.timestep_to_passive_mesh[0]
        maxx_init_passive_mesh, _ = torch.max(init_passive_mesh, dim=0)
        minn_init_passive_mesh, _ = torch.min(init_passive_mesh, dim=0)
        # maxx init passive mesh; minn init passvie mesh ##
        # contact passive mesh #
        self.maxx_init_passive_mesh = maxx_init_passive_mesh
        self.minn_init_passive_mesh = minn_init_passive_mesh
        
        pass


    def load_active_passive_timestep_to_mesh_v3(self, ):

        sv_fn = self.kinematic_mano_gt_sv_fn
        
        print(f'Loading from {sv_fn}')
        
        
        ''' Loading mano template '''
        mano_hand_template_fn = 'assets/mano_hand_template.obj'
        
        mano_hand_temp = trimesh.load(mano_hand_template_fn, force='mesh')
        hand_faces = mano_hand_temp.faces
        self.hand_faces = torch.from_numpy(hand_faces).long().to(self.device)
        
        print(f"Loading data from {sv_fn}")
        
        sv_dict = np.load(sv_fn, allow_pickle=True).item()
        
        print(f"sv_dict: {sv_dict.keys()}")
        
        obj_pcs = sv_dict['object_pc']
        obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
        # self.obj_pcs = obj_pcs
        
        
        
        obj_vertex_normals = sv_dict['obj_vertex_normals']
        obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
        self.obj_normals = obj_vertex_normals
        
        object_global_orient = sv_dict['object_global_orient'] # glboal orient 
        object_transl = sv_dict['object_transl']
        
        
        obj_faces = sv_dict['obj_faces']
        obj_faces = torch.from_numpy(obj_faces).long().cuda()
        self.obj_faces = obj_faces
        
        obj_verts = sv_dict['obj_verts']
        minn_verts = np.min(obj_verts, axis=0)
        maxx_verts = np.max(obj_verts, axis=0)
        extent = maxx_verts - minn_verts
        center_ori = (maxx_verts + minn_verts) / 2
        scale_ori = np.sqrt(np.sum(extent ** 2))
        obj_verts = torch.from_numpy(obj_verts).float().cuda()
        
        
        
        self.obj_verts = obj_verts
        
        
        
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
        
        self.obj_pcs = transformed_obj_verts
        
        rhand_verts = sv_dict['rhand_verts']
        rhand_verts = torch.from_numpy(rhand_verts).float().cuda()
        self.rhand_verts = rhand_verts ## rhand verts ## 
        
        
        
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


        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        tot_obj_quat = torch.stack(tot_obj_quat, dim=0)
        tot_reversed_obj_rot_mtx = torch.stack(tot_reversed_obj_rot_mtx, dim=0)
        self.tot_obj_quat = tot_obj_quat
        self.tot_reversed_obj_rot_mtx = tot_reversed_obj_rot_mtx
        
        ## should save self.object_global_orient and self.object_transl ##
        # object_global_orient, object_transl #
        self.object_global_orient = torch.from_numpy(object_global_orient).float().cuda()
        self.object_transl = torch.from_numpy(object_transl).float().cuda()
        return transformed_obj_verts, rhand_verts, self.obj_normals
    
    
    def load_active_passive_timestep_to_mesh_v3_taco(self, ):
        sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
        # /data1/xueyi/GRAB_extracted_test/train/20_sv_dict_real_obj.obj # data1
        
        # start_idx = 40
        
        start_idx = self.seq_start_idx
        maxx_ws = 150
        # maxx_ws = 90
        
        print(f"TACO loading data with start_idx: {start_idx}, maxx_ws: {maxx_ws}")
        
        
        sv_fn = self.kinematic_mano_gt_sv_fn
        
        ### get hand faces ###
        ''' Loading mano template '''
        mano_hand_template_fn = 'assets/mano_hand_template.obj'
        if not os.path.exists(mano_hand_template_fn):
            box_sv_fn = "/data2/xueyi/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy"
            box_sv_dict = np.load(box_sv_fn, allow_pickle=True).item()
            mano_hand_faces = box_sv_dict['hand_faces']
            mano_hand_verts = box_sv_dict['rhand_verts'][0]
            mano_hand_mesh = trimesh.Trimesh(mano_hand_verts, mano_hand_faces)
            mano_hand_mesh.export(mano_hand_template_fn)
        mano_hand_temp = trimesh.load(mano_hand_template_fn, force='mesh')
        hand_faces = mano_hand_temp.faces
        self.hand_faces = torch.from_numpy(hand_faces).long().to(self.device)
        
        
        
        print(f"Loading data from {sv_fn}")
        
        # sv_dict = np.load(sv_fn, allow_pickle=True).item()
        
        sv_dict = pkl.load(open(sv_fn, "rb"))
        
        self.hand_faces = torch.from_numpy(sv_dict['hand_faces']).float().cuda()
        
        print(f"sv_dict: {sv_dict.keys()}")
        
        maxx_ws = min(maxx_ws, len(sv_dict['obj_verts']) - start_idx)
        
        obj_pcs = sv_dict['obj_verts'][start_idx: start_idx + maxx_ws]
        obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
        
        self.obj_pcs = obj_pcs
        # obj_vertex_normals = sv_dict['obj_vertex_normals']
        # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
        self.obj_normals = torch.zeros_like(obj_pcs[0]) ### get the obj naormal vectors ##
        
        object_pose = sv_dict['obj_pose'][start_idx: start_idx + maxx_ws]
        object_pose = torch.from_numpy(object_pose).float().cuda() ### nn_frames x 4 x 4 ###
        object_global_orient_mtx = object_pose[:, :3, :3 ] ## nn_frames x 3 x 3 ##
        object_transl = object_pose[:, :3, 3] ## nn_frmaes x 3 ##
        
        
        # object_global_orient = sv_dict['object_global_orient'] # glboal orient 
        # object_transl = sv_dict['object_transl']
        
        
        obj_faces = sv_dict['obj_faces']
        obj_faces = torch.from_numpy(obj_faces).long().cuda()
        self.obj_faces = obj_faces # [0] ### obj faces ##
        
        # obj_verts = sv_dict['obj_verts']
        # minn_verts = np.min(obj_verts, axis=0)
        # maxx_verts = np.max(obj_verts, axis=0)
        # extent = maxx_verts - minn_verts
        # center_ori = (maxx_verts + minn_verts) / 2
        # scale_ori = np.sqrt(np.sum(extent ** 2))
        # obj_verts = torch.from_numpy(obj_verts).float().cuda()
        
        init_obj_verts = obj_pcs[0]
        init_obj_ornt_mtx = object_global_orient_mtx[0]
        init_obj_transl = object_transl[0]
        
        canon_obj_verts = torch.matmul(
            init_obj_ornt_mtx.contiguous().transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).transpose(1, 0).contiguous()
        ).transpose(1, 0).contiguous() ### 
        self.obj_verts = canon_obj_verts.clone()
        obj_verts = canon_obj_verts.clone()
        
        
        # self.obj_verts = obj_verts
        
        
        
        # mesh_scale = 0.8
        # bbmin, _ = obj_verts.min(0) #
        # bbmax, _ = obj_verts.max(0) #
        
        # center = (bbmin + bbmax) * 0.5
        # scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
        # # vertices = (vertices - center) * scale # (vertices - center) * scale # #
        
        # self.sdf_space_center = center
        # self.sdf_space_scale = scale
        
        
        sdf_sv_fn = self.obj_sdf_fn
        print(f'sdf_sv_fn: {sdf_sv_fn}')
        self.obj_sdf = np.load(sdf_sv_fn, allow_pickle=True)
        self.sdf_res = self.obj_sdf.shape[0]
        
        self.obj_sdf = torch.from_numpy(self.obj_sdf).float().cuda()
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
        
        tot_obj_quat = []
        
        for i_fr in range(object_global_orient_mtx.shape[0]):
            cur_ornt_mtx = object_global_orient_mtx[i_fr]
            cur_ornt_mtx_np = cur_ornt_mtx.detach().cpu().numpy() ### cur ornt mtx ## 
            cur_ornt_rot_struct = R.from_matrix(cur_ornt_mtx_np)
            cur_ornt_quat = cur_ornt_rot_struct.as_quat()
            cur_ornt_quat = cur_ornt_quat[[3, 0, 1, 2]]
            tot_obj_quat.append(torch.from_numpy(cur_ornt_quat).float().cuda()) ### float cuda ##
        
        # tot_obj_quat = np.stack(tot_obj_quat, axis=0) ## obj quat ##
        tot_obj_quat = torch.stack(tot_obj_quat, dim=0)
            
        
        # tot_reversed_obj_rot_mtx = []
        # tot_obj_quat = [] ## rotation matrix 
        
        
        # transformed_obj_verts = []
        # for i_fr in range(object_global_orient.shape[0]):
        #     cur_glb_rot = object_global_orient[i_fr]
        #     cur_transl = object_transl[i_fr]
        #     cur_transl = torch.from_numpy(cur_transl).float().cuda()
        #     cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
        #     cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
        #     cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
            
        #     cur_transformed_verts = torch.matmul(
        #         self.obj_verts, cur_glb_rot_mtx
        #     ) + cur_transl.unsqueeze(0)
            
        #     cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        #     tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
            
        #     cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
        #     cur_obj_quat = cur_glb_rot_struct.as_quat()
        #     cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
        #     cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
        #     tot_obj_quat.append(cur_obj_quat)

        #     # center_obj_verts = torch.mean(self.obj_verts, dim=0, keepdim=True)
        #     # cur_transformed_verts = torch.matmul(
        #     #     (self.obj_verts - center_obj_verts), cur_glb_rot_mtx
        #     # ) + cur_transl.unsqueeze(0) + center_obj_verts
            
        #     # cur_transformed_verts = torch.matmul(
        #     #     cur_glb_rot_mtx, self.obj_verts.transpose(1, 0)
        #     # ).contiguous().transpose(1, 0).contiguous() + cur_transl.unsqueeze(0)
        #     transformed_obj_verts.append(cur_transformed_verts)
        # transformed_obj_verts = torch.stack(transformed_obj_verts, dim=0)
        
        
        rhand_verts = sv_dict['hand_verts'][start_idx: start_idx + maxx_ws]
        rhand_verts = torch.from_numpy(rhand_verts).float().cuda()
        self.rhand_verts = rhand_verts ## rhand verts ## 
        
        
        
        # if '30_sv_dict' in sv_fn:
        #     bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
        #     obj_selected_verts = self.obj_verts[bbox_selected_verts_idxes]
        # else:
        #     obj_selected_verts = self.obj_verts.clone()
        
        # maxx_init_passive_mesh, _ = torch.max(obj_selected_verts, dim=0)
        # minn_init_passive_mesh, _ = torch.min(obj_selected_verts, dim=0)
        # self.maxx_init_passive_mesh = maxx_init_passive_mesh
        # self.minn_init_passive_mesh = minn_init_passive_mesh
        
        
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

        
        
        
        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        # tot_obj_quat = torch.stack(tot_obj_quat, dim=0)
        tot_reversed_obj_rot_mtx = object_global_orient_mtx.clone() # torch.stack(tot_reversed_obj_rot_mtx, dim=0)
        self.tot_obj_quat = tot_obj_quat
        self.tot_reversed_obj_rot_mtx = tot_reversed_obj_rot_mtx
        
        ## should save self.object_global_orient and self.object_transl ##
        # object_global_orient, object_transl #
        # self.object_global_orient = torch.from_numpy(object_global_orient).float().cuda()
        self.object_transl = object_transl.clone() #  torch.from_numpy(object_transl).float().cuda()
        return self.obj_pcs, rhand_verts, self.obj_normals
    
    
    
    def load_active_passive_timestep_to_mesh_twohands_arctic(self, ):
        import utils.utils as utils
        from manopth.manolayer import ManoLayer

        
        rgt_hand_pkl_fn = "assets/right_20230917_004.pkl"
        data_dict = pkl.load(open(rgt_hand_pkl_fn, "rb"))
        hand_faces = data_dict['hand_faces'] # ['faces']
        
        self.hand_faces = torch.from_numpy(hand_faces).long().to(self.device)
        
        self.start_idx = 20
        # self.window_size = 60
        self.window_size = self.window_size
        start_idx = self.start_idx
        window_size = self.window_size
        
        
        sv_fn = self.kinematic_mano_gt_sv_fn
        
        # gt_data_folder = "/".join(sv_fn.split("/")[:-1]) ## 
        gt_data_fn_name = sv_fn.split("/")[-1].split(".")[0]
        arctic_processed_data_sv_folder = "/home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj"
        if not os.path.exists(arctic_processed_data_sv_folder):
            arctic_processed_data_sv_folder = "/root/diffsim/quasi-dyn/raw_data/arctic_processed_canon_obj"
        gt_data_canon_obj_sv_fn = f"{arctic_processed_data_sv_folder}/{gt_data_fn_name}_canon_obj.obj"
            
        print(f"Loading data from {sv_fn}")
        
        sv_dict = np.load(sv_fn, allow_pickle=True).item()
        
        tot_frames_nn  = sv_dict["obj_rot"].shape[0]
        window_size = min(tot_frames_nn - self.start_idx, window_size)
        self.window_size = window_size
        
        
        object_global_orient = sv_dict["obj_rot"][start_idx: start_idx + window_size] # num_frames x 3 
        object_transl = sv_dict["obj_trans"][start_idx: start_idx + window_size] * 0.001 # num_frames x 3
        obj_pcs = sv_dict["verts.object"][start_idx: start_idx + window_size]
        
        # obj_pcs = sv_dict['object_pc']
        obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
        
        
        obj_vertex_normals = torch.zeros_like(obj_pcs)
        obj_tot_normals = obj_vertex_normals
        print(f"obj_normals: {obj_tot_normals.size()}")
        # /data/xueyi/sim/arctic_processed_data/processed_seqs/s01/espressomachine_use_01.npy
        
        # obj_vertex_normals = sv_dict['obj_vertex_normals']
        # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
        # self.obj_normals = obj_vertex_normals
        
        # object_global_orient = sv_dict['object_global_orient'] # glboal orient 
        # object_transl = sv_dict['object_transl']
        
        
        obj_faces = sv_dict['f'][0]
        obj_faces = torch.from_numpy(obj_faces).long().cuda()
        self.obj_faces = obj_faces

        
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
        #####  ## initial tarns of the object  and the hand ##
        # canon_obj_verts = torch.matmul(
        #     (init_obj_verts - init_obj_transl.unsqueeze(0)), init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        # )
        
        ## (R (v - t)^T)^T = (v - t) R^T
        canon_obj_verts = torch.matmul(
            init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
        ).contiguous().transpose(1, 0).contiguous()

        ## get canon obj verts ##
        
        # canon_obj_verts = obj_pcs[0].clone()
        self.obj_verts = canon_obj_verts.clone()
        obj_verts = canon_obj_verts.clone()
        
        
        #### save canonical obj mesh ####
        print(f"canon_obj_verts: {canon_obj_verts.size()}, obj_faces: {obj_faces.size()}")
        canon_obj_mesh = trimesh.Trimesh(vertices=canon_obj_verts.detach().cpu().numpy(), faces=obj_faces.detach().cpu().numpy())   
        canon_obj_mesh.export(gt_data_canon_obj_sv_fn)
        print(f"canonical obj exported to {gt_data_canon_obj_sv_fn}")
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
                cur_glb_rot_mtx_reversed, self.obj_verts.transpose(1, 0)
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
        if not os.path.exists(self.mano_path):
            self.mano_path = '/data/xueyi/mano_v1_2/models'
        self.rgt_mano_layer = ManoLayer(
            flat_hand_mean=False,
            side='right',
            mano_root=self.mano_path,
            ncomps=45,
            use_pca=False,
        ).cuda()
        
        self.lft_mano_layer = ManoLayer(
            flat_hand_mean=False,
            side='left',
            mano_root=self.mano_path,
            ncomps=45,
            use_pca=False,
        ).cuda()
        
        
        ##### rhand parameters #####
        rhand_global_orient_gt, rhand_pose_gt = sv_dict["rot_r"], sv_dict["pose_r"]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + self.window_size]
        
        rhand_global_orient_gt = rhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        rhand_pose_gt = rhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        rhand_transl, rhand_betas = sv_dict["trans_r"], sv_dict["shape_r"][0]
        rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + self.window_size], rhand_betas
        
        # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        rhand_transl = rhand_transl.reshape(self.window_size, -1).astype(np.float32)
        rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
        
        rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float().cuda()
        rhand_pose_var = torch.from_numpy(rhand_pose_gt).float().cuda()
        rhand_beta_var = torch.from_numpy(rhand_betas).float().cuda()
        rhand_transl_var = torch.from_numpy(rhand_transl).float().cuda()
        # R.from_rotvec(obj_rot).as_matrix()
        ##### rhand parameters #####
        
        
        ##### lhand parameters #####
        lhand_global_orient_gt, lhand_pose_gt = sv_dict["rot_l"], sv_dict["pose_l"]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
        lhand_global_orient_gt = lhand_global_orient_gt[start_idx: start_idx + self.window_size]
        # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
        lhand_pose_gt = lhand_pose_gt[start_idx: start_idx + self.window_size]
        
        lhand_global_orient_gt = lhand_global_orient_gt.reshape(self.window_size, -1).astype(np.float32)
        lhand_pose_gt = lhand_pose_gt.reshape(self.window_size, -1).astype(np.float32)
        
        lhand_transl, lhand_betas = sv_dict["trans_l"], sv_dict["shape_l"][0]
        lhand_transl, lhand_betas = lhand_transl[start_idx: start_idx + self.window_size], lhand_betas
        
        # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
        lhand_transl = lhand_transl.reshape(self.window_size, -1).astype(np.float32)
        lhand_betas = lhand_betas.reshape(-1).astype(np.float32)
        
        lhand_global_orient_var = torch.from_numpy(lhand_global_orient_gt).float().cuda()
        lhand_pose_var = torch.from_numpy(lhand_pose_gt).float().cuda()
        lhand_beta_var = torch.from_numpy(lhand_betas).float().cuda()
        lhand_transl_var = torch.from_numpy(lhand_transl).float().cuda() # self.window_size x 3
        # R.from_rotvec(obj_rot).as_matrix()
        ##### lhand parameters #####
        
    
        
        rhand_verts, rhand_joints = self.rgt_mano_layer(
            torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
            rhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), rhand_transl_var
        )
        ### rhand_joints: for joints ###
        rhand_verts = rhand_verts * 0.001
        rhand_joints = rhand_joints * 0.001
        
        
        lhand_verts, lhand_joints = self.lft_mano_layer(
            torch.cat([lhand_global_orient_var, lhand_pose_var], dim=-1),
            lhand_beta_var.unsqueeze(0).repeat(self.window_size, 1).view(-1, 10), lhand_transl_var
        )
        ### rhand_joints: for joints ###
        lhand_verts = lhand_verts * 0.001
        lhand_joints = lhand_joints * 0.001
        
        
        ### lhand and the rhand ###
        # rhand_verts, lhand_verts #
        self.rhand_verts = rhand_verts
        self.lhand_verts = lhand_verts 
        
        self.hand_faces = self.rgt_mano_layer.th_faces
        
        
        
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
        
        self.obj_normals = torch.zeros_like(obj_verts)
        
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
        
        re_transformed_obj_verts = torch.stack(re_transformed_obj_verts, dim=0)
        self.re_transformed_obj_verts = re_transformed_obj_verts
        
        # tot_obj_quat, tot_reversed_obj_rot_mtx #
        tot_obj_quat = torch.stack(tot_obj_quat, dim=0) ## tot obj quat ##
        tot_reversed_obj_rot_mtx = torch.stack(tot_reversed_obj_rot_mtx, dim=0)
        self.tot_obj_quat = tot_obj_quat # obj quat #
        
        # self.tot_obj_quat[0, 0] = 1.
        # self.tot_obj_quat[0, 1] = 0.
        # self.tot_obj_quat[0, 2] = 0.
        # self.tot_obj_quat[0, 3] = 0.
        
        self.tot_reversed_obj_rot_mtx = tot_reversed_obj_rot_mtx
        
        # self.tot_reversed_obj_rot_mtx[0] = torch.eye(3, dtype=torch.float32).cuda()
        
        ## should save self.object_global_orient and self.object_transl ##
        # object_global_orient, object_transl #
        self.object_global_orient = torch.from_numpy(object_global_orient).float().cuda()
        self.object_transl = torch.from_numpy(object_transl).float().cuda()
        
        # self.object_transl[0, :] = self.object_transl[0, :] * 0.0
        return transformed_obj_verts, rhand_verts, obj_tot_normals
    
    
    ### get ball primitives ###
    
    def get_ball_primitives(self, ):

        maxx_verts, _ = torch.max(self.obj_verts, dim=0)
        minn_verts, _ = torch.min(self.obj_verts, dim=0) # 
        center_verts = (maxx_verts + minn_verts) / 2.
        extent_verts = (maxx_verts - minn_verts)
        ball_d = max(extent_verts[0].item(), max(extent_verts[1].item(), extent_verts[2].item()))
        ball_r = ball_d / 2.
        return center_verts, ball_r
    



    # tot_l2loss = self.compute_loss_optimized_offset_with_preopt_offset()
    def compute_loss_optimized_offset_with_preopt_offset(self, tot_time_idx):
        # timestep_to_optimizable_offset
        optimized_offset = self.renderer.bending_network[1].timestep_to_optimizable_offset
        preopt_offset = self.ts_to_mesh_offset_for_opt # 
        # tot_l2loss = 0.
        tot_l2losses = []
        # for ts in range(0, self.n_timesteps):
        for ts in range(1, tot_time_idx + 1):
            if ts in optimized_offset and ts in preopt_offset:
                cur_optimized_offset = optimized_offset[ts]
                cur_preopt_offset = preopt_offset[ts]
                diff_optimized_preopt_offset = torch.mean(torch.sum((cur_preopt_offset - cur_optimized_offset) ** 2))
                # if ts == 1:
                #     tot_l2loss = 
                tot_l2losses.append(diff_optimized_preopt_offset)
            # tot_l2loss += diff_optimized_preopt_offset
        tot_l2losses = torch.stack(tot_l2losses, dim=0)
        tot_l2loss = torch.mean(tot_l2losses)
        # tot_l2loss = tot_l2loss / float(self.n_timesteps - 1)
        return tot_l2loss
    
    # tracking_loss = self.compute_loss_optimized_transformations(cur_time_idx)
    def compute_loss_optimized_transformations(self, cur_time_idx):
        # # 
        cur_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[cur_time_idx]
        cur_translations = self.other_bending_network.timestep_to_optimizable_total_def[cur_time_idx]
        init_passive_mesh = self.timestep_to_passive_mesh[0]
        center_passive_mesh = torch.mean(init_passive_mesh, dim=0)
        pred_passive_mesh = torch.matmul(
            cur_rot_mtx, (init_passive_mesh - center_passive_mesh.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) + center_passive_mesh.unsqueeze(0) + cur_translations.unsqueeze(0)
        gt_passive_mesh = self.timestep_to_passive_mesh[cur_time_idx]
        tracking_loss = torch.sum(
            (pred_passive_mesh - gt_passive_mesh) ** 2, dim=-1
        ).mean()
        return tracking_loss

    def compute_loss_optimized_transformations_v2(self, cur_time_idx, cur_passive_time_idx):
        # # ## get the 
        
        # timestep_to_optimizable_rot_mtx, timestep_to_optimizable_total_def
        cur_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[cur_time_idx]
        cur_translations = self.other_bending_network.timestep_to_optimizable_total_def[cur_time_idx]
        
        if self.other_bending_network.canon_passive_obj_verts is None:
            init_passive_mesh = self.timestep_to_passive_mesh[0]
            center_passive_mesh = torch.mean(init_passive_mesh, dim=0)
            # center_passive_mesh = torch.zeros((3, )).cuda()
        else:
            init_passive_mesh = self.other_bending_network.canon_passive_obj_verts
            center_passive_mesh = torch.zeros((3, )).cuda()
        pred_passive_mesh = torch.matmul(
            cur_rot_mtx, (init_passive_mesh - center_passive_mesh.unsqueeze(0)).transpose(1, 0)
        ).transpose(1, 0) + center_passive_mesh.unsqueeze(0) + cur_translations.unsqueeze(0)
        gt_passive_mesh = self.timestep_to_passive_mesh[cur_passive_time_idx]
        tracking_loss = torch.sum( # gt mehses # 
            (pred_passive_mesh - gt_passive_mesh) ** 2, dim=-1
        ).mean()
        return tracking_loss

    
    def construct_field_network(self, input_dim, hidden_dimensions,  output_dim):
        cur_field_network = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dimensions), nn.ReLU(),
                nn.Linear(hidden_dimensions, hidden_dimensions * 2), # with maxpoll layers # 
                nn.Linear(hidden_dimensions * 2, hidden_dimensions), nn.ReLU(), # 
                nn.Linear(hidden_dimensions, output_dim), # hidden
            ]
        )

        with torch.no_grad():
            for i, cc in enumerate(cur_field_network[:]):
                # for cc in layer:
                if isinstance(cc, nn.Linear):
                    torch.nn.init.kaiming_uniform_(
                        cc.weight, a=0, mode="fan_in", nonlinearity="relu"
                    )
                    if i < len(cur_field_network) - 1:
                        torch.nn.init.zeros_(cc.bias)
            torch.nn.init.zeros_(cur_field_network[-1].weight) 
            torch.nn.init.zeros_(cur_field_network[-1].bias) # initialize the field network for bendign and deofrmation
            
        return cur_field_network
    
    


    
    ''' GRAB clips --- expanded point set and expanded points for retargeting ''' 
    def train_point_set(self, ):
        
        ## ## GRAB clips ## ##
        # states -> the robot actions --- in this sim ##
        # chagne # # mano notjmano but the mano ---> optimize the mano delta states? # 
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # update learning rrate # 
        # robot actions ##
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        
        ''' Load the robot hand '''
        model_path = self.conf['model.sim_model_path'] # 
        # robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        self.hand_type = "redmax_hand"
        if model_path.endswith(".xml"):
            self.hand_type = "redmax_hand"
            robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        else:
            self.hand_type = "shadow_hand"
            robot_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path, args=None)
        ## shadow hand; redmax hand ##
        self.robot_agent = robot_agent
        robo_init_verts = self.robot_agent.robot_pts
        # if self.hand_type == "redmax_hand":
        robo_sampled_verts_idxes_fn = "robo_sampled_verts_idxes.npy"
        if os.path.exists(robo_sampled_verts_idxes_fn):
            sampled_verts_idxes = np.load(robo_sampled_verts_idxes_fn)
            sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
        else:
            n_sampling = 1000
            pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
            sampled_verts_idxes = pts_fps_idx
            np.save(robo_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
        # else:
        #     sampled_verts_idxes = None
        self.robo_hand_faces = self.robot_agent.robot_faces
        
        
        ## sampled verts idxes ## 
        self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- # # --- robot hand mani asset? ##  ## robot hand mani asse ##
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m 
        #### retargeting is also a problem here ####
        
        
        ''' Load the mano hand model '''
        model_path_mano = self.conf['model.mano_sim_model_path']
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.robo_hand_faces = self.mano_agent.robot_faces
        
        # if self.use_mano_hand_for_test: ##
        #     self.robo_hand_faces = self.hand_faces
        ## 
        
        ## start expanding the current visual pts ##
        print(f"Start expanding the current visual pts...")
        expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        
        self.expanded_visual_pts_nn = expanded_visual_pts.size(0)

        ## expanded_visual_pts of the expanded visual pts #
        expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        print(f"Saving expanded visual pts with shape {expanded_visual_pts.size()} to {expanded_visual_pts_sv_fn}")
        np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy) # 
        
        
        nn_substeps = 10
        
        ## 
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        # self.hand_faces # 

        
        ''' Expnad the current visual points ''' 
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        # params_to_train += list(self.robot_actions.parameters())
        
        self.mano_robot_delta_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        
        self.mano_robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # params_to_train += list(self.robot_init_states.parameters())
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        # params_to_train += list(self.robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        # params_to_train += list(self.robot_glb_trans.parameters())   
        
        ## 
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60, # embedding; a realistic thing # # ## so the optimizable modle deisgn --- approxmimate what you see and approximate the target simulator ## # at a distance; the asymmetric contact froces spring ks -- all of them wold affect model's behaviours ## ## mao robot glb 
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        
        
        
        self.mano_expanded_actuator_delta_offset = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_delta_offset.weight)
        # params_to_train += list(self.mano_expanded_actuator_delta_offset.parameters())
        
        ### mano friction forces ###
        # mano_expanded_actuator_friction_forces, mano_expanded_actuator_delta_offset # 
        self.mano_expanded_actuator_friction_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_friction_forces.weight)

        ## load mano states and actions ##
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            if 'mano_robot_init_states' in optimized_init_actions_ckpt:
                self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
            if 'mano_robot_glb_rotation' in optimized_init_actions_ckpt:
                self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])
            # if 'robot_delta_states' in optimized_init_actions_ckpt:
            #     self.mano_robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
            # self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            
            if 'mano_robot_states' in optimized_init_actions_ckpt:
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            if 'expanded_actuator_friction_forces' in optimized_init_actions_ckpt:
                try:
                    self.mano_expanded_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_friction_forces'])
                except:
                    pass
            if 'expanded_actuator_delta_offset' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_delta_offset.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_delta_offset'])


        ''' parameters for the real robot hand '''
        self.robot_actions = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_actions.weight)
        params_to_train += list(self.robot_actions.parameters())
        
        
        # self.robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        self.robot_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_states.weight)
        params_to_train += list(self.robot_states.parameters())
        
        self.robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_init_states.weight)
        params_to_train += list(self.robot_init_states.parameters())
        
        ## robot glb rotations ##
        self.robot_glb_rotation = nn.Embedding( ## robot hand rotation 
            num_embeddings=num_steps, embedding_dim=4
        ).cuda()
        self.robot_glb_rotation.weight.data[:, 0] = 1.
        self.robot_glb_rotation.weight.data[:, 1:] = 0.
        
        self.robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_glb_trans.weight)
        
        
        # ### local minimum -> ## robot 
        self.robot_actuator_friction_forces = nn.Embedding( # frictional forces ##
            num_embeddings=365428 * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_actuator_friction_forces.weight)
        
        
        
        if len(self.load_optimized_init_transformations) > 0:
            print(f"[Robot] Loading optimized init transformations from {self.load_optimized_init_transformations}")
            cur_optimized_init_actions_fn = self.load_optimized_init_transformations
            # cur_optimized_init_actions = # optimized init states # ## robot init states ##
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            try:
                self.robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
            except:
                pass
            self.robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
            if 'robot_delta_states' in optimized_init_actions_ckpt:
                try:
                    self.robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
                except:
                    pass
            if 'robot_states' in optimized_init_actions_ckpt:
                self.robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
            # if 'robot_delta_states'  ## robot delta states ##
            # self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
            self.robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
        
        
        if self.hand_type == "redmax_hand":
            self.maxx_robo_pts = 25.
            self.minn_robo_pts = -15.
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.5437551664260203
        else:
            self.minn_robo_pts = -0.1
            self.maxx_robo_pts = 0.2
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.437551664260203
        ## for grab ##
        self.mult_const_after_cent = self.mult_const_after_cent / 3. * 0.9507
        
        
        ### figners for the finger retargeting approach ###
        self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        # self.robot_fingers = [3591, 4768, 6358, 10228, 6629, 10566, 5631, 9673]
        self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877]
        # self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        if self.hand_type == "redmax_hand":
            self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
            self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        
        # self.mano_mult_const_after_cent = 3.
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']


        self.nn_ts = self.nn_timesteps - 1

        ''' parameters for the real robot hand ''' 
        
        
        self.timestep_to_active_mesh = {}
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        
        self.timestep_to_active_mesh_w_delta_states = {}
        
        
        # states -> get states -> only update the acitons #
        with torch.no_grad(): # init them to zero 
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                ''' Get rotations, translations, and actions of the current robot ''' ## mano robot glb rotation 
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7) # mano glb rot
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                self.mano_agent.set_init_states_target_value(link_cur_states)
                
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True) 

                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                ## 
                ### transform by the glboal transformation and the translation ### ## cur visual pts ## ## contiguous() ##
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## transformed pts ## 
                
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offset
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_w_delta_states[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_opt_ours_sim[cur_ts] = cur_visual_pts.detach()
                
        self.iter_step = 0
        
        ''' Set redmax robot actions '''


        params_to_train_kines = []
        # params_to_train_kines += list(self.mano_robot_glb_rotation.parameters())
        # params_to_train_kines += list(self.mano_robot_glb_trans.parameters())
        # params_to_train_kines += list(self.mano_robot_delta_states.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        # 
        # params_to_train_kines += list(self.robot_states.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        
        
        
        
        # can tray the optimizer ## ## mano states ##
        # if self.use_LBFGS:
        #     self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=self.learning_rate)
        # else:
        #     self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        
        
        mano_expanded_actuator_delta_offset_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone().detach()
        
        # if self.optimize_rules:
        #     params_to_train_kines = []
        #     params_to_train_kines += list(self.other_bending_network.parameters())
        #     # self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        #     if self.use_LBFGS:
        #         self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=self.learning_rate)
        #     else:
        #         self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        #     # self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=1e-2)
        
        
        
        ## policy and the controller ##
        # self.other_bending_network.spring_ks_values.weight.data[:, :] = 0.1395
        # self.other_bending_network.spring_ks_values.weight.data[0, :] = 0.1395
        # self.other_bending_network.spring_ks_values.weight.data[1, :] = 0.00
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 10.0
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 1000.0
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 100.0
        
        
        # load_redmax_robot_actions_fn = "/data3/datasets/diffsim/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v13__hieoptrotorules_penalty_friction_ct_01_thres_0001_avg_optrobo_nglbtrans_reg001_manorules_projk_0_nsp_res_sqrstiff_preprojfri_projk_0_optrules_diffh_subs_10_/checkpoints/redmax_robot_actions_ckpt_001900.pth"
        # if len(load_redmax_robot_actions_fn) > 0:
        #     redmax_robot_actions_ckpt = torch.load(load_redmax_robot_actions_fn, map_location=self.device, )
            # self.redmax_robot_actions.load_state_dict(redmax_robot_actions_ckpt['redmax_robot_actions'])

        ''' prepare for keeping the original global rotations, trans, and states '''
        # ori_mano_robot_glb_rot = self.mano_robot_glb_rotation.weight.data.clone()
        # ori_mano_robot_glb_trans = self.mano_robot_glb_trans.weight.data.clone()
        # ori_mano_robot_delta_states = self.mano_robot_delta_states.weight.data.clone()
        
        
        self.iter_step = 0


        for i_iter in tqdm(range(100000)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep #
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # # # #
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            # correspondence_pts_idxes = None
            
            # tot_penetrating_depth_penalty = []
            # tot_ragged_dist = []
            # tot_delta_offset_reg_motion = []
            # tot_dist_mano_visual_ori_to_cur = []
            # tot_reg_loss = []
            # tot_diff_cur_states_to_ref_states = []
            # tot_diff_tangential_forces = []
            # penetration_forces = None ###
            # sampled_visual_pts_joint_idxes = None
            
            # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            self.timestep_to_anchored_mano_pts = {}
            
            
            self.ts_to_contact_passive_normals = {}
            self.ts_to_passive_normals = {}
            self.ts_to_passive_pts = {}
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            self.ts_to_dyn_mano_pts = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            # tot_grad_qs = []
            
            robo_intermediates_states = []
            
            tot_penetration_depth = []
            
            robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            tot_interpenetration_nns = []
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            # for cur_ts in range(self.nn_ts):
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                # tot_redmax_actions = []
                # actions = {}

                self.free_def_bending_weight = 0.0

                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone()
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot)
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)


                # # # cur_ts_delta_rot, cur_ts_redmax_robot_trans # # #
                # cur_glb_rot = torch.matmul(cur_ts_delta_rot, cur_glb_rot)
                cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans # redmax robot transj## 

                link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_init_states_target_value(link_cur_states)
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True)  # init state visual pts #
                
                cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                ## get the friction forces ##
                cur_visual_pts_friction_forces = self.mano_expanded_actuator_friction_forces(cur_visual_pts_idxes)

                ### transform the visual pts ### ## fricton forces ##
                # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                # cur_visual_pts = cur_visual_pts * 2. - 1.  # cur visual pts # 
                # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # # mult_const #
                
                ### visual pts are expanded from xxx ###
                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent # mult cnst after cent #
                cur_dyn_mano_pts = cur_dyn_mano_pts * self.mano_mult_const_after_cent
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                cur_dyn_mano_pts = torch.matmul(cur_rot, cur_dyn_mano_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offset
                
                # diff_redmax_visual_pts_with_ori_visual_pts = torch.sum(
                #     (cur_visual_pts[sampled_verts_idxes] - self.timestep_to_active_mesh_opt_ours_sim[cur_ts].detach()) ** 2, dim=-1
                # )
                # diff_redmax_visual_pts_with_ori_visual_pts = diff_redmax_visual_pts_with_ori_visual_pts.mean()
                
                # train the friction net? how to train the friction net? #
                # if self.use_mano_hand_for_test:
                #     self.timestep_to_active_mesh[cur_ts] = self.rhand_verts[cur_ts] # .detach()
                # else:
                
                
                # timestep_to_anchored_mano_pts, timestep_to_raw_active_meshes # 
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts[cur_ts] = cur_dyn_mano_pts.detach().cpu().numpy()
                
                # # ragged_dist = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # # dist_transformed_expanded_visual_pts_to_ori_visual_pts = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # # diff_cur_states_to_ref_states = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                # cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                # robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # self.robot_agent.set_init_states_target_value(robo_links_states)
                # cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                

                # ### transform the visual pts ###
                # cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                # cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                # cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                
                
                # cur_rot = cur_robo_glb_rot
                # cur_trans = cur_glb_trans
                
                # timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                # timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                # ### transform by the glboal transformation and the translation ###
                # cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                
                # self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.clone() # robo visual pts #
                
                # if self.hand_type == 'redmax_hand':
                #     cur_robo_visual_pts = cur_robo_visual_pts[sampled_verts_idxes] # 
                
                
                
                self.free_def_bending_weight = 0.0
                # self.free_def_bending_weight = 0.5
                
                # if i_iter == 0 and cur_ts == 0: ## for the tiantianquan sequence ##
                #     dist_robot_pts_to_mano_pts = torch.sum(
                #         (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                #     )
                #     minn_dist_robot_pts_to_mano_pts, correspondence_pts_idxes = torch.min(dist_robot_pts_to_mano_pts, dim=-1)
                #     minn_dist_robot_pts_to_mano_pts = torch.sqrt(minn_dist_robot_pts_to_mano_pts)
                #     # dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.01
                #     dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.005
                    
                #     corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                    
                #     dist_corr_correspondence_pts_to_mano_visual_pts = torch.sum(
                #         (corr_correspondence_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                #     )
                #     dist_corr_correspondence_pts_to_mano_visual_pts = torch.sqrt(dist_corr_correspondence_pts_to_mano_visual_pts)
                #     minn_dist_to_corr_pts, _ = torch.min(dist_corr_correspondence_pts_to_mano_visual_pts, dim=0)
                #     anchored_mano_visual_pts = minn_dist_to_corr_pts < 0.005
                    
                # corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                # # corr_robo = cur_visual_pts[sampled_verts_idxes]
                # cd_robo_pts_to_corr_mano_pts = torch.sum(
                #     (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts[anchored_mano_visual_pts].unsqueeze(0)) ** 2, dim=-1
                # )
                
                # self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts[anchored_mano_visual_pts].detach().cpu().numpy()
                
                # cd_robo_to_mano, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=-1)
                # cd_mano_to_robo, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=0)
                # # diff_robo_to_corr_mano_pts = cd_mano_to_robo.mean()
                # diff_robo_to_corr_mano_pts = cd_robo_to_mano.mean()
                
                # mano_fingers = self.rhand_verts[cur_ts][self.mano_fingers]
                
                # if self.hand_type == 'redmax_hand':
                #     # sampled_verts_idxes
                #     robo_fingers = cur_robo_visual_pts[sampled_verts_idxes][self.robot_fingers]
                # else:
                #     robo_fingers = cur_robo_visual_pts[self.robot_fingers]
                
                # pure finger tracking ##
                # pure_finger_tracking_loss = torch.sum((mano_fingers - robo_fingers) ** 2)
                
                
                # diff_robo_to_corr_mano_pts_finger_tracking = torch.sum(
                #     (corr_correspondence_pts - cur_robo_visual_pts) ** 2, dim=-1
                # )
                # diff_robo_to_corr_mano_pts_finger_tracking = diff_robo_to_corr_mano_pts_finger_tracking[dist_smaller_than_thres]
                # diff_robo_to_corr_mano_pts_finger_tracking = diff_robo_to_corr_mano_pts_finger_tracking.mean()
                
                # loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef + pure_finger_tracking_loss * 0.5 # + diff_robo_to_corr_mano_pts_finger_tracking * self.finger_tracking_loss_coef
                
                ## TODO: add the glboal retargeting using fingers before conducting this approach 
                
                
                # def evaluate_tracking_loss():
                #     self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
                    
                #     ## 
                #     # init states  # 
                #     # cur_ts % mano_nn_substeps == 0:
                #     if (cur_ts + 1) % mano_nn_substeps == 0:
                #         cur_passive_big_ts = cur_ts // mano_nn_substeps
                #         in_func_tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                #         # tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                #     else:
                #         in_func_tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                #     return in_func_tracking_loss

                    
                if contact_pairs_set is None:
                    self.contact_pairs_set = None
                else:
                    self.contact_pairs_set = contact_pairs_set.copy()
                
                # ### if traiing the jrbpt h
                # print(self.timestep_to_active_mesh[cur_ts].size(), cur_visual_pts_friction_forces.size())
                contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set, pts_frictional_forces=cur_visual_pts_friction_forces)

                ### train with force to active ##
                # if self.train_with_forces_to_active and (not self.use_mano_inputs):
                #     # penetration_forces #
                #     if torch.sum(self.other_bending_network.penetrating_indicator.float()) > 0.5:
                #         net_penetrating_forces = self.other_bending_network.penetrating_forces
                #         net_penetrating_points = self.other_bending_network.penetrating_points

                        
                #         # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
                #         self.timestep_to_penetration_points[cur_ts] = net_penetrating_points.detach().cpu().numpy()
                #         self.timestep_to_penetration_points_forces[cur_ts] = net_penetrating_forces.detach().cpu().numpy()
                        
                        
                #         ### transform the visual pts ###
                #         # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                #         # cur_visual_pts = cur_visual_pts * 2. - 1.
                #         # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # mult_const #
                        
                #         # sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[finger_sampled_idxes][self.other_bending_network.penetrating_indicator]
                        
                #         # sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[self.other_bending_network.penetrating_indicator]
                        
                #         net_penetrating_forces = torch.matmul(
                #             cur_rot.transpose(1, 0), net_penetrating_forces.transpose(1, 0)
                #         ).transpose(1, 0)
                #         net_penetrating_forces = net_penetrating_forces / self.mult_const_after_cent
                #         net_penetrating_forces = net_penetrating_forces / 2
                #         net_penetrating_forces = net_penetrating_forces * self.extent_robo_pts
                        
                #         net_penetrating_points = torch.matmul(
                #             cur_rot.transpose(1, 0), (net_penetrating_points - cur_trans.unsqueeze(0)).transpose(1, 0)
                #         ).transpose(1, 0)
                #         net_penetrating_points = net_penetrating_points / self.mult_const_after_cent
                #         net_penetrating_points = (net_penetrating_points + 1.) / 2. # penetrating points #
                #         net_penetrating_points = (net_penetrating_points * self.extent_robo_pts) + self.minn_robo_pts
                        
                        
                #         link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        
                #     else:
                #         # penetration_forces = None
                #         link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        
                # if contact_pairs_set is not None:
                #     self.contact_pairs_sets[cur_ts] = contact_pairs_set.copy()
                
                # # contact force d ## ts to the passive normals ## 
                # self.ts_to_contact_passive_normals[cur_ts] = self.other_bending_network.tot_contact_passive_normals.detach().cpu().numpy()
                # self.ts_to_passive_pts[cur_ts] = self.other_bending_network.cur_passive_obj_verts.detach().cpu().numpy()
                # self.ts_to_passive_normals[cur_ts] = self.other_bending_network.cur_passive_obj_ns.detach().cpu().numpy()
                # self.ts_to_contact_force_d[cur_ts] = self.other_bending_network.contact_force_d.detach().cpu().numpy()
                # self.ts_to_penalty_frictions[cur_ts] = self.other_bending_network.penalty_friction_tangential_forces.detach().cpu().numpy()
                # if self.other_bending_network.penalty_based_friction_forces is not None:
                #     self.ts_to_penalty_disp_pts[cur_ts] = self.other_bending_network.penalty_based_friction_forces.detach().cpu().numpy()
                
                # # # get the penetration depth of the bending network #
                
                
                # ### optimize with intermediates ### # optimize with intermediates # 
                # if self.optimize_with_intermediates:
                #     tracking_loss = self.compute_loss_optimized_transformations(cur_ts + 1) # 
                # else:
                #     tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                    
                
                # # cur_ts % mano_nn_substeps == 0: # 
                if (cur_ts + 1) % mano_nn_substeps == 0:
                    cur_passive_big_ts = cur_ts // mano_nn_substeps
                    ## compute optimized transformations ##
                    tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                    tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                else:
                    tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()

                # # hand_tracking_loss = torch.sum( ## delta states? ##
                # #     (self.timestep_to_active_mesh_w_delta_states[cur_ts] - cur_visual_pts) ** 2, dim=-1
                # # )
                # # hand_tracking_loss = hand_tracking_loss.mean()
                
                
                # # loss = tracking_loss + self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                # # diff_redmax_visual_pts_with_ori_visual_pts.backward()
                penetraton_penalty = self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                
                tot_penetration_depth.append(penetraton_penalty.detach().item())
                
                # smaller_than_zero_level_set_indicator
                cur_interpenetration_nns = self.other_bending_network.smaller_than_zero_level_set_indicator.float().sum()
                
                tot_interpenetration_nns.append(cur_interpenetration_nns)
                
                # diff_hand_tracking = torch.zeros((1,), dtype=torch.float32).cuda().mean() ## 
                
                # ## diff
                # # diff_hand_tracking_coef
                # # kinematics_proj_loss = kinematics_trans_diff + penetraton_penalty + diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss
                
                # # if self.use_mano_hand_for_test: ## only the kinematics mano hand is optimized here ##
                # #     kinematics_proj_loss = tracking_loss
                
                # # kinematics_proj_loss = hand_tracking_loss * 1e2 ## 1e2 and the 1e2 ## 
                
                # kinematics_proj_loss = diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss + penetraton_penalty
                
                ## kinematics 
                # kinematics_proj_loss = loss_finger_tracking # + tracking_loss + penetraton_penalty
                
                reg_delta_offset_loss = torch.sum(
                    (mano_expanded_actuator_delta_offset_ori - self.mano_expanded_actuator_delta_offset.weight.data) ** 2, dim=-1
                )
                reg_delta_offset_loss = reg_delta_offset_loss.mean()
                # motion_reg_loss_coef
                reg_delta_offset_loss = reg_delta_offset_loss * self.motion_reg_loss_coef # ## motion reg loss ## #
                
                
                ### tracking loss and the penetration penalty ###
                kinematics_proj_loss = tracking_loss + penetraton_penalty + reg_delta_offset_loss ## tracking and the penetration penalty ##
                
                ### kinematics proj loss ###
                loss = kinematics_proj_loss # * self.loss_scale_coef ## get 


                
                self.kines_optimizer.zero_grad()
                
                try:
                    kinematics_proj_loss.backward(retain_graph=True)
                    
                    self.kines_optimizer.step()
                except:
                    pass
                
                
                # mano_expanded_actuator_delta_offset, # point to the gradient of a #
                ### get the gradient information ###
                # if self.iter_step > 1239 and self.mano_expanded_actuator_delta_offset.weight.grad is not None:
                #     grad_mano_expanded_actuator_delta_offset = self.mano_expanded_actuator_delta_offset.weight.grad.data
                #     grad_mano_expanded_actuator_delta_offset = torch.sum(grad_mano_expanded_actuator_delta_offset)
                #     print(f"iter_step: {self.iter_step}, grad_offset: {grad_mano_expanded_actuator_delta_offset}") 
                # if self.iter_step > 1239 and self.mano_expanded_actuator_friction_forces.weight.grad is not None:
                #     grad_mano_expanded_actuator_friction_forces = self.mano_expanded_actuator_friction_forces.weight.grad.data
                #     grad_mano_expanded_actuator_friction_forces = torch.sum(grad_mano_expanded_actuator_friction_forces)
                #     print(f"iter_step: {self.iter_step}, grad_friction_forces: {grad_mano_expanded_actuator_friction_forces}")
                    
                # if self.iter_step > 1239 and cur_visual_pts.grad is not None:
                #     grad_cur_visual_pts = torch.sum(cur_visual_pts.grad.data)
                #     print(f"iter_step: {self.iter_step}, grad_cur_visual_pts: {grad_cur_visual_pts}")
                
                
                # # 
                
                # if self.use_LBFGS:
                #     self.kines_optimizer.step(evaluate_tracking_loss) # 
                # else:
                #     self.kines_optimizer.step()

                # 
                # tracking_loss.backward(retain_graph=True)
                # if self.use_LBFGS:
                #     self.other_bending_network.reset_timestep_to_quantities(cur_ts)
                
                
                robot_states_actions_diff_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                robo_actions_diff_loss.append(reg_delta_offset_loss.item())

                
                tot_losses.append(loss.detach().item()) # total losses # # total losses # 
                # tot_penalty_dot_forces_normals.append(cur_penalty_dot_forces_normals.detach().item())
                # tot_penalty_friction_constraint.append(cur_penalty_friction_constraint.detach().item())
                
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint() # a smart solution for them ? # # save checkpoint # ## save checkpoint ##
                self.update_learning_rate() ## update learning rate ##
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            if len(tot_tracking_loss) > 0:
                tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            else:
                tot_tracking_loss = 0.0
            if len(tot_penetration_depth) > 0:
                tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            else:
                tot_penetration_depth = 0.0
            robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            if len(mano_tracking_loss) > 0:
                mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))
            else:
                mano_tracking_loss = 0.0
            
            avg_tot_interpenetration_nns = float(sum(tot_interpenetration_nns) ) / float(len(tot_interpenetration_nns))
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} penetration = {} / {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, avg_tot_interpenetration_nns, self.timestep_to_active_mesh[0].size(0), self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')


            if i_iter % self.val_mesh_freq == 0:
                self.validate_mesh_robo_g()
                self.validate_mesh_robo()
                self.validate_contact_info_robo()
                
            
            torch.cuda.empty_cache()
    
    
    ''' GRAB clips --- expanded point set and expanded points for retargeting ''' 
    def train_point_set_dyn(self, ):
        
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        
        ''' Load the robot hand '''
        model_path = self.conf['model.sim_model_path']
        self.hand_type = "redmax_hand"
        if model_path.endswith(".xml"):
            self.hand_type = "redmax_hand"
            robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        else:
            self.hand_type = "shadow_hand"
            robot_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path, args=None)
        ## shadow hand; redmax hand ##
        self.robot_agent = robot_agent
        robo_init_verts = self.robot_agent.robot_pts
        # if self.hand_type == "redmax_hand":
        robo_sampled_verts_idxes_fn = "robo_sampled_verts_idxes.npy"
        if os.path.exists(robo_sampled_verts_idxes_fn):
            sampled_verts_idxes = np.load(robo_sampled_verts_idxes_fn)
            sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
        else:
            n_sampling = 1000
            pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
            sampled_verts_idxes = pts_fps_idx
            np.save(robo_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
        # else:
        #     sampled_verts_idxes = None
        self.robo_hand_faces = self.robot_agent.robot_faces
        
        
        ## sampled verts idxes ## 
        self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- # # --- robot hand mani asset? ##  ## robot hand mani asse ##
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m 
        #### retargeting is also a problem here ####
        
        
        ''' Load the mano hand model '''
        model_path_mano = self.conf['model.mano_sim_model_path']
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        self.robo_hand_faces = self.mano_agent.robot_faces
        
        # if self.use_mano_hand_for_test: ##
        #     self.robo_hand_faces = self.hand_faces
        ## 
        
        ## start expanding the current visual pts ##
        print(f"Start expanding the current visual pts...")
        expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        
        self.expanded_visual_pts_nn = expanded_visual_pts.size(0)

        ## expanded_visual_pts of the expanded visual pts #
        expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        print(f"Saving expanded visual pts with shape {expanded_visual_pts.size()} to {expanded_visual_pts_sv_fn}")
        np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy) # 
        
        
        nn_substeps = 10
        
        ## 
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        # self.hand_faces # 

        
        ''' Expnad the current visual points ''' 
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        # params_to_train += list(self.robot_actions.parameters())
        
        self.mano_robot_delta_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        
        self.mano_robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # params_to_train += list(self.robot_init_states.parameters())
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        # params_to_train += list(self.robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        # params_to_train += list(self.robot_glb_trans.parameters())   
        
        ## 
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60, # embedding; a realistic thing # # ## so the optimizable modle deisgn --- approxmimate what you see and approximate the target simulator ## # at a distance; the asymmetric contact froces spring ks -- all of them wold affect model's behaviours ## ## mao robot glb 
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        
        
        
        self.mano_expanded_actuator_delta_offset = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_delta_offset.weight)
        # params_to_train += list(self.mano_expanded_actuator_delta_offset.parameters())
        
        
        self.mano_expanded_actuator_friction_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_friction_forces.weight)
        
        ##### expanded actuators pointact jforces ### 
        self.mano_expanded_actuator_pointact_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_forces.weight)
        
        self.mano_expanded_actuator_pointact_damping_coefs = nn.Embedding(
            num_embeddings=10, embedding_dim=1
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_damping_coefs.weight)
        

        ## load mano states and actions ##
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            if 'mano_robot_init_states' in optimized_init_actions_ckpt:
                self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
            if 'mano_robot_glb_rotation' in optimized_init_actions_ckpt:
                self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])

            if 'mano_robot_states' in optimized_init_actions_ckpt:
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            
            if 'mano_robot_actions' in optimized_init_actions_ckpt:
                self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['mano_robot_actions'])    
            
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            if 'expanded_actuator_friction_forces' in optimized_init_actions_ckpt:
                try:
                    self.mano_expanded_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_friction_forces'])
                except:
                    pass
            #### actuator point forces and actuator point offsets ####
            if 'mano_expanded_actuator_delta_offset' in optimized_init_actions_ckpt:
                print(f"loading mano_expanded_actuator_delta_offset...")
                self.mano_expanded_actuator_delta_offset.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_delta_offset'])
            if 'mano_expanded_actuator_pointact_forces' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_forces'])
            if 'mano_expanded_actuator_pointact_damping_coefs' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_damping_coefs.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_damping_coefs'])



        ## load 
        ''' parameters for the real robot hand '''
        self.robot_actions = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_actions.weight)
        params_to_train += list(self.robot_actions.parameters())
        
        
        # self.robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        self.robot_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_states.weight)
        params_to_train += list(self.robot_states.parameters())
        
        self.robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_init_states.weight)
        params_to_train += list(self.robot_init_states.parameters())
        
        ## robot glb rotations ##
        self.robot_glb_rotation = nn.Embedding( ## robot hand rotation 
            num_embeddings=num_steps, embedding_dim=4
        ).cuda()
        self.robot_glb_rotation.weight.data[:, 0] = 1.
        self.robot_glb_rotation.weight.data[:, 1:] = 0.
        
        self.robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_glb_trans.weight)
        
        
        # ### local minimum -> ## robot 
        self.robot_actuator_friction_forces = nn.Embedding( # frictional forces ##
            num_embeddings=365428 * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_actuator_friction_forces.weight)
        
        
        
        # if len(self.load_optimized_init_transformations) > 0:
        #     print(f"[Robot] Loading optimized init transformations from {self.load_optimized_init_transformations}")
        #     cur_optimized_init_actions_fn = self.load_optimized_init_transformations
        #     # cur_optimized_init_actions = # optimized init states # ## robot init states ##
        #     optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
        #     try:
        #         self.robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
        #     except:
        #         pass
        #     self.robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        #     if 'robot_delta_states' in optimized_init_actions_ckpt:
        #         try:
        #             self.robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
        #         except:
        #             pass
        #     if 'robot_states' in optimized_init_actions_ckpt:
        #         self.robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        #     # if 'robot_delta_states'  ## robot delta states ##
        #     # self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
        #     # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
        #     self.robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
        
        
        if self.hand_type == "redmax_hand":
            self.maxx_robo_pts = 25.
            self.minn_robo_pts = -15.
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.5437551664260203
        else:
            self.minn_robo_pts = -0.1
            self.maxx_robo_pts = 0.2
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.437551664260203
        ## for grab ##
        self.mult_const_after_cent = self.mult_const_after_cent / 3. * 0.9507
        
        
        ### figners for the finger retargeting approach ###
        self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        # self.robot_fingers = [3591, 4768, 6358, 10228, 6629, 10566, 5631, 9673]
        self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877]
        # self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        if self.hand_type == "redmax_hand":
            self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
            self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        
        # self.mano_mult_const_after_cent = 3.
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']


        self.nn_ts = self.nn_timesteps - 1

        ''' parameters for the real robot hand ''' 
        
        
        self.timestep_to_active_mesh = {}
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        
        self.timestep_to_active_mesh_w_delta_states = {}
        
        
        ### mano_expanded_actuator_pointact_forces ### 
        ### timestep_to_actuator_points_vels ###
        ### timestep_to_actuator_points_offsets ###
        self.mass_point_mass = 1.0
        
        self.timestep_to_actuator_points_vels = {}
        self.timestep_to_actuator_points_passive_forces = {}
        
        self.timestep_to_actuator_points_offsets = {}
        time_cons = 0.0005
        pointset_expansion_alpha = 0.1
        
        # ### 
        with torch.no_grad():
            cur_vel_damping_coef = self.mano_expanded_actuator_pointact_damping_coefs(torch.zeros((1,), dtype=torch.long).cuda()).squeeze(0) ### velocity damping coef 
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                ''' Get rotations, translations, and actions of the current robot ''' ## mano robot glb rotation 
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7) # mano glb rot
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                ### motivate via states ####
                # link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                # self.mano_agent.set_init_states_target_value(link_cur_states)
                ### motivate via states ####
                
                
                ### motivate via actions ####
                link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=None, sampled_visual_pts_joint_idxes=None)
                ### motivate via actions ####
                
                
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True) 

                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                # cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                cur_visual_pts_forces = self.mano_expanded_actuator_pointact_forces(cur_visual_pts_idxes) ## get vsual pts forces ### 
                
                ## cur visual pts forces ## ## cur visual pts forces ##
                cur_visual_pts_forces = cur_visual_pts_forces * pointset_expansion_alpha ## pts forces that combines pts forces nad the alpha ##
                
                ## --- linear damping here ? ## 
                cur_visual_pts_accs = cur_visual_pts_forces * time_cons / self.mass_point_mass ### get the mass pont accs ## 
                if cur_ts == 0:
                    cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                else:
                    prev_visual_pts_vels = self.timestep_to_actuator_points_vels[cur_ts - 1] ### nn_pts x 3 ##  ## nn_pts x 3 ##
                    cur_visual_pts_accs = cur_visual_pts_accs - cur_vel_damping_coef * prev_visual_pts_vels ### nn_pts x 3 ## ## prev visual pts vels ## ## prev visual pts vels ##
                    cur_visual_pts_vels = prev_visual_pts_vels + cur_visual_pts_accs * time_cons ## nn_pts x 3 ##  ## get the current vels --- cur_vels = prev_vels + cur_acc * time_cons ##
                self.timestep_to_actuator_points_vels[cur_ts] = cur_visual_pts_vels.detach().clone() # 
                cur_visual_pts_offsets = cur_visual_pts_vels * time_cons
                # 
                if cur_ts > 0:
                    prev_visual_pts_offset = self.timestep_to_actuator_points_offsets[cur_ts - 1]
                    cur_visual_pts_offsets = prev_visual_pts_offset + cur_visual_pts_offsets ## add to the visual pts offsets ##
                self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone() ### pts offset ###
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                ### transform by the glboal transformation and the translation ### ## cur visual pts ## ## contiguous() ##
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## transformed pts ## 
                
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offsets
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_w_delta_states[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_opt_ours_sim[cur_ts] = cur_visual_pts.detach()
                
        self.iter_step = 0
        
        ''' Set redmax robot actions '''


        params_to_train_kines = []
        # params_to_train_kines += list(self.mano_robot_glb_rotation.parameters())
        # params_to_train_kines += list(self.mano_robot_glb_trans.parameters())
        # params_to_train_kines += list(self.mano_robot_delta_states.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        # 
        # params_to_train_kines += list(self.robot_states.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        # train_point_set_dyn
        params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        # params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        
        # mano_expanded_actuator_pointact_forces ## st the forces ##
        params_to_train_kines += list(self.mano_expanded_actuator_pointact_forces.parameters())
        
        
        
        # can tray the optimizer
        # if self.use_LBFGS:
        #     self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=self.learning_rate)
        # else:
        #     self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
        
        
        mano_expanded_actuator_delta_offset_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone().detach()
        mano_expanded_actuator_pointact_forces_ori = self.mano_expanded_actuator_pointact_forces.weight.data.clone().detach()
        
        print(f"optimize_rules: {self.optimize_rules}")
        if self.optimize_rules:
            print(f"optimize_rules: {self.optimize_rules}")
            params_to_train_kines = []
            params_to_train_kines += list(self.other_bending_network.parameters())
            # self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
            # if self.use_LBFGS:
            #     self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=self.learning_rate)
            # else:
            self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate)
            # self.kines_optimizer = torch.optim.LBFGS(params_to_train_kines, lr=1e-2)
        
        
        
        ## policy and the controller ##
        # self.other_bending_network.spring_ks_values.weight.data[:, :] = 0.1395
        # self.other_bending_network.spring_ks_values.weight.data[0, :] = 0.1395
        # self.other_bending_network.spring_ks_values.weight.data[1, :] = 0.00
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 10.0
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 1000.0
        # self.other_bending_network.inertia_div_factor.weight.data[:, :] = 100.0
        
        
        # load_redmax_robot_actions_fn = "/data3/datasets/diffsim/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v13__hieoptrotorules_penalty_friction_ct_01_thres_0001_avg_optrobo_nglbtrans_reg001_manorules_projk_0_nsp_res_sqrstiff_preprojfri_projk_0_optrules_diffh_subs_10_/checkpoints/redmax_robot_actions_ckpt_001900.pth"
        # if len(load_redmax_robot_actions_fn) > 0:
        #     redmax_robot_actions_ckpt = torch.load(load_redmax_robot_actions_fn, map_location=self.device, )
            # self.redmax_robot_actions.load_state_dict(redmax_robot_actions_ckpt['redmax_robot_actions'])

        ''' prepare for keeping the original global rotations, trans, and states '''
        # ori_mano_robot_glb_rot = self.mano_robot_glb_rotation.weight.data.clone()
        # ori_mano_robot_glb_trans = self.mano_robot_glb_trans.weight.data.clone()
        # ori_mano_robot_delta_states = self.mano_robot_delta_states.weight.data.clone()
        
        
        self.iter_step = 0
        
        self.ts_to_dyn_mano_pts_th = {}


        for i_iter in tqdm(range(1000)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep #
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # # # #
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            # correspondence_pts_idxes = None
            
            # tot_penetrating_depth_penalty = []
            # tot_ragged_dist = []
            # tot_delta_offset_reg_motion = []
            # tot_dist_mano_visual_ori_to_cur = []
            # tot_reg_loss = []
            # tot_diff_cur_states_to_ref_states = []
            # tot_diff_tangential_forces = []
            penetration_forces = None ###
            sampled_visual_pts_joint_idxes = None
            
            # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            self.timestep_to_anchored_mano_pts = {}
            
            ## 
            self.ts_to_contact_passive_normals = {}
            self.ts_to_passive_normals = {}
            self.ts_to_passive_pts = {}
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            self.ts_to_dyn_mano_pts = {}
            
            self.timestep_to_actuator_points_passive_forces = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            # tot_grad_qs = []
            
            robo_intermediates_states = []
            
            tot_penetration_depth = []
            
            robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            tot_interpenetration_nns = []
            
            tot_diff_cur_visual_pts_offsets_with_ori = []
            
            tot_summ_grad_mano_expanded_actuator_delta_offset_weight = []
            tot_summ_grad_mano_expanded_actuator_pointact_forces_weight = []
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            
            
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                # tot_redmax_actions = []
                # actions = {}

                self.free_def_bending_weight = 0.0

                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone()
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot)
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)


                # # # cur_ts_delta_rot, cur_ts_redmax_robot_trans # # #
                # cur_glb_rot = torch.matmul(cur_ts_delta_rot, cur_glb_rot)
                cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans # redmax robot transj## 

                # link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                # self.mano_agent.set_init_states_target_value(link_cur_states)
                
                
                ### motivate via states ####
                # link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                # self.mano_agent.set_init_states_target_value(link_cur_states)
                ### motivate via states ####
                
                
                ### motivate via actions ####
                link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes)
                ### motivate via actions ####
                
                
                cur_visual_pts, visual_pts_joint_idxes = self.mano_agent.get_init_state_visual_pts(expanded_pts=True, ret_joint_idxes=True)  # init state visual pts #
                
                ## visual pts sampled ##
                
                ## visual pts sampled ##
                cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts # # expanded pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                # cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                
                
                if self.drive_pointset == "actions":
                    ''' Act-React-driven point motions '''
                    ###### get the point offset via actuation forces and reaction forces ######
                    ### actuation forces at this timestep ###
                    cur_visual_pts_forces = self.mano_expanded_actuator_pointact_forces(cur_visual_pts_idxes) * 1e7 ## get vsual pts forces ### 
                    
                    # if cur_ts > 0 and (cur_ts - 1 in self.timestep_to_actuator_points_passive_forces):
                    #     cur_visual_pts_passive_forces = self.timestep_to_actuator_points_passive_forces[cur_ts - 1] ## nn_visual_pts x 3 ##
                    #     cur_visual_pts_forces = cur_visual_pts_forces + cur_visual_pts_passive_forces ## two forces ### 
                        
                    cur_visual_pts_forces = cur_visual_pts_forces * pointset_expansion_alpha
                        
                    ## --- linear damping here ? ## 
                    cur_visual_pts_accs = cur_visual_pts_forces / self.mass_point_mass ### get the mass pont accs ## 
                    if cur_ts == 0:
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    else: # visual pts acc -> visual pts vels #
                        # prev_visual_pts_vels = self.timestep_to_actuator_points_vels[cur_ts - 1] ### nn_pts x 3 ## 
                        # cur_visual_pts_accs = cur_visual_pts_accs - cur_vel_damping_coef * prev_visual_pts_vels ### nn_pts x 3 ##
                        # cur_visual_pts_vels = prev_visual_pts_vels + cur_visual_pts_accs * time_cons ## nn_pts x 3 ## 
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    self.timestep_to_actuator_points_vels[cur_ts] = cur_visual_pts_vels.detach().clone()
                    cur_visual_pts_offsets = cur_visual_pts_vels * time_cons
                    # 
                    # if cur_ts > 0:
                    #     prev_visual_pts_offset = self.timestep_to_actuator_points_offsets[cur_ts - 1]
                    #     cur_visual_pts_offsets = prev_visual_pts_offset + cur_visual_pts_offsets
                        
                    
                    # train_pointset_acts_via_deltas, diff_cur_visual_pts_offsets_with_ori
                    # cur_visual_pts_offsets_from_delta = mano_expanded_actuator_delta_offset_ori[cur_ts]
                    # cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset.weight.data[cur_ts].detach()
                    cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes).detach()
                    ## 
                    diff_cur_visual_pts_offsets_with_ori = torch.sum((cur_visual_pts_offsets - cur_visual_pts_offsets_from_delta) ** 2, dim=-1).mean() ## mean of the avg offset differences ##
                    tot_diff_cur_visual_pts_offsets_with_ori.append(diff_cur_visual_pts_offsets_with_ori.item())
                        
                    
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ###### get the point offset via actuation forces and reaction forces ######
                    ''' Act-React-driven point motions '''
                elif self.drive_pointset == "states":
                    ''' Offset-driven point motions '''
                    ## points should be able to manipulate the object accordingly ###
                    ### we should avoid the penetrations between points and the object ###
                    ### we should restrict relative point displacement / the point offsets at each timestep to relatively small values ###
                    ## -> so we have three losses for the delta offset optimization ##
                    cur_visual_pts_offsets = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes)
                    # cur_visual_pts_offsets = cur_visual_pts_offsets * 10
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ''' Offset-driven point motions '''
                    
                else:
                    raise ValueError(f"Unknown drive_pointset: {self.drive_pointset}")
                
                
                
                ### add forces 
                cur_visual_pts_friction_forces = self.mano_expanded_actuator_friction_forces(cur_visual_pts_idxes)

                ### transform the visual pts ### ## fricton forces ##
                # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                # cur_visual_pts = cur_visual_pts * 2. - 1.  # cur visual pts # 
                # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # # mult_const #
                
                #### cur visual pts ####
                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                cur_dyn_mano_pts = cur_dyn_mano_pts * self.mano_mult_const_after_cent
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                cur_dyn_mano_pts = torch.matmul(cur_rot, cur_dyn_mano_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                
                ## update the visual points positions ##
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offsets
                
                # diff_redmax_visual_pts_with_ori_visual_pts = torch.sum(
                #     (cur_visual_pts[sampled_verts_idxes] - self.timestep_to_active_mesh_opt_ours_sim[cur_ts].detach()) ** 2, dim=-1
                # )
                # diff_redmax_visual_pts_with_ori_visual_pts = diff_redmax_visual_pts_with_ori_visual_pts.mean()
                
                # train the friction net? how to train the friction net? #
                # if self.use_mano_hand_for_test:
                #     self.timestep_to_active_mesh[cur_ts] = self.rhand_verts[cur_ts] # .detach()
                # else:
                
                ### timesttep to active mesh ###
                # timestep_to_anchored_mano_pts, timestep_to_raw_active_meshes # 
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts[cur_ts] = cur_dyn_mano_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts_th[cur_ts] = cur_dyn_mano_pts
                
                ## ragged dist ##
                # # ragged_dist = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # # dist_transformed_expanded_visual_pts_to_ori_visual_pts = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # # diff_cur_states_to_ref_states = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                # cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                # robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # self.robot_agent.set_init_states_target_value(robo_links_states)
                # cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                

                # ### transform the visual pts ###
                # cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                # cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                # cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                
                
                # cur_rot = cur_robo_glb_rot
                # cur_trans = cur_glb_trans
                
                # timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                # timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                # ### transform by the glboal transformation and the translation ###
                # cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                
                # self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.clone() # robo visual pts #
                
                # if self.hand_type == 'redmax_hand':
                #     cur_robo_visual_pts = cur_robo_visual_pts[sampled_verts_idxes] # 
                
                
                
                self.free_def_bending_weight = 0.0
                # self.free_def_bending_weight = 0.5
                
                # if i_iter == 0 and cur_ts == 0: ## for the tiantianquan sequence ##
                #     dist_robot_pts_to_mano_pts = torch.sum(
                #         (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                #     )
                #     minn_dist_robot_pts_to_mano_pts, correspondence_pts_idxes = torch.min(dist_robot_pts_to_mano_pts, dim=-1)
                #     minn_dist_robot_pts_to_mano_pts = torch.sqrt(minn_dist_robot_pts_to_mano_pts)
                #     # dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.01
                #     dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.005
                    
                #     corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                    
                #     dist_corr_correspondence_pts_to_mano_visual_pts = torch.sum(
                #         (corr_correspondence_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                #     )
                #     dist_corr_correspondence_pts_to_mano_visual_pts = torch.sqrt(dist_corr_correspondence_pts_to_mano_visual_pts)
                #     minn_dist_to_corr_pts, _ = torch.min(dist_corr_correspondence_pts_to_mano_visual_pts, dim=0)
                #     anchored_mano_visual_pts = minn_dist_to_corr_pts < 0.005
                    
                # corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                # # corr_robo = cur_visual_pts[sampled_verts_idxes]
                # cd_robo_pts_to_corr_mano_pts = torch.sum(
                #     (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts[anchored_mano_visual_pts].unsqueeze(0)) ** 2, dim=-1
                # )
                
                # self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts[anchored_mano_visual_pts].detach().cpu().numpy()
                
                # cd_robo_to_mano, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=-1)
                # cd_mano_to_robo, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=0)
                # # diff_robo_to_corr_mano_pts = cd_mano_to_robo.mean()
                # diff_robo_to_corr_mano_pts = cd_robo_to_mano.mean()
                
                # mano_fingers = self.rhand_verts[cur_ts][self.mano_fingers]
                
                # if self.hand_type == 'redmax_hand':
                #     # sampled_verts_idxes
                #     robo_fingers = cur_robo_visual_pts[sampled_verts_idxes][self.robot_fingers]
                # else:
                #     robo_fingers = cur_robo_visual_pts[self.robot_fingers]
                
                # pure finger tracking ##
                # pure_finger_tracking_loss = torch.sum((mano_fingers - robo_fingers) ** 2)
                
                
                # diff_robo_to_corr_mano_pts_finger_tracking = torch.sum(
                #     (corr_correspondence_pts - cur_robo_visual_pts) ** 2, dim=-1
                # )
                # diff_robo_to_corr_mano_pts_finger_tracking = diff_robo_to_corr_mano_pts_finger_tracking[dist_smaller_than_thres]
                # diff_robo_to_corr_mano_pts_finger_tracking = diff_robo_to_corr_mano_pts_finger_tracking.mean()
                
                # loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef + pure_finger_tracking_loss * 0.5 # + diff_robo_to_corr_mano_pts_finger_tracking * self.finger_tracking_loss_coef
                
                ## TODO: add the glboal retargeting using fingers before conducting this approach 
                
                
                # def evaluate_tracking_loss():
                #     self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
                    
                #     ## 
                #     # init states  # 
                #     # cur_ts % mano_nn_substeps == 0:
                #     if (cur_ts + 1) % mano_nn_substeps == 0:
                #         cur_passive_big_ts = cur_ts // mano_nn_substeps
                #         in_func_tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                #         # tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                #     else:
                #         in_func_tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                #     return in_func_tracking_loss

                
                ''' cache contact pair set for exporting contact information '''
                if contact_pairs_set is None:
                    self.contact_pairs_set = None
                else:
                    self.contact_pairs_set = contact_pairs_set.copy()
                
                
                ts_to_act_mesh = self.timestep_to_active_mesh
                # ts_to_act_mesh = self.ts_to_dyn_mano_pts_th
                # self.timestep_to_active_mesh = self.ts_to_dyn_mano_pts_th
                
                ## to passive mesh normals; to friction forces ##
                # print(self.timestep_to_active_mesh[cur_ts].size(), cur_visual_pts_friction_forces.size()) ## to active mesh ##
                # contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=ts_to_act_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set, pts_frictional_forces=cur_visual_pts_friction_forces)
                
                contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=ts_to_act_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set)

                ### train with force to active ##
                if self.train_with_forces_to_active and (not self.use_mano_inputs):
                    # penetration_forces #
                    if torch.sum(self.other_bending_network.penetrating_indicator.float()) > 0.5:
                        net_penetrating_forces = self.other_bending_network.penetrating_forces
                        net_penetrating_points = self.other_bending_network.penetrating_points

                        
                        # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
                        self.timestep_to_penetration_points[cur_ts] = net_penetrating_points.detach().cpu().numpy()
                        self.timestep_to_penetration_points_forces[cur_ts] = net_penetrating_forces.detach().cpu().numpy()
                        
                        
                        ### transform the visual pts ###
                        # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                        # cur_visual_pts = cur_visual_pts * 2. - 1.
                        # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # mult_const #
                        
                        # sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[finger_sampled_idxes][self.other_bending_network.penetrating_indicator]
                        
                        sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[self.other_bending_network.penetrating_indicator]
                        
                        ## from net penetration forces to to the 
                         
                        ### get the passvie force for each point ## ## 
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone() ## for
                        
                        net_penetrating_forces = torch.matmul(
                            cur_rot.transpose(1, 0), net_penetrating_forces.transpose(1, 0)
                        ).transpose(1, 0)
                        # net_penetrating_forces = net_penetrating_forces / self.mult_const_after_cent
                        # net_penetrating_forces = net_penetrating_forces / 2
                        # net_penetrating_forces = net_penetrating_forces * self.extent_robo_pts
                        
                        net_penetrating_forces = (1.0 - pointset_expansion_alpha) * net_penetrating_forces
                        
                        net_penetrating_points = torch.matmul(
                            cur_rot.transpose(1, 0), (net_penetrating_points - cur_trans.unsqueeze(0)).transpose(1, 0)
                        ).transpose(1, 0)
                        # net_penetrating_points = net_penetrating_points / self.mult_const_after_cent
                        # net_penetrating_points = (net_penetrating_points + 1.) / 2. # penetrating points #
                        # net_penetrating_points = (net_penetrating_points * self.extent_robo_pts) + self.minn_robo_pts
                        
                        penetration_forces = net_penetrating_forces
                        # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        # 
                    else:
                        penetration_forces = None
                        sampled_visual_pts_joint_idxes = None
                        # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        ''' the bending network still have this property and we can get force values here for the expanded visual points '''
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone()  
                        
                # if contact_pairs_set is not None:
                #     self.contact_pairs_sets[cur_ts] = contact_pairs_set.copy()
                
                # # contact force d ## ts to the passive normals ## 
                # self.ts_to_contact_passive_normals[cur_ts] = self.other_bending_network.tot_contact_passive_normals.detach().cpu().numpy()
                # self.ts_to_passive_pts[cur_ts] = self.other_bending_network.cur_passive_obj_verts.detach().cpu().numpy()
                # self.ts_to_passive_normals[cur_ts] = self.other_bending_network.cur_passive_obj_ns.detach().cpu().numpy()
                # self.ts_to_contact_force_d[cur_ts] = self.other_bending_network.contact_force_d.detach().cpu().numpy()
                # self.ts_to_penalty_frictions[cur_ts] = self.other_bending_network.penalty_friction_tangential_forces.detach().cpu().numpy()
                # if self.other_bending_network.penalty_based_friction_forces is not None:
                #     self.ts_to_penalty_disp_pts[cur_ts] = self.other_bending_network.penalty_based_friction_forces.detach().cpu().numpy()
                
                
                
                # if self.optimize_with_intermediates:
                #     tracking_loss = self.compute_loss_optimized_transformations(cur_ts + 1) # 
                # else:
                #     tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
                ##  expanded points set dyn ## 
                
                # # cur_ts % mano_nn_substeps == 0: # 
                if (cur_ts + 1) % mano_nn_substeps == 0:
                    cur_passive_big_ts = cur_ts // mano_nn_substeps
                    ## compute optimized transformations ##
                    tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                    tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                else:
                    tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()

                # # hand_tracking_loss = torch.sum( ## delta states? ##
                # #     (self.timestep_to_active_mesh_w_delta_states[cur_ts] - cur_visual_pts) ** 2, dim=-1
                # # )
                # # hand_tracking_loss = hand_tracking_loss.mean()
                
                
                # # loss = tracking_loss + self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                # # diff_redmax_visual_pts_with_ori_visual_pts.backward()
                ### penetrating depth penalty ### penetrating depth penatly ### ## penetration depth ##
                penetraton_penalty = self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                
                tot_penetration_depth.append(penetraton_penalty.detach().item())
                
                # smaller_than_zero_level_set_indicator
                cur_interpenetration_nns = self.other_bending_network.smaller_than_zero_level_set_indicator.float().sum()
                
                tot_interpenetration_nns.append(cur_interpenetration_nns)
                
                # diff_hand_tracking = torch.zeros((1,), dtype=torch.float32).cuda().mean() ## 
                
                
                # # kinematics_proj_loss = kinematics_trans_diff + penetraton_penalty + diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss
                
                # # if self.use_mano_hand_for_test: ## only the kinematics mano hand is optimized here ##
                # #     kinematics_proj_loss = tracking_loss
                
                # # kinematics_proj_loss = hand_tracking_loss * 1e2 ## 1e2 and the 1e2 ## 
                
                # kinematics_proj_loss = diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss + penetraton_penalty
                
                # kinematics_proj_loss = loss_finger_tracking # + tracking_loss + penetraton_penalty
                
                reg_delta_offset_loss = torch.sum(
                    (mano_expanded_actuator_delta_offset_ori - self.mano_expanded_actuator_delta_offset.weight.data) ** 2, dim=-1
                )
                reg_delta_offset_loss = reg_delta_offset_loss.mean()
                
                reg_act_force_loss = torch.sum(
                    (mano_expanded_actuator_pointact_forces_ori - self.mano_expanded_actuator_pointact_forces.weight.data) ** 2, dim=-1
                )
                reg_act_force_loss = reg_act_force_loss.mean()
                
                # motion_reg_loss_coef ## delta offset loss and the act force loss ###
                reg_delta_offset_loss = (reg_delta_offset_loss + reg_act_force_loss) * self.motion_reg_loss_coef
                
                
                ### tracking loss and the penetration penalty ###
                ## tracking; penetrations; delta offset ##
                
                # train_pointset_acts_via_deltas, diff_cur_visual_pts_offsets_with_ori
                
                if self.fix_obj:
                    kinematics_proj_loss = penetraton_penalty + reg_delta_offset_loss * 100
                    if self.train_pointset_acts_via_deltas and self.drive_pointset == "actions":
                        kinematics_proj_loss = reg_delta_offset_loss * 100 + diff_cur_visual_pts_offsets_with_ori * 100
                else:
                    ## 
                    # kinematics_proj_loss = tracking_loss + penetraton_penalty + reg_delta_offset_loss * 100
                    kinematics_proj_loss = tracking_loss + reg_delta_offset_loss * 100
                    if self.train_pointset_acts_via_deltas: ## via deltas ##
                        kinematics_proj_loss = kinematics_proj_loss + diff_cur_visual_pts_offsets_with_ori
                
                ### kinematics proj loss ###
                loss = kinematics_proj_loss


                self.kines_optimizer.zero_grad()
                
                try:
                    
                    # kinematics_proj_loss = kinematics_proj_loss * 1e15
                    if self.optimize_rules:
                        # kinematics_proj_loss = kinematics_proj_loss * 10
                        kinematics_proj_loss = kinematics_proj_loss * 100
                        
                    kinematics_proj_loss.backward(retain_graph=True)
                    
                    
                    ## mano_expanded_actuator_pointact_forces ## --> get the gradient of the pointact_forces here ##
                    if self.mano_expanded_actuator_pointact_forces.weight.grad is not None:
                        grad_mano_expanded_actuator_pointact_forces_weight = self.mano_expanded_actuator_pointact_forces.weight.grad.data
                        summ_grad_mano_expanded_actuator_pointact_forces_weight = torch.sum(grad_mano_expanded_actuator_pointact_forces_weight).item()
                        tot_summ_grad_mano_expanded_actuator_pointact_forces_weight.append(summ_grad_mano_expanded_actuator_pointact_forces_weight)
                        # print(f"i_iter: {i_iter}, cur_ts: {cur_ts}, grad_pointact_forces: {summ_grad_mano_expanded_actuator_pointact_forces_weight}") ## forces weight -> expanded forces weight ##
                    elif self.mano_expanded_actuator_delta_offset.weight.grad is not None:
                        grad_mano_expanded_actuator_delta_offset_weight = self.mano_expanded_actuator_delta_offset.weight.grad.data
                        summ_grad_mano_expanded_actuator_delta_offset_weight = torch.sum(grad_mano_expanded_actuator_delta_offset_weight).item()
                        tot_summ_grad_mano_expanded_actuator_delta_offset_weight.append(summ_grad_mano_expanded_actuator_delta_offset_weight)
                        # print(f"i_iter: {i_iter}, cur_ts: {cur_ts}, grad_pointact_offset: {summ_grad_mano_expanded_actuator_delta_offset_weight}") 
                    
                    self.kines_optimizer.step()
                except:
                    pass
                
                
                ### get the gradient information ###
                # if self.iter_step > 1239 and self.mano_expanded_actuator_delta_offset.weight.grad is not None:
                #     grad_mano_expanded_actuator_delta_offset = self.mano_expanded_actuator_delta_offset.weight.grad.data
                #     grad_mano_expanded_actuator_delta_offset = torch.sum(grad_mano_expanded_actuator_delta_offset)
                #     print(f"iter_step: {self.iter_step}, grad_offset: {grad_mano_expanded_actuator_delta_offset}") 
                # if self.iter_step > 1239 and self.mano_expanded_actuator_friction_forces.weight.grad is not None:
                #     grad_mano_expanded_actuator_friction_forces = self.mano_expanded_actuator_friction_forces.weight.grad.data
                #     grad_mano_expanded_actuator_friction_forces = torch.sum(grad_mano_expanded_actuator_friction_forces)
                #     print(f"iter_step: {self.iter_step}, grad_friction_forces: {grad_mano_expanded_actuator_friction_forces}")
                    
                # if self.iter_step > 1239 and cur_visual_pts.grad is not None:
                #     grad_cur_visual_pts = torch.sum(cur_visual_pts.grad.data)
                #     print(f"iter_step: {self.iter_step}, grad_cur_visual_pts: {grad_cur_visual_pts}")
                
                
                
                # if self.use_LBFGS:
                #     self.kines_optimizer.step(evaluate_tracking_loss) # 
                # else:
                #     self.kines_optimizer.step()

                # 
                # tracking_loss.backward(retain_graph=True)
                # if self.use_LBFGS:
                #     self.other_bending_network.reset_timestep_to_quantities(cur_ts)
                
                
                robot_states_actions_diff_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                robo_actions_diff_loss.append(reg_delta_offset_loss.item())

                
                tot_losses.append(loss.detach().item())
                # tot_penalty_dot_forces_normals.append(cur_penalty_dot_forces_normals.detach().item())
                # tot_penalty_friction_constraint.append(cur_penalty_friction_constraint.detach().item())
                
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                # if self.iter_step % self.save_freq == 0:
                if self.iter_step % 2000 == 0:
                    self.save_checkpoint()
                self.update_learning_rate()
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            if len(tot_tracking_loss) > 0:
                tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            else:
                tot_tracking_loss = 0.0
            if len(tot_penetration_depth) > 0:
                tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            else:
                tot_penetration_depth = 0.0
            robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            if len(mano_tracking_loss) > 0:
                mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))
            else:
                mano_tracking_loss = 0.0
                
            if len(tot_diff_cur_visual_pts_offsets_with_ori) > 0:
                diff_cur_visual_pts_offsets_with_ori = sum(tot_diff_cur_visual_pts_offsets_with_ori) / float(len(tot_diff_cur_visual_pts_offsets_with_ori))
            else:
                diff_cur_visual_pts_offsets_with_ori = 0.0
            
            if len(tot_summ_grad_mano_expanded_actuator_pointact_forces_weight) > 0:
                summ_grad_mano_expanded_actuator_pointact_forces_weight = sum(tot_summ_grad_mano_expanded_actuator_pointact_forces_weight) / float(len(tot_summ_grad_mano_expanded_actuator_pointact_forces_weight))
            else:
                summ_grad_mano_expanded_actuator_pointact_forces_weight = 0.0
                
            if len(tot_summ_grad_mano_expanded_actuator_delta_offset_weight) > 0:
                summ_grad_mano_expanded_actuator_delta_offset_weight = sum(tot_summ_grad_mano_expanded_actuator_delta_offset_weight) / float(len(tot_summ_grad_mano_expanded_actuator_delta_offset_weight))
            else:
                summ_grad_mano_expanded_actuator_delta_offset_weight = 0.0
            
            avg_tot_interpenetration_nns = float(sum(tot_interpenetration_nns) ) / float(len(tot_interpenetration_nns))
            
            
            ##### logging the losses information ##### 
            cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} diff_cur_visual_pts_offsets_with_ori = {} penetration = {} / {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, diff_cur_visual_pts_offsets_with_ori, avg_tot_interpenetration_nns, self.timestep_to_active_mesh[0].size(0), self.optimizer.param_groups[0]['lr'])
                
            print(cur_log_sv_str)
            ##### logging the losses information #####
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} pointact_forces_grad = {} point_delta_offset_grad = {}  penetration = {} / {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, summ_grad_mano_expanded_actuator_pointact_forces_weight, summ_grad_mano_expanded_actuator_delta_offset_weight, avg_tot_interpenetration_nns, self.timestep_to_active_mesh[0].size(0), self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')


            if i_iter % self.val_mesh_freq == 0:
                self.validate_mesh_robo_g()
                self.validate_mesh_robo()
                self.validate_contact_info_robo()
                
            
            torch.cuda.empty_cache()
    
    
    
    ''' GRAB clips --- expanded point set and expanded points for retargeting ''' 
    def train_point_set_retargeting(self, ):
        ## ## GRAB clips ## ##
        # states -> the robot actions --- in this sim ##
        # chagne # # mano notjmano but the mano ---> optimize the mano delta states? # 
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # update learning rrate # 
        # robot actions ##
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        # load # # load the robot hand # # load
        ''' Load the robot hand '''
        model_path = self.conf['model.sim_model_path'] # 
        # robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        self.hand_type = "redmax_hand" if "redmax" in model_path else "shadow_hand"
        # self.hand_type = "redmax_hand"
        if model_path.endswith(".xml"):
            # self.hand_type = "redmax_hand"
            robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        else:
            # self.hand_type = "shadow_hand"
            robot_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path, args=None)
        self.robot_agent = robot_agent
        
        robo_init_verts = self.robot_agent.robot_pts
        if self.hand_type == "redmax_hand":
            redmax_sampled_verts_idxes_fn = "redmax_robo_sampled_verts_idxes_new.npy"
            redmax_sampled_verts_idxes_fn = os.path.join("assets", redmax_sampled_verts_idxes_fn)
            if os.path.exists(redmax_sampled_verts_idxes_fn):
                sampled_verts_idxes = np.load(redmax_sampled_verts_idxes_fn)
                sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
            else:
                n_sampling = 1000
                pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
                sampled_verts_idxes = pts_fps_idx
                np.save(redmax_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
            self.sampled_verts_idxes = sampled_verts_idxes
        
        
        self.robo_hand_faces = self.robot_agent.robot_faces
        
        
        ## sampled verts idxes ## 
        # self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- # # --- robot hand mani asset? ##  ## robot hand mani asse ##
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m
        
        
        ''' Load the mano hand model '''
        model_path_mano = self.conf['model.mano_sim_model_path']
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.robo_hand_faces = self.mano_agent.robot_faces
        
        # if self.use_mano_hand_for_test: ## use
        #     self.robo_hand_faces = self.hand_faces
        ## 
        
        ## start expanding the current visual pts ##
        print(f"Start expanding current visual pts...")
        expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        
        self.expanded_visual_pts_nn = expanded_visual_pts.size(0)

        ## expanded_visual_pts of the expanded visual pts #
        expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        print(f"Saving expanded visual pts with shape {expanded_visual_pts.size()} to {expanded_visual_pts_sv_fn}")
        np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy) # 
        
        
        nn_substeps = 10
        
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        # self.hand_faces # 

        
        ''' Expnad the current visual points ''' 
        # expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.expanded_visual_pts_nn = expanded_visual_pts.size(0)
        # expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        # expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        # np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy)
        # # ''' Expnad the current visual points '''  #  # differentiate through the simulator? # # 
        
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        # params_to_train += list(self.robot_actions.parameters())
        
        self.mano_robot_delta_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        
        self.mano_robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # params_to_train += list(self.robot_init_states.parameters())
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        # params_to_train += list(self.robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        # params_to_train += list(self.robot_glb_trans.parameters())   
        
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        
        
        
        self.mano_expanded_actuator_delta_offset = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_delta_offset.weight)
        # params_to_train += list(self.mano_expanded_actuator_delta_offset.parameters())
        
        
        # mano_expanded_actuator_friction_forces, mano_expanded_actuator_delta_offset # 
        self.mano_expanded_actuator_friction_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_friction_forces.weight)
        
        
        ##### expanded actuators pointact jforces ### 
        self.mano_expanded_actuator_pointact_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_forces.weight)
        
        self.mano_expanded_actuator_pointact_damping_coefs = nn.Embedding(
            num_embeddings=10, embedding_dim=1
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_damping_coefs.weight)
        


        ## load mano states and actions ##
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            if 'mano_robot_init_states' in optimized_init_actions_ckpt:
                self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
            if 'mano_robot_glb_rotation' in optimized_init_actions_ckpt:
                self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])

            if 'mano_robot_states' in optimized_init_actions_ckpt:
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            
            if 'mano_robot_actions' in optimized_init_actions_ckpt:
                self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['mano_robot_actions'])    
            
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            if 'expanded_actuator_friction_forces' in optimized_init_actions_ckpt:
                try:
                    self.mano_expanded_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_friction_forces'])
                except:
                    pass
            #### actuator point forces and actuator point offsets ####
            if 'mano_expanded_actuator_delta_offset' in optimized_init_actions_ckpt:
                print(f"loading mano_expanded_actuator_delta_offset...")
                self.mano_expanded_actuator_delta_offset.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_delta_offset'])
            if 'mano_expanded_actuator_pointact_forces' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_forces'])
            if 'mano_expanded_actuator_pointact_damping_coefs' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_damping_coefs.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_damping_coefs'])




        ''' parameters for the real robot hand ''' 
        # # robot actions # # real robot hand ##
        self.robot_actions = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_actions.weight)
        params_to_train += list(self.robot_actions.parameters())
        
        # self.robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        self.robot_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_states.weight)
        params_to_train += list(self.robot_states.parameters())
        
        self.robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_init_states.weight)
        params_to_train += list(self.robot_init_states.parameters())
        
        ## robot glb rotations ##
        self.robot_glb_rotation = nn.Embedding( ## robot hand rotation 
            num_embeddings=num_steps, embedding_dim=4
        ).cuda()
        self.robot_glb_rotation.weight.data[:, 0] = 1.
        self.robot_glb_rotation.weight.data[:, 1:] = 0.
        
        self.robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_glb_trans.weight)
        
        
        # ### local minimum -> ## robot 
        self.robot_actuator_friction_forces = nn.Embedding( # frictional forces ##
            num_embeddings=365428 * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_actuator_friction_forces.weight)
        
        
        ### load optimized init transformations for robot actions ###
        if len(self.load_optimized_init_transformations) > 0:
            print(f"[Robot] Loading optimized init transformations from {self.load_optimized_init_transformations}")
            cur_optimized_init_actions_fn = self.load_optimized_init_transformations
            # cur_optimized_init_actions = # optimized init states # ## robot init states ##
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            try:
                self.robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
            except:
                pass
            self.robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
            if 'robot_delta_states' in optimized_init_actions_ckpt:
                try:
                    self.robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
                except:
                    pass
            if 'robot_states' in optimized_init_actions_ckpt:
                self.robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
            # if 'robot_delta_states'  ## robot delta states ##
            # self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
            
            self.robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])

        
        
        if self.hand_type == "redmax_hand":
            self.maxx_robo_pts = 25.
            self.minn_robo_pts = -15.
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.5437551664260203
        else:
            self.minn_robo_pts = -0.1
            self.maxx_robo_pts = 0.2
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.437551664260203
        ## for grab ##
        self.mult_const_after_cent = self.mult_const_after_cent / 3. * 0.9507
        
        
        # self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        # # self.robot_fingers = [3591, 4768, 6358, 10228, 6629, 10566, 5631, 9673]
        # self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877]
        # # self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        # if self.hand_type == "redmax_hand":
        #     self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        #     self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, ]
        self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877, ]
        
        if self.hand_type == "redmax_hand":
            self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
            self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
            self.robot_fingers = [14670, 321530, 36939, 125930, 200397, 257721, 333438, 338358]
        
        # params_to_train = []
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']
        
        
        self.nn_ts = self.nn_timesteps - 1

        ''' parameters for the real robot hand ''' 
        

        
        self.timestep_to_active_mesh = {}
        # ref_expanded_visual_pts, minn_idx_expanded_visual_pts_to_link_pts #
        # minn_idx_expanded_visual_pts_to_link_pts #
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        
        self.timestep_to_active_mesh_w_delta_states = {}
        
        
        self.mass_point_mass = 1.0
        
        self.timestep_to_actuator_points_vels = {}
        self.timestep_to_actuator_points_passive_forces = {}
        
        
        self.timestep_to_actuator_points_offsets = {}
        time_cons = 0.0005
        pointset_expansion_alpha = 0.1
        
        
        # states -> get states -> only update the acitons #
        with torch.no_grad(): # init them to zero 
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                ''' Get rotations, translations, and actions of the current robot '''
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7) # mano glb rot
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                ## mano robot states ## mano robot states ##
                link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_init_states_target_value(link_cur_states)
                
                ## set init visual pts ##
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True) 

                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                
                cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                ## 
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## transformed pts ## 
                
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offset
                
                # if not self.use_mano_inputs:
                # if self.use_mano_hand_for_test:
                #     self.timestep_to_active_mesh[cur_ts] = self.rhand_verts[cur_ts].detach()
                # else:
                #     self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_w_delta_states[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_opt_ours_sim[cur_ts] = cur_visual_pts.detach()
                
                
                cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.robot_agent.set_init_states_target_value(robo_links_states)
                cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                

                if not self.use_scaled_urdf:
                    ### transform the visual pts ###
                    cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                    cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                
                
                # cur_rot = cur_robo_glb_rot
                # cur_trans = cur_glb_trans
                
                # timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                # timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                ### transform by the glboal transformation and the translation ###
                cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.detach()
                
        self.iter_step = 0
        
        mano_expanded_actuator_delta_offset_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone().detach()
        mano_expanded_actuator_pointact_forces_ori = self.mano_expanded_actuator_pointact_forces.weight.data.clone().detach()
        
        ''' Set redmax robot actions '''
        
        
        # self.optimize_robot = False
        
        self.optimize_robot = True
        
        # self.optimize_anchored_pts = False
        self.optimize_anchored_pts = True
        

        params_to_train_kines = []
        # params_to_train_kines += list(self.mano_robot_glb_rotation.parameters())

        
        # if self.optimize_anchored_pts:
        #     params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        #     params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        
        if self.optimize_robot:
            params_to_train_kines += list(self.robot_states.parameters())
            

        
        
        self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate) #### kinematics optimizer ###
        
        
        if self.optimize_rules:
            params_to_train_kines = []
            params_to_train_kines += list(self.other_bending_network.parameters())
            self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate) 
            
        
        
        self.expanded_set_delta_motion_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone()
        
      
        ''' prepare for keeping the original global rotations, trans, and states '''
        # ori_mano_robot_glb_rot = self.mano_robot_glb_rotation.weight.data.clone()
        # ori_mano_robot_glb_trans = self.mano_robot_glb_trans.weight.data.clone()
        # ori_mano_robot_delta_states = self.mano_robot_delta_states.weight.data.clone()
        
        
        
        
        self.iter_step = 0
        
        self.ts_to_dyn_mano_pts_th = {}
        self.timestep_to_anchored_mano_pts = {}


        for i_iter in tqdm(range(200)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep #
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # # # #
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            
            # tot_penetrating_depth_penalty = []
            # tot_ragged_dist = []
            # tot_delta_offset_reg_motion = []
            # tot_dist_mano_visual_ori_to_cur = []
            # tot_reg_loss = []
            # tot_diff_cur_states_to_ref_states = []
            # tot_diff_tangential_forces = []
            # penetration_forces = None ###
            # sampled_visual_pts_joint_idxes = None
            
            # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            # self.timestep_to_anchored_mano_pts = {}
            
            
            self.ts_to_contact_passive_normals = {}
            self.ts_to_passive_normals = {}
            self.ts_to_passive_pts = {}
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            self.ts_to_dyn_mano_pts = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            # tot_grad_qs = []
            
            robo_intermediates_states = []
            
            tot_penetration_depth = []
            
            robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            
            tot_interpenetration_nns = []
            
            tot_diff_cur_visual_pts_offsets_with_ori = []
            
            tot_summ_grad_mano_expanded_actuator_delta_offset_weight = []
            tot_summ_grad_mano_expanded_actuator_pointact_forces_weight = []
            
            penetration_forces = None ###
            sampled_visual_pts_joint_idxes = None
            
            
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            # for cur_ts in range(self.nn_ts):
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                # tot_redmax_actions = []
                # actions = {}

                self.free_def_bending_weight = 0.0

                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone()
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations # # glb rot # trans #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)


                # # # cur_ts_delta_rot, cur_ts_redmax_robot_trans # # #
                # cur_glb_rot = torch.matmul(cur_ts_delta_rot, cur_glb_rot)
                # cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans # redmax robot transj## 

                # link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # self.mano_agent.set_init_states_target_value(link_cur_states)
                # cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True)  # init state visual pts #
                
                # cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                ### motivate via actions ####
                link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes)
                ### motivate via actions ####
                
                
                cur_visual_pts, visual_pts_joint_idxes = self.mano_agent.get_init_state_visual_pts(expanded_pts=True, ret_joint_idxes=True)  # init state visual pts #
                
                ## visual pts sampled ##
                
                ## visual pts sampled ##
                cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                
                
                if self.drive_pointset == "actions":
                    ''' Act-React-driven point motions '''
                    ###### get the point offset via actuation forces and reaction forces ######
                    ### actuation forces at this timestep ###
                    cur_visual_pts_forces = self.mano_expanded_actuator_pointact_forces(cur_visual_pts_idxes) * 1e7 ## get vsual pts forces ### 
                    
                    # if cur_ts > 0 and (cur_ts - 1 in self.timestep_to_actuator_points_passive_forces):
                    #     cur_visual_pts_passive_forces = self.timestep_to_actuator_points_passive_forces[cur_ts - 1] ## nn_visual_pts x 3 ##
                    #     cur_visual_pts_forces = cur_visual_pts_forces + cur_visual_pts_passive_forces ## two forces ### 
                        
                    cur_visual_pts_forces = cur_visual_pts_forces * pointset_expansion_alpha
                        
                    ## --- linear damping here ? ## 
                    cur_visual_pts_accs = cur_visual_pts_forces / self.mass_point_mass ### get the mass pont accs ## 
                    if cur_ts == 0:
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    else: # visual pts acc -> visual pts vels #
                        # prev_visual_pts_vels = self.timestep_to_actuator_points_vels[cur_ts - 1] ### nn_pts x 3 ## 
                        # cur_visual_pts_accs = cur_visual_pts_accs - cur_vel_damping_coef * prev_visual_pts_vels ### nn_pts x 3 ##
                        # cur_visual_pts_vels = prev_visual_pts_vels + cur_visual_pts_accs * time_cons ## nn_pts x 3 ## 
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    self.timestep_to_actuator_points_vels[cur_ts] = cur_visual_pts_vels.detach().clone()
                    cur_visual_pts_offsets = cur_visual_pts_vels * time_cons
                    # 
                    # if cur_ts > 0:
                    #     prev_visual_pts_offset = self.timestep_to_actuator_points_offsets[cur_ts - 1]
                    #     cur_visual_pts_offsets = prev_visual_pts_offset + cur_visual_pts_offsets
                        
                    
                    # train_pointset_acts_via_deltas, diff_cur_visual_pts_offsets_with_ori
                    # cur_visual_pts_offsets_from_delta = mano_expanded_actuator_delta_offset_ori[cur_ts]
                    # cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset.weight.data[cur_ts].detach()
                    cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes).detach()
                    ## 
                    diff_cur_visual_pts_offsets_with_ori = torch.sum((cur_visual_pts_offsets - cur_visual_pts_offsets_from_delta) ** 2, dim=-1).mean() ## mean of the avg offset differences ##
                    tot_diff_cur_visual_pts_offsets_with_ori.append(diff_cur_visual_pts_offsets_with_ori.item())
                        
                    
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ###### get the point offset via actuation forces and reaction forces ######
                    ''' Act-React-driven point motions '''
                elif self.drive_pointset == "states":
                    ''' Offset-driven point motions '''
                    ## points should be able to manipulate the object accordingly ###
                    ### we should avoid the penetrations between points and the object ###
                    ### we should restrict relative point displacement / the point offsets at each timestep to relatively small values ###
                    ## -> so we have three losses for the delta offset optimization ##
                    cur_visual_pts_offsets = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes)
                    # cur_visual_pts_offsets = cur_visual_pts_offsets * 10
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ''' Offset-driven point motions '''
                    
                else:
                    raise ValueError(f"Unknown drive_pointset: {self.drive_pointset}")
                
                
                
                
                # cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                # ## get the friction forces ##
                # cur_visual_pts_friction_forces = self.mano_expanded_actuator_friction_forces(cur_visual_pts_idxes)

                ### transform the visual pts ### ## fricton forces ##
                # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                # cur_visual_pts = cur_visual_pts * 2. - 1.  # cur visual pts # 
                # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # # mult_const #
                
                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent # mult cnst after cent #
                cur_dyn_mano_pts = cur_dyn_mano_pts * self.mano_mult_const_after_cent
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                ## 
                ### transform by the glboal transformation and the translation ###
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                cur_dyn_mano_pts = torch.matmul(cur_rot, cur_dyn_mano_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offsets
                
                # diff_redmax_visual_pts_with_ori_visual_pts = torch.sum(
                #     (cur_visual_pts[sampled_verts_idxes] - self.timestep_to_active_mesh_opt_ours_sim[cur_ts].detach()) ** 2, dim=-1
                # )
                # diff_redmax_visual_pts_with_ori_visual_pts = diff_redmax_visual_pts_with_ori_visual_pts.mean()
                
                # train the friction net? how to train the friction net? #
                # if self.use_mano_hand_for_test:
                #     self.timestep_to_active_mesh[cur_ts] = self.rhand_verts[cur_ts] # .detach()
                # else:
                
                
                # timestep_to_anchored_mano_pts, timestep_to_raw_active_meshes # 
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                # self.ts_to_dyn_mano_pts[cur_ts] = cur_dyn_mano_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts[cur_ts] = cur_dyn_mano_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts_th[cur_ts] = cur_dyn_mano_pts
                
                
                # ragged_dist = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # dist_transformed_expanded_visual_pts_to_ori_visual_pts = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # diff_cur_states_to_ref_states = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.robot_agent.set_init_states_target_value(robo_links_states)
                cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                

                if not self.use_scaled_urdf:
                    ### transform the visual pts ###
                    cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                    cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                    
                
                cur_rot = cur_robo_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                ### transform by the glboal transformation and the translation ###
                cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.clone() # robo visual pts #
                
                # self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                # if self.hand_type == 'redmax_hand':
                #     maxx_sampled_idxes = torch.max(sampled_verts_idxes)
                #     minn_sampled_idxes = torch.min(sampled_verts_idxes)
                #     # print(f"cur_robo_visual_pts: {cur_robo_visual_pts.size()}, maxx_sampled_idxes: {maxx_sampled_idxes}, minn_sampled_idxes: {minn_sampled_idxes}")
                #     cur_robo_visual_pts = cur_robo_visual_pts[sampled_verts_idxes] # 
                
                
                
                self.free_def_bending_weight = 0.0
                # self.free_def_bending_weight = 0.5
                
                if i_iter == 0 and cur_ts == 0: ## for the tiantianquan sequence ##
                    dist_robot_pts_to_mano_pts = torch.sum(
                        (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                    )
                    minn_dist_robot_pts_to_mano_pts, correspondence_pts_idxes = torch.min(dist_robot_pts_to_mano_pts, dim=-1)
                    minn_dist_robot_pts_to_mano_pts = torch.sqrt(minn_dist_robot_pts_to_mano_pts)
                    # dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.01
                    dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.005
                    
                    corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes] # in correspondence pts idxes ##
                    
                    dist_corr_correspondence_pts_to_mano_visual_pts = torch.sum(
                        (corr_correspondence_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                    )
                    dist_corr_correspondence_pts_to_mano_visual_pts = torch.sqrt(dist_corr_correspondence_pts_to_mano_visual_pts)
                    minn_dist_to_corr_pts, _ = torch.min(dist_corr_correspondence_pts_to_mano_visual_pts, dim=0)
                    anchored_mano_visual_pts = minn_dist_to_corr_pts < 0.005
                    
                corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                # corr_robo = cur_visual_pts[sampled_verts_idxes]
                cd_robo_pts_to_corr_mano_pts = torch.sum( # distance from robot pts to the anchored mano pts
                    (cur_robo_visual_pts.unsqueeze(1) - cur_visual_pts[anchored_mano_visual_pts].unsqueeze(0).detach()) ** 2, dim=-1
                )
                
                # self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts[anchored_mano_visual_pts] # .detach().cpu().numpy()
                self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts # .detach().cpu().numpy()
                
                
                cd_robo_to_mano, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=-1)
                cd_mano_to_robo, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=0)
                # diff_robo_to_corr_mano_pts = cd_mano_to_robo.mean()
                diff_robo_to_corr_mano_pts = cd_robo_to_mano.mean()
                
                mano_fingers = self.rhand_verts[cur_ts][self.mano_fingers]
                
                
                
                ##### finger cd loss -> to the anchored expanded actioning points #####
                ## fingeer tracking loss -> to each finger ##
                # loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef + pure_finger_tracking_loss * 0.5 # + diff_robo_to_corr_mano_pts_finger_tracking * self.finger_tracking_loss_coef
                loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef # + pure_finger_tracking_loss * 0.5 
                
                
                ## TODO: add the glboal retargeting using fingers before conducting this approach 
                
                
                # def evaluate_tracking_loss():
                #     self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
                    
                #     ## 
                #     # init states  # 
                #     # cur_ts % mano_nn_substeps == 0:
                #     if (cur_ts + 1) % mano_nn_substeps == 0:
                #         cur_passive_big_ts = cur_ts // mano_nn_substeps
                #         in_func_tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                #         # tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                #     else:
                #         in_func_tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                #     return in_func_tracking_loss

                    
                if contact_pairs_set is None:
                    self.contact_pairs_set = None
                else:
                    ## 
                    self.contact_pairs_set = contact_pairs_set.copy()
                
                # ### if traiing the jrbpt h
                # print(self.timestep_to_active_mesh[cur_ts].size(), cur_visual_pts_friction_forces.size()) # friction forces #
                ### optimize the robot tracing loss ###
                
                
                # if self.o
                
                if self.optimize_anchored_pts:
                    # anchored_cur_visual_pts_friction_forces = cur_visual_pts_friction_forces[anchored_mano_visual_pts]
                    contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_anchored_mano_pts, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set)
                else:
                    contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set, pts_frictional_forces=None)
                    
                    
                ### train with force to active ##
                if self.train_with_forces_to_active and (not self.use_mano_inputs):
                    # penetration_forces #
                    if torch.sum(self.other_bending_network.penetrating_indicator.float()) > 0.5:
                        net_penetrating_forces = self.other_bending_network.penetrating_forces
                        net_penetrating_points = self.other_bending_network.penetrating_points

                        
                        # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
                        self.timestep_to_penetration_points[cur_ts] = net_penetrating_points.detach().cpu().numpy()
                        self.timestep_to_penetration_points_forces[cur_ts] = net_penetrating_forces.detach().cpu().numpy()
                        
                        
                        ### transform the visual pts ###
                        # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                        # cur_visual_pts = cur_visual_pts * 2. - 1.
                        # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # mult_const #
                        
                        # sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[finger_sampled_idxes][self.other_bending_network.penetrating_indicator]
                        
                        sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[self.other_bending_network.penetrating_indicator]
                        
                        ## from net penetration forces to to the 
                         
                        ### get the passvie force for each point ## ## 
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone() ## for
                        
                        net_penetrating_forces = torch.matmul(
                            cur_rot.transpose(1, 0), net_penetrating_forces.transpose(1, 0)
                        ).transpose(1, 0)
                        # net_penetrating_forces = net_penetrating_forces / self.mult_const_after_cent
                        # net_penetrating_forces = net_penetrating_forces / 2
                        # net_penetrating_forces = net_penetrating_forces * self.extent_robo_pts
                        
                        net_penetrating_forces = (1.0 - pointset_expansion_alpha) * net_penetrating_forces
                        
                        net_penetrating_points = torch.matmul(
                            cur_rot.transpose(1, 0), (net_penetrating_points - cur_trans.unsqueeze(0)).transpose(1, 0)
                        ).transpose(1, 0)
                        # net_penetrating_points = net_penetrating_points / self.mult_const_after_cent
                        # net_penetrating_points = (net_penetrating_points + 1.) / 2. # penetrating points #
                        # net_penetrating_points = (net_penetrating_points * self.extent_robo_pts) + self.minn_robo_pts
                        
                        penetration_forces = net_penetrating_forces
                        # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        # 
                    else:
                        penetration_forces = None
                        sampled_visual_pts_joint_idxes = None
                        # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                        ''' the bending network still have this property and we can get force values here for the expanded visual points '''
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone()  
                        
                
                
                
                if (cur_ts + 1) % mano_nn_substeps == 0:
                    cur_passive_big_ts = cur_ts // mano_nn_substeps
                    ### tracking loss between the predicted transformation and te tracking ###
                    tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                    tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                else:
                    tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()





                # cur_interpenetration_nns = self.other_bending_network.smaller_than_zero_level_set_indicator.float().sum()
                
                diff_actions = torch.sum(
                    (self.expanded_set_delta_motion_ori - self.mano_expanded_actuator_delta_offset.weight) ** 2, dim=-1
                )
                
                reg_act_force_loss = torch.sum(
                    (mano_expanded_actuator_pointact_forces_ori - self.mano_expanded_actuator_pointact_forces.weight.data) ** 2, dim=-1
                )
                reg_act_force_loss = reg_act_force_loss.mean()
                
                
                diff_actions = diff_actions.mean()
                diff_actions = diff_actions + reg_act_force_loss
                
                
                ### the tracking loss ###
                # kinematics_proj_loss = loss_finger_tracking + tracking_loss + diff_actions # + tracking_loss + penetraton_penalty
                
                kinematics_proj_loss = loss_finger_tracking + tracking_loss
                
                loss = kinematics_proj_loss
                
                
                mano_tracking_loss.append(loss_finger_tracking.detach().cpu().item())
                
                
                self.kines_optimizer.zero_grad()
                
                kinematics_proj_loss.backward(retain_graph=True)
                
                self.kines_optimizer.step()
                
                # if self.use_LBFGS:
                #     self.kines_optimizer.step(evaluate_tracking_loss) # 
                # else:
                #     self.kines_optimizer.step()

                # 
                # tracking_loss.backward(retain_graph=True)
                # if self.use_LBFGS:
                #     self.other_bending_network.reset_timestep_to_quantities(cur_ts)
                
                
                # robot_states_actions_diff_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # diff_actions
                robot_states_actions_diff_loss = diff_actions
                robo_actions_diff_loss.append(robot_states_actions_diff_loss)

                
                tot_losses.append(loss.detach().item()) # total losses # # total losses # 
                # tot_penalty_dot_forces_normals.append(cur_penalty_dot_forces_normals.detach().item())
                # tot_penalty_friction_constraint.append(cur_penalty_friction_constraint.detach().item())
                
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)


                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                
                self.update_learning_rate()
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            if len(tot_tracking_loss) > 0:
                tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            else:
                tot_tracking_loss = 0.0
            if len(tot_penetration_depth) > 0:
                tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            else:
                tot_penetration_depth = 0.0
            robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            if len(mano_tracking_loss) > 0:
                mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))
            else:
                mano_tracking_loss = 0.0
                
                
            
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')
            else:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)

            
            # self.validate_mesh_robo_a()
            if i_iter % self.val_mesh_freq == 0:
                self.validate_mesh_robo_g()
                self.validate_mesh_robo()
                ### test for contact infos ###
                # self.validate_contact_info_robo()
                
            
            torch.cuda.empty_cache()
    
    
    
    ''' GRAB clips --- expanded point set and expanded points for retargeting ''' 
    def train_point_set_retargeting_pts(self, ):
        ## ## GRAB clips ## ##
        # states -> the robot actions --- in this sim ##
        # chagne # # mano notjmano but the mano ---> optimize the mano delta states? # 
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # update learning rrate # 
        # robot actions ##
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        # load # # load the robot hand # # load
        ''' Load the robot hand '''
        model_path = self.conf['model.sim_model_path'] # 
        # robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        self.hand_type = "redmax_hand" if "redmax" in model_path else "shadow_hand"
        # self.hand_type = "redmax_hand"
        if model_path.endswith(".xml"):
            # self.hand_type = "redmax_hand"
            robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        else:
            # self.hand_type = "shadow_hand"
            robot_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path, args=None)
        self.robot_agent = robot_agent
        
        robo_init_verts = self.robot_agent.robot_pts
        if self.hand_type == "redmax_hand":
            redmax_sampled_verts_idxes_fn = "redmax_robo_sampled_verts_idxes_new.npy"
            redmax_sampled_verts_idxes_fn = os.path.join("assets", redmax_sampled_verts_idxes_fn)
            if os.path.exists(redmax_sampled_verts_idxes_fn):
                sampled_verts_idxes = np.load(redmax_sampled_verts_idxes_fn)
                sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
            else:
                n_sampling = 1000
                pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
                sampled_verts_idxes = pts_fps_idx
                np.save(redmax_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
            self.sampled_verts_idxes = sampled_verts_idxes
        
        robo_expanded_visual_pts = self.robot_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        self.robo_expanded_visual_pts_nn = robo_expanded_visual_pts.size(0)
        
        self.robo_hand_faces = self.robot_agent.robot_faces
        
        
        
        ## sampled verts idxes ## 
        # self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- # # --- robot hand mani asset? ##  ## robot hand mani asse ##
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m
        
        
        ''' Load the mano hand model '''
        model_path_mano = self.conf['model.mano_sim_model_path']
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.robo_hand_faces = self.mano_agent.robot_faces
        
        
        
        # if self.use_mano_hand_for_test: ## use
        #     self.robo_hand_faces = self.hand_faces
        ## 
        
        ## start expanding the current visual pts ##
        print(f"Start expanding current visual pts...")
        expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        
        self.expanded_visual_pts_nn = expanded_visual_pts.size(0)
        ## expanded_visual_pts of the expanded visual pts #
        expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        print(f"Saving expanded visual pts with shape {expanded_visual_pts.size()} to {expanded_visual_pts_sv_fn}")
        np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy)
        
        
        nn_substeps = 10
        # 
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        # self.hand_faces # 

        
        ''' Expnad the current visual points ''' 
        # expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.expanded_visual_pts_nn = expanded_visual_pts.size(0)
        # expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        # expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        # np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy)
        # # ''' Expnad the current visual points '''  #  # differentiate through the simulator? # # 
        
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        # params_to_train += list(self.robot_actions.parameters())
        
        self.mano_robot_delta_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        
        self.mano_robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # params_to_train += list(self.robot_init_states.parameters())
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        # params_to_train += list(self.robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        # params_to_train += list(self.robot_glb_trans.parameters())   
        
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        
        
        
        self.mano_expanded_actuator_delta_offset = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_delta_offset.weight)
        # params_to_train += list(self.mano_expanded_actuator_delta_offset.parameters())
        
        
        # mano_expanded_actuator_friction_forces, mano_expanded_actuator_delta_offset # 
        self.mano_expanded_actuator_friction_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_friction_forces.weight)
        
        
        ##### expanded actuators pointact jforces ### 
        self.mano_expanded_actuator_pointact_forces = nn.Embedding(
            num_embeddings=self.expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_forces.weight)
        
        self.mano_expanded_actuator_pointact_damping_coefs = nn.Embedding(
            num_embeddings=10, embedding_dim=1
        ).cuda()
        torch.nn.init.zeros_(self.mano_expanded_actuator_pointact_damping_coefs.weight)
        


        ## load mano states and actions ##
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            if 'mano_robot_init_states' in optimized_init_actions_ckpt:
                self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
            if 'mano_robot_glb_rotation' in optimized_init_actions_ckpt:
                self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])

            if 'mano_robot_states' in optimized_init_actions_ckpt:
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            
            if 'mano_robot_actions' in optimized_init_actions_ckpt:
                self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['mano_robot_actions'])    
            
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            if 'expanded_actuator_friction_forces' in optimized_init_actions_ckpt:
                try:
                    self.mano_expanded_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_friction_forces'])
                except:
                    pass
            #### actuator point forces and actuator point offsets ####
            if 'mano_expanded_actuator_delta_offset' in optimized_init_actions_ckpt:
                print(f"loading mano_expanded_actuator_delta_offset...")
                self.mano_expanded_actuator_delta_offset.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_delta_offset'])
            if 'mano_expanded_actuator_pointact_forces' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_forces.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_forces'])
            if 'mano_expanded_actuator_pointact_damping_coefs' in optimized_init_actions_ckpt:
                self.mano_expanded_actuator_pointact_damping_coefs.load_state_dict(optimized_init_actions_ckpt['mano_expanded_actuator_pointact_damping_coefs'])




        ''' parameters for the real robot hand ''' 
        # # robot actions # # real robot hand ##
        self.robot_actions = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_actions.weight)
        params_to_train += list(self.robot_actions.parameters())
        
        # self.robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.robot_delta_states.weight)
        # params_to_train += list(self.robot_delta_states.parameters())
        self.robot_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_states.weight)
        params_to_train += list(self.robot_states.parameters())
        
        self.robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.robot_init_states.weight)
        params_to_train += list(self.robot_init_states.parameters())
        
        ## robot glb rotations ##
        self.robot_glb_rotation = nn.Embedding( ## robot hand rotation 
            num_embeddings=num_steps, embedding_dim=4
        ).cuda()
        self.robot_glb_rotation.weight.data[:, 0] = 1.
        self.robot_glb_rotation.weight.data[:, 1:] = 0.
        
        self.robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_glb_trans.weight)
        
        
        # ### local minimum -> ## robot 
        self.robot_actuator_friction_forces = nn.Embedding( # frictional forces ##
            num_embeddings=365428 * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_actuator_friction_forces.weight)
        
        
        self.expanded_actuator_delta_offset = nn.Embedding(
            num_embeddings=self.robo_expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.expanded_actuator_delta_offset.weight)
        # params_to_train += list(self.mano_expanded_actuator_delta_offset.parameters())
        
        
        
        ##### expanded actuators pointact jforces ### 
        self.expanded_actuator_pointact_forces = nn.Embedding(
            num_embeddings=self.robo_expanded_visual_pts_nn * 60, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.expanded_actuator_pointact_forces.weight)
        
        self.expanded_actuator_pointact_damping_coefs = nn.Embedding(
            num_embeddings=10, embedding_dim=1
        ).cuda()
        torch.nn.init.zeros_(self.expanded_actuator_pointact_damping_coefs.weight)
        


        
        
        ### load optimized init transformations for robot actions ###
        if len(self.load_optimized_init_transformations) > 0:
            print(f"[Robot] Loading optimized init transformations from {self.load_optimized_init_transformations}")
            cur_optimized_init_actions_fn = self.load_optimized_init_transformations
            # cur_optimized_init_actions = # optimized init states # ## robot init states ##
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            try:
                self.robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
            except:
                pass
            self.robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
            if 'robot_delta_states' in optimized_init_actions_ckpt:
                try:
                    self.robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
                except:
                    pass
            if 'robot_states' in optimized_init_actions_ckpt:
                self.robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
            # if 'robot_delta_states'  ## robot delta states ##
            # self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
            
            self.robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
            
            if 'expanded_actuator_delta_offset' in optimized_init_actions_ckpt:
                print(f"[Robot] loading actuator delta offsets from {self.load_optimized_init_transformations}")
                self.expanded_actuator_delta_offset.load_state_dict(optimized_init_actions_ckpt['expanded_actuator_delta_offset'])
            
            if 'expanded_actuator_pointact_forces' in optimized_init_actions_ckpt:
                print(f"[Robot] loading actuator pointact forces from {self.load_optimized_init_transformations}")
                self.expanded_actuator_pointact_forces.load_state_dict(optimized_init_actions_ckpt['expanded_actuator_pointact_forces'])
            
            if 'expanded_actuator_pointact_damping_coefs' in optimized_init_actions_ckpt:
                print(f"[Robot] loading actuator pointact damping coefs from {self.load_optimized_init_transformations}")
                self.expanded_actuator_pointact_damping_coefs.load_state_dict(optimized_init_actions_ckpt['expanded_actuator_pointact_damping_coefs'])

        
        
        if self.hand_type == "redmax_hand":
            self.maxx_robo_pts = 25.
            self.minn_robo_pts = -15.
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.5437551664260203
        else:
            self.minn_robo_pts = -0.1
            self.maxx_robo_pts = 0.2
            self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
            self.mult_const_after_cent = 0.437551664260203
        ## for grab ##
        self.mult_const_after_cent = self.mult_const_after_cent / 3. * 0.9507
        
        
        # self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        # # self.robot_fingers = [3591, 4768, 6358, 10228, 6629, 10566, 5631, 9673]
        # self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877]
        # # self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        # if self.hand_type == "redmax_hand":
        #     self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
        #     self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
        
        self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, ]
        self.robot_fingers = [6496, 10128, 53, 1623, 3209, 4495, 9523, 8877, ]
        
        if self.hand_type == "redmax_hand":
            self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121]
            self.robot_fingers = [521, 624, 846, 973, 606, 459, 383, 265]
            self.robot_fingers = [14670, 321530, 36939, 125930, 200397, 257721, 333438, 338358]
        
        # params_to_train = []
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']
        
        
        self.nn_ts = self.nn_timesteps - 1

        ''' parameters for the real robot hand ''' 
        

        
        self.timestep_to_active_mesh = {}
        # ref_expanded_visual_pts, minn_idx_expanded_visual_pts_to_link_pts #
        # minn_idx_expanded_visual_pts_to_link_pts #
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        
        self.timestep_to_active_mesh_w_delta_states = {}
        
        
        self.mass_point_mass = 1.0
        
        self.timestep_to_actuator_points_vels = {}
        self.timestep_to_actuator_points_passive_forces = {}
        
        self.timestep_to_robo_actuator_points_vels = {}
        self.timestep_to_robo_actuator_points_passive_forces = {}
        
        
        self.timestep_to_actuator_points_offsets = {}
        self.timestep_to_robo_actuator_points_offsets = {}
        
        time_cons = 0.0005
        pointset_expansion_alpha = 0.1
        
        
        # states -> get states -> only update the acitons #
        with torch.no_grad(): # init them to zero 
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                ''' Get rotations, translations, and actions of the current robot '''
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7) # mano glb rot
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                ## mano robot states ## mano robot states ##
                link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_init_states_target_value(link_cur_states)
                
                ## set init visual pts ##
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True) 

                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                
                cur_visual_pts_offset = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes) ## get the idxes ### 
                
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                ## 
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## transformed pts ## 
                
                
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offset
                
                # if not self.use_mano_inputs:
                # if self.use_mano_hand_for_test:
                #     self.timestep_to_active_mesh[cur_ts] = self.rhand_verts[cur_ts].detach()
                # else:
                #     self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_w_delta_states[cur_ts] = cur_visual_pts.detach()
                self.timestep_to_active_mesh_opt_ours_sim[cur_ts] = cur_visual_pts.detach()
                
                
                cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                
                robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.robot_agent.set_init_states_target_value(robo_links_states)
                cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                

                if not self.use_scaled_urdf:
                    ### transform the visual pts ###
                    cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                    cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                
                
                # cur_rot = cur_robo_glb_rot
                # cur_trans = cur_glb_trans
                
                # timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                # timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                ### transform by the glboal transformation and the translation ###
                cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.detach()
                
        self.iter_step = 0
        
        mano_expanded_actuator_delta_offset_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone().detach()
        mano_expanded_actuator_pointact_forces_ori = self.mano_expanded_actuator_pointact_forces.weight.data.clone().detach()
        
        # robo_expanded_set_delta_motion_ori, robo_expanded_actuator_pointact_forces_ori
        robo_expanded_set_delta_motion_ori = self.expanded_actuator_delta_offset.weight.data.clone().detach()
        robo_expanded_actuator_pointact_forces_ori = self.expanded_actuator_pointact_forces.weight.data.clone().detach()
        
        ''' Set redmax robot actions '''
        
        
        # self.optimize_robot = False
        
        # self.optimize_robot = True
        
        # self.optimize_anchored_pts = False
        # self.optimize_anchored_pts = True
        

        params_to_train_kines = []
        # params_to_train_kines += list(self.mano_robot_glb_rotation.parameters())

        
        # if self.optimize_anchored_pts:
        #     params_to_train_kines += list(self.mano_expanded_actuator_delta_offset.parameters())
        #     params_to_train_kines += list(self.mano_expanded_actuator_friction_forces.parameters())
        
        # expanded_actuator_pointact_forces, expanded_actuator_delta_offset
        # if self.optimize_robot:
        
        if self.optimize_pointset_motion_only:
            # params_to_train_kines += list(self.robot_states.parameters())
            ### 
            params_to_train_kines += list(self.expanded_actuator_pointact_forces.parameters())
            params_to_train_kines += list(self.expanded_actuator_delta_offset.parameters())
        else:
            params_to_train_kines += list(self.robot_states.parameters())
            params_to_train_kines += list(self.expanded_actuator_pointact_forces.parameters())
            params_to_train_kines += list(self.expanded_actuator_delta_offset.parameters())
            
        
        
        
        self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate) #### kinematics optimizer ###
        
        
        if self.optimize_rules:
            params_to_train_kines = []
            params_to_train_kines += list(self.other_bending_network.parameters())
            self.kines_optimizer = torch.optim.Adam(params_to_train_kines, lr=self.learning_rate) 
            
        
        
        self.expanded_set_delta_motion_ori = self.mano_expanded_actuator_delta_offset.weight.data.clone()
        
      
        ''' prepare for keeping the original global rotations, trans, and states '''
        # ori_mano_robot_glb_rot = self.mano_robot_glb_rotation.weight.data.clone()
        # ori_mano_robot_glb_trans = self.mano_robot_glb_trans.weight.data.clone()
        # ori_mano_robot_delta_states = self.mano_robot_delta_states.weight.data.clone()
        
        
        
        
        self.iter_step = 0
        
        self.ts_to_dyn_mano_pts_th = {}
        self.timestep_to_anchored_mano_pts = {}
        self.timestep_to_expanded_robo_visual_pts = {}
        
        self.sampled_robo_expanded_pts_idxes = None


        # for i_iter in tqdm(range(100000)):
        for i_iter in tqdm(range(200)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep #
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # # # #
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            
            # tot_penetrating_depth_penalty = []
            # tot_ragged_dist = []
            # tot_delta_offset_reg_motion = []
            # tot_dist_mano_visual_ori_to_cur = []
            # tot_reg_loss = []
            # tot_diff_cur_states_to_ref_states = []
            # tot_diff_tangential_forces = []
            # penetration_forces = None ###
            # sampled_visual_pts_joint_idxes = None
            
            # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            # self.timestep_to_anchored_mano_pts = {}
            
            
            self.ts_to_contact_passive_normals = {}
            self.ts_to_passive_normals = {}
            self.ts_to_passive_pts = {}
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            self.ts_to_dyn_mano_pts = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            # tot_grad_qs = []
            
            robo_intermediates_states = []
            
            tot_penetration_depth = []
            
            robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            
            tot_interpenetration_nns = []
            
            tot_diff_cur_visual_pts_offsets_with_ori = []
            tot_diff_robo_cur_visual_pts_offsets_with_ori = []
            
            tot_summ_grad_mano_expanded_actuator_delta_offset_weight = []
            tot_summ_grad_mano_expanded_actuator_pointact_forces_weight = []
            
            penetration_forces = None ###
            sampled_visual_pts_joint_idxes = None
            
            
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            # for cur_ts in range(self.nn_ts):
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                # tot_redmax_actions = []
                # actions = {}

                self.free_def_bending_weight = 0.0


                ''' Get dynamic mano pts, expanded pts, etc '''
                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone()
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations # # glb rot # trans #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)


                # # # cur_ts_delta_rot, cur_ts_redmax_robot_trans # # #
                # cur_glb_rot = torch.matmul(cur_ts_delta_rot, cur_glb_rot)
                # cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans # redmax robot transj## 

                # link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # self.mano_agent.set_init_states_target_value(link_cur_states)
                # cur_visual_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=True)  # init state visual pts #
                
                # cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                ### motivate via actions ####
                link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes)
                ### motivate via actions ####
                
                
                cur_visual_pts, visual_pts_joint_idxes = self.mano_agent.get_init_state_visual_pts(expanded_pts=True, ret_joint_idxes=True)  # init state visual pts #
                
                ## visual pts sampled ##
                
                ## visual pts sampled ##
                cur_dyn_mano_pts = self.mano_agent.get_init_state_visual_pts(expanded_pts=False)  # init s
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.expanded_visual_pts_nn, end=(cur_ts + 1) * self.expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                
                
                
                if self.drive_pointset == "actions":
                    ''' Act-React-driven point motions '''
                    ###### get the point offset via actuation forces and reaction forces ######
                    ### actuation forces at this timestep ###
                    cur_visual_pts_forces = self.mano_expanded_actuator_pointact_forces(cur_visual_pts_idxes) * 1e7 ## get vsual pts forces ### 
                    
                    # if cur_ts > 0 and (cur_ts - 1 in self.timestep_to_actuator_points_passive_forces):
                    #     cur_visual_pts_passive_forces = self.timestep_to_actuator_points_passive_forces[cur_ts - 1] ## nn_visual_pts x 3 ##
                    #     cur_visual_pts_forces = cur_visual_pts_forces + cur_visual_pts_passive_forces ## two forces ### 
                        
                    cur_visual_pts_forces = cur_visual_pts_forces * pointset_expansion_alpha
                        
                    ## --- linear damping here ? ## 
                    cur_visual_pts_accs = cur_visual_pts_forces / self.mass_point_mass ### get the mass pont accs ## 
                    if cur_ts == 0:
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    else: # visual pts acc -> visual pts vels #
                        # prev_visual_pts_vels = self.timestep_to_actuator_points_vels[cur_ts - 1] ### nn_pts x 3 ## 
                        # cur_visual_pts_accs = cur_visual_pts_accs - cur_vel_damping_coef * prev_visual_pts_vels ### nn_pts x 3 ##
                        # cur_visual_pts_vels = prev_visual_pts_vels + cur_visual_pts_accs * time_cons ## nn_pts x 3 ## 
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    self.timestep_to_actuator_points_vels[cur_ts] = cur_visual_pts_vels.detach().clone()
                    cur_visual_pts_offsets = cur_visual_pts_vels * time_cons
                    # 
                    # if cur_ts > 0:
                    #     prev_visual_pts_offset = self.timestep_to_actuator_points_offsets[cur_ts - 1]
                    #     cur_visual_pts_offsets = prev_visual_pts_offset + cur_visual_pts_offsets
                        
                    
                    # train_pointset_acts_via_deltas, diff_cur_visual_pts_offsets_with_ori
                    # cur_visual_pts_offsets_from_delta = mano_expanded_actuator_delta_offset_ori[cur_ts]
                    # cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset.weight.data[cur_ts].detach()
                    cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes).detach()
                    ## 
                    diff_cur_visual_pts_offsets_with_ori = torch.sum((cur_visual_pts_offsets - cur_visual_pts_offsets_from_delta) ** 2, dim=-1).mean() ## mean of the avg offset differences ##
                    tot_diff_cur_visual_pts_offsets_with_ori.append(diff_cur_visual_pts_offsets_with_ori.item())
                        
                    
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ###### get the point offset via actuation forces and reaction forces ######
                    ''' Act-React-driven point motions '''
                elif self.drive_pointset == "states":
                    ''' Offset-driven point motions '''
                    ## points should be able to manipulate the object accordingly ###
                    ### we should avoid the penetrations between points and the object ###
                    ### we should restrict relative point displacement / the point offsets at each timestep to relatively small values ###
                    ## -> so we have three losses for the delta offset optimization ##
                    cur_visual_pts_offsets = self.mano_expanded_actuator_delta_offset(cur_visual_pts_idxes)
                    # cur_visual_pts_offsets = cur_visual_pts_offsets * 10
                    self.timestep_to_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ''' Offset-driven point motions '''
                    
                else:
                    raise ValueError(f"Unknown drive_pointset: {self.drive_pointset}")
                
                
                
                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent # mult cnst after cent #
                cur_dyn_mano_pts = cur_dyn_mano_pts * self.mano_mult_const_after_cent
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                cur_dyn_mano_pts = torch.matmul(cur_rot, cur_dyn_mano_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                cur_visual_pts = cur_visual_pts + cur_visual_pts_offsets
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts[cur_ts] = cur_dyn_mano_pts.detach().cpu().numpy()
                self.ts_to_dyn_mano_pts_th[cur_ts] = cur_dyn_mano_pts
                ''' Get dynamic mano pts, expanded pts, etc '''
                
                
                
                cur_robo_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                cur_robo_glb_rot = cur_robo_glb_rot / torch.clamp(torch.norm(cur_robo_glb_rot, dim=-1, p=2), min=1e-7)
                cur_robo_glb_rot = dyn_model_act.quaternion_to_matrix(cur_robo_glb_rot) # mano glboal rotations #
                cur_robo_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                ### drive by states, not the acts ### 
                robo_links_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                self.robot_agent.set_init_states_target_value(robo_links_states)
                cur_robo_visual_pts = self.robot_agent.get_init_state_visual_pts()
                cur_robo_expanded_visual_pts, robo_visual_pts_joint_idxes = self.robot_agent.get_init_state_visual_pts(expanded_pts=True, ret_joint_idxes=True)  # init state visual pts #
                

                if not self.use_scaled_urdf:
                    cur_robo_visual_pts = (cur_robo_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_robo_visual_pts = cur_robo_visual_pts * 2. -1.
                    cur_robo_visual_pts = cur_robo_visual_pts * self.mult_const_after_cent # mult_const #
                    
                    cur_robo_expanded_visual_pts = (cur_robo_expanded_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_robo_expanded_visual_pts = cur_robo_expanded_visual_pts * 2. -1.
                    cur_robo_expanded_visual_pts = cur_robo_expanded_visual_pts * self.mult_const_after_cent # mult_const #
                
                cur_rot = cur_robo_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                ### transform by the glboal transformation and the translation ###
                cur_robo_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                ### transform by the glboal transformation and the translation ###
                cur_robo_expanded_visual_pts = torch.matmul(cur_robo_glb_rot, cur_robo_expanded_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_robo_glb_trans.unsqueeze(0) ## transformed pts ## 
                
                
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_robo_visual_pts.clone() # robo visual pts #
                
                # not taht the scale is differne but would not affect the final result 
                # expanded_pts #
                cur_robo_visual_pts_idxes = torch.arange(
                    start=cur_ts * self.robo_expanded_visual_pts_nn, end=(cur_ts + 1) * self.robo_expanded_visual_pts_nn, dtype=torch.long
                ).cuda()
                
                # expanded_actuator_pointact_forces, expanded_actuator_delta_offset
                if self.drive_pointset == "actions":
                    ''' Act-React-driven point motions '''
                    ###### get the point offset via actuation forces and reaction forces ######
                    ### actuation forces at this timestep ###
                    cur_visual_pts_forces = self.expanded_actuator_pointact_forces(cur_robo_visual_pts_idxes) * 1e7
                    cur_visual_pts_forces = cur_visual_pts_forces * pointset_expansion_alpha
                        
                    ## --- linear damping here ? ## 
                    cur_visual_pts_accs = cur_visual_pts_forces / self.mass_point_mass ### get the mass pont accs ## 
                    if cur_ts == 0:
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    else: # visual pts acc -> visual pts vels #
                        cur_visual_pts_vels = cur_visual_pts_accs * time_cons
                    self.timestep_to_robo_actuator_points_vels[cur_ts] = cur_visual_pts_vels.detach().clone()
                    cur_visual_pts_offsets = cur_visual_pts_vels * time_cons
            
                    # train_pointset_acts_via_deltas, diff_cur_visual_pts_offsets_with_ori
                    # cur_visual_pts_offsets_from_delta = mano_expanded_actuator_delta_offset_ori[cur_ts]
                    # cur_visual_pts_offsets_from_delta = self.mano_expanded_actuator_delta_offset.weight.data[cur_ts].detach()
                    cur_visual_pts_offsets_from_delta = self.expanded_actuator_delta_offset(cur_robo_visual_pts_idxes).detach()
                    ## 
                    diff_cur_visual_pts_offsets_with_ori = torch.sum((cur_visual_pts_offsets - cur_visual_pts_offsets_from_delta) ** 2, dim=-1).mean() 
                    tot_diff_robo_cur_visual_pts_offsets_with_ori.append(diff_cur_visual_pts_offsets_with_ori.item())
                    
                    self.timestep_to_robo_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ###### get the point offset via actuation forces and reaction forces ######
                    ''' Act-React-driven point motions '''
                elif self.drive_pointset == "states":
                    ''' Offset-driven point motions '''
                    cur_visual_pts_offsets = self.expanded_actuator_delta_offset(cur_robo_visual_pts_idxes)
                    # cur_visual_pts_offsets = cur_visual_pts_offsets * 10
                    self.timestep_to_robo_actuator_points_offsets[cur_ts] = cur_visual_pts_offsets.detach().clone()
                    ''' Offset-driven point motions '''
                    
                else:
                    raise ValueError(f"Unknown drive_pointset: {self.drive_pointset}")
                
                cur_robo_expanded_visual_pts = cur_robo_expanded_visual_pts + cur_visual_pts_offsets
                
                # print(f"cur_robo_expanded_visual_pts: {cur_robo_expanded_visual_pts.size()}")
                
                if self.sampled_robo_expanded_pts_idxes is None:
                    # if os.path.exists(redmax_sampled_verts_idxes_fn):
                    #     sampled_verts_idxes = np.load(redmax_sampled_verts_idxes_fn)
                    #     sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
                    # else:
                    # print(f"Sampling fps idxes for robot expanded pts...")
                    n_sampling = 10000
                    pts_fps_idx = data_utils.farthest_point_sampling(cur_robo_expanded_visual_pts.unsqueeze(0), n_sampling=n_sampling)
                    sampled_robo_expanded_pts_idxes = pts_fps_idx
                    # np.save(redmax_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
                    self.sampled_robo_expanded_pts_idxes = sampled_robo_expanded_pts_idxes
                self.timestep_to_expanded_robo_visual_pts[cur_ts] = cur_robo_expanded_visual_pts[self.sampled_robo_expanded_pts_idxes]
                
                
                self.free_def_bending_weight = 0.0
                # self.free_def_bending_weight = 0.5
                
                if i_iter == 0 and cur_ts == 0: ## for the tiantianquan sequence ##
                    dist_robot_pts_to_mano_pts = torch.sum(
                        (cur_robo_expanded_visual_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                    )
                    minn_dist_robot_pts_to_mano_pts, correspondence_pts_idxes = torch.min(dist_robot_pts_to_mano_pts, dim=-1)
                    minn_dist_robot_pts_to_mano_pts = torch.sqrt(minn_dist_robot_pts_to_mano_pts)
                    # dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.01
                    dist_smaller_than_thres = minn_dist_robot_pts_to_mano_pts < 0.005
                    
                    corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes] # in correspondence pts idxes ##
                    
                    dist_corr_correspondence_pts_to_mano_visual_pts = torch.sum(
                        (corr_correspondence_pts.unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1
                    )
                    dist_corr_correspondence_pts_to_mano_visual_pts = torch.sqrt(dist_corr_correspondence_pts_to_mano_visual_pts)
                    minn_dist_to_corr_pts, _ = torch.min(dist_corr_correspondence_pts_to_mano_visual_pts, dim=0)
                    anchored_mano_visual_pts = minn_dist_to_corr_pts < 0.005
                    
                corr_correspondence_pts = cur_visual_pts[correspondence_pts_idxes]
                # corr_robo = cur_visual_pts[sampled_verts_idxes]
                cd_robo_pts_to_corr_mano_pts = torch.sum( # distance from robot pts to the anchored mano pts
                    (cur_robo_expanded_visual_pts.unsqueeze(1) - cur_visual_pts[anchored_mano_visual_pts].unsqueeze(0).detach()) ** 2, dim=-1
                )
                
                # self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts[anchored_mano_visual_pts] # .detach().cpu().numpy()
                self.timestep_to_anchored_mano_pts[cur_ts] = cur_visual_pts # .detach().cpu().numpy()
                
                
                cd_robo_to_mano, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=-1)
                cd_mano_to_robo, _ = torch.min(cd_robo_pts_to_corr_mano_pts, dim=0)
                # diff_robo_to_corr_mano_pts = cd_mano_to_robo.mean()
                diff_robo_to_corr_mano_pts = cd_robo_to_mano.mean()
                
                mano_fingers = self.rhand_verts[cur_ts][self.mano_fingers]
                
                
                
                ##### finger cd loss -> to the anchored expanded actioning points #####
                ## fingeer tracking loss -> to each finger ##
                # loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef + pure_finger_tracking_loss * 0.5 # + diff_robo_to_corr_mano_pts_finger_tracking * self.finger_tracking_loss_coef
                loss_finger_tracking = diff_robo_to_corr_mano_pts * self.finger_cd_loss_coef # + pure_finger_tracking_loss * 0.5 
                
                
                ## TODO: add the glboal retargeting using fingers before conducting this approach 
                
                
                # def evaluate_tracking_loss():
                #     self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=False, contact_pairs_set=self.contact_pairs_set)
                    
                #     ## 
                #     # init states  # 
                #     # cur_ts % mano_nn_substeps == 0:
                #     if (cur_ts + 1) % mano_nn_substeps == 0:
                #         cur_passive_big_ts = cur_ts // mano_nn_substeps
                #         in_func_tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                #         # tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                #     else:
                #         in_func_tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                #     return in_func_tracking_loss

                    
                if contact_pairs_set is None:
                    self.contact_pairs_set = None
                else:
                    ## 
                    self.contact_pairs_set = contact_pairs_set.copy()
                
                # ### if traiing the jrbpt h
                # print(self.timestep_to_active_mesh[cur_ts].size(), cur_visual_pts_friction_forces.size()) # friction forces #
                ### optimize the robot tracing loss ###
                
                
                # if self.o
                
                if self.optimize_anchored_pts:
                    # anchored_cur_visual_pts_friction_forces = cur_visual_pts_friction_forces[anchored_mano_visual_pts]
                    contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_anchored_mano_pts, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set)
                else:
                    contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_expanded_robo_visual_pts, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=self.robot_actuator_friction_forces, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set, pts_frictional_forces=None)
                    
                    
                ### train with force to active ##
                if self.train_with_forces_to_active and (not self.use_mano_inputs):
                    # penetration_forces #
                    if torch.sum(self.other_bending_network.penetrating_indicator.float()) > 0.5:
                        net_penetrating_forces = self.other_bending_network.penetrating_forces
                        net_penetrating_points = self.other_bending_network.penetrating_points

                        
                        # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
                        self.timestep_to_penetration_points[cur_ts] = net_penetrating_points.detach().cpu().numpy()
                        self.timestep_to_penetration_points_forces[cur_ts] = net_penetrating_forces.detach().cpu().numpy()
                        
                        
                        sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[self.other_bending_network.penetrating_indicator]
                        
                        
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone() ## for
                        
                        net_penetrating_forces = torch.matmul(
                            cur_rot.transpose(1, 0), net_penetrating_forces.transpose(1, 0)
                        ).transpose(1, 0)
                        
                        net_penetrating_forces = (1.0 - pointset_expansion_alpha) * net_penetrating_forces
                        
                        net_penetrating_points = torch.matmul(
                            cur_rot.transpose(1, 0), (net_penetrating_points - cur_trans.unsqueeze(0)).transpose(1, 0)
                        ).transpose(1, 0)
                        
                        penetration_forces = net_penetrating_forces
                        
                    else:
                        penetration_forces = None
                        sampled_visual_pts_joint_idxes = None
                        ''' the bending network still have this property and we can get force values here for the expanded visual points '''
                        self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone()  
                
                
                if (cur_ts + 1) % mano_nn_substeps == 0:
                    cur_passive_big_ts = cur_ts // mano_nn_substeps
                    ### tracking loss between the predicted transformation and te tracking ###
                    tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                    tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                else:
                    tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()


                
                # diff_hand_tracking = torch.zeros((1,), dtype=torch.float32).cuda().mean() ## 
                
                # ## diff
                # # diff_hand_tracking_coef
                # # kinematics_proj_loss = kinematics_trans_diff + penetraton_penalty + diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss
                
                # # if self.use_mano_hand_for_test: ## only the kinematics mano hand is optimized here ##
                # #     kinematics_proj_loss = tracking_loss
                
                # # kinematics_proj_loss = hand_tracking_loss * 1e2
                
                # smaller_than_zero_level_set_indicator
                cur_interpenetration_nns = self.other_bending_network.smaller_than_zero_level_set_indicator.float().sum()
                
                
                # kinematics_proj_loss = diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss + penetraton_penalty
                # expanded_set_delta_motion_ori
                diff_actions = torch.sum(
                    (self.expanded_set_delta_motion_ori - self.mano_expanded_actuator_delta_offset.weight) ** 2, dim=-1
                )
                diff_actions = diff_actions.mean()
                
                reg_act_force_loss = torch.sum(
                    (mano_expanded_actuator_pointact_forces_ori - self.mano_expanded_actuator_pointact_forces.weight.data) ** 2, dim=-1
                )
                reg_act_force_loss = reg_act_force_loss.mean()
                
                # robo_expanded_set_delta_motion_ori, robo_expanded_actuator_pointact_forces_ori
                diff_robo_actions = torch.sum(
                    (robo_expanded_set_delta_motion_ori - self.expanded_actuator_delta_offset.weight) ** 2, dim=-1
                )
                diff_robo_actions = diff_robo_actions.mean()
                
                reg_robo_act_force_loss = torch.sum(
                    (robo_expanded_actuator_pointact_forces_ori - self.expanded_actuator_pointact_forces.weight.data) ** 2, dim=-1
                )
                reg_robo_act_force_loss = reg_robo_act_force_loss.mean()
                
                
                diff_actions = diff_actions + reg_act_force_loss + diff_robo_actions + reg_robo_act_force_loss
                
                
                kinematics_proj_loss = loss_finger_tracking + tracking_loss
                
                loss = kinematics_proj_loss # * self.loss_scale_coef ## get 
                
                
                mano_tracking_loss.append(loss_finger_tracking.detach().cpu().item())
                
                
                self.kines_optimizer.zero_grad()
                
                kinematics_proj_loss.backward(retain_graph=True)
                
                self.kines_optimizer.step()
                
                # if self.use_LBFGS:
                #     self.kines_optimizer.step(evaluate_tracking_loss) # 
                # else:
                #     self.kines_optimizer.step()

                # 
                # tracking_loss.backward(retain_graph=True)
                # if self.use_LBFGS:
                #     self.other_bending_network.reset_timestep_to_quantities(cur_ts)
                
                
                # robot_states_actions_diff_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # diff_actions
                robot_states_actions_diff_loss = diff_actions
                robo_actions_diff_loss.append(robot_states_actions_diff_loss)

                
                tot_losses.append(loss.detach().item()) # total losses # # total losses # 
                # tot_penalty_dot_forces_normals.append(cur_penalty_dot_forces_normals.detach().item())
                # tot_penalty_friction_constraint.append(cur_penalty_friction_constraint.detach().item())
                
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)


                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                
                self.update_learning_rate()
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            if len(tot_tracking_loss) > 0:
                tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            else:
                tot_tracking_loss = 0.0
            if len(tot_penetration_depth) > 0:
                tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            else:
                tot_penetration_depth = 0.0
            robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            if len(mano_tracking_loss) > 0:
                mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))
            else:
                mano_tracking_loss = 0.0
                
                
            
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')
            else:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} tracking_loss = {} mano_tracking_loss = {} penetration_depth = {} actions_diff_loss = {} lr={}'.format(self.iter_step, tot_losses, tot_tracking_loss, mano_tracking_loss, tot_penetration_depth, robo_actions_diff_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)

            
            # self.validate_mesh_robo_a()
            if i_iter % self.val_mesh_freq == 0:
                self.validate_mesh_robo_g()
                self.validate_mesh_robo()

            
            torch.cuda.empty_cache()
    

    ''' GRAB clips --- kinematics based finger retargeting ''' 
    def train_sparse_retar(self, ):
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()


        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        # load 
        ''' Load the robot hand '''
        model_path = self.conf['model.sim_model_path']
        self.hand_type = "shadow_hand"
        # if model_path.endswith(".xml"):
        #     # self.hand_type = "redmax_hand"
        #     robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        # else:
        # self.hand_type = "shadow_hand"
        robot_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path, args=None)
        
            
        self.robot_agent = robot_agent
        robo_init_verts = self.robot_agent.robot_pts
        robo_sampled_verts_idxes_fn = "robo_sampled_verts_idxes.npy"
        
        sampled_verts_idxes = np.load(robo_sampled_verts_idxes_fn)
        sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()

        self.robo_hand_faces = self.robot_agent.robot_faces
        
        
        ''' Load the robot hand '''
         ### adfte loadingjthe robot hand ##
        self.robot_delta_angles = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=4,
        ).cuda()
        torch.nn.init.zeros_(self.robot_delta_angles.weight)
        self.robot_delta_angles.weight.data[:, 0] = 1.0
        
        
        self.robot_delta_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3,
        ).cuda()
        torch.nn.init.zeros_(self.robot_delta_trans.weight)
        
        
        self.robot_delta_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3,
        ).cuda()
        torch.nn.init.zeros_(self.robot_delta_glb_trans.weight)
        
        ### adfte loadingjthe robot hand ##
        self.robot_delta_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_delta_states.weight)
        self.robot_delta_states.weight.data[0, 24] = 1.0
        
        self.robot_states = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_states.weight)
        self.robot_states.weight.data[:, 24] = 1.0
        
        self.robot_init_states = nn.Embedding(
            num_embeddings=1, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.robot_init_states.weight)
        self.robot_init_states.weight.data[0, 24] = 1.0
        
        
        self.robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=4
        ).cuda()
        # [6.123234e-17, 0.000000e+00 0.000000e+00 1.000000e+00 ]
        
        # if self.hand_type == "redmax":
            
        if self.hand_type == "shadow":
            self.robot_glb_rotation.weight.data[:, 0] = 6.123234e-17
            self.robot_glb_rotation.weight.data[:, 1:] = 0.
            self.robot_glb_rotation.weight.data[:, 3] = 1.0
        else:
            self.robot_glb_rotation.weight.data[:, 0] = 1.
            self.robot_glb_rotation.weight.data[:, 1:] = 0.
        
        self.robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.robot_glb_trans.weight)
        # params_to_train += list(self.robot_glb_trans.parameters())   
        
        load_delta_trans = False
        if 'model.load_optimized_init_transformations' in self.conf and len(self.conf['model.load_optimized_init_transformations']) > 0: 
            print(f"[Robot] Loading optimized init transformations from {self.conf['model.load_optimized_init_transformations']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_transformations']
            # cur_optimized_init_actions = # optimized init states 
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )

            try:
                if optimized_init_actions_ckpt['robot_init_states']['weight'].size(0) > self.robot_init_states.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_init_states']['weight'].data = optimized_init_actions_ckpt['robot_init_states']['weight'].data[:self.robot_init_states.weight.data.size(0)]
                self.robot_init_states.load_state_dict(optimized_init_actions_ckpt['robot_init_states'])
            except:
                pass
            if optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0) > self.robot_glb_rotation.weight.data.size(0):
                optimized_init_actions_ckpt['robot_glb_rotation']['weight'].data = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].data[:self.robot_glb_rotation.weight.data.size(0)]
            self.robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
            if 'robot_delta_states' in optimized_init_actions_ckpt:
                try:
                    if optimized_init_actions_ckpt['robot_delta_states']['weight'].size(0) > self.robot_delta_states.weight.data.size(0):
                        optimized_init_actions_ckpt['robot_delta_states']['weight'].data = optimized_init_actions_ckpt['robot_delta_states']['weight'].data[:self.robot_delta_states.weight.data.size(0)]
                    self.robot_delta_states.load_state_dict(optimized_init_actions_ckpt['robot_delta_states'])
                except:
                    pass
            if 'robot_states' in optimized_init_actions_ckpt:
                if optimized_init_actions_ckpt['robot_states']['weight'].size(0) > self.robot_states.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_states']['weight'].data = optimized_init_actions_ckpt['robot_states']['weight'].data[:self.robot_states.weight.data.size(0)]
                self.robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
            # if 'robot_delta_states'  ## robot delta states ##
            if 'robot_actions' in optimized_init_actions_ckpt:
                if optimized_init_actions_ckpt['robot_actions']['weight'].size(0) > self.robot_actions.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_actions']['weight'].data = optimized_init_actions_ckpt['robot_actions']['weight'].data[:self.robot_actions.weight.data.size(0)]
                self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            # self.robot_actions.load_state_dict(optimized_init_actions_ckpt['robot_actions'])
            # self.mano_robot_actuator_friction_forces.load_state_dict(optimized_init_actions_ckpt['robot_actuator_friction_forces'])
            if optimized_init_actions_ckpt['robot_glb_trans']['weight'].data.size(0) > self.robot_glb_trans.weight.data.size(0):
                optimized_init_actions_ckpt['robot_glb_trans']['weight'].data = optimized_init_actions_ckpt['robot_glb_trans']['weight'].data[:self.robot_glb_trans.weight.data.size(0)]
            self.robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
            
            if 'robot_delta_angles' in optimized_init_actions_ckpt:
                if optimized_init_actions_ckpt['robot_delta_angles']['weight'].data.size(0) > self.robot_delta_angles.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_delta_angles']['weight'].data = optimized_init_actions_ckpt['robot_delta_angles']['weight'].data[:self.robot_delta_angles.weight.data.size(0)]
                self.robot_delta_angles.load_state_dict(optimized_init_actions_ckpt['robot_delta_angles'])
            if 'robot_delta_trans' in optimized_init_actions_ckpt:
                if optimized_init_actions_ckpt['robot_delta_trans']['weight'].data.size(0) > self.robot_delta_trans.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_delta_trans']['weight'].data = optimized_init_actions_ckpt['robot_delta_trans']['weight'].data[:self.robot_delta_trans.weight.data.size(0)]
                self.robot_delta_trans.load_state_dict(optimized_init_actions_ckpt['robot_delta_trans'])

            if 'robot_delta_glb_trans' in optimized_init_actions_ckpt:
                load_delta_trans = True
                if optimized_init_actions_ckpt['robot_delta_glb_trans']['weight'].size(0) > self.robot_delta_glb_trans.weight.data.size(0):
                    optimized_init_actions_ckpt['robot_delta_glb_trans']['weight'].data = optimized_init_actions_ckpt['robot_delta_glb_trans']['weight'].data[:self.robot_delta_glb_trans.weight.data.size(0)]
                
                self.robot_delta_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_delta_glb_trans'])
            
        # robot_delta_glb_trans, robot_delta_glb_trans_lft
        if (not load_delta_trans):
            tot_robot_delta_trans = []
            for i_fr in range(num_steps):
                if i_fr == 0:
                    cur_robot_delta_trans = self.robot_glb_trans.weight.data[0]
                else:
                    cur_robot_delta_trans = self.robot_glb_trans.weight.data[i_fr] - self.robot_glb_trans.weight.data[i_fr - 1]
                tot_robot_delta_trans.append(cur_robot_delta_trans)
            tot_robot_delta_trans = torch.stack(tot_robot_delta_trans, dim=0)
            self.robot_delta_glb_trans.weight.data.copy_(tot_robot_delta_trans) # use delta states ##
            
            
        # ### load init transformations ckpts ### #
        for i_ts in range(self.robot_glb_trans.weight.size(0)):
            if i_ts == 0:
                self.robot_delta_trans.weight.data[i_ts, :] = self.robot_glb_trans.weight.data[i_ts, :].clone()
            else:
                prev_ts = i_ts - 1
                self.robot_delta_trans.weight.data[i_ts, :] = self.robot_glb_trans.weight.data[i_ts, :] - self.robot_glb_trans.weight.data[i_ts - 1, :]
        
        self.robot_delta_angles.weight.data[0, :] = self.robot_glb_rotation.weight.data[0, :].clone()
        
        params_to_train = [] # params to train #
        # params_to_train += list(self.robot_delta_states.parameters())
        params_to_train += list(self.robot_glb_rotation.parameters())
        # params_to_train += list(self.robot_init_states.parameters())
        params_to_train += list(self.robot_glb_trans.parameters())
        
        params_to_train += list(self.robot_delta_angles.parameters())

        params_to_train += list(self.robot_delta_trans.parameters())        
        
        if not self.retar_only_glb:
            params_to_train += list(self.robot_states.parameters())
            params_to_train += list(self.robot_delta_states.parameters())
            
        
        self.mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, 86, 364, 477, 588, 699]
        self.robot_fingers = [6684, 9174, 53, 1623, 3209, 4495, 10028, 8762, 1030, 2266, 3822, 5058, 7074]
        
        
        # self.minn_robo_pts = -0.1
        # self.maxx_robo_pts = 0.2
        # self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
        
        # if self.hand_type == "redmax_hand":
        #     self.maxx_robo_pts = 25.
        #     self.minn_robo_pts = -15.
        #     self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
        #     self.mult_const_after_cent = 0.5437551664260203
        # else:
        self.minn_robo_pts = -0.1
        self.maxx_robo_pts = 0.2
        self.extent_robo_pts = self.maxx_robo_pts - self.minn_robo_pts
        self.mult_const_after_cent = 0.437551664260203
        
        ## for grab ##
        self.mult_const_after_cent = self.mult_const_after_cent / 3. * 0.9507
        
        
        # # optimize with intermediates #
        self.mano_fingers = torch.tensor(self.mano_fingers, dtype=torch.long).cuda()
        self.robot_fingers = torch.tensor(self.robot_fingers, dtype=torch.long).cuda()
        
        self.nn_ts = self.nn_timesteps - 1
        # self.optimize_with_intermediates = False
        
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
 
        
        self.timestep_to_active_mesh = {}
        # ref_expanded_visual_pts, minn_idx_expanded_visual_pts_to_link_pts #
        # minn_idx_expanded_visual_pts_to_link_pts #
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
     
        
        self.iter_step = 0
        
        
        
        self.minn_retargeting_loss = 1e27
        
        
        nn_glb_retar_eps = 1000
        nn_wstates_retar_eps = 2000
        nn_dense_retar_eps = 2000
        
        tot_retar_eps = nn_glb_retar_eps + nn_wstates_retar_eps + nn_dense_retar_eps
        
        transfer_to_wstates = False
        transfer_to_denseretar = False
        
        finger_sampled_idxes = None
        
        minn_dist_mano_pts_to_visual_pts_idxes = None
        
        for i_iter in tqdm(range(tot_retar_eps)):
            tot_losses = []

            tot_tracking_loss = []
            
            # timestep 
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # self.timestep_to_
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}

            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            self.ts_to_robot_fingers = {}
            self.ts_to_mano_fingers = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            # tot_grad_qs = []
            
            self.ts_to_glb_quat = {}
            self.ts_to_glb_trans = {}
            
            
            # robo_intermediates_states = []
            
            # tot_penetration_depth = []
            
            # robo_actions_diff_loss = []
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            for cur_ts in range(self.nn_ts):
                tot_redmax_actions = []
                
                actions = {}
                
                self.free_def_bending_weight = 0.0

                if self.drive_glb_delta:
                    print(f"drive_glb_delta!")
                    if cur_ts == 0:
                        cur_glb_rot_quat = self.robot_delta_angles(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                        
                        cur_glb_rot_quat = cur_glb_rot_quat / torch.clamp(torch.norm(cur_glb_rot_quat, dim=-1, p=2), min=1e-7)
                        # cur_glb_rot_quat = cur_glb_rot.clone()
                        cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot_quat)
                        
                        self.ts_to_glb_quat[cur_ts] = cur_glb_rot_quat.detach().clone()
                        
                        cur_glb_trans = self.robot_delta_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)

                        self.ts_to_glb_trans[cur_ts] = cur_glb_trans.detach().clone()
                        
                        self.robot_glb_rotation.weight.data[cur_ts, :] = cur_glb_rot_quat.detach().clone()
                        self.robot_glb_trans.weight.data[cur_ts, :] = cur_glb_trans.detach().clone()
                        
                    else:
                        prev_glb_quat = self.ts_to_glb_quat[cur_ts - 1]
                        cur_glb_rot_angle = self.robot_delta_angles(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)[1:]
                        cur_glb_rot_quat = prev_glb_quat + fields.update_quaternion(cur_glb_rot_angle, prev_glb_quat)
                        cur_glb_rot_quat = cur_glb_rot_quat / torch.clamp(torch.norm(cur_glb_rot_quat, dim=-1, p=2), min=1e-7)
                        
                        cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot_quat)
                        
                        self.ts_to_glb_quat[cur_ts] = cur_glb_rot_quat.detach().clone()
                        
                        cur_delta_glb_trans = self.robot_delta_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                        prev_glb_trans = self.ts_to_glb_trans[cur_ts - 1].detach().clone()
                        cur_glb_trans = prev_glb_trans + cur_delta_glb_trans
                        
                        self.ts_to_glb_trans[cur_ts] = cur_glb_trans.detach().clone()
                        
                        self.robot_glb_rotation.weight.data[cur_ts, :] = cur_glb_rot_quat.detach().clone()
                        self.robot_glb_trans.weight.data[cur_ts, :] = cur_glb_trans.detach().clone()
                
                elif self.retar_delta_glb_trans:
                    print("fzzzzzzz")
                    if cur_ts == 0:
                        cur_glb_trans = self.robot_delta_glb_trans(torch.zeros((1,), dtype=torch.long).cuda()).squeeze(0)
                    else:
                        prev_trans = torch.sum( self.robot_delta_glb_trans.weight.data[:cur_ts], dim=0).detach()
                        cur_delta_glb_trans = self.robot_delta_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                        cur_glb_trans = prev_trans + cur_delta_glb_trans
                    self.robot_glb_trans.weight.data[cur_ts, :] = cur_glb_trans[:].detach().clone()

                    cur_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                    cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) 
                
                else:
                    cur_glb_rot = self.robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    
                    # cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                    
                    cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                    # cur_glb_rot_quat = cur_glb_rot.clone()
                
                    cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                
                
                    cur_glb_trans = self.robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                # cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans # 
                
                ########### Delta states ###########
                if self.drive_glb_delta:
                    robo_delta_states = self.robot_delta_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.robot_agent.active_robot.set_delta_state_and_update_v2(robo_delta_states, cur_ts) ## delta states ##
                    self.robot_states.weight.data[cur_ts, :] = self.robot_agent.get_joint_state(cur_ts - 1, self.robot_states.weight.data[cur_ts, :])
                    # self.robot_states_sv[cur_ts, : ] = self.robot_agent.get_joint_state(cur_ts - 1, self.robot_states_sv[cur_ts, :])
                else:
                    # if self.hand_type == "shadow_hand":
                    link_cur_states = self.robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.robot_agent.set_init_states_target_value(link_cur_states)
                    
                # robo_delta_states = self.robot_delta_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # self.robot_agent.active_robot.set_delta_state_and_update_v2(robo_delta_states, cur_ts) ## delta states ##
                # self.robot_states.weight.data[cur_ts, :] = self.robot_agent.get_joint_state(cur_ts - 1, self.robot_states.weight.data[cur_ts, :])
                cur_visual_pts = self.robot_agent.get_init_state_visual_pts()  # get init state visual pts
            
                

                if not self.use_scaled_urdf:
                    ### transform the visual pts ###
                    cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    cur_visual_pts = cur_visual_pts * 2. - 1.
                    cur_visual_pts = cur_visual_pts * self.mult_const_after_cent
                    
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                
                ### transform by the glboal transformation and the translation ###
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) 
                
                mano_fingers = self.rhand_verts[cur_ts][self.mano_fingers]
                
                
                if self.hand_type == 'shadow_hand':
                    if self.retar_dense_corres and cur_ts == 0:
                        dist_mano_pts_to_visual_pts = torch.sum(
                            (self.rhand_verts[cur_ts].unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1 #### nn_mano_pts x nn_robo_pts x 3
                        )
                        dist_mano_pts_to_visual_pts = torch.sqrt(dist_mano_pts_to_visual_pts)
                        minn_dist_mano_pts_to_visual_pts, minn_dist_mano_pts_to_visual_pts_idxes = torch.min(dist_mano_pts_to_visual_pts, dim=-1) ### nn_mano_pts
                        minn_dist_mano_pts_to_visual_pts_idxes = minn_dist_mano_pts_to_visual_pts_idxes.detach()
                elif self.hand_type == 'redmax_hand':
                    if self.retar_dense_corres and cur_ts == 0:
                        dist_mano_pts_to_visual_pts = torch.sum(
                            (self.rhand_verts[cur_ts].unsqueeze(1) - cur_visual_pts.unsqueeze(0)) ** 2, dim=-1 #### nn_mano_pts x nn_robo_pts x 3
                        )
                        dist_mano_pts_to_visual_pts = torch.sqrt(dist_mano_pts_to_visual_pts)
                        minn_dist_mano_pts_to_visual_pts, minn_dist_mano_pts_to_visual_pts_idxes = torch.min(dist_mano_pts_to_visual_pts, dim=-1) ### nn_mano_pts
                        minn_dist_mano_pts_to_visual_pts_idxes = minn_dist_mano_pts_to_visual_pts_idxes.detach()
                
                
                if self.hand_type == 'redmax_hand':
                    # sampled_verts_idxes
                    # print(f"cur_visual_pts: {cur_visual_pts.size()}, maxx_verts_idx: {torch.max(sampled_verts_idxes)}, minn_verts_idx: {torch.min(sampled_verts_idxes)}")
                    # robo_fingers = cur_visual_pts[sampled_verts_idxes][self.robot_fingers]
                    robo_fingers = cur_visual_pts[self.robot_fingers]
                else:
                    robo_fingers = cur_visual_pts[self.robot_fingers]
                self.ts_to_robot_fingers[cur_ts] = robo_fingers.detach().cpu().numpy()
                self.ts_to_mano_fingers[cur_ts] = mano_fingers.detach().cpu().numpy()
                
                # diff_redmax_visual_pts_with_ori_visual_pts = torch.sum(
                #     (cur_visual_pts[sampled_verts_idxes] - self.timestep_to_active_mesh_opt_ours_sim[cur_ts].detach()) ** 2, dim=-1
                # )
                # diff_redmax_visual_pts_with_ori_visual_pts = diff_redmax_visual_pts_with_ori_visual_pts.mean()
                
                # ts_to_robot_fingers, ts_to_mano_fingers
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                
                # ragged_dist = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # dist_transformed_expanded_visual_pts_to_ori_visual_pts = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                # diff_cur_states_to_ref_states = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                
                
                self.free_def_bending_weight = 0.0
                
                if self.retar_dense_corres and (minn_dist_mano_pts_to_visual_pts_idxes is not None):
                    corres_robo_pts = cur_visual_pts[minn_dist_mano_pts_to_visual_pts_idxes]
                    tracking_loss = torch.sum((self.rhand_verts[cur_ts] - corres_robo_pts) ** 2, dim=-1) ### nn_mano_pts ###
                    tracking_loss = torch.mean(tracking_loss) * 20.0
                else:   
                    tracking_loss = torch.sum((mano_fingers - robo_fingers) ** 2)
                
                loss = tracking_loss #  + self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                
                # kinematics_trans_diff = (robot_rotation_diff + robot_trans_diff + robot_delta_states_diff) * self.robot_actions_diff_coef
                
                # kinematics_proj_loss = kinematics_trans_diff + penetraton_penalty + tracking_loss
                
                loss = loss # + (robot_rotation_diff + robot_trans_diff) * self.robot_actions_diff_coef
                
                # loss = kinematics_proj_loss
                tot_losses.append(loss)
                # self.iter_step += 1
                # self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                
                
                self.optimizer.zero_grad()
                
                loss.backward(retain_graph=True)
                
                self.optimizer.step()


                
                # tot_losses.append(loss.detach().item())
                # # tot_penalty_dot_forces_normals.append(cur_penalty_dot_forces_normals.detach().item())
                # # tot_penalty_friction_constraint.append(cur_penalty_friction_constraint.detach().item())
                
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                
                # # if self.iter_step % self.save_freq == 0:
                # #     self.save_checkpoint() # a smart solution for them ? # # save checkpoint #
                self.update_learning_rate() ## update learning rate ##
                
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache() 
                torch.cuda.empty_cache() 
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''

            
            
            # tot_losses = sum(tot_losses) / float(len(tot_losses))
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            
            saved_best = False

            if self.retar_dense_corres:
                if tot_losses < self.minn_retargeting_loss:
                    self.minn_retargeting_loss = tot_losses
                    self.save_checkpoint(tag="denseretar_best", niter=True)
                    self.validate_mesh_robo()
                    saved_best = True
                    
            if tot_losses < self.minn_retargeting_loss:
                self.minn_retargeting_loss = tot_losses
                if self.retar_dense_corres:
                    self.save_checkpoint(tag="denseretar_best", niter=True)
                elif (self.retar_only_glb and transfer_to_wstates) or (not self.retar_only_glb):
                    self.save_checkpoint(tag="wstates_best", niter=True)
                elif self.retar_only_glb:
                    self.save_checkpoint(tag="glbonly_best", niter=True)
            
            
            if i_iter == 0 or  (i_iter % self.ckpt_sv_freq == 0):
                #### transfer to dense-retargeting ####
                if (self.retar_only_glb and transfer_to_denseretar) or ((not self.retar_only_glb) and transfer_to_denseretar):
                    self.save_checkpoint(tag="denseretar")
                elif self.retar_only_glb and transfer_to_wstates:
                    self.save_checkpoint(tag="towstates")
                else:
                    self.save_checkpoint() # a smart solution for them ? # # save checkpoint #
            
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'i_iter: {} iter:{:8>d} loss = {} lr={}'.format(i_iter, self.iter_step, tot_losses, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')
                
            ### train finger retargeting

            # self.validate_mesh_robo_a()
            if (not saved_best) and (i_iter % self.val_mesh_freq == 0):
                self.validate_mesh_robo()
                
            if self.retar_only_glb and (i_iter == 1000):
                params_to_train = [] # params to train #
                params_to_train += list(self.robot_glb_rotation.parameters())
                params_to_train += list(self.robot_glb_trans.parameters())
                
                params_to_train += list(self.robot_states.parameters())
                params_to_train += list(self.robot_delta_glb_trans.parameters())
                params_to_train += list(self.robot_delta_states.parameters())
                self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
                transfer_to_wstates = True

            if (self.retar_only_glb and (i_iter == nn_wstates_retar_eps + nn_glb_retar_eps)) or ((not self.retar_only_glb) and i_iter == nn_wstates_retar_eps): 
                params_to_train = [] # params to train # ### judge ###
                params_to_train += list(self.robot_glb_rotation.parameters())
                params_to_train += list(self.robot_glb_trans.parameters())
                params_to_train += list(self.robot_states.parameters())
                params_to_train += list(self.robot_delta_glb_trans.parameters())
                params_to_train += list(self.robot_delta_states.parameters())
                
                self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
                transfer_to_denseretar = True
                self.retar_dense_corres = True

    
    
            torch.cuda.empty_cache()
    
    
    
    ''' GRAB & TACO clips; MANO dynamic hand ''' 
    def train_dyn_mano_model(self, ):
        
        # chagne # # mano notjmano but the mano ---> optimize the mano delta states? #
        ### the real robot actions from mano model rules ###
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # update learning rrate # 
        # robot actions ##
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        # load --- and can load other states as well ##
        ''' Load the robot hand '''
        # model_path = self.conf['model.sim_model_path'] # 
        # robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        # self.robot_agent = robot_agent
        # robo_init_verts = self.robot_agent.robot_pts
        # robo_sampled_verts_idxes_fn = "robo_sampled_verts_idxes.npy"
        # if os.path.exists(robo_sampled_verts_idxes_fn):
        #     sampled_verts_idxes = np.load("robo_sampled_verts_idxes.npy")
        #     sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
        # else:
        #     n_sampling = 1000
        #     pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
        #     sampled_verts_idxes = pts_fps_idx
        #     np.save(robo_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
        # self.robo_hand_faces = self.robot_agent.robot_faces
        # self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        ## load the robot hand ##
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- 
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m ### ndof_m ### # redma # x_sim 
        
        
        ''' Load the mano hand ''' # dynamic mano hand jin it #
        model_path_mano = self.conf['model.mano_sim_model_path']
        if not os.path.exists(model_path_mano):
            model_path_mano = "rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507_nroot.urdf"
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.robo_hand_faces = self.mano_agent.robot_faces

        
        nn_substeps = 10
        
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        
        
        ''' Expnad the current visual points ''' 
        # expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.expanded_visual_pts_nn = expanded_visual_pts.size(0)
        # expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        # expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        # np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy)
        # # ''' Expnad the current visual points '''  #  # differentiate through the simulator? # # 
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        params_to_train += list(self.mano_robot_actions.parameters())
        
        # self.mano_robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # # params_to_train += list(self.robot_delta_states.parameters())
        
        # self.mano_robot_init_states = nn.Embedding(
        #     num_embeddings=1, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # # params_to_train += list(self.robot_init_states.parameters())
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        params_to_train += list(self.mano_robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        params_to_train += list(self.mano_robot_glb_trans.parameters())   
        # 
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        # self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        params_to_train += list(self.mano_robot_states.parameters())
        
        
        ''' Load optimized MANO hand actions and states '''
        # ### laod optimized init actions ####
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            # cur_optimized_init_actions = # optimized init states 
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            
            if 'mano_robot_states' in optimized_init_actions_ckpt:
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])
            # self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
            if 'mano_robot_actions' in optimized_init_actions_ckpt:
                self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['mano_robot_actions'])
               
        else:
            optimized_init_actions_ckpt = None

        mano_glb_trans_np_data = self.mano_robot_glb_trans.weight.data.detach().cpu().numpy()
        mano_glb_rotation_np_data = self.mano_robot_glb_rotation.weight.data.detach().cpu().numpy()
        mano_states_np_data = self.mano_robot_states.weight.data.detach().cpu().numpy()
        
        if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
            object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
            object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
        
        
        ''' Scaling constants '''
        self.mano_mult_const_after_cent = 0.9507
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']
        
        # mano_to_dyn_corr_pts_idxes_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/nearest_dyn_verts_idxes.npy"
        # if not os.path.exists(mano_to_dyn_corr_pts_idxes_fn):
        #     mano_to_dyn_corr_pts_idxes_fn = "/data/xueyi/diffsim/NeuS/rsc/mano/nearest_dyn_verts_idxes.npy"
        
        mano_to_dyn_corr_pts_idxes_fn = "./assets/nearest_dyn_verts_idxes.npy"
        self.mano_to_dyn_corr_pts_idxes = np.load(mano_to_dyn_corr_pts_idxes_fn, allow_pickle=True)
        self.mano_to_dyn_corr_pts_idxes = torch.from_numpy(self.mano_to_dyn_corr_pts_idxes).long().cuda() 
        
        print(f"mano_to_dyn_corr_pts_idxes: {self.mano_to_dyn_corr_pts_idxes.size()}")
        

        self.nn_ts = self.nn_timesteps
        
        
        
        ''' Set actions for the redmax simulation and add parameters to params-to-train '''
        
        # params_to_train = []
        # params_to_train += list(self.redmax_robot_actions.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        # init_rot = R.random().as_quat()
        # init_rot = torch.from_numpy(init_rot).float().cuda()
        # self.robot_glb_rotation.weight.data[0, :] = init_rot[:] # init rot
        
        
        
        # ### Constraint set ###
        # self.robot_hand_states_only_allowing_neg = torch.tensor( [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16], dtype=torch.long).cuda()
        
        
        self.timestep_to_active_mesh = {}
        # ref_expanded_visual_pts, minn_idx_expanded_visual_pts_to_link_pts #
        # minn_idx_expanded_visual_pts_to_link_pts #
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        self.timestep_to_active_mesh_w_delta_states = {}
        
          
        self.iter_step = 0


        self.minn_tracking_loss = 1e27
        
        # for i_iter in tqdm(range(100000)):
        for i_iter in tqdm(range(1000)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep 
            # self.timestep_to_active_mesh = {}
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            # self.timestep_to_
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            # correspondence_pts_idxes = None
            # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            
            # tot_grad_qs = []
            
            # robo_intermediates_states = []
            
            # tot_penetration_depth = []
            
            # robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            # for cur_ts in range(self.nn_ts):
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps):
                tot_redmax_actions = []
                
                
                actions = {}
                
                self.free_def_bending_weight = 0.0

                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone()
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans

                if self.optimize_dyn_actions:
                    ''' Articulated joint forces-driven mano robot '''
                    ### current -> no penetration setting; no contact setting ###
                    link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=None, sampled_visual_pts_joint_idxes=None)
                else:
                    ''' Articulated states-driven mano robot '''
                    link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.mano_agent.set_init_states_target_value(link_cur_states)
                
                # optimize_dyn_actions
                
                
                cur_visual_pts = self.mano_agent.get_init_state_visual_pts() 


                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                ## 
                ### transform by the glboal transformation and the translation ###
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## visual pts ##
                
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                
                # 0.1 ## as the initial one #
                
                
                cur_kine_rhand_verts = self.rhand_verts[cur_ts // mano_nn_substeps]
                cur_dyn_visual_pts_to_mano_verts = cur_visual_pts[self.mano_to_dyn_corr_pts_idxes]
                diff_hand_tracking = torch.mean(
                    torch.sum((cur_kine_rhand_verts - cur_dyn_visual_pts_to_mano_verts) ** 2, dim=-1)
                )
                
                
                
                self.optimizer.zero_grad()
                loss = diff_hand_tracking
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                # diff_hand_tracking # diff hand ## 
                mano_tracking_loss.append(diff_hand_tracking.detach().cpu().item())
                
                
                
                tot_losses.append(loss.detach().item())


                self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                self.iter_step += 1
                self.update_learning_rate() ## update learning rate ##
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            
            if (i_iter % self.ckpt_sv_freq) == 0:
                self.save_checkpoint() # a smart solution for them ? # # save checkpoint #
                
            
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            # tot_penalty_dot_forces_normals = sum(tot_penalty_dot_forces_normals) / float(len(tot_penalty_dot_forces_normals))
            # tot_penalty_friction_constraint = sum(tot_penalty_friction_constraint) / float(len(tot_penalty_friction_constraint))
            # tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            # tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            # robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))

            
            # if tot_losses < self.minn_tracking_loss:
            #     self.minn_tracking_loss = tot_losses
            #     self.save_checkpoint(tag="best")
                
                
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} mano_tracking_loss = {} lr={}'.format(self.iter_step, tot_losses, mano_tracking_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')

            # self.validate_mesh_robo_a()
            if i_iter % self.val_mesh_freq == 0:
                # self.validate_mesh_robo()
                self.validate_mesh_robo_e()

            torch.cuda.empty_cache()


    ''' GRAB & TACO clips; MANO dynamic hand ''' 
    def train_dyn_mano_model_wreact(self, ):
        
        # chagne # # mano notjmano but the mano ---> optimize the mano delta states? #
        ### the real robot actions from mano model rules ###
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate() # update learning rrate # 
        # robot actions ##
        
        nn_timesteps = self.timestep_to_passive_mesh.size(0)
        self.nn_timesteps = nn_timesteps
        num_steps = self.nn_timesteps
        
        # load --- and can load other states as well ##
        ''' Load the robot hand '''
        # model_path = self.conf['model.sim_model_path'] # 
        # robot_agent = dyn_model_act.RobotAgent(xml_fn=model_path, args=None)
        # self.robot_agent = robot_agent
        # robo_init_verts = self.robot_agent.robot_pts
        # robo_sampled_verts_idxes_fn = "robo_sampled_verts_idxes.npy"
        # if os.path.exists(robo_sampled_verts_idxes_fn):
        #     sampled_verts_idxes = np.load("robo_sampled_verts_idxes.npy")
        #     sampled_verts_idxes = torch.from_numpy(sampled_verts_idxes).long().cuda()
        # else:
        #     n_sampling = 1000
        #     pts_fps_idx = data_utils.farthest_point_sampling(robo_init_verts.unsqueeze(0), n_sampling=n_sampling)
        #     sampled_verts_idxes = pts_fps_idx
        #     np.save(robo_sampled_verts_idxes_fn, sampled_verts_idxes.detach().cpu().numpy())
        # self.robo_hand_faces = self.robot_agent.robot_faces
        # self.sampled_verts_idxes = sampled_verts_idxes
        ''' Load the robot hand '''
        
        ## load the robot hand ##
        
        ''' Load robot hand in DiffHand simulator '''
        # redmax_sim = redmax.Simulation(model_path)
        # redmax_sim.reset(backward_flag = True) # redmax_sim -- 
        # # ### redmax_ndof_u, redmax_ndof_r ### #
        # redmax_ndof_u = redmax_sim.ndof_u
        # redmax_ndof_r = redmax_sim.ndof_r
        # redmax_ndof_m = redmax_sim.ndof_m ### ndof_m ### # redma # x_sim 
        
        
        ''' Load the mano hand ''' # dynamic mano hand jin it #
        model_path_mano = self.conf['model.mano_sim_model_path']
        if not os.path.exists(model_path_mano): ## the model path mano ##
            model_path_mano = "rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507_nroot.urdf"
        # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
        mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
        self.mano_agent = mano_agent
        # ''' Load the mano hand '''
        self.robo_hand_faces = self.mano_agent.robot_faces

        
        nn_substeps = 10
        
        mano_nn_substeps = 1
        # mano_nn_substeps = 10 # 
        self.mano_nn_substeps = mano_nn_substeps
        
        
        
        ''' Expnad the current visual points ''' 
        # expanded_visual_pts = self.mano_agent.active_robot.expand_visual_pts(expand_factor=self.pointset_expand_factor, nn_expand_pts=self.pointset_nn_expand_pts)
        # self.expanded_visual_pts_nn = expanded_visual_pts.size(0)
        # expanded_visual_pts_npy = expanded_visual_pts.detach().cpu().numpy()
        # expanded_visual_pts_sv_fn = "expanded_visual_pts.npy"
        # np.save(expanded_visual_pts_sv_fn, expanded_visual_pts_npy)
        # # ''' Expnad the current visual points '''  #  # differentiate through the simulator? # # 
        
        params_to_train = [] # params to train #
        ### robot_actions, robot_init_states, robot_glb_rotation, robot_actuator_friction_forces, robot_glb_trans ###
        
        ''' Define MANO robot actions, delta_states, init_states, frictions, and others '''
        self.mano_robot_actions = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_actions.weight)
        params_to_train += list(self.mano_robot_actions.parameters())
        
        # self.mano_robot_delta_states = nn.Embedding(
        #     num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.mano_robot_delta_states.weight)
        # # params_to_train += list(self.robot_delta_states.parameters())
        
        # self.mano_robot_init_states = nn.Embedding(
        #     num_embeddings=1, embedding_dim=60,
        # ).cuda()
        # torch.nn.init.zeros_(self.mano_robot_init_states.weight)
        # # params_to_train += list(self.robot_init_states.parameters()) ## params to train the aaa ##
        
        self.mano_robot_glb_rotation = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
        ).cuda()
        self.mano_robot_glb_rotation.weight.data[:, 0] = 1.
        self.mano_robot_glb_rotation.weight.data[:, 1:] = 0.
        params_to_train += list(self.mano_robot_glb_rotation.parameters())
        
        
        self.mano_robot_glb_trans = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_glb_trans.weight)
        params_to_train += list(self.mano_robot_glb_trans.parameters())   
        # 
        self.mano_robot_states = nn.Embedding(
            num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.mano_robot_states.weight)
        # self.mano_robot_states.weight.data[0, :] = self.mano_robot_init_states.weight.data[0, :].clone()
        params_to_train += list(self.mano_robot_states.parameters())
        
        
        ''' Load optimized MANO hand actions and states '''
        # ### laod optimized init actions ####
        if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
            ''' load the optimized actions ''' #### init transformations ####
            print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
            cur_optimized_init_actions_fn = self.conf['model.load_optimized_init_actions']
            # cur_optimized_init_actions = # optimized init states 
            optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location=self.device, )
            
            if 'mano_robot_states' in optimized_init_actions_ckpt: ## 
                self.mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
            self.mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
            self.mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])
            # self.mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
            if 'mano_robot_actions' in optimized_init_actions_ckpt:
                self.mano_robot_actions.load_state_dict(optimized_init_actions_ckpt['mano_robot_actions'])
        else:
            optimized_init_actions_ckpt = None

        mano_glb_trans_np_data = self.mano_robot_glb_trans.weight.data.detach().cpu().numpy()
        mano_glb_rotation_np_data = self.mano_robot_glb_rotation.weight.data.detach().cpu().numpy()
        mano_states_np_data = self.mano_robot_states.weight.data.detach().cpu().numpy()
        
        if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
            object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
            object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
        
        
        ### scaling constaints ###
        ''' Scaling constants ''' ## scaling constants ##
        self.mano_mult_const_after_cent = 0.9507
        
        if 'model.mano_mult_const_after_cent' in self.conf:
            self.mano_mult_const_after_cent = self.conf['model.mano_mult_const_after_cent']
        
        
        mano_to_dyn_corr_pts_idxes_fn = "./assets/nearest_dyn_verts_idxes.npy"
        self.mano_to_dyn_corr_pts_idxes = np.load(mano_to_dyn_corr_pts_idxes_fn, allow_pickle=True)
        self.mano_to_dyn_corr_pts_idxes = torch.from_numpy(self.mano_to_dyn_corr_pts_idxes).long().cuda() 
        
        print(f"mano_to_dyn_corr_pts_idxes: {self.mano_to_dyn_corr_pts_idxes.size()}")
        

        self.nn_ts = self.nn_timesteps
        
        
        
        ''' Set actions for the redmax simulation and add parameters to params-to-train '''
        
        if self.optimize_rules: 
            params_to_train = []
            params_to_train += list(self.other_bending_network.parameters())
        else:
            params_to_train = []
            params_to_train += list(self.mano_robot_actions.parameters())
            params_to_train += list(self.mano_robot_glb_rotation.parameters())
            params_to_train += list(self.mano_robot_glb_trans.parameters())
            params_to_train += list(self.mano_robot_states.parameters())
        
        # params_to_train = []
        # params_to_train += list(self.redmax_robot_actions.parameters())
        ### construct optimizer ###
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        # init_rot = R.random().as_quat()
        # init_rot = torch.from_numpy(init_rot).float().cuda()
        # self.robot_glb_rotation.weight.data[0, :] = init_rot[:]
        
        
        
        self.timestep_to_active_mesh = {}
        # ref_expanded_visual_pts, minn_idx_expanded_visual_pts_to_link_pts #
        # minn_idx_expanded_visual_pts_to_link_pts #
        self.timestep_to_expanded_visual_pts = {}
        self.timestep_to_active_mesh_opt_ours_sim = {}
        self.timestep_to_active_mesh_w_delta_states = {}
        
          
        self.iter_step = 0


        self.minn_tracking_loss = 1e27
        
        for i_iter in tqdm(range(1000)):
            tot_losses = []
            tot_tracking_loss = []
            
            # timestep #
            # self.timestep_to_active_mesh = {} #
            self.timestep_to_posed_active_mesh = {}
            self.timestep_to_posed_mano_active_mesh = {}
            self.timestep_to_mano_active_mesh = {}
            self.timestep_to_corr_mano_pts = {}
            
            timestep_to_tot_rot = {}
            timestep_to_tot_trans = {}
            
            # correspondence_pts_idxes = None
            # # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces # timestep #
            self.timestep_to_raw_active_meshes = {}
            self.timestep_to_penetration_points = {}
            self.timestep_to_penetration_points_forces = {}
            self.joint_name_to_penetration_forces_intermediates = {}
            
            
            self.ts_to_contact_force_d = {}
            self.ts_to_penalty_frictions = {}
            self.ts_to_penalty_disp_pts = {}
            self.ts_to_redmax_states = {}
            # constraitns for states # 
            # with 17 dimensions on the states; [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16] # # with 17 dimensions # 
            
            contact_pairs_set = None
            self.contact_pairs_sets = {}
            
            # redmax_sim.reset(backward_flag = True)
            
            penetration_forces=  None
            sampled_visual_pts_joint_idxes = None
            
            # tot_grad_qs = []
            
            robo_intermediates_states = []
            
            tot_penetration_depth = []
            
            robo_actions_diff_loss = []
            mano_tracking_loss = []
            
            # init global transformations ##
            # cur_ts_redmax_delta_rotations = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_delta_rotations = torch.tensor([0., 0., 0., 0.], dtype=torch.float32).cuda()
            cur_ts_redmax_robot_trans = torch.zeros((3,), dtype=torch.float32).cuda()
            
            # for cur_ts in range(self.nn_ts):
            for cur_ts in range(self.nn_ts * self.mano_nn_substeps - 1):
                # tot_redmax_actions = []
                
                # actions = {}
                
                self.free_def_bending_weight = 0.0

                # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_delta_states #
                cur_glb_rot = self.mano_robot_glb_rotation(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_rot = cur_glb_rot + cur_ts_redmax_delta_rotations
                
                cur_glb_rot = cur_glb_rot / torch.clamp(torch.norm(cur_glb_rot, dim=-1, p=2), min=1e-7)
                # cur_glb_rot_quat = cur_glb_rot.clone() #
                
                cur_glb_rot = dyn_model_act.quaternion_to_matrix(cur_glb_rot) # mano glboal rotations #
                cur_glb_trans = self.mano_robot_glb_trans(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                
                cur_glb_trans = cur_glb_trans + cur_ts_redmax_robot_trans

                if self.optimize_dyn_actions:
                    ''' Articulated joint forces-driven mano robot '''
                    ### current -> no penetration setting; no contact setting ###
                    link_cur_actions = self.mano_robot_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.mano_agent.set_actions_and_update_states_v2( link_cur_actions, cur_ts, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes) ## 
                else: 
                    ''' Articulated states-driven mano robot '''
                    link_cur_states = self.mano_robot_states(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                    self.mano_agent.set_init_states_target_value(link_cur_states)
                
                # optimize_dyn_actions
                
                cur_visual_pts, visual_pts_joint_idxes = self.mano_agent.get_init_state_visual_pts(ret_joint_idxes=True) 


                cur_visual_pts = cur_visual_pts * self.mano_mult_const_after_cent
                
                
                cur_rot = cur_glb_rot
                cur_trans = cur_glb_trans
                
                timestep_to_tot_rot[cur_ts] = cur_rot.detach()
                timestep_to_tot_trans[cur_ts] = cur_trans.detach()
                
                ## 
                ### transform by the glboal transformation and the translation ###
                cur_visual_pts = torch.matmul(cur_rot, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) ## visual pts ##
                
                
                
                self.timestep_to_active_mesh[cur_ts] = cur_visual_pts
                self.timestep_to_raw_active_meshes[cur_ts] = cur_visual_pts.detach().cpu().numpy()
                
                # 0.1 ## as the initial one #
                ''' cache contact pair set for exporting contact information '''
                if contact_pairs_set is None:
                    self.contact_pairs_set = None
                else:
                    self.contact_pairs_set = contact_pairs_set.copy()
                
                # ### if traiing the jrbpt h ## act wiht reacts ##
                # print(self.timestep_to_active_mesh[cur_ts].size(), cur_visual_pts_friction_forces.size())
                ### get the jactive mesh and remember the active mesh ###
                contact_pairs_set = self.other_bending_network.forward2( input_pts_ts=cur_ts, timestep_to_active_mesh=self.timestep_to_active_mesh, timestep_to_passive_mesh=self.timestep_to_passive_mesh, timestep_to_passive_mesh_normals=self.timestep_to_passive_mesh_normals, friction_forces=None, sampled_verts_idxes=None, reference_mano_pts=None, fix_obj=self.fix_obj, contact_pairs_set=contact_pairs_set)

                ### train with force to active ##
                # if self.train_with_forces_to_active and (not self.use_mano_inputs):
                # penetration_forces #
                if torch.sum(self.other_bending_network.penetrating_indicator.float()) > 0.5:
                    net_penetrating_forces = self.other_bending_network.penetrating_forces
                    net_penetrating_points = self.other_bending_network.penetrating_points

                    
                    # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
                    self.timestep_to_penetration_points[cur_ts] = net_penetrating_points.detach().cpu().numpy()
                    self.timestep_to_penetration_points_forces[cur_ts] = net_penetrating_forces.detach().cpu().numpy()
                    
                    
                    ### transform the visual pts ###
                    # cur_visual_pts = (cur_visual_pts - self.minn_robo_pts) / self.extent_robo_pts
                    # cur_visual_pts = cur_visual_pts * 2. - 1.
                    # cur_visual_pts = cur_visual_pts * self.mult_const_after_cent # mult_const #
                    
                    # sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[finger_sampled_idxes][self.other_bending_network.penetrating_indicator]
                    
                    sampled_visual_pts_joint_idxes = visual_pts_joint_idxes[self.other_bending_network.penetrating_indicator]
                    
                    ## from net penetration forces to to the 
                        
                    ### get the passvie force for each point ## ## 
                    # self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone() ## for
                    
                    net_penetrating_forces = torch.matmul(
                        cur_rot.transpose(1, 0), net_penetrating_forces.transpose(1, 0)
                    ).transpose(1, 0)
                    # net_penetrating_forces = net_penetrating_forces / self.mult_const_after_cent
                    # net_penetrating_forces = net_penetrating_forces / 2
                    # net_penetrating_forces = net_penetrating_forces * self.extent_robo_pts
                    
                    # net_penetrating_forces = (1.0 - pointset_expansion_alpha) * net_penetrating_forces
                    
                    net_penetrating_points = torch.matmul(
                        cur_rot.transpose(1, 0), (net_penetrating_points - cur_trans.unsqueeze(0)).transpose(1, 0)
                    ).transpose(1, 0)
                    # net_penetrating_points = net_penetrating_points / self.mult_const_after_cent
                    # net_penetrating_points = (net_penetrating_points + 1.) / 2. # penetrating points #
                    # net_penetrating_points = (net_penetrating_points * self.extent_robo_pts) + self.minn_robo_pts
                    
                    penetration_forces = net_penetrating_forces ## get the penetration forces and net penetration forces ##
                    # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda()
                    
                    # penetration_forces_values = penetration_forces['penetration_forces'].detach()
                    # penetration_forces_points = penetration_forces['penetration_forces_points'].detach()
                    penetration_forces = {
                        'penetration_forces': penetration_forces,
                        'penetration_forces_points': net_penetrating_points
                    }
                    
                else:
                    penetration_forces = None
                    sampled_visual_pts_joint_idxes = None
                    # penetration_forces = {
                    #     'penetration_forces': penetration_forces, 
                    #     'penetration_forces_points': None, 
                    # }
                    # link_maximal_contact_forces = torch.zeros((redmax_ndof_r, 6), dtype=torch.float32).cuda() ## pointset ##
                    ''' the bending network still have this property and we can get force values here for the expanded visual points '''
                    # self.timestep_to_actuator_points_passive_forces[cur_ts] = self.other_bending_network.penetrating_forces_allpts.detach().clone()  
                        
                # if contact_pairs_set is not None:
                #     self.contact_pairs_sets[cur_ts] = contact_pairs_set.copy()
                
                # # contact force d ## ts to the passive normals ## 
                # self.ts_to_contact_passive_normals[cur_ts] = self.other_bending_network.tot_contact_passive_normals.detach().cpu().numpy()
                # self.ts_to_passive_pts[cur_ts] = self.other_bending_network.cur_passive_obj_verts.detach().cpu().numpy()
                # self.ts_to_passive_normals[cur_ts] = self.other_bending_network.cur_passive_obj_ns.detach().cpu().numpy()
                # self.ts_to_contact_force_d[cur_ts] = self.other_bending_network.contact_force_d.detach().cpu().numpy()
                # self.ts_to_penalty_frictions[cur_ts] = self.other_bending_network.penalty_friction_tangential_forces.detach().cpu().numpy()
                # if self.other_bending_network.penalty_based_friction_forces is not None:
                #     self.ts_to_penalty_disp_pts[cur_ts] = self.other_bending_network.penalty_based_friction_forces.detach().cpu().numpy()
                
                
                
                # if self.optimize_with_intermediates:
                #     tracking_loss = self.compute_loss_optimized_transformations(cur_ts + 1) # 
                # else:
                #     tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()
                    
                
                # # cur_ts % mano_nn_substeps == 0: # 
                if (cur_ts + 1) % mano_nn_substeps == 0:
                    cur_passive_big_ts = cur_ts // mano_nn_substeps
                    ## compute optimized transformations ##
                    tracking_loss = self.compute_loss_optimized_transformations_v2(cur_ts + 1, cur_passive_big_ts + 1)
                    tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                else:
                    tracking_loss = torch.zeros((1,), dtype=torch.float32).cuda().mean()

                # # hand_tracking_loss = torch.sum( ## delta states? ##
                # #     (self.timestep_to_active_mesh_w_delta_states[cur_ts] - cur_visual_pts) ** 2, dim=-1
                # # )
                # # hand_tracking_loss = hand_tracking_loss.mean()
                
                
                # # loss = tracking_loss + self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                # # diff_redmax_visual_pts_with_ori_visual_pts.backward()
                penetraton_penalty = self.other_bending_network.penetrating_depth_penalty * self.penetrating_depth_penalty_coef
                
                tot_penetration_depth.append(penetraton_penalty.detach().item())
                
                # smaller_than_zero_level_set_indicator
                # cur_interpenetration_nns = self.other_bending_network.smaller_than_zero_level_set_indicator.float().sum()
                
                # tot_interpenetration_nns.append(cur_interpenetration_nns)
                # 
                # diff_hand_tracking = torch.zeros((1,), dtype=torch.float32).cuda().mean() ## 
                
                
                # # kinematics_proj_loss = kinematics_trans_diff + penetraton_penalty + diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss
                
                # # if self.use_mano_hand_for_test: ## only the kinematics mano hand is optimized here ##
                # #     kinematics_proj_loss = tracking_loss
                
                # # kinematics_proj_loss = hand_tracking_loss * 1e2 ## 1e2 and the 1e2 ## 
                
                # kinematics_proj_loss = diff_hand_tracking * self.diff_hand_tracking_coef + tracking_loss + penetraton_penalty
                
                # kinematics_proj_loss = loss_finger_tracking # + tracking_loss + penetraton_penalty
                
                # reg_delta_offset_loss = torch.sum(
                #     (mano_expanded_actuator_delta_offset_ori - self.mano_expanded_actuator_delta_offset.weight.data) ** 2, dim=-1
                # )
                # reg_delta_offset_loss = reg_delta_offset_loss.mean()
                # motion_reg_loss_coef
                # reg_delta_offset_loss = reg_delta_offset_loss * self.motion_reg_loss_coef
                
                
                ### tracking loss and the penetration penalty ###
                # kinematics_proj_loss = tracking_loss + penetraton_penalty + reg_delta_offset_loss
                
                cur_kine_rhand_verts = self.rhand_verts[cur_ts // mano_nn_substeps]
                cur_dyn_visual_pts_to_mano_verts = cur_visual_pts[self.mano_to_dyn_corr_pts_idxes]
                diff_hand_tracking = torch.mean(
                    torch.sum((cur_kine_rhand_verts - cur_dyn_visual_pts_to_mano_verts) ** 2, dim=-1)
                )
                
                
                kinematics_proj_loss = tracking_loss + diff_hand_tracking
                ### kinematics proj loss ###
                loss = kinematics_proj_loss # * self.loss_scale_coef


                
                # self.kines_optimizer.zero_grad()
                
                # try:
                #     kinematics_proj_loss.backward(retain_graph=True)
                    
                #     self.kines_optimizer.step()
                # except:
                #     pass
                
                
                
                
                
                self.optimizer.zero_grad()
                # loss = diff_hand_tracking
                loss.backward(retain_graph=True) # 
                self.optimizer.step() ## update the weights ##
                 
                # diff_hand_tracking # diff hand ## 
                mano_tracking_loss.append(diff_hand_tracking.detach().cpu().item())
                tot_tracking_loss.append(tracking_loss.detach().cpu().item())
                
                
                tot_losses.append(loss.detach().item())

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)

                self.iter_step += 1
                self.update_learning_rate() ## update learning rate ##
                
                torch.cuda.empty_cache()
                
            
            ''' Get nn_forward_ts and backward through the actions for updating '''
            
            ## save checkpoint ## 
            
            if (i_iter % self.ckpt_sv_freq) == 0:
                self.save_checkpoint()
                
            
            tot_losses = sum(tot_losses) / float(len(tot_losses))
            # tot_penalty_dot_forces_normals = sum(tot_penalty_dot_forces_normals) / float(len(tot_penalty_dot_forces_normals))
            # tot_penalty_friction_constraint = sum(tot_penalty_friction_constraint) / float(len(tot_penalty_friction_constraint))
            tot_tracking_loss = sum(tot_tracking_loss) / float(len(tot_tracking_loss))
            # tot_penetration_depth = sum(tot_penetration_depth) / float(len(tot_penetration_depth))
            # robo_actions_diff_loss = sum(robo_actions_diff_loss) / float(len(robo_actions_diff_loss))
            mano_tracking_loss = sum(mano_tracking_loss) / float(len(mano_tracking_loss))

            # if tot_losses < self.minn_tracking_loss:
            #     self.minn_tracking_loss = tot_losses
            #     self.save_checkpoint(tag="best")
            
            if i_iter % self.report_freq == 0:
                logs_sv_fn = os.path.join(self.base_exp_dir, 'log.txt')
                
                cur_log_sv_str = 'iter:{:8>d} loss = {} mano_tracking_loss = {} tot_tracking_loss = {} lr={}'.format(self.iter_step, tot_losses, mano_tracking_loss, tot_tracking_loss, self.optimizer.param_groups[0]['lr'])
                
                print(cur_log_sv_str)
                ''' Dump to the file '''
                with open(logs_sv_fn, 'a') as log_file:
                    log_file.write(cur_log_sv_str + '\n')

            # self.validate_mesh_robo_a()
            if i_iter % self.val_mesh_freq == 0:
                self.validate_mesh_robo()
                # self.validate_mesh_robo_e() # validate meshes #
  
            torch.cuda.empty_cache()

    
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0 # cos anneal ratio #
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end: # warm up end and the w
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        ## g in self.
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    ## backup files ##
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        for i_obj in range(len(self.sdf_network)):
            self.sdf_network[i_obj].load_state_dict(checkpoint['sdf_network_fine'][i_obj])
            self.bending_network[i_obj].load_state_dict(checkpoint['bending_network_fine'][i_obj])
        # self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')


    def load_checkpoint_via_fn(self, checkpoint_fn):

        checkpoint = torch.load(checkpoint_fn, map_location=self.device, )
        
        self.other_bending_network.load_state_dict(checkpoint['dyn_model'], strict=False)

        logging.info(f"checkpoint with sdf_net and bending_net loaded from {checkpoint_fn}")

        logging.info('End')
        
    def load_checkpoint_prev_delta(self, delta_mesh_checkpoint_fn):
        delta_mesh_checkpoint = torch.load(delta_mesh_checkpoint_fn, map_location=self.device)
        self.prev_sdf_network.load_state_dict(delta_mesh_checkpoint['sdf_network_fine'])
        logging.info(f"delta_mesh checkpoint loaded from {delta_mesh_checkpoint_fn}")
    
    def save_checkpoint_delta_states(self, ):
        checkpoint = {
            'robot_delta_states': self.robot_delta_states.state_dict()
        }
        ckpt_sv_root_folder = os.path.join(self.base_exp_dir, 'checkpoints')
        os.makedirs(ckpt_sv_root_folder, exist_ok=True)
        ckpt_sv_fn = os.path.join(ckpt_sv_root_folder, 'robo_delta_states_ckpt_{:0>6d}.pth'.format(self.iter_step))
        
        torch.save(checkpoint, ckpt_sv_fn)
    
    
    def save_checkpoint_redmax_robot_actions(self, ):
        checkpoint = {
            'redmax_robot_actions': self.redmax_robot_actions.state_dict()
        }
        ckpt_sv_root_folder = os.path.join(self.base_exp_dir, 'checkpoints')
        os.makedirs(ckpt_sv_root_folder, exist_ok=True)
        ckpt_sv_fn = os.path.join(ckpt_sv_root_folder, 'redmax_robot_actions_ckpt_{:0>6d}.pth'.format(self.iter_step))
        
        torch.save(checkpoint, ckpt_sv_fn)


    def save_checkpoint(self, tag="", niter=False):

        checkpoint = {
            'dyn_model': self.other_bending_network.state_dict()
        }


        if self.mode in ['train_actions_from_model_rules', 'train_mano_actions_from_model_rules', 'train_actions_from_mano_model_rules', 'train_real_robot_actions_from_mano_model_rules', 'train_real_robot_actions_from_mano_model_rules_diffhand', 'train_real_robot_actions_from_mano_model_rules_diffhand_fortest', 'train_sparse_retar', 'train_real_robot_actions_from_mano_model_rules_manohand_fortest', 'train_real_robot_actions_from_mano_model_rules_manohand_fortest_states', 'train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_res_world', 'train_dyn_mano_model', 'train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_grab', 'train_point_set', 'train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab', 'train_point_set_retar', "train_point_set_retar_pts", "train_finger_kinematics_retargeting_arctic_twohands", "train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_arctic_twohands", "train_redmax_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab", "train_dyn_mano_model_wreact"]:
            try:    
                checkpoint['robot_actions'] = self.robot_actions.state_dict()    
            except:
                pass
            
            try:    
                checkpoint['robot_actions_lft'] = self.robot_actions_lft.state_dict()    
            except:
                pass
            try:
                checkpoint['robot_init_states'] = self.robot_init_states.state_dict()    
            except:
                pass
            try:
                checkpoint['robot_glb_rotation'] = self.robot_glb_rotation.state_dict()    
            except:
                pass
            
            try:
                checkpoint['robot_glb_rotation_lft'] = self.robot_glb_rotation_lft.state_dict()    
            except:
                pass
            
            try:
                checkpoint['robot_actuator_friction_forces'] = self.robot_actuator_friction_forces.state_dict()    
            except:
                pass
            
            try:
                checkpoint['robot_actuator_friction_forces'] = self.robot_actuator_friction_forces.state_dict() 
            except:
                pass
            
            try:
                checkpoint['robot_glb_trans'] = self.robot_glb_trans.state_dict() 
            except:
                pass
            # try:
            #     checkpoint['robot_delta_states'] = self.robot_delta_states.state_dict()
            # except:
            #     pass
            
            try:
                checkpoint['robot_delta_angles'] = self.robot_delta_angles.state_dict()
            except:
                pass
            
            try:
                checkpoint['robot_delta_trans'] = self.robot_delta_trans.state_dict()
            except:
                pass
            
            # try:
            #     checkpoint['robot_glb_trans_lft'] = self.robot_glb_trans_lft.state_dict() 
            # except:
            #     pass
            
            try:
                checkpoint['robot_glb_trans_lft'] = self.robot_glb_trans_lft.state_dict() 
            except:
                pass
            try:
                checkpoint['robot_delta_states'] = self.robot_delta_states.state_dict()
            except:
                pass
            
            
            if self.mode in ['train_actions_from_mano_model_rules']:
                
                checkpoint['expanded_actuator_friction_forces'] = self.expanded_actuator_friction_forces.state_dict() 
            # robot_delta_states 
            
            try:
                checkpoint['expanded_actuator_delta_offset'] = self.expanded_actuator_delta_offset.state_dict()
            except:
                pass
            
            # mano_expanded_actuator_pointact_forces
            # mano_expanded_actuator_friction_forces, mano_expanded_actuator_delta_offset # 
            try:
                checkpoint['mano_expanded_actuator_friction_forces'] = self.mano_expanded_actuator_friction_forces.state_dict()
            except:
                pass
            
            try:
                checkpoint['mano_expanded_actuator_pointact_forces'] = self.mano_expanded_actuator_pointact_forces.state_dict()
            except:
                pass
                
            
            try:
                checkpoint['mano_expanded_actuator_delta_offset'] = self.mano_expanded_actuator_delta_offset.state_dict()
            except:
                pass
            
            # mano_expanded_actuator_delta_offset_nex
            try:
                checkpoint['mano_expanded_actuator_delta_offset_nex'] = self.mano_expanded_actuator_delta_offset_nex.state_dict()
            except:
                pass
            
            # expanded_actuator_pointact_forces
            try:
                checkpoint['expanded_actuator_pointact_forces'] = self.expanded_actuator_pointact_forces.state_dict()
            except:
                pass
            
            try:
                checkpoint['expanded_actuator_delta_offset'] = self.expanded_actuator_delta_offset.state_dict()
            except:
                pass
            
            # mano_robot_glb_rotation, mano_robot_glb_trans, mano_robot_init_states, mano_robot_delta_states #
            try:
                checkpoint['mano_robot_glb_rotation'] = self.mano_robot_glb_rotation.state_dict()
            except:
                pass
            
            try:
                checkpoint['mano_robot_glb_trans'] = self.mano_robot_glb_trans.state_dict()
            except:
                pass
            
            try:
                checkpoint['mano_robot_init_states'] = self.mano_robot_init_states.state_dict()
            except:
                pass
            
            try:
                checkpoint['mano_robot_delta_states'] = self.mano_robot_delta_states.state_dict()
            except:
                pass
        
            try:
                checkpoint['mano_robot_states'] = self.mano_robot_states.state_dict()
            except:
                pass
            
            try:
                checkpoint['mano_robot_actions'] = self.mano_robot_actions.state_dict()
            except:
                pass
            
            try:
                checkpoint['redmax_robot_actions'] = self.redmax_robot_actions.state_dict()
            except:
                pass
            
            # residual_controller, residual_dynamics_model # 
            try:
                checkpoint['residual_controller'] = self.residual_controller.state_dict()
            except:
                pass
            
            try:
                checkpoint['residual_dynamics_model'] = self.residual_dynamics_model.state_dict()
            except:
                pass
            
            # robot_states
            try:
                checkpoint['robot_states'] = self.robot_states.state_dict()
            except:
                pass
            
            try:
                checkpoint['robot_states_sv_from_act'] = self.robot_states_sv
            except:
                pass
        
            try:
                checkpoint['robot_states_lft'] = self.robot_states_lft.state_dict()
            except:
                pass
            
            try:
                checkpoint['robot_delta_states_lft'] = self.robot_delta_states_lft.state_dict()
            except:
                pass
                
            # robot_delta_glb_trans, robot_delta_glb_trans_lft
            try:
                checkpoint['robot_delta_glb_trans'] = self.robot_delta_glb_trans.state_dict()
            except:
                pass
        
            try:
                checkpoint['robot_delta_glb_trans_lft'] = self.robot_delta_glb_trans_lft.state_dict()
            except:
                pass
            
            # # object_global_orient, object_transl #
            try:
                checkpoint['object_transl'] = self.object_transl
                checkpoint['object_global_orient'] = self.object_global_orient
            except:
                pass
            
            try:
                # optimized_quat
                cur_optimized_quat = np.stack(self.optimized_quat, axis=0)
                cur_optimized_trans = np.stack(self.optimized_trans, axis=0)
                checkpoint['optimized_quat'] = cur_optimized_quat
                checkpoint['optimized_trans'] = cur_optimized_trans
            except:
                pass
                
            
        
        print(f"Saving checkpoint with keys {checkpoint.keys()}")
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        
        ckpt_sv_fn = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step))
        if len(tag) > 0:
            if tag == "best": ## 
                ckpt_sv_fn = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{}.pth'.format(tag))
            else:
                if niter: ### 
                    ckpt_sv_fn = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{}.pth'.format(tag))
                else:
                    # ckpt_sv_fn = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{}_{}.pth'.format(self.iter_step, tag))
                    ckpt_sv_fn = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{}_{}.pth'.format(self.iter_step, tag))
        
        torch.save(checkpoint, ckpt_sv_fn)
        
        
        
        
        bending_net_save_values = self.other_bending_network.save_values
        os.makedirs(os.path.join(self.base_exp_dir, 'miscs'), exist_ok=True)
        bending_net_save_values_sv_fn = os.path.join(self.base_exp_dir, 'miscs', 'bending_net_save_values_{:0>6d}.npy'.format(self.iter_step))
        np.save(bending_net_save_values_sv_fn, bending_net_save_values) 
        return ckpt_sv_fn



    def validate_mesh_robo(self, ):
        ## merge meshes ##
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # 
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        # self.hand_faces, self.obj_faces # # hand faces #
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        hand_faces_np = self.robo_hand_faces.detach().cpu().numpy()
        
        if self.use_mano_inputs:
            hand_faces_np = mano_hand_faces_np
            
        if self.optim_sim_model_params_from_mano:
            hand_faces_np = mano_hand_faces_np
        
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        
        
        if self.other_bending_network.canon_passive_obj_verts is None:
            init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
            init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        else:
            init_passive_obj_verts = self.other_bending_network.canon_passive_obj_verts.detach().cpu().numpy()
            init_passive_obj_verts_center = torch.zeros((3, )).cuda().detach().cpu().numpy()
        
        # init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        # init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        ts_to_hand_obj_verts = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0, (self.nn_ts - 1) * self.mano_nn_substeps, 1):
            if self.optim_sim_model_params_from_mano:
                cur_hand_mesh = self.rhand_verts[i_ts] # .detach().cpu().numpy()
            else:
                cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            
            if i_ts % self.mano_nn_substeps == 0:
                cur_mano_rhand = self.rhand_verts[i_ts // self.mano_nn_substeps].detach().cpu().numpy()
            cur_rhand_verts = cur_hand_mesh
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx: # to optimizable rot mtx #
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))
            
            ## the training 
            if self.mode in ['train_sparse_retar']:
                cur_transformed_obj = self.obj_pcs[i_ts].detach().cpu().numpy()

            if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
                ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
                ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## # 
                ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
                ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            else:
                ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
                ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            ts_to_hand_obj_verts[i_ts] = (cur_rhand_verts_np, cur_transformed_obj) # not correct.... #
            # merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            if i_ts % 10 == 0:
                # print(f"exporting meshes i_ts: {i_ts}, cur_hand_verts_np: {cur_rhand_verts_np.shape}, hand_faces_np: {hand_faces_np.shape}")
                merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj], [hand_faces_np, obj_faces_np])
                maxx_hand_faces, minn_hand_faces = np.max(hand_faces_np), np.min(hand_faces_np)
                maxx_obj_faces, minn_obj_faces = np.max(obj_faces_np), np.min(obj_faces_np)
                
                # print(f"cur_rhand_verts_np: {cur_rhand_verts_np.shape}, cur_transformed_obj: {cur_transformed_obj.shape}, maxx_hand_faces: {maxx_hand_faces}, minn_hand_faces: {minn_hand_faces}, maxx_obj_faces: {maxx_obj_faces}, minn_obj_faces: {minn_obj_faces}")
                mesh = trimesh.Trimesh(merged_verts, merged_faces)
                mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
                mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                if i_ts % self.mano_nn_substeps == 0:
                    ### overlayed with the mano mesh ###
                    merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
                    mesh = trimesh.Trimesh(merged_verts, merged_faces)
                    mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_wmano.ply'.format(self.iter_step, i_ts)
                    mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
            'ts_to_hand_obj_verts': ts_to_hand_obj_verts
        }
        
        if self.mode not in ['train_sparse_retar']:
            hand_obj_verts_faces_sv_dict = {
                'ts_to_hand_obj_verts': ts_to_hand_obj_verts,
                'hand_faces': hand_faces_np,
                'obj_faces': obj_faces_np
            }
            
            hand_obj_verts_faces_sv_dict_sv_fn = 'hand_obj_verts_faces_sv_dict_{:0>8d}.npy'.format(self.iter_step)
            hand_obj_verts_faces_sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, hand_obj_verts_faces_sv_dict_sv_fn)
            np.save(hand_obj_verts_faces_sv_dict_sv_fn, hand_obj_verts_faces_sv_dict)
            # collision detection and # hand obj verts faces sv dict #
        try:
            timestep_to_mano_active_mesh = {ts: self.timestep_to_mano_active_mesh[ts].detach().cpu().numpy() for ts in self.timestep_to_mano_active_mesh}
            mano_sv_dict_sv_fn = 'mano_act_pts_{:0>8d}.npy'.format(self.iter_step)
            mano_sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, mano_sv_dict_sv_fn)
            np.save(mano_sv_dict_sv_fn, timestep_to_mano_active_mesh)
            
            timestep_to_robot_mesh = {ts: self.timestep_to_active_mesh[ts].detach().cpu().numpy() for ts in self.timestep_to_active_mesh}
            rhand_verts = self.rhand_verts.detach().cpu().numpy()
            #  ts_to_robot_fingers, ts_to_mano_fingers
            retar_sv_dict = {
                'timestep_to_robot_mesh': timestep_to_robot_mesh,
                'rhand_verts': rhand_verts,
                'ts_to_robot_fingers': self.ts_to_robot_fingers, 
                'ts_to_mano_fingers': self.ts_to_mano_fingers,
            }
            retar_sv_fn =  'retar_info_dict_{:0>8d}.npy'.format(self.iter_step)
            retar_sv_fn = os.path.join(mesh_sv_root_dir, retar_sv_fn)
            np.save(retar_sv_fn, retar_sv_dict)

            timestep_to_corr_mano_pts = {ts: self.timestep_to_corr_mano_pts[ts].detach().cpu().numpy() for ts in self.timestep_to_corr_mano_pts}
            mano_sv_dict_sv_fn = 'corr_mano_act_pts_{:0>8d}.npy'.format(self.iter_step)
            mano_sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, mano_sv_dict_sv_fn)
            np.save(mano_sv_dict_sv_fn, timestep_to_corr_mano_pts)
        except:
            pass
        sv_dict_sv_fn = '{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
        
        try:
            robo_intermediates_states_np = self.robo_intermediates_states.numpy()
            robo_intermediates_states_sv_fn = f"robo_intermediates_states_{self.iter_step}.npy"
            robo_intermediates_states_sv_fn = os.path.join(mesh_sv_root_dir, robo_intermediates_states_sv_fn)
            np.save(robo_intermediates_states_sv_fn, robo_intermediates_states_np) ## robot 
        except:
            pass
    
    def validate_mesh_robo_redmax_acts(self, i_iter, tag=None):
        optimized_redmax_meshes = {
            ts: self.ts_to_act_opt_pts_woglb[ts].detach().cpu().numpy() for ts in self.ts_to_act_opt_pts_woglb
        }
        states_optimized_meshes = {
            ts: self.timestep_to_active_mesh_wo_glb_from_states[ts].detach().cpu().numpy() for ts in self.timestep_to_active_mesh_wo_glb_from_states
        }
        act_optimized_sv_dict = {
            'optimized_redmax_meshes': optimized_redmax_meshes,
            'states_optimized_meshes': states_optimized_meshes,
        }
        
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        if (tag is None) or (len(tag) == 0):
            act_optimized_sv_dict_fn = f"act_optimized_sv_dict_{i_iter}.npy"
        else:
            act_optimized_sv_dict_fn = f"act_optimized_sv_dict_{i_iter}_tag_{tag}.npy"
        act_optimized_sv_dict_fn = os.path.join(mesh_sv_root_dir, act_optimized_sv_dict_fn)
        np.save(act_optimized_sv_dict_fn, act_optimized_sv_dict) #
        print(f"Redmax acts optimized info saved to {act_optimized_sv_dict_fn}")
    
    
    def validate_mesh_robo_d(self, ):
        init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        
        ts_to_active_mesh = {
            ts: self.ts_to_target_sim_active_meshes[ts].detach().cpu().numpy() for ts in self.ts_to_target_sim_active_meshes
        }
        ts_to_passive_mesh = {}
        for ts in self.ts_to_target_sim_active_meshes:
            cur_pred_quat = self.ts_to_target_sim_obj_quat[ts]
            cur_pred_trans = self.ts_to_target_sim_obj_trans[ts].detach().cpu().numpy()
            
            cur_pred_rot_mtx = fields.quaternion_to_matrix(cur_pred_quat).detach().cpu().numpy()
            
            ## cur pred trnas ##
            # cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))
            ts_to_passive_mesh[ts] = cur_transformed_obj
            
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        hand_faces_np = self.robo_hand_faces.detach().cpu().numpy() # # 
        
        # obj faces; hand faces; ts to active meshes; ts to passive meshes #
        
        sv_dict = {
            'ts_to_active_mesh': ts_to_active_mesh, 'ts_to_passive_mesh': ts_to_passive_mesh, 'obj_faces': obj_faces_np, 'hand_faces': hand_faces_np
        }
        
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_active_mesh_sv_fn = '{:0>8d}_ts_to_target_sim_active_mesh.npy'.format(self.iter_step)
        ts_to_active_mesh_sv_fn = os.path.join(mesh_sv_root_dir, ts_to_active_mesh_sv_fn)
        np.save(ts_to_active_mesh_sv_fn, sv_dict)
        

    def validate_mesh_robo_a(self, ):
        ts_to_active_mesh = {
            ts: self.timestep_to_active_mesh[ts][self.sampled_verts_idxes].detach().cpu().numpy() for ts in self.timestep_to_active_mesh
        }
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_active_mesh_sv_fn = '{:0>8d}_ts_to_active_mesh.ply'.format(self.iter_step)
        ts_to_active_mesh_sv_fn = os.path.join(mesh_sv_root_dir, ts_to_active_mesh_sv_fn)
        np.save(ts_to_active_mesh_sv_fn, ts_to_active_mesh)
        
    def validate_mesh_robo_b(self, ):
        ts_to_active_mesh = {
            ts: self.cur_ts_to_optimized_visual_pts[ts] for ts in self.cur_ts_to_optimized_visual_pts
        }
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_active_mesh_sv_fn = '{:0>8d}_ts_to_opt_act_pts.ply'.format(self.iter_step)
        ts_to_active_mesh_sv_fn = os.path.join(mesh_sv_root_dir, ts_to_active_mesh_sv_fn)
        np.save(ts_to_active_mesh_sv_fn, ts_to_active_mesh)
    
    
    def save_redmax_actions(self, ):
        
        # tot_redmax_actions
        redmax_act_sv_folder = os.path.join(self.base_exp_dir, 'checkpoint')
        os.makedirs(redmax_act_sv_folder, exist_ok=True)
        redmax_act_sv_fn = '{:0>8d}_redmax_act.npy'.format(self.iter_step)
        redmax_act_sv_fn = os.path.join(redmax_act_sv_folder, redmax_act_sv_fn)
        
        
        np.save(redmax_act_sv_fn, self.tot_redmax_actions.detach().cpu().numpy())
        
    
    def validate_mesh_robo_c(self, ):
        ts_to_active_mesh = {
            ts: self.tot_visual_pts[ts] for ts in range(self.tot_visual_pts.shape[0])
        }
        
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_active_mesh_sv_fn = '{:0>8d}_ts_to_opt_intermediate_act_pts.npy'.format(self.iter_step)
        ts_to_active_mesh_sv_fn = os.path.join(mesh_sv_root_dir, ts_to_active_mesh_sv_fn)
        np.save(ts_to_active_mesh_sv_fn, ts_to_active_mesh)
        
        ts_to_ref_act_mesh = {
            ts: self.timestep_to_active_mesh_wo_glb[ts].detach().cpu().numpy() for ts in self.timestep_to_active_mesh_wo_glb
        }
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_ref_act_mesh_sv_fn = '{:0>8d}_ts_to_ref_act_pts.npy'.format(self.iter_step)
        ts_to_ref_act_mesh_sv_fn = os.path.join(mesh_sv_root_dir, ts_to_ref_act_mesh_sv_fn)
        np.save(ts_to_ref_act_mesh_sv_fn, ts_to_ref_act_mesh)
        

    
    
    def validate_mesh_robo_e(self, ):
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # 
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        hand_faces_np = self.robo_hand_faces.detach().cpu().numpy()
        
        if self.use_mano_inputs:
            hand_faces_np = mano_hand_faces_np
        
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        
        
        # init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        # init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        
        
        retargeting_info = {}
        ts_to_retargeted_info = {}
        
        for i_ts in range(0, self.nn_ts - 1, 1):
            cur_mano_rhand = self.rhand_verts[i_ts].detach().cpu().numpy()
            cur_hand_verts = self.timestep_to_active_mesh[i_ts].detach().cpu().numpy()
            obj_verts = self.timestep_to_passive_mesh[i_ts].detach().cpu().numpy()
            obj_faces = obj_faces_np
            hand_faces = hand_faces_np
            
            ts_to_retargeted_info[i_ts] = (cur_hand_verts, cur_mano_rhand,  obj_verts)
            
            
            if i_ts % 10 == 0: # merged verts #
                merged_verts, merged_faces = merge_meshes([cur_hand_verts, obj_verts], [hand_faces, obj_faces])
                ## ## merged 
                mesh = trimesh.Trimesh(merged_verts, merged_faces)
                
                mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
                mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                
                merged_verts, merged_faces = merge_meshes([cur_hand_verts, obj_verts, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
                mesh = trimesh.Trimesh(merged_verts, merged_faces)
                mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_wmano.ply'.format(self.iter_step, i_ts)
                mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                cur_hand_mesh = trimesh.Trimesh(cur_hand_verts, hand_faces)
                mesh_sv_fn = 'dyn_mano_hand_{:0>8d}_ts_{:0>3d}.obj'.format(self.iter_step, i_ts)
                cur_hand_mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                cur_obj_mesh = trimesh.Trimesh(obj_verts, obj_faces_np)
                mesh_sv_fn = 'obj_{:0>8d}_ts_{:0>3d}.obj'.format(self.iter_step, i_ts)
                cur_obj_mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                cur_mano_hand_mesh = trimesh.Trimesh(cur_mano_rhand, mano_hand_faces_np)
                mesh_sv_fn = 'kine_mano_hand_{:0>8d}_ts_{:0>3d}.obj'.format(self.iter_step, i_ts)
                cur_mano_hand_mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))

        retargeting_info = {
            'ts_to_retargeted_info': ts_to_retargeted_info,
            'obj_verts': obj_verts,
            'obj_faces': obj_faces_np,
            'hand_faces': hand_faces,
        }
        retargeting_info_sv_fn = 'retargeting_info_{:0>8d}.npy'.format(self.iter_step)
        retargeting_info_sv_fn = os.path.join(mesh_sv_root_dir, retargeting_info_sv_fn)
        np.save(retargeting_info_sv_fn, retargeting_info)
        # save the retargeted info here ##
        
    
    
    def validate_mesh_robo_g(self, tag=None):
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # 
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        # one single hand or # not very easy to 
        # self.hand_faces, self.obj_faces 
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        # hand_faces_np = self.robo_hand_faces.detach().cpu().numpy() # ### robot faces # 
        
        if self.use_mano_inputs: ## mano inputs ## ## validate mesh robo g ##
            hand_faces_np = mano_hand_faces_np
        
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        
        
        if self.other_bending_network.canon_passive_obj_verts is None:
            init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
            init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        else:
            init_passive_obj_verts = self.other_bending_network.canon_passive_obj_verts.detach().cpu().numpy()
            init_passive_obj_verts_center = torch.zeros((3, )).cuda().detach().cpu().numpy()
        
        # init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        # init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        ts_to_hand_obj_verts = {}
        ts_to_transformed_obj_pts = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0, (self.nn_ts - 1) * self.mano_nn_substeps, 1):
            cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            if i_ts % self.mano_nn_substeps == 0:
                cur_mano_rhand = self.rhand_verts[i_ts // self.mano_nn_substeps].detach().cpu().numpy()
            cur_rhand_verts = cur_hand_mesh # [: cur_hand_mesh.size(0) // 2]
            # cur_lhand_verts = cur_hand_mesh # [cur_hand_mesh.size(0) // 2:]
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx: # to optimizable rot mtx #
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
                ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
                ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## # 
                ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
                ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            else:
                ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
                ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            ts_to_hand_obj_verts[i_ts] = (cur_rhand_verts_np, cur_transformed_obj) # not correct.... #
            # merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            # if i_ts % 10 == 0:
            #     # print(f"exporting meshes i_ts: {i_ts}, cur_hand_verts_np: {cur_rhand_verts_np.shape}, hand_faces_np: {hand_faces_np.shape}")
            #     merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj], [hand_faces_np, obj_faces_np])
            #     mesh = trimesh.Trimesh(merged_verts, merged_faces)
            #     mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
            #     mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
            #     if i_ts % self.mano_nn_substeps == 0:
            #         ### overlayed with the mano mesh ###
            #         merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            #         mesh = trimesh.Trimesh(merged_verts, merged_faces)
            #         mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_wmano.ply'.format(self.iter_step, i_ts)
            #         mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
            #### 
            ##
        # try:
        #     timestep_to_anchored_mano_pts = self.timestep_to_anchored_mano_pts
        # except:
        #     timestep_to_anchored_mano_pts = {}
        
        # try:
        #     timestep_to_raw_active_meshes = self.timestep_to_raw_active_meshes
        # except:
        #     timestep_to_raw_active_meshes = {}
        
        # try:
        #     ts_to_dyn_mano_pts = self.ts_to_dyn_mano_pts
        # except:
        #     ts_to_dyn_mano_pts = {}
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
            'ts_to_hand_obj_verts': ts_to_hand_obj_verts, # hand vertices, obj vertices #
            'obj_faces_np': obj_faces_np,
            # 'timestep_to_anchored_mano_pts': timestep_to_anchored_mano_pts, 
            # 'timestep_to_raw_active_meshes': timestep_to_raw_active_meshes, # ts to raw active meshes ##
            # 'ts_to_dyn_mano_pts': ts_to_dyn_mano_pts,
            # 'timestep_to_expanded_robo_visual_pts': { ts: self.timestep_to_expanded_robo_visual_pts[ts].detach().cpu().numpy() for ts in  self.timestep_to_expanded_robo_visual_pts},
        }
        try:
            ts_sv_dict['timestep_to_expanded_robo_visual_pts']  = { ts: self.timestep_to_expanded_robo_visual_pts[ts].detach().cpu().numpy() for ts in  self.timestep_to_expanded_robo_visual_pts},
        except:
            pass
        

        timestep_to_mano_active_mesh = {ts: self.timestep_to_active_mesh[ts].detach().cpu().numpy() for ts in self.timestep_to_active_mesh}
        if tag is not None:
            mano_sv_dict_sv_fn = 'mano_act_pts_{:0>8d}_{}.npy'.format(self.iter_step, tag)
        else:
            mano_sv_dict_sv_fn = 'mano_act_pts_{:0>8d}.npy'.format(self.iter_step)
        mano_sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, mano_sv_dict_sv_fn)
        np.save(mano_sv_dict_sv_fn, timestep_to_mano_active_mesh)
        
        
        if tag is not None:
            sv_dict_sv_fn = 'retar_info_dict_{:0>8d}_{}.npy'.format(self.iter_step, tag)
        else:
            sv_dict_sv_fn = 'retar_info_dict_{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
    
    
        sv_dict_sv_fn_ts_to_hand_obj_verts = 'hand_obj_verts_faces_sv_dict_{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn_ts_to_hand_obj_verts = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn_ts_to_hand_obj_verts)
        np.save(sv_dict_sv_fn_ts_to_hand_obj_verts, ts_to_hand_obj_verts)
    
    def validate_mesh_robo_h(self, ): # validate mesh robo ###
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # 
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        # one single hand or # not very easy to ## no
        # self.hand_faces, self.obj_faces
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        hand_faces_np = self.robo_hand_faces.detach().cpu().numpy() # ### robot faces # 
        
        if self.use_mano_inputs:
            hand_faces_np = mano_hand_faces_np
            
        if self.optim_sim_model_params_from_mano:
            hand_faces_np = mano_hand_faces_np
        else:
            hand_faces_np_left  =  self.robot_agent_left.robot_faces.detach().cpu().numpy()
        
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        
        
        if self.other_bending_network.canon_passive_obj_verts is None:
            init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
            # init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
            init_passive_obj_verts_center = torch.zeros((3, )).cuda().detach().cpu().numpy()
        else:
            init_passive_obj_verts = self.other_bending_network.canon_passive_obj_verts.detach().cpu().numpy()
            init_passive_obj_verts_center = torch.zeros((3, )).cuda().detach().cpu().numpy()
        
        # init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        # init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        ts_to_hand_obj_verts = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0, (self.nn_ts - 1) * self.mano_nn_substeps, 1):
            if self.optim_sim_model_params_from_mano:
                cur_hand_mesh = self.rhand_verts[i_ts] # .detach().cpu().numpy()
            else:
                cur_hand_mesh = self.ts_to_act_rgt[i_ts]
            
            # cur_rhand_mano_gt, cur_lhand_mano_gt, gt_obj_pcs, re_trans_obj_pcs # 
            cur_rhand_mano_gt = self.rhand_verts[i_ts].detach().cpu().numpy()
            cur_lhand_mano_gt = self.lhand_verts[i_ts].detach().cpu().numpy() # 
            gt_obj_pcs = self.timestep_to_passive_mesh[i_ts].detach().cpu().numpy() # 
            re_trans_obj_pcs = self.re_transformed_obj_verts[i_ts].detach().cpu().numpy() # 
            
            # ts_to_mano_rhand_meshes ## why we have zero gradients ## ## why the values cannot be updated? ##
            if self.optim_sim_model_params_from_mano:
                cur_hand_mesh_left = self.lhand_verts[i_ts] ## retargeting ##
                cur_hand_mesh_faces_left = self.hand_faces.detach().cpu().numpy()
            else:
                cur_hand_mesh_left = self.ts_to_act_lft[i_ts]
                cur_hand_mesh_faces_left = self.robot_agent_left.robot_faces.detach().cpu().numpy()
            
            # cur_hand_mesh_left = self.ts_to_act_lft[i_ts]
            cur_hand_mesh_left_np = cur_hand_mesh_left.detach().cpu().numpy()
            
            if i_ts % self.mano_nn_substeps == 0:
                cur_mano_rhand = self.rhand_verts[i_ts // self.mano_nn_substeps].detach().cpu().numpy()
            cur_rhand_verts = cur_hand_mesh # [: cur_hand_mesh.size(0) // 2]
            # cur_lhand_verts = cur_hand_mesh # [cur_hand_mesh.size(0) // 2:]
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx: # to optimizable rot mtx #
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
                ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
                ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## # 
                ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
                ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            else:
                ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
                ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            ts_to_hand_obj_verts[i_ts] = (cur_rhand_verts_np, cur_hand_mesh_left_np, cur_transformed_obj) # not correct.... #
            # cur_transformed_obj = self.timestep_to_passive_mesh[i_ts].detach().cpu().numpy()
            # merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            if i_ts % 10 == 0:
                # print(f"exporting meshes i_ts: {i_ts}, cur_hand_verts_np: {cur_rhand_verts_np.shape}, hand_faces_np: {hand_faces_np.shape}")
                # print(f"cur_rhand_verts_np: {cur_rhand_verts_np.shape}, cur_hand_mesh_left_np: {cur_hand_mesh_left_np.shape}, cur_transformed_obj: {cur_transformed_obj.shape}, hand_faces_np: {hand_faces_np.shape}, obj_faces_np: {obj_faces_np.shape}")
                merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_hand_mesh_left_np, cur_transformed_obj], [hand_faces_np, cur_hand_mesh_faces_left,  obj_faces_np])
                # print(f"merged_verts: {merged_verts.shape}, merged_faces: {merged_faces.shape}")
                mesh = trimesh.Trimesh(merged_verts, merged_faces)
                mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
                mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                
                if i_ts % self.mano_nn_substeps == 0:
                    ### overlayed with the mano mesh ###
                    merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_hand_mesh_left_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, cur_hand_mesh_faces_left, obj_faces_np, mano_hand_faces_np])
                    mesh = trimesh.Trimesh(merged_verts, merged_faces)
                    mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_wmano.ply'.format(self.iter_step, i_ts)
                    mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                    
                    ### overlayed with the mano mesh ###
                    merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_hand_mesh_left_np, cur_mano_rhand], [hand_faces_np, cur_hand_mesh_faces_left, mano_hand_faces_np]) # 
                    mesh = trimesh.Trimesh(merged_verts, merged_faces)
                    mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_onlyhand.ply'.format(self.iter_step, i_ts)
                    mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                    
                    # cur_rhand_mano_gt, cur_lhand_mano_gt, gt_obj_pcs, re_trans_obj_pcs # 
                    merged_verts, merged_faces = merge_meshes([cur_rhand_mano_gt, cur_lhand_mano_gt, gt_obj_pcs], [mano_hand_faces_np, mano_hand_faces_np, obj_faces_np])
                    mesh  = trimesh.Trimesh(merged_verts, merged_faces)
                    mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_gt_mano_obj.ply'.format(self.iter_step, i_ts)
                    mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                    
                    merged_verts, merged_faces = merge_meshes([cur_rhand_mano_gt, cur_lhand_mano_gt, re_trans_obj_pcs], [mano_hand_faces_np, mano_hand_faces_np, obj_faces_np])
                    mesh  = trimesh.Trimesh(merged_verts, merged_faces)
                    mesh_sv_fn = '{:0>8d}_ts_{:0>3d}_gt_mano_retrans_obj.ply'.format(self.iter_step, i_ts)
                    mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
                    
                
                
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
            'ts_to_hand_obj_verts': ts_to_hand_obj_verts,
            'hand_faces_np': hand_faces_np,
            'cur_hand_mesh_faces_left': cur_hand_mesh_faces_left,
            'obj_faces_np': obj_faces_np
        }
        
        
        sv_dict_sv_fn = '{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
        

    
    def validate_contact_info_robo(self, ): # validate mesh robo #
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # and you
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        # one single hand or # not very easy to 
        # self.hand_faces, self.obj_faces # # j
        # mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        # hand_faces_np = self.robo_hand_faces.detach().cpu().numpy() # ### robot faces # 
        # obj_faces_np = self.obj_faces.detach().cpu().numpy()
        init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        # ts_to_obj_quaternion = {}
        # ts_to_obj_rot_mtx = {}
        # ts_to_obj_trans = {}
        # ts_to_hand_obj_verts = {}
        ts_to_obj_verts = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0,( self.nn_ts - 1) * self.mano_nn_substeps, 1):
            cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            # cur_mano_rhand = self.rhand_verts[i_ts].detach().cpu().numpy() # 
            # cur_rhand_verts = cur_hand_mesh # [: cur_hand_mesh.size(0) // 2]
            # cur_lhand_verts = cur_hand_mesh # [cur_hand_mesh.size(0) // 2:]
            # cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx:
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            # if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
            #     ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
            #     ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## #
            #     ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
            #     ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            # else: # 
            #     ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
            #     ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
            #     ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            ts_to_obj_verts[i_ts] = cur_transformed_obj # cur_transformed_obj #
            # ts_to_hand_obj_verts[i_ts] = (cur_rhand_verts_np, cur_transformed_obj)
            # # merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            # if i_ts % 10 == 0:
            #     merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj], [hand_faces_np, obj_faces_np])
            #     mesh = trimesh.Trimesh(merged_verts, merged_faces)
            #     mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
            #     mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
            
        # joint torque # joint torque #
        cur_sv_penetration_points_fn = os.path.join(self.base_exp_dir, "meshes", f"penetration_points_{self.iter_step}.npy")
        # timestep_to_raw_active_meshes, timestep_to_penetration_points, timestep_to_penetration_points_forces
        cur_timestep_to_accum_acc = {
            ts: self.other_bending_network.timestep_to_accum_acc[ts].detach().cpu().numpy() for ts in self.other_bending_network.timestep_to_accum_acc
        }
        # save penetration points # joint
        cur_sv_penetration_points = {
            'timestep_to_raw_active_meshes': self.timestep_to_raw_active_meshes,
            'timestep_to_penetration_points': self.timestep_to_penetration_points,
            'timestep_to_penetration_points_forces': self.timestep_to_penetration_points_forces,
            'ts_to_obj_verts': ts_to_obj_verts,
            'cur_timestep_to_accum_acc': cur_timestep_to_accum_acc
        }
        np.save(cur_sv_penetration_points_fn, cur_sv_penetration_points)
        
        # joint_name_to_penetration_forces_intermediates
        cur_sv_joint_penetration_intermediates_fn = os.path.join(self.base_exp_dir, "meshes", f"joint_penetration_intermediates_{self.iter_step}.npy")
        np.save(cur_sv_joint_penetration_intermediates_fn, self.joint_name_to_penetration_forces_intermediates)
        
        
        # [upd_contact_active_point_pts, upd_contact_point_position, (upd_contact_active_idxes, upd_contact_passive_idxes), (upd_contact_frame_rotations, upd_contact_frame_translations)]
        tot_contact_pairs_set = {}
        
        for ts in self.contact_pairs_sets:
            cur_contact_pairs_set = self.contact_pairs_sets[ts]
            if isinstance(cur_contact_pairs_set, dict):
                tot_contact_pairs_set[ts] = {
                    'upd_contact_active_idxes': cur_contact_pairs_set['contact_active_idxes'].detach().cpu().numpy(),
                    'upd_contact_passive_pts': cur_contact_pairs_set['contact_passive_pts'].detach().cpu().numpy(), # in contact indexes #
                }
            else:
                [upd_contact_active_point_pts, upd_contact_point_position, (upd_contact_active_idxes, upd_contact_passive_idxes), (upd_contact_frame_rotations, upd_contact_frame_translations)] = cur_contact_pairs_set
                tot_contact_pairs_set[ts] = {
                    'upd_contact_active_idxes': upd_contact_active_idxes.detach().cpu().numpy(),
                    'upd_contact_passive_idxes': upd_contact_passive_idxes.detach().cpu().numpy(), # in contact indexes #
                    
                }
        # sampled_verts_idxes
        tot_contact_pairs_set['sampled_verts_idxes'] = self.sampled_verts_idxes.detach().cpu().numpy()
        tot_contact_pairs_set_fn = os.path.join(self.base_exp_dir, "meshes", f"tot_contact_pairs_set_{self.iter_step}.npy")
        np.save(tot_contact_pairs_set_fn, tot_contact_pairs_set)
        
        # self.ts_to_contact_force_d[cur_ts] = self.other_bending_network.contact_force_d.detach().cpu().numpy()
        #         self.ts_to_penalty_frictions[cur_ts] = self.other_bending_network.penalty_friction_tangential_forces.detach().cpu().numpy()
            
        contact_forces_dict = {
            'ts_to_contact_force_d': self.ts_to_contact_force_d, 
            'ts_to_penalty_frictions': self.ts_to_penalty_frictions,
            'ts_to_penalty_disp_pts': self.ts_to_penalty_disp_pts,
            'cur_timestep_to_accum_acc': cur_timestep_to_accum_acc,
            'ts_to_passive_normals': self.ts_to_passive_normals,
            'ts_to_passive_pts': self.ts_to_passive_pts,
            'ts_to_contact_passive_normals': self.ts_to_contact_passive_normals, ## the normal directions of the inc contact apssive object points #
        }
        contact_forces_dict_sv_fn = os.path.join(self.base_exp_dir, "meshes", f"contact_forces_dict_{self.iter_step}.npy")
        np.save(contact_forces_dict_sv_fn, contact_forces_dict)
        
        # [upd_contact_active_point_pts, upd_contact_point_position, (upd_contact_active_idxes, upd_contact_passive_idxes), (upd_contact_frame_rotations, upd_contact_frame_translations)] = self.contact_pairs_set
        # contact_pairs_info = {
        #     'upd_contact_active_idxes': upd_contact_active_idxes.detach().cpu().numpy()
        # }
        
        
        
    # train_robot_actions_from_mano_model_rules
    def validate_mesh_expanded_pts(self, ):
        # one single hand or
        # self.hand_faces, self.obj_faces #
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        # validate_mesh_expanded_pts = self.faces.detach().cpu().numpy() # ### robot faces # 
        # obj_faces_np = self.obj_faces.detach().cpu().numpy()
        init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        ts_to_transformed_obj = {}
        ts_to_active_pts = {}
        ts_to_mano_hand_verts = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0, self.nn_ts, 10):
            cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            cur_mano_rhand = self.rhand_verts[i_ts].detach().cpu().numpy()
            cur_rhand_verts = cur_hand_mesh # [: cur_hand_mesh.size(0) // 2]
            # cur_lhand_verts = cur_hand_mesh # [cur_hand_mesh.size(0) // 2:]
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx:
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
                ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
                ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## # 
                ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
                ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            else:
                ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
                ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            ts_to_transformed_obj[i_ts] = cur_transformed_obj # .detach().cpu().numpy()
            ts_to_active_pts[i_ts] = cur_hand_mesh.detach().cpu().numpy()
            ts_to_mano_hand_verts[i_ts] = cur_mano_rhand
            # merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            # mesh = trimesh.Trimesh(merged_verts, merged_faces)
            # mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
            # mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn)) # mesh_sv_fn #
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
            'ts_to_transformed_obj': ts_to_transformed_obj,
            'ts_to_active_pts': ts_to_active_pts,
            'ts_to_mano_hand_verts': ts_to_mano_hand_verts,
            'hand_faces': mano_hand_faces_np,
            # 'fobj_faces': obj_faces_np,
        }
        sv_dict_sv_fn = '{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
    
    def validate_mesh_mano(self, ):
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # and you
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        # one single hand or
        # self.hand_faces, self.obj_faces #
        mano_hand_faces_np = self.hand_faces.detach().cpu().numpy()
        hand_faces_np = self.faces.detach().cpu().numpy() # ### robot faces # 
        obj_faces_np = self.obj_faces.detach().cpu().numpy()
        init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        # for i_ts in range(1, self.nn_timesteps - 1, 10):
        for i_ts in range(0, self.nn_ts, 10):
            cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            cur_mano_rhand = self.rhand_verts[i_ts].detach().cpu().numpy()
            cur_rhand_verts = cur_hand_mesh # [: cur_hand_mesh.size(0) // 2]
            # cur_lhand_verts = cur_hand_mesh # [cur_hand_mesh.size(0) // 2:]
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            # cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            if i_ts not in self.other_bending_network.timestep_to_optimizable_rot_mtx:
                cur_pred_rot_mtx = np.eye(3, dtype=np.float32)
                cur_pred_trans = np.zeros((3,), dtype=np.float32)
            else:
                cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            if i_ts not in self.other_bending_network.timestep_to_optimizable_quaternion:
                ts_to_obj_quaternion[i_ts] = np.zeros((4,), dtype=np.float32)
                ts_to_obj_quaternion[i_ts][0] = 1. # ## quaternion ## # 
                ts_to_obj_rot_mtx[i_ts] = np.eye(3, dtype=np.float32)
                ts_to_obj_trans[i_ts] = np.zeros((3,),  dtype=np.float32)
            else:
                ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
                ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
                ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_transformed_obj, cur_mano_rhand], [hand_faces_np, obj_faces_np, mano_hand_faces_np])
            mesh = trimesh.Trimesh(merged_verts, merged_faces)
            mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
            mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
        }
        sv_dict_sv_fn = '{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
        

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        def merge_meshes(verts_list, faces_list):
            tot_verts_nn = 0
            merged_verts = []
            merged_faces = []
            for i_mesh in range(len(verts_list)):
                merged_verts.append(verts_list[i_mesh])
                merged_faces.append(faces_list[i_mesh] + tot_verts_nn) # and you
                tot_verts_nn += verts_list[i_mesh].shape[0]
            merged_verts = np.concatenate(merged_verts, axis=0)
            merged_faces = np.concatenate(merged_faces, axis=0)
            return merged_verts, merged_faces
        
        
        
        if self.train_multi_seqs:
            # seq_idx = torch.randint(low=0, high=len(self.rhand_verts), size=(1,)).item()
            seq_idx = self.seq_idx
            cur_hand_faces = self.hand_faces[seq_idx]
            cur_obj_faces = self.obj_faces[seq_idx]
            timestep_to_passive_mesh = self.obj_verts[seq_idx]
            timestep_to_active_mesh = self.hand_verts[seq_idx]
        else:
            cur_hand_faces = self.hand_faces
            cur_obj_faces = self.obj_faces
            timestep_to_passive_mesh = self.timestep_to_passive_mesh
            timestep_to_active_mesh = self.timestep_to_active_mesh
            
        
        # one single hand or
        # self.hand_faces, self.obj_faces #
        # hand_faces_np = self.hand_faces.detach().cpu().numpy()
        # obj_faces_np = self.obj_faces.detach().cpu().numpy()
        # init_passive_obj_verts = self.timestep_to_passive_mesh[0].detach().cpu().numpy()
        
        hand_faces_np = cur_hand_faces.detach().cpu().numpy()
        obj_faces_np = cur_obj_faces.detach().cpu().numpy()
        init_passive_obj_verts = timestep_to_passive_mesh[0].detach().cpu().numpy()
        init_passive_obj_verts_center = np.mean(init_passive_obj_verts, axis=0, keepdims=True)
        mesh_sv_root_dir = os.path.join(self.base_exp_dir, 'meshes')
        os.makedirs(mesh_sv_root_dir, exist_ok=True)
        ts_to_obj_quaternion = {}
        ts_to_obj_rot_mtx = {}
        ts_to_obj_trans = {}
        for i_ts in range(1, self.n_timesteps, 10):
            # cur_hand_mesh = self.timestep_to_active_mesh[i_ts]
            cur_hand_mesh = timestep_to_active_mesh[i_ts]
            cur_rhand_verts = cur_hand_mesh[: cur_hand_mesh.size(0) // 2]
            cur_lhand_verts = cur_hand_mesh[cur_hand_mesh.size(0) // 2: ]
            cur_rhand_verts_np = cur_rhand_verts.detach().cpu().numpy()
            cur_lhand_verts_np = cur_lhand_verts.detach().cpu().numpy()
            cur_pred_rot_mtx = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
            cur_pred_trans = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            cur_transformed_obj = np.matmul(cur_pred_rot_mtx, (init_passive_obj_verts - init_passive_obj_verts_center).T).T + init_passive_obj_verts_center + np.reshape(cur_pred_trans, (1, 3))

            ts_to_obj_quaternion[i_ts] = self.other_bending_network.timestep_to_optimizable_quaternion[i_ts].detach().cpu().numpy()
            ts_to_obj_rot_mtx[i_ts] = self.other_bending_network.timestep_to_optimizable_rot_mtx[i_ts].detach().cpu().numpy()
            ts_to_obj_trans[i_ts] = self.other_bending_network.timestep_to_optimizable_total_def[i_ts].detach().cpu().numpy()
            # 
            merged_verts, merged_faces = merge_meshes([cur_rhand_verts_np, cur_lhand_verts_np, cur_transformed_obj], [hand_faces_np, hand_faces_np, obj_faces_np])
            mesh = trimesh.Trimesh(merged_verts, merged_faces)
            mesh_sv_fn = '{:0>8d}_ts_{:0>3d}.ply'.format(self.iter_step, i_ts)
            mesh.export(os.path.join(mesh_sv_root_dir, mesh_sv_fn))
        ts_sv_dict = {
            'ts_to_obj_quaternion': ts_to_obj_quaternion,
            'ts_to_obj_rot_mtx': ts_to_obj_rot_mtx,
            'ts_to_obj_trans': ts_to_obj_trans,
        }
        sv_dict_sv_fn = '{:0>8d}.npy'.format(self.iter_step)
        sv_dict_sv_fn = os.path.join(mesh_sv_root_dir, sv_dict_sv_fn)
        np.save(sv_dict_sv_fn, ts_sv_dict)
        
        
    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()




if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    bending_net_type = runner.conf['model.bending_net_type']
    

    
    ############# dynamic mano optimization and the pointset representation trajectory optimization #############
    ########## Dynamic MANO optimization ##########
    if args.mode == "train_dyn_mano_model":
        runner.train_dyn_mano_model() # 
        
    elif args.mode == "train_dyn_mano_model_wreact":
        runner.train_dyn_mano_model_wreact()
        
    ########## Retargeting -- Expanded set motion ##########
    elif args.mode == "train_point_set":
        runner.train_point_set_dyn()
    
    
    ########## Retargeting -- Expanded set motion retargeting ##########
    elif args.mode == "train_point_set_retar": ### retargeting ###
        runner.train_point_set_retargeting() ### retargeting ###

    
    ########## Retargeting -- Expanded set motion retargeting ##########
    elif args.mode == "train_point_set_retar_pts": ### retargeting ###
        runner.train_point_set_retargeting_pts() ### retargeting ###

    
    ########## Retargeting -- GRAB & TACO ##########
    elif args.mode == 'train_sparse_retar':
        runner.train_sparse_retar()
    


