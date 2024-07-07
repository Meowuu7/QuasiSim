
export PYTHONPATH=.

export policy_model="control_vae"
export wandb="--wandb"

export cuda_ids="1"


# export policy_rollout_length=3
export policy_rollout_length=24

export policy_rollout_length=19

# export policy_rollout_length=10
# bash scripts/run_train_policy_hoi_wana.sh

export wana="--wana"

export save_dir="./exp"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_kh_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_200_kh_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter64_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_exana_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_svmore_"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter2048_svmore_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter2048_svmore_ptsfeats_"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter2048_svmore_nobs_"


# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm5000_wm2_"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_"

# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_defaulparams_"


# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_defaulparamsj__"

# with the analytical sim and only to close the gap between the analytical and the sim #

# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507.urdf"
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_camera_wcollision.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_30_camera_data.npy"


export wana=""

export wnorm="--wnorm"
export wnorm=""

export bullet_nn_substeps=100

export bullet_nn_substeps=200

export policy_weight_l1=0.1
export policy_weight_l2=0.01



## use the analytical only with trajectory optimization ##
export conf_path="/home/xueyi/diffsim/NeuS/confs/dyn_grab_mano_model_states.conf"
export use_ana="--use_ana"
export traj_opt="--traj_opt"
# export use_ana=""

# control in the target sim # control in the current sim -> only positions # 
# via simple pd conroller #
## and end to end optimization #
# states right pgrain and dgain #

# delta at most velocity ##
### sue not the analytical but still traj opt? ## the wolrd model's learning ##
export use_ana=""
export traj_opt="--traj_opt"

export use_ana=""
export wana="--wana"

export sub_iter=1024

export world_model_lr=0.0001

export sub_iter=2048
export nn_traj_samples=2048

## test for point features ##
export point_features="--point_features"
export nn_traj_samples=2048


## test for normalizing observations for network input ##
export wnorm_obs="--wnorm_obs"
export point_features=""
export nn_traj_samples=2048

## test for lighter obj masss ##
export wnorm_obs=""
export point_features=""
export nn_traj_samples=1024
export sub_iter=1024

export train_res_contact_forces="--train_res_contact_forces"

export penetration_proj_k_to_robot=400000000.0

export reset_mano_states="--reset_mano_states"


# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_cube_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_defaulparamsj__"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_defaulparamsj__2_"

# # ## the mano urdf fn ## # urdf fn ##
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507.urdf"
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_cube_wcollision.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_20_cube_data.npy"


#### GRAB train split 25-seq ball ####
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_ball_wcollision.urdf"
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507.urdf"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball_adc_1_"
export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_25_ball_data.npy"
export reset_mano_states="--reset_mano_states"

### GRAB train split 54-seq cylinder ###
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_cylinder_wcollision.urdf"
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_54_cylinder_data.npy"
# export reset_mano_states=""
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_54_cylinder_0_3_"
# export tag="obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_54_cylinder_1_"


export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/54_obj.obj"


## generate the SDF and the SDF mesh scales for using in the collision detections ##

## generate the SDF and the SDF mesh scales for using in the collision detections ##

# and all ohter GRAB clips ###
### GRAB train split 1-seq ###
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_dingshuji_wcollision.urdf"
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_1_dingshuji_data.npy"
# export tag="1024_subiters_optim_params_1_dingshuji_1_mas10000_"


## can this aproach fulfill our vision ##
### GRAB train split 224-seq ###
# export conf_path="/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states.conf"
# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states.conf
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_tiantianquan_wcollision.urdf"
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_224_tiantianquan_data.npy"
# export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/224_obj.obj"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv2_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv3_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv4_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv5_"
# export wana="--wana"

# ### for the 54 sequence ###
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_cylinder_wcollision.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_54_cylinder_data.npy"
# export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/54_obj.obj"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas5000_"


# get the ano urdf ffile ##
# save gt referecnes fn ## ---- the tiantianquan's reference gt file ##
## obj urdf fn ## ## not 

export plane_z=-0.2
export cuda_ids="1"

export world_model_rollout_length=19
export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_c_"


export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_224.conf

# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_tiantianquan_wcollision.urdf"
# export mano_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_224_tiantianquan_data.npy"
# # shadow_grab_train_split_54_cylinder_data_v2
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_224_tiantianquan_data_v2.npy"
# export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/224_obj.obj"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_b_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_"
# export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_"


export cuda_ids="3"



export only_residual="--only_residual"

export penetration_proj_k_to_robot_friction=10000000.0
export penetration_proj_k_to_robot_friction=100000000.0


# export nn_traj_samples=124
export nn_traj_samples=1024
# export sub_iter=14

export cuda_ids="7"
export use_mano_delta_states="--use_mano_delta_states"
export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_deltastates_"

export use_mano_delta_states=""
export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_stdv2_"


export pred_delta_offsets="--pred_delta_offsets"
export finger_state_action_sigma=0.0001
export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_stdv1_predelta_"
# export nn_traj_samples=124
export use_contact_region_pts=""
export pred_delta_offsets="--pred_delta_offsets"
export finger_state_action_sigma=0.01
export tag="1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_stdv1_predelta_new_"


############# 54 - cylinder #############
# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states.conf
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_cylinder_wcollision.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_54_cylinder_data.npy"
# export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/54_obj.obj"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_wnet_"

# # export nn_traj_samples=124
# export nn_traj_samples=1024
# # export sub_iter=14

# export cuda_ids="0"
# export use_mano_delta_states="--use_mano_delta_states"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_wnet_deltastates_"

# export use_mano_delta_states=""
# export pred_delta_offsets="--pred_delta_offsets"
# export finger_state_action_sigma=0.0001
# export finger_state_action_sigma=0.01
# export finger_state_action_sigma=0.1
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_wnet_preddelta_"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_wnet_preddelta_stdv2_"
# export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_wnet_preddelta_stdv4_"
# # export nn_traj_samples=124
# ################ 54 - cylinder #################


############# 89 - flashlight #############
export cuda_ids="0"
# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_89.conf
# export obj_urdf_fn="/home/xueyi/diffsim/NeuS/rsc/mano/grab_flashlight_wcollision.urdf"
# export sv_gt_refereces_fn="/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_89_flashlight_data.npy"
# export canon_obj_fn="/data1/xueyi/GRAB_extracted_test/train/89_obj.obj"
# export tag="1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_stdv1_predelta_new_"
export pred_delta_offsets="--pred_delta_offsets"
export use_contact_region_pts=""
export pred_delta_offsets=""

## several wm variants ##
# train residual 
# only residual
export only_residual="--only_residual"
# pred_delta_offsets
# use_contact_region_pts


export tag="1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_"

export cuda_ids="1"
export pred_delta_offsets="--pred_delta_offsets"
export use_contact_region_pts="--use_contact_region_pts"
export tag="1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_cont_region_"

export use_contact_network="--use_contact_network"

## use the network to agument contact normal and tangential forces ##
export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export tag="1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_nnet_"
############# 89 - flashlight #############


############# 102 - mouse #############
export finger_state_action_sigma=0.01
export obj_name="mouse"
export obj_idx=102

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""

export use_preset_inertia="--use_preset_inertia"

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_102.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# # export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_


export use_preset_inertia=""
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_npresetinertia_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_npresetinertia_mass7000_

export use_contact_network="--use_contact_network"
export pred_delta_offsets="--pred_delta_offsets"
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_npresetinertia_wconnet_predelta_

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data_woptrobo.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_test_woptrobo_v1_
############# 102 - mouse #############

############# 7 - handle #############
export obj_name="handle"
export obj_idx=7

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass7000_
## mass distributions ##
############# 7 - handle #############


############# 47 - scissor #############
export obj_name="scissor"
export obj_idx=47

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_
## mass distributions ##

export use_contact_network="--use_contact_network"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_wcnet_
############# 47 - scissor #############



############# 67 - banana #############
export obj_name="banana"
export obj_idx=67

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_

export angular_damping_coef=0.7
# export use_preset_inertia="--use_preset_inertia"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_
export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_coretest_

## mass distributions ##

# export use_contact_network="--use_contact_network"
# export pred_delta_offsets="--pred_delta_offsets"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_wcnet_preddelta_

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_wcnet_preddelta_woffsetfeats_

# export finger_state_action_sigma=0.0001
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d0001_netv1_mass10000_wcnet_preddelta_woffsetfeats_
############# 67 - banana #############


############# 76 - quadrangular #############
export obj_name="quadrangular"
export obj_idx=76

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj


export use_contact_network="--use_contact_network"
export pred_delta_offsets="--pred_delta_offsets"
export angular_damping_coef=0.7
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_
############# 76 - quadrangular #############


############# 85 - bunny #############
export obj_name="bunny"
export obj_idx=85

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
export angular_damping_coef=1.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_objts2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_objts2_tsintervalv2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_objts2_tsintervalv3_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_


# export use_cmaes=""
# export use_ambient_contact_net=""


# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_data.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_data_v2.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v3.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v4.npy
# # shadow_grab_train_split_85_bunny_wact_wctftohand_data
# # export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_wctftohand_data.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_1_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_3_


# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_4_

# export use_cmaes="--use_cmaes"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_4_wcmaes_

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_5_wcmaes_
# export use_contact_network="--use_contact_network"
# export pred_delta_offsets="--pred_delta_offsets"
# export angular_damping_coef=0.7
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_
############# 85 - bunny #############

# export obj_name="mouse"
# export obj_idx=102

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_102.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data_woptrobo.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_test_woptrobo_v1_

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_test_v2_

############# 91 - hammer #############
# export obj_name="hammer"
# export obj_idx=91

# export use_contact_network=""
# export pred_delta_offsets=""
# export use_contact_region_pts=""
# export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export angular_damping_coef=1.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_defaultdt_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv2_
# # export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_objts2_


# export use_optimizable_timecons="--use_optimizable_timecons"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_

# export angular_damping_coef=2.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_

# export angular_damping_coef=2.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_lmass_

# # shadow_grab_train_split_91_hammer_wact_data
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_data.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_lmass_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_


# export nn_traj_samples=124
# export nn_traj_samples=1024
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_

# # shadow_grab_train_split_91_hammer_wact_thres0d0_data
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_thres0d0_data.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_std0d05_fromthres0d0_


# # use_ambient_contact_net
# # export nn_traj_samples=124
# export use_ambient_contact_net="--use_ambient_contact_net"

# export use_cmaes="--use_cmaes"

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_std0d05_fromthres0d0_ambientcnet_

# # export use_ambient_contact_net=""
# # export angular_damping_coef=2.0
# # export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_
# # export use_contact_network="--use_contact_network"
# # export pred_delta_offsets="--pred_delta_offsets"
# # # export angular_damping_coef=0.7
# # export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_cnetpreddelta_
############# 91 - hammer #############

export damping_coef=100.0

############# 167 - hammer #############
# export obj_name="hammer167"
# export obj_idx=167

# export use_contact_network=""
# export pred_delta_offsets=""
# export use_contact_region_pts=""
# export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export nn_traj_samples=1024
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_

# export angular_damping_coef=1.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp1d0_

# export angular_damping_coef=0.7
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_

# export angular_damping_coef=0.7
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_


# export damping_coef=10.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_ldp10d0_
############# 167 - hammer #############

############# 110 - phone #############
# export obj_name="phone"
# export obj_idx=110

# export use_contact_network=""
# export pred_delta_offsets=""
# export use_contact_region_pts=""
# export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
# export angular_damping_coef=1.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_defaultdt_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv2_
# # export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_objts2_


# export use_optimizable_timecons="--use_optimizable_timecons"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_

# # export use_contact_network="--use_contact_network"
# # export pred_delta_offsets="--pred_delta_offsets"
# # export angular_damping_coef=0.7
# # export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_
############# 110 - phone #############


############# 107 - stapler #############
export obj_name="stapler107" 
export obj_idx=107

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data_fingerretar.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
export angular_damping_coef=2.0


export use_cmaes=""
export use_ambient_contact_net=""

# # export use_cmaes="--use_cmaes" # grab with collision ##
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv2_
# 2.0
export angular_damping_coef=2.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_
# export angular_damping_coef=3.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv4_

export abd_tail_frame_nn=11

export friction_coef=10.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_fr100_


export add_table=""
export gravity_scale=0.0

# export abd_tail_frame_nn=2
# export tag=subiterPs1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv5_

# export use_cmaes="--use_cmaes"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv5_wcmaes_


# export world_model_rollout_length=25
# export policy_rollout_length=25
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv5_wcmaes_rollout25_
############# 107 - stapler #############


############# 85 - bunny #############
export obj_name="bunny"
export obj_idx=85

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj
export angular_damping_coef=1.0

export use_cmaes=""
export use_ambient_contact_net=""

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_data.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_data_v2.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v3.npy
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v4.npy
# # shadow_grab_train_split_85_bunny_wact_wctftohand_data
# # export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_wact_wctftohand_data.npy

export gravity_scale=-9.8
export table_height=-0.67
export add_table="--add_table"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_wtable_gn2_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_wtable_gn9d8_
# # export use_contact_network="--use_contact_network"
# export pred_delta_offsets="--pred_delta_offsets"
# export angular_damping_coef=0.7
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_
############# 85 - bunny #############

############# 102 - mouse #############
export obj_name="mouse"
export obj_idx=102

export use_ambient_contact_net=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_102.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export canon_obj_fn=/data1/xueyi/GRAB_extracted_test/train/${obj_idx}_obj.obj

export gravity_scale=-9.8
export table_height=-0.652
export add_table="--add_table"
export use_cmaes=""

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data_woptrobo.npy
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_test_woptrobo_wtable_gn9d8_

# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data.npy
############# 102 - mouse #############


############# 91 - hammer #############
export obj_name="hammer"
export obj_idx=91

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export conf_path=/home/xueyi/diffsim/NeuS/confs/dyn_grab_shadow_model_states_${obj_idx}.conf
# export obj_urdf_fn=/home/xueyi/diffsim/NeuS/rsc/mano/grab_${obj_name}_wcollision.urdf
# export sv_gt_refereces_fn=/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy
# export canon_obj_fn=./data/GRAB_extracted_test/train/${obj_idx}_obj.obj
export nn_traj_samples=1024

export angular_damping_coef=2.0

export use_cmaes="--use_cmaes"

export add_table="--add_table"
export gravity_scale=-9.8
export table_height=-0.64
export table_y=-0.48
export table_x=0.73

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_std0d05_fromthres0d0_ambientcnet_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_wtable_gn

export use_cmaes=""
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wtable_gn


export angular_damping_coef=1.0
export use_cmaes="--use_cmaes"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_wtable_gn_adp1d0_

export policy_lr=0.001

export policy_lr=0.0001
export use_cmaes=""
export use_cmaes="--use_cmaes"
export angular_damping_coef=0.7 ## setthe angular damping coef ##
export nn_traj_samples=124
export sub_iter=124
export add_table=""
export gravity_scale=0.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wocmase_wtable_gn_adp0d7_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d7_


export train_wm_only="--train_wm_only"
export nn_traj_samples=1024
export angular_damping_coef=1.0
export use_cmaes=""
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_


export train_wm_only=""
# export add_table="--add_table"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d7_

# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_cs0d6_



### save period ###

export train_wm_only="--train_wm_only"
export save_period=10
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_


# export use_ambient_contact_net="--use_ambient_contact_net"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_ambientnet_

export sub_iter=500
export train_wm_only=""
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_cs0d6_ambientnet_


#### 
export pred_tar_mano_states="--pred_tar_mano_states"
export train_wm_only="--train_wm_only"
export nn_traj_samples=124
export sub_iter=124
# export use_ambient_contact_net=""
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_

# export use_ambient_contact_net="--use_ambient_contact_net"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_


export train_policy_only=""


export train_policy_only="--train_policy_only"
export train_wm_only=""
# export use_ambient_contact_net="--use_ambient_contact_net"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_glbsigma0_wanbientnet_



############# 91 - hammer #############


############# 102 - mouse #############
export obj_name="mouse"
export obj_idx=102

export use_ambient_contact_net=""

export conf_path=./confs_new/dyn_grab_shadow_model_states_102.conf
export obj_urdf_fn=./rsc/mano/grab_${obj_name}_wcollision.urdf
export canon_obj_fn=./data/GRAB_extracted_test/train/${obj_idx}_obj.obj
export sv_gt_refereces_fn=./ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy

export tag=mouse_102


# export load_ckpt="/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_/model_30.data"


# export train_wm_only=""
# export train_policy_only="--train_policy_only"
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_predtarmano_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_predtarmano_plossv2_

############# 102 - mouse #############




export save_dir="./exp"



# conda activate control-vae-4

export cuda_ids="0"



CUDA_VISIBLE_DEVICES=${cuda_ids}  python train/train_policy_ha.py --policy_model=${policy_model} ${wandb} --policy_rollout_length=${policy_rollout_length} ${wana} --mano_urdf_fn=${mano_urdf_fn} --obj_urdf_fn=${obj_urdf_fn} --sv_gt_refereces_fn=${sv_gt_refereces_fn} --bullet_nn_substeps=${bullet_nn_substeps} --save_dir=${save_dir} --tag=${tag} --policy_weight_l1=${policy_weight_l1} --policy_weight_l2=${policy_weight_l2} ${wnorm} ${use_ana} --conf_path=${conf_path} ${traj_opt} --sub_iter=${sub_iter} --world_model_lr=${world_model_lr} --nn_traj_samples=${nn_traj_samples} ${point_features} ${wnorm_obs} ${train_res_contact_forces} --plane_z=${plane_z} ${reset_mano_states} --canon_obj_fn=${canon_obj_fn} --world_model_rollout_length=${world_model_rollout_length} ${only_residual} --penetration_proj_k_to_robot_friction=${penetration_proj_k_to_robot_friction} ${use_mano_delta_states} --finger_state_action_sigma=${finger_state_action_sigma} ${pred_delta_offsets} ${use_contact_region_pts} ${use_contact_network} ${use_preset_inertia} --angular_damping_coef=${angular_damping_coef} ${use_optimizable_timecons} --damping_coef=${damping_coef} ${use_ambient_contact_net} ${use_cmaes} --abd_tail_frame_nn=${abd_tail_frame_nn} --friction_coef=${friction_coef} ${add_table} --gravity_scale=${gravity_scale} --table_height=${table_height} --table_y=${table_y} --table_x=${table_x} --policy_lr=${policy_lr} ${train_wm_only} --save_period=${save_period} ${pred_tar_mano_states} --load_ckpt=${load_ckpt} ${train_policy_only} --kinematic_mano_gt_sv_fn=${kinematic_mano_gt_sv_fn} ${two_hands} --left_mano_urdf_fn=${left_mano_urdf_fn} ${use_optimized_obj}
 
