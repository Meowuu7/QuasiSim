
export PYTHONPATH=.

export policy_model="control_vae"
export wandb="--wandb"

export cuda_ids="1"

export policy_rollout_length=19


export save_dir="./exp"

export wnorm="--wnorm"
export wnorm=""
export wnorm_obs=""
export use_ana=""
export wana="--wana"

export bullet_nn_substeps=200

export policy_weight_l1=0.1
export policy_weight_l2=0.01
export finger_state_action_sigma=0.01

export damping_coef=1.0
export angular_damping_coef=1.0

export use_ana=""
export traj_opt="--traj_opt"



export sub_iter=1024

export world_model_lr=0.0001

export sub_iter=2048
# export nn_traj_samples=2048


export point_features="--point_features"
# export nn_traj_samples=2048


## test for normalizing observations for network input ##
export point_features=""
# export nn_traj_samples=2048


export point_features=""
export nn_traj_samples=1024
export sub_iter=1024

export train_res_contact_forces="--train_res_contact_forces"
export penetration_proj_k_to_robot=400000000.0
export penetration_proj_k_to_robot_friction=100000000.0

export reset_mano_states=""


export plane_z=-0.2
export cuda_ids="1"

export world_model_rollout_length=19
export tag="1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_b_wnl19_c_"


# export conf_path=./confs_new/dyn_grab_shadow_model_states_224.conf
export mano_urdf_fn=./rsc/shadow_hand_description/shadowhand_new_scaled.urdf 

export cuda_ids="3"



export only_residual="--only_residual"



############# 107 - stapler #############
export obj_name="stapler107" 
export obj_idx=107

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

# export angular_damping_coef=2.0


export use_cmaes=""
export use_ambient_contact_net=""

# # export use_cmaes="--use_cmaes" # grab with collision ##
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv2_
# 2.0
# export angular_damping_coef=2.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_
# export angular_damping_coef=3.0
# export tag=subiters1024_optim_params_shadow_${obj_idx}_${obj_name}_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv4_

export abd_tail_frame_nn=11

export friction_coef=10.0

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
# export angular_damping_coef=1.0

export use_cmaes=""
export use_ambient_contact_net=""

export gravity_scale=-9.8
export table_height=-0.67
export add_table="--add_table"
############# 85 - bunny #############

############# 102 - mouse #############
export obj_name="mouse"
export obj_idx=102

export use_ambient_contact_net=""

export gravity_scale=-9.8
export table_height=-0.652
export add_table="--add_table"
export use_cmaes=""

############# 102 - mouse #############


############# 91 - hammer #############
export obj_name="hammer"
export obj_idx=91

export use_contact_network=""
export pred_delta_offsets=""
export use_contact_region_pts=""
export use_preset_inertia=""

export nn_traj_samples=1024

# export angular_damping_coef=2.0

export use_cmaes="--use_cmaes"

export add_table="--add_table"
export gravity_scale=-9.8
export table_height=-0.64
export table_y=-0.48
export table_x=0.73

export use_cmaes=""
# export angular_damping_coef=1.0
export use_cmaes="--use_cmaes"
export policy_lr=0.001

export policy_lr=0.0001
export use_cmaes=""
export use_cmaes="--use_cmaes"
# export angular_damping_coef=0.7
export nn_traj_samples=124
export sub_iter=124
export add_table=""
export gravity_scale=0.0


export train_wm_only="--train_wm_only"
export nn_traj_samples=1024
# export angular_damping_coef=1.0
export use_cmaes=""

export train_wm_only=""
export train_wm_only="--train_wm_only"
export save_period=10
export train_wm_only=""

#### 
export pred_tar_mano_states="--pred_tar_mano_states"
export train_wm_only="--train_wm_only"
export nn_traj_samples=124
export sub_iter=124

export train_policy_only=""


export train_policy_only="--train_policy_only"
export train_wm_only=""
############# 91 - hammer #############


############# 102 - mouse #############
export obj_name="mouse"
export obj_idx=102

export use_ambient_contact_net=""
# export conf_path=./confs_new/dyn_grab_shadow_model_states_102.conf
export conf_path=./confs_new/dyn_grab_arti_shadow_dm_curriculum.conf
export obj_urdf_fn=./rsc/mano/grab_${obj_name}_wcollision.urdf
export canon_obj_fn=./rsc/mano/meshes/${obj_idx}_obj.obj
export sv_gt_refereces_fn=./ReferenceData/shadow_grab_train_split_${obj_idx}_${obj_name}_data.npy

export tag=mouse_102
############# 102 - mouse #############




export save_dir="./exp"

# bash ./scripts_new/run_train_policy_hoi_wana.sh


# conda activate control-vae-4

export cuda_ids="0"



CUDA_VISIBLE_DEVICES=${cuda_ids}  python train/train_policy_ha.py --policy_model=${policy_model} ${wandb} --policy_rollout_length=${policy_rollout_length} ${wana} --mano_urdf_fn=${mano_urdf_fn} --obj_urdf_fn=${obj_urdf_fn} --sv_gt_refereces_fn=${sv_gt_refereces_fn} --bullet_nn_substeps=${bullet_nn_substeps} --save_dir=${save_dir} --tag=${tag} --policy_weight_l1=${policy_weight_l1} --policy_weight_l2=${policy_weight_l2} ${wnorm} ${use_ana} --conf_path=${conf_path} ${traj_opt} --sub_iter=${sub_iter} --world_model_lr=${world_model_lr} --nn_traj_samples=${nn_traj_samples} ${point_features} ${wnorm_obs} ${train_res_contact_forces} --plane_z=${plane_z} ${reset_mano_states} --canon_obj_fn=${canon_obj_fn} --world_model_rollout_length=${world_model_rollout_length} ${only_residual} --penetration_proj_k_to_robot_friction=${penetration_proj_k_to_robot_friction} ${use_mano_delta_states} --finger_state_action_sigma=${finger_state_action_sigma} ${pred_delta_offsets} ${use_contact_region_pts} ${use_contact_network} ${use_preset_inertia} --angular_damping_coef=${angular_damping_coef} ${use_optimizable_timecons} --damping_coef=${damping_coef} ${use_ambient_contact_net} ${use_cmaes} --abd_tail_frame_nn=${abd_tail_frame_nn} --friction_coef=${friction_coef} ${add_table} --gravity_scale=${gravity_scale} --table_height=${table_height} --table_y=${table_y} --table_x=${table_x} --policy_lr=${policy_lr} ${train_wm_only} --save_period=${save_period} ${pred_tar_mano_states} --load_ckpt=${load_ckpt} ${train_policy_only} --kinematic_mano_gt_sv_fn=${kinematic_mano_gt_sv_fn} ${two_hands} --left_mano_urdf_fn=${left_mano_urdf_fn} ${use_optimized_obj}
 
