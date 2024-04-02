
export PYTHONPATH=.

export cuda_ids="3"



export trainer=exp_runner_arti_forward.py
export conf=wmask_refine_passive_rigidtrans_forward.conf
export data_case=hand_test_routine_2_light_color_wtime_active_passive


# export trainer=exp_runner_arti_multi_objs_compositional.py
export trainer=exp_runner_arti_multi_objs_compositional_ks.py
export trainer=exp_runner_arti_multi_objs_arti_dyn.py
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v2.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v4.conf

# /home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf
export conf=dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf

# /home/xueyi/diffsim/NeuS/confs/dyn_grab_mano_model_states.conf
export conf=dyn_grab_mano_model_states.conf

export conf=dyn_grab_shadow_model_states.conf




export mode="train_def"
# export mode="train_actions_from_model_rules"
export mode="train_mano_actions_from_model_rules"

# virtual force #
# /data/xueyi/diffsim/NeuS/confs/dyn_arctic_ks_robohand_from_mano_model_rules.conf ##

export mode="train_real_robot_actions_from_mano_model_rules_diffhand_fortest"



export mode="train_real_robot_actions_from_mano_model_rules_manohand_fortest_states"


### using the diffsim ###

###### diffsim ######
# /home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v3.conf 


export mode="train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_res_world"
export mode="train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_res_rl"

export mode="train_dyn_mano_model_states"

# 
export mode="train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab"
## dyn grab shadow 


### optimize for the manipulatable hand actions ###
export mode="train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab_redmax_acts"



# ## acts ##
# export conf=dyn_grab_shadow_model_states_224.conf
# export conf=dyn_grab_shadow_model_states_89.conf
# export conf=dyn_grab_shadow_model_states_102.conf
# export conf=dyn_grab_shadow_model_states_7.conf
# export conf=dyn_grab_shadow_model_states_47.conf
# export conf=dyn_grab_shadow_model_states_67.conf
# export conf=dyn_grab_shadow_model_states_76.conf
# export conf=dyn_grab_shadow_model_states_85.conf
# # export conf=dyn_grab_shadow_model_states_91.conf
# # export conf=dyn_grab_shadow_model_states_167.conf
# export conf=dyn_grab_shadow_model_states_107.conf
# export conf=dyn_grab_shadow_model_states_306.conf
# export conf=dyn_grab_shadow_model_states_313.conf
# export conf=dyn_grab_shadow_model_states_322.conf


# /home/xueyi/diffsim/NeuS/confs_new/dyn_grab_arti_shadow_multi_stages.conf
export conf=dyn_grab_arti_shadow_multi_stages.conf
# export conf=dyn_grab_shadow_model_states_398.conf
# export conf=dyn_grab_shadow_model_states_363.conf
# export conf=dyn_grab_shadow_model_states_358.conf

# bash scripts_new/train_grab_shadow_multistages.sh





export conf_root="./confs_new"


export cuda_ids="4"


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

