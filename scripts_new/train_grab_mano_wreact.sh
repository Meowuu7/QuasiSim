
export PYTHONPATH=.

export cuda_ids="4"

# export cuda_ids="5"


export trainer=exp_runner_arti_forward.py
export conf=wmask_refine_passive_rigidtrans_forward.conf
export data_case=hand_test_routine_2_light_color_wtime_active_passive


# export trainer=exp_runner_arti_multi_objs_compositional.py
export trainer=exp_runner_arti_multi_objs_compositional_ks.py
# export trainer=exp_runner_arti_multi_objs_dyn.py
export trainer=exp_runner_arti_multi_objs_pointset.py
# /home/xueyi/diffsim/NeuS/confs/wmask_refine_passive_compositional.conf
export conf=wmask_refine_passive_compositional.conf
export conf=dyn_arctic_ks.conf

export conf=dyn_arctic_ks_robohand.conf
# /data/xueyi/diffsim/NeuS/confs/dyn_arctic_ks_robohand_from_mano_model_rules.conf
export conf=dyn_arctic_ks_robohand_from_mano_model_rules.conf
export conf=dyn_arctic_robohand_from_mano_model_rules.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v2.conf
export conf=dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v4.conf

# /home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf
export conf=dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf
# /home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf
# export conf=dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf

export conf=dyn_arctic_robohand_from_mano_model_train_mano_dyn_model_states.conf
# export conf=dyn_arctic_ks_robohand_from_mano_model_rules_arti.conf
# a very stiff system for # 
export conf=dyn_grab_pointset_mano.conf

export mode="train_from_model_rules"
export mode="train_from_model_rules"

export mode="train_sdf_from_model_rules"
export mode="train_actions_from_model_rules"
# export mode="train_actions_from_sim_rules"



export mode="train_def"
# export mode="train_actions_from_model_rules"
export mode="train_mano_actions_from_model_rules"

# virtual force # #  ## virtual forces ##
# /data/xueyi/diffsim/NeuS/confs/dyn_arctic_ks_robohand_from_mano_model_rules.conf ##
export mode="train_actions_from_model_rules"

export mode="train_actions_from_mano_model_rules"

export mode="train_real_robot_actions_from_mano_model_rules"
export mode="train_real_robot_actions_from_mano_model_rules_diffhand"

export mode="train_real_robot_actions_from_mano_model_rules_diffhand_fortest"
export mode="train_real_robot_actions_from_mano_model_rules_manohand_fortest"


export mode="train_real_robot_actions_from_mano_model_rules_diffhand_fortest"



export mode="train_real_robot_actions_from_mano_model_rules_manohand_fortest_states"

###### diffsim ######
# /home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v3.conf 

export mode="train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_res_world"
export mode="train_real_robot_actions_from_mano_model_rules_v5_manohand_fortest_states_res_rl"

export mode="train_dyn_mano_model_states"



####### steup mode and conf here #######
## train dyn mano model states wreact ##
## train dyn mano model states wreact ##
# <<<<<<< HEAD:scripts_new/train_grab_pointset_dyn.sh
# export mode="train_dyn_mano_model_states_wreact" ## wreact ## wreact ##

# export conf=dyn_grab_pointset_mano_dyn.conf ## philosophical interesting ##
# ## train dyn mano model states wreact ##
# =======
# export mode="train_dyn_mano_model_states_wreact"
# >>>>>>> 3a9e049570b238cfcb4b2c92529950177d2a81f5:scripts_new/train_grab_mano_wreact.sh


export mode="train_dyn_mano_model_states_wreact" ## wreact ## wreact ##

export conf=dyn_grab_pointset_mano_dyn.conf #

# export conf=dyn_grab_pointset_mano.conf #

export conf_root="./confs_new"



export cuda_ids="0"

# <<<<<<< HEAD:scripts_new/train_grab_pointset_dyn.sh
# export cuda_ids="0"
# # 
# =======
# # bash scripts_new/train_grab_mano_wreact.sh #
# # bash #

# export cuda_ids="0"
# >>>>>>> 3a9e049570b238cfcb4b2c92529950177d2a81f
#  bash scripts_new/train_grab_mano_wreact.sh

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

