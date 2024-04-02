
export PYTHONPATH=.

# export cuda_ids="3"



export conf=wmask_refine_passive_rigidtrans_forward.conf
export data_case=hand_test_routine_2_light_color_wtime_active_passive

export trainer=exp_runner_stage_2.py




export mode="train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab"
## dyn grab shadow 

### optimze for the redmax hand actions from joint states ###
export mode="train_diffhand_model"

## optimize for the manipulatable hand actions ###
# export mode="train_real_robot_actions_from_mano_model_rules_v5_shadowhand_fortest_states_grab_redmax_acts"



export conf=dyn_grab_arti_shadow_diffhand.conf




export conf_root="./confs_new"


export cuda_ids="3"


CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

