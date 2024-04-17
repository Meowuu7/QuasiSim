
export PYTHONPATH=.



export data_case=hand_test_routine_2_light_color_wtime_active_passive



export trainer=exp_runner_stage_1.py

export mode="train_dyn_mano_model_wreact"
# 
# export mode="train_dyn_mano_model_wreact_"

export conf=dyn_grab_pointset_mano_dyn_optacts.conf


export conf_root="./confs_new"



export cuda_ids="2"


# bash scripts_new/train_grab_mano_wreact_optacts.sh


CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

