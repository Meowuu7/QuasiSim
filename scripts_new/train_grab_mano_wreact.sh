
export PYTHONPATH=.



export data_case=hand_test_routine_2_light_color_wtime_active_passive



export trainer=exp_runner_stage_1.py

export mode="train_dyn_mano_model_wreact"

export conf=dyn_grab_pointset_mano_dyn.conf


export conf_root="./confs_new"



export cuda_ids="0"


# bash scripts_new/train_grab_mano_wreact.sh

CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

