
export PYTHONPATH=.




export data_case=hand_test_routine_2_light_color_wtime_active_passive



export trainer=exp_runner_stage_1.py


export mode="train_point_set"

export conf=dyn_grab_pointset_points_dyn_s3.conf

export conf_root="./confs_new"


# bash scripts_new/train_grab_pointset_points_dyn_s3.sh

export cuda_ids="0"


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

