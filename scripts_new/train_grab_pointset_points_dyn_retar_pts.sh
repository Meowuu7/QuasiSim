
export PYTHONPATH=.




export data_case=hand_test_routine_2_light_color_wtime_active_passive



export trainer=exp_runner_stage_1.py



# ### stage 1 -> tracking MANO expanded set using Shadow's object mesh points ###
# export mode="train_expanded_set_motions_retar"
# export conf=dyn_grab_pointset_points_dyn_retar.conf

### stage 2 -> tracking MANO expanded set using Shadow's expanded points ###
export mode="train_point_set_retar_pts"
export conf=dyn_grab_pointset_points_dyn_retar_pts.conf




export conf_root="./confs_new"


# bash scripts_new/train_grab_pointset_points_dyn_retar.sh

export cuda_ids="0"



PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

