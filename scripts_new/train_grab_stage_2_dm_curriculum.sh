
export PYTHONPATH=.

export cuda_ids="3"



export conf=wmask_refine_passive_rigidtrans_forward.conf
export data_case=hand_test_routine_2_light_color_wtime_active_passive


export trainer=exp_runner_stage_2.py


### optimize for the manipulatable hand actions ###
export mode="train_manip_acts_params"


# export conf=dyn_grab_arti_shadow_multi_stages.conf
export conf=dyn_grab_arti_shadow_dm_curriculum.conf


# bash scripts_new/train_grab_stage_2_dm_curriculum.sh


export conf_root="./confs_new"


export cuda_ids="1"


CUDA_VISIBLE_DEVICES=${cuda_ids} python ${trainer} --mode ${mode} --conf ${conf_root}/${conf} --case ${data_case}

