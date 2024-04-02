
import torch


def check_checkpoint_file(ckpt_fn):
    loaded_state_dict = torch.load(ckpt_fn, map_location='cpu')
    mano_robot_init_states_dict = loaded_state_dict['mano_robot_init_states']
    print(type(mano_robot_init_states_dict))
    print(mano_robot_init_states_dict.keys())
    mano_init_states_weight = mano_robot_init_states_dict['weight']
    print(f"weight: {mano_init_states_weight.shape}, {type(mano_init_states_weight)}")

if __name__=='__main__':
    ckpt_fn = "/data3/datasets/diffsim/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__thres_000_wglb_onlyhandtrack_sf_10000_/checkpoints/ckpt_052000.pth"
    
    check_checkpoint_file(ckpt_fn)
