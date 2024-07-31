import os
from ControlVAECore.Env.bullet_hoi_track_env import VCLODETrackEnv
from ControlVAECore.Env.bullet_hoi_track_env_twohands import VCLODETrackEnv as VCLODETrackEnvAnaTwoHands
import numpy as np
import wandb
from utils.parser import train_policy_args
from train import control_vae_hoi_wana
from train import control_vae_hoi_wana_twohands
# from train import rl_ppo
# from train.diffusion_policy import PolicyAgent
# from utils.model_util import create_gaussian_diffusion


def launch_rlg_hydra():
    
    cfg = load_hydra_conf("../isaacgymenvs/cfg", "config")

    import isaacgymenvs
    from datetime import datetime
    from omegaconf import open_dict
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    with open_dict(cfg):
        cfg.task.test = cfg.test
        
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # # ensure checkpoints can be specified as relative paths
    # if cfg.checkpoint:
    #     cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    
    
    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
    if cfg.capture_video:
        envs.is_vector_env = True
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )
        
    act_dim = envs.num_hand_acts
    return envs




def load_hydra_conf(conf_path, conf_name="config"):
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    with initialize(version_base=None, config_path=conf_path, job_name="test_app"):
        cfg = compose(config_name=conf_name, overrides=["db=mysql", "db.user=me"])

    cfg = OmegaConf.to_yaml(cfg)
    return cfg



if __name__ == '__main__':
    args = train_policy_args()
    if args['wandb']:
        wandb.init(project='Short')
        
    ##### Unorganized part #####
    # if args['get_data_fns_from_objinfo']:
    #     if args['dataset_type'] == 'taco':
    #         CTL_ROOT = "/home/xueyi/diffsim/Control-VAE"
    #         if not os.path.exists(CTL_ROOT):
    #             CTL_ROOT = "/root/diffsim/control-vae-2"
    #         NEUS_ROOT = "/home/xueyi/diffsim/NeuS"
    #         if not os.path.exists(NEUS_ROOT):
    #             NEUS_ROOT = "/root/diffsim/quasi-dyn"
    #         TACO_DATA_ROOT = "/data2/datasets/xueyi/taco/processed_data"
    #         if not os.path.exists(TACO_DATA_ROOT):
    #             TACO_DATA_ROOT = "/data/xueyi/taco/processed_data"
    #         URDF_ROOT = "/home/xueyi/diffsim/Control-VAE/assets"
    #         if not os.path.exists(URDF_ROOT):
    #             URDF_ROOT = "/root/diffsim/control-vae-2/assets"
    #         #### Get object information --- obj_name, obj_idxx, obj_idxx_root ####
    #         obj_name = args['obj_name']
    #         obj_idxx = args['obj_idxx']
    #         obj_idxx_root = obj_idxx.split('_')[0]
            
    #         args['kinematic_mano_gt_sv_fn'] = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.pkl"
            
    #         gt_reference_tag = args['gt_reference_tag']
    #         if len(gt_reference_tag) > 0:
    #             args['sv_gt_refereces_fn'] = f"ReferenceData/shadow_taco_train_split_{obj_idxx}_{obj_name}_data_opt_tag_{gt_reference_tag}.npy"
    #         else:
    #             args['sv_gt_refereces_fn'] = f"ReferenceData/shadow_taco_train_split_{obj_idxx}_{obj_name}_data.npy"
            
    #         args['obj_urdf_fn'] = f"{URDF_ROOT}/taco_{obj_idxx}_wcollision.urdf"
    #         args['canon_obj_fn'] = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
    #         args['mano_urdf_fn'] = f"{NEUS_ROOT}/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"
    #         args['left_mano_urdf_fn'] = f"{NEUS_ROOT}/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"
    #         args['conf_path'] = f"{NEUS_ROOT}/confs/dyn_grab_shadow_model_states_taco_{obj_idxx}.conf"
    #         args['tag'] = f"taco_shadow_{obj_idxx}__mpc_v3_tag_{gt_reference_tag}"
    #     else:
    #         raise NotImplementedError("Not implemented yet!")
            
            
    
    # if args['use_ana']:
    #     env = VCLODETrackEnvAna(**args)
    # else:
    if args['use_isaac']:
        print("Using isaac!")
        # from isaacgymenvs.tasks.
        env = launch_rlg_hydra()
    # else:
    elif args['policy_model'] == 'control_vae_twohands':
        env = VCLODETrackEnvAnaTwoHands(**args)   
    else:
        env = VCLODETrackEnv(**args)   
    
    
    if args['policy_model'] == 'control_vae':
        PolicyAgent = control_vae_hoi_wana.PolicyAgent
    elif args['policy_model'] == 'control_vae_twohands':
        PolicyAgent = control_vae_hoi_wana_twohands.PolicyAgent
    else:
        raise NotImplementedError
    
    nn_obs = env.bullet_mano_num_joints + 3 + 4

    
    nn_act = env.bullet_mano_num_joints
    nn_delta = env.bullet_mano_num_joints + 3 + 3
    
    

    keys = ['mano_trans', 'mano_rot', 'mano_states', 'obj_rot', 'obj_trans']
    for k in keys:
        args[f'world_model_weight_{k}'] = 1.0
        args[f'policy_weight_{k}'] = 1.0
        
    args['bullet_mano_num_joints'] = env.bullet_mano_num_joints
    args['bullet_mano_finger_num_joints'] = env.bullet_mano_finger_num_joints
    
    
    PolicyAgent(
        observation_size=nn_obs,
        action_size=nn_act,
        delta_size=nn_delta,
        env=env,
        diffusion=None,
        **args,
    ).run_loop()
    
    # python train/train_policy_hoi.py