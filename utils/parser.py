from argparse import ArgumentParser
import argparse
import os
import json
import yaml

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file: {args_path} was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--wandb", action="store_true", help="Whether use wandb for logging.")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=3, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--mini_batch", default=64, type=int, help="Mini batch size during training")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--world_window", default=8, type=int,
                       help="Number of world model window")
    group.add_argument("--policy_window", default=24, type=int,
                       help="Number of policy model window")
    group.add_argument("--weight_pos", default=0.2, type=float)
    group.add_argument("--weight_rot", default=0.1, type=float)
    group.add_argument("--weight_vel", default=0.5, type=float)
    group.add_argument("--weight_avel", default=0.5, type=float)
    group.add_argument("--weight_height", default=1.2, type=float)
    group.add_argument("--weight_up_dir", default=3, type=float)
    group.add_argument("--weight_action_l1", default=0.01, type=float)
    group.add_argument("--weight_action_l2", default=0.001, type=float)


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_motion_training_options(parser):
    group = parser.add_argument_group('motion_training')
    group.add_argument('--save_dir', default='results/', type=str, help='Path to save checkpoints and results.')
    # group.add_argument("--save_dir", required=True, type=str,
    #                    help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=11, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='results/', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_frames", default=11, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

def add_sampling_options_policy(parser):
    group = parser.add_argument_group('sampling_policy')
    group.add_argument("--policy_model_file", default='results/diffusionh42/model_2400.data', type=str,
                       help="Path to model_###.data file to be sampled.")
    group.add_argument("--output_path", default='results/', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=5.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_motion_training_options(parser)
    return parser.parse_args()

def add_policy_options(parser):
    group = parser.add_argument_group('policy')
    group.add_argument("--policy_model", type = str, default = "diffusion_policy", help = "The type of policy agent")
    group.add_argument("--latent_size", type = int, default = 64, help = "dim of latent space")
    group.add_argument("--max_iteration", type = int, default = 50001, help = "iteration for controlVAE training")
    group.add_argument("--collect_size", type = int, default = 2048, help = "number of transition collect for each iteration")
    group.add_argument("--sub_iter", type = int, default = 8, help = "num of batch in each iteration")
    group.add_argument("--save_period", type = int, default = 500, help = "save checkpoints for every * iterations")
    group.add_argument("--evaluate_period", type = int, default = 50, help = "save checkpoints for every * iterations")
    group.add_argument("--replay_buffer_size", type = int, default = 50000, help = "buffer size of replay buffer")

def add_diffusion_policy_options(parser):
    group = parser.add_argument_group('diffusion_policy')
    group.add_argument("--obs_horizon", default=8, type=int)
    group.add_argument("--pred_horizon", default=8, type=int)

def add_env_options(parser):
    group = parser.add_argument_group('track_env')
    group.add_argument("--env_contact_type", default=0, type=int, help="contact type, 0 for LCP and 1 for maxforce")
    group.add_argument("--env_close_self_collision", default=False, help="flag for closing self collision", action = 'store_true')
    group.add_argument("--env_min_length", default=26, type=int, help="episode won't terminate if length is less than this")
    group.add_argument("--env_max_length", default=1024, type=int, help="episode will terminate if length reach this")
    group.add_argument("--env_err_threshod", default = 0.5, type = float, help="height error threshod between simulated and tracking character")
    group.add_argument("--env_err_length", default = 20, type = int, help="episode will terminate if error accumulated ")
    group.add_argument("--env_scene_fname", default = "odecharacter_scene.pickle", type = str, help="pickle file for scene")
    group.add_argument("--env_fps", default = 50, type = int, help="fps of control policy")
    group.add_argument("--env_substep", default = 6, type = int, help="substeps for simulation")
    group.add_argument("--env_no_balance", default = False, help="whether to use distribution balance when choose initial state", action = 'store_true')
    group.add_argument("--env_use_com_height", default = False, help="if true, calculate com in terminate condition, else use head height", action = 'store_true')
    # group.add_argument("--motion_dataset", default = "datasets/HumanML3D/sfu.pickle", type = str, help="path to motion dataset")
    group.add_argument("--motion_dataset", default = "Data/ReferenceData/binary_data/runwalkjumpgetup.pickle", type = str, help="path to motion dataset")
    # group.add_argument("--data_folder", default = "Data/ReferenceData/runwalkjumpgetup", type = str, help="path to motion dataset in bvh")
    group.add_argument("--data_json", default=None, type=str)
    group.add_argument("--env_recompute_velocity", default = True, help = "whether to resample velocity")
    group.add_argument("--env_random_count", default=96, type=int, help = "target will be random switched for every 96 frame")

def add_policy_training_options(parser):
    group = parser.add_argument_group('policy_training')
    group.add_argument('--load_dir', default=None, type=str, help='Path to load checkpoints and results.')
    group.add_argument('--save_dir', default='results2/', type=str, help='Path to save checkpoints and results.')
    group.add_argument("--diffusion_steps", default=100, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--actor_activation", default='ELU', type=str)
    group.add_argument("--actor_gate_hidden_layer_size", default=64, type=int)
    group.add_argument("--actor_hidden_layer_num", default=3, type=int)
    group.add_argument("--actor_hidden_layer_size", default=512, type=int)
    group.add_argument("--actor_num_experts", default=6, type=int)
    group.add_argument("--bvh_folder", default='./Data/ReferenceData/runwalkjumpgetup', type=str)
    group.add_argument("--policy_batch_size", default=512, type=int)
    group.add_argument("--policy_lr", default=1.0e-05, type=float)
    group.add_argument("--diffusion_lr", default=1.0e-04, type=float)
    group.add_argument("--denoising_steps", default=10, type=int)
    group.add_argument("--policy_rollout_length", default=24, type=int)
    group.add_argument("--policy_weight_avel", default=0.5, type=float)
    group.add_argument("--policy_weight_height", default=1.2, type=float)
    group.add_argument("--policy_weight_l1", default=0.01, type=float)
    group.add_argument("--policy_weight_l2", default=0.001, type=float)
    group.add_argument("--policy_weight_pos", default=0.2, type=float)
    group.add_argument("--policy_weight_rot", default=0.1, type=float)
    group.add_argument("--policy_weight_up_dir", default=3, type=int)
    group.add_argument("--policy_weight_vel", default=0.5, type=float)
    group.add_argument("--cpu_b", default=34, type=int)
    group.add_argument("--cpu_e", default=44, type=int)
    group.add_argument("--encoder_activation", default='ELU', type=str)
    group.add_argument("--encoder_fix_var", default=0.3, type=float)
    group.add_argument("--encoder_hidden_layer_num", default=2, type=int)
    group.add_argument("--encoder_hidden_layer_size", default=512, type=int)
    group.add_argument("--experiment_name", default='add_prior', type=str)
    group.add_argument("--world_model_activation", default='ELU', type=str)
    group.add_argument("--world_model_batch_size", default=512, type=int)
    # group.add_argument("--world_model_hidden_layer_num", default=4, type=int)
    # group.add_argument("--world_model_hidden_layer_size", default=512, type=int)
    group.add_argument("--world_model_hidden_layer_num", default=10, type=int)
    group.add_argument("--world_model_hidden_layer_size", default=1024, type=int)
    group.add_argument("--world_model_lr", default=0.002, type=float)
    group.add_argument("--world_model_rollout_length", default=8, type=int)
    group.add_argument("--world_model_weight_avel", default=4, type=int)
    group.add_argument("--world_model_weight_pos", default=1, type=int)
    group.add_argument("--world_model_weight_rot", default=1, type=int)
    group.add_argument("--world_model_weight_vel", default=4, type=int)
    group.add_argument("--wana", default=False,  action="store_true",  help = "whether to resample velocity")
    group.add_argument("--mano_urdf_fn", default='/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled.urdf', type=str)
    # left_mano_urdf_fn
    group.add_argument("--left_mano_urdf_fn", default='/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled.urdf', type=str)
    group.add_argument("--obj_urdf_fn", default='/home/xueyi/diffsim/NeuS/rsc/mano/redmax_box_test_3_wcollision.urdf', type=str)
    # sv_gt_refereces_fn
    group.add_argument("--sv_gt_refereces_fn", default='/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/bullet_hoi_data.npy', type=str)
    group.add_argument("--bullet_nn_substeps", default=100, type=int)
    group.add_argument("--tag", default='', type=str)
    group.add_argument("--wnorm", default=False,  action="store_true",  help = "whether to resample velocity")
    # conf_path
    group.add_argument("--conf_path", default="/home/xueyi/diffsim/NeuS/confs/dyn_arctic_robohand_from_mano_model_rules_actions_f2_diffhand_v4.conf", type=str)
    group.add_argument("--use_ana", default=False,  action="store_true",  help = "whether to resample velocity")
    # traj_opt
    group.add_argument("--traj_opt", default=False,  action="store_true",  help = "whether to resample velocity")
    group.add_argument("--nn_traj_samples", default=512, type=int)
    # point_features
    group.add_argument("--point_features", default=False,  action="store_true",  help = "whether to resample velocity")
    group.add_argument("--canon_obj_fn", default='/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes/camera.obj', type=str)
    # wnorm_obs
    group.add_argument("--wnorm_obs", default=False,  action="store_true",  help = "whether to resample velocity")
    # train_res_contact_forces
    group.add_argument("--train_res_contact_forces", default=False,  action="store_true",  help = "whether to resample velocity")
    
    # penetration_proj_k_to_robot = 4000000000.0
    group.add_argument("--penetration_proj_k_to_robot", default=4000000000.0, type=float)
    # plane_z
    group.add_argument("--plane_z", default=-0.045, type=float)
    group.add_argument("--reset_mano_states", default=False,  action="store_true", )
    # only_residual
    group.add_argument("--only_residual", default=False,  action="store_true", )
    # penetration_proj_k_to_robot_friction = 1000000.0
    group.add_argument("--penetration_proj_k_to_robot_friction", default=1000000.0, type=float)
    # use_mano_delta_states
    group.add_argument("--use_mano_delta_states", default=False,  action="store_true", )
    # pred_delta_offsets
    group.add_argument("--pred_delta_offsets", default=False,  action="store_true", )
    # finger_state_action_sigma
    group.add_argument("--finger_state_action_sigma", default=0.01, type=float)
    # use_contact_region_pts
    group.add_argument("--use_contact_region_pts", default=False,  action="store_true", )
    # use_contact_network
    group.add_argument("--use_contact_network", default=False,  action="store_true", )
    # use_preset_inertia
    group.add_argument("--use_preset_inertia", default=False,  action="store_true", )
    # angular_damping_coef
    group.add_argument("--angular_damping_coef", default=1.0, type=float)
    # damping_coef
    group.add_argument("--damping_coef", default=100.0, type=float)
    # use_optimizable_timecons
    group.add_argument("--use_optimizable_timecons", default=False,  action="store_true",)
    # use_ambient_contact_net
    group.add_argument("--use_ambient_contact_net", default=False,  action="store_true",)
    # whether to use cmaes ##
    group.add_argument("--use_cmaes", default=False,  action="store_true",)
    # abd_tail_frame_nn
    group.add_argument("--abd_tail_frame_nn", default=100, type=int)
    # friction_coef
    group.add_argument("--friction_coef", default=10.0, type=float)
    # gravity_scale
    group.add_argument("--gravity_scale", default=0.0, type=float)
    # add_table
    group.add_argument("--add_table", default=False,  action="store_true",)
    # table_height
    group.add_argument("--table_height", default=0.0, type=float)
    # table_x
    group.add_argument("--table_x", default=0.0, type=float)
    # table_y
    group.add_argument("--table_y", default=0.0, type=float)
    # train_wm_only
    group.add_argument("--train_wm_only", default=False,  action="store_true",)
    # pred_tar_mano_states
    group.add_argument("--pred_tar_mano_states", default=False,  action="store_true",)
    # load_ckpt
    group.add_argument("--load_ckpt", default='', type=str) 
    # train_policy_only
    group.add_argument("--train_policy_only", default=False,  action="store_true",)
    # two_hands
    group.add_argument("--two_hands", default=False,  action="store_true",)
    # kinematic_mano_gt_sv_fn
    group.add_argument("--kinematic_mano_gt_sv_fn", default='', type=str) 
    group.add_argument("--use_optimized_obj", default=False,  action="store_true",)
    # pred_mano_obj_states_woana
    group.add_argument("--pred_mano_obj_states_woana", default=False,  action="store_true",)
    # train_visualfeats_policy
    group.add_argument("--train_visualfeats_policy", default=False,  action="store_true",)
    # use_isaac
    group.add_argument("--use_isaac", default=False,  action="store_true",)
    ## 
    group.add_argument("--get_data_fns_from_objinfo", default=False,  action="store_true",)
    # dataset_type #
    group.add_argument("--dataset_type", default='grab', type=str)  ### choices: ['grab', 'taco', 'arctic'] ###
    group.add_argument("--obj_name", default='', type=str)  #
    group.add_argument("--obj_idxx", default='', type=str)  #
    group.add_argument("--gt_reference_tag", default='', type=str)  #
    # use_base_sim_mpc ##
    group.add_argument("--use_base_sim_mpc", default=False,  action="store_true",)
    # soft_thres
    group.add_argument("--soft_thres", default=0.0, type=float)
    


def train_policy_args():
    parser = ArgumentParser()
       
    parser.add_argument("--wandb", action="store_true", help="Whether use wandb for logging.")
    
    add_env_options(parser)
    add_policy_options(parser)
    add_diffusion_options(parser)
    add_diffusion_policy_options(parser)
    add_policy_training_options(parser)
    
    args = parser.parse_args()
    
    return vars(args)

def generate_policy_args():
    parser = ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Whether use wandb for logging.")
    add_env_options(parser)
    add_policy_options(parser)
    add_diffusion_options(parser)
    add_diffusion_policy_options(parser)
    add_policy_training_options(parser)
    add_sampling_options_policy(parser)
    
    args = parser.parse_args()
    
    return vars(args)

def generate_motion_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def generate_combined_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_env_options(parser)
    add_policy_options(parser)
    add_diffusion_policy_options(parser)
    add_policy_training_options(parser)
    add_sampling_options_policy(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')
    
    return args

def generate_args_policy():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options_policy(parser)
    add_generate_options(parser)
    return parser.parse_args()
    # args = parse_and_load_from_model(parser)
    # cond_mode = get_cond_mode(args)

    # if (args.input_text or args.text_prompt) and cond_mode != 'text':
    #     raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    # elif (args.action_file or args.action_name) and cond_mode != 'action':
    #     raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    # return args

def eval_args():
    parser = ArgumentParser()
    add_env_options(parser)
    add_policy_options(parser)
    add_diffusion_options(parser)
    add_diffusion_policy_options(parser)
    add_policy_training_options(parser)
    add_sampling_options_policy(parser)
    
    parser.add_argument("--wandb", action="store_true", help="Whether use wandb for logging.")
    parser.add_argument("--noise", default=0, type=float)
    parser.add_argument("--result_dir", required=True, type=str)
    parser.add_argument("--test_dir", default='Eval/motions/humanml_test', type=str)
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()
    
    return vars(args)

def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)