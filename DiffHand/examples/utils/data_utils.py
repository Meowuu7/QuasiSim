import taichi as ti
import math
import torch
# from ..utils import Timer
import numpy as np
import torch.nn.functional as F
import os

import argparse


ACTIVE_OBJ_TYPE = "active"
PASSIVE_OBJ_TYPE = "passive"
ACTIVE_PASSIVE_OBJ_TYPE = "active_passive"


def create_arguments():
    # parser = argparse.ArgumentParser(
    #                 prog='mpmlangsimulator',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gui', '--ngui', action='store_true',  # activate the argument to disable gui #
                    help='whether to use gui in the program')
    parser.add_argument('-c', '--use_cuda', action='store_true', # activate the argument to disable gui #
                    help='whether to use gui in the program')
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
    # optim_act
    parser.add_argument("-a", "--optim_act", help="increase output verbosity",
                    action="store_true")
    parser.add_argument("-t", "--task", type=str, default="tracking")
    # args = parser.parse_args() # num_frames
    parser.add_argument("-nf", "--num_frames", type=int, default=200)
    parser.add_argument("-ns", "--n_substeps", type=int, default=30)
    parser.add_argument("-ex", "--export_pcs", action='store_true', help='whether to export pcs')
    parser.add_argument("-to", "--draw_optimized", action='store_true', help='whether to export pcs')
    parser.add_argument("-mu", "--mu", type=float, default=1666666.6666666667)
    parser.add_argument("-op", "--optimize_params", action='store_true', help='whether to export pcs')
    # --optim_act --task="reach_goal"
    parser.add_argument("-s", "--sv_root", type=str, default="tmp")
    parser.add_argument("-act", "--act_fn", type=str, default="")
    parser.add_argument("-tag", "--tag", type=str, default="")
    
    # general args
    parser.add_argument("-optim", "--optim", action='store_true', help='the optimization process or the inverse reconstruction process')
    parser.add_argument("-load_target_iter", "--load_target_iter", type=int, default=61)

    # rendering model
    parser.add_argument("-view_divide", "--view_divide", type=float, default=1.)
    parser.add_argument("-res", "--res", type=int, default=512)
    parser.add_argument("-dx", "--dx", type=float, default=0.02)
    parser.add_argument("-density_res", "--density_res", type=int, default=128)
    parser.add_argument("-n_views", "--n_views", type=int, default=7)
    parser.add_argument("-torus_r1", "--torus_r1", type=float, default=0.4)
    parser.add_argument("-torus_r2", "--torus_r2", type=float, default=0.1)
    parser.add_argument("-fov", "--fov", type=int, default=1)
    parser.add_argument("-camera_origin_radius", "--camera_origin_radius", type=float, default=1)
    parser.add_argument("-marching_steps", "--marching_steps", type=int, default=1000)
    parser.add_argument("-learning_rate", "--learning_rate", type=float, default=15)
    parser.add_argument("-mu_scale_factor", "--mu_scale_factor", type=float, default=0.5)
    # dynamical model
    parser.add_argument("-x_goal", "--x_goal", type=float, default=10.5)
    parser.add_argument("-p_mass", "--p_mass", type=float, default=1.)
    parser.add_argument("-dt", "--dt", type=float, default=1e-4)
    parser.add_argument("-g", "--gravity", type=float, default=9.8)
    
    parser.add_argument("--model", type = str, default = 'finger_torque')
    parser.add_argument('--gradient', action = 'store_true')
    parser.add_argument("--record", action = "store_true")
    parser.add_argument("--render", action = "store_true")
    
    parser.add_argument('--passive_only', action = 'store_true')
    
    # obj_type
    parser.add_argument("-obj_type", "--obj_type", type=str, default="active")
    parser.add_argument("-dim", "--dim", type=int, default=3)
    # tet_grid_res
    parser.add_argument("-tet_grid_res", "--tet_grid_res", type=int, default=64)
    parser.add_argument("-tet_scale", "--tet_scale", type=float, default=2.0)
    parser.add_argument("-tet_render_type", "--tet_render_type", type=str, default="neural_render")
    parser.add_argument("-initial_radius", "--initial_radius", type=float, default=2.0)
    # use_loss_type
    parser.add_argument("-use_loss_type", "--use_loss_type", type=str, default="rendering")
    parser.add_argument("-optimized_u_fn", "--optimized_u_fn", type=str, default="")
    parser.add_argument("-use_grid_loss_type", "--use_grid_loss_type", type=str, default="density")
    parser.add_argument("-optim_target", "--optim_target", type=str, default="states")
    parser.add_argument("-n_frames", "--n_frames", type=int, default=10)
    
    parser.add_argument('--use_three_planes', action = 'store_true')
    
    parser.add_argument("--img_sv_root", type = str, default = './save_res')
    # nn_rendering_poses
    parser.add_argument("-nn_rendering_poses", "--nn_rendering_poses", type=int, default=7)
    parser.add_argument("--cam_configs", type = str, default = 'routine_1') ## routine_1 or routine_2 here for camera view selection ##
    return parser

