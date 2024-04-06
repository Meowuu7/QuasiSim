'''
test forward and rendering
'''
from utils.renderer import SimRenderer
import numpy as np
import redmax_py as redmax
import os
import argparse
import time
import scipy.optimize

import torch.nn as nn
import torch


import math

from utils.volume_render_deformation_th import VolumeRender
from examples.utils import volume_render_deformation as volume_render
from imageio import imwrite, imread
import utils.load_utils_th as load_utils
from examples.utils import data_utils as data_utils
from examples.utils.data_utils import *

import random
import taichi as ti

import utils.deftet_geometry as deftet_geometry
import trimesh
from nerf import DirectVoxGO, get_rays_of_a_view, batch_indices_generator

import engine.reconstructor_dvgo as recon_utils


# 
if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Test forward and rendering of simulation')
    # parser.add_argument("--model", type = str, default = 'finger_torque')
    # parser.add_argument('--gradient', action = 'store_true')
    # parser.add_argument("--record", action = "store_true")
    # parser.add_argument("--render", action = "store_true")
    real = ti.f32
    ti.init(default_fp=real, arch=ti.cuda, device_memory_fraction=0.2) # f
    torch.cuda.set_device(0)
    
    th_cuda_idx = 0
    dim = 3
    grid_res = 128
    xyz_min = [0.] * dim
    # xyz_max = [self.grid_res] * self.dim
    xyz_max = [1.] * dim
    # selxyz_min = xyz_min
    # self.xyz_max = xyz_max ## 
    xyz_min_th = torch.tensor(xyz_min, dtype=torch.float32).cuda(th_cuda_idx)
    xyz_max_th = torch.tensor(xyz_max, dtype=torch.float32).cuda(th_cuda_idx)
    alpha_init = 1e-4
    fast_color_thres=1e-7
    rgbnet_direct=True
    rgbnet_depth=3
    rgbnet_width=128
    rgbnet_dim=12
    viewbase_pe=4
    mask_thres=1e-3
    dx = 1. / float(grid_res)
    base_dir = None
    ### nerf marcher that convert the input grids to the final images ### 
    nerf = DirectVoxGO(xyz_min=xyz_min, xyz_max=xyz_max, # xyz min; xyz max #
                            alpha_init=alpha_init, fast_color_thres=fast_color_thres,
                            rgbnet_dim=rgbnet_dim, rgbnet_direct=rgbnet_direct, 
                            rgbnet_depth=rgbnet_depth, rgbnet_width=rgbnet_width, # rbgnet depth; rgbnet width #
                            viewbase_pe=viewbase_pe, 
                            mask_thres=mask_thres,
                            dx=dx, base_dir=base_dir)
    recon_utils.test_tot_forward_stepping_with_rgb(nerf=nerf)
    
    print("After stepping!!!")
    
    
    # ti.init(default_fp=real, arch=ti.cpu)
    # ti.init(default_fp=real, arch=ti.cuda, kernel_profiler=True, device_memory_fraction=0.9)
    # ti.init(device_memory_fraction=0.9)
    ti.init(default_fp=real, arch=ti.cuda, device_memory_fraction=0.5)
        
    parser = data_utils.create_arguments()
    
    
    args = parser.parse_args()
    # args.image_sv_folder = "/data1/sim/diffsim/DiffRedMax/save_res"
    # args.image_sv_folder = "./save_res/hand_sphere_demo"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target"
    args.image_sv_folder = "./save_res/hand_sphere_demo_optim_test"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_point_tracking"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_point_tracking_test"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_v2" ## a special joint value for the finger root joint #
    args.image_sv_folder = f"{args.image_sv_folder}_obj_type_{args.obj_type}_target_v2" #### target_v2 ####
    if args.model[-4:] == '.xml':
        pure_model_name = args.model[:-4]
    else:
        pure_model_name = args.model
    args.image_sv_folder = f"./save_res/goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}" ## obj type would affect saved images ##
    args.image_sv_folder = f"./save_res/goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_nfr_{args.n_frames}_view_divide_{args.view_divide}_n_views_{args.n_views}_three_planes_{args.use_three_planes}_recon_dvgo"
    
    cur_img_sv_folder = f"goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_nfr_{args.n_frames}_view_divide_{args.view_divide}_n_views_{args.n_views}_three_planes_{args.use_three_planes}_recon_dvgo"
    cur_img_sv_folder = f"goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_nfr_{args.n_frames}_view_divide_{args.view_divide}_n_views_{args.n_views}_three_planes_{args.use_three_planes}_recon_dvgo_new_Nposes_{args.nn_rendering_poses}"
    cur_img_sv_folder = f"goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_nfr_{args.n_frames}_view_divide_{args.view_divide}_n_views_{args.n_views}_three_planes_{args.use_three_planes}_recon_dvgo_new_Nposes_{args.nn_rendering_poses}_cam_configs_{args.cam_configs}"
    cur_img_sv_folder = f"goal_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_nfr_{args.n_frames}_view_divide_{args.view_divide}_n_views_{args.n_views}_three_planes_{args.use_three_planes}_recon_dvgo_new_Nposes_{args.nn_rendering_poses}_cam_configs_{args.cam_configs}_tag_{args.tag}"
    
    args.image_sv_folder = os.path.join(args.img_sv_root, cur_img_sv_folder)
    
    # 
    os.makedirs(args.image_sv_folder, exist_ok=True)
    args.num_steps = 1000
    args.mod_sv_timesteps = 500
    # args.mod_sv_timesteps = 1000 # 1000
    args.use_deformation = False
    args.use_deformation = True
    args.use_kinematic = False
    args.th_cuda_idx = 0
    args.use_loss_type = "point_tracking" ## ["point_tracking", "rendering"] ### point tracking; point tracking ###
    # args = parser.parse_args() #

    ## ##
    asset_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))

    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')

    sim = redmax.Simulation(model_path) ## hand sphere model path ##

    nn_render_steps = args.n_frames
    num_steps = 1000
    nn_mod_steps = num_steps // nn_render_steps
    
    ### set basic timestepping arguments ###
    args.mod_sv_timesteps = nn_mod_steps
    args.num_steps = num_steps
    args.n_timesteps = nn_render_steps
    

    sim.reset(backward_flag = args.gradient) # reset the simulation to start a new trajectory

    ndof_u = sim.ndof_u # number of actions
    ndof_r = sim.ndof_r # number of degrees of freedom
    ndof_var = sim.ndof_var # number of auxiliary variables
    
    args.n_act_states = ndof_u
    args.n_passive_states = ndof_r - ndof_u
    args.active_robot_name = "active" ## active ##
    args.passive_robot_name = "passive" ## passive ##
    
    
    loss = 0.
    df_du = np.zeros(ndof_u * num_steps)
    df_dq = np.zeros(ndof_r * num_steps)
    
    print(f"ndof_u: {ndof_u}, ndof_r: {ndof_r}, ndof_var: {ndof_var}")

    tot_qs = []
    t0 = time.time()
    u = np.zeros(ndof_u * num_steps)
    for i in range(num_steps):
        sim.set_u(np.zeros(ndof_u))
        # sim.set_u((np.random.rand(ndof_u) - 0.5) * 2.)
        # sim.set_u(np.ones(ndof_u))
        # sim.set_u(np.ones(ndof_u) * np.sin(i / 100 * np.pi))
        # if i < 50:
        #     sim.set_u(np.ones(ndof_u) * -1)
        # else:
        #     sim.set_u(np.ones(ndof_u))
        sim.forward(1, verbose = False)
        q = sim.get_q()
        if (i + 1) % nn_mod_steps == 0:
            tot_qs.append(q)
        loss += np.sum(q)
        df_dq[ndof_u * i:ndof_u * (i + 1)] = 1.
        
        # print(f"state of the sphere at {i}-th step: {q[-3:]}")
        # draw the geometry at each timestep here? 
    print(f'q = {q}') # chagne the goal of it ###
    
    # last_q_state = """-0.42442816 -0.42557961 -0.40366201 -0.3977891 -0.40947627 -0.4201424 -0.3799285 -0.3808375 -0.37953552 -0.42039598 -0.4058405 -0.39808804 -0.40947487 -0.42012458 -0.41822534 -0.41917521 -0.4235266 -0.87189658 -1.42093761 0.21977979"""
    # last_q_state = last_q_state.split(" ")
    # last_q_state = [float(cur_q) for cur_q in last_q_state]
    # last_q_state = np.array(last_q_state, dtype=np.float32)
    # tot_qs[0] = last_q_state
    
    # tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
    # # forward 1) kienmatic state transformation; 2) rasterization; and 3) the rendering process ## 
    # # backward 1) ---> need target images; 2) need gradients; 3) needs losses and the way of getting supervision from pixels # 
    # tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
    # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    # robots = load_utils.parse_data_from_xml(xml_fn, args)
    # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
    
    
    # target_image_sv_fn = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target/rendered_images_optim_False_passive_False_nv_7_test.npy"
    # reconstructor.load_target_rendered_images(target_image_sv_fn)
    # print(f"after setting target images...")
    
    # target_act_xs_fn = ""
    # reconstructor.load_target_act_xs(target_act_xs_fn) ### load the target act xs 
    
    # reconstructor.set_states(tot_qs)
    # reconstructor.forward_stepping()
    
    # if args.use_loss_type == "rendering": # ["rendering", "point_tracking"]
    #     reconstructor.forward_p2g()
    #     reconstructor.forward_ti_rendering()
    #     print(f"start backwarding")
    #     reconstructor.backward_stepping()
    # elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"]
    #     reconstructor.forward_tensor_to_ti()
    #     reconstructor.backward_stepping_tracking_target()
    # else:
    #     raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
    
    
    # reconstructor.forward_p2g()
    # # reconstructor.forward_rendering()
    # reconstructor.forward_ti_rendering()
    
    # print(f"start backwarding")
    # reconstructor.backward_stepping()
    
    # print(f"Start initializing...")
    # reconstructor.initialize()
    
    
    ## target q and the q ##
    ## loss; gradients; optimization ##
    #### optimization -> if we use a simple loss formulation #### per point at each frame -> per point tracking problem ###
    #### reduce the difficulty of the problem them? #### -> if there is not too much self collisions or not too much collisions ###
    
    # i_iter = 0
    # def loss_and_grad_ref_states(u): ## loss and grad ref states ##
    #     global i_iter, args
    #     sim.reset(True)
    #     tot_qs = []
    #     for i in range(num_steps):
    #         sim.set_u(u[ndof_u * i : ndof_u * (i + 1)])
    #         sim.forward(1)
    #         if (i + 1) % nn_mod_steps == 0:
    #             tot_qs.append(sim.get_q()) ## sim get q ##
    #         # tot_qs.append(sim.get_q())
    #     q = sim.get_q()
        
    #     tot_qs = np.stack(tot_qs, axis=0) ### tot_qs for the qs ###
    #     tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
    #     reconstructor.set_states(tot_qs)
    #     reconstructor.forward_stepping()
        
    #     if args.use_loss_type == "rendering": # ["rendering", "point_tracking"]
    #         reconstructor.forward_p2g()
    #         reconstructor.forward_ti_rendering()
    #         print(f"start backwarding")
    #         reconstructor.backward_stepping()
    #     elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"]
    #         reconstructor.forward_tensor_to_ti()
    #         reconstructor.backward_stepping_tracking_target()
    #     else:
    #         raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
    
    #     # reconstructor.forward_p2g()
    #     # reconstructor.forward_ti_rendering()
    #     # reconstructor.backward_stepping()
        
        
    #     df_dq = np.zeros(ndof_r * num_steps)
    #     # df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
    #     for i_s in range(reconstructor.n_timesteps):
    #         cur_st_idx = (i_s + 1) * nn_mod_steps - 1
    #         df_dq[cur_st_idx: cur_st_idx + ndof_r] = reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy()
    #     f = reconstructor.tot_cd_loss
        
    #     f = f * 1000
    #     ### iter and the final qs ###
    #     print(f"iter: {i_iter}, q: {q}, grad: {reconstructor.state_vals.grad.data[-1].detach().cpu().numpy()}")
    #     sim.backward_info.set_flags(False, False, False, True) 
    #     sim.backward_info.df_du = np.zeros(ndof_u * num_steps)
        
    #     # df_dq = np.zeros(ndof_r * num_steps) 
    #     # df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
    #     # df_dq[-ndof_r: -(ndof_r - ndof_u)] = 2 * (q - init_q_goal)[: -(ndof_r - ndof_u)]
    #     ### dense states supervision ####
    #     # tot_qs = np.concatenate(tot_qs, axis=0)
    #     # df_dq = 2 * tot_qs ### tot_q
    #     # f = np.sum(tot_qs ** 2)
        
    #     print(f"iter: {i_iter}, f: {f}")
        
    #     sim.backward_info.df_dq = df_dq
        
    #     sim.backward()
    #     grad = np.copy(sim.backward_results.df_du)
        
    #     reconstructor.initialize()
        
    #     i_iter += 1
    #     return f, grad

    
    # def callback_func_ref_states(u, render=False):
    #     sim.reset(False)
        
    #     tot_qs = []
    #     for i in range(num_steps):
    #         sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
    #         sim.forward(1)
    #         q = sim.get_q()
    #         if (i + 1) % nn_mod_steps == 0:
    #             tot_qs.append(q)
        
    #     tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
        
    #     save_q_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_q_iter_{i_iter}.npy")
    #     np.save(save_q_res_fn, tot_qs)
    #     print(f"active_only q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn

    #     q = sim.get_q()
        
    #     print('q = ', q)
    #     if render:
    #         print('q = ', q)
    #         SimRenderer.replay(sim, record = True, record_path = "./optim_only_active_states.gif")
    #     return
    
    # res = scipy.optimize.minimize(loss_and_grad_ref_states, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func_ref_states)

    # callback_func_ref_states(u = res.x, render = args.render)
    
    # if args.render:
    #     SimRenderer.replay(sim, record = True, record_path = "./optim_initial.gif")
        
    # u[11] = 1e3
    # for i in range(num_steps):
    #     u[ndof_u * i + 11: ndof_u * i + 14] = 1e5
    #     # u[ndof_u * i + 6: ndof_u * i + 10] = 1e3
    #     # setting a good initilziation; but i still do not know what happens here... ## 
    #     ## q[-1] is the rotation variable? q[-2] is the direction along the gravity direction ##
    #     ## q[-3] is the direction along the forwarding x direction ###
    
    ####### Set initial states via PD controlls #######
    sim.reset(False)
    q_goal = np.zeros(ndof_r - 3) - 0.001
    P_q = np.ones(ndof_r - 3) *0.001
    u = np.zeros(ndof_u * num_steps)
    # ## set the initial value of q ##
    for i in range(num_steps):
        # if i % 1 == 0:
        #     print(f"step: {i}")
        q = sim.get_q()
        error = q_goal - q[:-3]
        ui = error * P_q
        u[i * ndof_u: (i + 1) * ndof_u] = ui
        sim.set_u(ui) # rigid, articulated, interactions with other objects #
        sim.forward(1, False)
    q = sim.get_q()
    print('q = ', q)
    
    # if args.render: ## and render them at different status ## ## track the active object v.s. track both the active and the passive object ##
    #     SimRenderer.replay(sim, record = True, record_path = "./optim_after_pd.gif")
    
    
    xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    robots = load_utils.parse_data_from_xml(xml_fn, args)
    robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    ### construct the recons ###
    reconstructor = recon_utils.Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot) ## reconstructor ### 
    
    
    
    recon_us = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo/active_with_goal_optim_res_u_iter_79.npy"
    recon_us = np.load(recon_us, allow_pickle=True)
    print(f"recon_us: {recon_us.shape}")
    sim.reset(False)
    tot_qs = []
    for i in range(num_steps):
        ###### set_u for us ######
        sim.set_u(recon_us[ndof_u * i : ndof_u * (i + 1)])
        sim.forward(1)
        if (i + 1) % nn_mod_steps == 0:
            tot_qs.append(sim.get_q())
    tot_qs = np.stack(tot_qs, axis=0) ## 
    tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
    reconstructor.set_states(tot_qs) # # use the states -> set states # tot stepping with rgb #
    
    reconstructor.tot_forward_stepping_with_rgb()
    reconstructor.initialize()
    exit(0)


    # init_q_goal = -1
    init_q_goal = -0.5
    init_q_goal = -4
    init_q_goal_vec = np.zeros((ndof_r, ), dtype=np.float32) ### init q goal vec ###
    init_q_goal_vec[:] = -4. # [joint1, joint3, joint7, joint11, joint15] #
    # finger_root_joint_idx = [0, 2, 6, 10, 14] #
    # finger_root_joint_idx = np.array(finger_root_joint_idx, dtype=np.int32)
    # # init_q_goal_vec[]
    
    
    
    
    i_iter = 0
    def loss_and_grad_qs(u):
        global i_iter
        sim.reset(True)
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i : ndof_u * (i + 1)])
            sim.forward(1)
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(sim.get_q())
        q = sim.get_q()
        # f = np.sum((q - init_q_goal)[: -(ndof_r - ndof_u)] ** 2) #
        f = np.sum((q - init_q_goal) ** 2) # 
        
        
        # tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
        
        # ### get qs from the tot_qs and set states for the reconstructor and forard the rendering ###
        # tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        # reconstructor.set_states(tot_qs)
        
        # reconstructor.tot_forward_stepping_with_rgb()
        
        # reconstructor.forward_stepping()
        
        ####### save the optimized pts ####### ### points ### 
        # # tot_frame_act_xs, tot_frame_passive_xs #
        # cur_optimized_act_xs = reconstructor.tot_frame_act_xs.detach().cpu().numpy() ### act xs ##
        # cur_optimized_passive_xs = reconstructor.tot_frame_passive_xs.detach().cpu().numpy() ### passive xs ###
        # cur_act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_act_xs.npy")
        # cur_passive_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_passive_xs.npy")
        # np.save(cur_act_xs_sv_fn, cur_optimized_act_xs) 
        # np.save(cur_passive_xs_sv_fn, cur_optimized_passive_xs)
        # print(f"current optimized act xs saved to {cur_act_xs_sv_fn}")
        # print(f"current optimized passive xs saved to {cur_passive_xs_sv_fn}")
        ####### save the optimized pts #######
        
        
        # reconstructor.forward_p2g() # goal optimize -> then with the dense sup optimize #
        # reconstructor.forward_ti_rendering()
        
        # reconstructor.initialize()
        
        
        print(f"iter: {i_iter}, q: {q}")
        sim.backward_info.set_flags(False, False, False, True) 
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps) ## df_du ##
        # df_dq = np.ones(ndof_r * num_steps) 
        
        df_dq = np.zeros(ndof_r * num_steps) 
        df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
        # df_dq[-ndof_r: -(ndof_r - ndof_u)] = 2 * (q - init_q_goal)[: -(ndof_r - ndof_u)]
        ### dense states supervision ####
        # tot_qs = np.concatenate(tot_qs, axis=0)
        # df_dq = 2 * tot_qs ### tot_q
        # f = np.sum(tot_qs ** 2)
        
        print(f"iter: {i_iter}, f: {f}")
        
        sim.backward_info.df_dq = df_dq
        
        sim.backward()
        grad = np.copy(sim.backward_results.df_du)
        
        i_iter += 1
        return f, grad
    
    final_u = np.zeros_like(u)
    
    
    def callback_func_qs(u, render=False):
        sim.reset(False)
        
        tot_qs = []
        for i in range(num_steps): ## 
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
            q = sim.get_q()
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(q)
        
        tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
        
        ### get qs from the tot_qs and set states for the reconstructor and forard the rendering ###
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        reconstructor.set_states(tot_qs)
        
        reconstructor.tot_forward_stepping_with_rgb()
        
        # reconstructor.forward_stepping()
        
        ####### save the optimized pts ####### ### points ### 
        # # tot_frame_act_xs, tot_frame_passive_xs #
        # cur_optimized_act_xs = reconstructor.tot_frame_act_xs.detach().cpu().numpy() ### act xs ##
        # cur_optimized_passive_xs = reconstructor.tot_frame_passive_xs.detach().cpu().numpy() ### passive xs ###
        # cur_act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_act_xs.npy")
        # cur_passive_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_passive_xs.npy")
        # np.save(cur_act_xs_sv_fn, cur_optimized_act_xs) 
        # np.save(cur_passive_xs_sv_fn, cur_optimized_passive_xs)
        # print(f"current optimized act xs saved to {cur_act_xs_sv_fn}")
        # print(f"current optimized passive xs saved to {cur_passive_xs_sv_fn}")
        ####### save the optimized pts #######
        
        
        # reconstructor.forward_p2g()
        # reconstructor.forward_ti_rendering()
        
        reconstructor.initialize()
        
        
        ##### save q res fn #####
        # save_q_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_q_iter_{i_iter}.npy")
        # np.save(save_q_res_fn, tot_qs.detach().cpu().numpy())
        # print(f"active_only q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn
        
        # save_u_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_u_iter_{i_iter}.npy")
        # np.save(save_u_res_fn, u)
        # print(f"active_only u optimized at iter {i_iter} saved to {save_u_res_fn}") ## save_q_res_fn
        ##### save q res fn #####
        
        ####### save q res fn #######
        # save_q_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_q_iter_{i_iter}.npy")
        # np.save(save_q_res_fn, tot_qs)
        # print(f"active_only q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn

        q = sim.get_q()
        
        print('q = ', q)
        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = True, record_path = "./optim_only_active_states.gif")

        # global final_u
        # final_u[:] = u[:]
        
        return
    
    # sim.reset(False)
    # export_data_folder = "./save_res/hand_sphere_demo"
    # os.makedirs(export_data_folder, exist_ok=True)
    # sim.replay()
    # # sim.export_replay(export_data_folder)
    
    res = scipy.optimize.minimize(loss_and_grad_qs, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func_qs)
    # callback_func(u = res.x, render = True)
    u[:] = res.x[:]
    
    # callback_func_qs(u = res.x, render = args.render)
        
    # sim.reset(False)
    # export_data_folder = "./save_res/hand_sphere_demo"
    # os.makedirs(export_data_folder, exist_ok=True)
    # # sim.export_replay(export_data_folder)
    # sim.replay()
    # SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) # render the simulation replay video
    
    # SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) 
    
    # u[:] = final_u[:]
    
    ## set the u to the res.x, the previous optimization result ## ## result ##
    # 
    
    y_idx = -2
    # y_idx = -3
    i_iter = 0
    y_goal =  -8  #  -3. # -8 # -5.
    def loss_and_grad(u):
        global i_iter
        sim.reset(True)
        
        tot_df_dq = np.zeros(ndof_r * num_steps)
        
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i: ndof_u * (i + 1)])
            sim.forward(1)
            cur_q = sim.get_q()
            tot_df_dq[ndof_r * (i + 1) + y_idx] = 2 * (cur_q[y_idx] - y_goal)
            
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(cur_q)
        
        tot_qs = np.stack(tot_qs, axis=0)
        
        save_q_res_fn = os.path.join(args.image_sv_folder, f"res_q_iter_{i_iter}.npy")
        np.save(save_q_res_fn, tot_qs)
        print(f"q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn
        
        save_u_res_fn = os.path.join(args.image_sv_folder, f"res_u_iter_{i_iter}.npy")
        np.save(save_u_res_fn, u)
        print(f"u optimized at iter {i_iter} saved to {save_u_res_fn}") ## save_q_res_fn
        
        q = sim.get_q()
        f = (q[y_idx] - y_goal) ** 2
        
        print(f"i_iter: {i_iter}, current_y: {q[y_idx]}, y_goal: {y_goal}, f: {f}")
        
        sim.backward_info.set_flags(False, False, False, True) 
        
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps)
        df_dq = np.zeros(ndof_r * num_steps)
        
        df_dq[ndof_r * (num_steps) + y_idx] = 2. * (q[y_idx] - y_goal)
        # df_dq[:] = tot_df_dq[:]
        # sim.backward_info.df_dq = df_dq
        
        sim.backward()
        grad = np.copy(sim.backward_results.df_du)
        # for i in range(num_steps):
        #     if np.abs(grad[i]) > 0:
        #         print(f"{i}-th step's {i  % ndof_u} torque's grad: {grad[i]}")
            # print(f"step {i}'s grad of interest:", grad[ndof_u * i + 11: ndof_u * i + 14])
        # grad[:11] = 0. 
        # grad[12:] = 0.
        # f = f *
        # grad = grad * 1e8
        i_iter += 1
        return f, grad
    
    i_iter = 0
    def callback_func(u, render = False):
        global nn_mod_steps, args, i_iter
        sim.reset(False)
        
        tot_qs = []
        for i in range(num_steps): ## 
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
            q = sim.get_q()
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(q)
        
        ### tot_qs ---> tot_qs ###
        tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ## ## the rendering equations ## ##
        
        
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        
        
        ####### save the optimized pts #######
        # tot_frame_act_xs, tot_frame_passive_xs #
        cur_optimized_act_xs = reconstructor.tot_frame_act_xs.detach().cpu().numpy() ### act xs ##
        cur_optimized_passive_xs = reconstructor.tot_frame_passive_xs.detach().cpu().numpy() ### passive xs ###
        cur_act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_with_goal_optim_act_xs.npy")
        cur_passive_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_with_goal_optim_passive_xs.npy")
        np.save(cur_act_xs_sv_fn, cur_optimized_act_xs) 
        np.save(cur_passive_xs_sv_fn, cur_optimized_passive_xs)
        print(f"current optimized act xs saved to {cur_act_xs_sv_fn}")
        print(f"current optimized passive xs saved to {cur_passive_xs_sv_fn}")
        ####### save the optimized pts ####### ## pts optimized ##
        
        
        
        reconstructor.forward_p2g()
        reconstructor.forward_ti_rendering()
        
        reconstructor.initialize()
        
        ## active optimize ##
        save_q_res_fn = os.path.join(args.image_sv_folder, f"active_with_goal_optim_res_q_iter_{i_iter}.npy")
        np.save(save_q_res_fn, tot_qs.detach().cpu().numpy())
        print(f"active_only q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn
        
        save_u_res_fn = os.path.join(args.image_sv_folder, f"active_with_goal_optim_res_u_iter_{i_iter}.npy")
        np.save(save_u_res_fn, u)
        print(f"active_only u optimized at iter {i_iter} saved to {save_u_res_fn}") ## save_q_res_fn

        
        # save_q_res_fn = os.path.join(args.image_sv_folder, f"res_q_iter_{i_iter}.npy")
        # np.save(save_q_res_fn, tot_qs)
        # print(f"q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn
        
        # save_u_res_fn = os.path.join(args.image_sv_folder, f"res_u_iter_{i_iter}.npy")
        # np.save(save_u_res_fn, u)
        # print(f"u optimized at iter {i_iter} saved to {save_u_res_fn}") ## save_q_res_fn ## and should export many 
        
        # forward 1) kienmatic state transformation; 2) rasterization; and 3) the rendering process ##
        # backward 1) ---> need target images; 2) need gradients; 3) needs losses and the way of getting supervision from pixels #
        # tot_qs = torch.from_numpy(tot_qs).float()
        # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
        # robots = load_utils.parse_data_from_xml(xml_fn, args=args)
        # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
        # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
        # print(f"tot_qs: {len(tot_qs)}, n_timesteps: {reconstructor.n_timesteps}")
        # reconstructor.set_states(tot_qs)
        # reconstructor.forward_stepping()
        # reconstructor.forward_p2g()
        # # reconstructor.forward_rendering()
        # reconstructor.forward_ti_rendering()
        # reconstructor.initialize()

        q = sim.get_q()
        print('q = ', q)
        ## render ##
        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = True, record_path = "./optim_with_goals.gif")
        pass

    res = scipy.optimize.minimize(loss_and_grad, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func) # func and with goals #



    # res = scipy.optimize.minimize(loss_and_grad_qs, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func_qs)
    # callback_func(u = res.x, render = True)
    callback_func(u = res.x, render = args.render)
    
    # server -> cannot use the render function; cannot get a satisfactory optimization result #
    # local 

    # t1 = time.time()

    # if args.gradient: # only test for the gradient ... ###
    #     sim.backward_info.set_flags(False, False, False, True)
    #     sim.backward_info.df_du = df_du
    #     sim.backward_info.df_dq = df_dq
    #     sim.backward()

    # t2 = time.time()

    # fps_forward_only = num_steps / (t1 - t0)
    # fps_with_gradient = num_steps / (t2 - t0)
    
    

    # print('FPS (forward only) = {:.1f}, FPS (with gradient) = {:.1f}'.format(fps_forward_only, fps_with_gradient))

    # SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) # render the simulation replay video

