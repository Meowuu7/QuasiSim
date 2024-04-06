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

from nerf import DirectVoxGO, get_rays_of_a_view, batch_indices_generator

# import engine.reconstructor_dvgo as recon_utils



if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Test forward and rendering of simulation')
    # parser.add_argument("--model", type = str, default = 'finger_torque')
    # parser.add_argument('--gradient', action = 'store_true')
    # parser.add_argument("--record", action = "store_true")
    # parser.add_argument("--render", action = "store_true")
    
    real = ti.f32
    ti.init(default_fp=real, arch=ti.cuda, device_memory_fraction=0.4)
    
    parser = data_utils.create_arguments()
    
    args = parser.parse_args()
    # args.image_sv_folder = "/data1/sim/diffsim/DiffRedMax/save_res"
    # args.image_sv_folder = "./save_res/hand_sphere_demo"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target"
    args.image_sv_folder = "./save_res/hand_sphere_demo_optim_test"
    args.image_sv_folder = "./save_res/hand_sphere_demo_optim_test_rendering"
    args.image_sv_folder = "./save_res/hand_sphere_demo_optim_test_rendering_optimizable_geometry"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_with_passive"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_with_passive_only_passive" # only passive #
    args.image_sv_folder = "./save_res/hand_sphere_demo_target_with_passive_optimize" # only passive #
    
    #### optimize that geometry such that; or optimize that poses ###
    ### -> change the optimization spaces v.s. chagne the optimization intermedaite targets ###
    ### -> the natural and prior losses v.s. covnerto priors (states with model -> the hand poses and the geometry)
    ### -> you have the hand poses and priors; and how to use it to affect the optimization process? # ##
    
    
    
    args.num_steps = 1000
    args.mod_sv_timesteps = 50
    args.use_deformation = False
    args.use_deformation = True
    args.use_kinematic = False
    args.th_cuda_idx = 0
    # args.use_loss_type = "point_tracking" ## ["point_tracking", "rendering"] ### point tracking ### test hand sphere optimize rgb # ##
    # args.use_loss_type = "rendering"
    # args = parser.parse_args()
    # args.optimize = True
    args.optimize = False
    
    if args.model[-4:] == '.xml':
        pure_model_name = args.model[:-4]
    else:
        pure_model_name = args.model
    # args.image_sv_folder = f"./save_res/tracking_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}_nfr_{args.n_frames}_nv_{args.n_views}_view_divide_{args.view_divide}_three_planes_{args.use_three_planes}" ## obj type would affect saved images ##
    
    image_sv_fn = f"tracking_optimize_model_{pure_model_name}_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}_nfr_{args.n_frames}_nv_{args.n_views}_view_divide_{args.view_divide}_three_planes_{args.use_three_planes}"
    args.image_sv_folder = os.path.join(args.img_sv_root, image_sv_fn)
    
    if len(args.tag) > 0:
        args.image_sv_folder = args.image_sv_folder + f"_tag_{args.tag}"
    
    # hand_sphere_demo_target_v2_obj_type_active_target_v2 #
    # if args.obj_type == ACTIVE_OBJ_TYPE: #
    #     if "joint_test" in args.model: #
    #         ###### Target v2 ######
    #         args.image_sv_folder = f"./save_res/hand_sphere_demo_target_with_passive_optimize_r2_target_v2_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}" # only passive #
    #         ###### Target v2 ######
    #     else:
    #         args.image_sv_folder = f"./save_res/hand_sphere_demo_target_with_passive_optimize_r2_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}" # only passive #
    # elif args.obj_type == ACTIVE_PASSIVE_OBJ_TYPE:
    #     # args.image_sv_folder = "./save_res/hand_sphere_demo_target_with_passive_optimize_r2" # only passive #
    #     if "joint_test" in args.model:
    #         ###### Target v2 ######
    #         args.image_sv_folder = f"./save_res/hand_sphere_demo_target_with_passive_optimize_r2_target_v2_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}" # only passive #
    #         ###### Target v2 ######
    #     else:
    #         args.image_sv_folder = f"./save_res/hand_sphere_demo_target_with_passive_optimize_r2_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}" # only passive #
    # else:
    #     args.image_sv_folder = f"./save_res/hand_sphere_demo_target_with_passive_optimize_r2_passive_target_v2_obj_type_{args.obj_type}_loss_type_{args.use_loss_type}" # only passive #
        
    os.makedirs(args.image_sv_folder, exist_ok=True)
    
    
    ## the dmtet source ##
    ''' Creating the dmtet source '''
    # ## TODO: change the args.model parameter ##
    # ## TODO: transfor mthe object to get its supervision from pixels ### pixels and supervisions #
    # deftet = create_optimizable_deftet_geometry(args=args)
    # # deftet of deftet # # args.model #
    # verts, faces, tet_verts, tets = deftet.get_tet_mesh_optimizable()
    # ## trimesh save obj ##
    # verts = verts.detach().cpu().numpy()
    # faces = faces.detach().cpu().numpy()
    # save_mesh = trimesh.Trimesh(verts, faces)
    # exported = trimesh.exchange.obj.export_obj(save_mesh)
    # target_optimizable_obj_fn = "/home/xueyi/diffsim/DiffHand/assets/optimizable_geometry/passive_obj.obj"
    # # with open(target_optimizable_obj_fn, "w") as wf: ## wf as the target optimizable obj fn ##
    # #     wf.write(target_optimizable_obj_fn)
    # #     wf.close()
    # save_obj_file(verts, faces.tolist(), target_optimizable_obj_fn, add_one=False)
    # print(f"optimizable object saved to {target_optimizable_obj_fn}") ### optimizable obj fn ###
    ####### saved to target_optimizable_obj_fn #######
    ''' Creating the dmtet source  '''


    asset_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))


    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')

    print(f"Loading sim file: {model_path}")
    sim = redmax.Simulation(model_path) # simulation model_path #

    print(f"After loading sim file")
    
    nn_render_steps = args.n_frames #  40 #  10
    num_steps =  1000
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
    args.active_robot_name = "active"
    args.passive_robot_name = "passive"
    
    # last_q_state = """-0.42442816 -0.42557961 -0.40366201 -0.3977891 -0.40947627 -0.4201424 -0.3799285 -0.3808375 -0.37953552 -0.42039598 -0.4058405 -0.39808804 -0.40947487 -0.42012458 -0.41822534 -0.41917521 -0.4235266 -0.87189658 -1.42093761 0.21977979"""
    # last_q_state = last_q_state.split(" ")
    # last_q_state = [float(cur_q) for cur_q in last_q_state]
    # last_q_state = np.array(last_q_state, dtype=np.float32)
    # tot_qs[0] = last_q_state ## # ### 
    
    
    # tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
    # # forward 1) kienmatic state transformation; 2) rasterization; and 3) the rendering process ## 
    # # backward 1) ---> need target images; 2) need gradients; 3) needs losses and the way of getting supervision from pixels # 
    # tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
    # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    # robots = load_utils.parse_data_from_xml(xml_fn, args)
    # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
    
    
    ### xml_fn; robot_name_to_robot; 
    # ## load robots and load many other things #
    xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    robots = load_utils.parse_data_from_xml(xml_fn, args) # parse data form xml #
    robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    ### construct the recons ###
    reconstructor = recon_utils.Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
    
    # ### get the reconstructo ###
    # transformed_joints = []
    # transformed_joints = robots[0].get_tot_transformed_joints(transformed_joints)
    # transformed_joints = torch.stack(transformed_joints, dim=0)
    # print(f"get transformed_joints: {transformed_joints.size()}")
    # sv_transformed_joints_fn = os.path.join("/home/xueyi/diffsim/DiffHand/examples/save_res", "hand_transformed_joints.npy")
    # np.save(sv_transformed_joints_fn, transformed_joints.detach().cpu().numpy()) ## 
    # print(f"transformed joints pts saved to {sv_transformed_joints_fn}")


    # if args.obj_type == ACTIVE_OBJ_TYPE:
    #     if "joint_test" in args.model:
    #         target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_v2_obj_type_active_target_v2"
    #     else:
    #         if nn_render_steps == 10:
    #             target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_point_tracking"
    #             if not args.use_three_planes:
    #                 if args.n_views > 7: # rbg color and the rendering # rendering #
    #                     target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5"
    #                 else:
    #                     target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7"
    #             else:
    #                 target_sv_folder = f"/home/xueyi/diffsim/DiffHand/examples/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_{args.use_three_planes}"
    #         elif nn_render_steps == 40:
    #             target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_100"
    #         else:
    #             raise ValueError(f"Targets for {nn_render_steps} frmaes have not been created yet.")
    # if args.obj_type == ACTIVE_PASSIVE_OBJ_TYPE:
    #     if "joint_test" in args.model:
    #         target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_v2_obj_type_active_passive_target_v2"
    #     else:
    #         if nn_render_steps == 10:
    #             target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_with_passive_only_passive"
    #         elif nn_render_steps == 40:
    #             target_sv_folder = "/home/xueyi/diffsim/DiffHand/examples/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_passive_nfr_100"
    #         else:
    #             raise ValueError(f"Targets for {nn_render_steps} frmaes have not been created yet.")
        
    # print(f"target_sv_folder: {target_sv_folder}")
    # print(f'target_image_sv_fn: {target_image_sv_fn}')
    # # target_image_sv_fn = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_with_passive_only_passive/rendered_images_optim_False_passive_False_nv_7_test.npy"
    # reconstructor.load_target_rendered_images(target_image_sv_fn) ## target image saving fn ##
    # print(f"after setting target images...")
    
    # if args.use_loss_type == "point_tracking":
    #     # target_act_xs_fn = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_v2_obj_type_active_target_v2/active_only_optim_act_xs.npy"
    #     # reconstructor.load_target_act_xs(target_act_xs_fn) ### load the target act xs 
    #     target_act_xs_fn = os.path.join(target_sv_folder, "active_only_optim_act_xs.npy")
    #     reconstructor.load_target_act_xs(target_act_xs_fn)
    #     target_passive_xs_fn = os.path.join(target_sv_folder, "active_only_optim_passive_xs.npy")
    #     reconstructor.load_target_passive_xs(target_passive_xs_fn)
    
    
    ## upate the xml file ###
    # reconstructor.passive_robot.children[0].body.update_xml_file()
    ## upate the xml file ###
    


    target_rgb_img_sv_fn = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo/tot_frame_rendered_rgb.npy"
    target_rgb_img_sv_fn = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo_new_Nposes_7_routine_2/tot_frame_rendered_rgb.npy"
    # acrive_passive; light_colors # TODO: what assumptions made in the light colors? #
    target_rgb_img_sv_fn = "/data2/datasets/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_passive_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo_new_Nposes_7_cam_configs_routine_2_tag_light_color/tot_frame_rendered_rgb.npy"
    
    reconstructor.load_target_rendering_img(target_rgb_img_sv_fn)
    
    target_depth_img_sv_fn = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo/tot_frame_rendered_depth.npy"
    target_depth_img_sv_fn = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo_new_Nposes_7_routine_2/tot_frame_rendered_depth.npy"
    target_depth_img_sv_fn = "/data2/datasets/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_passive_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo_new_Nposes_7_cam_configs_routine_2_tag_light_color/tot_frame_rendered_depth.npy"
    reconstructor.load_target_rendering_depth(target_depth_img_sv_fn)
    
    
    ### set the confg via the initial radius ###
    conf_fn = "/home/xueyi/diffsim/DiffHand/assets/hand_sphere_only_hand_test_with_obj_wopt.xml"
    # initial_radius = 0.2047
    initial_radius = 2.0375
    eps_radius = 1e-3
    def config_reconstructor_sim_via_target_radius(target_radius):
        global initial_radius, sim, reconstructor, conf_fn
        ### config fn ###
        load_utils.dump_robot_configurations(target_radius, conf_fn) # 
        
        print(f"Loading sim file: {conf_fn}")
        sim = redmax.Simulation(conf_fn) # simulation model_path #
        sim.reset(True) 

        reconstructor.passive_robot.children[0].body.set_radius(target_radius) ## set_radius 
        
    initial_radius = initial_radius #  + eps_radius
    ### config constructor sim via traget radius ###
    config_reconstructor_sim_via_target_radius(target_radius=initial_radius) ### 
    
    
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
        q = sim.get_q() ## loss += np.sum(q)
        if (i + 1) % nn_mod_steps == 0:
            tot_qs.append(q)
        loss += np.sum(q) # 
        df_dq[ndof_u * i:ndof_u * (i + 1)] = 1. # df_dq
        
        # print(f"state of the sphere at {i}-th step: {q[-3:]}") 
        # draw the geometry at each timestep here? 
    print(f'q = {q}')
    
    
    '''########### nothing ###########'''
    # target_act_xs_fn = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_point_tracking/act_xs_optim_False_passive_False_nv_7.npy"
    # target_act_xs_fn = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target_point_tracking/act_xs_optim_False_passive_False_nv_7.npy"
    # reconstructor.load_target_act_xs(target_act_xs_fn) ### load the target act xs ##
    
    # reconstructor.set_states(tot_qs)
    # reconstructor.forward_stepping()
    
    # ## unrecognized loss type ## ###
    # if args.use_loss_type == "rendering": # ["rendering", "point_tracking"] ### 
    #     reconstructor.forward_p2g()
    #     reconstructor.forward_ti_rendering()
    #     print(f"start backwarding")
    #     reconstructor.backward_stepping()
    # elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"] ### 
    #     reconstructor.forward_tensor_to_ti()
    #     reconstructor.backward_stepping_tracking_target()
    # else:
    #     raise ValueError(f"Unrecognized loss type: {args.use_loss_type}") # ######### #
    
    # # reconstructor.forward_p2g()
    # # # reconstructor.forward_rendering()
    # # reconstructor.forward_ti_rendering() ##
    # # ### ### # #
    
    # # print(f"start backwarding")
    # # reconstructor.backward_stepping()
    
    # print(f"Start initializing...")
    # reconstructor.initialize()
    '''########### nothing ###########'''
    # rigidity and articulated 
    # rotation and translation #
    
    
    
    if args.optim_target == "states":
        ''' optimize the state spaces '''
        ### with optimized q states ####
        tot_qs = np.stack(tot_qs, axis=0) ### tot_qs for the qs
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        for i_step in range(500): # state sequences #  # 
            
            reconstructor.set_states(tot_qs)
            reconstructor.forward_stepping()
            
            # if args.optimize:
            if args.use_loss_type == "rendering": # ["rendering", "point_tracking"] # [rendering, "point_tracking"] #
                reconstructor.forward_p2g()
                
                if args.use_grid_loss_type == "density":
                    reconstructor.forward_p2g_with_rgb()
                elif args.use_grid_loss_type == "deformation":
                    ##### foward p2g with deformation ####
                    reconstructor.forward_p2g_with_deformation()
                else:
                    raise ValueError(f"Unrecognized use_grid_loss_type: {args.use_grid_loss_type}")
                # forward p2g with deformaiton #
                # reconstructor.forward_ti_rendering() # ti rendering #
                print(f"start backwarding") # poster printing #
                # reconstructor.backward_stepping() #
            elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"] # [  ] #
                reconstructor.forward_tensor_to_ti()
                reconstructor.backward_stepping_tracking_target()
            else:
                raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
            
            cur_act_xs = reconstructor.ti_act_xs.to_numpy()
            act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_xs.npy")
            np.save(act_xs_sv_fn, cur_act_xs)
            print(f"Active xs saved to {act_xs_sv_fn}")
            
            cur_state_qs_np = tot_qs.detach().cpu().numpy()
            qs_sv_fn = os.path.join(args.image_sv_folder, f"cur_qs_{i_step}.npy")
            np.save(qs_sv_fn, cur_state_qs_np)
            print(f"Current optimized qs saved to {qs_sv_fn}")
            
            print(f"tot_cd_loss: {reconstructor.tot_cd_loss}")
            lr = 1.0 / 3. #  0.1 #  1.0
            lr = 1.0 #  1.0 / 10. #  0.1 #  1.0
            lr = 10.0 
            # lr = 5e5 # 1e6 # 1e5 #### learning rate setting ##
            print(f"q: {tot_qs[-1]}, grad: {reconstructor.state_vals.grad.data[tot_qs.size(0) - 1]}")
            print(f"q: {tot_qs[0]}, grad: {reconstructor.state_vals.grad.data[0]}")
            
            # i_s -> 
            
            for i_s in range(tot_qs.size(0)):
                tot_qs[i_s] = tot_qs[i_s] - reconstructor.state_vals.grad.data[i_s] * lr
            ## prior 1 -> non ##
            ## prior 1 -> non ##
            # prior 1 -> non-positive state increases #
            for i_s in range(1, tot_qs.size(0)):
                prev_q = tot_qs[i_s - 1]
                cur_q = tot_qs[i_s]
                
                delta_q = cur_q - prev_q
                delta_q[delta_q > 0.] = 0.
                ### non-positive state changes ###
                tot_qs[i_s] = prev_q + delta_q
            
            reconstructor.initialize() # 
            # reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy()[:-1]
        ''' optimize the state spaces '''
        exit(0)
    elif args.optim_target == "points":
        ''' optimize the state spaces '''
        ### with optimized q states ####
        tot_qs = np.stack(tot_qs, axis=0) ### tot_qs for the qs # with qs and the qs #
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx) 
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        tot_act_xs = reconstructor.ti_act_xs.to_numpy() ### get act xs ###
        for i_step in range(500): # state sequences #  # 
            # if i_step == 0:
            #     reconstructor.set_states(tot_qs)
            #     reconstructor.forward_stepping()
            reconstructor.forward_stepping_direct_from_pts(act_xs_np=tot_act_xs)
            
            # if args.optimize:
            if args.use_loss_type == "rendering": # ["rendering", "point_tracking"] #
                reconstructor.forward_p2g()
                
                if args.use_grid_loss_type == "density":
                    reconstructor.forward_p2g_with_rgb()
                elif args.use_grid_loss_type == "deformation":
                    ##### foward p2g with deformation ####
                    reconstructor.forward_p2g_with_deformation()
                else:
                    raise ValueError(f"Unrecognized use_grid_loss_type: {args.use_grid_loss_type}")
                ## 
                # forward p2g with deformaiton # # and also not a natural pose #
                # reconstructor.forward_ti_rendering() # ti rendering #
                # NeuS reconsturction -> the differences ->
                print(f"start backwarding")
                # reconstructor.backward_stepping() # 
            elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"]
                reconstructor.forward_tensor_to_ti()
                reconstructor.backward_stepping_tracking_target()
            else:
                raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
            
            cur_act_xs = reconstructor.ti_act_xs.to_numpy()
            act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_xs.npy")
            np.save(act_xs_sv_fn, cur_act_xs)
            print(f"Active xs saved to {act_xs_sv_fn}")
            
            # Mesh reconstruction and NeuS
            # cur_state_qs_np = tot_qs.detach().cpu().numpy()
            # qs_sv_fn = os.path.join(args.image_sv_folder, f"cur_qs_{i_step}.npy")
            # np.save(qs_sv_fn, cur_state_qs_np)
            # print(f"Current optimized qs saved to {qs_sv_fn}")
            
            print(f"tot_cd_loss: {reconstructor.tot_cd_loss}")
            lr = 1.0 / 3. #  0.1 #  1.0
            lr = 1.0 #  1.0 / 10. #  0.1 #  1.0
            # lr = 5e5 # 1e6 # 1e5 #### learning rate setting ## 
            # 
            # print(f"q: {tot_qs[-1]}, grad: {reconstructor.state_vals.grad.data[tot_qs.size(0) - 1]}") # geneoh-dfifusion #
            # print(f"q: {tot_qs[0]}, grad: {reconstructor.state_vals.grad.data[0]}")
            # for i_s in range(tot_qs.size(0)): #  
            #     tot_qs[i_s] = tot_qs[i_s] - reconstructor.state_vals.grad.data[i_s] * lr
            
            # print(f"act_xs: {tot_act_xs[i_s, ]}")
            for i_s in range(tot_act_xs.shape[0]):
                tot_act_xs[i_s] = tot_act_xs[i_s] - reconstructor.ti_act_xs.grad.to_numpy()[i_s] * lr
            
            reconstructor.initialize()
            # reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy()[:-1]
        ''' optimize the state spaces '''
        exit(0)
    
    
    
    sim = redmax.Simulation(model_path)
    
    
    rendering_timestep = 1
    
    # ##### load pre-optimized us ##### #
    pre_optimized_us_fn = "/home/xueyi/diffsim/NeuS/optimized_us.npy" # optimized_us # optimized_us # optimized_us # # # 
    pre_optimized_us = np.load(pre_optimized_us_fn, allow_pickle=True)
    u = pre_optimized_us.copy() # #### pre_optimized_us #### #
    
    
    i_iter = 0
    def loss_and_grad_ref_states(u): ## loss and grad ref states
        global i_iter, args, model_path, rendering_timestep, initial_radius # load the rgb file #
        
        sim.reset(True)
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i : ndof_u * (i + 1)])
            sim.forward(1)
            if (i + 1) % nn_mod_steps == 0:
                # print(f"cur_step: {i}, nn_mod_steps: {nn_mod_steps}")
                tot_qs.append(sim.get_q())
            # tot_qs.append(sim.get_q())
        q = sim.get_q()
        
        # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
        # robots = load_utils.parse_data_from_xml(xml_fn, args)
        # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
        # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
        
        tot_qs = np.stack(tot_qs, axis=0) ### tot_qs for the qs
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        
        # if args.optimize: # from states #
        if args.use_loss_type == "rendering": # ["rendering", "point_tracking"] #
            # reconstructor.forward_p2g()
            # reconstructor.forward_ti_rendering()
            # reconstructor.forward_dvgo_rendering()
            # reconstructor.forward_p2g_with_rgb()
            if args.use_grid_loss_type == "density": # with rgb # 
                    reconstructor.forward_p2g_with_rgb(rendering_timestep=rendering_timestep)
            elif args.use_grid_loss_type == "deformation":
                ##### foward p2g with deformation #### # density #
                reconstructor.forward_p2g_with_deformation()
            else:
                raise ValueError(f"Unrecognized use_grid_loss_type: {args.use_grid_loss_type}")
                ## 
            print(f"start backwarding")
            # reconstructor.backward_stepping()
        elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"]
            reconstructor.forward_tensor_to_ti()
            reconstructor.backward_stepping_tracking_target()
        else:
            raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
    
        #### ndof_r * num_steps #### # num_steps #
        
        
        
        state_vals_grad = reconstructor.state_vals.grad.data.detach().cpu().numpy()
        
        df_dq = np.zeros(ndof_r * num_steps) # ndof_r * num_steps #
        # df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
        # for i_s in range(reconstructor.n_timesteps): ## n_timesteps ##
        # for i_s in [reconstructor.n_timesteps - 1]: ## n_timesteps ##
        #     cur_st_idx = (i_s + 1) * nn_mod_steps - 1
        #     # print(f"cur_step: {i_s}, cur_st_idx: {cur_st_idx}")
        #     # df_dq[cur_st_idx * ndof_r: (cur_st_idx + 1) * ndof_r] = reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy() #
        #     ###### detach; cpu; numpy ######
        #     df_dq[cur_st_idx * ndof_r: (cur_st_idx + 1) * ndof_r - 1] = reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy()[:-1] 
        
        for i_s in range(rendering_timestep): ## n_timesteps ##
            cur_st_idx = (i_s + 1) * nn_mod_steps - 1
            # print(f"cur_step: {i_s}, cur_st_idx: {cur_st_idx}")
            # df_dq[cur_st_idx * ndof_r: (cur_st_idx + 1) * ndof_r] = reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy() #
            ###### detach; cpu; numpy ######
            df_dq[cur_st_idx * ndof_r: (cur_st_idx + 1) * ndof_r - 1] = reconstructor.state_vals.grad.data[i_s].detach().cpu().numpy()[:-1] 
        
        
        if args.use_loss_type == "rendering": 
            f = reconstructor.tot_cd_loss
            f = f * 1000
            # f = f * 500
        elif args.use_loss_type == "point_tracking":  ## point tracking loss; point ##
            f = reconstructor.point_tracking_loss[None]
            f = f * 0.5 ## active ##
            # f = f * 1.0 ## active ##
            # f = f * 0.3 ## active ##
            # f = f * 10.0
        else:
            raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
        
        ### iter and the final qs ###
        print(f"iter: {i_iter}, q: {q}, grad: {reconstructor.state_vals.grad.data[-1].detach().cpu().numpy()}")
        sim.backward_info.set_flags(False, False, False, True)  #### 
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps)
        
        # df_dq = np.zeros(ndof_r * num_steps)
        # df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
        # df_dq[-ndof_r: -(ndof_r - ndof_u)] = 2 * (q - init_q_goal)[: -(ndof_r - ndof_u)]
        ### dense states supervision #### ## supervision ##
        # tot_qs = np.concatenate(tot_qs, axis=0)
        # df_dq = 2 * tot_qs
        # f = np.sum(tot_qs ** 2)
        
        print(f"iter: {i_iter}, f: {f}, last_q: {tot_qs[rendering_timestep - 1]}, df_dq: {state_vals_grad[rendering_timestep - 1]}")
        
        
        grad_multiplying_ratio = 1.
        grad_multiplying_ratio = 10.
        
        sim.backward_info.df_dq = df_dq * grad_multiplying_ratio ## ## df_dq ## ### grad_multiplying_ratio ###
        
        sim.backward()
        grad = np.copy(sim.backward_results.df_du)
        
        last_rendering_timestep_st = (rendering_timestep * nn_mod_steps - 1) * ndof_u # rendering time stpes; nn_mod_steps #
        last_rendering_timestep_ed = last_rendering_timestep_st + ndof_u
        grad_rendeirng_lasttime_u = grad[last_rendering_timestep_st : last_rendering_timestep_ed] 
        
        print(f"grad to u: {grad_rendeirng_lasttime_u}") ### rendering last timestep u ###
        
        reconstructor.initialize()
        
        
        
        ''' config the reconstructor via target radius (perturbed radius) '''
        config_reconstructor_sim_via_target_radius(initial_radius + eps_radius)
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i : ndof_u * (i + 1)])
            sim.forward(1)
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(sim.get_q())
        q = sim.get_q()
        
        tot_qs = np.stack(tot_qs, axis=0) ### tot_qs for the qs
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        
        # if args.optimize:
        if args.use_loss_type == "rendering":
            if args.use_grid_loss_type == "density":
                    reconstructor.forward_p2g_with_rgb(rendering_timestep=rendering_timestep)
            elif args.use_grid_loss_type == "deformation":
                reconstructor.forward_p2g_with_deformation()
            else:
                raise ValueError(f"Unrecognized use_grid_loss_type: {args.use_grid_loss_type}")
            print(f"start backwarding")
        elif args.use_loss_type == "point_tracking":
            reconstructor.forward_tensor_to_ti()
            reconstructor.backward_stepping_tracking_target()
        else:
            raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
        
        if args.use_loss_type == "rendering":
            perturb_f = reconstructor.tot_cd_loss
            perturb_f = perturb_f * 1000
        elif args.use_loss_type == "point_tracking":
            perturb_f = reconstructor.point_tracking_loss[None]
            perturb_f = perturb_f * 0.5
        else:
            raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
        
        print(f"iter: {i_iter}, with delta_radius, f: {perturb_f}, last_q: {tot_qs[rendering_timestep - 1]}")
        gradient_over_radius = (perturb_f - f) / eps_radius
        lr_radius = 0.001
        print(f"iter: {i_iter}, radius: {initial_radius}, grad: {gradient_over_radius}")
        initial_radius = initial_radius - lr_radius * gradient_over_radius # 
        
        reconstructor.initialize()
        
        ''' config the reconstructor via target radius (updated radius) '''
        config_reconstructor_sim_via_target_radius(initial_radius) # /home/xueyi/sim/arctic/prepared_data_s01_laptop_use_01_1.npy #
        
        i_iter += 1
        return f, grad


    def callback_func_ref_states(u, render=False): # 
        global reconstructor, args, rendering_timestep
        
        # sim = redmax.Simulation(model_path) #
        sim.reset(False)
        
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
            q = sim.get_q()
            if (i + 1) % nn_mod_steps == 0:
                tot_qs.append(q)
        
        tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
        
        tot_qs = torch.from_numpy(tot_qs).float().cuda(args.th_cuda_idx)
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        
         ####### save the optimized pts #######
        # tot_frame_act_xs, tot_frame_passive_xs #
        cur_optimized_act_xs = reconstructor.tot_frame_act_xs.detach().cpu().numpy() ### act xs ##
        cur_optimized_passive_xs = reconstructor.tot_frame_passive_xs.detach().cpu().numpy() ### passive xs ###
        cur_act_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_act_xs.npy")
        cur_passive_xs_sv_fn = os.path.join(args.image_sv_folder, f"active_only_optim_passive_xs.npy")
        np.save(cur_act_xs_sv_fn, cur_optimized_act_xs) 
        np.save(cur_passive_xs_sv_fn, cur_optimized_passive_xs)
        print(f"current optimized act xs saved to {cur_act_xs_sv_fn}")
        print(f"current optimized passive xs saved to {cur_passive_xs_sv_fn}")
        last_frame_passive_avg = np.mean(cur_optimized_passive_xs[-1], axis=0)
        print(f"last_frame_passive_avg: {last_frame_passive_avg}")
        ####### save the optimized pts #######
        
        # reconstructor.forward_p2g()
        # reconstructor.forward_ti_rendering()
        # reconstructor.forward_dvgo_rendering()
        reconstructor.forward_p2g_with_rgb(rendering_timestep=rendering_timestep)
        # print(f"start backwarding") ### def tet ## def tet ##
        # reconstructor.backward_stepping() ## ti rendering ## ti rendering ##
    
        ######## use loss type ######## loss type # get 3d -> the def tet ##
        # if args.use_loss_type == "rendering": # ["rendering", "point_tracking"]
        #     reconstructor.forward_p2g()
        #     reconstructor.forward_ti_rendering() # learning thing; and the contact model of it # ## 
        #     print(f"start backwarding")
        #     reconstructor.backward_stepping()
        # elif args.use_loss_type == "point_tracking": # ["rendering", "point_tracking"]
        #     reconstructor.forward_tensor_to_ti()
        #     reconstructor.backward_stepping_tracking_target()
        # else:
        #     raise ValueError(f"Unrecognized loss type: {args.use_loss_type}")
        #### get total rendering #### ## for the q_res function ##
        
        # image #
        save_q_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_q_iter_{i_iter}.npy")
        np.save(save_q_res_fn, tot_qs.detach().cpu().numpy())
        print(f"active_only q optimized at iter {i_iter} saved to {save_q_res_fn}") ## save_q_res_fn
        
        save_u_res_fn = os.path.join(args.image_sv_folder, f"active_only_optim_res_u_iter_{i_iter}.npy")
        np.save(save_u_res_fn, u)
        print(f"active_only u optimized at iter {i_iter} saved to {save_u_res_fn}") ## save_q_res_fn

        reconstructor.initialize()
        q = sim.get_q()
        
        print('q = ', q)
        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = True, record_path = "./optim_only_active_states.gif")
        return
    
    ## loss and grad ref states ## # callback_func_ref_states -> ref states ##
    res = scipy.optimize.minimize(loss_and_grad_ref_states, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func_ref_states)

    callback_func_ref_states(u = res.x, render = args.render)
    exit(0)
