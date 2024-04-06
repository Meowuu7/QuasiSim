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

import random
import taichi as ti

### test the hand sphere optimize ###

# ti.init(arch=ti.cuda)
real = ti.f32
ti.init(default_fp=real, arch=ti.cuda, device_memory_fraction=0.5)

@ti.data_oriented
class Reconstructor(nn.Module):
    def __init__(self, args, robot_name_to_robot) -> None:
        super().__init__()
        
        self.th_cuda_idx = 0
        
        self.args = args
        self.n_timesteps = args.n_timesteps
        
        self.n_act_states = args.n_act_states
        self.n_passive_states = args.n_passive_states
        self.dim = args.dim
        
        self.res = args.res ### the image resolution ###
        self.n_views = args.n_views
        
        
        self.n_states = self.n_act_states + self.n_passive_states ## states ##
        
        self.state_vals = torch.zeros((self.n_timesteps, self.n_states), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx) # , requires_grad=True
        ## self.passive_x ##3
        
        ### TODO: for the robot structure -> should change them in the torch envieonemt; and check whethe it is enough to retain gradients in the calculation
        ### 1) load initial robots ####
        ## robot 
        self.name_to_robot = robot_name_to_robot ## robot name to robot ##
        # give state vals gradients
        self.active_robot_name = args.active_robot_name
        self.passive_robot_name = args.passive_robot_name
        self.active_robot = self.name_to_robot[self.active_robot_name]
        self.passive_robot = self.name_to_robot[self.passive_robot_name]
        
        self.nn_active_pts = self.active_robot.nn_pts
        self.nn_passive_pts = self.passive_robot.nn_pts
        
        
        
        
        
        ### in the torch space ###
        self.act_xs = torch.zeros((self.n_timesteps, self.nn_active_pts, self.dim), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx) ## requires grad here
        self.passive_xs = torch.zeros((self.n_timesteps, self.nn_passive_pts, self.dim), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx) ## passive_xs ##
        
        ### for girds ###
        self.grid_res = args.density_res ## density resolution
        self.buffered_grids = torch.zeros((self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx)
        self.buffered_grid_deformation = torch.zeros((self.n_timesteps, self.grid_res, self.grid_res, self.grid_res, self.dim), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx)
        self.act_xs_deformation = torch.zeros((self.n_timesteps, self.nn_active_pts, self.dim), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx)
        self.passive_x_deformation = torch.zeros((self.n_timesteps, self.nn_passive_pts, self.dim), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx)
        
        ### for images ###
        self.buffered_images = torch.zeros((self.n_timesteps, self.n_views, self.res, self.res), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx)
        self.buffered_images_deformations = torch.zeros((self.n_timesteps, self.n_views, self.res, self.res, 2), dtype=torch.float32, requires_grad=True).cuda(self.th_cuda_idx) #### buffered images deformations ###
        
        self.ti_act_xs = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=( self.n_timesteps, self.nn_active_pts))
        self.ti_passive_x = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, self.nn_passive_pts))
        self.ti_buffered_images = [ # self.
            ti.field(dtype=ti.float32, shape=(self.n_views, self.res, self.res), needs_grad=True) for _ in range(self.n_timesteps)
        ]
        self.ti_buffered_images_deformation = [ # self.
            ti.Vector.field(n=2, dtype=ti.float32, shape=(self.n_views, self.res, self.res), needs_grad=True) for _ in range(self.n_timesteps)
        ]
        self.ti_buffered_grids = ti.field(dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), needs_grad=True)
        
        # self.ti_buffered_grids_deformation = ti.field(dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), needs_grad=True)
        
        self.ti_buffered_grids_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), needs_grad=True)
        
        
        self.ti_act_xs_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=( self.n_timesteps, self.nn_active_pts))
        self.ti_passive_x_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, self.nn_passive_pts))
        
        self.n_neighbour = (3,) * self.dim
        # self.neighbour = (3,) * self.dim
        
        
        ### 
        
        self.maxx_pts = 25.
        self.minn_pts = -15.
        self.extent = self.maxx_pts - self.minn_pts
        
        
        ### for grid neighbours ###
        self.neighbour = 3
        self.neighbour_offsets = []
        for i_x in range(self.neighbour):
            for i_y in range(self.neighbour):
                for i_z in range(self.neighbour):
                    self.neighbour_offsets.append(torch.tensor([i_x, i_y, i_z], dtype=torch.long).cuda(self.th_cuda_idx))
                    
        self.render = VolumeRender(
            res=args.res, density_res=self.grid_res, dx=args.dx, n_views=self.n_views, torus_r1=args.torus_r1, torus_r2=args.torus_r2, fov=args.fov, camera_origin_radius=args.camera_origin_radius, marching_steps=args.marching_steps, learning_rate=args.learning_rate, tot_frames=self.n_timesteps
        )
        
        self.ti_render = volume_render.VolumeRender(res=args.res, density_res=self.grid_res, dx=args.dx, n_views=self.n_views, torus_r1=args.torus_r1, torus_r2=args.torus_r2, fov=args.fov, camera_origin_radius=args.camera_origin_radius, marching_steps=args.marching_steps, learning_rate=args.learning_rate, tot_frames=self.n_timesteps)
    
    def initialize(self, ):
        self.act_xs = self.act_xs * 0.
        self.passive_xs = self.passive_xs * 0.
        self.buffered_grids = self.buffered_grids * 0.
        self.buffered_grid_deformation = self.buffered_grid_deformation * 0.
        self.act_xs_deformation = self.act_xs_deformation * 0.
        self.passive_x_deformation = self.passive_x_deformation * 0.
        self.buffered_images = self.buffered_images * 0.
        self.buffered_images_deformations = self.buffered_images_deformations * 0.
        
        
        self.ti_act_xs.fill(0.)
        self.ti_act_xs.grad.fill(0.)
        self.ti_passive_x.fill(0.)
        self.ti_passive_x.grad.fill(0.)
        for i_fr in range(self.n_timesteps):
            self.ti_buffered_images[i_fr].fill(0.)
            self.ti_buffered_images[i_fr].grad.fill(0.)
            self.ti_buffered_images_deformation[i_fr].fill(0.)
            self.ti_buffered_images_deformation[i_fr].grad.fill(0.)
        self.ti_buffered_grids.fill(0.)
        self.ti_buffered_grids.grad.fill(0.)
        self.ti_buffered_grids_deformation.fill(0.)
        self.ti_buffered_grids_deformation.grad.fill(0.)
        self.ti_act_xs_deformation.fill(0.)
        self.ti_act_xs_deformation.grad.fill(0.)
        self.ti_passive_x_deformation.fill(0.)
        self.ti_passive_x_deformation.grad.fill(0.)
        self.ti_render.initialize()
        
        
        
        
    def set_states(self, tot_states):
        # active_states = tot_states[:, :self.n_]
        self.state_vals[:, :] = tot_states[:, :] ### state values here ###
        
        
    def forward_stepping(self, ):
        # for each frame; step the robot to extract the object points #
        for i_fr in range(self.n_timesteps):
            cur_fr_active_states = self.state_vals[i_fr, :self.n_act_states]
            cur_fr_passive_states = self.state_vals[i_fr, self.n_act_states: ]
            self.active_robot.set_state_via_vec(cur_fr_active_states)
            self.passive_robot.set_state_via_vec(cur_fr_passive_states) ### for the active and passive ###
            
            cur_active_robot_pts = [] # cur active robot pts # 
            cur_active_robot_pts = self.active_robot.compute_transformation_via_state_vecs(cur_fr_active_states, cur_active_robot_pts)
            cur_active_robot_pts = torch.cat(cur_active_robot_pts, dim=0) ### cat pts
            self.act_xs[i_fr] = cur_active_robot_pts
            
            
            # cur_active_robot_pts = []
            # cur_active_robot_pts = self.active_robot.compute_transformation_via_state_vecs(cur_fr_active_states, cur_active_robot_pts)
            # cur_active_robot_pts = torch.cat(cur_active_robot_pts, dim=0) ### cat pts
            # self.act_xs[i_fr] = cur_active_robot_pts
            
            
            cur_active_robot_pts = []
        #     def get_visual_pts_list(self, visual_pts_list):
        #         for cur_link in self.children:
        #             cur_link.get_visual_pts_list(visual_pts_list)
            self.active_robot.get_visual_pts_list(cur_active_robot_pts)
            cur_active_robot_pts = torch.cat(cur_active_robot_pts, dim=0) ### cat pts
            self.act_xs[i_fr] = cur_active_robot_pts ### get current frame act robots 
            
            ### TODO: add the passive robot transformations
            cur_passive_robot_pts = []
            self.passive_robot.get_visual_pts_list(cur_passive_robot_pts)
            cur_passive_robot_pts = torch.cat(cur_passive_robot_pts, dim=0)
            self.passive_xs[i_fr] = cur_passive_robot_pts ### get passive robot pts ###
            
        # ## forward stepping ##
        # pass
    
    # forward active pts to grids #
    def forward_active_pts_to_grids(self, ):
        ## act_xs ### nn_frames x nn_pts x 3 #
        maxx_act_xs = torch.max(self.act_xs, dim=0)[0]
        maxx_act_xs = torch.max(maxx_act_xs, dim=0)[0]
        minn_act_xs = torch.min(self.act_xs, dim=0)[0]
        minn_act_xs = torch.min(minn_act_xs, dim=0)[0]
        print(f"maxx_act_xs: {maxx_act_xs}, minn_act_xs: {minn_act_xs}")
        ### the active object only setting ### 
        random_sample_idx = list(range(self.nn_active_pts))
        random.shuffle(random_sample_idx)
        random_sample_idx = random_sample_idx[:500]
        # random_sample_idx = torch.tensor(random_sample_idx, )
        print(f"nn_active_pts: {self.nn_active_pts}")
        # for i_fr in range(self.n_timesteps):
        for i_fr in range(1):
            print(f'cur_fr: {i_fr}')
            # for i_pts in range(self.nn_active_pts): #### active pts #### ## active pts #### 
            for i_pts in random_sample_idx:
                ### TODO: scale the object pts here ###
                cur_pts = self.act_xs[i_fr, i_pts] ## scale the object pts ##
                cur_pts = (cur_pts - self.minn_pts) / self.extent
                X_cur_pts = cur_pts * self.grid_res ## to the grid index ## ## 
                base = (X_cur_pts - 0.5).long()
                
                fx = X_cur_pts - base
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
                
                for offset in self.neighbour_offsets:
                    weight = 1.0
                    for i in range(self.dim):
                        weight *= w[offset[i]][i]
                    offseted_grid_idx = offset + base
                    self.buffered_grids[i_fr, offseted_grid_idx[0], offseted_grid_idx[1], offseted_grid_idx[2]] += weight * self.args.p_mass
                    self.buffered_grid_deformation[i_fr, offseted_grid_idx[0], offseted_grid_idx[1], offseted_grid_idx[2]] += weight * self.act_xs_deformation[i_fr, i_pts] ### add to the grid deformation ###

    @ti.kernel
    def rasterize_act_cur_step_cur_frame(self, i_fr: ti.i32): ## i_fr >= 1 ##
        for i_v in range(self.nn_active_pts):  ### nn_samples ###
            # i_link, i_vert = self.get_i_link_i_v(i_v=i_v) #
            x = (self.ti_act_xs[i_fr, i_v] - self.minn_pts) / self.extent
            Xp = x * self.grid_res
            base = ti.cast(Xp - 0.5, ti.int32)
            # print(f"x: {x}, base: {base}") #
            fx = Xp - base # grid indexes #
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.n_neighbour))):
                # dpos = (offset - fx) * self.dx #
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                    
                self.ti_buffered_grids[i_fr, base + offset] += weight * self.args.p_mass # aggregate mass from particles ## aggregate mass from 

                self.ti_buffered_grids_deformation[i_fr, base + offset] += weight * self.ti_act_xs_deformation[i_fr, i_v]
    
    # @ti.kernel
    def forward_ti_rendering(self, ):
        tot_rendered_imgs = []
        # self.render.fill_density_tot_frames(self.buffered_grids) ##s
        # self.render.fill_density_tot_frames
        self.ti_render.fill_density_fr_ti_with_fr_density_fr_single_loop(self.ti_buffered_grids)
        self.ti_render.fill_density_deformation_fr_ti_with_fr_density_fr_single_loop(self.ti_buffered_grids_deformation) ### fill density ###
        for i_frame in range(self.n_timesteps):
            #### fill density fr ti ###
            t1 = time.time()
            # self.render.fill_density_fr_ti_with_fr(self.buffered_grids, i_frame) # fill density fr ti #
            # self.render.fill_density_fr_ti_with_fr_density_fr(self.buffered_grids, i_frame)
            t2 = time.time()
            # print(f"time used for filling density field: {t2 - t1}")
            for view in range(self.ti_render.n_views): ### i_frame ###
                t3 = time.time()
                # self.render.ray_march_single_loop_with_density_frames_with_deformation(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                self.ti_render.ray_march_single_loop_with_density_frames_with_deformation(self.ti_buffered_images[i_frame], self.ti_buffered_images_deformation[i_frame], math.pi / (self.ti_render.n_views * self.args.view_divide) * view - math.pi / 2.0, view, i_frame)
                # # ## i_fr: ti.i32, angle: ti.f32, view_id: ti.i32 ### ## ray marching ##
                # self.render.ray_march_single_loop_with_frame(
                #     i_frame, math.pi / self.render.n_views * view - math.pi / 2.0, view
                # )
                # t4 = time.time()
                # print(f"view {view} rendering used time {t4 - t3}")
            
        # self.get_rendered_iamges_from_render()
        
        for i_frame in range(self.n_timesteps):
            ### views and images ###
            views = self.ti_buffered_images[i_frame].to_numpy() # ti render 
            # tot_rendered_imgs.append(views)
            for view in range(self.ti_render.n_views): ## ti render for rendering ##
                img = views[view]
                m = np.max(img)
                if m > 0:
                    img /= m
                img = 1 - img
                ### #### imwrite for iamges #### #
                imwrite(
                    os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_nv_{:d}.png".format(i_frame, view, self.args.optim, self.args.passive_only, self.n_views)),
                    (255 * img).astype(np.uint8))
            
            # #### save target views ###
            # target_views = self.target_images[i_frame].to_numpy()
            # for view in range(self.render.n_views): ## 
            #     img = target_views[view]
            #     m = np.max(img)
            #     if m > 0:
            #         img /= m # img /= m #
            #     img = 1 - img
            #     ### #### imwrite for iamges #### #
            #     imwrite(
            #         os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_target_nv_{:d}.png".format(i_frame, view, optim, passive_only, self.n_views)),
            #         (255 * img).astype(np.uint8))
            
        #### tot rendered imgs #### 
        # tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ## nn_frames x nn_views x nn_img_res x nn_img_res
        tot_rendered_imgs = [cur_frame_image.to_numpy() for cur_frame_image in self.ti_buffered_images]
        tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ### numpy 
        # tot_rendered_imgs = self.buffered_images[i_frame].to_numpy() # 
        
        rendered_imgs_sv_fn = os.path.join(self.args.image_sv_folder, f"rendered_images_optim_{self.args.optim}_passive_{self.args.passive_only}_nv_{self.n_views}_test.npy")
        np.save(rendered_imgs_sv_fn, tot_rendered_imgs) ## rendered imgaes ##
        print(f"Rendered images saved to {rendered_imgs_sv_fn}")
    
    
    #### TODO: implement the forward passive pts to grid function ####
    ## implement ##
    
    def forward_p2g(self,):
        self.ti_act_xs.from_numpy(self.act_xs.detach().cpu().numpy())
        self.ti_passive_x.from_numpy(self.passive_xs.detach().cpu().numpy())
        for i_fr in range(self.n_timesteps):
            print(f"Rasterizing the frame: {i_fr}")
            self.rasterize_act_cur_step_cur_frame(i_fr=i_fr)
        # self.forward_active_pts_to_grids()
    
    ## init the reconstructor ###
    def forward_rendering(self, ):
        # for i_fr in range(self.n_timesteps):
        for i_fr in range(1):
            for view in range(self.n_views):
                print(f"i_frame: {i_fr}, i_view: {view}")
                cur_angle = torch.tensor([math.pi / (self.render.n_views * self.args.view_divide) * view - math.pi / 2.0], dtype=torch.float32).cuda()
                self.render.ray_march_single_loop_with_density_frames_with_deformation_th(self.buffered_grids[i_fr], self.buffered_grid_deformation[i_fr], self.buffered_images[i_fr], self.buffered_images_deformations[i_fr], angle=cur_angle, view_id=view)
        
        for i_frame in range(self.n_timesteps):
            ### views and images ###
            views = self.buffered_images[i_frame].detach().cpu().numpy() 
            # tot_rendered_imgs.append(views)
            
            for view in range(self.n_views): ## 
                print(f"i_frame: {i_frame}, i_view: {view}")
                img = views[view]
                m = np.max(img)
                if m > 0:
                    img /= m # img /= m #
                img = 1 - img
                ### #### imwrite for iamges #### #
                imwrite(
                    os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_nv_{:d}.png".format(i_frame, view, self.args.optim, self.args.passive_only, self.n_views)),
                    (255 * img).astype(np.uint8))
            
            #### save target views ###
            target_views = self.target_images[i_frame].detach().cpu().numpy()
            for view in range(self.render.n_views): ## 
                img = target_views[view]
                m = np.max(img)
                if m > 0:
                    img /= m # img /= m #
                img = 1 - img
                ### #### imwrite for iamges #### #
                imwrite(
                    os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_target_nv_{:d}.png".format(i_frame, view, self.args.optim, self.args.passive_only, self.n_views)),
                    (255 * img).astype(np.uint8))
            
        #### tot rendered imgs ####
        # tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ## nn_frames x nn_views x nn_img_res x nn_img_res
        tot_rendered_imgs = [cur_frame_image.detach().cpu().numpy() for cur_frame_image in self.buffered_images]
        tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ### numpy 
        # tot_rendered_imgs = self.buffered_images[i_frame].to_numpy() # 
        
        rendered_imgs_sv_fn = os.path.join(self.args.image_sv_folder, f"rendered_images_optim_{self.args.optim}_passive_{self.args.passive_only}_nv_{self.n_views}_n_iter_{i_iter}.npy")
        np.save(rendered_imgs_sv_fn, tot_rendered_imgs) ## rendered imgaes ##
        print(f"Rendered images saved to {rendered_imgs_sv_fn}")
    




if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Test forward and rendering of simulation')
    # parser.add_argument("--model", type = str, default = 'finger_torque')
    # parser.add_argument('--gradient', action = 'store_true')
    # parser.add_argument("--record", action = "store_true")
    # parser.add_argument("--render", action = "store_true")
    
    parser = data_utils.create_arguments()
    
    
    args = parser.parse_args()
    # args.image_sv_folder = "/data1/sim/diffsim/DiffRedMax/save_res"
    # args.image_sv_folder = "./save_res/hand_sphere_demo"
    args.image_sv_folder = "./save_res/hand_sphere_demo_target"
    os.makedirs(args.image_sv_folder, exist_ok=True)
    args.num_steps = 1000
    args.mod_sv_timesteps = 50
    args.use_deformation = False
    args.use_deformation = True
    args.use_kinematic = False

    # args = parser.parse_args()

    ## ##
    asset_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))

    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')

    sim = redmax.Simulation(model_path) ## hand sphere model path ##

    nn_render_steps = 10
    num_steps = 1000
    nn_mod_steps = num_steps // nn_render_steps
    
    ### set basic timestepping arguments ###
    args.mod_sv_timesteps = nn_mod_steps
    args.num_steps = num_steps
    args.n_timesteps = nn_render_steps
    args.n_timesteps = 24
    

    sim.reset(backward_flag = args.gradient) # reset the simulation to start a new trajectory

    ndof_u = sim.ndof_u # number of actions
    ndof_r = sim.ndof_r # number of degrees of freedom
    ndof_var = sim.ndof_var # number of auxiliary variables
    
    args.n_act_states = ndof_u
    args.n_passive_states = ndof_r - ndof_u
    args.active_robot_name = "active"
    args.passive_robot_name = "passive"
    ## should 
    
    
    
    loss = 0.
    df_du = np.zeros(ndof_u * num_steps)
    df_dq = np.zeros(ndof_r * num_steps) ## free2d
    
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
        #     sim.set_u(np.ones(ndof_u)) ## sim forward and backward ##
        sim.forward(1, verbose = False)
        q = sim.get_q() ## loss += np.sum(q)
        if (i + 1) % nn_mod_steps == 0:
            tot_qs.append(q)
        loss += np.sum(q) # 
        df_dq[ndof_u * i:ndof_u * (i + 1)] = 1.
        
        # print(f"state of the sphere at {i}-th step: {q[-3:]}")
        # draw the geometry at each timestep here? 
    print(f'q = {q}')
    
    args.th_cuda_idx = 0
    
    xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    robots = load_utils.parse_data_from_xml(xml_fn, args=args)
    robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
    
    
    # tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
    # # forward 1) kienmatic state transformation; 2) rasterization; and 3) the rendering process ## 
    # # backward 1) ---> need target images; 2) need gradients; 3) needs losses and the way of getting supervision from pixels # 
    # tot_qs = torch.from_numpy(tot_qs).float()
    # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
    # robots = load_utils.parse_data_from_xml(xml_fn)
    # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
    # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
    # reconstructor.set_states(tot_qs)
    # reconstructor.forward_stepping()
    # reconstructor.forward_p2g()
    # # reconstructor.forward_rendering()
    # reconstructor.forward_ti_rendering()
    
    
    
    # if args.render:
    #     SimRenderer.replay(sim, record = True, record_path = "./optim_initial.gif")
        
    # u[11] = 1e3
    # for i in range(num_steps):
    #     u[ndof_u * i + 11: ndof_u * i + 14] = 1e5
    #     # u[ndof_u * i + 6: ndof_u * i + 10] = 1e3
    #     # setting a good initilziation; but i still do not know what happens here... ## 
    #     ## q[-1] is the rotation variable? q[-2] is the direction along the gravity direction ##
    #     ## q[-3] is the direction along the forwarding x direction ###
    
    sim.reset(False)
    q_goal = np.zeros(ndof_r - 3) - 0.001
    P_q = np.ones(ndof_r - 3) *0.001
    u = np.zeros(ndof_u * num_steps)
    # ## set the initial value of q ##
    for i in range(num_steps):
        # if i % 1 == 0:
        #     print(f"step: {i}")
        q = sim.get_q() # 
        error = q_goal - q[:-3]
        ui = error * P_q
        u[i * ndof_u: (i + 1) * ndof_u] = ui
        sim.set_u(ui)
        sim.forward(1, False)
    q = sim.get_q()
    print('q = ', q)
    
    
    # test_qs_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/res_q_iter_49.npy"
    # test_qs = np.load(test_qs_fn, allow_pickle=True)
    
    test_us_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/res_u_iter_49.npy"
    # res_u_iter_79.npy
    test_us_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/res_u_iter_79.npy"
    # res_u_iter_78.npy ### u_iter_78 #
    test_us_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/res_u_iter_78.npy"
    test_us_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/active_only_optim_res_u_iter_155.npy"
    test_us_fn = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/active_with_goal_optim_res_u_iter_30.npy"
    test_us_fn = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo/active_with_goal_optim_res_u_iter_79.npy"
    # test_us_fn #
    test_us = np.load(test_us_fn, allow_pickle=True)
    
    
    step_idxes_with_contacts = [147, 148, 154, 155, 194, 195, 344, 345, 463, 464, 578, 579, 581, 582, 594, 595, 627, 628, 659, 660, 661, 662, 681, 682]
    
    
    def callback_func_qs_test(render=False):
        global test_us
        sim.reset(False)
        
        tot_qs = []
        
        for i in range(num_steps): # 
            sim.set_u(test_us[ndof_u * i:ndof_u * (i + 1)])
            # sim.set_q
            sim.forward(1)
            
            if i in step_idxes_with_contacts:
                cur_q = sim.get_q()
                print(f"i: {cur_q}, cur_q: {cur_q}")
                tot_qs.append(cur_q) #### cur_q ###

        
        
        tot_qs = np.stack(tot_qs, axis=0)
        tot_qs = torch.from_numpy(tot_qs).float().cuda()
        reconstructor.set_states(tot_qs)
        reconstructor.forward_stepping()
        cur_act_xs = reconstructor.act_xs.detach().cpu().numpy()
        cur_passive_xs = reconstructor.passive_xs.detach().cpu().numpy()
        tot_sv_stats = {
            'act_xs': cur_act_xs, "passive_xs": cur_passive_xs # passive xs and act_xs #
        }
        # tot_sv_states #
        np.save(f"act_passive_stats.npy", tot_sv_stats)
        print(f"act passive saved to {'act_passive_stats.npy'}")
        
        
        q = sim.get_q()
        
        print('q = ', q)
        if render:
            # print('q = ', q)
            SimRenderer.replay(sim, record = True, record_path = "./save_res/res_gif/optim_active_with_goal_states_test_qs_target_v2.gif")

        # global final_u
        # final_u[:] = u[:]
        
        return
    
    callback_func_qs_test( render = args.render)
    exit(0)
    
    # if args.render: ## and render them at different status ## ## track the active object v.s. track both the active and the passive object ##
    #     SimRenderer.replay(sim, record = True, record_path = "./optim_after_pd.gif")
    
    init_q_goal = -1
    # init_q_goal = -2 # deftet #
    init_q_goal = -2 #  -4
    i_iter = 0
    def loss_and_grad_qs(u):
        global i_iter
        sim.reset(True)
        tot_qs = []
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i : ndof_u * (i + 1)])
            sim.forward(1)
            tot_qs.append(sim.get_q())
        q = sim.get_q()
        f = np.sum((q - init_q_goal) ** 2) # 
        
        print(f"iter: {i_iter}, q: {q}")
        sim.backward_info.set_flags(False, False, False, True) 
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps) ## df_du ##
        df_dq = np.ones(ndof_r * num_steps) 
        
        df_dq = np.zeros(ndof_r * num_steps) 
        df_dq[-ndof_r: ] = 2 * (q - init_q_goal)
        ### dense states supervision ####
        # tot_qs = np.concatenate(tot_qs, axis=0)
        # df_dq = 2 * tot_qs ### tot_q
        # f = np.sum(tot_qs ** 2)
        
        print(f"iter: {i_iter}, f: {f}")
        
        sim.backward_info.df_dq = df_dq ## df_dq
        
        sim.backward()
        grad = np.copy(sim.backward_results.df_du)
        
        i_iter += 1
        return f, grad
    
    final_u = np.zeros_like(u)
    
    
    ## 
    # def callback_func(u, render = False):
    #     sim.reset(False)
    #     for i in range(num_steps): ## 
    #         sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
    #         sim.forward(1)
        
    #     q = sim.get_q()

    #     f = (q[3] - x_goal) ** 2

    #     print('f = ', f)

    #     if render:
    #         print('q = ', q)
    #         SimRenderer.replay(sim, record = False, record_path = "./torque_finger_flick_optimized.gif")
    
    
    def callback_func_qs(u, render=False):
        sim.reset(False)
        
        for i in range(num_steps): ## 
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)

        q = sim.get_q()
        
        print('q = ', q)
        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = True, record_path = "./save_res/res_gif/optim_only_active_states.gif")

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
    
    callback_func_qs(u = res.x, render = args.render)
        
    # sim.reset(False)
    # export_data_folder = "./save_res/hand_sphere_demo"
    # os.makedirs(export_data_folder, exist_ok=True)
    # # sim.export_replay(export_data_folder)
    # sim.replay()
    # SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) # render the simulation replay video
    
    # SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) 
    
    # u[:] = final_u[:]
    
    ## set the u to the res.x, the previous optimization result ##
    u[:] = res.x[:]
    
    y_idx = -2
    # y_idx = -3
    i_iter = 0
    y_goal = -5 # -8  #  -3. # -8 # -5. #  -3. #
    def loss_and_grad(u):
        global i_iter
        sim.reset(True)
        
        tot_df_dq = np.zeros(ndof_r * num_steps)
        
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i: ndof_u * (i + 1)])
            sim.forward(1)
            cur_q = sim.get_q()
            tot_df_dq[ndof_r * (i + 1) + y_idx] = 2 * (cur_q[y_idx] - y_goal)
        
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
    
    def callback_func(u, render = False):
        # global reconstructor
        sim.reset(False)
        
        tot_qs = []
        for i in range(num_steps): ## 
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
            q = sim.get_q()
            if (i + 1) % nn_mod_steps:
                tot_qs.append(q)
        
        # tot_qs = np.stack(tot_qs, axis=0) ## nn rendering steps ##
        # # forward 1) kienmatic state transformation; 2) rasterization; and 3) the rendering process ## 
        # # backward 1) ---> need target images; 2) need gradients; 3) needs losses and the way of getting supervision from pixels # 
        # tot_qs = torch.from_numpy(tot_qs).float()
        # xml_fn = os.path.join("/home/xueyi/diffsim/DiffHand/assets", f"{args.model}.xml")
        # robots = load_utils.parse_data_from_xml(xml_fn)
        # robot_name_to_robot = {"active": robots[0], "passive": robots[1]}
        # reconstructor = Reconstructor(args=args, robot_name_to_robot=robot_name_to_robot)
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
            SimRenderer.replay(sim, record = True, record_path = "./save_res/res_gif/optim_with_goals.gif")
        pass

    res = scipy.optimize.minimize(loss_and_grad, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func) # func and with goals ##



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