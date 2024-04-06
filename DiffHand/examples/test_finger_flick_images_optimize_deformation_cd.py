'''
optimize an open-loop action sequence for finger to push a box to a target location
'''
from utils.renderer import SimRenderer
import numpy as np
import scipy.optimize
import redmax_py as redmax
import taichi as ti
from scipy.spatial.transform import Rotation as R
# import torch
# from examples.utils import volume_render as volume_render
from examples.utils import volume_render_deformation as volume_render
from examples.utils import data_utils as data_utils
import math
from imageio import imwrite, imread

import time
import os
import torch

### getting to other goals/optimizable variables/different scenes ###
### the rendering process -> taichi? only density? and what is the one 
### the realistic scenes here? ###


## data oriented 

@ti.data_oriented
class Reconstructor:
    def __init__(self, args) -> None: 
        # self.sim = redmax.make_sim("TorqueFingerFlick-Demo", "BDF2")
        
        self.nn_links = 3
        self.nn_samples = 500
        self.dim = 3
        # self.n_timesteps =  11 + 1 #  10 + 1
        ## self.n_timesteps = 11 + 1
        
        
        self.n_timesteps = args.num_steps // args.mod_sv_timesteps
        self.n_timesteps = self.n_timesteps + 1
        
        # self.dx = args.dx ## single iamges comparisons ### ### single images comparison ### 
        self.args = args
        ## TODO: 1) sample points from the cubes; 2) transform sampled points via the state at each time step; 3) rasterize points; 4) get the rendering loss ##
        self.act_xs = []
        self.act_states = []
        # need joint direction, joint offset and the joint states to determine the transformation matrices of them
        self.active_link_dofs = [1, 1, 1, 1]
        self.passive_link_dof = 3
        # self.active_joint_dirs = [
        #     ti.Vector([0., 0., -1.], dt=ti.float32) for _ in range(self.nn_links) ## 
        # ]
        
        # self.active_joint_dirs = ti.Matrix.rows(self.active_joint_dirs)
        
        ### active joint dirs ### ### active joint dirs ### ### 
        active_joint_dirs = [
            ti.Vector([0., 0., -1.], dt=ti.float32) for _ in range(self.nn_links)
        ]
        self.active_joint_dirs = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.nn_links))
        for i_link in range(self.nn_links):
            self.active_joint_dirs[i_link] = active_joint_dirs[i_link]
        
        
        active_joint_offsets = [ ### joint offsets ###
            ti.Vector([0., 3.2, 0.], dt=ti.float32), 
            ti.Vector([4., 3.2, 0.], dt=ti.float32),
            ti.Vector([6., 3.2, 0.], dt=ti.float32),
        ]
        
        self.active_joint_offsets = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.nn_links))
        for i_link in range(self.nn_links):
            self.active_joint_offsets[i_link] = active_joint_offsets[i_link]
        
        
        ### initial configurations ###
        self.active_links_centers = [
            np.array([ 2, 3.2, 0. ], dtype=np.float32), 
            np.array([ 5, 3.2, 0. ], dtype=np.float32), 
            np.array( [6.5, 3.2, 0. ], dtype=np.float32)
        ]
        self.active_links_sizes = [
            np.array([ 4, 1, 1 ], dtype=np.float32), 
            np.array([2, 1, 1,], dtype=np.float32), 
            np.array([1., 1., 1.], dtype=np.float32)
        ]
        
        
        self.neighbour = (3,) * self.dim
        ### the i_link of the active links ###
        # for i_link in range(self.nn_links):
        #     # cur_link_dof = 1 if i_link < self.nn_links - 1 else 3
        #     cur_link_Fxs = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, self.nn_samples))
            
        #     ### current link states ###
        #     cur_link_states = ti.Vector.field(n=self.active_link_dofs[i_link], dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps,)) ## cur_link_states ##
            
        #     self.act_xs.append(cur_link_Fxs)
        #     self.act_states.append(cur_link_states)
        ##### init act xs #####
        self.act_xs = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.nn_links, self.n_timesteps, self.nn_samples))
        
        
        self.init_act_xs = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.nn_links, self.nn_samples))
        
        self.act_states = ti.Vector.field(n=self.active_link_dofs[0], dtype=ti.float32, needs_grad=True, shape=(self.nn_links, self.n_timesteps,))
        
        self.passive_x = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, self.nn_samples))
        self.init_passive_x = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.nn_samples))
        
        self.passive_state = ti.Vector.field(n=self.passive_link_dof, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, )) ### passive link center; link size ###
        
        self.passive_link_center = np.array([5.2, 0.5, 0.0], dtype=np.float32)
        self.passive_link_size = np.array([1, 1, 1], dtype=np.float32)
                
        print(f"start sampling points from the initial condition")
        # self.sample_points()

        
        #### the gradient to each deformation vector at each time ###
        ### the deformation vector is calculated via the current position and the target position ####
        ### then connect it with the point deformations and the gradient w.r.t. point state at each time step ###
        ### self.sample_points ## chagne buffer here for init ##
        # add the grid buffer and the image buffer here for initialization #
        self.grid_res = 128 #  256 # 128
        self.grid_res = args.density_res
        self.dx = 1. / float(self.grid_res)
        self.img_res = 512
        ### use number of views = 14 ###
        # self.n_views = 20 #  7 # 3 #  1 #   7 # 14 # 1 #  100 # 14 # 100 #  14 #  7 # 14 ### 
        self.n_views = args.n_views
        # self.buffered_grids = [
        #     ti.field(dtype=ti.float32, shape=(self.grid_res, self.grid_res, self.grid_res), needs_grad=True) for _ in range(self.n_timesteps)
        # ]
        #### 
        ### buffered images #### target images ####
        self.buffered_images = [ # self.
            ti.field(dtype=ti.float32, shape=(self.n_views, self.img_res, self.img_res), needs_grad=True) for _ in range(self.n_timesteps)
        ]
        self.target_images = [
            ti.field(dtype=ti.float32, shape=(self.n_views, self.img_res, self.img_res)) for _ in range(self.n_timesteps)
        ]
        
        
        self.buffered_images_deformation = [ # self.
            ti.Vector.field(n=2, dtype=ti.float32, shape=(self.n_views, self.img_res, self.img_res), needs_grad=True) for _ in range(self.n_timesteps)
        ]
        
        self.buffered_grids = ti.field(dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), needs_grad=True)
        
        
        # self.buffered_grids_deformation_tot_frames = ti.Vector.field(n=3, dtype=ti.float32, shape=(self.tot_frames, self.density_res, self.density_res, self.density_res), needs_grad=True)
        
        ## buffered_grids_deformation, act_xs_deformation, passive_x_deformation ## ### grid deformation ####
        self.buffered_grids_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res), needs_grad=True)
        
        
        self.act_xs_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.nn_links, self.n_timesteps, self.nn_samples))
        self.passive_x_deformation = ti.Vector.field(n=self.dim, dtype=ti.float32, needs_grad=True, shape=(self.n_timesteps, self.nn_samples))
        # self.buffered_images = ti.field(dtype=ti.float32, shape=(self.n_timesteps, self.img_res, self.img_res), needs_grad=True)
        ## ## ## buffered grids ###
        ## intermediate grids for exporting ##
        self.intermediate_buffered_grids = ti.field(dtype=ti.float32, shape=(self.grid_res, self.grid_res, self.grid_res))
        
        ### and load the target space from the numpy array value ###
        self.target_grid_space = ti.field(dtype=ti.float32, shape=(self.n_timesteps, self.grid_res, self.grid_res, self.grid_res))
        
        # exported_points = {
        #     'act': cur_exported_act_points, 
        #     'passive': cur_exported_passive_points
        # }
        # target_act_mesh_verts, target_passive_mesh_verts #
        self.target_act_mesh_verts = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.nn_links, self.n_timesteps, self.nn_samples))
        self.target_passive_mesh_verts = ti.Vector.field(n=self.dim, dtype=ti.float32, shape=(self.n_timesteps, self.nn_samples))
        ### mesh vertices ## 
        
        
        self.loss = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
        self.loss.fill(0.)
        
        self.render = volume_render.VolumeRender(res=args.res, density_res=self.grid_res, dx=args.dx, n_views=self.n_views, torus_r1=args.torus_r1, torus_r2=args.torus_r2, fov=args.fov, camera_origin_radius=args.camera_origin_radius, marching_steps=args.marching_steps, learning_rate=args.learning_rate, tot_frames=self.n_timesteps)
        
        ### deformation gradients? to F_xs ##
        # self.maxx_val = 12.; self.minn_val = -1.
        self.maxx_val = 14.; self.minn_val = -1.
        self.maxx_val = 17.; self.minn_val = -3.
        self.extent = self.maxx_val - self.minn_val
        
        self.nrendered = True
        
        self.tot_cd_loss = 0.
        
        self.initialize()
          
    ## sample points for the points ##     
    def sample_points(self, ):
        for i_link in range(self.nn_links):
            print(f"sampling points for link {i_link}")
            cur_link_size = self.active_links_sizes[i_link] # x, y, z 
            samples_points_rnd = np.random.uniform(low=-0.5, high=0.5, size=(self.nn_samples, 3)) ## nn_samples x 3 ### nn_samples 
            samples_points_scaled = samples_points_rnd * cur_link_size.reshape(1, 3) # 0.5 scale by the link size
            samples_points = samples_points_scaled + self.active_links_centers[i_link]
            minn_samples_points = np.min(samples_points, axis=0)
            maxx_samples_points = np.max(samples_points, axis=0)
            print(f"i_link: {i_link}, minn_pts: {minn_samples_points}, maxx_pts: {maxx_samples_points}")
            for i_s in range(samples_points.shape[0]):
                self.act_xs[i_link, 0, i_s] = ti.Vector(samples_points[i_s], dt=ti.float32)
                self.init_act_xs[i_link, i_s] = ti.Vector(samples_points[i_s], dt=ti.float32)
        samples_points = np.random.uniform(low=-0.5, high=0.5, size=(self.nn_samples, 3)) ## nn_samples x 3
        samples_points = samples_points * self.passive_link_size.reshape(1, 3) 
        samples_points = samples_points + self.passive_link_center
        print(f"Sampling points for the passive link")
        for i_s in range(samples_points.shape[0]):
            self.passive_x[0, i_s] = ti.Vector(samples_points[i_s], dt=ti.float32)
            self.init_passive_x[i_s] = ti.Vector(samples_points[i_s], dt=ti.float32)
        act_pts_np = self.act_xs.to_numpy()
        passive_pts_np = self.passive_x.to_numpy()
        for i_link in range(self.nn_links):
            cur_link_init_pts = act_pts_np[i_link, 0]
            minn_init_pts, maxx_init_pts = np.min(cur_link_init_pts, axis=0), np.max(cur_link_init_pts, axis=0)
            print(f"i_link: {i_link}, minn_init_pts: {minn_init_pts}, maxx_init_pts: {maxx_init_pts}")
        # for i_link in range(self.nn_links):
        #     cur_sampled_pts_np = self.init_act_xs.
    
    def load_target_grid_space(self, target_grid_space_sv_fn):
        target_grid_numpy = np.load(target_grid_space_sv_fn, allow_pickle=True)
        self.target_grid_space.from_numpy(target_grid_numpy) ### from numpy
        
    # exported_points = {
    #     'act': cur_exported_act_points, 
    #     'passive': cur_exported_passive_points ### 
    # }
    # target_act_mesh_verts, target_passive_mesh_verts #
    def load_target_mesh_verts(self, verts_fn):
        act_passive_mesh = np.load(verts_fn, allow_pickle=True).item()
        act_mesh = act_passive_mesh["act"]
        passive_mesh = act_passive_mesh["passive"]
        # print()
        for i_link in range(act_mesh.shape[0]):
            cur_link_init_mesh = act_mesh[i_link, 0]
            minn_cur_init_mesh, maxx_cur_init_mesh = np.min(cur_link_init_mesh, axis=0), np.max(cur_link_init_mesh, axis=0)
            print(f"minn_cur_init_mesh: {minn_cur_init_mesh}, maxx_cur_init_mesh: {maxx_cur_init_mesh}")
        self.target_act_mesh_verts.from_numpy(act_mesh)
        self.target_passive_mesh_verts.from_numpy(passive_mesh)
    
        
    @ti.kernel
    def get_loss_curr_to_target_grid_spaces(self,):
        # self.target_grid_space
        # self.buffered_grids
        for I in ti.grouped(self.target_grid_space):
            self.loss[None] += ((self.target_grid_space[I] - self.buffered_grids[I]) ** 2) / float((self.grid_res ** 3) * self.n_timesteps)
    
    @ti.kernel
    def get_loss_curr_to_target_active(self,):
        for I in ti.grouped(self.target_act_mesh_verts):
            cur_vert_mse_loss = (self.target_act_mesh_verts[I] - self.act_xs[I]) ** 2
            self.loss[None] += (cur_vert_mse_loss.x + cur_vert_mse_loss.y + cur_vert_mse_loss.z) / float(self.nn_links * self.nn_samples)
            # self.loss[None] += ((self.target_act_mesh_verts[I] - self.act_xs[I]) ** 2) / float()
    
    @ti.kernel
    def get_loss_curr_to_target_passive(self,):
        for I in ti.grouped(self.target_passive_mesh_verts):
            cur_vert_mse_loss = (self.target_passive_mesh_verts[I] - self.passive_x[I]) ** 2
            nn_passive_scale = 10
            self.loss[None] += (cur_vert_mse_loss.x + cur_vert_mse_loss.y + cur_vert_mse_loss.z) * nn_passive_scale / float(self.nn_samples) ### the loss from the passive object
    
    @ti.kernel
    def get_loss_curr_to_target_rendered_images_with_frame(self, cur_image: ti.template(), target_image: ti.template()):
        for I in ti.grouped(cur_image):
            has_density_weight = 1.
            if cur_image[I] > 0. or target_image[I] > 0.:
                self.loss[None] += (has_density_weight * (cur_image[I] -  target_image[I]) ** 2) / float(self.img_res * self.img_res * self.n_timesteps * self.n_views)
    
    def get_loss_curr_to_target_rendered_images_with_frame_cd_th(self, cur_image: ti.template(), cur_image_def: ti.template(), target_image: ti.template()):
        cuda_idx = 1
        cur_image_th = torch.from_numpy(cur_image.to_numpy()).float().cuda(cuda_idx)
        cur_target_th = torch.from_numpy(target_image.to_numpy()).float().cuda(cuda_idx)
        # cur_image_grad = torch.zeros_like(cur_image_th)
        cur_image_def_th = torch.from_numpy(cur_image_def.to_numpy()).float().cuda(cuda_idx)
        cur_image_def_grad = torch.zeros_like(cur_image_def_th)
        for i_v in range(cur_image_th.size(0)):
            cur_img_pts = []
            cur_img_pts_deformation = []
            # cur_img_pts_deformation = ## cd loss ##
            cur_img_pts = torch.zeros((cur_image_th.size(1), cur_image_th.size(2), 2), dtype=torch.float32).cuda(cuda_idx)
            cur_img_pts[:, :, 0] = torch.arange(cur_image_th.size(1)).float().cuda(cuda_idx).unsqueeze(-1) / float(cur_image_th.size(1)) ## nn x 1 -> set the x axis
            cur_img_pts[:, :, 1] = torch.arange(cur_image_th.size(2)).float().cuda(cuda_idx).unsqueeze(0) / float(cur_image_th.size(2)) ## nn x 1 -> set the x axis
            cur_img_pts = cur_img_pts.contiguous().view(cur_img_pts.size(0) * cur_img_pts.size(1), 2).contiguous()
            
            # print(f"cur_img_pts: {cur_img_pts.size()}, cur_image_th: {cur_image_th.size()}, cur_image_def_th: {cur_image_def_th.size()}")
            cur_img_pts = cur_img_pts[cur_image_th[i_v].contiguous().view(cur_image_th.size(1) * cur_image_th.size(2)).contiguous() > 0.]
            cur_img_pts_deformation = cur_image_def_th.clone().contiguous().view(cur_image_def_th.size(0), cur_image_def_th.size(1) * cur_image_def_th.size(2), -1)[i_v, cur_image_th[i_v].contiguous().view(cur_image_th.size(1) * cur_image_th.size(2)).contiguous() > 0.]
            # for i_x in range(cur_image_th.size(1)):
            #     for i_y in range(cur_image_th.size(2)):
            #         if cur_image_th[i_v, i_x, i_y].item() > 0.:
            #             cur_img_pts.append([float(i_x) / float(cur_image_th.size(1)), float(i_y) / float(cur_image_th.size(2))])
            #             cur_img_pts_deformation.append(cur_image_def_th[i_v, i_x, i_y])
            
            cur_target_pts = torch.zeros((cur_image_th.size(1), cur_image_th.size(2), 2), dtype=torch.float32).cuda(cuda_idx)
            cur_target_pts[:, :, 0] = torch.arange(cur_image_th.size(1)).float().cuda(cuda_idx).unsqueeze(-1) / float(cur_image_th.size(1)) ## nn x 1 -> set the x axis
            cur_target_pts[:, :, 1] = torch.arange(cur_image_th.size(2)).float().cuda(cuda_idx).unsqueeze(0) / float(cur_image_th.size(2)) ## nn x 1 -> set the x axis
            cur_target_pts = cur_target_pts.contiguous().view(cur_target_pts.size(0) * cur_target_pts.size(1), 2).contiguous()
            
            # print(f"cur_target_pts: {cur_target_pts.size()}, cur_target_th: {cur_target_th.size()}")
            cur_target_pts = cur_target_pts[cur_target_th[i_v].contiguous().view(cur_image_th.size(1) * cur_image_th.size(2)).contiguous() > 0.]
            
            # cur_img_pts_deformation = cur_image_def_th[i_v, cur_image_th > 0.]
            
            # cur_target_pts = []
            # for i_x in range(cur_target_th.size(1)):
            #     for i_y in range(cur_target_th.size(2)):
            #         if cur_target_th[i_v, i_x, i_y].item() > 0.:
            #             cur_target_pts.append([float(i_x) / float(cur_image_th.size(1)), float(i_y) / float(cur_image_th.size(2))])
            # cur_img_pts = torch.tensor(cur_img_pts, dtype=torch.float32, requires_grad=True).cuda()
            # print(len(cur_img_pts_deformation), cur_img_pts_deformation[0].size())
            
            
            cur_img_pts_deformation = cur_img_pts_deformation.clone() # torch.stack(cur_img_pts_deformation, dim=0).cuda()
            cur_img_pts_deformation.requires_grad_ = True
            cur_img_pts_deformation.requires_grad = True
            
            # cur_img_pts_deformation # 
            # maxx_cur_img_pts_deformation, _ = torch.max(cur_img_pts_deformation, dim=0)
            # minn_cur_img_pts_deformation, _ = torch.min(cur_img_pts_deformation, dim=0)
            # maxx_cur_img_pts, _ = torch.max(cur_img_pts, dim=0)
            # minn_cur_img_pts, _ = torch.min(cur_img_pts, dim=0)
            # print(f"maxx_cur_img_pts_deformation: {maxx_cur_img_pts_deformation}, minn_cur_img_pts_deformation: {minn_cur_img_pts_deformation}, maxx_cur_img_pts: {maxx_cur_img_pts}, minn_cur_img_pts: {minn_cur_img_pts}") ### maxx_cur_img
            
            cur_img_pts.requires_grad_ = True
            cur_img_pts.requires_grad = True
            # cur_img_pts_deformation = torch.zeros((cur_img_pts_deformation_ori.size(0), cur_img_pts_deformation_ori.size(1)), dtype=torch.float32, requires_grad=True).cuda()
            # cur_img_pts_deformation.data = cur_img_pts_deformation_ori.data
            # img pts deformation ##
            cur_img_pts = cur_img_pts_deformation + cur_img_pts ### nn_img_pts x 3 
            # cur_target_pts = torch.tensor(cur_target_pts, dtype=torch.float32).cuda()
            
            # print(f"cur_img_pts: {cur_img_pts.size()}, cur_target_pts: {cur_target_pts.size()}")
            
            dist_img_pts_target_pts = torch.sum( ### 
                (cur_img_pts.unsqueeze(1) - cur_target_pts.unsqueeze(0)) ** 2, dim=-1
            )
            minn_dist, minn_idx = torch.min(dist_img_pts_target_pts, dim=-1) ## nn_img_pts
            minn_dist_rever, minn_idx_rever = torch.min(dist_img_pts_target_pts, dim=0)
            cd_loss = 0.5 * (minn_dist.mean() + minn_dist_rever.mean())
            cd_loss.backward()
            self.tot_cd_loss += cd_loss / float(cur_image_th.size(0) * self.n_timesteps)
            # print(f"cd_loss: {cd_loss}")
            # cur_image_pts_grad = cur_img_pts.grad
            cur_image_cur_view_def_grad = cur_img_pts_deformation.grad
            
            cur_img_pts_Xp = cur_img_pts * torch.tensor([cur_image_th.size(1), cur_image_th.size(2)], dtype=torch.float32).unsqueeze(0).cuda(cuda_idx)
            cur_img_pts_Xp = cur_img_pts_Xp.long() ### nn_pts x 2 3##
            ### 
            # print("grad:", cur_img_pts_deformation.grad)
            # print(f"cur_img_pts_Xp: {cur_img_pts_Xp.size()}")
            # print(f"cur_image_def_grad[i_v, cur_img_pts_Xp, :]: {cur_image_def_grad[i_v, cur_img_pts_Xp[:, 0],  cur_img_pts_Xp[:, 1]].size()}", f"cur_img_pts_Xp: {cur_img_pts_Xp.size()}")
            # cur_image_def_grad[i_v, cur_img_pts_Xp, :] = cur_image_def_grad #### assign the gradients
            ### cur_img_pts_Xp ###
            # cur_image_def_grad[i_v, cur_img_pts_Xp] = cur_image_cur_view_def_grad
            
            # maxx_cur_img_pts_Xp, _ = torch.max(cur_img_pts_Xp, dim=0)
            # minn_cur_img_pts_Xp, _ = torch.min(cur_img_pts_Xp, dim=0)
            # print(f"maxx_cur_img_pts_Xp: {maxx_cur_img_pts_Xp}, minn_cur_img_pts_Xp: {minn_cur_img_pts_Xp}, cur_image_def_grad: {cur_image_def_grad.size()}")
            
            cur_image_def_grad[i_v, cur_img_pts_Xp[:, 0], cur_img_pts_Xp[:, 1]] = cur_image_cur_view_def_grad
            # for i_p in range(cur_img_pts_Xp.size(0)):
            #     cur_image_def_grad[i_v, cur_img_pts_Xp[i_p, 0], cur_img_pts_Xp[i_p, 1]] = cur_image_cur_view_def_grad[i_p] ##
            
        return cur_image_def_grad.detach().cpu().numpy()
            
    
    
    def get_loss_curr_to_target_rendered_images(self, ):
        for i_fr in range(self.n_timesteps):
            self.get_loss_curr_to_target_rendered_images_with_frame(self.buffered_images[i_fr], self.target_images[i_fr])
    
    def backward_loss_images(self, ):
        for i_fr in reversed(range(self.n_timesteps)):
            self.get_loss_curr_to_target_rendered_images_with_frame.grad(self.buffered_images[i_fr], self.target_images[i_fr])
    
    
    def backward_image_deformations(self, ):
        for i_fr in range(self.n_timesteps):
            ## for each view ##
            cur_fr_image_def_grad = self.get_loss_curr_to_target_rendered_images_with_frame_cd_th(self.buffered_images[i_fr], self.buffered_images_deformation[i_fr], self.target_images[i_fr])
            # set the buffered images deformation grad ##
            self.buffered_images_deformation[i_fr].grad.from_numpy(cur_fr_image_def_grad)
        
    
    
    def initialize(self, ): ### initialization ###
        # for i_link in range(self.nn_links):
        #     self.act_xs[i_link].grad.fill(0.)
        #     self.act_states[i_link].grad.fill(0.)
        self.act_xs.fill(0.)
        self.act_xs.grad.fill(0.)
        self.act_states.fill(0.)
        self.act_states.grad.fill(0.)
        self.passive_x.fill(0.)
        self.passive_x.grad.fill(0.)
        self.passive_state.fill(0.)
        self.passive_state.grad.fill(0.)
        
        ### buffered grids; buffered iamges ###
        # for i_fr in range(self.n_timesteps):
        #     self.buffered_grids[i_fr].fill(0.)
        #     self.buffered_grids[i_fr].grad.fill(0.)
        #     self.buffered_images[i_fr].fill(0.)
        #     self.buffered_images[i_fr].grad.fill(0.)
        self.buffered_grids.fill(0.)
        self.buffered_grids.grad.fill(0.)
        
        for i_fr in range(self.n_timesteps):
            self.buffered_images[i_fr].fill(0.)
            self.buffered_images[i_fr].grad.fill(0.)
            
            self.buffered_images_deformation[i_fr].fill(0.)
            self.buffered_images_deformation[i_fr].grad.fill(0.)
            
        ## buffered_grids_deformation, act_xs_deformation, passive_x_deformation ##
        self.buffered_grids_deformation.fill(0.)
        self.buffered_grids_deformation.grad.fill(0.)
        self.act_xs_deformation.fill(0.)
        self.act_xs_deformation.grad.fill(0.)
        self.passive_x_deformation.fill(0.)
        self.passive_x_deformation.grad.fill(0.)
        
        # self.sample_points()
        
        ## init ##
        if optim:
            ## the initial vertex positions ###
            for i_link in range(self.nn_links):
                for i_s in range(self.nn_samples): ### i_s-th sample #
                    # target_act_mesh_verts
                    # self.act_xs[i_link, 0, i_s] = self.init_act_xs[i_link, i_s]
                    self.act_xs[i_link, 0, i_s] = self.target_act_mesh_verts[i_link, 0, i_s]
            for i_s in range(self.nn_samples):
                # self.passive_x[0, i_s] = self.init_passive_x[i_s]
                self.passive_x[0, i_s] = self.target_passive_mesh_verts[0, i_s]
            # minn_act_xs, maxx_act_xs = np.min(self.act_xs)
            
            cur_exported_act_points = self.act_xs.to_numpy()
            cur_exported_target_points = self.target_act_mesh_verts.to_numpy() ### nn_links x nn_timesteps x nn_samples
            
            for i_link in range(self.nn_links):
                cur_link_init_pts = cur_exported_act_points[i_link, 0]
                minn_pts, maxx_pts = np.min(cur_link_init_pts, axis=0), np.max(cur_link_init_pts, axis=0)
                print(f"i_link: {i_link}, minn_pts: {minn_pts}, maxx_pts: {maxx_pts}")
                
                cur_link_init_target_pts = cur_exported_target_points[i_link, 0]
                minn_target_pts, maxx_target_pts = np.min(cur_link_init_target_pts, axis=0), np.max(cur_link_init_target_pts, axis=0)
                print(f"i_link: {i_link}, minn_target_pts: {minn_target_pts}, maxx_target_pts: {maxx_target_pts}")
                
                # cur_link_real_init_pts = init_act_pts_numpy[i_link]
                # minn_real_init_pts, maxx_real_init_pts = np.min(cur_link_real_init_pts, axis=0), np.max(cur_link_real_init_pts, axis=0)
                # print(f"i_link: {i_link}, minn_real_init_pts: {minn_real_init_pts}, maxx_real_init_pts: {maxx_real_init_pts}")
        
        else:
            self.sample_points()

        self.act_xs.grad.fill(0.)
        self.passive_x.grad.fill(0.)

        self.loss.fill(0.)
        self.loss.grad.fill(0.)
        
        
        self.render.initialize()
        self.tot_cd_loss = 0.
        
    
    
    def apply_initial_transformation(self, init_q):
        # init_q: 6-dim numpy array 
        # TODO: a more general initlization since the initial state of the passive object may not be zero
        for i_link in range(self.nn_links):
            # axis angle?
            cur_link_angle = init_q[i_link].item()
            # TODO: a more general axis-angle implementation is needed 
            # TODO: here we use the rotation axis as [0., 0., -1.]
            r = R.from_euler('z', -1. * cur_link_angle, degrees=False)
            rot_mtx = r.as_matrix()
            rot_mtx = ti.Matrix(rot_mtx, dt=ti.float32)
            # rot_mtx (pts - joint) + joint
            for i_s in range(self.nn_samples):
                self.act_xs[i_link, 0, i_s] = rot_mtx @ (self.act_xs[i_link, 0, i_s] - self.active_joint_offsets[i_link]) + self.active_joint_offsets[i_link]
    
    
    @ti.kernel
    def get_rendered_images_per_frame(self, i_fr: ti.i32):
        ### the i_fr-th frame in the ti.field ###
        tmp_buffered_images = ti.field(dtype=ti.float32, shape=(self.n_views, self.img_res, self.img_res), needs_grad=True)
        for I in ti.grouped(self.buffered_images[i_fr]):
            tmp_buffered_images[I] = self.render.buffered_images[i_fr, I] ### buffered images ###
            # self.buffered_images[i_fr][I] = self.render.buffered_images[i_fr, I] ### buffered images ###
        return tmp_buffered_images

    def get_rendered_iamges_from_render(self, ):
        # buffered_images_tot_frames
        for i_frame in range(self.n_timesteps):
            ### self.buffered_images[i_frame]
            self.get_rendered_images_per_frame(i_fr=i_frame)
            
    def backward_get_rendered_images_from_render(self, ):
        for i_frame in reversed(range(self.n_timesteps)):
            self.get_rendered_images_per_frame.grad(i_fr=i_frame)
            
    
    def run_render(self, ):
        tot_rendered_imgs = []
        # self.render.fill_density_tot_frames(self.buffered_grids)
        # self.render.fill_density_tot_frames
        self.render.fill_density_fr_ti_with_fr_density_fr_single_loop(self.buffered_grids)
        self.render.fill_density_deformation_fr_ti_with_fr_density_fr_single_loop(self.buffered_grids_deformation) ### fill density ###
        for i_frame in range(self.n_timesteps):
            #### fill density fr ti ###
            t1 = time.time()
            # self.render.fill_density_fr_ti_with_fr(self.buffered_grids, i_frame) # fill density fr ti #
            # self.render.fill_density_fr_ti_with_fr_density_fr(self.buffered_grids, i_frame)
            t2 = time.time()
            # print(f"time used for filling density field: {t2 - t1}")
            for view in range(self.render.n_views): ### i_frame ###
                t3 = time.time()
                # self.render.ray_march_single_loop(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0,
                #           view)
                # self.render.ray_march_single_loop_with_density_frames(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                # self.render.ray_march_single_loop_with_density_frames(self.buffered_images[i_frame], math.pi / (self.render.n_views * 4) * view - math.pi / 2.0, view, i_frame)
                # self.render.ray_march_single_loop_with_density_frames(self.buffered_images[i_frame], math.pi / (self.render.n_views * 2) * view - math.pi / 2.0, view, i_frame)
                # self.render.ray_march_single_loop_with_density_frames(self.buffered_images[i_frame], math.pi / (self.render.n_views * 8) * view - math.pi / 2.0, view, i_frame)
                # if not self.args.use_deformation:
                #     self.render.ray_march_single_loop_with_density_frames(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                # else:
                #     # buffered_images_deformation
                #     self.render.ray_march_single_loop_with_density_frames_with_deformation(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                
                
                # self.render.ray_march_single_loop_with_density_frames_with_deformation(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                
                self.render.ray_march_single_loop_with_density_frames_with_deformation(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / (self.render.n_views * self.args.view_divide) * view - math.pi / 2.0, view, i_frame)
                # # ## i_fr: ti.i32, angle: ti.f32, view_id: ti.i32 ### ## ray marching ##
                # self.render.ray_march_single_loop_with_frame(
                #     i_frame, math.pi / self.render.n_views * view - math.pi / 2.0, view
                # )
                t4 = time.time()
                # print(f"view {view} rendering used time {t4 - t3}")
            
        # self.get_rendered_iamges_from_render()
        
        for i_frame in range(self.n_timesteps):
            ### views and images ###
            views = self.buffered_images[i_frame].to_numpy() # 
            # tot_rendered_imgs.append(views)
            for view in range(self.render.n_views): ## 
                img = views[view]
                m = np.max(img)
                if m > 0:
                    img /= m # img /= m #
                img = 1 - img
                ### #### imwrite for iamges #### #
                imwrite(
                    os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_nv_{:d}.png".format(i_frame, view, optim, passive_only, self.n_views)),
                    (255 * img).astype(np.uint8))
            
            #### save target views ###
            target_views = self.target_images[i_frame].to_numpy()
            for view in range(self.render.n_views): ## 
                img = target_views[view]
                m = np.max(img)
                if m > 0:
                    img /= m # img /= m #
                img = 1 - img
                ### #### imwrite for iamges #### #
                imwrite(
                    os.path.join(self.args.image_sv_folder, "image_frame_{:04d}_view_{:04d}_optim_{}_passive_{}_target_nv_{:d}.png".format(i_frame, view, optim, passive_only, self.n_views)),
                    (255 * img).astype(np.uint8))
            
        #### tot rendered imgs ####
        # tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ## nn_frames x nn_views x nn_img_res x nn_img_res
        tot_rendered_imgs = [cur_frame_image.to_numpy() for cur_frame_image in self.buffered_images]
        tot_rendered_imgs = np.stack(tot_rendered_imgs, axis=0) ### numpy 
        # tot_rendered_imgs = self.buffered_images[i_frame].to_numpy() # 
        
        rendered_imgs_sv_fn = os.path.join(self.args.image_sv_folder, f"rendered_images_optim_{optim}_passive_{passive_only}_nv_{self.n_views}_n_iter_{i_iter}.npy")
        np.save(rendered_imgs_sv_fn, tot_rendered_imgs) ## rendered imgaes ##
        print(f"Rendered images saved to {rendered_imgs_sv_fn}")
    
    def run_render_grad(self, ):
        # self.backward_get_rendered_images_from_render() ### backward rendered images ####
        
        for i_frame in reversed(range(self.n_timesteps)):
            for view in reversed(range(self.render.n_views)):
                # self.render.ray_march_single_loop.grad(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0,
                #           view)
                ## from image grad to density grad ##
                # self.render.ray_march_single_loop_with_density_frames.grad(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                # if not self.args.use_deformation:
                #     self.render.ray_march_single_loop_with_density_frames.grad(self.buffered_images[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                # else:
                #     #### ray_march_single_loop_with_density_frames_with_deformation ####
                #     self.render.ray_march_single_loop_with_density_frames_with_deformation.grad(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / self.render.n_views * view - math.pi / 2.0, view, i_frame)
                ### iamge deformation ###
                self.render.ray_march_single_loop_with_density_frames_with_deformation.grad(self.buffered_images[i_frame], self.buffered_images_deformation[i_frame], math.pi / (self.render.n_views * self.args.view_divide)  * view - math.pi / 2.0, view, i_frame)
                # self.render.ray_march_single_loop_with_frame.grad(
                #     i_frame, math.pi / self.render.n_views * view - math.pi / 2.0, view
                # )
            # self.render.fill_density_fr_ti_with_fr.grad(self.buffered_grids, i_frame)
            # self.render.fill_density_fr_ti_with_fr_density_fr.grad(self.buffered_grids, i_frame)
        
        # print(f"Start backwarding density to deformation...")
        # self.render.get_density_grad_finite_difference()
        
        # if i_iter == 0:
        #     render_density_grad_finite_differences = self.render.density_grad_finite_difference.to_numpy()
        #     density_grad_fd_sv_fn = os.path.join(args.image_sv_folder, f"density_grad_fd.npy")
        #     np.save(density_grad_fd_sv_fn, render_density_grad_finite_differences)
        #     print(f"density grad fd saved to {density_grad_fd_sv_fn}") ### density grad fd sv ####
        
        # self.render.backward_density_to_deformation()
        self.render.fill_density_deformation_fr_ti_with_fr_density_fr_single_loop.grad(self.buffered_grids_deformation)
        self.render.fill_density_fr_ti_with_fr_density_fr_single_loop.grad(self.buffered_grids)
    
    
    def load_target_rendered_images(self, rendered_images_sv_fn):
        rendered_images = np.load(rendered_images_sv_fn, allow_pickle=True)
        for i_fr in range(self.n_timesteps): ## load images; load images ##
            self.target_images[i_fr].from_numpy(rendered_images[i_fr])
            
    ### construct 
    def set_states_via_q(self, qs):
        for i_frame in range(len(qs)):
            cur_q = qs[i_frame]
            for i_link in range(self.nn_links):
                # print(f"cur_q: {cur_q}")
                # self.act_states[i_link][i_frame + 1][0] = cur_q[i_link] # .item()
                self.act_states[i_link, i_frame + 1][0] = cur_q[i_link] # .item()
            self.passive_state[i_frame + 1] = ti.Vector(cur_q[self.nn_links: ], dt=ti.float32)
        
    

    @ti.func
    def rotation_matrix_from_axis_angle(self, axis: ti.template(), angle: ti.template()): # rotation_matrix_from_axis_angle -> 
        sin_ = ti.math.sin(angle)
        cos_ = ti.math.cos(angle)
        u_x, u_y, u_z = axis.x, axis.y, axis.z
        u_xx = axis.x * axis.x
        u_yy = axis.y * axis.y
        u_zz = axis.z * axis.z
        u_xy = axis.x * axis.y
        u_xz = axis.x * axis.z
        u_yz = axis.y * axis.z
        rot_mtx = ti.Matrix.cols(
            [
                ti.Vector([cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dt=ti.float32), 
                ti.Vector([u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dt=ti.float32), 
                ti.Vector([u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dt=ti.float32)
            ]
        )
        # rot_mtx_numpy = rot_mtx.to_numpy()
        # rot_mtx_at_rot_mtx = rot_mtx @ rot_mtx.transpose()
        # print(rot_mtx_at_rot_mtx)
        return rot_mtx
        
    @ti.kernel
    def rasterize_act_cur_step_cur_frame(self, i_link: ti.i32, i_fr: ti.i32): ## i_fr >= 1 here 
        for i_v in range(self.nn_samples):  ### nn_samples ###
            # i_link, i_vert = self.get_i_link_i_v(i_v=i_v)
            x = (self.act_xs[i_link, i_fr, i_v] - self.minn_val) / self.extent
            # x = self.obj_F_xs[self.active_obj_idx][i_f, i_v]
            # x_prev = (self.act_xs[i_link, i_fr - 1, i_v] - self.minn_val) / self.extent
            # x_def = x - x_prev
            
            # x_def = ti.Vector.zero(dt=ti.float32, n=3)
            # if i_fr < self.n_timesteps - 1:
            #     x_nex = (self.act_xs[i_link, i_fr + 1, i_v] - self.minn_val) / self.extent ### x
            #     x_def = x_nex - x
                
            # v = self.obj_F_vs[i_obj][i_f, i_v]
            # Xp = x / self.dx # grid indexes
            Xp = x * self.grid_res
            base = ti.cast(Xp - 0.5, ti.int32)
            # print(f"x: {x}, base: {base}")
            fx = Xp - base # grid indexes 
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                # dpos = (offset - fx) * self.dx #
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                #### mass x vertices; dt x forces; affine @ dpos ####
                # self.grid_v[base + offset] += weight * (self.obj_F_ms[i_obj][i_v] * self.obj_F_vs[i_obj][i_f, i_v] + self.dt * self.obj_F_fs[i_obj][i_f, i_v] + affine @ dpos) # m * velocity #
                self.buffered_grids[i_fr, base + offset] += weight * self.args.p_mass # aggregate mass from particles ## aggregate mass from 
                # if self.args.use_deformation:
                
                self.buffered_grids_deformation[i_fr, base + offset] += weight * self.act_xs_deformation[i_link, i_fr, i_v]
                
                ## weight * def
                # self.buffered grids ### deformation of the previous one?
                # self.buffered_grids_deformation[i_fr, base + offset] += weight * x_def ###d de ### gradein 
                
    

    @ti.kernel
    def rasterize_passive_cur_step_cur_frame(self, i_fr: ti.i32):
        for i_v in range(self.nn_samples):      
            # i_link, i_vert = self.get_i_link_i_v(i_v=i_v)
            x = (self.passive_x[i_fr, i_v] - self.minn_val) / self.extent
            # x = self.obj_F_xs[self.active_obj_idx][i_f, i_v]
            
            # x_def = ti.Vector.zero(dt=ti.float32,n=3)
            # if i_fr < self.n_timesteps - 1:
            #     x_nex = (self.passive_x[i_fr + 1, i_v] - self.minn_val) / self.extent
            #     x_def = x_nex - x
                
            # v = self.obj_F_vs[i_obj][i_f, i_v]
            # Xp = x / self.dx # grid indexes
            Xp = x * self.grid_res
            base = ti.cast(Xp - 0.5, ti.int32)
            # print(f"x: {x}, base: {base}")
            fx = Xp - base # grid indexes 
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                # dpos = (offset - fx) * self.dx #
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                #### mass x vertices; dt x forces; affine @ dpos ####
                # self.grid_v[base + offset] += weight * (self.obj_F_ms[i_obj][i_v] * self.obj_F_vs[i_obj][i_f, i_v] + self.dt * self.obj_F_fs[i_obj][i_f, i_v] + affine @ dpos) # m * velocity #
                self.buffered_grids[i_fr, base + offset] += weight * self.args.p_mass # aggregate mass from particles
                # if self.args.use_deformation:
                #     # self.buffered_grids_deformation[i_fr, base + offset] += weight * self.passive_x_deformation[i_fr, i_v]
                #     self.buffered_grids_deformation[i_fr, base + offset] +=  self.passive_x_deformation[i_fr, i_v]
                # self.buffered_grids_deformation[i_fr, base + offset] += weight * x_def ###d de ### gradein 
                self.buffered_grids_deformation[i_fr, base + offset] +=  weight * self.passive_x_deformation[i_fr, i_v]

    @ti.kernel
    def backward_rasterize_passive_cur_step_cur_frame(self, i_fr: ti.i32):
        for i_v in range(self.nn_samples):      
            # i_link, i_vert = self.get_i_link_i_v(i_v=i_v)
            x = (self.passive_x[i_fr, i_v] - self.minn_val) / self.extent
            # x = self.obj_F_xs[self.active_obj_idx][i_f, i_v]

            # v = self.obj_F_vs[i_obj][i_f, i_v]
            # Xp = x / self.dx # grid indexes
            Xp = x * self.grid_res
            base = ti.cast(Xp - 0.5, ti.int32)
            # print(f"x: {x}, base: {base}")
            fx = Xp - base # grid indexes 
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                # dpos = (offset - fx) * self.dx #
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                
                self.passive_x_deformation.grad[i_fr, i_v] += weight * self.buffered_grids_deformation.grad[i_fr, base + offset]

    @ti.kernel
    def backward_rasterize_act_cur_step_cur_frame(self, i_link: ti.i32, i_fr: ti.i32): ## i_fr >= 1 here 
        for i_v in range(self.nn_samples):  ### nn_samples ###
            # i_link, i_vert = self.get_i_link_i_v(i_v=i_v)
            x = (self.act_xs[i_link, i_fr, i_v] - self.minn_val) / self.extent
            # x = self.obj_F_xs[self.active_obj_idx][i_f, i_v]
            # x_prev = (self.act_xs[i_link, i_fr - 1, i_v] - self.minn_val) / self.extent
            # x_def = x - x_prev
            
            # x_def = ti.Vector.zero(dt=ti.float32, n=3)
            # if i_fr < self.n_timesteps - 1:
            #     x_nex = (self.act_xs[i_link, i_fr + 1, i_v] - self.minn_val) / self.extent ### x
            #     x_def = x_nex - x
                
            # v = self.obj_F_vs[i_obj][i_f, i_v]
            # Xp = x / self.dx # grid indexes
            Xp = x * self.grid_res
            base = ti.cast(Xp - 0.5, ti.int32)
            # print(f"x: {x}, base: {base}")
            fx = Xp - base # grid indexes 
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                # dpos = (offset - fx) * self.dx #
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                #### mass x vertices; dt x forces; affine @ dpos ####
                # self.grid_v[base + offset] += weight * (self.obj_F_ms[i_obj][i_v] * self.obj_F_vs[i_obj][i_f, i_v] + self.dt * self.obj_F_fs[i_obj][i_f, i_v] + affine @ dpos) # m * velocity #
                # self.buffered_grids[i_fr, base + offset] += weight * self.args.p_mass # aggregate mass from particles ## aggregate mass from 
                # if self.args.use_deformation:
                self.act_xs_deformation.grad[i_link, i_fr, i_v] += weight * self.buffered_grids_deformation.grad[i_fr, base + offset]
                ## weight * def
                # self.buffered grids ### deformation of the previous one?
                # self.buffered_grids_deformation[i_fr, base + offset] += weight * x_def ###d de ### gradein 
       

    ### link step -> the articulated motion here ###
    @ti.kernel
    def act_link_step_cur_frame(self, i_link: ti.i32, i_fr: ti.i32):
        ### act
        for i_s in range(self.nn_samples): # 
            # rot_vec = ti.Vector(
            #     [self.active_joint_dirs[i_link, 0], self.active_joint_dirs[i_link, 1], self.active_joint_dirs[i_link, 2]], dt=ti.float32
            # )
            
            # rot_vecs = [self.active_joint_dirs[ii_link_idx] for ii_link_idx in range(self.nn_links)]
            # cur_rots = [self.rotation_matrix_from_axis_angle(rot_vecs[ii_link_idx], self.act_states[ii_link_idx, i_fr]) for ii_link_idx in range(self.nn_links)]
            transformed_val = ti.Vector([0., 0., 0.], dt=ti.float32)
            if i_link == 0:
                rot_vec = self.active_joint_dirs[i_link]
                cur_rot = self.rotation_matrix_from_axis_angle(rot_vec, self.act_states[i_link, i_fr])
                transformed_val = cur_rot @ (self.act_xs[i_link, 0, i_s] - self.active_joint_offsets[i_link]) + self.active_joint_offsets[i_link]
            elif i_link == 1:
                rot_vec = self.active_joint_dirs[i_link]
                cur_rot = self.rotation_matrix_from_axis_angle(rot_vec, self.act_states[i_link, i_fr])
                transformed_val = cur_rot @ (self.act_xs[i_link, 0, i_s] - self.active_joint_offsets[i_link]) + self.active_joint_offsets[i_link]
                rot_vec_p = self.active_joint_dirs[i_link - 1]
                cur_rot_p = self.rotation_matrix_from_axis_angle(rot_vec_p, self.act_states[i_link - 1, i_fr])
                transformed_val = cur_rot_p @ (transformed_val - self.active_joint_offsets[i_link - 1]) + self.active_joint_offsets[i_link - 1]
            elif i_link == 2:
                rot_vec = self.active_joint_dirs[i_link]
                cur_rot = self.rotation_matrix_from_axis_angle(rot_vec, self.act_states[i_link, i_fr])
                transformed_val = cur_rot @ (self.act_xs[i_link, 0, i_s] - self.active_joint_offsets[i_link]) + self.active_joint_offsets[i_link]
                rot_vec_p = self.active_joint_dirs[i_link - 1]
                cur_rot_p = self.rotation_matrix_from_axis_angle(rot_vec_p, self.act_states[i_link - 1, i_fr])
                transformed_val = cur_rot_p @ (transformed_val - self.active_joint_offsets[i_link - 1]) + self.active_joint_offsets[i_link - 1]
                rot_vec_pp = self.active_joint_dirs[i_link - 2] ### parent's parent link 
                cur_rot_pp = self.rotation_matrix_from_axis_angle(rot_vec_pp, self.act_states[i_link - 2, i_fr]) ### parent's parent rotation matrix
                transformed_val = cur_rot_pp @ (transformed_val - self.active_joint_offsets[i_link - 2]) + self.active_joint_offsets[i_link - 2]
            
            # self.act_xs[i_link, i_fr, i_s] = cur_rot @ (self.init_act_xs[i_link, i_s] - self.active_joint_offsets[i_link]) + self.active_joint_offsets[i_link]
            self.act_xs[i_link, i_fr, i_s] = transformed_val
    
    @ti.kernel
    def passive_link_step_cur_frame(self, i_fr: ti.i32):
        for i_s in range(self.nn_samples):
            # self.passive_x[i_fr, i_s] = self.init_passive_x[i_s] + self.passive_state[i_fr] ### passive x state value ###
            self.passive_x[i_fr, i_s] = self.passive_x[0, i_s] + self.passive_state[i_fr]

    @ti.kernel
    def backward_passive_link_step_cur_frame(self, i_fr: ti.i32):
        for i_s in range(self.nn_samples):
            self.passive_state.grad[i_fr] += self.passive_x.grad[i_fr, i_s]


    def forward_stepping(self, ): # observations
        # for i_link in range(self.nn_links):
        for i_fr in range(self.n_timesteps - 1):
            for i_link in range(self.nn_links):
                self.act_link_step_cur_frame(i_link=i_link, i_fr=i_fr + 1) # step forward the act_link
                ### just debugging here ---- if using no act link stepping here ###
                # self.rasterize_act_cur_step_cur_frame(i_link=i_link, i_fr=i_fr + 1)
        
        if not passive_only:
            for i_fr in range(self.n_timesteps):
                for i_link in range(self.nn_links):
                    self.rasterize_act_cur_step_cur_frame(i_link=i_link, i_fr=i_fr)
                
        
        for i_fr in range(self.n_timesteps - 1):
            self.passive_link_step_cur_frame(i_fr=i_fr + 1)
            ### just debugging here ---- if using no act link stepping here ###
            # self.rasterize_passive_cur_step_cur_frame(i_fr=i_fr + 1)
        for i_fr in range(self.n_timesteps):
            self.rasterize_passive_cur_step_cur_frame(i_fr=i_fr) ### for the passive object ###
    
    def backward_stepping(self, ):
        self.buffered_grids.grad.fill(0.)
        for i_fr in reversed(range(self.n_timesteps)):
            self.rasterize_passive_cur_step_cur_frame.grad(i_fr=i_fr)
            # if self.args.use_deformation:
            self.backward_rasterize_passive_cur_step_cur_frame(i_fr=i_fr)
        
        # if i_iter == 0:
        # passive_x_grad = self.passive_x.grad.to_numpy()
        # passive_x_grad_sv_fn = os.path.join(args.image_sv_folder, f"passive_x_grad_iter_{i_iter}.npy")
        # np.save(passive_x_grad_sv_fn, passive_x_grad)
        # print(f"passive_x grad saved to {passive_x_grad_sv_fn}") ### density grad fd sv ####
            
        #### setting the gradients to deformation's gradients ####
        # if self.args.use_deformation: ### set passive x gradient ###
        
        # self.passive_x.grad = self.passive_x_deformation.grad
        
        self.passive_x.grad.from_numpy(self.passive_x_deformation.grad.to_numpy())
        
        passive_deformation_grad = np.sum(self.passive_x_deformation.grad.to_numpy())
        print(f"passive_deformation_grad: {passive_deformation_grad}")
        
        passive_x_grad_sum = np.sum(self.passive_x.grad.to_numpy())
        print(f"passive_x-grad_sum: {passive_x_grad_sum}")
        
        # state_grad_numpy_array = []
        # for i_fr in range(self.n_timesteps):
        for i_fr in reversed(range(self.n_timesteps - 1)):
            
            # self.passive_link_step_cur_frame(i_fr=i_fr + 1)
            # self.rasterize_passive_cur_step_cur_frame(i_fr=i_fr + 1)
            
            ### passive link step cur frame ###
            # self.rasterize_passive_cur_step_cur_frame.grad(i_fr=i_fr + 1) 
            self.passive_link_step_cur_frame(i_fr=i_fr + 1)
            
            ## check passive x's grdients below ## # i sample 
            # print(f"passive x's value: {self.passive_x[i_fr + 1, 0]}, passive x's grad value: {self.passive_x.grad[i_fr + 1, 0]}")
            self.passive_link_step_cur_frame.grad(i_fr=i_fr + 1)
            
            # ### use gradient #####?
            # if self.args.use_deformation: # #### add to the passive states?
            #     self.backward_passive_link_step_cur_frame(i_fr=i_fr + 1)
            
            # self.backward_passive_link_step_cur_frame(i_fr=i_fr + 1)
            
            # print(f"passive x's state value: {self.passive_state[i_fr + 1]}, grad: {self.passive_state.grad[i_fr + 1]}") # backward stepping for passive x
            # state_grad_numpy_array.append(self.passive_state.grad[i_fr + 1])
            # self.rasterize_passive_cur_step_cur_frame.grad(i_fr=i_fr + 1)
        # for i_fr in range(self.n_timesteps):
        #     print(f"passive x's state value: {self.passive_state[i_fr + 1]}, grad: {self.passive_state.grad[i_fr + 1]}")
        
        if not passive_only: #### passive only not ####
            for i_fr in reversed(range(self.n_timesteps)):
                for i_link in reversed(range(self.nn_links)):
                    self.rasterize_act_cur_step_cur_frame.grad(i_link=i_link, i_fr=i_fr) ### grad through ###
                    self.backward_rasterize_act_cur_step_cur_frame(i_link=i_link, i_fr=i_fr)
            # if i_iter == 0:
            # act_xs_grad = self.act_xs.grad.to_numpy()
            # act_xs_grad_sv_fn = os.path.join(args.image_sv_folder, f"act_xs_grad_iter_{i_iter}.npy")
            # np.save(act_xs_grad_sv_fn, act_xs_grad)
            # print(f"act_xs grad saved to {act_xs_grad_sv_fn}") ### density grad fd sv ####
            
            #### use gradient #####?
            # if self.args.use_deformation:
            # self.act_xs.grad = self.act_xs_deformation.grad
            self.act_xs.grad.from_numpy(self.act_xs_deformation.grad.to_numpy())
        # self.act_xs.grad
        #### act_xs_grad_sum of the sum ####
        act_xs_grad_sum = np.sum(self.act_xs.grad.to_numpy()) ## 
        print(f"act_xs_grad_sum: {act_xs_grad_sum.item()}") ## 
        
        
        for i_fr in reversed(range(self.n_timesteps - 1)):
            for i_link in range(self.nn_links):
                # self.act_link_step_cur_frame(i_link=i_link, i_fr=i_fr + 1) 
                # self.rasterize_act_cur_step_cur_frame(i_link=i_link, i_fr=i_fr + 1)
                
                self.act_link_step_cur_frame(i_link=i_link, i_fr=i_fr + 1)
                
                # self.rasterize_act_cur_step_cur_frame.grad(i_link=i_link, i_fr=i_fr + 1)
                self.act_link_step_cur_frame.grad(i_link=i_link, i_fr=i_fr + 1) # step forward the act_link
                # self.rasterize_act_cur_step_cur_frame.grad(i_link=i_link, i_fr=i_fr + 1)
                
        # for i_fr in range(self.n_timesteps):
        #     print(f"passive x's state value: {self.passive_state[i_fr + 1]}, grad: {self.passive_state.grad[i_fr + 1]}")
    
    
    def get_grid_loss(self,):
        pass
    
    @ti.kernel
    def export_intermediate_grid_spaces(self, i_fr: ti.i32):
        # intermediate_buffered_grids
        for I in ti.grouped(self.intermediate_buffered_grids):
            self.intermediate_buffered_grids[I] = self.buffered_grids[i_fr, I] ### 
        
    
    def export_grid_spaces(self, sv_grid_space_fn):
        # the rasterized grids at each frmae
        tot_grids_numpy = []
        for i_fr in range(self.n_timesteps):
            self.export_intermediate_grid_spaces(i_fr=i_fr)
            cur_fr_grid_np = self.intermediate_buffered_grids.to_numpy()
            # print(f"frame: {i_fr}, sum of current grid values: {np.sum(cur_fr_grid_np)}")
            tot_grids_numpy.append(cur_fr_grid_np) ### res x res x res
        tot_grids_numpy = np.stack(tot_grids_numpy, axis=0) ### 
        print(f"Converted grids_numpy with shape: {tot_grids_numpy.shape}")
        np.save(sv_grid_space_fn, tot_grids_numpy)
        print(f"Converted grid spaces saved to {sv_grid_space_fn}")
    
    def export_mesh_vertices(self, sv_mesh_vertices_fn): # transformed value 
        # transformed_val #
        cur_exported_act_points = self.act_xs.to_numpy() ### nn_link x nn_frames x nn_points x 3 -> the act points 
        cur_exported_passive_points = self.passive_x.to_numpy() ##
        init_act_pts_numpy = self.init_act_xs.to_numpy()
        init_passive_pts_numpy = self.init_passive_x.to_numpy()
        # cur_exported_act_points_sv_fn = os.path.join("./save_res", f"exported_act_points")
        for i_link in range(self.nn_links):
            cur_link_init_pts = cur_exported_act_points[i_link, 0]
            minn_pts, maxx_pts = np.min(cur_link_init_pts, axis=0), np.max(cur_link_init_pts, axis=0)
            print(f"i_link: {i_link}, minn_pts: {minn_pts}, maxx_pts: {maxx_pts}")
            
            cur_link_real_init_pts = init_act_pts_numpy[i_link]
            minn_real_init_pts, maxx_real_init_pts = np.min(cur_link_real_init_pts, axis=0), np.max(cur_link_real_init_pts, axis=0)
            print(f"i_link: {i_link}, minn_real_init_pts: {minn_real_init_pts}, maxx_real_init_pts: {maxx_real_init_pts}")
        
        exported_points = {
            'act': cur_exported_act_points, 
            'passive': cur_exported_passive_points
        }
        np.save(sv_mesh_vertices_fn, exported_points) ### sv mesh vertices 
    
    def forward(self, qs):
        # qs -> list of r-dim numpy vectors 
        self.set_states_via_q(qs=qs) # set states via q #
        # for i_link in range(s)
        self.forward_stepping() # forward stepping with q #
        
        if optim: 
            if self.nrendered:
                self.run_render()
                # self.nrendered = False
            else:
                self.nrendered = False
        # self.get_grid_loss() # 
        # self.get_loss_curr_to_target_grid_spaces()
    
    ### the loss  to the density value -> the gradient of the loss over each deformation vector #### 
    ## inputs are points here ###
    def backward(self, qs):
        
        # self.loss.grad[None] = 1. ## loss grad ## to target grid spa
        
        # self.get_loss_curr_to_target_grid_spaces.grad()
        
        # self.buffered_grids.fill(0.)
        # self.buffered_grids.grad.fill(0.)
        
        # self.set_states_via_q(qs=qs) # every states
        # self.forward_stepping() ## 
        
        ##### points or images space losses #####
        # #### losses from the grid spaces ####
        # self.get_loss_curr_to_target_grid_spaces()
        
        # #### losses from mesh vertices ####
        # # self.get_loss_curr_to_target_active()
        # # self.get_loss_curr_to_target_passive()
        
        # self.loss.grad[None] = 1 ### cost to 
        
        # # self.get_loss_curr_to_target_passive.grad()
        # # self.get_loss_curr_to_target_active.grad()
        
        # #### losses from the grid spaces ####
        # self.get_loss_curr_to_target_grid_spaces.grad() ## frames ##
        # #### losses from the grid spaces #### # how about the evaluated points? ## 10 frames ##
        ##### points or images space losses #####
        
        
        self.get_loss_curr_to_target_rendered_images()
        self.loss.grad[None] = 1
        
        ### 
        if self.args.use_deformation:
            print(f"Start backwarding deformations!!")
            self.backward_image_deformations() ## backward image deformation ##
        else:
            self.backward_loss_images() ### backward loss images ##
            
        # self.backward_loss_images()
        
        
        # deformation_grad_np = self.render.deformation.grad.to_numpy()
        # deformation_grad_np_sum = np.sum(deformation_grad_np)
        # print(f"brefore running rendering grad: deformation grad sum: {deformation_grad_np_sum}")
        
        # print(f"tot_cd_loss: {self.tot_cd_loss}")
        
        self.loss[None] = self.tot_cd_loss
        
        
        self.run_render_grad()
        
        
        if self.args.use_deformation:
            print(f"Setting buffered grids grad!")
            # for i_fr in range(self.n_timesteps):
            # self.buffered_grids.grad = self.render.deformation.grad
            # self.buffered_grids_deformation.grad = self.render.deformation.grad
            # deformation # ## 
            deformation_grad_np = self.buffered_grids_deformation.grad.to_numpy()
            deformation_grad_np_sum = np.sum(deformation_grad_np)
            print(f"deformation grad sum: {deformation_grad_np_sum}")
            
            buffered_grid_deformation_grad_sum = np.sum(self.buffered_grids_deformation.grad.to_numpy())
            print(f"buffered grid deformation grad sum: {buffered_grid_deformation_grad_sum}")
        
        # 
        # print(f"begin backward stepping") ## backward stepping ## ### backward stepping ### ## 
        self.backward_stepping() ### get the loss w.r.t. self.act_states and self.passive_state # ### gradients of the 
        # print(f"after backward stepping")
        ### act_states: n_linkxs x n_timesteps x state_dof
        ### n_timesteps x passive_obj_state_dim
        df_dqs = np.zeros((len(qs), qs[0].shape[0]), dtype=np.float32)
        # for i_link in range(self.nn_links):
        for i_fr in range(len(qs)):
            for i_link in range(self.nn_links):
                # cur_act_link_q_grad = 
                df_dqs[i_fr, i_link] = self.act_states.grad[i_link, i_fr + 1].to_numpy()[0] # export to the numpy array #
            # for i_dim in range(3): ### passive states ###
            df_dqs[i_fr, self.nn_links: ] = self.passive_state.grad[i_fr + 1].to_numpy()
            # print(f"i_fr: {i_fr + 1}, passive x's state grad: {self.passive_state.grad[i_fr + 1]}")
        # print(f"printing")
        print(df_dqs)
        return self.loss[None], df_dqs
        #### TODO: ==== then we can get the gradients of qs in the taichi ####
        #### TODO: ==== get qs in the numpy array ###

##  TODO: how to use the deformation -> a strategy is modify this version to the use deformation version ##

        
## TODO: 1) save the tracked result; 2) the forward and the backward proces for optimizing actions; 3) optimziing the initial states, geometry, and actions; 
## save the tracked result

### the system and the geometry here 
### the optimization problem ### 
## rendering -> geometry ##
## physics -> actions ##
## physics -> actions ##
## the initial state and the x_goal ##
## the initial state with the x_goal ##
## the initial state with the x_goal ##
## initi
## 
# passive object geometry -> co-optimize the geometry and the action 
# optimizable geometry? # density field with density field #
# tets with the NeRF; what about neumanifold? #

# and the optimize for the initial states ###
## articulated object ##
## articulated ## articulated object ##


if __name__ == '__main__':
    
    parser = data_utils.create_arguments()
    
    
    args = parser.parse_args()
    # args.image_sv_folder = "/data1/sim/diffsim/DiffRedMax/save_res"
    args.image_sv_folder = "./save_res"
    args.num_steps = 1000
    args.mod_sv_timesteps = 50
    args.use_deformation = False
    args.use_deformation = True
    args.use_kinematic = False
    # args.use_kinematic = True
    # args.view_divide = 1
    
    ##
    
    # this system with different masses and the in
    
    os.makedirs(args.image_sv_folder, exist_ok=True)
    
    if len(args.tag) > 0:
        args.image_sv_folder = os.path.join(args.image_sv_folder, f"tag_{args.tag}")
        os.makedirs(args.image_sv_folder, exist_ok=True)
    
    optim = False
    optim = True
    optim = args.optim
    
    passive_only = False
    # passive_only = True
    
    ### reconstructor 
    reconstructor = Reconstructor(args=args)
    
    
    sim = redmax.make_sim("TorqueFingerFlick-Demo", "BDF2")
    
    ndof_r, ndof_m, ndof_u = sim.ndof_r, sim.ndof_m, sim.ndof_u

    print('ndof_r = ', ndof_r)
    print('ndof_m = ', ndof_m)
    print('ndof_u = ', ndof_u)

    num_steps = args.num_steps

    ### flick demo ####
    q0 = np.array([0., np.pi / 2., np.pi / 4., 0., 0., 0.])
    sim.set_q_init(q0)

    # x_goal = 10.5

    x_goal = args.x_goal

    q_goal = np.zeros(3)
    P_q = np.array([10., 2., 3.])

    sim.reset(False)

    # initial guess #
    u = np.zeros(ndof_u * num_steps)
    for i in range(num_steps):
        q = sim.get_q()
        error = q_goal - q[:3]
        ui = error * P_q
        
        u[i * ndof_u:(i + 1) * ndof_u] = ui

        sim.set_u(ui)

        sim.forward(1, False)

    q = sim.get_q()
    print('q = ', q)
    
    i_iter = 49
    i_iter = 61
    i_iter = args.load_target_iter
    if optim:
        # target_meshes_sv_fn = f"./save_res/exported_mesh_verts_iter_{i_iter}.npy"
        # reconstructor.load_target_mesh_verts(target_meshes_sv_fn)
        # target_images_sv_fn = f"./save_res/rendered_images_optim_False.npy"
        # target_images_sv_fn = f"./save_res/rendered_images_optim_False_nv_14.npy"
        # reconstructor.load_target_rendered_images(target_images_sv_fn)
        ### target grid spaces sv fn ###
        
        # target_grid_spaces_sv_fn = f"./save_res/exported_grid_spaces_iter_{i_iter}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
        # reconstructor.load_target_grid_space(target_grid_space_sv_fn=target_grid_spaces_sv_fn)
        
        if reconstructor.n_views < 1:
            if reconstructor.grid_res == 256:
                target_meshes_sv_fn = f"./save_res/exported_mesh_verts_iter_{i_iter}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
            else:
                target_meshes_sv_fn = f"./save_res/tag_{args.tag}/exported_mesh_verts_iter_{61}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
        else:
            if reconstructor.grid_res == 256:
                target_meshes_sv_fn = f"./save_res/tag_{args.tag}/exported_mesh_verts_iter_{61}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
            else:
                # target_meshes_sv_fn = f"./save_res/tag_{args.tag}/exported_mesh_verts_iter_{61}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
                target_meshes_sv_fn = f"./save_res/tag_{args.tag}/exported_mesh_verts_iter_{i_iter}_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
        reconstructor.load_target_mesh_verts(target_meshes_sv_fn)
        
        # target_images_sv_fn = f"./save_res/rendered_images_optim_False.npy"
        if reconstructor.n_views < 1:
            if reconstructor.grid_res == 256:
                target_images_sv_fn = f"./save_res/rendered_images_optim_False_passive_{passive_only}_nv_{reconstructor.n_views}.npy"
                
                if not os.path.exists(target_images_sv_fn):
                    target_images_sv_fn = f"./save_res/rendered_images_optim_False_passive_{passive_only}_nv_{reconstructor.n_views}_n_iter_{62}.npy"
            else:
                target_images_sv_fn = f"./save_res/tag_{args.tag}/rendered_images_optim_False_passive_False_nv_{reconstructor.n_views}_n_iter_62.npy"
        else:
            if reconstructor.grid_res == 256:
                target_images_sv_fn = f"/home/xueyi/diffsim/DiffHand/examples/save_res/tag_{args.tag}/rendered_images_optim_False_passive_False_nv_{reconstructor.n_views}_n_iter_62.npy"
            else: ### nv_n_iter_ ###
                # target_images_sv_fn = f"/home/xueyi/diffsim/DiffHand/examples/save_res/tag_{args.tag}/rendered_images_optim_False_passive_False_nv_{reconstructor.n_views}_n_iter_62.npy"
                target_images_sv_fn = f"/home/xueyi/diffsim/DiffHand/examples/save_res/tag_{args.tag}/rendered_images_optim_False_passive_False_nv_{reconstructor.n_views}_n_iter_{i_iter + 1}.npy"
        reconstructor.load_target_rendered_images(target_images_sv_fn)
        
    # save_res/exported_grid_spaces_iter_9_optim_passive_True_nv_14.npy 
    # 
    
    reconstructor.initialize()

    # args.mod_sv_timesteps = 100
    
    buffered_qs = []
    
    
    i_iter = 0
    # SimRenderer.replay(sim, record = False, record_path = "./torque_finger_flick_init.gif")

    def loss_and_grad(u): # compute loss and gradient through diff redmax
        global i_iter
        global reconstructor
        
        sim.reset(True)  # #  # ## sim
        
        tot_qs = []
        
        # cur_q = sim.get_q() #### the current q state ####
        # tot_qs.append(cur_q) #### get the current state here ####
        
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            # n_timesteps ###
            sim.forward(1)
            if (i + 1) % args.mod_sv_timesteps == 0: # mod sv timesteps ##
                cur_q = sim.get_q()
                tot_qs.append(cur_q)
                if i_iter == 0:
                    buffered_qs.append(cur_q)
        
        if args.use_kinematic:
            q = buffered_qs[-1]
        else:
            q = sim.get_q()
        
        # tot_qs.append(q)
        
        
        if args.use_kinematic:
            reconstructor.forward(buffered_qs)
        else:
            reconstructor.forward(tot_qs)
        
        # cur_grid_space_sv_fn = os.path.join(args.image_sv_folder, f"exported_grid_spaces_iter_{i_iter}.npy")
        # reconstructor.export_grid_spaces(sv_grid_space_fn=cur_grid_space_sv_fn)
        # cur_mesh_verts_sv_fn = os.path.join(args.image_sv_folder, f"exported_mesh_verts_iter_{i_iter}.npy")
        # reconstructor.export_mesh_vertices(cur_mesh_verts_sv_fn) ### exported mesh fns ### ## fns ##
        
        # print(f"{i_iter}-th iteraction; start backwarding steps")
        
        if optim:
            
            # cur_grid_space_sv_fn = os.path.join(args.image_sv_folder, f"exported_grid_spaces_iter_{i_iter}_optim_passive_{passive_only}_nv_{reconstructor.n_views}.npy")
            
            # reconstructor.export_grid_spaces(sv_grid_space_fn=cur_grid_space_sv_fn)
            cur_mesh_verts_sv_fn = os.path.join(args.image_sv_folder, f"exported_mesh_verts_iter_{i_iter}_optim_passive_{passive_only}_nv_{reconstructor.n_views}.npy")
            
            reconstructor.export_mesh_vertices(cur_mesh_verts_sv_fn) ### exported mesh fns ### ## fns ## ### passive only?
            if args.use_kinematic:
                f, df_dqs_recon = reconstructor.backward(buffered_qs)
            else:
                f, df_dqs_recon = reconstructor.backward(tot_qs)
        else:
            # cur_grid_space_sv_fn = os.path.join(args.image_sv_folder, f"exported_grid_spaces_iter_{i_iter}_passive_{passive_only}_nv_{reconstructor.n_views}.npy")
            
            # reconstructor.export_grid_spaces(sv_grid_space_fn=cur_grid_space_sv_fn)
            
            cur_mesh_verts_sv_fn = os.path.join(args.image_sv_folder, f"exported_mesh_verts_iter_{i_iter}_passive_{passive_only}_nv_{reconstructor.n_views}.npy")
            
            reconstructor.export_mesh_vertices(cur_mesh_verts_sv_fn) ### exported mesh fns ### ## fns ##
            
            f = (q[3] - x_goal) ** 2

        # f = (q[3] - x_goal) ** 2 # 
        cost_to_goal = (q[3] - x_goal) ** 2
        
        print(f"iter: {i_iter}, cost_to_goal: {cost_to_goal}, q[3]: {q[3]}, x_goal: {x_goal}, q: {q}")

        # set backward info # 
        sim.backward_info.set_flags(False, False, False, True) # specify which gradient we want, here we only want the gradients w.r.t. control actions
        
        # set terminal derivatives # temporal derivatives #
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps) # specify the partial derivative of the control actions in loss function
        df_dq = np.zeros(ndof_r * num_steps) 
        # over control variables # 
        
        if optim:
            # ### state driven gradients ###
            i_recon = 0
            for i_recon in range(df_dqs_recon.shape[0]):
                # print() # mod sv timesteps #
                st_step_idx = (i_recon + 1) * args.mod_sv_timesteps - 1
                print(f"st idx: {st_step_idx}, ed idx: {st_step_idx + 1}, n_timesteps: {num_steps}")
                ### df_dq ###
                
                # df_dqs_recon[i_recon, :3] = 0.
                # df_dqs_recon[i_recon, 4:] = 0.
                
                # if i_recon == df_dqs_recon.shape[0] - 1:
                #     df_dqs_recon[i_recon, :] = df_dqs_recon[i_recon, :] * 1e3
                #     df_dq[st_step_idx * ndof_r: (st_step_idx + 1) * ndof_r] = df_dqs_recon[i_recon, :]
                    
                #     # df_dqs_recon[i_recon, 3] = 2. * (q[3] - x_goal)
                #     # df_dq[st_step_idx * ndof_r: (st_step_idx + 1) * ndof_r] = df_dqs_recon[i_recon, :]
                    
                #     # df_dq[ndof_r * (num_steps - 1) + 3] = 2. * (q[3] - x_goal) # specify 
                #     # f = f * 200
                #     f = (q[3] - x_goal) ** 2
                #     print(f"i_recon: {i_recon}, df_dqs: {df_dqs_recon[i_recon]}, st_step_idx: {st_step_idx * ndof_r}, idxx: {ndof_r * (num_steps - 1)}")
                # df_dqs_recon[i_recon, 3] = 2. * (q[3] - x_goal)
                df_dq[st_step_idx * ndof_r: (st_step_idx + 1) * ndof_r] = df_dqs_recon[i_recon, :]  * 1e3
                
                # df_dq[ndof_r * (num_steps - 1) + 3] = 2. * (q[3] - x_goal)
                # f = f * 200
                # f = (q[3] - x_goal) ** 2
                # 
                # print(f"i_recon: {i_recon}, df_dqs: {df_dqs_recon[i_recon]}, st_step_idx: {st_step_idx * ndof_r}, idxx: {ndof_r * (num_steps - 1)}")
                    
                ### passive ###
                # df_dq[st_step_idx * ndof_r + 3: (st_step_idx + 1) * ndof_r] = df_dqs_recon[i_recon, 3:] # only rely on passive object's 
                # df_dq[st_step_idx * ndof_r: (st_step_idx + 1) * ndof_r - 3] = df_dqs_recon[i_recon, :3] 
            # ### state driven gradients ###
        else:
            ### goal driven gradients ### ## visual status v.s. visual status ## ## noisy gradients 
            df_dq[ndof_r * (num_steps - 1) + 3] = 2. * (q[3] - x_goal) # specify the partial derivative of the state q in loss function
            ### goal driven gradients ###
        
        if optim:
            f = f * 1000 #  2000 #  10000
        
        ### backward info ###
        if i_iter >= 0 and optim:
            # sim.backward_info.df_dq = df_dq * 1e4
            sim.backward_info.df_dq = df_dq  #   * 1e4 # * 1e3
        else:
            sim.backward_info.df_dq = df_dq
        
        if optim and args.use_kinematic:
            for i_i in range(len(buffered_qs)):
                df_dqs_recon[i_i][4:] = 0.
                buffered_qs[i_i] = buffered_qs[i_i] - df_dqs_recon[i_i] * 1e5 
                # if i_i == 
                print("buffered qs")
                print(buffered_qs[i_i])
            # print(buffered_qs)
            # buffered_qs[i_i][3] = buffered_qs[i_i][3] - df_dqs_recon[i_i][3] * 1e5 

        # backward
        sim.backward()

        grad = np.copy(sim.backward_results.df_du) ### df_du
        
        if optim:
            reconstructor.initialize() # forward qs #
        
        i_iter += 1 ### i_iter for the i_iter ##
        
        print(f"f: {f}")
        
        return f, grad
    
    def callback_func(u, render = False): # 
        global reconstructor
        sim.reset(False)
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
        
        q = sim.get_q()

        # f = (q[3] - x_goal) ** 2 ## q and q ## 
        f = reconstructor.loss[None]
        
        # if not optim:
        #     reconstructor.run_render()

        # print('f = ', f) ### file xxx ###

        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = False, record_path = "./torque_finger_flick_optimized.gif")

    res = scipy.optimize.minimize(loss_and_grad,  np.copy(u),  method = "L-BFGS-B", jac = True, callback = callback_func)
    
    if not optim:
        reconstructor.run_render()
    # res = scipy.optimize.minimize(loss_and_grad, np.copy(u), method = "BFGS", jac = True, callback = callback_func)

    ## callback func ## ## loss and grad ##
    # callback_func(u = res.x, render = True)
    callback_func(u = res.x, render = False)
    