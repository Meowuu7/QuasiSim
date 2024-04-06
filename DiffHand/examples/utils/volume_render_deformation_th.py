import taichi as ti
import argparse
import numpy as np
import math
import os
from imageio import imwrite
import urllib

import torch

# os.makedirs('output_volume_renderer', exist_ok=True)
# 
# real = ti.f32
# # ti.init(default_fp=real, arch=ti.cpu)
# ti.init(default_fp=real, arch=ti.cuda, kernel_profiler=True, device_memory_fraction=0.9)
# ti.init(device_memory_fraction=0.9)
# ti.init(default_fp=real, arch=ti.cuda, device_memory_fraction=0.9)
### density res ###
# res = 512
# density_res = 128
# inv_density_res = 1.0 / density_res
# res_f32 = float(res)
# dx = 0.02
# n_views = 7
# torus_r1 = 0.4
# torus_r2 = 0.1
# fov = 1 # camera #
# camera_origin_radius = 1
# marching_steps = 1000
# learning_rate = 15

# scalar = lambda: ti.field(dtype=real)

# density = scalar()
# target_images = scalar()
# images = scalar()
# loss = scalar()

# ti.root.dense(ti.ijk, density_res).place(density)
# ti.root.dense(ti.i, n_views).dense(ti.jk, res).place(target_images, images)
# ti.root.place(loss)
# ti.root.lazy_grad()

## volume render; raseter the volume into
## u_0 x + u_1 y = z -> 
## u_00 u_11 ~ U(0,1); randomly sample them U(0,1) and transform u_00 and u_11
## u_00 (u_0 x + u_1 y) + u_11 z = final_z? -> to get the final z values here ##
## u_00 


# @ti.data_oriented
class VolumeRender:
    def __init__(self, res, density_res, dx, n_views, torus_r1, torus_r2, fov, camera_origin_radius, marching_steps, learning_rate, tot_frames=1):
        self.res = res 
        self.res_f32 = float(res)
        self.density_res = density_res
        self.dx = dx
        self.n_views = n_views
        self.torus_r1 = torus_r1
        self.torus_r2 = torus_r2
        self.fov = fov
        self.camera_origin_radius = camera_origin_radius
        self.marching_steps = marching_steps
        self.learning_rate = learning_rate
        
        self.tot_frames = tot_frames
        
        
        # self.
        
        # self.target_images = ti.field(dtype=real, shape=(self.n_views, self.res, self.res))
        # self.density = ti.field(dtype=real, shape=(self.density_res, self.density_res, self.density_res), needs_grad=True)
        
        # self.deformation = ti.Vector.field(n=3, dtype=real, shape=(self.tot_frames, self.density_res, self.density_res, self.density_res), needs_grad=True)
        # density_tot_frames ## density tot frames ##
        
        # self.density_tot_frames = ti.field(dtype=real, shape=(self.tot_frames, self.density_res, self.density_res, self.density_res), needs_grad=True)
        
        # self.images = ti.field(dtype=real, shape=(self.n_views, self.res, self.res),  needs_grad=True)
        
        # ### for each frame ###
        
        # self.loss = ti.field(dtype=real, shape=(),  needs_grad=True)

        # self.density_tot_frames = ti.field(dtype=real, shape=(self.tot_frames, self.density_res, self.density_res, self.density_res), needs_grad=True)
        # ### total buffered images tot frames ###
        # self.buffered_images_tot_frames = ti.field(dtype=real, shape=(self.tot_frames, self.n_views, self.res, self.res),  needs_grad=True)

        # self.density_grad_finite_difference = ti.Vector.field(n=3, dtype=real, shape=(self.tot_frames, self.density_res, self.density_res, self.density_res), needs_grad=True)
        


    # @ti.func
    def in_box(self, x, y, z):
        # The density grid is contained in a unit box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
        return x >= -0.5 and x < 0.5 and y >= -0.5 and y < 0.5 and z >= -0.5 and z < 0.5


    # @ti.kernel
    # def ray_march(self, field: ti.template(), angle: ti.f32, view_id: ti.i32):
        for pixel in range(self.res * self.res):
            for k in range(self.marching_steps):
                x = pixel // self.res
                y = pixel - x * self.res

                camera_origin = ti.Vector([
                    self.camera_origin_radius * ti.sin(angle), 0,
                    self.camera_origin_radius * ti.cos(angle)
                ])
                dir = ti.Vector([
                    self.fov * (ti.cast(x, ti.f32) /
                        (self.res_f32 / 2.0) - self.res_f32 / self.res_f32),
                    self.fov * (ti.cast(y, ti.f32) / (self.res_f32 / 2.0) - 1.0), -1.0
                ])

                length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                                dir[2] * dir[2])
                dir /= length

                # rotated x #
                rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
                rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
                dir[0] = rotated_x
                dir[2] = rotated_z
                point = camera_origin + (k + 1) * self.dx * dir

                # Convert to coordinates of the density grid box
                box_x = point[0] + 0.5
                box_y = point[1] + 0.5
                box_z = point[2] + 0.5

                # Density grid location
                index_x = ti.cast(ti.floor(box_x * self.density_res), ti.i32)
                index_y = ti.cast(ti.floor(box_y * self.density_res), ti.i32)
                index_z = ti.cast(ti.floor(box_z * self.density_res), ti.i32)
                index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
                index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
                index_z = ti.max(0, ti.min(index_z, self.density_res - 1))

                flag = 0
                if self.in_box(point[0], point[1], point[2]):
                    flag = 1

                contribution = self.density[index_z, index_y, index_x] * flag

                field[view_id, y, x] += contribution

    # ## ray marchi single loop ##
    # @ti.kernel
    # def ray_march_single_loop(self, field: ti.template(), angle: ti.f32, view_id: ti.i32): # run the render single loop #
        for pixel in range(self.res * self.res * self.marching_steps):
            # for k in range(self.marching_steps):
            x = (pixel % (self.res * self.res)) // self.res
            y = (pixel % (self.res * self.res)) - x * self.res
            k = pixel // (self.res * self.res)


            camera_origin = ti.Vector([
                self.camera_origin_radius * ti.sin(angle), 0,
                self.camera_origin_radius * ti.cos(angle)
            ])
            dir = ti.Vector([
                self.fov * (ti.cast(x, ti.f32) /
                    (self.res_f32 / 2.0) - self.res_f32 / self.res_f32),
                self.fov * (ti.cast(y, ti.f32) / (self.res_f32 / 2.0) - 1.0), -1.0
            ])

            length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                            dir[2] * dir[2])
            dir /= length

            # rotated x #
            rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
            rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
            dir[0] = rotated_x
            dir[2] = rotated_z
            point = camera_origin + (k + 1) * self.dx * dir

            # Convert to coordinates of the density grid box
            box_x = point[0] + 0.5
            box_y = point[1] + 0.5
            box_z = point[2] + 0.5

            # Density grid location
            index_x = ti.cast(ti.floor(box_x * self.density_res), ti.i32)
            index_y = ti.cast(ti.floor(box_y * self.density_res), ti.i32)
            index_z = ti.cast(ti.floor(box_z * self.density_res), ti.i32)
            index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
            index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
            index_z = ti.max(0, ti.min(index_z, self.density_res - 1))

            flag = 0
            if self.in_box(point[0], point[1], point[2]):
                flag = 1

            contribution = self.density[index_z, index_y, index_x] * flag
            ## deformation ##

            field[view_id, y, x] += contribution
      
      
    def ray_march_single_loop_with_density_frames_with_deformation_th(self, density_field, density_def, img_field, img_def_field, angle, view_id):
        # for pixel in range(self.res * self.res * self.marching_steps):
        for pixel in range(10 * 10 * self.marching_steps):
            # for k in range(self.marching_steps):
            x = (pixel % (self.res * self.res)) // self.res
            y = (pixel % (self.res * self.res)) - x * self.res
            k = pixel // (self.res * self.res)
            
            camera_origin = torch.tensor(
                [self.camera_origin_radius * torch.sin(angle), 0., self.camera_origin_radius * torch.cos(angle)], dtype=torch.float32
            ).cuda()
            dir = torch.tensor(
                [self.fov * (float(x) /
                    (self.res_f32 / 2.0) - self.res_f32 / self.res_f32), self.fov * (float(y) / (self.res_f32 / 2.0) - 1.0), -1.0], dtype=torch.float32
            ).cuda()
            length = torch.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                            dir[2] * dir[2])
            dir = dir / length
            
            rotated_x = dir[0] * torch.cos(angle) + dir[2] * torch.sin(angle)
            rotated_z = dir[0] * torch.sin(angle) + dir[2] * torch.cos(angle)
            dir[0] = rotated_x
            dir[2] = rotated_z
            point = camera_origin + (k + 1) * self.dx * dir
            
            R = torch.stack(
                [
                    torch.tensor([torch.cos(angle), 0., -1. * torch.sin(angle)], dtype=torch.float32).cuda(), 
                    torch.tensor([0., 1., 0.], dtype=torch.float32).cuda(), 
                    torch.tensor([torch.sin(angle), 0., torch.cos(angle)], dtype=torch.float32).cuda(), 
                ], dim=-1 ### stack for the matrix
            )
            
            ## dir[0] dir[1] dir[2] -> the direction ### 
            # Convert to coordinates of the density grid box
            box_x = point[0] + 0.5
            box_y = point[1] + 0.5
            box_z = point[2] + 0.5

            # Density grid location
            index_x = (box_x * self.density_res).long()
            index_y = (box_y * self.density_res).long()
            index_z = (box_z * self.density_res).long()
            # index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
            # index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
            # index_z = ti.max(0, ti.min(index_z, self.density_res - 1))
            # print(f"index_x: {index_x}")
            # torch.max()
            # index_x = torch.max(0, torch.min(index_x, self.density_res - 1))
            # index_y = torch.max(0, torch.min(index_y, self.density_res - 1)[0])[0]
            # index_y = torch.max(0, torch.min(index_z, self.density_res - 1)[0])[0]
            
            index_x = torch.clamp(index_x, min=0, max=self.density_res - 1)
            index_y = torch.clamp(index_y, min=0, max=self.density_res - 1)
            index_z = torch.clamp(index_z, min=0, max=self.density_res - 1)

            flag = 0
            if self.in_box(point[0], point[1], point[2]):
                flag = 1

            
            # 3 -> 2 matrix  # deformation'; # # 
            # rotated_def = R @ density_def[index_z, index_y, index_x] * flag
            rotated_def = (torch.matmul(R, density_def[index_z, index_y, index_x].unsqueeze(-1)) * flag).squeeze(-1)
            
            select_element_matrix = torch.stack(
                [
                    torch.tensor([0., 0.], dtype=torch.float32).cuda(), torch.tensor([1., 0.], dtype=torch.float32).cuda(), torch.tensor([0., 1.], dtype=torch.float32).cuda()
                ], dim=-1 ### element matrix
            )
            selected_rotated_def = torch.matmul(select_element_matrix, rotated_def.unsqueeze(-1)).squeeze(-1)
            # selected_rotated_def = select_element_matrix @ rotated_def ### 2-dim 

            # contribution = self.density[index_z, index_y, index_x] * flag
            # contribution = self.density_tot_frames[fr, index_z, index_y, index_x] * flag
            ## deformation ##
            
            contribution = density_field[index_z, index_y, index_x] * flag
            img_field[view_id, y, x] += contribution
            img_def_field[view_id, y, x] += selected_rotated_def * flag

            # field[view_id, y, x] += contribution
            # def_field[view_id, y, x] += selected_rotated_def * flag
            
      
    # @ti.kernel
    # def ray_march_single_loop_with_density_frames_with_deformation(self, field: ti.template(), def_field: ti.template(), angle: ti.f32, view_id: ti.i32, fr: ti.i32): # run the render single loop #
    #     ## field; ## iamge; image deformation ##
    #     for pixel in range(self.res * self.res * self.marching_steps):
    #         # for k in range(self.marching_steps):
    #         x = (pixel % (self.res * self.res)) // self.res
    #         y = (pixel % (self.res * self.res)) - x * self.res
    #         k = pixel // (self.res * self.res)


    #         camera_origin = ti.Vector([
    #             self.camera_origin_radius * ti.sin(angle), 0,
    #             self.camera_origin_radius * ti.cos(angle)
    #         ])
    #         dir = ti.Vector([
    #             self.fov * (ti.cast(x, ti.f32) /
    #                 (self.res_f32 / 2.0) - self.res_f32 / self.res_f32),
    #             self.fov * (ti.cast(y, ti.f32) / (self.res_f32 / 2.0) - 1.0), -1.0
    #         ])

    #         length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
    #                         dir[2] * dir[2])
    #         dir /= length

    #         # rotated x #
    #         rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
    #         rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
    #         dir[0] = rotated_x
    #         dir[2] = rotated_z
    #         point = camera_origin + (k + 1) * self.dx * dir
            
    #         R = ti.Matrix.cols(
    #             [
    #                 ti.Vector([ti.cos(angle), 0., -1. * ti.sin(angle)], dt=ti.float32), 
    #                 ti.Vector([0., 1., 0.], dt=ti.float32), 
    #                 ti.Vector([ti.sin(angle), 0., ti.cos(angle)], dt=ti.float32)
    #             ]
    #         )
            

    #         ## dir[0] dir[1] dir[2] -> the direction ### 
    #         # Convert to coordinates of the density grid box
    #         box_x = point[0] + 0.5
    #         box_y = point[1] + 0.5
    #         box_z = point[2] + 0.5

    #         # Density grid location
    #         index_x = ti.cast(ti.floor(box_x * self.density_res), ti.i32)
    #         index_y = ti.cast(ti.floor(box_y * self.density_res), ti.i32)
    #         index_z = ti.cast(ti.floor(box_z * self.density_res), ti.i32)
    #         index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
    #         index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
    #         index_z = ti.max(0, ti.min(index_z, self.density_res - 1))

    #         flag = 0
    #         if self.in_box(point[0], point[1], point[2]):
    #             flag = 1

            
    #         # 3 -> 2 matrix  # deformation'; # # 
    #         rotated_def = R @ self.deformation[fr, index_z, index_y, index_x] * flag
    #         # select_element_matrix = ti.Matrix.cols(
    #         #     [
    #         #         ti.Vector([0., 1.], dt=ti.float32), ti.Vector([1., 0.], dt=ti.float32), ti.Vector([0., 0.], dt=ti.float32)
    #         #     ]
    #         # )
    #         select_element_matrix = ti.Matrix.cols(
    #             [
    #                 # ti.Vector([0., 1.], dt=ti.float32), ti.Vector([1., 0.], dt=ti.float32), ti.Vector([0., 0.], dt=ti.float32)
    #                 ti.Vector([0., 0.], dt=ti.float32), ti.Vector([1., 0.], dt=ti.float32), ti.Vector([0., 1.], dt=ti.float32)
    #                 # ti.Vector([1., 0.], dt=ti.float32), ti.Vector([0., 1.], dt=ti.float32), ti.Vector([0., 0.], dt=ti.float32)
    #             ]
    #         )
    #         selected_rotated_def = select_element_matrix @ rotated_def ### 2-dim 

    #         # contribution = self.density[index_z, index_y, index_x] * flag
    #         contribution = self.density_tot_frames[fr, index_z, index_y, index_x] * flag
    #         ## deformation ##
            
            

    #         field[view_id, y, x] += contribution
    #         def_field[view_id, y, x] += selected_rotated_def * flag
            
            
    # @ti.kernel
    # def ray_march_single_loop_with_density_frames(self, field: ti.template(), angle: ti.f32, view_id: ti.i32, fr: ti.i32): # run the render single loop #
    #     for pixel in range(self.res * self.res * self.marching_steps):
    #         # for k in range(self.marching_steps):
    #         x = (pixel % (self.res * self.res)) // self.res
    #         y = (pixel % (self.res * self.res)) - x * self.res
    #         k = pixel // (self.res * self.res)


    #         camera_origin = ti.Vector([
    #             self.camera_origin_radius * ti.sin(angle), 0,
    #             self.camera_origin_radius * ti.cos(angle)
    #         ])
    #         dir = ti.Vector([
    #             self.fov * (ti.cast(x, ti.f32) /
    #                 (self.res_f32 / 2.0) - self.res_f32 / self.res_f32),
    #             self.fov * (ti.cast(y, ti.f32) / (self.res_f32 / 2.0) - 1.0), -1.0
    #         ])

    #         length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
    #                         dir[2] * dir[2])
    #         dir /= length

    #         # rotated x #
    #         rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
    #         rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
    #         dir[0] = rotated_x
    #         dir[2] = rotated_z
    #         point = camera_origin + (k + 1) * self.dx * dir

    #         ## dir[0] dir[1] dir[2] -> the direction ### 
    #         # Convert to coordinates of the density grid box
    #         box_x = point[0] + 0.5
    #         box_y = point[1] + 0.5
    #         box_z = point[2] + 0.5

    #         # Density grid location
    #         index_x = ti.cast(ti.floor(box_x * self.density_res), ti.i32)
    #         index_y = ti.cast(ti.floor(box_y * self.density_res), ti.i32)
    #         index_z = ti.cast(ti.floor(box_z * self.density_res), ti.i32)
    #         index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
    #         index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
    #         index_z = ti.max(0, ti.min(index_z, self.density_res - 1))

    #         flag = 0
    #         if self.in_box(point[0], point[1], point[2]):
    #             flag = 1

    #         # contribution = self.density[index_z, index_y, index_x] * flag
    #         contribution = self.density_tot_frames[fr, index_z, index_y, index_x] * flag
    #         ## deformation ##

    #         field[view_id, y, x] += contribution


    # @ti.kernel
    # def ray_march_single_loop_with_frame(self, i_fr: ti.i32, angle: ti.f32, view_id: ti.i32): # run the render single loop #
        # for pixel in range(self.res * self.res * self.marching_steps):
        #     # for k in range(self.marching_steps):
        #     x = (pixel % (self.res * self.res)) // self.res
        #     y = (pixel % (self.res * self.res)) - x * self.res
        #     k = pixel // (self.res * self.res)

        #     ### camera origin ###
        #     camera_origin = ti.Vector([
        #         self.camera_origin_radius * ti.sin(angle), 0,
        #         self.camera_origin_radius * ti.cos(angle)
        #     ]) ## 
        #     dir = ti.Vector([
        #         self.fov * (ti.cast(x, ti.f32) /
        #             (self.res_f32 / 2.0) - self.res_f32 / self.res_f32),
        #         self.fov * (ti.cast(y, ti.f32) / (self.res_f32 / 2.0) - 1.0), -1.0
        #     ])

        #     length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
        #                     dir[2] * dir[2])
        #     dir /= length

            # rotated x #
            # rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
            # rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
            # dir[0] = rotated_x
            # dir[2] = rotated_z
            # point = camera_origin + (k + 1) * self.dx * dir

            # # Convert to coordinates of the density grid box
            # box_x = point[0] + 0.5
            # box_y = point[1] + 0.5
            # box_z = point[2] + 0.5

            # # Density grid location
            # index_x = ti.cast(ti.floor(box_x * self.density_res), ti.i32)
            # index_y = ti.cast(ti.floor(box_y * self.density_res), ti.i32)
            # index_z = ti.cast(ti.floor(box_z * self.density_res), ti.i32)
            # index_x = ti.max(0, ti.min(index_x, self.density_res - 1))
            # index_y = ti.max(0, ti.min(index_y, self.density_res - 1))
            # index_z = ti.max(0, ti.min(index_z, self.density_res - 1))

            # flag = 0
            # if self.in_box(point[0], point[1], point[2]):
            #     flag = 1

            # contribution = self.density[index_z, index_y, index_x] * flag
            # self.buffered_images_tot_frames[i_fr, view_id, y, x] += contribution
            # # field[view_id, y, x] += contribution ### field[view_id, y, x] += 
            # # self.density_tot_frames[i_fr, view_id, y, x] += contribution


    def initialize(self,):
        self.density.fill(0.)
        self.density.grad.fill(0.)
        
        self.buffered_images_tot_frames.fill(0.)
        self.buffered_images_tot_frames.grad.fill(0.)
        
        self.density_tot_frames.fill(0.)
        self.density_tot_frames.grad.fill(0.)
        
        self.images.fill(0.)
        self.images.grad.fill(0.)
        
        self.density_tot_frames.fill(0.)
        self.density_tot_frames.grad.fill(0.)
        
        self.deformation.fill(0.)
        self.deformation.grad.fill(0.)


        self.density_grad_finite_difference.fill(0.)
        self.density_grad_finite_difference.grad.fill(0.)

    @ti.kernel
    def compute_loss(self, view_id: ti.i32):
        for i in range(self.res):
            for j in range(self.res):
                self.loss[None] += (self.images[view_id, i, j] -
                            self.target_images[view_id, i, j])**2 * (1.0 /
                                                                (self.res * self.res))


    @ti.kernel # self.images #
    def clear_images(self,): ##### clear rendered images ####
        for v, i, j in self.images:
            self.images[v, i, j] = 0


    @ti.kernel
    def clear_density(self,): #### clear set densities ####
        for i, j, k in self.density:
            self.density[i, j, k] = 0
            self.density.grad[i, j, k] = 0


    @ti.kernel
    def extract_target_image(self, n: ti.i32, arr: ti.types.ndarray()):
        for i, j in ti.ndrange(self.res, self.res):
            arr[i, j] = self.target_images[n, i, j]


    def create_target_images(self,):
        for view in range(self.n_views):
            self.ray_march(self.target_images, math.pi / self.n_views * view - math.pi / 2.0,
                    view)
            ### 
            img = np.zeros((self.res, self.res), dtype=np.float32)
            self.extract_target_image(view, img)
            img /= np.max(img)
            img = 1 - img

            fn = "{}/target_{:04d}.png".format("output_volume_renderer", view)
            print("Saving {}".format(fn))
            imwrite(fn, (255 * img).astype(np.uint8))


    @ti.func
    def in_torus(self, x, y, z):
        len_xz = ti.sqrt(x * x + z * z)
        qx = len_xz - self.torus_r1
        len_q = ti.sqrt(qx * qx + y * y)
        dist = len_q - self.torus_r2
        return dist < 0


    @ti.kernel
    def create_torus_density(self,):
        for i, j, k in self.density:
            # Convert to density coordinates
            x = ti.cast(k, ti.f32) * self.inv_density_res - 0.5
            y = ti.cast(j, ti.f32) * self.inv_density_res - 0.5
            z = ti.cast(i, ti.f32) * self.inv_density_res - 0.5

            # Swap x, y to rotate the torus
            if self.in_torus(y, x, z):
                self.density[i, j, k] = self.inv_density_res
            else:
                self.density[i, j, k] = 0.0


    @ti.kernel
    def apply_grad(self,):
        # gradient descent
        for i, j, k in self.density:
            self.density[i, j, k] -= self.learning_rate * self.density.grad[i, j, k]
            self.density[i, j, k] = ti.max(self.density[i, j, k], 0)


    @ti.kernel
    def fill_density(self, volume: ti.types.ndarray()):
        res = volume.shape[0]
        for i, j, k in self.density:
            self.density[i, j, k] = volume[i, res - j - 1, k]

    @ti.kernel
    def fill_density_fr_ti(self, volume: ti.template()):
        # res = self.density_res
        for i, j, k in self.density:
            self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
        
    @ti.kernel
    def fill_density_fr_ti_with_fr(self, volume: ti.template(), fr: ti.i32):
        # res = self.density_res
        for i, j, k in self.density:
            # self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
            self.density[i, j, k] = volume[fr,  i, self.density_res - j - 1, k]
            
    @ti.kernel
    def fill_density_fr_ti_with_fr_density_fr(self, volume: ti.template(), fr: ti.i32):
        # res = self.density_res
        for i, j, k in self.density:
            # self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
            self.density_tot_frames[fr, i, j, k] = volume[fr,  i, self.density_res - j - 1, k]
            
    @ti.kernel
    def fill_density_fr_ti_with_fr_density_fr_single_loop(self, volume: ti.template()):
        # res = self.density_res
        
        for fr, i, j, k in self.density_tot_frames:
            # self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
            self.density_tot_frames[fr, i, j, k] = volume[fr,  i, self.density_res - j - 1, k]
            
    @ti.kernel
    def fill_density_deformation_fr_ti_with_fr_density_fr_single_loop(self, volume: ti.template()):
        # res = self.density_res
        
        for fr, i, j, k in self.density_tot_frames:
            # self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
            # self.deformation[fr, i, j, k] = volume[fr,  i, self.density_res - j - 1, k]
            ### 
            reverse_y_transform_matrix = ti.Matrix.cols(
                [
                    ti.Vector([1., 0, 0], dt=ti.float32), 
                    ti.Vector([0, -1, 0], dt=ti.float32),
                    ti.Vector([0, 0, 1], dt=ti.float32)
                ]
            )
            self.deformation[fr, i, j, k] = reverse_y_transform_matrix @ volume[fr,  i, self.density_res - j - 1, k]
            
    @ti.func
    def get_finite_difference_of_density(self, fr: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32):
        x_density_difference = 0.
        x_nei_n = 0
        if i > 0:
            x_density_difference += self.density_tot_frames[fr, i, j, k] - self.density_tot_frames[fr, i - 1, j, k]
            x_nei_n += 1
        if i < self.density_res - 1:
            x_density_difference += self.density_tot_frames[fr, i + 1, j, k] - self.density_tot_frames[fr, i, j, k]
            x_nei_n += 1
        x_density_difference = x_density_difference / float(x_nei_n)
        
        y_density_difference = 0.
        y_nei_n = 0
        if j > 0:
            y_density_difference += self.density_tot_frames[fr, i, j, k] - self.density_tot_frames[fr, i, j - 1, k]
            y_nei_n += 1
        if j < self.density_res - 1:
            y_density_difference += self.density_tot_frames[fr, i, j + 1, k] - self.density_tot_frames[fr, i, j, k]
            y_nei_n += 1
        y_density_difference = y_density_difference / float(y_nei_n)
        
        z_density_difference = 0.
        z_nei_n = 0
        if k > 0:
            z_density_difference += self.density_tot_frames[fr, i, j, k] - self.density_tot_frames[fr, i, j, k - 1]
            z_nei_n += 1
        if k < self.density_res - 1:
            z_density_difference += self.density_tot_frames[fr, i, j, k + 1] - self.density_tot_frames[fr, i, j, k]
            z_nei_n += 1
            
        z_density_difference = z_density_difference / float(z_nei_n)
        
        xyz_difference = ti.Vector([x_density_difference, y_density_difference, z_density_difference], dt=ti.float32)
        return xyz_difference
        
    
    @ti.func
    def get_density_grad_finite_difference_cur_grid(self, fr: ti.i32, i: ti.i32, j: ti.i32, k: ti.i32):
        x_density_difference = 0.
        x_nei_n = 0
        if i > 0:
            x_density_difference += self.density_tot_frames.grad[fr, i, j, k] - self.density_tot_frames.grad[fr, i - 1, j, k]
            x_nei_n += 1
        if i < self.density_res - 1:
            x_density_difference += self.density_tot_frames.grad[fr, i + 1, j, k] - self.density_tot_frames.grad[fr, i, j, k]
            x_nei_n += 1
        x_density_difference = x_density_difference / float(x_nei_n)
        
        y_density_difference = 0.
        y_nei_n = 0
        if j > 0:
            y_density_difference += self.density_tot_frames.grad[fr, i, j, k] - self.density_tot_frames.grad[fr, i, j - 1, k]
            y_nei_n += 1
        if j < self.density_res - 1:
            y_density_difference += self.density_tot_frames.grad[fr, i, j + 1, k] - self.density_tot_frames.grad[fr, i, j, k]
            y_nei_n += 1
        y_density_difference = y_density_difference / float(y_nei_n)
        
        z_density_difference = 0.
        z_nei_n = 0
        if k > 0:
            z_density_difference += self.density_tot_frames.grad[fr, i, j, k] - self.density_tot_frames.grad[fr, i, j, k - 1]
            z_nei_n += 1
        if k < self.density_res - 1:
            z_density_difference += self.density_tot_frames.grad[fr, i, j, k + 1] - self.density_tot_frames.grad[fr, i, j, k]
            z_nei_n += 1
            
        z_density_difference = z_density_difference / float(z_nei_n)
        
        xyz_difference = ti.Vector([x_density_difference, y_density_difference, z_density_difference], dt=ti.float32)
        return xyz_difference
    
    @ti.kernel
    def get_density_grad_finite_difference(self, ): ## density grad finite difference ##
        for fr, i, j, k in self.density_tot_frames:
            ### density_grad_tot_frames
            self.density_grad_finite_difference[fr, i, j, k] = self.get_density_grad_finite_difference_cur_grid(fr, i, j, k)
        
    
    @ti.kernel
    def backward_density_to_deformation(self,):
        for fr, i, j, k in self.deformation:
            # deformation with size: tot_framees x density_res x density_res x density_res
            intger_deformation = ti.cast(self.deformation[fr, i, j, k], ti.i32)
            nex_density_index = ti.Vector([i, j, k], dt=ti.i32) + intger_deformation # 3
            density_grid_upper_bound = ti.Vector([self.density_res - 1, self.density_res - 1, self.density_res - 1], dt=ti.i32)
            ti.atomic_min(nex_density_index, density_grid_upper_bound)
            density_grid_lower_bound = ti.Vector([0, 0, 0], dt=ti.i32)
            ti.atomic_max(nex_density_index, density_grid_lower_bound)
            if fr < self.tot_frames - 1: ### fr
                # cur_density = self.density_tot_frames[fr, i, j, k]  ### the fr, i, j, k ####
                # nex_density = self.density_tot_frames[fr + 1, nex_density_index]
                # df / d\sigma_t = df / d(\sigma_t-1(())
                
                #### strategy 1 ####
                # xyz_difference = self.get_finite_difference_of_density(fr + 1, nex_density_index.x, nex_density_index.y, nex_density_index.z)
                # cur_def_grad = -1. * self.density_tot_frames.grad[fr, i, j, k] * xyz_difference ### TODO: the finite difference of the next density index ##
                # # cur_def_grad =  self.density_tot_frames.grad[fr, i, j, k] * xyz_difference
                # self.deformation.grad[fr, i, j, k] = cur_def_grad
                #### strategy 1 ####
                
                ##3 density to 
                self.deformation.grad[fr, i, j, k] =  -1. * self.density_grad_finite_difference[fr + 1, nex_density_index]
        

    @ti.kernel
    def fill_density_tot_frames(self, volume: ti.template()):
        for i_fr, i, j, k in self.density_tot_frames:
            self.density_tot_frames[i_fr, i, j, k] = volume[i_fr, i, self.density_res - j - 1, k]
        # res = self.density_res
        # for i, j, k in self.density:
        #     # self.density[i, j, k] = volume[i, self.density_res - j - 1, k]
        #     self.density[i, j, k] = volume[fr,  i, self.density_res - j - 1, k]

    # @ti.kernel
    # def fill_buffered_images_to_curr_images(self, )
    

## and you have many views as supervisions #
## 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    if not os.path.exists('bunny_128.bin'):
        url = 'https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/bunny_128.bin'
        print(':: Downloading {}'.format(url))
        report = lambda bn, bs, size: print(':: Downloading {:.2f}/{:.2f} MiB ...'.format(bn * bs / 1e6, size / 1e6), end='\r')
        urllib.request.urlretrieve(url, 'bunny_128.bin', reporthook=report)
        print(':: Download finished' + ' ' * 10)

    render = VolumeRender()
    print(':: Loading bunny_128 ...')
    volume = np.fromfile("bunny_128.bin", dtype=np.float32).reshape(
        (render.density_res, render.density_res, render.density_res))

    print(':: Fill density ...') # fill dens for the volume #
    render.fill_density(volume)

    #create_torus_density()
    print(':: Create target images ...')
    render.create_target_images()
    render.clear_density()

    for iter in range(options.iters):
        render.clear_images()
        with ti.ad.Tape(render.loss):
            for view in range(render.n_views):
                render.ray_march(render.images, math.pi / render.n_views * view - math.pi / 2.0,
                          view)
                render.compute_loss(view)

        views = render.images.to_numpy()
        for view in range(render.n_views):
            img = views[view]
            m = np.max(img)
            if m > 0:
                img /= m
            img = 1 - img
            imwrite(
                "{}/image_{:04d}_{:04d}.png".format("output_volume_renderer",
                                                    iter, view),
                (255 * img).astype(np.uint8))

        print('Iter', iter, ' Loss =', render.loss[None])
        render.apply_grad()


if __name__ == '__main__':
    main()
