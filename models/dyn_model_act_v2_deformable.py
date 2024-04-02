
import math
# import torch
# from ..utils import Timer
import numpy as np
# import torch.nn.functional as F
import os

import argparse

from xml.etree.ElementTree import ElementTree

import trimesh
import torch
import torch.nn as nn
# import List
# class link; joint; body
### 

from scipy.spatial.transform import Rotation as R
from torch.distributions.uniform import Uniform

# deformable articulated objects with the articulated models #

DAMPING = 1.0
DAMPING = 0.3

def plane_rotation_matrix_from_angle_xz(angle):
    sin_ = torch.sin(angle)
    cos_ = torch.cos(angle)
    zero_padding = torch.zeros_like(cos_)
    one_padding = torch.ones_like(cos_)
    col_a = torch.stack(
        [cos_, zero_padding, sin_], dim=0
    )
    col_b = torch.stack(
        [zero_padding, one_padding, zero_padding], dim=0
    )
    col_c = torch.stack(
        [-1. * sin_, zero_padding, cos_], dim=0
    )
    rot_mtx = torch.stack(
        [col_a, col_b, col_c], dim=-1
    )
    return rot_mtx

def plane_rotation_matrix_from_angle(angle):
    ## angle of 
    sin_ = torch.sin(angle)
    cos_ = torch.cos(angle)
    col_a = torch.stack(
        [cos_, sin_], dim=0 ### col of the rotation matrix
    )
    col_b = torch.stack(
        [-1. * sin_, cos_], dim=0 ## cols of the rotation matrix
    )
    rot_mtx = torch.stack(
        [col_a, col_b], dim=-1 ### rotation matrix
    )
    return rot_mtx

def rotation_matrix_from_axis_angle(axis, angle): # rotation_matrix_from_axis_angle -> 
        # sin_ = np.sin(angle) #  ti.math.sin(angle)
        # cos_ = np.cos(angle) #  ti.math.cos(angle)
        sin_ = torch.sin(angle) #  ti.math.sin(angle)
        cos_ = torch.cos(angle) #  ti.math.cos(angle)
        u_x, u_y, u_z = axis[0], axis[1], axis[2]
        u_xx = u_x * u_x
        u_yy = u_y * u_y
        u_zz = u_z * u_z
        u_xy = u_x * u_y
        u_xz = u_x * u_z
        u_yz = u_y * u_z
        
        row_a = torch.stack(
            [cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dim=0
        )
        # print(f"row_a: {row_a.size()}")
        row_b = torch.stack(
            [u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dim=0
        )
        # print(f"row_b: {row_b.size()}")
        row_c = torch.stack(
            [u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dim=0
        )
        # print(f"row_c: {row_c.size()}")
        
        ### rot_mtx for the rot_mtx ###
        rot_mtx = torch.stack(
            [row_a, row_b, row_c], dim=-1 ### rot_matrix of he matrix ##
        )
        
        return rot_mtx


def update_quaternion(delta_angle, prev_quat):
    s1 = 0
    s2 = prev_quat[0]
    v2 = prev_quat[1:]
    v1 = delta_angle / 2
    new_v = s1 * v2 + s2 * v1 + torch.cross(v1, v2)
    new_s = s1 * s2 - torch.sum(v1 * v2)
    new_quat = torch.cat([new_s.unsqueeze(0), new_v], dim=0)
    return new_quat


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1) # -1 for the quaternion matrix # 
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    
    return o.reshape(quaternions.shape[:-1] + (3, 3))

    


class Inertial:
    def __init__(self, origin_rpy, origin_xyz, mass, inertia) -> None:
        self.origin_rpy = origin_rpy
        self.origin_xyz = origin_xyz
        self.mass = mass
        self.inertia = inertia
        if torch.sum(self.inertia).item() < 1e-4:
            self.inertia = self.inertia + torch.eye(3, dtype=torch.float32).cuda()
        pass
    
class Visual:
    def __init__(self, visual_xyz, visual_rpy, geometry_mesh_fn, geometry_mesh_scale) -> None:
        # self.visual_origin = visual_origin
        self.visual_xyz = visual_xyz
        self.visual_rpy = visual_rpy
        self.mesh_nm = geometry_mesh_fn.split("/")[-1].split(".")[0]
        mesh_root = "/home/xueyi/diffsim/NeuS/rsc/mano" ## mano models of the mesh root ##
        if not os.path.exists(mesh_root):
            mesh_root = "/data/xueyi/diffsim/NeuS/rsc/mano"
        self.mesh_root = mesh_root
        self.geometry_mesh_fn = os.path.join(mesh_root, geometry_mesh_fn)
        self.geometry_mesh_scale = geometry_mesh_scale
        # tranformed by xyz #
        self.vertices, self.faces = self.load_geoemtry_mesh()
        self.cur_expanded_visual_pts = None
        pass
    
    def load_geoemtry_mesh(self, ):
        # mesh_root = 
        mesh = trimesh.load_mesh(self.geometry_mesh_fn)
        vertices = mesh.vertices
        faces = mesh.faces
        
        vertices = torch.from_numpy(vertices).float().cuda()
        faces =torch.from_numpy(faces).long().cuda()
        
        vertices = vertices * self.geometry_mesh_scale.unsqueeze(0) + self.visual_xyz.unsqueeze(0)
        
        return vertices, faces
    
    # init_visual_meshes = get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes)
    def get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes):
        # cur_vertices = torch.matmul(parent_rot, self.vertices.transpose(1, 0)).contiguous().transpose(1, 0).contiguous() + parent_trans.unsqueeze(0)
        cur_vertices = self.vertices
        # print(f"adding mesh loaded from {self.geometry_mesh_fn}")
        init_visual_meshes['vertices'].append(cur_vertices) # cur vertices # trans # 
        init_visual_meshes['faces'].append(self.faces)
        return init_visual_meshes
    
    def expand_visual_pts(self, ):
        expand_factor = 0.2
        nn_expand_pts = 20
        
        expand_factor = 0.4
        nn_expand_pts = 40 ### number of the expanded points ### ## points ##
        expand_save_fn = f"{self.mesh_nm}_expanded_pts_factor_{expand_factor}_nnexp_{nn_expand_pts}.npy"
        expand_save_fn = os.path.join(self.mesh_root, expand_save_fn)
        
        if not os.path.exists(expand_save_fn):
            cur_expanded_visual_pts = []
            if self.cur_expanded_visual_pts is None:
                cur_src_pts = self.vertices
            else:
                cur_src_pts = self.cur_expanded_visual_pts
            maxx_verts, _ = torch.max(cur_src_pts, dim=0)
            minn_verts, _ = torch.min(cur_src_pts, dim=0)
            extent_verts = maxx_verts - minn_verts ## (3,)-dim vecotr
            norm_extent_verts = torch.norm(extent_verts, dim=-1).item() ## (1,)-dim vector
            expand_r = norm_extent_verts * expand_factor
            # nn_expand_pts = 5 # expand the vertices to 5 times of the original vertices 
            for i_pts in range(self.vertices.size(0)):
                cur_pts = cur_src_pts[i_pts]
                # sample from the circile with cur_pts as thejcenter and the radius as expand_r
                # (-r, r) # sample the offset vector in the size of (nn_expand_pts, 3)
                offset_dist = Uniform(-1. * expand_r, expand_r)
                offset_vec = offset_dist.sample((nn_expand_pts, 3)).cuda()
                cur_expanded_pts = cur_pts + offset_vec
                cur_expanded_visual_pts.append(cur_expanded_pts)
            cur_expanded_visual_pts = torch.cat(cur_expanded_visual_pts, dim=0)
            np.save(expand_save_fn, cur_expanded_visual_pts.detach().cpu().numpy())
        else:
            print(f"Loading visual pts from {expand_save_fn}") # load from the fn #
            cur_expanded_visual_pts = np.load(expand_save_fn, allow_pickle=True)
            cur_expanded_visual_pts = torch.from_numpy(cur_expanded_visual_pts).float().cuda()
        self.cur_expanded_visual_pts = cur_expanded_visual_pts # expanded visual pts #
        return self.cur_expanded_visual_pts
        # cur_pts #
        # use r as the search direction # # expande save fn #
    def get_transformed_visual_pts(self, visual_pts_list):
        visual_pts_list.append(self.cur_expanded_visual_pts) # 
        return visual_pts_list
            
        

## link urdf ## expand the visual pts to form the expanded visual grids pts #
# use get_name_to_visual_pts_faces to get the transformed visual pts and faces #
## Link_urdf ##
class Link_urdf: # get_transformed_visual_pts #
    def __init__(self, name, inertial: Inertial, visual: Visual=None) -> None:
        
        self.name = name
        self.inertial = inertial
        self.visual = visual # vsiual meshes #
        
        # self.joint = joint
        # self.body = body
        # self.children = children
        # self.name = name
        
        self.link_idx = ...
        
        # self.args = args
        
        self.joint = None # joint name to struct
        # self.join
        self.children = ...
        self.children = {} # joint name to child sruct
        
        ### dyn_model_act ###
        # parent_rot_mtx, parent_trans_vec #
        # parent_rot_mtx, parent_trans_vec # # link urdf # 
        # self.parent_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        # self.parent_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)
        # self.curr_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        # self.curr_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)
        # # 
        # self.tot_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        # self.tot_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)

    # expand visual pts #
    def expand_visual_pts(self, expanded_visual_pts, link_name_to_visited, link_name_to_link_struct):
        link_name_to_visited[self.name] = 1
        if self.visual is not None:
            cur_expanded_visual_pts = self.visual.expand_visual_pts()
            expanded_visual_pts.append(cur_expanded_visual_pts)
        
        for cur_link in self.children:
            cur_link_struct = link_name_to_link_struct[self.children[cur_link]]
            cur_link_name = cur_link_struct.name
            if cur_link_name in link_name_to_visited:
                continue
            expanded_visual_pts = cur_link_struct.expand_visual_pts(expanded_visual_pts, link_name_to_visited, link_name_to_link_struct)
        return expanded_visual_pts
    
    def get_transformed_visual_pts(self, visual_pts_list, link_name_to_visited, link_name_to_link_struct):
        link_name_to_visited[self.name] = 1
        
        if self.joint is not None:
            for cur_joint_name in self.joint:
                cur_joint = self.joint[cur_joint_name]
                cur_child_name = self.children[cur_joint_name]
                cur_child = link_name_to_link_struct[cur_child_name] # parent and the child_visual, cur_child.visual #
                # parent #
                # print(f"joint: {cur_joint.name}, child: {cur_child_name}, parent: {self.name}, child_visual: {cur_child.visual is not None}")
                # print(f"joint: {cur_joint.name}, child: {cur_child_name}, parent: {self.name}, child_visual: {cur_child.visual is not None}")
                # joint_origin_xyz = cur_joint.origin_xyz ## transformed_visual_pts ####
                if cur_child_name in link_name_to_visited:
                    continue
                # cur_child_visual_pts = {'vertices': [], 'faces': [], 'link_idxes': [], 'transformed_joint_pos': [], 'joint_link_idxes': []}
                cur_child_visual_pts_list = []
                cur_child_visual_pts_list = cur_child.get_transformed_visual_pts(cur_child_visual_pts_list, link_name_to_visited, link_name_to_link_struct)
                
                if len(cur_child_visual_pts_list) > 0:
                    cur_child_visual_pts = torch.cat(cur_child_visual_pts_list, dim=0)
                    # cur_child_verts, cur_child_faces = cur_child_visual_pts['vertices'], cur_child_visual_pts['faces']
                    # cur_child_link_idxes = cur_child_visual_pts['link_idxes']
                    # cur_transformed_joint_pos = cur_child_visual_pts['transformed_joint_pos']
                    # joint_link_idxes = cur_child_visual_pts['joint_link_idxes']
                    # if len(cur_child_verts) > 0:
                    # cur_child_verts, cur_child_faces = merge_meshes(cur_child_verts, cur_child_faces)
                    cur_child_visual_pts = cur_child_visual_pts + cur_joint.origin_xyz.unsqueeze(0)
                    cur_joint_rot, cur_joint_trans = cur_joint.compute_transformation_from_current_state()
                    cur_child_visual_pts = torch.matmul(cur_joint_rot, cur_child_visual_pts.transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_joint_trans.unsqueeze(0)
                    
                    # if len(cur_transformed_joint_pos) > 0:
                    #     cur_transformed_joint_pos = torch.cat(cur_transformed_joint_pos, dim=0)
                    #     cur_transformed_joint_pos = cur_transformed_joint_pos + cur_joint.origin_xyz.unsqueeze(0)
                    #     cur_transformed_joint_pos = torch.matmul(cur_joint_rot, cur_transformed_joint_pos.transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_joint_trans.unsqueeze(0)
                    #     cur_joint_pos = cur_joint_trans.unsqueeze(0).clone()
                    #     cur_transformed_joint_pos = torch.cat(
                    #         [cur_transformed_joint_pos, cur_joint_pos], dim=0 ##### joint poses #####
                    #     )
                    # else:
                    #     cur_transformed_joint_pos = cur_joint_trans.unsqueeze(0).clone()
                    
                    # if len(joint_link_idxes) > 0:
                    #     joint_link_idxes = torch.cat(joint_link_idxes, dim=-1) ### joint_link idxes ###
                    #     cur_joint_idx = cur_child.link_idx
                    #     joint_link_idxes = torch.cat(
                    #         [joint_link_idxes, torch.tensor([cur_joint_idx], dtype=torch.long).cuda()], dim=-1
                    #     )
                    # else:
                    #     joint_link_idxes = torch.tensor([cur_child.link_idx], dtype=torch.long).cuda().view(1,)
                    
                    
                    visual_pts_list.append(cur_child_visual_pts)
                    # cur_child_verts = cur_child_verts + # transformed joint pos #
                    # cur_child_link_idxes = torch.cat(cur_child_link_idxes, dim=-1)
                    # # joint_link_idxes = torch.cat(joint_link_idxes, dim=-1)
                    # init_visual_meshes['vertices'].append(cur_child_verts)
                    # init_visual_meshes['faces'].append(cur_child_faces)
                    # init_visual_meshes['link_idxes'].append(cur_child_link_idxes)
                    # init_visual_meshes['transformed_joint_pos'].append(cur_transformed_joint_pos)
                    # init_visual_meshes['joint_link_idxes'].append(joint_link_idxes)
                    
        
        if self.visual is not None:
            # get_transformed_visual_pts #
            visual_pts_list = self.visual.get_transformed_visual_pts(visual_pts_list)
        
        # for cur_link in self.children:
        #     cur_link_name = cur_link.name
        #     if cur_link_name in link_name_to_visited: # link name to visited # 
        #         continue
        #     visual_pts_list = cur_link.get_transformed_visual_pts(visual_pts_list, link_name_to_visited, link_name_to_link_struct)
        return visual_pts_list
    
    # use both the articulated motion and the frre form 
    def set_initial_state(self, states, action_joint_name_to_joint_idx, link_name_to_visited, link_name_to_link_struct):
        
        link_name_to_visited[self.name] = 1
        
        if self.joint is not None:
            for cur_joint_name in self.joint:
                cur_joint = self.joint[cur_joint_name]
                cur_joint_name = cur_joint.name
                cur_child = self.children[cur_joint_name]
                cur_child_struct = link_name_to_link_struct[cur_child]
                cur_child_name = cur_child_struct.name
                
                if cur_child_name in link_name_to_visited:
                    continue
                if cur_joint.type in ['revolute']:
                    cur_joint_idx = action_joint_name_to_joint_idx[cur_joint_name] # action joint name to joint idx #
                    # cur_joint_idx = action_joint_name_to_joint_idx[cur_joint_name] # 
                    # cur_joint = self.joint[cur_joint_name]
                    cur_state = states[cur_joint_idx] ### joint state ###
                    cur_joint.set_initial_state(cur_state)
                cur_child_struct.set_initial_state(states, action_joint_name_to_joint_idx, link_name_to_visited, link_name_to_link_struct)
        
        
    
    def get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes, link_name_to_link_struct, link_name_to_visited):
        link_name_to_visited[self.name] = 1
        
        # 'transformed_joint_pos': [], 'link_idxes': []
        if self.joint is not None:
            # for i_ch, (cur_joint, cur_child) in enumerate(zip(self.joint, self.children)):
            #     print(f"joint: {cur_joint.name}, child: {cur_child.name}, parent: {self.name}, child_visual: {cur_child.visual is not None}")
            #     joint_origin_xyz = cur_joint.origin_xyz
            #     init_visual_meshes = cur_child.get_init_visual_meshes(parent_rot, parent_trans + joint_origin_xyz, init_visual_meshes)
            # print(f"name: {self.name}, keys: {self.joint.keys()}")
            for cur_joint_name in self.joint: # 
                cur_joint = self.joint[cur_joint_name]
                cur_child_name = self.children[cur_joint_name]
                cur_child = link_name_to_link_struct[cur_child_name]
                # print(f"joint: {cur_joint.name}, child: {cur_child_name}, parent: {self.name}, child_visual: {cur_child.visual is not None}")
                # print(f"joint: {cur_joint.name}, child: {cur_child_name}, parent: {self.name}, child_visual: {cur_child.visual is not None}")
                joint_origin_xyz = cur_joint.origin_xyz
                if cur_child_name in link_name_to_visited:
                    continue
                cur_child_visual_pts = {'vertices': [], 'faces': [], 'link_idxes': [], 'transformed_joint_pos': [], 'joint_link_idxes': []}
                cur_child_visual_pts = cur_child.get_init_visual_meshes(parent_rot, parent_trans + joint_origin_xyz, cur_child_visual_pts, link_name_to_link_struct, link_name_to_visited)
                cur_child_verts, cur_child_faces = cur_child_visual_pts['vertices'], cur_child_visual_pts['faces']
                cur_child_link_idxes = cur_child_visual_pts['link_idxes']
                cur_transformed_joint_pos = cur_child_visual_pts['transformed_joint_pos']
                joint_link_idxes = cur_child_visual_pts['joint_link_idxes']
                if len(cur_child_verts) > 0:
                    cur_child_verts, cur_child_faces = merge_meshes(cur_child_verts, cur_child_faces)
                    cur_child_verts = cur_child_verts + cur_joint.origin_xyz.unsqueeze(0)
                    cur_joint_rot, cur_joint_trans = cur_joint.compute_transformation_from_current_state()
                    cur_child_verts = torch.matmul(cur_joint_rot, cur_child_verts.transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_joint_trans.unsqueeze(0)
                    
                    if len(cur_transformed_joint_pos) > 0:
                        cur_transformed_joint_pos = torch.cat(cur_transformed_joint_pos, dim=0)
                        cur_transformed_joint_pos = cur_transformed_joint_pos + cur_joint.origin_xyz.unsqueeze(0)
                        cur_transformed_joint_pos = torch.matmul(cur_joint_rot, cur_transformed_joint_pos.transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_joint_trans.unsqueeze(0)
                        cur_joint_pos = cur_joint_trans.unsqueeze(0).clone()
                        cur_transformed_joint_pos = torch.cat(
                            [cur_transformed_joint_pos, cur_joint_pos], dim=0 ##### joint poses #####
                        )
                    else:
                        cur_transformed_joint_pos = cur_joint_trans.unsqueeze(0).clone()
                    
                    if len(joint_link_idxes) > 0:
                        joint_link_idxes = torch.cat(joint_link_idxes, dim=-1) ### joint_link idxes ###
                        cur_joint_idx = cur_child.link_idx
                        joint_link_idxes = torch.cat(
                            [joint_link_idxes, torch.tensor([cur_joint_idx], dtype=torch.long).cuda()], dim=-1
                        )
                    else:
                        joint_link_idxes = torch.tensor([cur_child.link_idx], dtype=torch.long).cuda().view(1,)
                    
                    
                    
                    # cur_child_verts = cur_child_verts + # transformed joint pos #
                    cur_child_link_idxes = torch.cat(cur_child_link_idxes, dim=-1)
                    # joint_link_idxes = torch.cat(joint_link_idxes, dim=-1)
                    init_visual_meshes['vertices'].append(cur_child_verts)
                    init_visual_meshes['faces'].append(cur_child_faces)
                    init_visual_meshes['link_idxes'].append(cur_child_link_idxes)
                    init_visual_meshes['transformed_joint_pos'].append(cur_transformed_joint_pos)
                    init_visual_meshes['joint_link_idxes'].append(joint_link_idxes)
                    
                    
            # joint_origin_xyz = self.joint.origin_xyz
        else:
            joint_origin_xyz = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()
        # self.parent_rot_mtx = parent_rot
        # self.parent_trans_vec = parent_trans + joint_origin_xyz
        
        
        if self.visual is not None:
            init_visual_meshes = self.visual.get_init_visual_meshes(parent_rot, parent_trans, init_visual_meshes)
            cur_visual_mesh_pts_nn = self.visual.vertices.size(0)
            cur_link_idxes = torch.zeros((cur_visual_mesh_pts_nn, ), dtype=torch.long).cuda()+ self.link_idx
            init_visual_meshes['link_idxes'].append(cur_link_idxes)
        
        # for cur_link in self.children: # 
        #     init_visual_meshes = cur_link.get_init_visual_meshes(self.parent_rot_mtx, self.parent_trans_vec, init_visual_meshes)
        return init_visual_meshes ## init visual meshes ## 
    
    # calculate inerti
    def calculate_inertia(self, link_name_to_visited, link_name_to_link_struct):
        link_name_to_visited[self.name] = 1
        self.cur_inertia = torch.zeros((3, 3), dtype=torch.float32).cuda()
        
        if self.joint is not None:
            for joint_nm in self.joint:
                cur_joint = self.joint[joint_nm]
                cur_child = self.children[joint_nm]
                cur_child_struct = link_name_to_link_struct[cur_child]
                cur_child_name = cur_child_struct.name
                if cur_child_name in link_name_to_visited:
                    continue
                joint_rot, joint_trans = cur_joint.compute_transformation_from_current_state(n_grad=True)
                # cur_parent_rot = torch.matmul(parent_rot, joint_rot) # 
                # cur_parent_trans = torch.matmul(parent_rot, joint_trans.unsqueeze(-1)).squeeze(-1) + parent_trans # 
                child_inertia = cur_child_struct.calculate_inertia(link_name_to_visited, link_name_to_link_struct)
                child_inertia = torch.matmul(
                    joint_rot.detach(), torch.matmul(child_inertia, joint_rot.detach().transpose(1, 0).contiguous())
                ).detach()
                self.cur_inertia += child_inertia
        # if self.visual is not None:
        # self.cur_inertia += self.visual.inertia
        self.cur_inertia += self.inertial.inertia.detach()
        return self.cur_inertia
    
    
    def set_delta_state_and_update(self, states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, link_name_to_link_struct):
        
        link_name_to_visited[self.name] = 1
        
        if self.joint is not None:
            for cur_joint_name in self.joint:
                
                cur_joint = self.joint[cur_joint_name] # joint model 
                
                cur_child = self.children[cur_joint_name] # child model #
                
                cur_child_struct = link_name_to_link_struct[cur_child]
                
                cur_child_name = cur_child_struct.name
                
                if cur_child_name in link_name_to_visited:
                    continue
                
                ## cur child inertia ##
                # cur_child_inertia = cur_child_struct.cur_inertia
                
                
                if cur_joint.type in ['revolute']:
                    cur_joint_idx = action_joint_name_to_joint_idx[cur_joint_name]
                    cur_state = states[cur_joint_idx]
                    ### get the child struct ###
                    # set_actions_and_update_states(self, action, cur_timestep, time_cons, cur_inertia):
                    # set actions and update states #
                    cur_joint.set_delta_state_and_update(cur_state, cur_timestep)
                
                cur_child_struct.set_delta_state_and_update(states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, link_name_to_link_struct)

        
        
    # the joint #
    # set_actions_and_update_states(actions, cur_timestep, time_cons, action_joint_name_to_joint_idx, link_name_to_visited, self.link_name_to_link_struct)
    def set_actions_and_update_states(self, actions, cur_timestep, time_cons, action_joint_name_to_joint_idx, link_name_to_visited, link_name_to_link_struct):
        
        link_name_to_visited[self.name] = 1
        
        # the current joint of  the
        if self.joint is not None:
            for cur_joint_name in self.joint:
                
                cur_joint = self.joint[cur_joint_name] # joint model 
                
                cur_child = self.children[cur_joint_name] # child model #
                
                cur_child_struct = link_name_to_link_struct[cur_child]
                
                cur_child_name = cur_child_struct.name
                
                if cur_child_name in link_name_to_visited:
                    continue
                
                cur_child_inertia = cur_child_struct.cur_inertia
                
                
                if cur_joint.type in ['revolute']:
                    cur_joint_idx = action_joint_name_to_joint_idx[cur_joint_name]
                    cur_action = actions[cur_joint_idx]
                    ### get the child struct ###
                    # set_actions_and_update_states(self, action, cur_timestep, time_cons, cur_inertia):
                    # set actions and update states #
                    cur_joint.set_actions_and_update_states(cur_action, cur_timestep, time_cons, cur_child_inertia.detach())
                
                cur_child_struct.set_actions_and_update_states(actions, cur_timestep, time_cons, action_joint_name_to_joint_idx, link_name_to_visited, link_name_to_link_struct)


    def set_init_states_target_value(self, init_states):
        if self.joint.type == 'revolute':
            self.joint_angle = init_states[self.joint.joint_idx]
            joint_axis = self.joint.axis
            self.rot_vec = self.joint_angle * joint_axis
            self.joint.state = torch.tensor([1, 0, 0, 0], dtype=torch.float32).cuda()
            self.joint.state = self.joint.state + update_quaternion(self.rot_vec, self.joint.state)
            self.joint.timestep_to_states[0] = self.joint.state.detach()
            self.joint.timestep_to_vels[0] = torch.zeros((3,), dtype=torch.float32).cuda().detach() ## velocity ##
        for cur_link in self.children:
            cur_link.set_init_states_target_value(init_states)
    
    # should forward for one single step -> use the action #
    def set_init_states(self, ):
        self.joint.state = torch.tensor([1, 0, 0, 0], dtype=torch.float32).cuda()
        self.joint.timestep_to_states[0] = self.joint.state.detach()
        self.joint.timestep_to_vels[0] = torch.zeros((3,), dtype=torch.float32).cuda().detach() ## velocity ##
        for cur_link in self.children:
            cur_link.set_init_states() 
    
    
    def get_visual_pts(self, visual_pts_list):
        visual_pts_list = self.body.get_visual_pts(visual_pts_list)
        for cur_link in self.children:
            visual_pts_list = cur_link.get_visual_pts(visual_pts_list)
        visual_pts_list = torch.cat(visual_pts_list, dim=0)
        return visual_pts_list
    
    def get_visual_faces_list(self, visual_faces_list):
        visual_faces_list = self.body.get_visual_faces_list(visual_faces_list)
        for cur_link in self.children:
            visual_faces_list = cur_link.get_visual_faces_list(visual_faces_list)
        return visual_faces_list
        # pass

    
    def set_state(self, name_to_state):
        self.joint.set_state(name_to_state=name_to_state)
        for child_link in self.children:
            child_link.set_state(name_to_state)
            
    def set_state_via_vec(self, state_vec):
        self.joint.set_state_via_vec(state_vec)
        for child_link in self.children:
            child_link.set_state_via_vec(state_vec)




class Joint_Limit:
    def __init__(self, effort, lower, upper, velocity) -> None:
        self.effort = effort
        self.lower = lower
        self.velocity = velocity
        self.upper = upper
        pass

# Joint_urdf(name, joint_type, parent_link, child_link, origin_xyz, axis_xyz, limit: Joint_Limit)
class Joint_urdf: # 
    
    def __init__(self, name, joint_type, parent_link, child_link, origin_xyz, axis_xyz, limit: Joint_Limit) -> None:
        self.name = name
        self.type = joint_type
        self.parent_link = parent_link
        self.child_link = child_link
        self.origin_xyz = origin_xyz
        self.axis_xyz = axis_xyz
        self.limit = limit
        
        # joint angle; joint state #
        self.timestep_to_vels = {}
        self.timestep_to_states = {}
        
        self.init_pos = self.origin_xyz.clone()
        
        #### only for the current state #### # joint urdf #
        self.state = nn.Parameter(
            torch.tensor([1., 0., 0., 0.], dtype=torch.float32, requires_grad=True).cuda(), requires_grad=True
        )
        self.action = nn.Parameter(
            torch.zeros((1,), dtype=torch.float32, requires_grad=True).cuda(), requires_grad=True
        )
        # self.rot_mtx = np.eye(3, dtypes=np.float32)
        # self.trans_vec = np.zeros((3,), dtype=np.float32) ## rot m
        self.rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32, requires_grad=True).cuda(), requires_grad=True)
        self.trans_vec  = nn.Parameter(torch.zeros((3,), dtype=torch.float32,  requires_grad=True).cuda(), requires_grad=True)
        
    def set_initial_state(self, state):
        # joint angle as the state value #
        self.timestep_to_vels[0] = torch.zeros((3,), dtype=torch.float32).cuda().detach() ## velocity ##
        delta_rot_vec = self.axis_xyz * state
        # self.timestep_to_states[0] = state.detach()
        cur_state = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
        init_state = cur_state + update_quaternion(delta_rot_vec, cur_state)
        self.timestep_to_states[0] = init_state.detach()
        self.state = init_state
        
    def set_delta_state_and_update(self, state, cur_timestep):
        self.timestep_to_vels[cur_timestep] = torch.zeros((3,), dtype=torch.float32).cuda().detach() 
        delta_rot_vec = self.axis_xyz * state
        if cur_timestep == 0:
            prev_state = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
        else:
            prev_state = self.timestep_to_states[cur_timestep - 1].detach()
        cur_state = prev_state + update_quaternion(delta_rot_vec, prev_state)
        self.timestep_to_states[cur_timestep] = cur_state.detach()
        self.state = cur_state
        
        
    
    def compute_transformation_from_current_state(self, n_grad=False):
        # together with the parent rot mtx and the parent trans vec #
        # cur_joint_state = self.state
        if self.type == "revolute":
            # rot_mtx = rotation_matrix_from_axis_angle(self.axis, cur_joint_state)
            # trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
            if n_grad:
                rot_mtx = quaternion_to_matrix(self.state.detach())
            else:
                rot_mtx = quaternion_to_matrix(self.state)
            # trans_vec = self.pos - torch.matmul(rot_mtx, self.pos.view(3, 1)).view(3).contiguous()
            trans_vec = self.origin_xyz - torch.matmul(rot_mtx, self.origin_xyz.view(3, 1)).view(3).contiguous()
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec
        elif self.type == "fixed":
            rot_mtx = torch.eye(3, dtype=torch.float32).cuda()
            trans_vec = torch.zeros((3,), dtype=torch.float32).cuda()
            # trans_vec = self.origin_xyz
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec # 
        else:
            pass
        return self.rot_mtx, self.trans_vec

    
    # set actions # set actions and udpate states #
    def set_actions_and_update_states(self, action, cur_timestep, time_cons, cur_inertia):
        
        # timestep_to_vels, timestep_to_states, state #
        if self.type in ['revolute']:
            
            self.action = action
            # 
            # visual_pts and visual_pts_mass #
            # cur_joint_pos = self.joint.pos #
            # TODO: check whether the following is correct #
            torque = self.action * self.axis_xyz
            
            # # Compute inertia matrix #
            # inertial = torch.zeros((3, 3), dtype=torch.float32).cuda()
            # for i_pts in range(self.visual_pts.size(0)):
            #     cur_pts = self.visual_pts[i_pts]
            #     cur_pts_mass = self.visual_pts_mass[i_pts]
            #     cur_r = cur_pts - cur_joint_pos # r_i
            #     # cur_vert = init_passive_mesh[i_v]
            #     # cur_r = cur_vert - init_passive_mesh_center
            #     dot_r_r = torch.sum(cur_r * cur_r)
            #     cur_eye_mtx = torch.eye(3, dtype=torch.float32).cuda()
            #     r_mult_rT = torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0))
            #     inertial += (dot_r_r * cur_eye_mtx - r_mult_rT) * cur_pts_mass
            # m = torch.sum(self.visual_pts_mass)
            # # Use torque to update angular velocity -> state #
            # inertia_inv = torch.linalg.inv(inertial)
            
            # axis-angle of 
            # inertia_inv = self.cur_inertia_inv
            # print(f"updating actions and states for the joint {self.name} with type {self.type}")
            inertia_inv = torch.linalg.inv(cur_inertia).detach()
            
            delta_omega = torch.matmul(inertia_inv, torque.unsqueeze(-1)).squeeze(-1)
            
            # delta_omega = torque / 400 # 
            
            # timestep_to_vels, timestep_to_states, state #
            
            # TODO: dt should be an optimizable constant? should it be the same value as that optimized for the passive object? # 
            delta_angular_vel = delta_omega * time_cons #  * self.args.dt
            delta_angular_vel = delta_angular_vel.squeeze(0)
            if cur_timestep > 0: ## cur_timestep - 1 ##
                prev_angular_vel = self.timestep_to_vels[cur_timestep - 1].detach()
                cur_angular_vel = prev_angular_vel + delta_angular_vel * DAMPING
            else:
                cur_angular_vel = delta_angular_vel
            
            self.timestep_to_vels[cur_timestep] = cur_angular_vel.detach()

            cur_delta_quat = cur_angular_vel * time_cons #  * self.args.dt
            cur_delta_quat = cur_delta_quat.squeeze(0)
            cur_state = self.timestep_to_states[cur_timestep].detach() # quaternion #
            # print(f"cur_delta_quat: {cur_delta_quat.size()}, cur_state: {cur_state.size()}")
            nex_state = cur_state + update_quaternion(cur_delta_quat, cur_state)
            self.timestep_to_states[cur_timestep + 1] = nex_state.detach()
            self.state = nex_state  # set the joint state #


# get_transformed_visual_pts() --- transformed_visual_pts ##
# use the transformed visual # the articulated motion field # 
# then we should add the free motion field here # # add the free motion field #  # hwo to use that? #
# another rules for optimizing articulation motion field # 
# -> the articulated model predicted transformations #
# -> the free motion field -> the motion field predicted by the network for each timestep -> an implicit motion field #

class Robot_urdf:
    def __init__(self, links, link_name_to_link_idxes, link_name_to_link_struct, joint_name_to_joint_idx, actions_joint_name_to_joint_idx) -> None:
        self.links = links
        self.link_name_to_link_idxes = link_name_to_link_idxes
        self.link_name_to_link_struct = link_name_to_link_struct
        # joint_name_to_joint_idx, actions_joint_name_to_joint_idx
        self.joint_name_to_joint_idx = joint_name_to_joint_idx
        self.actions_joint_name_to_joint_idx = actions_joint_name_to_joint_idx
        
        # 
        # particles 
        # sample particles 
        # how to sample particles? 
        # how to expand the particles? # -> you can use weights in the model dict # 
        # from grids and jample from grids # 
        # link idx to the 
        # robot # 
        # init vertices, init faces # 
        # expande the aprticles # 
        # expanede particles # 
        # use particles to conduct the simulation # 
        
        
        
        self.init_vertices, self.init_faces = self.get_init_visual_pts()
        
        
        init_visual_pts_sv_fn = "robot_expanded_visual_pts.npy"
        np.save(init_visual_pts_sv_fn, self.init_vertices.detach().cpu().numpy())
        
        joint_name_to_joint_idx_sv_fn = "mano_joint_name_to_joint_idx.npy"
        np.save(joint_name_to_joint_idx_sv_fn, self.joint_name_to_joint_idx)
        
        actions_joint_name_to_joint_idx_sv_fn = "mano_actions_joint_name_to_joint_idx.npy"
        np.save(actions_joint_name_to_joint_idx_sv_fn, self.actions_joint_name_to_joint_idx)
        
        tot_joints = len(self.joint_name_to_joint_idx)
        tot_actions_joints = len(self.actions_joint_name_to_joint_idx)
        
        print(f"tot_joints: {tot_joints}, tot_actions_joints: {tot_actions_joints}")
        
        pass
    
    
    
    
    def expand_visual_pts(self, ):
        link_name_to_visited = {}
        # transform the visual pts #
        # action_joint_name_to_joint_idx = self.actions_joint_name_to_joint_idx
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        expanded_visual_pts = []
        # expanded the visual pts # # transformed viusal pts # or the translations of the visual pts #
        expanded_visual_pts = palm_link.expand_visual_pts(expanded_visual_pts, link_name_to_visited, self.link_name_to_link_struct) 
        expanded_visual_pts = torch.cat(expanded_visual_pts, dim=0)
        # pass 
        return expanded_visual_pts
    
    
    # get_transformed_visual_pts() # get_transformed_visual_pts of the visual pts ### get_transformed_visual_pts ## get_transformed_visual_pts ### # 
    def get_transformed_visual_pts(self, ):
        init_visual_pts = []
        link_name_to_visited = {}
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        ### init_visual_pts # from the pal mink to get the total transformed visual pts ## 
        init_visual_pts = palm_link.get_transformed_visual_pts(init_visual_pts, link_name_to_visited, self.link_name_to_link_struct)
        
        init_visual_pts = torch.cat(init_visual_pts, dim=0) ## get the inita visual pts from the palm link ### 
        return init_visual_pts
    
    
    ### samping issue? --- TODO`    `
    def get_init_visual_pts(self, ):
        init_visual_meshes = {
            'vertices': [], 'faces': [], 'link_idxes': [], 'transformed_joint_pos': [], 'link_idxes': [],  'transformed_joint_pos': [], 'joint_link_idxes': []
        }
        init_parent_rot = torch.eye(3, dtype=torch.float32).cuda()
        init_parent_trans = torch.zeros((3,), dtype=torch.float32).cuda()
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        init_visual_meshes = palm_link.get_init_visual_meshes(init_parent_rot, init_parent_trans, init_visual_meshes, self.link_name_to_link_struct, link_name_to_visited)
        
        self.link_idxes = torch.cat(init_visual_meshes['link_idxes'], dim=-1)
        self.transformed_joint_pos = torch.cat(init_visual_meshes['transformed_joint_pos'], dim=0)
        self.joint_link_idxes = torch.cat(init_visual_meshes['joint_link_idxes'], dim=-1) ### 
        
        
        
        # for cur_link in self.links:
        #     init_visual_meshes = cur_link.get_init_visual_meshes(init_parent_rot, init_parent_trans, init_visual_meshes, self.link_name_to_link_struct, link_name_to_visited)

        init_vertices, init_faces = merge_meshes(init_visual_meshes['vertices'], init_visual_meshes['faces'])
        return init_vertices, init_faces
    
    
    def set_delta_state_and_update(self, states, cur_timestep):
        link_name_to_visited = {}
        
        action_joint_name_to_joint_idx = self.actions_joint_name_to_joint_idx
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        palm_link.set_delta_state_and_update(states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, self.link_name_to_link_struct)
    
    # cur_joint.set_actions_and_update_states(cur_action, cur_timestep, time_cons, cur_child_inertia)
    def set_actions_and_update_states(self, actions, cur_timestep, time_cons,):
        # actions 
        # self.actions_joint_name_to_joint_idx as the action joint name to joint idx 
        link_name_to_visited = {}
        
        action_joint_name_to_joint_idx = self.actions_joint_name_to_joint_idx
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        palm_link.set_actions_and_update_states(actions, cur_timestep, time_cons, action_joint_name_to_joint_idx, link_name_to_visited, self.link_name_to_link_struct)
        
        # for cur_joint in 
        
        # for cur_link in self.links:
        #     if cur_link.joint is not None:
        #         for cur_joint_nm in cur_link.joint:
        #             if cur_link.joint[cur_joint_nm].type in ['revolute']:
        #                 cur_link_joint_name = cur_link.joint[cur_joint_nm].name
        #                 cur_link_joint_idx = self.actions_joint_name_to_joint_idx[cur_link_joint_name]
                        
        
        # for cur_link in self.links:
        #     cur_link.set_actions_and_update_states(actions, cur_timestep, time_cons, action_joint_name_to_joint_idx, link_name_to_visited, self.link_name_to_link_struct)
        
    ### TODO: add the contact torque when calculating the nextstep states ###
    ### TODO: not an accurate implementation since differen joints should be jconsidered for one single link ###
    ### TODO: the articulated force modle is not so easy as this one .... ###
    def set_contact_forces(self, hard_selected_forces, hard_selected_manipulating_points, hard_selected_sampled_input_pts_idxes):
        # transformed_joint_pos, joint_link_idxes, link_idxes #
        selected_pts_link_idxes = self.link_idxes[hard_selected_sampled_input_pts_idxes]
        # use the selected link idxes # 
        # selected pts idxes #
        
        # self.joint_link_idxes, transformed_joint_pos #
        self.link_idx_to_transformed_joint_pos = {}
        for i_link in range(self.transformed_joint_pos.size(0)):
            cur_link_idx = self.link_idxes[i_link].item()
            cur_link_pos = self.transformed_joint_pos[i_link]
            # if cur_link_idx not in self.link_idx_to_transformed_joint_pos:
            self.link_idx_to_transformed_joint_pos[cur_link_idx] = cur_link_pos
            # self.link_idx_to_transformed_joint_pos[cur_link_idx].append(cur_link_pos)
        
        # from the 
        self.link_idx_to_contact_forces = {}
        for i_c_pts in range(hard_selected_forces.size(0)):
            cur_contact_force = hard_selected_forces[i_c_pts] ## 
            cur_link_idx = selected_pts_link_idxes[i_c_pts].item()
            cur_link_pos = self.link_idx_to_transformed_joint_pos[cur_link_idx]
            cur_link_action_pos = hard_selected_manipulating_points[i_c_pts]
            # (action_pos - link_pos) x (-contact_force) #
            cur_contact_torque = torch.cross(
                cur_link_action_pos - cur_link_pos, -cur_contact_force
            )
            if cur_link_idx not in self.link_idx_to_contact_forces:
                self.link_idx_to_contact_forces[cur_link_idx] = [cur_contact_torque]
            else:
                self.link_idx_to_contact_forces[cur_link_idx].append(cur_contact_torque)
        for link_idx in self.link_idx_to_contact_forces:
            self.link_idx_to_contact_forces[link_idx] = torch.stack(self.link_idx_to_contact_forces[link_idx], dim=0)
            self.link_idx_to_contact_forces[link_idx] = torch.sum(self.link_idx_to_contact_forces[link_idx] , dim=0)
        for link_idx, link_struct in enumerate(self.links):
            if link_idx in self.link_idx_to_contact_forces:
                cur_link_contact_force = self.link_idx_to_contact_forces[link_idx]
                link_struct.contact_torque = cur_link_contact_force
            else:
                link_struct.contact_torque = None
        
    
    # def se ### from the optimizable initial states ###
    def set_initial_state(self, states):
        action_joint_name_to_joint_idx = self.actions_joint_name_to_joint_idx
        link_name_to_visited = {}
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        palm_link.set_initial_state(states, action_joint_name_to_joint_idx, link_name_to_visited, self.link_name_to_link_struct)
        
        # for cur_link in self.links:
        #     cur_link.set_initial_state(states, action_joint_name_to_joint_idx, link_name_to_visited, self.link_name_to_link_struct)
            
    ### after each timestep -> re-calculate the inertial matrix using the current simulated states and the set the new actiosn and forward the simulation # 
    def calculate_inertia(self):
        link_name_to_visited = {}
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        palm_link.calculate_inertia(link_name_to_visited, self.link_name_to_link_struct)
        
        # for cur_link in self.links:
        #     cur_link.calculate_inertia(link_name_to_visited, self.link_name_to_link_struct)

    
    


def parse_nparray_from_string(strr, args=None):
    vals = strr.split(" ")
    vals = [float(val) for val in vals]
    vals = np.array(vals, dtype=np.float32)
    vals = torch.from_numpy(vals).float()
    ## vals ##
    vals = nn.Parameter(vals.cuda(), requires_grad=True)
    
    return vals


### parse link data ###
def parse_link_data(link, args):
    
    link_name = link.attrib["name"]
    # print(f"parsing link: {link_name}") ## joints body meshes #
    
    joint = link.find("./joint")
    
    joint_name = joint.attrib["name"]
    joint_type = joint.attrib["type"]
    if joint_type in ["revolute"]: ## a general xml parser here? 
        axis = joint.attrib["axis"]
        axis = parse_nparray_from_string(axis, args=args)
    else:
        axis = None
    pos = joint.attrib["pos"] # 
    pos = parse_nparray_from_string(pos, args=args)
    quat = joint.attrib["quat"]
    quat = parse_nparray_from_string(quat, args=args)
    
    try:
        frame = joint.attrib["frame"]
    except:
        frame = "WORLD"
    
    if joint_type not in ["fixed"]:
        damping = joint.attrib["damping"]
        damping = float(damping)
    else:
        damping = 0.0
    
    cur_joint = Joint(joint_name, joint_type, axis, pos, quat, frame, damping, args=args)
    
    body = link.find("./body")
    body_name = body.attrib["name"]
    body_type = body.attrib["type"]
    if body_type == "mesh":
        filename = body.attrib["filename"]
    else:
        filename = ""
    
    if body_type == "sphere":
        radius = body.attrib["radius"]
        radius = float(radius)
    else:
        radius = 0.
    
    pos = body.attrib["pos"]
    pos = parse_nparray_from_string(pos, args=args)
    quat = body.attrib["quat"]
    quat = joint.attrib["quat"]
    try:
        transform_type = body.attrib["transform_type"]
    except:
        transform_type = "OBJ_TO_WORLD"
    density = body.attrib["density"]
    density = float(density)
    mu = body.attrib["mu"]
    mu = float(mu)
    try: ## rgba ##
        rgba = body.attrib["rgba"]
        rgba = parse_nparray_from_string(rgba, args=args)
    except:
        rgba = np.zeros((4,), dtype=np.float32)
    
    cur_body = Body(body_name, body_type, filename, pos, quat, transform_type, density, mu, rgba, radius, args=args)
    
    children_link = []
    links = link.findall("./link")
    for child_link in links: # 
        cur_child_link = parse_link_data(child_link, args=args)
        children_link.append(cur_child_link)
    
    link_name = link.attrib["name"]
    link_obj = Link(link_name, joint=cur_joint, body=cur_body, children=children_link, args=args)
    return link_obj


### parse link data ###
def parse_link_data_urdf(link):
    
    link_name = link.attrib["name"]
    # print(f"parsing link: {link_name}") ## joints body meshes #
    
    inertial = link.find("./inertial")
    
    origin = inertial.find("./origin")
    inertial_pos = origin.attrib["xyz"]
    inertial_pos = parse_nparray_from_string(inertial_pos)
    
    inertial_rpy = origin.attrib["rpy"]
    inertial_rpy = parse_nparray_from_string(inertial_rpy)
    
    inertial_mass = inertial.find("./mass")
    inertial_mass = inertial_mass.attrib["value"]
    
    inertial_inertia = inertial.find("./inertia")
    inertial_ixx = inertial_inertia.attrib["ixx"]
    inertial_ixx = float(inertial_ixx)
    inertial_ixy = inertial_inertia.attrib["ixy"]
    inertial_ixy = float(inertial_ixy)
    inertial_ixz = inertial_inertia.attrib["ixz"]
    inertial_ixz = float(inertial_ixz)
    inertial_iyy = inertial_inertia.attrib["iyy"]
    inertial_iyy = float(inertial_iyy)
    inertial_iyz = inertial_inertia.attrib["iyz"]
    inertial_iyz = float(inertial_iyz)
    inertial_izz = inertial_inertia.attrib["izz"]
    inertial_izz = float(inertial_izz)
    
    inertial_inertia_mtx = torch.zeros((3, 3), dtype=torch.float32).cuda()
    inertial_inertia_mtx[0, 0] = inertial_ixx
    inertial_inertia_mtx[0, 1] = inertial_ixy
    inertial_inertia_mtx[0, 2] = inertial_ixz
    inertial_inertia_mtx[1, 0] = inertial_ixy
    inertial_inertia_mtx[1, 1] = inertial_iyy
    inertial_inertia_mtx[1, 2] = inertial_iyz
    inertial_inertia_mtx[2, 0] = inertial_ixz
    inertial_inertia_mtx[2, 1] = inertial_iyz
    inertial_inertia_mtx[2, 2] = inertial_izz
    
    # [xx, xy, xz] # 
    # [0, yy, yz] #
    # [0, 0, zz] #
    
    # a strange inertia value ... #
    # TODO: how to compute the inertia matrix? #
    
    visual = link.find("./visual")
    
    if visual is not None:
        origin = visual.find("./origin")
        visual_pos = origin.attrib["xyz"]
        visual_pos = parse_nparray_from_string(visual_pos)
        visual_rpy = origin.attrib["rpy"]
        visual_rpy = parse_nparray_from_string(visual_rpy)
        geometry = visual.find("./geometry")
        geometry_mesh = geometry.find("./mesh")
        mesh_fn = geometry_mesh.attrib["filename"]
        mesh_scale = geometry_mesh.attrib["scale"]
        
        mesh_scale = parse_nparray_from_string(mesh_scale)
        mesh_fn = str(mesh_fn)
        
    
    link_struct = Link_urdf(name=link_name, inertial=Inertial(origin_rpy=inertial_rpy, origin_xyz=inertial_pos, mass=inertial_mass, inertia=inertial_inertia_mtx), visual=Visual(visual_rpy=visual_rpy, visual_xyz=visual_pos, geometry_mesh_fn=mesh_fn, geometry_mesh_scale=mesh_scale) if visual is not None else None)

    return link_struct

def parse_joint_data_urdf(joint):
    joint_name = joint.attrib["name"]
    joint_type = joint.attrib["type"]
    
    parent = joint.find("./parent")
    child = joint.find("./child")
    parent_name = parent.attrib["link"]
    child_name = child.attrib["link"]
    
    joint_origin = joint.find("./origin")
    # if joint_origin.
    try:
        origin_xyz = joint_origin.attrib["xyz"]
        origin_xyz = parse_nparray_from_string(origin_xyz)
    except:
        origin_xyz = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()
        
    joint_axis = joint.find("./axis")
    if joint_axis is not None:
        joint_axis = joint_axis.attrib["xyz"]
        joint_axis = parse_nparray_from_string(joint_axis)
    else:
        joint_axis = torch.tensor([1, 0., 0.], dtype=torch.float32).cuda()
    
    joint_limit = joint.find("./limit")
    if joint_limit is not None:
        joint_lower = joint_limit.attrib["lower"]
        joint_lower = float(joint_lower)
        joint_upper = joint_limit.attrib["upper"]
        joint_upper = float(joint_upper)
        joint_effort = joint_limit.attrib["effort"]
        joint_effort = float(joint_effort)
        joint_velocity = joint_limit.attrib["velocity"]
        joint_velocity = float(joint_velocity)
    else:
        joint_lower = -0.5000
        joint_upper = 1.57
        joint_effort = 1000
        joint_velocity = 0.5
    
    # cosntruct the joint data #      
    joint_limit = Joint_Limit(effort=joint_effort, lower=joint_lower, upper=joint_upper, velocity=joint_velocity)
    cur_joint_struct = Joint_urdf(joint_name, joint_type, parent_name, child_name, origin_xyz, joint_axis, joint_limit)
    return cur_joint_struct
    


def parse_data_from_urdf(xml_fn):
    
    tree = ElementTree()
    tree.parse(xml_fn)
    print(f"{xml_fn}")
    ### get total robots ###
    # robots = tree.findall("link")
    cur_robot = tree
    # i_robot = 0 #
    # tot_robots = [] #
    # for cur_robot in robots: #
    # print(f"Getting robot: {i_robot}") #
    # i_robot += 1 #
    # print(f"len(robots): {len(robots)}") #
    # cur_robot = robots[0] #
    cur_links = cur_robot.findall("./link")
    # i_link = 0
    link_name_to_link_idxes = {}
    cur_robot_links = []
    link_name_to_link_struct = {}
    for i_link_idx, cur_link in enumerate(cur_links):
        cur_link_struct = parse_link_data_urdf(cur_link)
        print(f"Adding link {cur_link_struct.name}")
        cur_link_struct.link_idx = i_link_idx
        cur_robot_links.append(cur_link_struct)
        
        link_name_to_link_idxes[cur_link_struct.name] = i_link_idx
        link_name_to_link_struct[cur_link_struct.name] = cur_link_struct
    # for cur_link in cur_links:
    #     cur_robot_links.append(parse_link_data_urdf(cur_link, args=args))   
    
    print(f"link_name_to_link_struct: {len(link_name_to_link_struct)}, ")
    
    tot_robot_joints = []
    
    joint_name_to_joint_idx = {}
    
    actions_joint_name_to_joint_idx = {}
    
    cur_joints = cur_robot.findall("./joint")
    for i_joint, cur_joint in enumerate(cur_joints):
        cur_joint_struct = parse_joint_data_urdf(cur_joint)
        cur_joint_parent_link = cur_joint_struct.parent_link
        cur_joint_child_link = cur_joint_struct.child_link
        
        cur_joint_idx = len(tot_robot_joints)
        cur_joint_name = cur_joint_struct.name
        
        joint_name_to_joint_idx[cur_joint_name] = cur_joint_idx
        
        cur_joint_type = cur_joint_struct.type
        if cur_joint_type in ['revolute']:
            actions_joint_name_to_joint_idx[cur_joint_name] = cur_joint_idx
        
        
        #### add the current joint to tot joints ###
        tot_robot_joints.append(cur_joint_struct)
        
        parent_link_idx = link_name_to_link_idxes[cur_joint_parent_link]
        cur_parent_link_struct = cur_robot_links[parent_link_idx]
        
        
        child_link_idx = link_name_to_link_idxes[cur_joint_child_link]
        cur_child_link_struct = cur_robot_links[child_link_idx]
        # parent link struct #
        if link_name_to_link_struct[cur_joint_parent_link].joint is not None:
            link_name_to_link_struct[cur_joint_parent_link].joint[cur_joint_struct.name] = cur_joint_struct
            link_name_to_link_struct[cur_joint_parent_link].children[cur_joint_struct.name] = cur_child_link_struct.name
            # cur_child_link_struct
            # cur_parent_link_struct.joint.append(cur_joint_struct)
            # cur_parent_link_struct.children.append(cur_child_link_struct)
        else:
            link_name_to_link_struct[cur_joint_parent_link].joint = {
                cur_joint_struct.name: cur_joint_struct
            }
            link_name_to_link_struct[cur_joint_parent_link].children = {
                cur_joint_struct.name: cur_child_link_struct.name
                    # cur_child_link_struct
            }
            # cur_parent_link_struct.joint = [cur_joint_struct]
            # cur_parent_link_struct.children.append(cur_child_link_struct)
        # pass
        
        
    cur_robot_obj = Robot_urdf(cur_robot_links, link_name_to_link_idxes, link_name_to_link_struct, joint_name_to_joint_idx, actions_joint_name_to_joint_idx)
        # tot_robots.append(cur_robot_obj)
    
    # for the joint robots #
    # for every joint 
    # tot_actuators = []
    # actuators = tree.findall("./actuator/motor")
    # joint_nm_to_joint_idx = {}
    # i_act = 0
    # for cur_act in actuators:
    #     cur_act_joint_nm  = cur_act.attrib["joint"]
    #     joint_nm_to_joint_idx[cur_act_joint_nm] = i_act
    #     i_act += 1 ### add the act ###
    
    # tot_robots[0].set_joint_idx(joint_nm_to_joint_idx) ### set joint idx here ### # tot robots #
    # tot_robots[0].get_nn_pts()
    # tot_robots[1].get_nn_pts()
    
    return cur_robot_obj


def get_name_to_state_from_str(states_str):
    tot_states = states_str.split(" ")
    tot_states = [float(cur_state) for cur_state in tot_states]
    joint_name_to_state = {}
    for i in range(len(tot_states)):
        cur_joint_name = f"joint{i + 1}"
        cur_joint_state = tot_states[i]
        joint_name_to_state[cur_joint_name] = cur_joint_state
    return joint_name_to_state


def merge_meshes(verts_list, faces_list):
    nn_verts = 0
    tot_verts_list = []
    tot_faces_list = []
    for i_vv, cur_verts in enumerate(verts_list):
        cur_verts_nn = cur_verts.size(0)
        tot_verts_list.append(cur_verts)
        tot_faces_list.append(faces_list[i_vv] + nn_verts)
        nn_verts = nn_verts + cur_verts_nn
    tot_verts_list = torch.cat(tot_verts_list, dim=0)
    tot_faces_list = torch.cat(tot_faces_list, dim=0)
    return tot_verts_list, tot_faces_list



class RobotAgent: # robot and the robot #
    def __init__(self, xml_fn) -> None:
        self.xml_fn = xml_fn
        # self.args = args
        
        ## 
        active_robot =  parse_data_from_urdf(xml_fn)
        
        self.time_constant = nn.Embedding(
            num_embeddings=3, embedding_dim=1
        ).cuda()
        torch.nn.init.ones_(self.time_constant.weight) #
        self.time_constant.weight.data = self.time_constant.weight.data * 0.2 ### time_constant data #
        
        self.optimizable_actions = nn.Embedding(
            num_embeddings=100, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(self.optimizable_actions.weight) #
        
        self.learning_rate = 5e-4
    
        self.active_robot = active_robot
        
        # # get init states # #
        self.set_init_states()
        init_visual_pts = self.get_init_state_visual_pts()
        self.init_visual_pts = init_visual_pts
        
        
    def set_init_states_target_value(self, init_states):
        # glb_rot = torch.eye(n=3, dtype=torch.float32).cuda()
        # glb_trans = torch.zeros((3,), dtype=torch.float32).cuda() ### glb_trans #### and the rot 3##

        # tot_init_states = {}
        # tot_init_states['glb_rot'] = glb_rot;
        # tot_init_states['glb_trans'] = glb_trans; 
        # tot_init_states['links_init_states'] = init_states
        # self.active_robot.set_init_states_target_value(tot_init_states)
        # init_joint_states = torch.zeros((60, ), dtype=torch.float32).cuda()
        self.active_robot.set_initial_state(init_states)
    
    def set_init_states(self):
        # glb_rot = torch.eye(n=3, dtype=torch.float32).cuda()
        # glb_trans = torch.zeros((3,), dtype=torch.float32).cuda() ### glb_trans #### and the rot 3##
        
        # ### random rotation ###
        # # glb_rot_np = R.random().as_matrix()
        # # glb_rot = torch.from_numpy(glb_rot_np).float().cuda()
        # ### random rotation ###
        
        # # glb_rot, glb_trans #
        # init_states = {} # init states #
        # init_states['glb_rot'] = glb_rot; # 
        # init_states['glb_trans'] = glb_trans; 
        # self.active_robot.set_init_states(init_states)
        
        init_joint_states = torch.zeros((60, ), dtype=torch.float32).cuda()
        self.active_robot.set_initial_state(init_joint_states)
    
    def get_init_state_visual_pts(self, ):
        # visual_pts_list = [] # compute the transformation via current state #
        # visual_pts_list, visual_pts_mass_list = self.active_robot.compute_transformation_via_current_state( visual_pts_list)
        cur_verts, cur_faces = self.active_robot.get_init_visual_pts()
        self.faces = cur_faces
        # init_visual_pts = visual_pts_list
        return cur_verts
    
    def set_actions_and_update_states(self, actions, cur_timestep):
        # 
        time_cons = self.time_constant(torch.zeros((1,), dtype=torch.long).cuda()) ### time constant of the system ##
        self.active_robot.set_actions_and_update_states(actions, cur_timestep, time_cons) ### 
        pass
    
    def forward_stepping_test(self, ):
        # delta_glb_rot; delta_glb_trans #
        timestep_to_visual_pts = {}
        for i_step in range(50):
            actions = {}
            actions['delta_glb_rot']  = torch.eye(3, dtype=torch.float32).cuda()
            actions['delta_glb_trans'] = torch.zeros((3,), dtype=torch.float32).cuda()
            actions_link_actions = torch.ones((22, ), dtype=torch.float32).cuda()
            # actions_link_actions = actions_link_actions * 0.2
            actions_link_actions = actions_link_actions * -1. # 
            actions['link_actions'] = actions_link_actions
            self.set_actions_and_update_states(actions=actions, cur_timestep=i_step)
    
            cur_visual_pts = robot_agent.get_init_state_visual_pts()
            cur_visual_pts = cur_visual_pts.detach().cpu().numpy()
            timestep_to_visual_pts[i_step + 1] = cur_visual_pts
        return timestep_to_visual_pts
    
    def initialize_optimization(self, reference_pts_dict):
        self.n_timesteps = 50
        # self.n_timesteps = 19 # first 19-timesteps optimization #
        self.nn_tot_optimization_iters = 100
        # self.nn_tot_optimization_iters = 57
        # TODO: load reference points #
        self.ts_to_reference_pts = np.load(reference_pts_dict, allow_pickle=True).item() #### 
        self.ts_to_reference_pts = {
            ts // 2 + 1: torch.from_numpy(self.ts_to_reference_pts[ts]).float().cuda() for ts in self.ts_to_reference_pts
        }
    
    
    def forward_stepping_optimization(self, ):
        nn_tot_optimization_iters = self.nn_tot_optimization_iters
        params_to_train = []
        params_to_train += list(self.optimizable_actions.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        for i_iter in range(nn_tot_optimization_iters):
            
            tot_losses = []
            ts_to_robot_points = {}
            for cur_ts in range(self.n_timesteps):
                # print(f"iter: {i_iter}, cur_ts: {cur_ts}")
                # actions = {}
                # actions['delta_glb_rot']  = torch.eye(3, dtype=torch.float32).cuda()
                # actions['delta_glb_trans'] = torch.zeros((3,), dtype=torch.float32).cuda()
                actions_link_actions = self.optimizable_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # actions_link_actions = actions_link_actions * 0.2
                # actions_link_actions = actions_link_actions * -1. # 
                # actions['link_actions'] = actions_link_actions
                # self.set_actions_and_update_states(actions=actions, cur_timestep=cur_ts) # update the interaction # 
                
                with torch.no_grad():
                    self.active_robot.calculate_inertia()
    
                self.active_robot.set_actions_and_update_states(actions_link_actions, cur_ts, 0.2)

                cur_visual_pts, cur_faces = self.active_robot.get_init_visual_pts()
                ts_to_robot_points[cur_ts + 1] = cur_visual_pts.clone()
                
                cur_reference_pts = self.ts_to_reference_pts[cur_ts + 1]
                diff = torch.sum((cur_visual_pts - cur_reference_pts) ** 2, dim=-1)
                diff = diff.mean()
                
                # diff.
                self.optimizer.zero_grad()
                diff.backward(retain_graph=True)
                # diff.backward(retain_graph=False)
                self.optimizer.step()
                
                tot_losses.append(diff.item())

                
            loss = sum(tot_losses) / float(len(tot_losses))
            print(f"Iter: {i_iter}, average loss: {loss}")
            # print(f"Iter: {i_iter}, average loss: {loss.item()}, start optimizing")
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
        
        self.ts_to_robot_points = {
            ts: ts_to_robot_points[ts].detach().cpu().numpy() for ts in ts_to_robot_points
        }
        self.ts_to_ref_points = {
            ts: self.ts_to_reference_pts[ts].detach().cpu().numpy() for ts in ts_to_robot_points
        }
        return self.ts_to_robot_points, self.ts_to_ref_points




def rotation_matrix_from_axis_angle(axis, angle): # rotation_matrix_from_axis_angle -> 
        # sin_ = np.sin(angle) #  ti.math.sin(angle)
        # cos_ = np.cos(angle) #  ti.math.cos(angle)
        sin_ = torch.sin(angle) #  ti.math.sin(angle)
        cos_ = torch.cos(angle) #  ti.math.cos(angle)
        u_x, u_y, u_z = axis[0], axis[1], axis[2]
        u_xx = u_x * u_x
        u_yy = u_y * u_y
        u_zz = u_z * u_z
        u_xy = u_x * u_y
        u_xz = u_x * u_z
        u_yz = u_y * u_z ## 
        
        
        row_a = torch.stack(
            [cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dim=0
        )
        # print(f"row_a: {row_a.size()}")
        row_b = torch.stack(
            [u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dim=0
        )
        # print(f"row_b: {row_b.size()}")
        row_c = torch.stack(
            [u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dim=0
        )
        # print(f"row_c: {row_c.size()}")
        
        ### rot_mtx for the rot_mtx ###
        rot_mtx = torch.stack(
            [row_a, row_b, row_c], dim=-1 ### rot_matrix of he matrix ##
        )
        
        return rot_mtx



#### Big TODO: the external contact forces from the manipulated object to the robot ####
if __name__=='__main__':
    
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_nocoll_simplified.urdf"
    robot_agent = RobotAgent(urdf_fn)
    
    ref_dict_npy = "reference_verts.npy"
    robot_agent.initialize_optimization(ref_dict_npy)
    ts_to_robot_points, ts_to_ref_points = robot_agent.forward_stepping_optimization()
    np.save(f"ts_to_robot_points.npy", ts_to_robot_points)
    np.save(f"ts_to_ref_points.npy", ts_to_ref_points)
    exit(0)
    
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_nocoll_simplified.urdf"
    cur_robot = parse_data_from_urdf(urdf_fn)
    # self.init_vertices, self.init_faces
    init_vertices, init_faces = cur_robot.init_vertices, cur_robot.init_faces
    
    
    
    init_vertices = init_vertices.detach().cpu().numpy()
    init_faces = init_faces.detach().cpu().numpy()
    
    
    ## initial  states ehre ##3
    # mesh_obj = trimesh.Trimesh(vertices=init_vertices, faces=init_faces)
    # mesh_obj.export(f"hand_urdf.ply")
    
    ##### Test the set initial state function #####
    init_joint_states = torch.zeros((60, ), dtype=torch.float32).cuda()
    cur_robot.set_initial_state(init_joint_states)
    ##### Test the set initial state function #####
    
    
    
    
    cur_zeros_actions = torch.zeros((60, ), dtype=torch.float32).cuda()
    cur_ones_actions = torch.ones((60, ), dtype=torch.float32).cuda() #  * 100
    
    ts_to_mesh_verts = {}
    for i_ts in range(50):
        cur_robot.calculate_inertia()
    
        cur_robot.set_actions_and_update_states(cur_ones_actions, i_ts, 0.2) ###
    
    
        cur_verts, cur_faces = cur_robot.get_init_visual_pts()
        cur_mesh = trimesh.Trimesh(vertices=cur_verts.detach().cpu().numpy(), faces=cur_faces.detach().cpu().numpy())
        
        ts_to_mesh_verts[i_ts + i_ts] = cur_verts.detach().cpu().numpy()
        # cur_mesh.export(f"stated_mano_mesh.ply")
        # cur_mesh.export(f"zero_actioned_mano_mesh.ply")
        cur_mesh.export(f"ones_actioned_mano_mesh_ts_{i_ts}.ply")
    
    np.save(f"reference_verts.npy", ts_to_mesh_verts)
    
    exit(0)
    
    xml_fn = "/home/xueyi/diffsim/DiffHand/assets/hand_sphere.xml"
    robot_agent = RobotAgent(xml_fn=xml_fn, args=None)
    init_visual_pts = robot_agent.init_visual_pts.detach().cpu().numpy()
    exit(0)
    