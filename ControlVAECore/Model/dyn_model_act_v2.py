
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

urdf_fn = ""

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
        mesh_root = "/home/xueyi/diffsim/NeuS/rsc/mano"
        if not os.path.exists(mesh_root):
            mesh_root = "/data/xueyi/diffsim/NeuS/rsc/mano"
            
        if "shadow" in urdf_fn and "left" in urdf_fn:
            mesh_root = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left"
            if not os.path.exists(mesh_root):
                mesh_root = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description_left"
        elif "shadow" in urdf_fn:
            mesh_root = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description"
            if not os.path.exists(mesh_root):
                mesh_root = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description"
        elif "redmax" in urdf_fn:
            mesh_root = "/home/xueyi/diffsim/NeuS/rsc/redmax_hand"
            if not os.path.exists(mesh_root):
                mesh_root = "/root/diffsim/quasi-dyn/rsc/redmax_hand"
            
        
        # if "shadow" in urdf_fn:
        #     mesh_root = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description"
        # if not os.path.exists(mesh_root):
        #     mesh_root = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description"
            
        self.mesh_root = mesh_root
        geometry_mesh_fn = geometry_mesh_fn.replace(".dae", ".obj")
        self.geometry_mesh_fn = os.path.join(mesh_root, geometry_mesh_fn)
        
        self.geometry_mesh_scale = geometry_mesh_scale
        # tranformed by xyz #
        self.vertices, self.faces = self.load_geoemtry_mesh()
        self.cur_expanded_visual_pts = None
        pass
    
    def load_geoemtry_mesh(self, ):
        # mesh_root = 
        # if self.geometry_mesh_fn.end
        mesh = trimesh.load_mesh(self.geometry_mesh_fn)
        vertices = mesh.vertices
        faces = mesh.faces
        
        vertices = torch.from_numpy(vertices).float().cuda()
        faces =torch.from_numpy(faces).long().cuda()
        
        vertices = vertices * self.geometry_mesh_scale.unsqueeze(0) + self.visual_xyz.unsqueeze(0)
        
        return vertices, faces
    
    # init_visual_meshes = get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes)
    def get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes, expanded_pts=False):
        # cur_vertices = torch.matmul(parent_rot, self.vertices.transpose(1, 0)).contiguous().transpose(1, 0).contiguous() + parent_trans.unsqueeze(0)
        
        if not expanded_pts: ### not using the expanded pts 
            cur_vertices = self.vertices
            # print(f"adding mesh loaded from {self.geometry_mesh_fn}")
            init_visual_meshes['vertices'].append(cur_vertices) # cur vertices # trans # 
            init_visual_meshes['faces'].append(self.faces)
        else:
            cur_vertices = self.cur_expanded_visual_pts
            init_visual_meshes['vertices'].append(cur_vertices) 
            init_visual_meshes['faces'].append(self.faces)
        return init_visual_meshes

    def expand_visual_pts(self, ):
        # expand_factor = 0.2
        # nn_expand_pts = 20
        
        expand_factor = 0.4
        nn_expand_pts = 40 ### number of the expanded points ### ## points ##
        
        expand_factor = 0.2
        nn_expand_pts = 20 ##
        expand_save_fn = f"{self.mesh_nm}_expanded_pts_factor_{expand_factor}_nnexp_{nn_expand_pts}_new.npy"
        expand_save_fn = os.path.join(self.mesh_root, expand_save_fn) # 
        
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


## link urdf ## expand the visual pts to form the expanded visual grids pts #
# use get_name_to_visual_pts_faces to get the transformed visual pts and faces #
class Link_urdf:
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
        
        
    
    def get_init_visual_meshes(self, parent_rot, parent_trans, init_visual_meshes, link_name_to_link_struct, link_name_to_visited, expanded_pts=False):
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
                cur_child_visual_pts = cur_child.get_init_visual_meshes(parent_rot, parent_trans + joint_origin_xyz, cur_child_visual_pts, link_name_to_link_struct, link_name_to_visited, expanded_pts=expanded_pts)
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
                    
                    # joint link idxes #
                    
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
            init_visual_meshes = self.visual.get_init_visual_meshes(parent_rot, parent_trans, init_visual_meshes, expanded_pts=expanded_pts)
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

    def set_delta_state_and_update_v2(self, states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, link_name_to_link_struct):
        link_name_to_visited[self.name] = 1
        
        if self.joint is not None:
            for cur_joint_name in self.joint:
                cur_joint = self.joint[cur_joint_name] # joint model 
                cur_child = self.children[cur_joint_name] # child model #
                cur_child_struct = link_name_to_link_struct[cur_child]
                cur_child_name = cur_child_struct.name
                if cur_child_name in link_name_to_visited:
                    continue
                # cur_child_inertia = cur_child_struct.cur_inertia
                if cur_joint.type in ['revolute']:
                    cur_joint_idx = action_joint_name_to_joint_idx[cur_joint_name]
                    cur_state = states[cur_joint_idx]
                    ### get the child struct ###
                    # set_actions_and_update_states(self, action, cur_timestep, time_cons, cur_inertia):
                    # set actions and update states #
                    cur_joint.set_delta_state_and_update_v2(cur_state, cur_timestep)
                cur_child_struct.set_delta_state_and_update_v2(states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, link_name_to_link_struct)
        
        
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
    
    def __init__(self, name, joint_type, parent_link, child_link, origin_xyz, axis_xyz, limit: Joint_Limit, origin_xyz_string="") -> None:
        self.name = name
        self.type = joint_type
        self.parent_link = parent_link
        self.child_link = child_link
        self.origin_xyz = origin_xyz
        self.axis_xyz = axis_xyz
        self.limit = limit
        
        self.origin_xyz_string = origin_xyz_string
        
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
            # prev_state = self.timestep_to_states[cur_timestep - 1].detach()
            prev_state = self.timestep_to_states[cur_timestep - 1] # .detach() # not detach? #
        cur_state = prev_state + update_quaternion(delta_rot_vec, prev_state)
        self.timestep_to_states[cur_timestep] = cur_state.detach()
        self.state = cur_state
    
    
    def set_delta_state_and_update_v2(self, delta_state, cur_timestep):
        self.timestep_to_vels[cur_timestep] = torch.zeros((3,), dtype=torch.float32).cuda().detach() 
        
        if cur_timestep == 0:
            cur_state = delta_state
        else:
            # prev_state = self.timestep_to_states[cur_timestep - 1].detach()
            # prev_state = self.timestep_to_states[cur_timestep - 1]
            cur_state = self.timestep_to_states[cur_timestep - 1].detach() + delta_state
        ## cur_state ## # 
        self.timestep_to_states[cur_timestep] = cur_state  # .detach()
        
        
        # delta_rot_vec = self.axis_xyz * state #
        
        cur_rot_vec = self.axis_xyz * cur_state ### cur_state #### # 
        # angle to the quaternion ? # 
        init_state = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
        cur_quat_state = init_state + update_quaternion(cur_rot_vec, init_state)
        self.state = cur_quat_state
        
        # if cur_timestep == 0:
        #     prev_state = torch.tensor([1., 0., 0., 0.], dtype=torch.float32).cuda()
        # else:
        #     # prev_state = self.timestep_to_states[cur_timestep - 1].detach()
        #     prev_state = self.timestep_to_states[cur_timestep - 1] # .detach() # not detach? #
        # cur_state = prev_state + update_quaternion(delta_rot_vec, prev_state)
        # self.timestep_to_states[cur_timestep] = cur_state.detach()
        # self.state = cur_state
    
    
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
            # TODO: check whether the following is correct # # set 
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
            
            # delta_omega = torque / 400 # # axis_xyz #
            
            
            # TODO: dt should be an optimizable constant? should it be the same value as that optimized for the passive object? # 
            delta_angular_vel = delta_omega * time_cons #  * self.args.dt
            delta_angular_vel = delta_angular_vel.squeeze(0)
            if cur_timestep > 0: ## cur_timestep - 1 ##
                prev_angular_vel = self.timestep_to_vels[cur_timestep - 1].detach()
                # cur_angular_vel = prev_angular_vel + delta_angular_vel * DAMPING
                cur_angular_vel = prev_angular_vel * DAMPING + delta_angular_vel
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



class Robot_urdf:
    def __init__(self, links, link_name_to_link_idxes, link_name_to_link_struct, joint_name_to_joint_idx, actions_joint_name_to_joint_idx, tot_joints=None) -> None:
        self.links = links
        self.link_name_to_link_idxes = link_name_to_link_idxes
        self.link_name_to_link_struct = link_name_to_link_struct
        
        # joint_name_to_joint_idx, actions_joint_name_to_joint_idx
        self.joint_name_to_joint_idx = joint_name_to_joint_idx
        self.actions_joint_name_to_joint_idx = actions_joint_name_to_joint_idx
        
        self.tot_joints = tot_joints
        # 
        
        
        self.init_vertices, self.init_faces = self.get_init_visual_pts()
        
        joint_name_to_joint_idx_sv_fn = "mano_joint_name_to_joint_idx.npy"
        np.save(joint_name_to_joint_idx_sv_fn, self.joint_name_to_joint_idx)
        
        actions_joint_name_to_joint_idx_sv_fn = "mano_actions_joint_name_to_joint_idx.npy"
        np.save(actions_joint_name_to_joint_idx_sv_fn, self.actions_joint_name_to_joint_idx)
        
        tot_joints = len(self.joint_name_to_joint_idx)
        tot_actions_joints = len(self.actions_joint_name_to_joint_idx)
        
        print(f"tot_joints: {tot_joints}, tot_actions_joints: {tot_actions_joints}")
        
        pass
    
    # robot.expande
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
    
    
    ### samping issue? --- TODO`    `
    def get_init_visual_pts(self, expanded_pts=False):
        init_visual_meshes = {
            'vertices': [], 'faces': [], 'link_idxes': [], 'transformed_joint_pos': [], 'link_idxes': [],  'transformed_joint_pos': [], 'joint_link_idxes': []
        }
        init_parent_rot = torch.eye(3, dtype=torch.float32).cuda()
        init_parent_trans = torch.zeros((3,), dtype=torch.float32).cuda()
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        ### from the palm linke ##
        init_visual_meshes = palm_link.get_init_visual_meshes(init_parent_rot, init_parent_trans, init_visual_meshes, self.link_name_to_link_struct, link_name_to_visited, expanded_pts=expanded_pts)
        
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


    def set_delta_state_and_update_v2(self, states, cur_timestep):
        link_name_to_visited = {}
        
        action_joint_name_to_joint_idx = self.actions_joint_name_to_joint_idx
        
        palm_idx = self.link_name_to_link_idxes["palm"]
        palm_link = self.links[palm_idx]
        
        link_name_to_visited = {}
        
        palm_link.set_delta_state_and_update_v2(states, cur_timestep, link_name_to_visited, action_joint_name_to_joint_idx, self.link_name_to_link_struct)
    
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
            
    ### 
    
    


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

    if origin is not None:
        inertial_pos = origin.attrib["xyz"]
        try:
            inertial_rpy = origin.attrib["rpy"]
        except:
            inertial_rpy = "0.0 0.0 0.0"
    else:
        inertial_pos = "0.0 0.0 0.0"
        inertial_rpy = "0.0 0.0 0.0"
    inertial_pos = parse_nparray_from_string(inertial_pos)

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
        if geometry_mesh is None:
            visual = None
        else:
            mesh_fn = geometry_mesh.attrib["filename"]

            try:
                mesh_scale = geometry_mesh.attrib["scale"]
            except:
                mesh_scale = "1 1 1"
            
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
        origin_xyz_string = joint_origin.attrib["xyz"]
        origin_xyz = parse_nparray_from_string(origin_xyz_string)
    except:
        origin_xyz = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()
        origin_xyz_string = ""
        
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
        if "velocity" in joint_limit.attrib:
            joint_velocity = joint_limit.attrib["velocity"]
            joint_velocity = float(joint_velocity)
        else:
            joint_velocity = 0.5
    else:
        joint_lower = -0.5000
        joint_upper = 1.57
        joint_effort = 1000
        joint_velocity = 0.5
    
    # cosntruct the joint data #      
    joint_limit = Joint_Limit(effort=joint_effort, lower=joint_lower, upper=joint_upper, velocity=joint_velocity)
    cur_joint_struct = Joint_urdf(joint_name, joint_type, parent_name, child_name, origin_xyz, joint_axis, joint_limit, origin_xyz_string)
    return cur_joint_struct
    


def parse_data_from_urdf(xml_fn):
    
    tree = ElementTree()
    tree.parse(xml_fn)
    print(f"{xml_fn}")
    ### get total robots ###
    # robots = tree.findall("link")
    cur_robot = tree
    # i_robot = 0
    # tot_robots = []
    # for cur_robot in robots:
    # print(f"Getting robot: {i_robot}")
    # i_robot += 1
    # print(f"len(robots): {len(robots)}")
    # cur_robot = robots[0]
    cur_links = cur_robot.findall("./link")
    # curlinks
    # i_link = 0
    link_name_to_link_idxes = {}
    cur_robot_links = []
    link_name_to_link_struct = {}
    for i_link_idx, cur_link in enumerate(cur_links):
        cur_link_struct = parse_link_data_urdf(cur_link)
        print(f"Adding link {cur_link_struct.name}, link_idx: {i_link_idx}")
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
        
        print(f"cur_joint_name: {cur_joint_name}, cur_joint_idx: {cur_joint_idx}, axis: {cur_joint_struct.axis_xyz}, origin: {cur_joint_struct.origin_xyz}")
        
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
        
        
    cur_robot_obj = Robot_urdf(cur_robot_links, link_name_to_link_idxes, link_name_to_link_struct, joint_name_to_joint_idx, actions_joint_name_to_joint_idx, tot_robot_joints)
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


### get init s
class RobotAgent: # robot and the robot #
    def __init__(self, xml_fn, args=None) -> None:
        global urdf_fn
        urdf_fn = xml_fn
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
        
        
        self.set_init_states()
        init_visual_pts = self.get_init_state_visual_pts()
        self.init_visual_pts = init_visual_pts
        
        cur_verts, cur_faces = self.active_robot.get_init_visual_pts()
        self.robot_pts = cur_verts
        self.robot_faces = cur_faces
        
        
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
        # init_states = {}
        # init_states['glb_rot'] = glb_rot;
        # init_states['glb_trans'] = glb_trans; 
        # self.active_robot.set_init_states(init_states)
        
        init_joint_states = torch.zeros((60, ), dtype=torch.float32).cuda()
        self.active_robot.set_initial_state(init_joint_states)
    
    def get_init_state_visual_pts(self, expanded_pts=False):
        # visual_pts_list = [] # compute the transformation via current state #
        # visual_pts_list, visual_pts_mass_list = self.active_robot.compute_transformation_via_current_state( visual_pts_list)
        cur_verts, cur_faces = self.active_robot.get_init_visual_pts(expanded_pts=expanded_pts)
        self.faces = cur_faces
        # self.robot_pts = cur_verts
        # self.robot_faces = cur_faces
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




# tot_obj_rot_quat, object_transl , lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_ref_fn)
def load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_ref_fn):
    # train_dyn_mano_model_states ## rhand 
    from manopth.manolayer import ManoLayer
    # sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
    # /data1/xueyi/GRAB_extracted_test/train/20_sv_dict_real_obj.obj # data1
    # import utils.utils as utils
    # from manopth.manolayer import ManoLayer
    
    start_idx = 20 # start_idx 
    window_size = 150
    # start_idx = self.start_idx
    # window_size = self.window_size
    
    # if 'model.kinematic_mano_gt_sv_fn' in self.conf:
    # sv_fn = self.conf['model.kinematic_mano_gt_sv_fn']
    sv_fn = arctic_gt_ref_fn
    # 
    # gt_data_folder = "/".join(sv_fn.split("/")[:-1]) ## 
    # gt_data_fn_name = sv_fn.split("/")[-1].split(".")[0]
    # arctic_processed_data_sv_folder = "/root/diffsim/quasi-dyn/raw_data/arctic_processed_canon_obj"
    # gt_data_canon_obj_sv_fn = f"{arctic_processed_data_sv_folder}/{gt_data_fn_name}_canon_obj.obj"
        
    print(f"Loading data from {sv_fn}")
    
    sv_dict = np.load(sv_fn, allow_pickle=True).item()
    
    
    tot_frames_nn  = sv_dict["obj_rot"].shape[0] ## obj rot ##
    window_size = min(tot_frames_nn - start_idx, window_size)
    window_size = window_size
    
    
    object_global_orient = sv_dict["obj_rot"][start_idx: start_idx + window_size] # num_frames x 3 
    object_transl = sv_dict["obj_trans"][start_idx: start_idx + window_size] * 0.001 # num_frames x 3
    obj_pcs = sv_dict["verts.object"][start_idx: start_idx + window_size]
    
    # obj_pcs = sv_dict['object_pc']
    obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    
    obj_vertex_normals = torch.zeros_like(obj_pcs[0])
    obj_normals = obj_vertex_normals
    
    # /data/xueyi/sim/arctic_processed_data/processed_seqs/s01/espressomachine_use_01.npy
    
    # obj_vertex_normals = sv_dict['obj_vertex_normals']
    # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
    # self.obj_normals = obj_vertex_normals
    
    # object_global_orient = sv_dict['object_global_orient'] # glboal orient 
    # object_transl = sv_dict['object_transl']
    
    
    obj_faces = sv_dict['f'][0]
    obj_faces = torch.from_numpy(obj_faces).long().cuda()
    obj_faces = obj_faces
    
    # obj_verts = sv_dict['verts.object']
    # minn_verts = np.min(obj_verts, axis=0)
    # maxx_verts = np.max(obj_verts, axis=0)
    # extent = maxx_verts - minn_verts
    # # center_ori = (maxx_verts + minn_verts) / 2
    # # scale_ori = np.sqrt(np.sum(extent ** 2))
    # obj_verts = torch.from_numpy(obj_verts).float().cuda()
    
    
    # obj_sv_path = "/data3/datasets/xueyi/arctic/arctic_data/data/meta/object_vtemplates"
    # obj_name = sv_fn.split("/")[-1].split("_")[0]
    # obj_mesh_fn = os.path.join(obj_sv_path, obj_name, "mesh.obj")
    # print(f"loading from {obj_mesh_fn}")
    # # template_obj_vs, template_obj_fs = trimesh.load(obj_mesh_fn, force='mesh')
    # template_obj_vs, template_obj_fs = utils.read_obj_file_ours(obj_mesh_fn, sub_one=True)
    
    
    
    
    # self.obj_verts = obj_verts
    init_obj_verts = obj_pcs[0]
    init_obj_rot_vec = object_global_orient[0]
    init_obj_transl = object_transl[0]
    
    init_obj_transl = torch.from_numpy(init_obj_transl).float().cuda()
    init_rot_struct = R.from_rotvec(init_obj_rot_vec)
    
    init_glb_rot_mtx = init_rot_struct.as_matrix()
    init_glb_rot_mtx = torch.from_numpy(init_glb_rot_mtx).float().cuda()
    # ## reverse the global rotation matrix ##
    init_glb_rot_mtx_reversed = init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
    # nn_obj_verts x  3 ##
    ##### 
    # canon_obj_verts = torch.matmul(
    #     (init_obj_verts - init_obj_transl.unsqueeze(0)), init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
    # )
    
    ''' canonical object verts '''
    # rot.T 
    canon_obj_verts = torch.matmul(
        init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
    ).contiguous().transpose(1, 0).contiguous()


    obj_verts = canon_obj_verts.clone()  
    # self.obj_verts = obj_verts.clone()  
    
    mesh_scale = 0.8
    bbmin, _ = obj_verts.min(0) #
    bbmax, _ = obj_verts.max(0) #
    
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
    # vertices = (vertices - center) * scale # (vertices - center) * scale # #
    
    
    tot_reversed_obj_rot_mtx = []
    tot_obj_quat = [] ## rotation matrix 
    
    re_transformed_obj_verts = []
    
    tot_obj_rot_quat = []
    
    # transformed_obj_verts = []
    for i_fr in range(object_global_orient.shape[0]):
        cur_glb_rot = object_global_orient[i_fr]
        cur_transl = object_transl[i_fr]
        cur_transl = torch.from_numpy(cur_transl).float().cuda()
        cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
        cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
        
        reversed_cur_glb_rot_mtx = cur_glb_rot_mtx.T
        reversed_cur_glb_rot_struct = R.from_matrix(reversed_cur_glb_rot_mtx)
        reversed_cur_glb_rot_quat = reversed_cur_glb_rot_struct.as_quat()
        reversed_cur_glb_rot_quat = reversed_cur_glb_rot_quat[[3, 0, 1, 2]]
        tot_obj_rot_quat.append(reversed_cur_glb_rot_quat)
        
        
        cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
        
        # transformed verts ## canon_verts x R + t = transformed_verts #
        # (transformed_verts - t) x R^T = canon_verts #
        # cur_transformed_verts = torch.matmul(
        #     self.obj_verts, cur_glb_rot_mtx
        # ) + cur_transl.unsqueeze(0)
        
        cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
        
        cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
        cur_obj_quat = cur_glb_rot_struct.as_quat()
        cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
        cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
        tot_obj_quat.append(cur_obj_quat)
        
        cur_re_transformed_obj_verts = torch.matmul(
            cur_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), obj_verts.transpose(1, 0)
        ).transpose(1, 0) + cur_transl.unsqueeze(0)
        re_transformed_obj_verts.append(cur_re_transformed_obj_verts)


    
    transformed_obj_verts = obj_pcs.clone()
    
    
    tot_obj_rot_quat = np.stack(tot_obj_rot_quat, axis=0)
    
    # tot_obj_rot_quat, object_transl #
    # tot_obj_rot_quat = torch.from_numpy(tot_obj_rot_quat).float().cuda()   ### get the rot quat ---- the gtobj rotations daata
    # object_transl = torch.from_numpy(object_transl).float().cuda() ## get obj transl ### ## obj transl ##
    
    # mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
   
    
    if '30_sv_dict' in sv_fn:
        bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
        obj_selected_verts = obj_verts[bbox_selected_verts_idxes]
    else:
        obj_selected_verts = obj_verts.clone()
    
    # maxx_init_passive_mesh, _ = torch.max(obj_selected_verts, dim=0)
    # minn_init_passive_mesh, _ = torch.min(obj_selected_verts, dim=0)
    # # self.maxx_init_passive_mesh = maxx_init_passive_mesh
    # self.minn_init_passive_mesh = minn_init_passive_mesh
    
    init_obj_verts = obj_verts
    
    mesh_scale = 0.8
    bbmin, _ = init_obj_verts.min(0) #
    bbmax, _ = init_obj_verts.max(0) #
    print(f"bbmin: {bbmin}, bbmax: {bbmax}")
    # center = (bbmin + bbmax) * 0.5
    
    
    mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
    if not os.path.exists(mano_path):
        mano_path = '/data/xueyi/mano_v1_2/models'
    rgt_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='right',
        mano_root=mano_path,
        ncomps=45,
        use_pca=False,
    ).cuda()
    
    lft_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='left',
        mano_root=mano_path,
        ncomps=45,
        use_pca=False,
    ).cuda()
    
    
    ##### rhand parameters #####
    rhand_global_orient_gt, rhand_pose_gt = sv_dict["rot_r"], sv_dict["pose_r"]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
    rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + window_size]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
    rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + window_size]
    
    rhand_global_orient_gt = rhand_global_orient_gt.reshape(window_size, -1).astype(np.float32)
    rhand_pose_gt = rhand_pose_gt.reshape(window_size, -1).astype(np.float32)
    
    rhand_transl, rhand_betas = sv_dict["trans_r"], sv_dict["shape_r"][0]
    rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + window_size], rhand_betas
    
    # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
    rhand_transl = rhand_transl.reshape(window_size, -1).astype(np.float32)
    rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
    
    rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float().cuda()
    rhand_pose_var = torch.from_numpy(rhand_pose_gt).float().cuda()
    rhand_beta_var = torch.from_numpy(rhand_betas).float().cuda()
    rhand_transl_var = torch.from_numpy(rhand_transl).float().cuda()
    # R.from_rotvec(obj_rot).as_matrix()
    ##### rhand parameters #####
    
    
    ##### lhand parameters #####
    lhand_global_orient_gt, lhand_pose_gt = sv_dict["rot_l"], sv_dict["pose_l"]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
    lhand_global_orient_gt = lhand_global_orient_gt[start_idx: start_idx + window_size]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
    lhand_pose_gt = lhand_pose_gt[start_idx: start_idx + window_size]
    
    lhand_global_orient_gt = lhand_global_orient_gt.reshape(window_size, -1).astype(np.float32)
    lhand_pose_gt = lhand_pose_gt.reshape(window_size, -1).astype(np.float32)
    
    lhand_transl, lhand_betas = sv_dict["trans_l"], sv_dict["shape_l"][0]
    lhand_transl, lhand_betas = lhand_transl[start_idx: start_idx + window_size], lhand_betas
    
    # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
    lhand_transl = lhand_transl.reshape(window_size, -1).astype(np.float32)
    lhand_betas = lhand_betas.reshape(-1).astype(np.float32)
    
    lhand_global_orient_var = torch.from_numpy(lhand_global_orient_gt).float().cuda()
    lhand_pose_var = torch.from_numpy(lhand_pose_gt).float().cuda()
    lhand_beta_var = torch.from_numpy(lhand_betas).float().cuda()
    lhand_transl_var = torch.from_numpy(lhand_transl).float().cuda() # self.window_size x 3
    # R.from_rotvec(obj_rot).as_matrix()
    ##### lhand parameters #####
    

    
    rhand_verts, rhand_joints = rgt_mano_layer(
        torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
        rhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), rhand_transl_var
    )
    ### rhand_joints: for joints ###
    rhand_verts = rhand_verts * 0.001
    rhand_joints = rhand_joints * 0.001
    
    
    lhand_verts, lhand_joints = lft_mano_layer(
        torch.cat([lhand_global_orient_var, lhand_pose_var], dim=-1),
        lhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), lhand_transl_var
    )
    ### rhand_joints: for joints ###
    lhand_verts = lhand_verts * 0.001
    lhand_joints = lhand_joints * 0.001
    
    
    ### lhand and the rhand ###
    # rhand_verts, lhand_verts #
    # self.rhand_verts = rhand_verts
    # self.lhand_verts = lhand_verts 
    
    hand_faces = rgt_mano_layer.th_faces
    
    
    #### tot_obj_rot_quat, object_transl, lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic
    return tot_obj_rot_quat, object_transl , lhand_verts.detach().cpu().numpy(), rhand_verts.detach().cpu().numpy()
    # return transformed_obj_verts, self.obj_normals





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


def calibreate_urdf_files(urdf_fn):
    # active_robot =  parse_data_from_urdf(xml_fn)
    active_robot =  parse_data_from_urdf(urdf_fn)
    tot_joints = active_robot.tot_joints
    
    # class Joint_urdf: # 
    # def __init__(self, name, joint_type, parent_link, child_link, origin_xyz, axis_xyz, limit: Joint_Limit) -> None:
    #     self.name = name
    #     self.type = joint_type
    #     self.parent_link = parent_link
    #     self.child_link = child_link
    #     self.origin_xyz = origin_xyz
    #     self.axis_xyz = axis_xyz
    #     self.limit = limit
    
    with open(urdf_fn) as rf:
        urdf_string = rf.read()
    for cur_joint in tot_joints:
        print(f"type: {cur_joint.type}, origin: {cur_joint.origin_xyz}")
        cur_joint_origin = cur_joint.origin_xyz
        scaled_joint_origin = cur_joint_origin * 3.
        cur_joint_origin_string = cur_joint.origin_xyz_string
        if len(cur_joint_origin_string) == 0 or torch.sum(cur_joint_origin).item() == 0.:
            continue
            # <origin xyz="0.0 0.0 0.0"/>
        cur_joint_origin_string_wtag = "<origin xyz=" + "\"" + cur_joint_origin_string + "\"" + "/>"
        scaled_joint_origin_string_wtag = "<origin xyz=" + "\"" + f"{scaled_joint_origin[0].item()} {scaled_joint_origin[1].item()} {scaled_joint_origin[2].item()}" + "\"" + "/>"
        # scaled_joint_origin_string = f"{scaled_joint_origin[0].item()} {scaled_joint_origin[1].item()} {scaled_joint_origin[2].item()}"
        # urdf_string = urdf_string.replace(cur_joint_origin_string, scaled_joint_origin_string)
        urdf_string = urdf_string.replace(cur_joint_origin_string_wtag, scaled_joint_origin_string_wtag)
    changed_urdf_fn = urdf_fn.replace(".urdf", "_scaled.urdf")
    with open(changed_urdf_fn, "w") as wf:
        wf.write(urdf_string)
    print(f"changed_urdf_fn: {changed_urdf_fn}")
    # exit(0)
    
    
def get_GT_states_data_from_ckpt(ckpt_fn):
    mano_nn_substeps = 1
    num_steps = 60
    mano_robot_actions = nn.Embedding(
        num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
    )
    torch.nn.init.zeros_(mano_robot_actions.weight)
    # params_to_train += list(self.robot_actions.parameters())
    
    mano_robot_delta_states = nn.Embedding(
        num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
    )
    torch.nn.init.zeros_(mano_robot_delta_states.weight)
    # params_to_train += list(self.robot_delta_states.parameters())
    
    mano_robot_init_states = nn.Embedding(
        num_embeddings=1, embedding_dim=60,
    )
    torch.nn.init.zeros_(mano_robot_init_states.weight)
    # params_to_train += list(self.robot_init_states.parameters())
    
    mano_robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps * mano_nn_substeps, embedding_dim=4
    )
    mano_robot_glb_rotation.weight.data[:, 0] = 1.
    mano_robot_glb_rotation.weight.data[:, 1:] = 0.
    # params_to_train += list(self.robot_glb_rotation.parameters())
    
    
    mano_robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps * mano_nn_substeps, embedding_dim=3
    )
    torch.nn.init.zeros_(mano_robot_glb_trans.weight)
    # params_to_train += list(self.robot_glb_trans.parameters())   
    
    mano_robot_states = nn.Embedding(
        num_embeddings=num_steps * mano_nn_substeps, embedding_dim=60,
    )
    torch.nn.init.zeros_(mano_robot_states.weight)
    mano_robot_states.weight.data[0, :] = mano_robot_init_states.weight.data[0, :].clone()
    
    
    ''' Load optimized MANO hand actions and states '''
    # ### laod optimized init actions #### # 
    # if 'model.load_optimized_init_actions' in self.conf and len(self.conf['model.load_optimized_init_actions']) > 0: 
    #     print(f"[MANO] Loading optimized init transformations from {self.conf['model.load_optimized_init_actions']}")
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    if 'mano_robot_states' in optimized_init_actions_ckpt:
        mano_robot_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_states'])
    
    if 'mano_robot_init_states' in optimized_init_actions_ckpt:
        mano_robot_init_states.load_state_dict(optimized_init_actions_ckpt['mano_robot_init_states'])
        
    if 'mano_robot_glb_rotation' in optimized_init_actions_ckpt:
        mano_robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_rotation'])
    
    if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
        mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    mano_glb_trans_np_data = mano_robot_glb_trans.weight.data.detach().cpu().numpy()
    mano_glb_rotation_np_data = mano_robot_glb_rotation.weight.data.detach().cpu().numpy()
    mano_states_np_data = mano_robot_states.weight.data.detach().cpu().numpy()
    
    if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
        object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
        object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': mano_glb_rotation_np_data,
        'mano_glb_trans': mano_glb_trans_np_data,
        'mano_states': mano_states_np_data,
        'obj_rot': object_global_orient,
        'obj_trans': object_transl
    }
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_20_cube_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_25_ball_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_54_cylinder_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_1_dingshuji_data.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####




def scale_and_save_meshes(meshes_folder):
    minn_robo_pts = -0.1
    maxx_robo_pts = 0.2
    extent_robo_pts = maxx_robo_pts - minn_robo_pts
    mult_const_after_cent = 0.437551664260203 ## should modify
    
    mult_const_after_cent = mult_const_after_cent / 3. * 0.9507
    
    meshes_fn = os.listdir(meshes_folder)
    meshes_fn = [fn for fn in meshes_fn if fn.endswith(".obj") and "scaled" not in fn]
    for cur_fn in meshes_fn:
        cur_mesh_name = cur_fn.split(".")[0]
        print(f"cur_mesh_name: {cur_mesh_name}")
        scaled_mesh_name = cur_mesh_name + "_scaled_bullet.obj"
        full_mesh_fn = os.path.join(meshes_folder, cur_fn)
        scaled_mesh_fn = os.path.join(meshes_folder, scaled_mesh_name)
        try:
            cur_mesh = trimesh.load_mesh(full_mesh_fn)
        except:
            continue
        cur_mesh.vertices = cur_mesh.vertices
        
        if 'palm' in cur_mesh_name:
            cur_mesh.vertices = (cur_mesh.vertices - minn_robo_pts) / extent_robo_pts
            cur_mesh.vertices = cur_mesh.vertices * 2. -1.
            cur_mesh.vertices = cur_mesh.vertices * mult_const_after_cent # mult_const #
        else:
            cur_mesh.vertices = (cur_mesh.vertices) / extent_robo_pts
            cur_mesh.vertices = cur_mesh.vertices * 2. # -1.
            cur_mesh.vertices = cur_mesh.vertices * mult_const_after_cent # mult_const #
        
        cur_mesh.export(scaled_mesh_fn)
        print(f"scaled_mesh_fn: {scaled_mesh_fn}")
    exit(0)



def calibreate_urdf_files_v2(urdf_fn):
    # active_robot =  parse_data_from_urdf(xml_fn)
    active_robot =  parse_data_from_urdf(urdf_fn)
    tot_joints = active_robot.tot_joints
    tot_links = active_robot.link_name_to_link_struct

    minn_robo_pts = -0.1
    maxx_robo_pts = 0.2
    extent_robo_pts = maxx_robo_pts - minn_robo_pts
    mult_const_after_cent = 0.437551664260203 ## should modify
    
    mult_const_after_cent = mult_const_after_cent / 3. * 0.9507
    
    with open(urdf_fn) as rf:
        urdf_string = rf.read()
    for cur_joint in tot_joints:
        # print(f"type: {cur_joint.type}, origin: {cur_joint.origin_xyz}")
        cur_joint_origin = cur_joint.origin_xyz
        
        # cur_joint_origin = (cur_joint_origin / extent_robo_pts) * 2.0 * mult_const_after_cent
        
        # cur_joint_origin = (cur_joint_origin / extent_robo_pts) * 2.0 * mult_const_after_cent
        
        if cur_joint.name in ['FFJ4' , 'MFJ4' ,'RFJ4' ,'LFJ5' ,'THJ5']:
            cur_joint_origin = (cur_joint_origin - minn_robo_pts) / extent_robo_pts
            cur_joint_origin = cur_joint_origin * 2.0 - 1.0
            cur_joint_origin = cur_joint_origin * mult_const_after_cent
        else:
            cur_joint_origin = (cur_joint_origin) / extent_robo_pts
            cur_joint_origin = cur_joint_origin * 2.0 # - 1.0
            cur_joint_origin = cur_joint_origin * mult_const_after_cent
            
        
        origin_list = cur_joint_origin.detach().cpu().tolist()
        origin_list = [str(cur_val) for cur_val in origin_list]
        origin_str = " ".join(origin_list)
        print(f"name: {cur_joint.name}, cur_joint_origin: {origin_str}")
        
        # scaled_joint_origin = cur_joint_origin * 3.
        # cur_joint_origin_string = cur_joint.origin_xyz_string
        # if len(cur_joint_origin_string) == 0 or torch.sum(cur_joint_origin).item() == 0.:
        #     continue
        #     # <origin xyz="0.0 0.0 0.0"/>
        # cur_joint_origin_string_wtag = "<origin xyz=" + "\"" + cur_joint_origin_string + "\"" + "/>"
        # scaled_joint_origin_string_wtag = "<origin xyz=" + "\"" + f"{scaled_joint_origin[0].item()} {scaled_joint_origin[1].item()} {scaled_joint_origin[2].item()}" + "\"" + "/>"
        # # scaled_joint_origin_string = f"{scaled_joint_origin[0].item()} {scaled_joint_origin[1].item()} {scaled_joint_origin[2].item()}"
        # # urdf_string = urdf_string.replace(cur_joint_origin_string, scaled_joint_origin_string)
        # urdf_string = urdf_string.replace(cur_joint_origin_string_wtag, scaled_joint_origin_string_wtag)
    # changed_urdf_fn = urdf_fn.replace(".urdf", "_scaled.urdf")
    # with open(changed_urdf_fn, "w") as wf:
    #     wf.write(urdf_string)
    # print(f"changed_urdf_fn: {changed_urdf_fn}")
    # # exit(0)
    
    # for cur_link_nm in tot_links:
    #     cur_link = tot_links[cur_link_nm]
    #     if cur_link.visual is None:
    #         continue
    #     xyz_visual = cur_link.visual.visual_xyz
    #     xyz_visual = (xyz_visual / extent_robo_pts) * 2.0 * mult_const_after_cent
    #     xyz_visual_list = xyz_visual.detach().cpu().tolist()
    #     xyz_visual_list = [str(cur_val) for cur_val in xyz_visual_list]
    #     xyz_visual_str = " ".join(xyz_visual_list)
    #     print(f"name: {cur_link.name}, xyz_visual: {xyz_visual_str}")


 
def get_shadow_GT_states_data_from_ckpt(ckpt_fn, obj_name, obj_idx):
    mano_nn_substeps = 1
    num_steps = 60
    
    
    # robot actions # # 
    # robot_actions = nn.Embedding(
    #     num_embeddings=num_steps, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_actions.weight)
    # # params_to_train += list(robot_actions.parameters())
    
    # robot_delta_states = nn.Embedding(
    #     num_embeddings=num_steps, embedding_dim=60,
    # ).cuda()
    # torch.nn.init.zeros_(robot_delta_states.weight)
    # # params_to_train += list(robot_delta_states.parameters())
    
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
        
    ''' Load optimized MANO hand actions and states '''
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy()
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy()
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy()
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_states_sv_from_act' in optimized_init_actions_ckpt:
        print(f"Using robot_states_sv_from_act!!")
        robot_states_np_data = optimized_init_actions_ckpt['robot_states_sv_from_act'].detach().cpu().numpy()
    
    if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
        object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
        object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    # print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'obj_rot': object_global_orient,
        'obj_trans': object_transl
    }
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_20_cube_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_25_ball_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/grab_train_split_54_cylinder_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_224_tiantianquan_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_54_cylinder_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_54_cylinder_data_v2.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_89_flashlight_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_7_handle_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_47_scissor_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_67_banana_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_76_quadrangular_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_110_phone_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_167_hammer167_wact_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_thres0d0_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_wctftohand_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_data_v2.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v3.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_102_mouse_data_woptrobo.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_107_stapler_data_fingerretar.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_data_v4.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_277_watch_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_296_bottle_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_267_dount267_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_358_cup_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data_wact.npy"
    sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_{obj_idx}_{obj_name}_data.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####



def load_active_passive_timestep_to_mesh_v3_taco(pkl_fn , start_idx=None):
    # train_dyn_mano_model_states ## rhand 
    # sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
    # # /data1/xueyi/GRAB_extracted_test/train/20_sv_dict_real_obj.obj # data1
    import pickle as pkl
    start_idx = 40 if start_idx is None else start_idx
    maxx_ws = 150
    # maxx_ws = 90
  
    
    sv_dict = pkl.load(open(pkl_fn, "rb"))
    
    # self.hand_faces = torch.from_numpy(sv_dict['hand_faces']).float().cuda()
    
    print(f"sv_dict: {sv_dict.keys()}")
    
    maxx_ws = min(maxx_ws, len(sv_dict['obj_verts']) - start_idx)
    
    obj_pcs = sv_dict['obj_verts'][start_idx: start_idx + maxx_ws]
    obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    # self.obj_pcs = obj_pcs
    # obj_vertex_normals = sv_dict['obj_vertex_normals']
    # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
    # self.obj_normals = torch.zeros_like(obj_pcs[0]) ### get the obj naormal vectors ##
    
    object_pose = sv_dict['obj_pose'][start_idx: start_idx + maxx_ws]
    object_pose = torch.from_numpy(object_pose).float().cuda() ### nn_frames x 4 x 4 ###
    object_global_orient_mtx = object_pose[:, :3, :3 ] ## nn_frames x 3 x 3 ##
    object_transl = object_pose[:, :3, 3] ## nn_frmaes x 3 ##
    
    
    tot_obj_quat = []
    
    for i_fr in range(object_global_orient_mtx.shape[0]):
        cur_ornt_mtx = object_global_orient_mtx[i_fr]
        cur_ornt_mtx_np = cur_ornt_mtx.detach().cpu().numpy() ### cur ornt mtx ## 
        cur_ornt_rot_struct = R.from_matrix(cur_ornt_mtx_np)
        cur_ornt_quat = cur_ornt_rot_struct.as_quat()
        cur_ornt_quat = cur_ornt_quat[[3, 0, 1, 2]]
        tot_obj_quat.append(torch.from_numpy(cur_ornt_quat).float().cuda()) ### float cuda ##
    
    # tot_obj_quat = np.stack(tot_obj_quat, axis=0) ## obj quat ##
    tot_obj_quat = torch.stack(tot_obj_quat, dim=0)
        
    return   tot_obj_quat.detach().cpu().numpy(), object_transl.detach().cpu().numpy()
    
    
def get_shadow_GT_states_data_from_ckpt_taco(ckpt_fn, obj_name, obj_idx, obj_states_fn=None, tag=None, start_idx=None):
    mano_nn_substeps = 1
    num_steps = 60
    
    
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        # robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        # print(optimized_init_actions_ckpt['robot_glb_rotation'].keys())
        num_steps = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0)
    
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
        
    ''' Load optimized MANO hand actions and states '''
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy()
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy()
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy()
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_states_sv_from_act' in optimized_init_actions_ckpt:
        print(f"Using robot_states_sv_from_act!!")
        robot_states_np_data = optimized_init_actions_ckpt['robot_states_sv_from_act'].detach().cpu().numpy()
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    

    
    if obj_states_fn is not None:
        optimized_info = np.load(obj_states_fn, allow_pickle=True).item()
        ts_to_obj_quaternion = optimized_info['ts_to_obj_quaternion']
        ts_to_obj_trans = optimized_info['ts_to_obj_trans']
        tot_optimized_quat = []
        tot_optimized_trans = []
        for ts in range(len(ts_to_obj_quaternion)):
            cur_ts_obj_quat = ts_to_obj_quaternion[ts]
            cur_ts_obj_trans = ts_to_obj_trans[ts]
            tot_optimized_quat.append(cur_ts_obj_quat)
            tot_optimized_trans.append(cur_ts_obj_trans)
        tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
        tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
        object_global_orient = tot_optimized_quat
        object_transl = tot_optimized_trans
        use_opt_data = True
    else:
        data_root = "/data3/datasets/xueyi/taco"
        if not os.path.exists(data_root):
            data_root = "/data/xueyi/taco"
        obj_idx_sub = obj_idx.split("_")[0]
        # pkl_fn = f"/data/xueyi/taco/processed_data/{obj_idx_sub}/right_{obj_idx}.pkl"
        pkl_fn = os.path.join(data_root, "processed_data", f"{obj_idx_sub}", f"right_{obj_idx}.pkl")
        object_global_orient, object_transl =  load_active_passive_timestep_to_mesh_v3_taco(pkl_fn, start_idx=start_idx)
        use_opt_data = False
    
    
    
    # print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'obj_rot': object_global_orient,
        'obj_trans': object_transl
    }

    # ref_sv_root = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData"
    # if not os.path.exists(ref_sv_root):
    #     ref_sv_root = "/root/diffsim/control-vae-2/Data/ReferenceData"
    
    ref_sv_root = "/home/xueyi/diffsim/Control-VAE/ReferenceData"
    if not os.path.exists(ref_sv_root):
        ref_sv_root = "/root/diffsim/control-vae-2/ReferenceData"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_267_dount267_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_358_cup_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data_wact.npy"
    # sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_{obj_idx}_{obj_name}_data.npy"
    # sv_gt_refereces_fn = f"/root/diffsim/control-vae-2/Data/ReferenceData/shadow_taco_train_split_{obj_idx}_{obj_name}_data.npy"
    if not use_opt_data:
        sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data.npy"
    else:
        if tag is not None and len(tag) > 0:
            sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data_opt_tag_{tag}.npy"
        else:
            sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data_opt.npy"
    sv_gt_refereces_fn = os.path.join(ref_sv_root, sv_gt_refereces_fn)
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####



def get_shadow_GT_states_data_from_optinfo_grab(opt_data_info, obj_name, obj_idx, obj_states_fn=None, tag=None, start_idx=None):
    mano_nn_substeps = 1
    num_steps = 60
    
    
    # sv_gt_refereces = {
    #     'mano_glb_rot': robot_glb_rotation_np_data,
    #     'mano_glb_trans': robot_glb_trans_np_data,
    #     'mano_states': robot_states_np_data,
    #     'obj_rot': object_global_orient,
    #     'obj_trans': object_transl
    # }
    
    opt_data_dict = np.load(opt_data_info, allow_pickle=True).item() ### opt data dict ###
    opt_data_states = opt_data_dict['state']
    
    
    tot_mano_glb_rot = [] ### qaut in w-xyz form ###
    tot_mano_glb_trans = []
    tot_mano_states = []
    tot_obj_rot = [] ### quat in w-xyz form ###
    tot_obj_trans = []
    
    for i_fr in range(len(opt_data_states)):
        cur_state = opt_data_states[i_fr]
        opt_hand_state = cur_state[:-7]
        opt_obj_state = cur_state[-7:]
        opt_hand_trans, opt_hand_rot, opt_hand_finger_states = opt_hand_state[:3], opt_hand_state[3:6], opt_hand_state[6:]
        opt_obj_trans, opt_obj_rot = opt_obj_state[:3], opt_obj_state[3:]
        
        opt_hand_rot_euler_zyx = np.array([opt_hand_rot[2], opt_hand_rot[1], opt_hand_rot[0]], dtype=np.float32) ## optimized hand rotation zyx
        opt_hand_rot_struct = R.from_euler( 'zyx', opt_hand_rot_euler_zyx, degrees=False)
        opt_hand_rot_quat = opt_hand_rot_struct.as_quat()
        opt_hand_rot_quat_wxyz = opt_hand_rot_quat[[3, 0, 1, 2]]
        opt_obj_rot_wxyz = opt_obj_rot[[3, 0, 1, 2]]
        
        tot_mano_glb_rot.append(opt_hand_rot_quat_wxyz.astype(np.float32))
        tot_mano_glb_trans.append(opt_hand_trans.astype(np.float32))
        
        opt_hand_finger_states = np.concatenate(
            [ np.zeros((2,), dtype=np.float32), opt_hand_finger_states ], axis=0
        )
        
        
        tot_mano_states.append(opt_hand_finger_states)
        tot_obj_rot.append(opt_obj_rot_wxyz.astype(np.float32))
        tot_obj_trans.append(opt_obj_trans.astype(np.float32))
        
    tot_mano_glb_rot = np.stack(tot_mano_glb_rot, axis=0)
    tot_mano_glb_trans = np.stack(tot_mano_glb_trans, axis=0)
    tot_mano_states = np.stack(tot_mano_states, axis=0)
    tot_obj_rot = np.stack(tot_obj_rot, axis=0)
    tot_obj_trans = np.stack(tot_obj_trans, axis=0)
    
    
    sv_gt_refereces = {
        'mano_glb_rot': tot_mano_glb_rot,
        'mano_glb_trans': tot_mano_glb_trans,
        'mano_states': tot_mano_states,
        'obj_rot': tot_obj_rot,
        'obj_trans': tot_obj_trans
    }
    
    ref_sv_root = "/home/xueyi/diffsim/Control-VAE/ReferenceData"
    if not os.path.exists(ref_sv_root):
        ref_sv_root = "/root/diffsim/control-vae-2/ReferenceData"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_267_dount267_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_358_cup_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data_wact.npy"
    # sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_{obj_idx}_{obj_name}_data.npy"
    # # sv_gt_refereces_fn = f"/root/diffsim/control-vae-2/Data/ReferenceData/shadow_taco_train_split_{obj_idx}_{obj_name}_data.npy"
    # if not use_opt_data:
    #     sv_gt_refereces_fn = f"shadow_grab_train_split_{obj_idx}_{obj_name}_data.npy"
    # else:
    if tag is not None and len(tag) > 0:
        sv_gt_refereces_fn = f"shadow_grab_train_split_{obj_idx}_{obj_name}_data_opt_tag_{tag}.npy"
    else:
        sv_gt_refereces_fn = f"shadow_grab_train_split_{obj_idx}_{obj_name}_data_opt.npy"
    sv_gt_refereces_fn = os.path.join(ref_sv_root, sv_gt_refereces_fn)
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
        
    return
    
    
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        # robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        # print(optimized_init_actions_ckpt['robot_glb_rotation'].keys())
        num_steps = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0)
    
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
        
    ''' Load optimized MANO hand actions and states '''
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy()
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy()
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy()
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_states_sv_from_act' in optimized_init_actions_ckpt:
        print(f"Using robot_states_sv_from_act!!")
        robot_states_np_data = optimized_init_actions_ckpt['robot_states_sv_from_act'].detach().cpu().numpy()
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    

    
    if obj_states_fn is not None:
        # 
        optimized_info = np.load(obj_states_fn, allow_pickle=True).item()
        ts_to_obj_quaternion = optimized_info['ts_to_obj_quaternion']
        ts_to_obj_trans = optimized_info['ts_to_obj_trans']
        tot_optimized_quat = []
        tot_optimized_trans = []
        for ts in range(len(ts_to_obj_quaternion)):
            cur_ts_obj_quat = ts_to_obj_quaternion[ts]
            cur_ts_obj_trans = ts_to_obj_trans[ts]
            tot_optimized_quat.append(cur_ts_obj_quat)
            tot_optimized_trans.append(cur_ts_obj_trans)
        tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
        tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
        object_global_orient = tot_optimized_quat
        object_transl = tot_optimized_trans
        use_opt_data = True
    else:
        data_root = "/data3/datasets/xueyi/taco"
        if not os.path.exists(data_root):
            data_root = "/data/xueyi/taco"
        obj_idx_sub = obj_idx.split("_")[0]
        # pkl_fn = f"/data/xueyi/taco/processed_data/{obj_idx_sub}/right_{obj_idx}.pkl"
        pkl_fn = os.path.join(data_root, "processed_data", f"{obj_idx_sub}", f"right_{obj_idx}.pkl")
        object_global_orient, object_transl =  load_active_passive_timestep_to_mesh_v3_taco(pkl_fn, start_idx=start_idx)
        use_opt_data = False
    
    
    
    # print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'obj_rot': object_global_orient,
        'obj_trans': object_transl
    }

    # ref_sv_root = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData"
    # if not os.path.exists(ref_sv_root):
    #     ref_sv_root = "/root/diffsim/control-vae-2/Data/ReferenceData"
    
    ref_sv_root = "/home/xueyi/diffsim/Control-VAE/ReferenceData"
    if not os.path.exists(ref_sv_root):
        ref_sv_root = "/root/diffsim/control-vae-2/ReferenceData"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_267_dount267_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_358_cup_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_313_mouse313_data_wact.npy"
    # sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_{obj_idx}_{obj_name}_data.npy"
    # sv_gt_refereces_fn = f"/root/diffsim/control-vae-2/Data/ReferenceData/shadow_taco_train_split_{obj_idx}_{obj_name}_data.npy"
    if not use_opt_data:
        sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data.npy"
    else:
        if tag is not None and len(tag) > 0:
            sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data_opt_tag_{tag}.npy"
        else:
            sv_gt_refereces_fn = f"shadow_taco_train_split_{obj_idx}_{obj_name}_data_opt.npy"
    sv_gt_refereces_fn = os.path.join(ref_sv_root, sv_gt_refereces_fn)
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####



def get_shadow_GT_states_data_from_ckpt_arctic_twohands(ckpt_fn, st_idx=0, obj_name="phone"):
    mano_nn_substeps = 1
    num_steps = 60
    
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        # robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        # print(optimized_init_actions_ckpt['robot_glb_rotation'].keys())
        num_steps = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0)
    
    print(f"num_steps: {num_steps}")
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
    
    
    robot_states_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states_lft.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation_lft.weight.data[:, 0] = 1.
    robot_glb_rotation_lft.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans_lft.weight)
    
    
        
    ''' Load optimized MANO hand actions and states '''
    # cur_optimized_init_actions_fn = ckpt_fn
    # optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    # print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
    ## rotation, states, trans left ##
    if 'robot_glb_rotation_lft' in optimized_init_actions_ckpt:
        robot_glb_rotation_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation_lft'])
    
    if 'robot_states_lft' in optimized_init_actions_ckpt:
        robot_states_lft.load_state_dict(optimized_init_actions_ckpt['robot_states_lft'])
        
    if 'robot_glb_trans_lft' in optimized_init_actions_ckpt:
        robot_glb_trans_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans_lft']) ## 
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    robot_glb_trans_lft_np_data = robot_glb_trans_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_lft_np_data = robot_glb_rotation_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_lft_np_data = robot_states_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    
    # tot_obj_rot_quat, object_transl, lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic
    
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    # obj_optimized_quat = optimized_init_actions_ckpt['optimized_quat'][st_idx: ]
    # obj_optimized_trans = optimized_init_actions_ckpt['optimized_trans'][st_idx: ]
    
    # print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'left_mano_glb_rot': robot_glb_rotation_lft_np_data,
        'left_mano_glb_trans': robot_glb_trans_lft_np_data,
        'left_mano_states': robot_states_lft_np_data,
        # 'optimized_quat': obj_optimized_quat,
        # 'optimized_trans': obj_optimized_trans ## optimized trans and rot ##
    } ## sv gt refeces e
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_110_phone_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_167_hammer167_wact_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_thres0d0_data.npy"
    # sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_arctic_epressomachine_data.npy"
    sv_gt_refereces_fn = f"/root/diffsim/control-vae-2/Data/ReferenceData/shadow_arctic_{obj_name}_data_st_{st_idx}.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}') ## sv gt refereces fn ##
    #### TODO: add an arg to control where to save the gt-reference-data ####




def get_shadow_GT_states_data_from_ckpt_arctic_twohands_woptimizedobj(ckpt_fn, obj_states_fn, st_idx=0, obj_name="phone"):
    mano_nn_substeps = 1
    num_steps = 60
    
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        # robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        # print(optimized_init_actions_ckpt['robot_glb_rotation'].keys())
        num_steps = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0)
    
    print(f"num_steps: {num_steps}")
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
    
    
    robot_states_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states_lft.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation_lft.weight.data[:, 0] = 1.
    robot_glb_rotation_lft.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans_lft.weight)
    
    
        
    ''' Load optimized MANO hand actions and states '''
    # cur_optimized_init_actions_fn = ckpt_fn
    # optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    # print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
    ## rotation, states, trans left ##
    if 'robot_glb_rotation_lft' in optimized_init_actions_ckpt:
        robot_glb_rotation_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation_lft'])
    
    if 'robot_states_lft' in optimized_init_actions_ckpt:
        robot_states_lft.load_state_dict(optimized_init_actions_ckpt['robot_states_lft'])
        
    if 'robot_glb_trans_lft' in optimized_init_actions_ckpt:
        robot_glb_trans_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans_lft']) ## 
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    robot_glb_trans_lft_np_data = robot_glb_trans_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_lft_np_data = robot_glb_rotation_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_lft_np_data = robot_states_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    # obj_optimized_quat = optimized_init_actions_ckpt['optimized_quat'][st_idx: ]
    # obj_optimized_trans = optimized_init_actions_ckpt['optimized_trans'][st_idx: ]
    
    # print(mano_robot_states.weight.data[1])
    
    # optimized obj states ##
    optimized_info = np.load(obj_states_fn, allow_pickle=True).item()
    ts_to_obj_quaternion = optimized_info['ts_to_obj_quaternion']
    ts_to_obj_trans = optimized_info['ts_to_obj_trans']
    tot_optimized_quat = []
    tot_optimized_trans = []
    for ts in range(len(ts_to_obj_quaternion)):
        cur_ts_obj_quat = ts_to_obj_quaternion[ts]
        cur_ts_obj_trans = ts_to_obj_trans[ts]
        tot_optimized_quat.append(cur_ts_obj_quat)
        tot_optimized_trans.append(cur_ts_obj_trans)
    
    tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
    tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'left_mano_glb_rot': robot_glb_rotation_lft_np_data,
        'left_mano_glb_trans': robot_glb_trans_lft_np_data,
        'left_mano_states': robot_states_lft_np_data,
        'optimized_quat': tot_optimized_quat, ## 
        'optimized_trans': tot_optimized_trans ## optimized trans and rot ##
    }
    SV_ROOT = "/root/diffsim/control-vae-2"
    if not os.path.exists(SV_ROOT):
        SV_ROOT = "/home/xueyi/diffsim/Control-VAE"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_110_phone_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_167_hammer167_wact_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_thres0d0_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_arctic_epressomachine_data.npy"
    sv_gt_refereces_fn = f"/root/diffsim/control-vae-2/Data/ReferenceData/shadow_arctic_{obj_name}_data_st_{st_idx}.npy"
    sv_gt_refereces_fn = f"{SV_ROOT}/Data/ReferenceData/shadow_arctic_{obj_name}_data_st_{st_idx}_optobj.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####



def get_shadow_GT_states_data_from_ckpt_arctic_single_right_hand(ckpt_fn, st_idx=0):
    mano_nn_substeps = 1
    num_steps = 60
    
    
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
    
    
    robot_states_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states_lft.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation_lft.weight.data[:, 0] = 1.
    robot_glb_rotation_lft.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans_lft.weight)
    
    
        
    ''' Load optimized MANO hand actions and states '''
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
    ## rotation, states, trans left ##
    if 'robot_glb_rotation_lft' in optimized_init_actions_ckpt:
        robot_glb_rotation_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation_lft'])
    
    if 'robot_states_lft' in optimized_init_actions_ckpt:
        robot_states_lft.load_state_dict(optimized_init_actions_ckpt['robot_states_lft'])
        
    if 'robot_glb_trans_lft' in optimized_init_actions_ckpt:
        robot_glb_trans_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans_lft']) ## 
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy()[st_idx: ]
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy()[st_idx: ]
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy()[st_idx: ]
    
    robot_glb_trans_lft_np_data = robot_glb_trans_lft.weight.data.detach().cpu().numpy()[st_idx: ]
    robot_glb_rotation_lft_np_data = robot_glb_rotation_lft.weight.data.detach().cpu().numpy()[st_idx: ]
    robot_states_lft_np_data = robot_states_lft.weight.data.detach().cpu().numpy()[st_idx: ]
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    obj_optimized_quat = optimized_init_actions_ckpt['optimized_quat'][st_idx: ]
    obj_optimized_trans = optimized_init_actions_ckpt['optimized_trans'][st_idx: ]
    
    # print(mano_robot_states.weight.data[1])
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'left_mano_glb_rot': robot_glb_rotation_lft_np_data,
        'left_mano_glb_trans': robot_glb_trans_lft_np_data,
        'left_mano_states': robot_states_lft_np_data,
        'obj_rot': obj_optimized_quat,
        'obj_trans': obj_optimized_trans ## optimized trans and rot ##
    }
    
    
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_91_hammer_wact_thres0d0_data.npy"
    sv_gt_refereces_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_arctic_epressomachine_data.npy"
    sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_arctic_epressomachine_data_st_{st_idx}.npy"
    sv_gt_refereces_fn = f"/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_arctic_epressomachine_data_st_{st_idx}_singlerighthand.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}')
    #### TODO: add an arg to control where to save the gt-reference-data ####



def get_optimized_dyn_info(ckpt_fn):
    optimized_info = torch.load(ckpt_fn, map_location='cpu', )
    print(f"optimized_info: {optimized_info.keys()}")
    state_dict_model = optimized_info['dyn_model']
    print(f"state_dict_model: {state_dict_model.keys()}")
    obj_mass = state_dict_model['optimizable_obj_mass.weight']
    obj_inertia = state_dict_model['obj_inertia.weight']
    print(obj_mass)
    print(obj_inertia)
    # 12.6832 # 
    
    
def create_grab_obj_urdf(obj_name, obj_idx, split='train'):
    
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="grab_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="/data1/xueyi/GRAB_extracted_test/{split}/{obj_idx}_obj.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="/data1/xueyi/GRAB_extracted_test/{split}/{obj_idx}_obj.obj" scale="1.0 1.0 1.0"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    urdf_sv_fn = f"/home/xueyi/diffsim/NeuS/rsc/mano/grab_{obj_name}_wcollision.urdf"
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")
    

def create_grab_obj_urdf_v2(obj_name, obj_idx, split='train'):
    
    obj_ori_fn = f"/data1/xueyi/GRAB_extracted_test/{split}/{obj_idx}_obj.obj"
    
    ctl_root_folder = f"/home/xueyi/diffsim/Control-VAE"
    ctl_asset_folder = os.path.join(ctl_root_folder, "assets")
    obj_dst_fn = os.path.join(ctl_asset_folder, "grab")
    os.makedirs(obj_dst_fn, exist_ok=True)
    
    obj_dst_fn = os.path.join(obj_dst_fn, f"{obj_idx}_obj.obj")
    
    os.system(f"cp {obj_ori_fn} {obj_dst_fn}") ### get obj file ###
    
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="grab_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="grab/{obj_idx}_obj.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="grab/{obj_idx}_obj.obj" scale="1.0 1.0 1.0"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    urdf_sv_fn = os.path.join(ctl_asset_folder, f"grab_{obj_name}_wcollision.urdf")
    # urdf_sv_fn = f"/home/xueyi/diffsim/NeuS/rsc/mano/grab_{obj_name}_wcollision.urdf"
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")


def create_taco_obj_urdf(obj_name, obj_idx):
    
    obj_main_idx = obj_idx.split('_')[0]
    
    data_root = "/data/xueyi/taco/processed_data"
    if not os.path.exists(data_root):
        data_root = "/data2/datasets/xueyi/taco/processed_data"
    
    obj_fn = os.path.join(data_root, f"{obj_main_idx}", f"right_{obj_idx}.obj")
    asset_obj_folder = f"./assets/taco/"
    os.system(f"cp {obj_fn} {asset_obj_folder}")
    
    data_asset_root = "taco"
    
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="taco_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="{data_asset_root}/right_{obj_idx}.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="{data_asset_root}/right_{obj_idx}.obj" scale="1.0 1.0 1.0"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    # urdf_sv_root = "/home/xueyi/diffsim/NeuS/rsc/mano"
    # if not os.path.exists(urdf_sv_root):
    #     urdf_sv_root = "/root/diffsim/quasi-dyn/rsc/mano"
    
    urdf_sv_root = "/home/xueyi/diffsim/Control-VAE/assets"
    if not os.path.exists(urdf_sv_root):
        urdf_sv_root = "/root/diffsim/control-vae-2/assets"
    
    # urdf_sv_fn = f"/home/xueyi/diffsim/NeuS/rsc/mano/taco_{obj_idx}_wcollision.urdf"
    urdf_sv_fn = f"taco_{obj_idx}_wcollision.urdf"
    urdf_sv_fn = os.path.join(urdf_sv_root, urdf_sv_fn)
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")
   

def create_arctic_obj_urdf(obj_name, split='train'):
    # /home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/espressomachine_grab_01_canon_obj.obj
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="grab_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="/home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/{obj_name}_grab_01_canon_obj.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="/home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/{obj_name}_grab_01_canon_obj.obj" scale="1 1 1"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    urdf_sv_fn = f"/home/xueyi/diffsim/NeuS/rsc/mano/arctic_{obj_name}_wcollision.urdf"
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")


def create_arctic_obj_urdf_v2(obj_name, split='train'):
    # /home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/espressomachine_grab_01_canon_obj.obj
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="arctic_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="arctic/{obj_name}.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="arctic/{obj_name}.obj" scale="1 1 1"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    # NEUS_ROOT = "/root/diffsim/quasi-dyn"
    # if not os.path.exists(NEUS_ROOT):
    #     NEUS_ROOT = "/home/xueyi/diffsim/NeuS"
    
    URDF_ROOT = "/home/xueyi/diffsim/Control-VAE/assets"
    if not os.path.exists(URDF_ROOT):
        URDF_ROOT = "/root/diffsim/control-vae-2/assets"
    
    urdf_sv_fn = f"{URDF_ROOT}/arctic_{obj_name}_wcollision.urdf"
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")


def test_states():
    
    gt_data_wact_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_wctftohand_data.npy"
    gt_data_wact = np.load(gt_data_wact_fn, allow_pickle=True).item()
    gt_states_wact = gt_data_wact['mano_states'][:, 2:]
    
    for i_fr in range(gt_states_wact.shape[0]):
        print(i_fr, gt_states_wact[i_fr, :].tolist())
    
    gt_data_fn = "/home/xueyi/diffsim/Control-VAE/Data/ReferenceData/shadow_grab_train_split_85_bunny_wact_data.npy"
    gt_data = np.load(gt_data_fn, allow_pickle=True).item()
    gt_states = gt_data['mano_states'][:, 2:]
    
    for i_fr in range(gt_states.shape[0]):
        print(i_fr, gt_states[i_fr, :].tolist())
    
    diff_gt_states_wact = np.sum(
        (gt_states_wact - gt_states) ** 2, axis=-1
    )
    diff_gt_states_wact = diff_gt_states_wact.mean()
    print(f"diff: {diff_gt_states_wact}")

## saved the robot file ##


def create_files(ckpt_fn, obj_idx, obj_name):
    # obj_name = 'plane'
    # obj_idx = 322
    # obj_idx = 91
    create_grab_obj_urdf(obj_name, obj_idx, split='train')
    # exit(0)
    
    get_shadow_GT_states_data_from_ckpt_arctic_twohands(ckpt_fn=ckpt_fn, st_idx=20)
    
    return


def create_files_taco(ckpt_fn, obj_idx, obj_name, obj_states_fn=None, tag=None, start_idx=None):
    # obj_name = 'plane'
    # obj_idx = 322
    # obj_idx = 91
    create_taco_obj_urdf(obj_name, obj_idx)
    # exit(0)
    
    get_shadow_GT_states_data_from_ckpt_taco(ckpt_fn=ckpt_fn, obj_name=obj_name, obj_idx=obj_idx, obj_states_fn=obj_states_fn, tag=tag, start_idx=start_idx)
    # get_shadow_GT_states_data_from_ckpt_arctic_twohands(ckpt_fn=ckpt_fn, st_idx=20)
    
    return

def create_files_arctic(ckpt_fn, obj_name):
    # obj_name = 'plane'
    # obj_idx = 322
    # obj_idx = 91
    # create_grab_obj_urdf(obj_name, obj_idx, split='train')
    # exit(0)
    
    get_shadow_GT_states_data_from_ckpt_arctic_twohands(ckpt_fn, st_idx=20, obj_name=obj_name)
    
    create_arctic_obj_urdf_v2(obj_name, split='train')
    
    return

def create_files_arctic_optobj(ckpt_fn, obj_states_fn, obj_name):
    
    create_arctic_obj_urdf_v2(obj_name, split='train')
    
    get_shadow_GT_states_data_from_ckpt_arctic_twohands_woptimizedobj(ckpt_fn, obj_states_fn, st_idx=20, obj_name=obj_name)
    


def export_canon_obj_file(kinematic_mano_gt_sv_fn, obj_name):
    sv_dict = np.load(kinematic_mano_gt_sv_fn, allow_pickle=True).item()
    
    
    # tot_frames_nn  = sv_dict["obj_rot"].shape[0] ## obj rot ##
    # window_size = min(tot_frames_nn - self.start_idx, window_size)
    # self.window_size = window_size
    
    
    object_global_orient = sv_dict["obj_rot"]  
    object_transl = sv_dict["obj_trans"] * 0.001
    obj_pcs = sv_dict["verts.object"]
    
    # obj_pcs = sv_dict['object_pc']
    obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    
    # self.obj_verts = obj_verts
    init_obj_verts = obj_pcs[0]
    init_obj_rot_vec = object_global_orient[0]
    init_obj_transl = object_transl[0]
    
    init_obj_transl = torch.from_numpy(init_obj_transl).float().cuda()
    init_rot_struct = R.from_rotvec(init_obj_rot_vec)
    
    init_glb_rot_mtx = init_rot_struct.as_matrix()
    init_glb_rot_mtx = torch.from_numpy(init_glb_rot_mtx).float().cuda()
    # ## reverse the global rotation matrix ##
    init_glb_rot_mtx_reversed = init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
    # nn_obj_verts x  3 ##
    ##### 
    # canon_obj_verts = torch.matmul(
    #     (init_obj_verts - init_obj_transl.unsqueeze(0)), init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
    # )
    
    ''' canonical object verts '''
    # rot.T 
    canon_obj_verts = torch.matmul(
        init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
    ).contiguous().transpose(1, 0).contiguous()

    # ## get canon obj verts ##
    
    canon_obj_sv_folder = "assets/arctic"
    canon_obj_mesh = trimesh.Trimesh(vertices=canon_obj_verts.detach().cpu().numpy(), faces=sv_dict['f'][0])
    
    canon_obj_mesh_sv_fn = f"{obj_name}.obj"
    canon_obj_mesh_sv_fn = os.path.join(canon_obj_sv_folder, canon_obj_mesh_sv_fn)
    canon_obj_mesh.export(canon_obj_mesh_sv_fn)
    


def create_arctic_obj_urdf_v3(obj_name, split='train'):
    # /home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/espressomachine_grab_01_canon_obj.obj
    urdf_str = f"""<?xml version="0.0" ?>
    <robot name="arctic_{obj_name}">
    
        <material name="obj_color">
            <!-- <color rgba="0.91796875 0.765 0.5234 1"/> -->
            <color rgba="0 0.7490196078431373 1.0 1"/>
        </material>
        
        <link name="link0">
            <inertial>
                <origin rpy="0.00000 -0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
                <!-- <mass value="200000000"/> -->
                <!-- <mass value="160.78"/> -->
                <mass value="10000.0"/>
                <!-- <mass value="1000.0"/> -->
                <inertia ixx="50" ixy="0" ixz="0" iyy="50" iyz="0" izz="50"/>
            </inertial>
            <visual>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="arctic/{obj_name}.obj" scale="1 1 1"/>
                </geometry>
                <material name="obj_color"/>
            </visual>
            <collision>
                <origin rpy="0 0 0.0" xyz="0 0 0"/>
                <geometry>
                    <mesh filename="arctic/{obj_name}.obj" scale="1 1 1"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """
    # NEUS_ROOT = "/root/diffsim/quasi-dyn"
    # if not os.path.exists(NEUS_ROOT):
    #     NEUS_ROOT = "/home/xueyi/diffsim/NeuS"
    
    URDF_ROOT = "/home/xueyi/diffsim/Control-VAE/assets"
    if not os.path.exists(URDF_ROOT):
        URDF_ROOT = "/root/diffsim/control-vae-2/assets"
    
    urdf_sv_fn = f"{URDF_ROOT}/arctic_{obj_name}_wcollision.urdf"
    with open(urdf_sv_fn, "w") as wf:
        wf.write(urdf_str)
        wf.write("\n")
        wf.close()
        
    print(f"{obj_name} urdf saved to {urdf_sv_fn}")



    
def get_shadow_GT_states_data_from_ckpt_arctic_twohands_v3(ckpt_fn, obj_optimized_info, st_idx=0, subj_obj_name="phone", obj_states_fn=None):
    mano_nn_substeps = 1
    num_steps = 60
    
    cur_optimized_init_actions_fn = ckpt_fn
    optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        # robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
        # print(optimized_init_actions_ckpt['robot_glb_rotation'].keys())
        num_steps = optimized_init_actions_ckpt['robot_glb_rotation']['weight'].size(0)
    
    print(f"num_steps: {num_steps}")
    
    robot_states = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation.weight.data[:, 0] = 1.
    robot_glb_rotation.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans.weight)
    
    
    robot_states_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=60,
    ).cuda()
    torch.nn.init.zeros_(robot_states_lft.weight)
    # params_to_train += list(robot_states.parameters())
    
    # robot_init_states = nn.Embedding(
    #     num_embeddings=1, embedding_dim=22,
    # ).cuda()
    # torch.nn.init.zeros_(robot_init_states.weight)
    # # params_to_train += list(robot_init_states.parameters())
    
    robot_glb_rotation_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=4
    ).cuda()
    robot_glb_rotation_lft.weight.data[:, 0] = 1.
    robot_glb_rotation_lft.weight.data[:, 1:] = 0.
    
    
    robot_glb_trans_lft = nn.Embedding(
        num_embeddings=num_steps, embedding_dim=3
    ).cuda()
    torch.nn.init.zeros_(robot_glb_trans_lft.weight)
    
    
        
    ''' Load optimized MANO hand actions and states '''
    # cur_optimized_init_actions_fn = ckpt_fn
    # optimized_init_actions_ckpt = torch.load(cur_optimized_init_actions_fn, map_location='cpu', )
    
    # print(f"optimized_init_actions_ckpt: {optimized_init_actions_ckpt.keys()}")
    
    if 'robot_glb_rotation' in optimized_init_actions_ckpt:
        robot_glb_rotation.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation'])
    
    if 'robot_states' in optimized_init_actions_ckpt:
        robot_states.load_state_dict(optimized_init_actions_ckpt['robot_states'])
        
    if 'robot_glb_trans' in optimized_init_actions_ckpt:
        robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans'])
        
    ## rotation, states, trans left ##
    if 'robot_glb_rotation_lft' in optimized_init_actions_ckpt:
        robot_glb_rotation_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_rotation_lft'])
    
    if 'robot_states_lft' in optimized_init_actions_ckpt:
        robot_states_lft.load_state_dict(optimized_init_actions_ckpt['robot_states_lft'])
        
    if 'robot_glb_trans_lft' in optimized_init_actions_ckpt:
        robot_glb_trans_lft.load_state_dict(optimized_init_actions_ckpt['robot_glb_trans_lft']) ## 
    
    # if 'mano_robot_glb_trans' in optimized_init_actions_ckpt: # mano_robot_glb_trans
    #     mano_robot_glb_trans.load_state_dict(optimized_init_actions_ckpt['mano_robot_glb_trans'])
    
    robot_glb_trans_np_data = robot_glb_trans.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_np_data = robot_glb_rotation.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_np_data = robot_states.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    robot_glb_trans_lft_np_data = robot_glb_trans_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_glb_rotation_lft_np_data = robot_glb_rotation_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    robot_states_lft_np_data = robot_states_lft.weight.data.detach().cpu().numpy() # [st_idx: ]
    
    
    # tot_obj_rot_quat, object_transl, lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic
    
    
    # if optimized_init_actions_ckpt is not None and 'object_transl' in optimized_init_actions_ckpt:
    #     object_transl = optimized_init_actions_ckpt['object_transl'].detach().cpu().numpy()
    #     object_global_orient = optimized_init_actions_ckpt['object_global_orient'].detach().cpu().numpy()
    
    # obj_optimized_quat = optimized_init_actions_ckpt['optimized_quat'][st_idx: ]
    # obj_optimized_trans = optimized_init_actions_ckpt['optimized_trans'][st_idx: ]
    
    # print(mano_robot_states.weight.data[1])
    
    if obj_states_fn is not None:
        # optimized obj states ##
        optimized_info = np.load(obj_states_fn, allow_pickle=True).item()
        ts_to_obj_quaternion = optimized_info['ts_to_obj_quaternion']
        ts_to_obj_trans = optimized_info['ts_to_obj_trans']
        tot_optimized_quat = []
        tot_optimized_trans = []
        for ts in range(len(ts_to_obj_quaternion)):
            cur_ts_obj_quat = ts_to_obj_quaternion[ts]
            cur_ts_obj_trans = ts_to_obj_trans[ts]
            tot_optimized_quat.append(cur_ts_obj_quat)
            tot_optimized_trans.append(cur_ts_obj_trans)
        
        tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
        tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
        print(f"Loading optimized info from {obj_states_fn}")
    else:
        
        tot_optimized_quat = obj_optimized_info['optimized_quat']
        tot_optimized_trans = obj_optimized_info['optimized_trans'] ## opt 
        
    
    #### TODO: add an arg to control where to save the gt-reference-data ####
    sv_gt_refereces = {
        'mano_glb_rot': robot_glb_rotation_np_data,
        'mano_glb_trans': robot_glb_trans_np_data,
        'mano_states': robot_states_np_data,
        'left_mano_glb_rot': robot_glb_rotation_lft_np_data,
        'left_mano_glb_trans': robot_glb_trans_lft_np_data,
        'left_mano_states': robot_states_lft_np_data,
        # 'optimized_quat': obj_optimized_info['optimized_quat'],
        # 'optimized_trans': obj_optimized_info['optimized_trans'] ## optimized trans and rot ##
        'optimized_quat': tot_optimized_quat, ## tot optimized quat ##
        'optimized_trans': tot_optimized_trans #
    }
    
    if obj_states_fn is None:
        sv_gt_refereces_fn = f"ReferenceData/shadow_arctic_{subj_obj_name}_data_v3.npy"
    else:
        sv_gt_refereces_fn = f"ReferenceData/shadow_arctic_{subj_obj_name}_data_v3_opt2.npy"
    
    # # sv_gt_refereces_fn = f"ReferenceData/shadow_arctic_{subj_obj_name}_data_v3.npy"
    # sv_gt_refereces_fn = f"ReferenceData/shadow_arctic_{subj_obj_name}_data_v3_opt2.npy"
    np.save(sv_gt_refereces_fn, sv_gt_refereces)
    print(f'gt reference data saved to {sv_gt_refereces_fn}') 


    
def create_files_arctic_v3(ckpt_fn, subj_obj_name, obj_states_fn=None):
    
    #### get arctic gt-reference file ####
    #### load the gt-references files into rhand_verts, lhand_verts, and object transformations ####
    subj_name, obj_name = subj_obj_name.split('_')
    arctic_gt_ref_fn = f"/data/xueyi/sim/arctic_processed_data/processed_seqs/{subj_name}/{obj_name}_grab_01.npy" ## get the arctic reference data ##
    
    arctic_gt_ref_folder = f"/data/xueyi/sim/arctic_processed_data/processed_seqs/{subj_name}"
    # tot_fns = 
    
    tot_obj_rot_quat, object_transl , lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_ref_fn)
    
    # 
    
    #### ####
    optimized_obj_info = {
        'optimized_quat': tot_obj_rot_quat, 
        'optimized_trans': object_transl
    }
    
    export_canon_obj_file(arctic_gt_ref_fn, obj_name)
    
    get_shadow_GT_states_data_from_ckpt_arctic_twohands_v3(ckpt_fn, obj_optimized_info=optimized_obj_info, st_idx=20, subj_obj_name=subj_obj_name, obj_states_fn=obj_states_fn)
    
    # create_arctic_obj_urdf_v2(obj_name, split='train')
    
    # obj_name = obj_name
    create_arctic_obj_urdf_v3(obj_name, split='train')
    
    return
        

#### Big TODO: the external contact forces from the manipulated object to the robot ####
if __name__=='__main__': # # #
    arctic_processed_folder = "/data/xueyi/arctic/arctic_processed_data"
    if not os.path.exists(arctic_processed_folder):
        arctic_processed_folder = "/data/xueyi/sim/arctic_processed_data"
    obj_name = "phone"
    obj_name = "scissors"
    obj_name = "mixer"
    obj_name = "laptop"
    kinematic_mano_gt_sv_fn = f"{arctic_processed_folder}/processed_seqs/s01/{obj_name}_grab_01.npy"
    
    
    
    # obj_name = 'mouse313'
    # obj_idx = 313
    # obj_name = 'cup'
    # obj_idx = 358
    # obj_name = 'mouse'
    # obj_idx = 102
    
    obj_name = 'bunny'
    obj_idx = 85
    
    
    # # obj_idx = 91
    # # create_grab_obj_urdf(obj_name, obj_idx, split='train')
    
    # create_grab_obj_urdf_v2(obj_name, obj_idx, split='train')
    # exit(0)
    
    opt_data_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient__step_19.npy"
    
    opt_data_info =  "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2__step_1.npy"
    
    opt_data_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_wtable_gn9d8_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_wtable_gn9d8__step_2.npy"
    
    opt_data_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_wtable_gn9d8_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_wtable_gn9d8__step_3.npy"
    
    # get_shadow_GT_states_data_from_optinfo_grab(opt_data_info, obj_name, obj_idx, obj_states_fn=None, tag=None, start_idx=None)
    # exit(0)
    
    # export_canon_obj_file(kinematic_mano_gt_sv_fn, obj_name)
    # exit(0)
    
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_phone_optmanorules_wresum_/checkpoints/ckpt_049933.pth"
    # obj_states_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_phone_optmanorules_wresum_/meshes/00072059.npy"
    
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s01_scissors_twohands_glbonly_towstates_/checkpoints/ckpt_293000.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_scissors_optmanorules_softv0_thres0d2_adp1d0_/meshes/01415649.npy"
    # obj_name = "scissors"
    
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_mixer_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/checkpoints/ckpt_256131_best.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_mixer_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/meshes/00256131.npy"
    # obj_name = "mixer"
    
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_laptop_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/checkpoints/ckpt_323479_best.pth"
    obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_laptop_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/meshes/00323479.npy"
    obj_name = "laptop"
    
    # # # obj_name = "phone"
    # create_files_arctic_optobj(ckpt_fn, obj_states_fn, obj_name)
    # exit(0)
    
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s01_phone_twohands_glbonly_towstates_/checkpoints/ckpt_188000.pth"
    # obj_name = "phone"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_mixer_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/checkpoints/ckpt_256131_best.pth"
    obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_mixer_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/meshes/00256131.npy"
    obj_name = "mixer"
    
    
    subj_obj_name = "s07_microwave"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s07_microwave_twohands_glbonly_towstates_/checkpoints/ckpt_denseretar_best.pth"
    
    subj_obj_name = "s07_espressomachine"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s07_espressomachine_twohands_glbonly_towstates_/checkpoints/ckpt_denseretar_best.pth"
    
    
    subj_obj_name = "s05_mixer"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s05_mixer_twohands_glbonly_towstates_/checkpoints/ckpt_denseretar_best.pth"
    
    
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_mixer_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_mixer_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/meshes/00019519.npy"
    
    
    # /data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_epressomachine_optoptrules_fromrobo_thres0d1_ressave_/checkpoints/ckpt_137175.pth
    # subj_obj_name = "s02_epressomachine" ## obj state fn? #
    subj_obj_name = "s02_espressomachine" ## obj state fn? #
    # espressomachine_grab_01.npy
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    
    
    
    subj_obj_name = "s02_phone" ##
    obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_phone_optmanorules_softv0_thres0d3_nres_/meshes/00569329.npy"
    ckpt_fn  = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_phone_optmanorules_softv0_thres0d3_nres_/checkpoints/ckpt_566349.pth"
    
    
    subj_obj_name = "s04_espressomachine" ##
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    
    
    subj_obj_name = "s05_capsulemachine" ##
    obj_states_fn=  "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/checkpoints/ckpt_best.pth"
    
    
    subj_obj_name = "s05_espressomachine" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    
    
    subj_obj_name = "s05_microwave" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    
    subj_obj_name = "s05_phone" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_phone_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_phone_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    
    subj_obj_name = "s06_capsulemachine" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    
    
    subj_obj_name = "s06_espressomachine" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    
    
    subj_obj_name = "s06_microwave" ##
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    
    subj_obj_name = "s06_scissors" ##
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_scissors_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_scissors_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    
    
    subj_obj_name = "s07_ketchup" ##
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s07_ketchup_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s07_ketchup_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    
    subj_obj_name = "s07_phone" ##
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s07_phone_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s07_phone_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
    
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_arctic_seq_s07_phone_twohands_glbonly_towstates_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None 
    
    
    
    # create_files_arctic_v3(ckpt_fn, subj_obj_name, obj_states_fn=obj_states_fn)
    # exit(0)
    
    
    
    # create_files_arctic(ckpt_fn, obj_name)
    # exit(0) 
    
    
    
    # test_states()
    # exit(0)
    
    # obj_idx = "20230917_030"
    # obj_name = "spoon"
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230917_030_glbonly_towstates_/checkpoints/ckpt_202000.pth"
    
    # obj_idx = "20230917_004"
    # obj_name = "spoon"
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230917_glbonly_towstatesnew_/checkpoints/ckpt_257000.pth"
    
    # obj_idx = "20230917_030"
    # obj_name = "ab"
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230917_030_glbonly_towstates_/checkpoints/ckpt_202000.pth"
    
    # obj_idx = "20230919_043"
    # obj_name = "brush"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230919_043_glbonly_towstates_real_/checkpoints/ckpt_221000.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230919_043_optmanorules_softv1_/meshes/00092529.npy"
    
    # ### 20231104_035 sponge ###
    # obj_idx = "20231104_035"
    # obj_name = "sponge"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231104_035_glbonly_towstates_/checkpoints/ckpt_136000.pth"
    # # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231104_035_optmanorules_softv0_thres0d0_/meshes/00147679.npy"
    
    
    # obj_idx = "20231104_017"
    # obj_name = "brush2"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231104_017_glbonly_towstates_real_real_/checkpoints/ckpt_125000.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231104_017_optmanorules_softv0_thres0d0_/meshes/00159705.npy"
    obj_states_fn = None
    
    # obj_idx = "20230919_043"
    # obj_name = "brush"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230919_043_glbonly_towstates_real_/checkpoints/ckpt_221000.pth"
    # obj_states_fn = None
    
    # obj_idx = "20231105_067"
    # obj_name = "spoon2"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231105_067_glbonly_real_towstates_/checkpoints/ckpt_295000.pth"
    # obj_states_fn = None
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_optrulesfrrobo_walltks_/meshes/00542360.npy"
    
    tag = None
    
    # obj_idx = "20231105_067"
    # obj_name = "spoon2"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_optrobo_/checkpoints/ckpt_2831_best.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_optrobo_/meshes/00002831.npy"
    # tag = "optrulesrobo"
    
    
    # obj_idx = "20231105_067"
    # obj_name = "spoon2"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_thres0d25_optrobo_optrulesthres0d21_/checkpoints/ckpt_149_best.pth"
    # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_thres0d25_optrobo_optrulesthres0d21_/meshes/00500789.npy"
    # tag = "optrulesrobo2"
    
    
    # obj_idx = "20230930_001"
    # obj_name = "plank1"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20230930_001_glbonly_todense_/checkpoints/ckpt_491849_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    
    
    # obj_idx = "20231031_184"
    # obj_name = "hammer20231031_184"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231031_184_thres0d3_/thres0d25_robo_thres0d2_/checkpoints/ckpt_best.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    # obj_idx = "20231031_171"
    # obj_name = "hammer20231031_171"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231031_171_thres0d3_/thres0d25_robo_thres0d2_/checkpoints/ckpt_best.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_067"
    # obj_name = "shovel20231027_067"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_067_glbonly_todense_/checkpoints/ckpt_1221110_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_066"
    # obj_name = "shovel20231027_066"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_066_glbonly_todense_/checkpoints/ckpt_1170090_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_068"
    # obj_name = "shovel20231027_068"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_068_glbonly_todense_/checkpoints/ckpt_1195539_denseretar.pth"
    
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_074"
    # obj_name = "shovel20231027_074"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_074_glbonly_todense_/checkpoints/ckpt_1283493_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_087"
    # obj_name = "shovel20231027_087"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_087_glbonly_todense_/checkpoints/ckpt_1256722_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_022"
    # obj_name = "shovel20231027_022"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_022_glbonly_todense_/checkpoints/ckpt_1339641_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_027"
    # obj_name = "20231027_027"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_027_glbonly_todense_/checkpoints/ckpt_1375110_denseretar.pth"
    # obj_states_fn = None
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_066"
    # obj_name = "20231027_066"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_066_glbonly_todense_/checkpoints/ckpt_1278090_denseretar.pth"
    # obj_states_fn = None
    
    # tag = "gt"
    # start_idx = 20
    
    
    # obj_idx = "20231027_114"
    # obj_name = "20231027_114"
    
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_114_glbonly_todense_/checkpoints/ckpt_1333627_denseretar.pth"
    # obj_states_fn = None
    
    # tag = "gt"
    # start_idx = 20
    
    
    obj_idx = "20231027_130"
    obj_name = "20231027_130"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_130_glbonly_todense_/checkpoints/ckpt_1368120_denseretar.pth"
    obj_states_fn = None
    
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231027_086"
    obj_name = "20231027_086"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_086_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231027_113"
    obj_name = "20231027_113"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_113_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231026_002"
    obj_name = "20231026_002"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_002_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    obj_idx = "20231026_005"
    obj_name = "20231026_005"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_005_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231026_006"
    obj_name = "20231026_006"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_006_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    obj_idx = "20231020_195"
    obj_name = "20231020_195"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231020_195_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231020_197"
    obj_name = "20231020_197"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231020_197_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    obj_idx = "20231020_199"
    obj_name = "20231020_199"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231020_199_glbonly_todense_/checkpoints/ckpt_2771549_denseretar.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    obj_idx = "20231020_201"
    obj_name = "20231020_201"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231020_201_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    obj_idx = "20231020_203"
    obj_name = "20231020_203"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231020_203_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    obj_states_fn = None
    tag = "gt"
    start_idx = 20
    
    
    obj_idx = "20231024_044"
    obj_name = "20231024_044"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231024_044_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    obj_idx = "20231024_043"
    obj_name = "20231024_043"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231024_043_glbonly_todense_/checkpoints/ckpt_1085841_denseretar.pth"
    
    
    obj_idx = "20231024_045"
    obj_name = "20231024_045"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231024_045_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    
    start_idx = 21
    obj_idx = "20231026_017"
    obj_name = "20231026_017"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_017_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    ### start idxx ###
    
    start_idx = 24
    obj_idx = "20231026_018"
    obj_name = "20231026_018"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_018_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    start_idx = 23
    obj_idx = "20231026_015"
    obj_name = "20231026_015"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_015_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    start_idx = 20
    obj_idx = "20231026_005"
    obj_name = "20231026_005"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_005_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    obj_idx = "20231026_006"
    obj_name = "20231026_006"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_006_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    obj_idx = "20231026_002"
    obj_name = "20231026_002"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231026_002_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    
    obj_idx = "20231027_130"
    obj_name = "20231027_130"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_130_glbonly_todense_/checkpoints/ckpt_2340120_denseretar.pth"
    
    
    obj_idx = "20231027_132"
    obj_name = "20231027_132"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_132_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    obj_idx = "20231027_086"
    obj_name = "20231027_086"
    ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_taco_20231027_086_glbonly_todense_/checkpoints/ckpt_denseretar_best.pth"
    
    # obj_idx = "20230927_037"
    # obj_name = "brush3"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_037_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = None
    # # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_optrobo_/meshes/00002831.npy"
    # tag = "gt"
    
    # obj_idx = "20230926_035"
    # obj_name = "knife1"
    # ckpt_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_035_thres0d3_multi_stages_/ckpt_20230926_035_best.pth"
    # obj_states_fn = None
    # # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_optrulesfrrobo_walltks_optrulesv2_optrobo_v22_optrobo_/meshes/00002831.npy"
    # tag = "gt"
    
    # /data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_018_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/checkpoints/ckpt_best.pth
    
    
    # obj_idx = "20230929_018"
    # obj_name = "bowl1"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_018_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/checkpoints/ckpt_best.pth"
    # # obj_states_fn = None
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_018_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00000149.npy"
    # tag = "optrobo"
    
    # obj_idx = "20230929_005"
    # obj_name = "knife2"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_005_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_005_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_/meshes/00137229.npy"
    # tag = "optrobo"
    
    
    # obj_idx = "20230929_002"
    # obj_name = "knife1"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_002_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_002_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/meshes/00007748.npy"
    # tag = "optrobo"
    
    
    # obj_idx = "20230927_037"
    # obj_name = "brush3"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_037_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth" 
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_037_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000125.npy"
    # tag = "optrobo1"
    
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_037_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_037_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00000125.npy"
    # tag = "optrobo2"
    
    
    # obj_idx = "20230927_023"
    # obj_name = "brush20230927_023"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_023_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_023_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00012069.npy"
    # tag = "optrobo"
    
    # obj_idx = "20230927_031"
    # obj_name = "20230927_031"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_031_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_031_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00001490.npy"
    # tag = "optrobo"
    
    
    # obj_idx = "20230928_036"
    # obj_name = "dish1"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_036_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_036_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    # tag = "optrobo"
    
    # obj_idx = "20230928_034"
    # obj_name = "dish2"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_034_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_034_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    # tag = "optrobo"
    
    # obj_idx = "20230926_035"
    # obj_name = "knife20230926_035"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_035_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_035_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00012069.npy"
    # tag = "optrobo"
    
    
    # obj_idx = "20230926_002"
    # obj_name = "20230926_002"
    # ckpt_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_002_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/checkpoints/ckpt_best.pth"
    # obj_states_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_002_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00003483.npy"
    # tag = "optrobo"
    
    
    
    
    create_files_taco(ckpt_fn, obj_idx, obj_name, obj_states_fn=obj_states_fn, tag=tag, start_idx=start_idx)
    exit(0)
    
    obj_idx = 398
    obj_name = "train"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_398_optmanorules_/checkpoints/ckpt_510000.pth"
    
    obj_idx = 322
    obj_name = "plane"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_322_plane_glbonly_towstates_/checkpoints/ckpt_884000.pth"
    
    
    obj_idx = 47
    obj_name = "scissors"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_47_scissors_glbonly_towstates_/checkpoints/ckpt_406000.pth"
    
    
    
    
    create_files(ckpt_fn=ckpt_fn, obj_idx=obj_idx, obj_name=obj_name)
    exit(0)
    
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_epressomachine_optoptrules_fromrobo_thres0d1_ressave_/checkpoints/ckpt_000059.pth"
    # st_idx = 20
    # get_shadow_GT_states_data_from_ckpt_arctic_single_right_hand(ckpt_fn=ckpt_fn, st_idx=st_idx)
    # exit(0)
    
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_epressomachine_optoptrules_fromrobo_thres0d1_/checkpoints/ckpt_090000.pth"
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_epressomachine_optoptrules_fromrobo_thres0d1_ressave_/checkpoints/ckpt_000059.pth"
    # st_idx = 20
    # get_shadow_GT_states_data_from_ckpt_arctic_twohands(ckpt_fn=ckpt_fn, st_idx=st_idx)
    # exit(0)
    
    
    # obj_name = 'mouse'
    # obj_idx = 102
    # obj_name = 'handle'
    # obj_idx = 7
    # obj_name = 'scissor'
    # obj_idx = 47
    # obj_name = 'banana'
    # obj_idx = 67
    # obj_name = 'quadrangular'
    # obj_idx = 76
    # obj_name = 'bunny'
    # obj_idx = 85
    # obj_name = 'hammer'
    # obj_idx = 91
    # obj_name = 'phone'
    # obj_idx = 110
    # obj_name = 'hammer167'
    # obj_idx = 167
    # obj_name = 'stapler107'
    # obj_idx = 107
    # obj_name = 'watch'
    # obj_idx = 277
    # obj_name = 'bottle'
    # obj_idx = 296
    # obj_name = 'dount267'
    # # obj_idx = 267
    # obj_name = 'mouse313'
    # obj_idx = 313
    # obj_name = 'cup'
    # obj_idx = 358
    obj_name = 'plane'
    obj_idx = 322
    # obj_idx = 91
    # create_grab_obj_urdf(obj_name, obj_idx, split='train')
    # exit(0)
    
    ## mouse -> ##
    # ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_89_optrules_frommano_thre00_/checkpoints/ckpt_240000.pth"
    
    # get_optimized_dyn_info(ckpt_fn)
    # exit(0)
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_/checkpoints/ckpt_320000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_54_optrules_/checkpoints/ckpt_030000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_224_optrules_frommano_toshadow_/checkpoints/ckpt_220000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_89_optrules_frommano_thre01_optrobo_/checkpoints/ckpt_010000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_102_optrules_frommano_thre00_robooptrules_/checkpoints/ckpt_030000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_7_handle_glbonly_towstates_real_/checkpoints/ckpt_085000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_47_optrules_frommano_thres00_robooptrules_/checkpoints/ckpt_060000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_67_banana_glbonly_towstates_v2_/checkpoints/ckpt_026000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_76_knife_glbonly_/checkpoints/ckpt_050000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_85_bunny_glbonly_towstates_/checkpoints/ckpt_036000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_91_glbonly_towstates_towstates_/checkpoints/ckpt_144000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_110_phone_glbonly_towstates_/checkpoints/ckpt_185000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_91_optoptrules_optrobo_act_/checkpoints/ckpt_020000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_91_optoptrules_optrobo_act_tst_/checkpoints/ckpt_000059.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_167_optoptrules_/checkpoints/ckpt_040000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_91_optoptrules_optrobo_act_tst_thres00_/checkpoints/ckpt_040000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_/checkpoints/ckpt_080000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_wcftorobo_/checkpoints/ckpt_160000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_wcftorobo_svstates_/checkpoints/ckpt_000059.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_wcftorobo_tstact_/checkpoints/ckpt_020000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_wcftorobo_tstact_/checkpoints/ckpt_000059.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_102_optrules_frommano_thre00_robooptrules_optrobo_/checkpoints/ckpt_070000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_107_stapler_glbonly_towstates_/checkpoints/ckpt_024000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_85_optrules_frommano_thres00_optrulesfromrobo_optroboact_onlyhandtracking_toobjtracking_/checkpoints/ckpt_030000.pth"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_277_watch_glbonly_towstates_/checkpoints/ckpt_059000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_296_bottle_glbonly_towstates_/checkpoints/ckpt_111000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_267_dount_glbonly_towstates_/checkpoints/ckpt_119000.pth"
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_267_dount_glbonly_towstates_/checkpoints/ckpt_124000.pth"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_313_optmanorules_thres00_optroboacts_/checkpoints/ckpt_010000.pth"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_358_optmanorules_thres00_optroboacts_/checkpoints/ckpt_020000.pth"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_313_optmanorules_thres00_optroboacts_/checkpoints/ckpt_080000.pth"
    
    ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_forces_rule_v18__shadow_finger_retargeting_scale_v2_winit_grab_seq_322_plane_glbonly_towstates_/checkpoints/ckpt_884000.pth"
    
    get_shadow_GT_states_data_from_ckpt(ckpt_fn)
    exit(0)
    
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled.urdf"
    calibreate_urdf_files_v2(urdf_fn)
    exit(0)
    
    meshes_folder = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/meshes"
    scale_and_save_meshes(meshes_folder)
    exit(0)
    
    sv_ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_mano_states_grab_train_54_cylinder_tst_/checkpoints/ckpt_070000.pth"
    sv_ckpt_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_mano_states_grab_train_1_dingshuji_tst_/checkpoints/ckpt_070000.pth"
    get_GT_states_data_from_ckpt(sv_ckpt_fn)
    exit(0)
    
    
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled_scaled_0_9507_nroot.urdf"
    robot_agent = RobotAgent(urdf_fn)
    exit(0)
    
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_nocoll_simplified.urdf"
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_wcollision_scaled.urdf"
    calibreate_urdf_files(urdf_fn)
    exit(0)
    
    # urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_nocoll_simplified.urdf"
    urdf_fn = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new.urdf"
    robot_agent = RobotAgent(urdf_fn)
    
    init_vertices, init_faces = robot_agent.active_robot.init_vertices, robot_agent.active_robot.init_faces
    init_vertices = init_vertices.detach().cpu().numpy()
    init_faces = init_faces.detach().cpu().numpy()
    
    shadow_hand_mesh = trimesh.Trimesh(vertices=init_vertices, faces=init_faces)
    shadow_hand_sv_fn = "/home/xueyi/diffsim/NeuS/raw_data/shadow_hand.obj"
    shadow_hand_mesh.export(shadow_hand_sv_fn)
    exit(0)
    
    
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
    