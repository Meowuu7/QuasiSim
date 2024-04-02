
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


DAMPING = 0.3

# DAMPING = 0.2

DAMPING = 0.0


def get_body_name_to_main_axis():
    # negative y; positive x #
    body_name_to_main_axis = {
        "body2": -2, "body6": 1, "body10": 1, "body14": 1, "body17": 1
    }
    return body_name_to_main_axis ## get the body name to main axis ##

## insert one 
def plane_rotation_matrix_from_angle_xz(angle):
    ## angle of 
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


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


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


## the optimization strategy: incremental optimization ##
class Joint:
    def __init__(self, name, joint_type, axis, pos, quat, frame, damping, args) -> None:
        self.name = name
        self.type = joint_type
        self.axis = axis # joint axis # 
        self.pos = pos # joint position #
        self.quat = quat
        self.frame = frame
        self.damping = damping
        
        self.args = args
        
        # self.timestep_to_actions = {} # torques #
        self.timestep_to_vels = {}
        self.timestep_to_states = {}
        
        self.init_pos = self.pos.clone()
        
        #### only for the current state ####
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
        # self.rot_mtx = np.eye(3, dtype=np.float32)
        # self.trans_vec = np.zeros((3,), dtype=np.float32)
        
        self.axis_rot_mtx = torch.tensor(
            [
                [1, 0, 0], [0, -1, 0], [0, 0, -1]
            ], dtype=torch.float32
        ).cuda()
        
        self.joint_idx = -1
        
        self.transformed_joint_pts = self.pos.clone()
    
    def print_grads(self, ):
        print(f"rot_mtx: {self.rot_mtx.grad}")
        print(f"trans_vec: {self.trans_vec.grad}")
        
    def clear_grads(self,):
        if self.rot_mtx.grad is not None:
            self.rot_mtx.grad.data = self.rot_mtx.grad.data * 0.
        if self.trans_vec.grad is not None:
            self.trans_vec.grad.data = self.trans_vec.grad.data * 0.
        
    def compute_transformation(self,):
        # use the state to transform them # # transform # ## transform the state ##
        # use the state to transform them # # transform them for the state #
        if self.type == "revolute":
            # print(f"computing transformation matrices with axis: {self.axis}, state: {self.state}")
            # rotation matrix from the axis angle # 
            rot_mtx = rotation_matrix_from_axis_angle(self.axis, self.state)
            # rot_mtx(p - p_v) + p_v -> rot_mtx p - rot_mtx p_v + p_v
            # trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
            # self.rot_mtx = np.copy(rot_mtx)
            # self.trans_vec = np.copy(trans_vec)
            trans_vec = self.pos - torch.matmul(rot_mtx, self.pos.view(3, 1)).view(3).contiguous()
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec
        else:
            ### TODO: implement transformations for joints in other types ###
            pass
    
     
    def set_state(self, name_to_state):
        if self.name in name_to_state:
            # self.state = name_to_state["name"]
            self.state = name_to_state[self.name] ## 
            
    def set_state_via_vec(self, state_vec): ### transform points via the state vectors here ###
        if self.joint_idx >= 0:
            self.state = state_vec[self.joint_idx] ## give the parameter to the parameters ##
            
    def set_joint_idx(self, joint_name_to_idx):
        if self.name in joint_name_to_idx:
            self.joint_idx = joint_name_to_idx[self.name]
            
    
    def set_args(self, args):
        self.args = args
        
        
    def compute_transformation_via_state_vals(self, state_vals):
        if self.joint_idx >= 0:
            cur_joint_state = state_vals[self.joint_idx]
        else:
            cur_joint_state = self.state
        # use the state to transform them # # transform # ## transform the state ##
        # use the state to transform them # # transform them for the state #
        if self.type == "revolute":
            # print(f"computing transformation matrices with axis: {self.axis}, state: {self.state}")
            # rotation matrix from the axis angle # 
            rot_mtx = rotation_matrix_from_axis_angle(self.axis, cur_joint_state)
            # rot_mtx(p - p_v) + p_v -> rot_mtx p - rot_mtx p_v + p_v
            # trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
            # self.rot_mtx = np.copy(rot_mtx)
            # self.trans_vec = np.copy(trans_vec)
            trans_vec = self.pos - torch.matmul(rot_mtx, self.pos.view(3, 1)).view(3).contiguous()
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec
        elif self.type == "free2d":
            cur_joint_state = state_vals # still only for the current scene #
            # cur_joint_state
            cur_joint_rot_val = state_vals[2]
            rot_mtx = plane_rotation_matrix_from_angle_xz(cur_joint_rot_val)
            # rot_mtx = plane_rotation_matrix_from_angle(cur_joint_rot_val) ### 2 x 2 rot matrix # 
            # R_axis^T ( R R_axis (p) + trans (with the y-axis padded) )
            cur_trans_vec = torch.stack(
                [state_vals[0], torch.zeros_like(state_vals[0]), state_vals[1]], dim=0
            )
            # cur_trans_vec # 
            rot_mtx = torch.matmul(self.axis_rot_mtx.transpose(1, 0), torch.matmul(rot_mtx, self.axis_rot_mtx))
            trans_vec = torch.matmul(self.axis_rot_mtx.transpose(1, 0), cur_trans_vec.unsqueeze(-1).contiguous()).squeeze(-1).contiguous() + self.pos
            
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec ## rot_mtx and trans_vec # 
        else:
            ### TODO: implement transformations for joints in other types ###
            pass
        return self.rot_mtx, self.trans_vec


    def compute_transformation_from_current_state(self):
        # if self.joint_idx >= 0:
        #     cur_joint_state = state_vals[self.joint_idx]
        # else:
        cur_joint_state = self.state
        if self.type == "revolute":
            # print(f"computing transformation matrices with axis: {self.axis}, state: {self.state}")
            # rotation matrix from the axis angle # 
            # rot_mtx = rotation_matrix_from_axis_angle(self.axis, cur_joint_state)
            # rot_mtx(p - p_v) + p_v -> rot_mtx p - rot_mtx p_v + p_v
            # trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
            # self.rot_mtx = np.copy(rot_mtx)
            # self.trans_vec = np.copy(trans_vec)
            rot_mtx = quaternion_to_matrix(self.state)
            # print(f"state: {self.state}, rot_mtx: {rot_mtx}")
            # trans_vec = self.pos - torch.matmul(rot_mtx, self.pos.view(3, 1)).view(3).contiguous()
            trans_vec = self.init_pos - torch.matmul(rot_mtx, self.init_pos.view(3, 1)).view(3).contiguous()
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec
        elif self.type == "free2d":
            state_vals = cur_joint_state
            cur_joint_state = state_vals # still only for the current scene #
            # cur_joint_state
            cur_joint_rot_val = state_vals[2]
            ### rot_mtx ### ### rot_mtx ###
            rot_mtx = plane_rotation_matrix_from_angle_xz(cur_joint_rot_val)
            # rot_mtx = plane_rotation_matrix_from_angle(cur_joint_rot_val) ### 2 x 2 rot matrix # 
            # R_axis^T ( R R_axis (p) + trans (with the y-axis padded) )
            cur_trans_vec = torch.stack(
                [state_vals[0], torch.zeros_like(state_vals[0]), state_vals[1]], dim=0
            )
            # cur_trans_vec # 
            rot_mtx = torch.matmul(self.axis_rot_mtx.transpose(1, 0), torch.matmul(rot_mtx, self.axis_rot_mtx))
            trans_vec = torch.matmul(self.axis_rot_mtx.transpose(1, 0), cur_trans_vec.unsqueeze(-1).contiguous()).squeeze(-1).contiguous() + self.pos
            
            self.rot_mtx = rot_mtx
            self.trans_vec = trans_vec
        else:
            ### TODO: implement transformations for joints in other types ###
            pass
        return self.rot_mtx, self.trans_vec



    def transform_joints_via_parent_rot_trans_infos(self, parent_rot_mtx, parent_trans_vec):
        # 
        # if self.type == "revolute" or self.type == "free2d":
        transformed_joint_pts = torch.matmul(parent_rot_mtx, self.pos.view(3 ,1).contiguous()).view(3).contiguous() + parent_trans_vec
        self.pos = torch.matmul(parent_rot_mtx, self.init_pos.view(3 ,1).contiguous()).view(3).contiguous() + parent_trans_vec
        
        return self.pos
    
    
    
    
# initialize the robot with states set to zeros #
# update the robot via states #
# set a new action #
# update states via actions #
# update robot (visual points and parameters) via states #


## transform from the root of the robot; pass qs from the root to the leaf node ##
## visual meshes  or visual meshes from the basic description of robots ##
## visual meshes; or visual points ##
## visual meshes -> transform them into the visual density values here ##
## visual meshes -> transform them into the ## into the visual counterparts ##
## ## visual meshes -> ## ## ## 
# <body name="body0" type="mesh"  filename="hand/body0.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.700000 0.700000 0.700000 1"/>
class Body:
    def __init__(self, name, body_type, filename, pos, quat, transform_type, density, mu, rgba, radius, args) -> None:
        self.name = name
        self.body_type = body_type
        ### for mesh object ###
        self.filename = filename
        self.args = args

        self.pos = pos
        self.quat = quat
        self.transform_type = transform_type
        self.density = density
        self.mu = mu
        self.rgba = rgba
        
        # 
        self.radius = radius
        
        self.visual_pts_ref = None
        self.visual_faces_ref = None
        
        self.visual_pts = None
        
        self.body_name_to_main_axis = get_body_name_to_main_axis() ### get the body name to main axis here #
        
        self.get_visual_counterparts()
        # inertial_ref, inertial_ref_inv
        
    def get_visual_faces_list(self, visual_faces_list):
        visual_faces_list.append(self.visual_faces_ref)
        return visual_faces_list
        
    def compute_inertial_inv(self, rot_mtx):
        cur_inertia_inv = torch.matmul(
            rot_mtx, torch.matmul(self.inertial_ref_inv, rot_mtx.transpose(1, 0).contiguous()) ### passive obj rot transpose 
        )
        self.cur_inertial_inv = cur_inertia_inv
        return cur_inertia_inv
        
    def compute_inertia(self, rot_mtx):
        cur_inertia = torch.matmul(
            rot_mtx, torch.matmul(self.inertial_ref, rot_mtx.transpose(1, 0))
        )
        self.cur_inertia = cur_inertia
        return cur_inertia
        
    def update_radius(self,):
        self.radius.data = self.radius.data - self.radius.grad.data
        
        self.radius.grad.data = self.radius.grad.data * 0.
        
      
    ### get visual pts colorrs ### ### 
    def get_visual_pts_colors(self, ):
        tot_visual_pts_nn = self.visual_pts_ref.size(0)
        # self.pts_rgba = [torch.from_numpy(self.rgba).float().cuda(self.args.th_cuda_idx) for _ in range(tot_visual_pts_nn)] # total visual pts nn 
        self.pts_rgba = [torch.tensor(self.rgba.data).cuda() for _ in range(tot_visual_pts_nn)] # total visual pts nn  skeletong
        self.pts_rgba = torch.stack(self.pts_rgba, dim=0) # 
        return self.pts_rgba
    ## optimize the action sequneces ##
    def get_visual_counterparts(self,):
        if self.body_type == "sphere":
            filename = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo/meshes/18.obj"
            if not os.path.exists(filename):
                filename = "/data/xueyi/diffsim/DiffHand/assets/18.obj"
            body_mesh = trimesh.load(filename, process=False) 
        elif self.body_type == "mesh":
            filename = self.filename
            if "shadow" in xml_fn:
                rt_asset_path = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description"
            else:
                rt_asset_path = "/home/xueyi/diffsim/DiffHand/assets"
                if not os.path.exists(rt_asset_path):
                    rt_asset_path = "/data/xueyi/diffsim/DiffHand/assets"
                
            filename = os.path.join(rt_asset_path, filename) # 
            body_mesh = trimesh.load(filename, process=False) 
        elif self.body_type == "abstract":
            body_mesh = trimesh.Trimesh(vertices=np.empty((0, 3), dtype=np.float32), faces=np.empty((0, 3), dtype=np.int32))
            # body_mesh = trimesh.load(filename, process=False) 

        self.pos = nn.Parameter(
            torch.tensor(self.pos.detach().cpu().tolist(), dtype=torch.float32,  requires_grad=True).cuda(), requires_grad=True
        )
        
        ### Step 1 ### -> set the pos to the correct initial pose ###
        # self.radius = nn.Parameter(
        #     torch.tensor([self.args.initial_radius], dtype=torch.float32, requires_grad=True).cuda(), requires_grad=True
        # )
        self.radius = nn.Parameter(
            torch.tensor([2.], dtype=torch.float32, requires_grad=True).cuda(), requires_grad=True
        )
        ### visual pts ref ### ## body_mesh.vertices -> # 
        self.visual_pts_ref = torch.tensor(body_mesh.vertices, dtype=torch.float32).cuda()

        self.visual_faces_ref = torch.tensor(body_mesh.faces, dtype=torch.long).cuda()
        
        minn_pts, _ = torch.min(self.visual_pts_ref, dim=0) ### get the visual pts minn ###
        maxx_pts, _ = torch.max(self.visual_pts_ref, dim=0) ### visual pts maxx ###
        mean_pts = torch.mean(self.visual_pts_ref, dim=0) ### mean_pts of the mean_pts ###
        
        if self.name in self.body_name_to_main_axis:
            cur_main_axis = self.body_name_to_main_axis[self.name] ## get the body name ##
            
            if cur_main_axis == -2:
                main_axis_pts = minn_pts[1] # the main axis pts
                full_main_axis_pts = torch.tensor([mean_pts[0], main_axis_pts, mean_pts[2]], dtype=torch.float32).cuda()
            elif cur_main_axis == 1:
                main_axis_pts = maxx_pts[0] # the maxx axis pts
                full_main_axis_pts = torch.tensor([main_axis_pts, mean_pts[1], mean_pts[2]], dtype=torch.float32).cuda()
            self.full_main_axis_pts_ref = full_main_axis_pts
        else:
            self.full_main_axis_pts_ref = mean_pts.clone()


    def transform_visual_pts_ref(self,):
        if self.name == "sphere":
            visual_pts_ref =  self.visual_pts_ref  / 2. #
            visual_pts_ref = visual_pts_ref * self.radius
        else:
            visual_pts_ref = self.visual_pts_ref
        return visual_pts_ref

    def transform_visual_pts(self, rot_mtx, trans_vec):
        visual_pts_ref = self.transform_visual_pts_ref()
        # rot_mtx: 3 x 3 numpy array 
        # trans_vec: 3 numpy array
        # print(f"transforming body with rot_mtx: {rot_mtx} and trans_vec: {trans_vec}")
        # self.visual_pts = np.matmul(rot_mtx, self.visual_pts_ref.T).T + trans_vec.reshape(1, 3) # reshape #
        # print(f"rot_mtx: {rot_mtx}, trans_vec: {trans_vec}")
        self.visual_pts = torch.matmul(rot_mtx, visual_pts_ref.transpose(1, 0)).transpose(1, 0) + trans_vec.unsqueeze(0)
        
        # full_main_axis_pts -> 
        self.full_main_axis_pts = torch.matmul(rot_mtx, self.full_main_axis_pts_ref.unsqueeze(-1)).contiguous().squeeze(-1) + trans_vec
        self.full_main_axis_pts = self.full_main_axis_pts.unsqueeze(0)
        
        return self.visual_pts
    
    def transform_expanded_visual_pts(self, rot_mtx, trans_vec):
        expanded_visual_pts_ref = self.expanded_visual_pts_ref
        self.expanded_visual_pts = torch.matmul(rot_mtx, expanded_visual_pts_ref.transpose(1, 0)).transpose(1, 0) + trans_vec.unsqueeze(0)
        
        return self.expanded_visual_pts
    
    def get_tot_transformed_joints(self, transformed_joints):
        if self.name in self.body_name_to_main_axis:
            transformed_joints.append(self.full_main_axis_pts)
        return transformed_joints
    
    def get_nn_pts(self,):
        self.nn_pts = self.visual_pts_ref.size(0)
        return self.nn_pts
    
    def set_args(self, args):
        self.args = args
        
    def clear_grad(self, ):
        if self.pos.grad is not None:
            self.pos.grad.data = self.pos.grad.data  * 0.
        if self.radius.grad is not None:
            self.radius.grad.data = self.radius.grad.data  * 0.
    
    def get_visual_pts(self, visual_pts_list):
        visual_pts_list.append(self.visual_pts.detach())
        return visual_pts_list
    
    # get the visual counterparts of the boyd mesh or elements #
    
    # xyz attribute ## ## xyz attribute #

# use get_name_to_visual_pts
# use get_name_to_visual_pts_faces to get the transformed visual pts and faces #
class Link: 
    def __init__(self, name, joint: Joint, body: Body, children, args) -> None:
        
        self.joint = joint
        self.body = body
        self.children = children
        self.name = name
        
        self.args = args
        
        ### dyn_model_act ###
        # parent_rot_mtx, parent_trans_vec #
        # parent_rot_mtx, parent_trans_vec #
        self.parent_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        self.parent_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)
        self.curr_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        self.curr_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)
        # 
        self.tot_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(), requires_grad=True)
        self.tot_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(), requires_grad=True)
        
        self.compute_inertia() 
    
    def print_grads(self, ):
        print(f"parent_rot_mtx: {self.parent_rot_mtx.grad}")
        print(f"parent_trans_vec: {self.parent_trans_vec.grad}")
        print(f"curr_rot_mtx: {self.curr_rot_mtx.grad}")
        print(f"curr_trans_vec: {self.curr_trans_vec.grad}")
        print(f"tot_rot_mtx: {self.tot_rot_mtx.grad}")
        print(f"tot_trans_vec: {self.tot_trans_vec.grad}")
        print(f"Joint")
        self.joint.print_grads()
        for cur_link in self.children:
            cur_link.print_grads()
    
    def compute_inertia(self, ):
        joint_pos = self.joint.pos
        joint_rot, joint_trans = self.joint.compute_transformation_from_current_state() # from current state # 
        body_pts = self.body.transform_visual_pts(joint_rot, joint_trans)
        self.inertial_ref = torch.zeros((3, 3), dtype=torch.float32).cuda()
        body_pts_mass = 1. / float(body_pts.size(0))
        for i_pts in range(body_pts.size(0)):
            cur_pts = body_pts[i_pts]
            cur_pts_mass = body_pts_mass
            cur_r = cur_pts - joint_pos
            # cur_vert = init_passive_mesh[i_v] # 
            # cur_r = cur_vert - init_passive_mesh_center
            dot_r_r = torch.sum(cur_r * cur_r)
            cur_eye_mtx = torch.eye(3, dtype=torch.float32).cuda()
            r_mult_rT = torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0))
            self.inertial_ref += (dot_r_r * cur_eye_mtx - r_mult_rT) * cur_pts_mass
        self.inertial_ref_inv = torch.linalg.inv(self.inertial_ref)
        self.body.inertial_ref = self.inertial_ref.clone()
        self.body.inertial_ref_inv = self.inertial_ref_inv.clone() ### inertial ref ###
        # print(f"body_invertia_matrix_inv: {self.body.inertial_ref_inv}")
    
    
    
    # 
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
        
    # with link states # # stpe 
    
    
    def get_joint_states(self, joint_states):
        if self.joint.type == 'revolute':
            # joint_states.append(self.joint.state)
            joint_idx = self.joint.joint_idx
    
    def set_penetration_forces(self, penetration_forces, sampled_visual_pts_joint_idxes, joint_penetration_forces):
        # penetration_forces
        if self.children is not None and len(self.children) > 0:
            for cur_link in self.children:
                cur_link.set_penetration_forces(penetration_forces,  sampled_visual_pts_joint_idxes, joint_penetration_forces)
        
        if self.joint.type in ['revolute']:
            # penetration_forces_values = penetration_forces['penetration_forces'] # 
            # penetration_forces_points = penetration_forces['penetration_forces_points'] # 
            
            # penetration forces #
            penetration_forces_values = penetration_forces['penetration_forces'].detach()
            penetration_forces_points = penetration_forces['penetration_forces_points'].detach()
            
            ####### use a part of peentration points and forces #######
            if sampled_visual_pts_joint_idxes is not None:
                selected_forces_mask = sampled_visual_pts_joint_idxes == self.joint.joint_idx
            else:
                selected_forces_mask = torch.ones_like(penetration_forces_values[:, 0]).bool()
            ####### use a part of peentration points and forces #######
            
            ####### use all peentration points and forces #######
            # selected_forces_mask = torch.ones_like(penetration_forces_values[:, 0]).bool()
            ####### use all peentration points and forces #######
            
            if torch.sum(selected_forces_mask.float()) > 0.5:
                
                penetration_forces_values = penetration_forces_values[selected_forces_mask]
                penetration_forces_points = penetration_forces_points[selected_forces_mask]
                # tot_rot_mtx, tot_trans_vec
                # cur_joint_rot =  self.tot_rot_mtx
                # cur_joint_trans = self.tot_trans_vec
                cur_joint_rot =  self.tot_rot_mtx.detach()
                cur_joint_trans = self.tot_trans_vec.detach()
                local_frame_penetration_forces_values = torch.matmul(cur_joint_rot.transpose(1, 0), penetration_forces_values.transpose(1, 0)).transpose(1, 0)
                local_frame_penetration_forces_points = torch.matmul(cur_joint_rot.transpose(1, 0), (penetration_forces_points - cur_joint_trans.unsqueeze(0)).transpose(1, 0)).transpose(1, 0)
                
                body_visual_pts_ref = self.body.visual_pts_ref
                center_pts = torch.mean(body_visual_pts_ref, dim=0)
                
                joint_pos_to_forces_points = local_frame_penetration_forces_points - center_pts.unsqueeze(0)
                forces_torques = torch.cross(joint_pos_to_forces_points, local_frame_penetration_forces_values) # forces values of the local frame #
                forces_torques = torch.sum(forces_torques, dim=0)
                
                forces = torch.sum(local_frame_penetration_forces_values, dim=0)
                
                cur_joint_maximal_forces = torch.cat(
                    [forces, forces_torques], dim=0
                )
                cur_joint_idx = self.joint.joint_idx
                joint_penetration_forces[cur_joint_idx][:] = cur_joint_maximal_forces[:].clone()
                
                # forces_torques_dot_axis = torch.sum(self.joint.axis * forces_torques)
                # forces_torques = self.joint.axis * forces_torques_dot_axis
                
                ####### use children penetrations torqeus #######
                # if children_penetration_torques is not None:
                #     children_penetration_torques_dot_axis = torch.sum(self.joint.axis * children_penetration_torques)
                #     children_penetration_torques = self.joint.axis * children_penetration_torques_dot_axis
                #     forces_torques = forces_torques + children_penetration_torques  # * 0.5 # damping #
                #     children_penetration_torques = forces_torques.clone() * 0.5 # damping #
                # else:
                #     children_penetration_torques = forces_torques.clone() * 0.5 #
                ####### use children penetrations torqeus #######
                    
                # force torques #
                # torque = torque + forces_torques # * 0.001
                    
            
            
    
    # forward dynamics --- from actions to states #
    # inertia matrix --- from the inertia matrix to the inertia matrix # # set actiosn and update states #
    def set_actions_and_update_states(self, actions, cur_timestep, time_cons, penetration_forces=None, sampled_visual_pts_joint_idxes=None, joint_name_to_penetration_forces_intermediates=None, children_penetration_torques=None, buffered_intertia_matrix=None):
        
        if self.children is not None and len(self.children) > 0:
            # tot_children_intertia_matrix = torch.zeros((3, 3), dtype=torch.float32).cuda()
            for cur_link in self.children:
                tot_children_intertia_matrix = torch.zeros((3, 3), dtype=torch.float32).cuda()
                cur_link.set_actions_and_update_states(actions, cur_timestep, time_cons, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes, joint_name_to_penetration_forces_intermediates=joint_name_to_penetration_forces_intermediates, children_penetration_torques=children_penetration_torques, buffered_intertia_matrix=tot_children_intertia_matrix)
                if buffered_intertia_matrix is not None:
                    buffered_intertia_matrix = buffered_intertia_matrix + tot_children_intertia_matrix
                else:
                    buffered_intertia_matrix = tot_children_intertia_matrix
        
            # tot_children_intertia_matri 
            
        # tot_children_intertia_matri = tot_children_intertia_matri + torch.eye(3, dtype=torch.float32).cuda() * 0.0001
        
        # buffered_intertia_matrix 
        
        if self.joint.type in ['revolute']:
            # return # 
            self.joint.action = actions[self.joint.joint_idx]
            # 
            # visual_pts and visual_pts_mass
            cur_joint_pos = self.joint.pos
            # TODO: check whether the following is correct #
            torque = self.joint.action * self.joint.axis ## joint.axis ##
            
            # should along the axis # 
            # torques added to the joint # 
            # penetration forces -- a list of the forces ## penetration forces ###
            if penetration_forces is not None:
                # a series of the #
                # penetration_forces: { 'global_rotation': xxx, 'global_translation': xxx,  'penetration_forces': xxx, 'penetration_forces_points': xxx }
                # glb_rot = penetration_forces['global_rotation'] # 
                # # glb_trans = penetration_forces['global_translation'] # 
                # penetration_forces_values = penetration_forces['penetration_forces'] # 
                # penetration_forces_points = penetration_forces['penetration_forces_points'] # 
                
                # penetration forces # # values points # 
                penetration_forces_values = penetration_forces['penetration_forces'].detach()
                penetration_forces_points = penetration_forces['penetration_forces_points'].detach()
                
                ####### use a part of peentration points and forces #######
                if sampled_visual_pts_joint_idxes is not None:
                    selected_forces_mask = sampled_visual_pts_joint_idxes == self.joint.joint_idx
                else:
                    selected_forces_mask = torch.ones_like(penetration_forces_values[:, 0]).bool()
                ####### use a part of peentration points and forces #######
                
                ####### use all peentration points and forces #######
                selected_forces_mask = torch.ones_like(penetration_forces_values[:, 0]).bool()
                ####### use all peentration points and forces #######
                
                if torch.sum(selected_forces_mask.float()) > 0.5:
                    
                    penetration_forces_values = penetration_forces_values[selected_forces_mask]
                    penetration_forces_points = penetration_forces_points[selected_forces_mask]
                    # tot_rot_mtx, tot_trans_vec
                    # cur_joint_rot =  self.tot_rot_mtx
                    # cur_joint_trans = self.tot_trans_vec
                    cur_joint_rot =  self.tot_rot_mtx.detach()
                    cur_joint_trans = self.tot_trans_vec.detach()
                    local_frame_penetration_forces_values = torch.matmul(cur_joint_rot.transpose(1, 0), penetration_forces_values.transpose(1, 0)).transpose(1, 0)
                    local_frame_penetration_forces_points = torch.matmul(cur_joint_rot.transpose(1, 0), (penetration_forces_points - cur_joint_trans.unsqueeze(0)).transpose(1, 0)).transpose(1, 0)
                    
                    joint_pos_to_forces_points = local_frame_penetration_forces_points - cur_joint_pos.unsqueeze(0)
                    forces_torques = torch.cross(joint_pos_to_forces_points, local_frame_penetration_forces_values) # forces values of the local frame #
                    forces_torques = torch.sum(forces_torques, dim=0)
                    
                    forces_torques_dot_axis = torch.sum(self.joint.axis * forces_torques)
                    forces_torques = self.joint.axis * forces_torques_dot_axis
                    
                    ####### use children penetrations torqeus #######
                    # if children_penetration_torques is not None:
                    #     children_penetration_torques_dot_axis = torch.sum(self.joint.axis * children_penetration_torques)
                    #     children_penetration_torques = self.joint.axis * children_penetration_torques_dot_axis
                    #     forces_torques = forces_torques + children_penetration_torques  # * 0.5 # damping #
                    #     children_penetration_torques = forces_torques.clone() * 0.5 # damping #
                    # else:
                    #     children_penetration_torques = forces_torques.clone() * 0.5 #
                    ####### use children penetrations torqeus #######
                        
                    # force torques #
                    torque = torque + forces_torques # * 0.001
                    
                    if joint_name_to_penetration_forces_intermediates is not None:
                        visual_pts = self.body.visual_pts_ref.detach().cpu().numpy()
                        forces_points_local_frame = local_frame_penetration_forces_points.detach().cpu().numpy()
                        forces_values_local_frame = local_frame_penetration_forces_values.detach().cpu().numpy()
                        joint_pos = cur_joint_pos.detach().cpu().numpy()
                        joint_axis = self.joint.axis.detach().cpu().numpy()
                        joint_name_to_penetration_forces_intermediates[self.joint.name] = {
                            'visual_pts': visual_pts, 'forces_points_local_frame': forces_points_local_frame, 'forces_values_local_frame': forces_values_local_frame, 'joint_pos': joint_pos, 'joint_axis': joint_axis
                        }
                else:
                    if children_penetration_torques is not None:
                        children_penetration_torques = children_penetration_torques * 0.5
                        
                        
                
                # # TODO: transform the forces to the joint frame #
                # cur_penetration_torque = torch.zeros_like(torque)
                # for cur_pene_force_set in penetration_forces:
                #     cur_pene_force, cur_pene_point = cur_pene_force_set
                #     joint_pos_to_pene_point = cur_pene_point - cur_joint_pos ## joint pos ##
                #     cur_point_pene_torque = torch.cross(joint_pos_to_pene_point, cur_pene_force)
                #     cur_penetration_torque += cur_point_pene_torque
                # # # ## 
                # dot_axis_with_penetration_torque = torch.sum(self.joint.axis * cur_penetration_torque)
                # cur_penetration_torque = self.joint.axis * dot_axis_with_penetration_torque
                # torque = torque + cur_penetration_torque
            
            # # Compute inertia matrix # #
            # inertial = torch.zeros((3, 3), dtype=torch.float32).cuda()
            # for i_pts in range(self.visual_pts.size(0)):
            #     cur_pts = self.visual_pts[i_pts]
            #     cur_pts_mass = self.visual_pts_mass[i_pts]
            #     cur_r = cur_pts - cur_joint_pos # r_i # 
            #     # cur_vert = init_passive_mesh[i_v]
            #     # cur_r = cur_vert - init_passive_mesh_center
            #     dot_r_r = torch.sum(cur_r * cur_r)
            #     cur_eye_mtx = torch.eye(3, dtype=torch.float32).cuda()
            #     r_mult_rT = torch.matmul(cur_r.unsqueeze(-1), cur_r.unsqueeze(0))
            #     inertial += (dot_r_r * cur_eye_mtx - r_mult_rT) * cur_pts_mass
            # m = torch.sum(self.visual_pts_mass)
            # # Use torque to update angular velocity -> state #
            # inertia_inv = torch.linalg.inv(inertial)
            
            # axis-angle of # axis-angle # # a) joint torque; # b) external  force and torque #
            # potision of the force # # link a; body a # body to the joint # # body to the joint # #
            # force applied to the joint torque # # torque #
            # change the angles #
            # inertia_inv = self.cur_inertia_inv
            inertia_inv = torch.linalg.inv(self.cur_inertia).detach()
            
            inertia_inv = torch.eye(n=3, dtype=torch.float32).cuda()
            
            
            if buffered_intertia_matrix is not None:
                buffered_intertia_matrix = buffered_intertia_matrix + torch.eye(n=3, dtype=torch.float32).cuda()
            else:
                buffered_intertia_matrix = torch.eye(n=3, dtype=torch.float32).cuda()
            
            inertia_inv = torch.linalg.inv(buffered_intertia_matrix).detach()
            

            delta_omega = torch.matmul(inertia_inv, torque.unsqueeze(-1)).squeeze(-1)
            
            # 
            # delta_omega = torque / 400 # # apply the force onto the link; apply the force onto the link 
            
            # TODO: dt should be an optimizable constant? should it be the same value as that optimized for the passive object? # 
            delta_angular_vel = delta_omega * time_cons #  * self.args.dt # delta quat # 
            delta_angular_vel = delta_angular_vel.squeeze(0)
            if cur_timestep > 0:
                prev_angular_vel = self.joint.timestep_to_vels[cur_timestep - 1].detach()
                cur_angular_vel = prev_angular_vel * DAMPING + delta_angular_vel
            else:
                cur_angular_vel = delta_angular_vel # delta 
            
            self.joint.timestep_to_vels[cur_timestep] = cur_angular_vel.detach()
            # TODO: about args.dt 
            cur_delta_quat = cur_angular_vel * time_cons
            cur_delta_quat = cur_delta_quat.squeeze(0) # delta quat #
            cur_state = self.joint.timestep_to_states[cur_timestep].detach()
            nex_state = cur_state + update_quaternion(cur_delta_quat, cur_state)
            self.joint.timestep_to_states[cur_timestep + 1] = nex_state.detach() # 
            self.joint.state = nex_state  
            # followed by updating visual pts using states # # 
            # print(f"updated_joint_state: {self.joint.state}")
        
        
    
    ## for the robot: iterate over links and get the states ##
    def get_joint_nm_to_states(self, joint_nm_to_states):
        if self.joint.type in ['revolute']:
            joint_nm_to_states[self.joint.name] = self.joint.state
        if self.children is not None and len(self.children) > 0:
            for cur_link in self.children:
                joint_nm_to_states = cur_link.get_joint_nm_to_states(joint_nm_to_states)
        return joint_nm_to_states
    
    def get_timestep_to_states(self, joint_nm_to_ts_to_states):
        if self.joint.type in ['revolute']:
            joint_nm_to_ts_to_states[self.joint.name] = self.joint.timestep_to_states
        if self.children is not None and len(self.children) > 0:
            for cur_link in self.children:
                joint_nm_to_ts_to_states = cur_link.get_timestep_to_states(joint_nm_to_ts_to_states)
        return joint_nm_to_ts_to_states
    
    # current delta states # get states -- reference states # joint 
    def set_and_update_states(self, states, cur_timestep, time_cons):
        #
        if self.joint.type in ['revolute']:
            # return # # prev 
            cur_state = states[self.joint.joint_idx] # joint idx # 
            
            # 
            # self.joint.timestep_to_states[cur_timestep + 1] = cur_state.detach()
            delta_rot_vec = self.joint.axis * cur_state # states --> 
            prev_state = self.joint.timestep_to_states[cur_timestep].detach()
            cur_state = prev_state + update_quaternion(delta_rot_vec, prev_state)
            self.joint.timestep_to_states[cur_timestep + 1] = cur_state.detach()
            self.joint.state = cur_state  
            # followed by updating visual pts using states #
            # print(f"updated_joint_state: {self.joint.state}")
        
        # link and the states # 
        if self.children is not None and len(self.children) > 0:
            for cur_link in self.children: # glb trans # 
                cur_link.set_and_update_states(states, cur_timestep, time_cons)
        
    # 
    def set_state(self, name_to_state):
        self.joint.set_state(name_to_state=name_to_state)
        for child_link in self.children:
            child_link.set_state(name_to_state)
            
            
    def set_state_via_vec(self, state_vec): # 
        self.joint.set_state_via_vec(state_vec)
        for child_link in self.children:
            child_link.set_state_via_vec(state_vec)
    
    ## 
    def get_tot_transformed_joints(self, transformed_joints):
        cur_joint_transformed_pts = self.joint.transformed_joint_pts.unsqueeze(0) ### 3 pts 
        transformed_joints.append(cur_joint_transformed_pts)
        transformed_joints = self.body.get_tot_transformed_joints(transformed_joints)
        # if self.joint.name 
        for cur_link in self.children:
            transformed_joints = cur_link.get_tot_transformed_joints(transformed_joints)
        return transformed_joints
        
    def compute_transformation_via_state_vecs(self, state_vals, parent_rot_mtx, parent_trans_vec, visual_pts_list):
        # state vecs and rot mtx # state vecs #####
        joint_rot_mtx, joint_trans_vec = self.joint.compute_transformation_via_state_vals(state_vals=state_vals)

        self.curr_rot_mtx = joint_rot_mtx
        self.curr_trans_vec = joint_trans_vec
        
        self.joint.transform_joints_via_parent_rot_trans_infos(parent_rot_mtx=parent_rot_mtx, parent_trans_vec=parent_trans_vec) ## get rot and trans mtx and vecs ### 

        # current rot #
        tot_parent_rot_mtx = torch.matmul(parent_rot_mtx, joint_rot_mtx)
        tot_parent_trans_vec = torch.matmul(parent_rot_mtx, joint_trans_vec.unsqueeze(-1)).view(3) + parent_trans_vec
        
        self.tot_rot_mtx = tot_parent_rot_mtx
        self.tot_trans_vec = tot_parent_trans_vec
        
        # self.tot_rot_mtx = np.copy(tot_parent_rot_mtx)
        # self.tot_trans_vec = np.copy(tot_parent_trans_vec)
        
        ### visual_pts_list for recording visual pts ###
        
        cur_body_visual_pts = self.body.transform_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        visual_pts_list.append(cur_body_visual_pts)
        
        for cur_link in self.children:
            # cur_link.parent_rot_mtx = np.copy(tot_parent_rot_mtx) ### set children parent rot mtx and the trans vec
            # cur_link.parent_trans_vec = np.copy(tot_parent_trans_vec) ## 
            cur_link.parent_rot_mtx = tot_parent_rot_mtx  ### set children parent rot mtx and the trans vec #
            cur_link.parent_trans_vec = tot_parent_trans_vec  ## 
            # cur_link.compute_transformation() ## compute self's transformations
            cur_link.compute_transformation_via_state_vecs(state_vals, tot_parent_rot_mtx, tot_parent_trans_vec, visual_pts_list)
        
    def compute_transformation_via_current_state(self, parent_rot_mtx, parent_trans_vec, visual_pts_list, visual_pts_mass, link_name_to_transformations_and_transformed_pts, joint_idxes=None):
        # state vecs and rot mtx # state vecs #
        joint_rot_mtx, joint_trans_vec = self.joint.compute_transformation_from_current_state()

        # cur_inertia_inv
        self.cur_inertia_inv = torch.zeros((3, 3), dtype=torch.float32).cuda()
        self.cur_inertia = torch.zeros((3, 3), dtype=torch.float32).cuda()
        self.curr_rot_mtx = joint_rot_mtx
        self.curr_trans_vec = joint_trans_vec
        
        self.joint.transform_joints_via_parent_rot_trans_infos(parent_rot_mtx=parent_rot_mtx, parent_trans_vec=parent_trans_vec) ## get rot and trans mtx and vecs ### 

        # get the parent rot mtx and the joint rot mtx # # joint rot mtx #
        tot_parent_rot_mtx = torch.matmul(parent_rot_mtx, joint_rot_mtx)
        tot_parent_trans_vec = torch.matmul(parent_rot_mtx, joint_trans_vec.unsqueeze(-1)).view(3) + parent_trans_vec
        
        # tot_rot_mtx, tot_trans_vec
        self.tot_rot_mtx = tot_parent_rot_mtx
        self.tot_trans_vec = tot_parent_trans_vec
        
        # self.tot_rot_mtx = np.copy(tot_parent_rot_mtx)
        # self.tot_trans_vec = np.copy(tot_parent_trans_vec)
        
        ### visual_pts_list for recording visual pts ### # !!!! damping is an important technique here ! ####
        ### visual pts list for recoding visual pts ###
        # so the inertial should be transformed by the tot_rot_mtx #  # transform visual pts # # tr
        cur_body_visual_pts = self.body.transform_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        # visual_pts_list.append(cur_body_visual_pts)
        self.cur_inertia_inv = self.cur_inertia_inv + self.body.compute_inertial_inv(self.tot_rot_mtx)
        self.cur_inertia = self.cur_inertia + self.body.compute_inertia(self.tot_rot_mtx)
        
        # 
        cur_body_transformations = (self.tot_rot_mtx, self.tot_trans_vec)
        link_name_to_transformations_and_transformed_pts[self.body.name] = (cur_body_visual_pts.detach().clone(), cur_body_transformations)
        
        if joint_idxes is not None:
            cur_body_joint_idx = self.joint.joint_idx
            cur_body_joint_idxes = [cur_body_joint_idx for _ in range(cur_body_visual_pts.size(0))]
            cur_body_joint_idxes = torch.tensor(cur_body_joint_idxes, dtype=torch.long).cuda()
            joint_idxes.append(cur_body_joint_idxes)
        
        # 
        
        children_visual_pts_list = []
        children_pts_mass = []
        children_pts_mass.append(torch.ones((cur_body_visual_pts.size(0), ), dtype=torch.float32).cuda() / float(cur_body_visual_pts.size(0)))
        children_visual_pts_list.append(cur_body_visual_pts)
        for cur_link in self.children:
            # cur_link.parent_rot_mtx = np.copy(tot_parent_rot_mtx) ### set children parent rot mtx and the trans vec
            # cur_link.parent_trans_vec = np.copy(tot_parent_trans_vec) ## 
            cur_link.parent_rot_mtx = tot_parent_rot_mtx  ### set children parent rot mtx and the trans vec #
            cur_link.parent_trans_vec = tot_parent_trans_vec  ## 
            # cur_link.compute_transformation() ## compute self's transformations
            children_visual_pts_list, children_pts_mass = cur_link.compute_transformation_via_current_state(tot_parent_rot_mtx, tot_parent_trans_vec, children_visual_pts_list, children_pts_mass, link_name_to_transformations_and_transformed_pts, joint_idxes=joint_idxes)
            ## inertia_inv ## 
            self.cur_inertia_inv = self.cur_inertia_inv + cur_link.cur_inertia_inv
            self.cur_inertia = self.cur_inertia + cur_link.cur_inertia ### get the current inertia ###
            
        children_visual_pts = torch.cat(children_visual_pts_list, dim=0)
        self.visual_pts = children_visual_pts.detach() # 
        visual_pts_list.append(children_visual_pts)
        children_pts_mass = torch.cat(children_pts_mass, dim=0)
        self.visual_pts_mass = children_pts_mass.detach()
        visual_pts_mass.append(children_pts_mass)
        # print(f"children_pts_mass: {children_pts_mass.size()}")
        return visual_pts_list, visual_pts_mass


    def compute_expanded_visual_pts_transformation_via_current_state(self, parent_rot_mtx, parent_trans_vec, visual_pts_list, visual_pts_mass):
        # state vecs and rot mtx # state vecs ##### # 
        joint_rot_mtx, joint_trans_vec = self.joint.compute_transformation_from_current_state()

        # cur_inertia_inv
        self.cur_inertia_inv = torch.zeros((3, 3), dtype=torch.float32).cuda()
        self.cur_inertia = torch.zeros((3, 3), dtype=torch.float32).cuda()
        self.curr_rot_mtx = joint_rot_mtx
        self.curr_trans_vec = joint_trans_vec
        
        self.joint.transform_joints_via_parent_rot_trans_infos(parent_rot_mtx=parent_rot_mtx, parent_trans_vec=parent_trans_vec) ## get rot and trans mtx and vecs ### 

        # get the parent rot mtx and the joint rot mtx # # joint rot mtx #
        tot_parent_rot_mtx = torch.matmul(parent_rot_mtx, joint_rot_mtx)
        tot_parent_trans_vec = torch.matmul(parent_rot_mtx, joint_trans_vec.unsqueeze(-1)).view(3) + parent_trans_vec
        
        self.tot_rot_mtx = tot_parent_rot_mtx
        self.tot_trans_vec = tot_parent_trans_vec
        
        # self.tot_rot_mtx = np.copy(tot_parent_rot_mtx)
        # self.tot_trans_vec = np.copy(tot_parent_trans_vec)
        
        ### visual_pts_list for recording visual pts ###
        # so the inertial should be transformed by the tot_rot_mtx #  # transform visual pts # 
        cur_body_visual_pts = self.body.transform_expanded_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        # visual_pts_list.append(cur_body_visual_pts)
        self.cur_inertia_inv = self.cur_inertia_inv + self.body.compute_inertial_inv(self.tot_rot_mtx)
        self.cur_inertia = self.cur_inertia + self.body.compute_inertia(self.tot_rot_mtx)
        
        # 
        # cur_body_transformations = (self.tot_rot_mtx.detach().clone(), self.tot_trans_vec.detach().clone())
        # link_name_to_transformations_and_transformed_pts[self.body.name] = (cur_body_visual_pts.detach().clone(), cur_body_transformations)
        
        # 
        
        children_visual_pts_list = []
        children_pts_mass = []
        children_pts_mass.append(torch.ones((cur_body_visual_pts.size(0), ), dtype=torch.float32).cuda() / float(cur_body_visual_pts.size(0)))
        children_visual_pts_list.append(cur_body_visual_pts)
        for cur_link in self.children:
            # cur_link.parent_rot_mtx = np.copy(tot_parent_rot_mtx) ### set children parent rot mtx and the trans vec
            # cur_link.parent_trans_vec = np.copy(tot_parent_trans_vec) ## 
            cur_link.parent_rot_mtx = tot_parent_rot_mtx  ### set children parent rot mtx and the trans vec #
            cur_link.parent_trans_vec = tot_parent_trans_vec  ## 
            # cur_link.compute_transformation() ## compute self's transformations
            children_visual_pts_list, children_pts_mass = cur_link.compute_expanded_visual_pts_transformation_via_current_state(tot_parent_rot_mtx, tot_parent_trans_vec, children_visual_pts_list, children_pts_mass)
            
            self.cur_inertia_inv = self.cur_inertia_inv + cur_link.cur_inertia_inv
            self.cur_inertia = self.cur_inertia + cur_link.cur_inertia ### get the current inertia ###
            
        children_visual_pts = torch.cat(children_visual_pts_list, dim=0)
        self.expanded_visual_pts = children_visual_pts.detach() # 
        visual_pts_list.append(children_visual_pts)
        children_pts_mass = torch.cat(children_pts_mass, dim=0)
        self.expanded_visual_pts_mass = children_pts_mass.detach()
        visual_pts_mass.append(children_pts_mass)
        # print(f"children_pts_mass: {children_pts_mass.size()}")
        return visual_pts_list, visual_pts_mass

    
    def set_body_expanded_visual_pts(self, link_name_to_ragged_expanded_visual_pts):
        self.body.expanded_visual_pts_ref = link_name_to_ragged_expanded_visual_pts[self.body.name].detach().clone()
        
        for cur_link in self.children:
            cur_link.set_body_expanded_visual_pts(link_name_to_ragged_expanded_visual_pts)
    
    
    def get_visual_pts_rgba_values(self, pts_rgba_vals_list):
        
        cur_body_visual_rgba_vals = self.body.get_visual_pts_colors()
        pts_rgba_vals_list.append(cur_body_visual_rgba_vals)
        
        for cur_link in self.children:
            cur_link.get_visual_pts_rgba_values(pts_rgba_vals_list)
        
        
    
    def compute_transformation(self,):
        self.joint.compute_transformation()
        # self.curr_rot_mtx = np.copy(self.joint.rot_mtx)
        # self.curr_trans_vec = np.copy(self.joint.trans_vec)
        
        self.curr_rot_mtx = self.joint.rot_mtx
        self.curr_trans_vec = self.joint.trans_vec
        # rot_p (rot_c p + trans_c) + trans_p # 
        # rot_p rot_c p + rot_p trans_c + trans_p #
        #### matmul ####
        # tot_parent_rot_mtx = np.matmul(self.parent_rot_mtx, self.curr_rot_mtx)
        # tot_parent_trans_vec = np.matmul(self.parent_rot_mtx, self.curr_trans_vec.reshape(3, 1)).reshape(3) + self.parent_trans_vec
        
        tot_parent_rot_mtx = torch.matmul(self.parent_rot_mtx, self.curr_rot_mtx)
        tot_parent_trans_vec = torch.matmul(self.parent_rot_mtx, self.curr_trans_vec.unsqueeze(-1)).view(3) + self.parent_trans_vec
        
        self.tot_rot_mtx = tot_parent_rot_mtx
        self.tot_trans_vec = tot_parent_trans_vec
        
        # self.tot_rot_mtx = np.copy(tot_parent_rot_mtx)
        # self.tot_trans_vec = np.copy(tot_parent_trans_vec)
        
        for cur_link in self.children:
            # cur_link.parent_rot_mtx = np.copy(tot_parent_rot_mtx) ### set children parent rot mtx and the trans vec
            # cur_link.parent_trans_vec = np.copy(tot_parent_trans_vec) ## 
            cur_link.parent_rot_mtx = tot_parent_rot_mtx  ### set children parent rot mtx and the trans vec #
            cur_link.parent_trans_vec = tot_parent_trans_vec  ## 
            cur_link.compute_transformation() ## compute self's transformations
            
    def get_name_to_visual_pts_faces(self, name_to_visual_pts_faces):
        # transform_visual_pts # ## rot_mt
        self.body.transform_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        name_to_visual_pts_faces[self.body.name] = {"pts": self.body.visual_pts, "faces": self.body.visual_faces_ref}
        for cur_link in self.children:
            cur_link.get_name_to_visual_pts_faces(name_to_visual_pts_faces) ## transform the pts faces 

    def get_visual_pts_list(self, visual_pts_list):
        # transform_visual_pts # ## rot_mt
        self.body.transform_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        visual_pts_list.append(self.body.visual_pts) # body template #
        # name_to_visual_pts_faces[self.body.name] = {"pts": self.body.visual_pts, "faces": self.body.visual_faces_ref}
        for cur_link in self.children:
            # cur_link.get_name_to_visual_pts_faces(name_to_visual_pts_faces) ## transform the pts faces 
            visual_pts_list = cur_link.get_visual_pts_list(visual_pts_list)
        return visual_pts_list
    
        
    def set_joint_idx(self, joint_name_to_idx):
        self.joint.set_joint_idx(joint_name_to_idx)
        for cur_link in self.children:
            cur_link.set_joint_idx(joint_name_to_idx)
        # if self.name in joint_name_to_idx:
        #     self.joint_idx = joint_name_to_idx[self.name]

    def get_nn_pts(self,):
        nn_pts = 0
        nn_pts += self.body.get_nn_pts()
        for cur_link in self.children:
            nn_pts += cur_link.get_nn_pts()
        self.nn_pts = nn_pts
        return self.nn_pts
    
    def clear_grads(self,):
        
        if self.parent_rot_mtx.grad is not None:
            self.parent_rot_mtx.grad.data = self.parent_rot_mtx.grad.data * 0.
        if self.parent_trans_vec.grad is not None:
            self.parent_trans_vec.grad.data = self.parent_trans_vec.grad.data * 0.
        if self.curr_rot_mtx.grad is not None:
            self.curr_rot_mtx.grad.data = self.curr_rot_mtx.grad.data * 0.
        if self.curr_trans_vec.grad is not None:
            self.curr_trans_vec.grad.data = self.curr_trans_vec.grad.data * 0.
        if self.tot_rot_mtx.grad is not None:
            self.tot_rot_mtx.grad.data = self.tot_rot_mtx.grad.data * 0.
        if self.tot_trans_vec.grad is not None:
            self.tot_trans_vec.grad.data = self.tot_trans_vec.grad.data * 0.

        self.joint.clear_grads()
        self.body.clear_grad()
        for cur_link in self.children:
            cur_link.clear_grads()
    
    def set_args(self, args):
        self.args = args
        for cur_link in self.children:
            cur_link.set_args(args)
            
    
    
        
class Robot: # robot and the robot #
    def __init__(self, children_links, args) -> None:
        self.children = children_links
        ### global rotation quaternion ###
        self.glb_rotation = nn.Parameter(torch.eye(3, dtype=torch.float32,  requires_grad=True).cuda(), requires_grad=True)
        ### global translation vectors ##
        self.glb_trans = nn.Parameter(torch.tensor([ 0., 0., 0.], dtype=torch.float32,  requires_grad=True).cuda(), requires_grad=True)
        self.args = args
        
    def set_state(self, name_to_state):
        for cur_link in self.children:
            cur_link.set_state(name_to_state)
    
    def compute_transformation(self,):
        for cur_link in self.children:
            cur_link.compute_transformation()
    
    def get_name_to_visual_pts_faces(self, name_to_visual_pts_faces):
        for cur_link in self.children:
            cur_link.get_name_to_visual_pts_faces(name_to_visual_pts_faces)
            
    def get_visual_pts_list(self, visual_pts_list):
        for cur_link in self.children:
            visual_pts_list = cur_link.get_visual_pts_list(visual_pts_list)
        return visual_pts_list
            
    def get_visual_faces_list(self, visual_faces_list):
        for cur_link in self.children:
            visual_faces_list = cur_link.get_visual_faces_list(visual_faces_list) 
        return visual_faces_list
            
    def set_joint_idx(self, joint_name_to_idx):
        for cur_link in self.children:
            cur_link.set_joint_idx(joint_name_to_idx) ### set joint idx ###

    def set_state_via_vec(self, state_vec): ### set the state vec for the state vec ###
        for cur_link in self.children: ### set the state vec for the state vec ###
            cur_link.set_state_via_vec(state_vec)
        # self.joint.set_state_via_vec(state_vec)
        # for child_link in self.children:
        #     child_link.set_state_via_vec(state_vec)
        
    # get_tot_transformed_joints
    def get_tot_transformed_joints(self, transformed_joints): # i
        for cur_link in self.children: # 
            transformed_joints = cur_link.get_tot_transformed_joints(transformed_joints)
        return transformed_joints
    
    def get_joint_states(self, joint_states):
        for cur_link in self.children:
            joint_states = cur_link.get_joint_states(joint_states)
        return joint_states
    
    def get_nn_pts(self):
        nn_pts = 0
        for cur_link in self.children:
            nn_pts += cur_link.get_nn_pts()
        self.nn_pts = nn_pts
        return self.nn_pts
    
    def set_args(self, args):
        self.args = args
        for cur_link in self.children: ## args ##
            cur_link.set_args(args)
    
    def print_grads(self):
        for cur_link in self.children:
            cur_link.print_grads()
        
    def clear_grads(self,): ## clear grads ##
        for cur_link in self.children:
            cur_link.clear_grads()
            
    def compute_transformation_via_state_vecs(self, state_vals, visual_pts_list):
        for cur_link in self.children:
            cur_link.compute_transformation_via_state_vecs(state_vals, cur_link.parent_rot_mtx, cur_link.parent_trans_vec, visual_pts_list)
        return visual_pts_list
    
    # get_visual_pts_rgba_values(self, pts_rgba_vals_list):
    def get_visual_pts_rgba_values(self, pts_rgba_vals_list):
        for cur_link in self.children:
            cur_link.get_visual_pts_rgba_values(pts_rgba_vals_list)
        return pts_rgba_vals_list ## compute pts rgba vals list ##
    
    def set_init_states(self, init_states):
        # glb_rot, glb_trans #
        ###### set the initial state ######
        glb_rot = init_states['glb_rot']
        self.glb_rotation.data[:, :] = glb_rot[:, :]
        glb_trans = init_states['glb_trans']
        self.glb_trans.data[:] = glb_trans[:] # glb trans #
        # parent_rot_mtx, parent_trans_vec #
        for cur_link in self.children:
            cur_link.parent_rot_mtx.data[:, :] = self.glb_rotation.data[:, :]
            cur_link.parent_trans_vec.data[:] = self.glb_trans.data[:]
            cur_link.set_init_states()
        
    def set_init_states_target_value(self, tot_init_states):
        glb_rot = tot_init_states['glb_rot']
        self.glb_rotation.data[:, :] = glb_rot[:, :]
        glb_trans = tot_init_states['glb_trans']
        self.glb_trans.data[:] = glb_trans[:] # glb trans #
        links_init_states = tot_init_states['links_init_states']
        for cur_link in self.children:
            cur_link.parent_rot_mtx.data[:, :] = self.glb_rotation.data[:, :]
            cur_link.parent_trans_vec.data[:] = self.glb_trans.data[:]
            cur_link.set_init_states_target_value(links_init_states)
    
    
    def get_timestep_to_states(self, joint_nm_to_ts_to_states):
        for cur_link in self.children:
            joint_nm_to_ts_to_states = cur_link.get_timestep_to_states(joint_nm_to_ts_to_states)
        return joint_nm_to_ts_to_states
    
    
    def get_joint_nm_to_states(self, joint_nm_to_states):
        for cur_link in self.children:
            joint_nm_to_states = cur_link.get_joint_nm_to_states(joint_nm_to_states)
        return joint_nm_to_states
    
    # def set_penetration_forces(self, penetration_forces):
    def set_penetration_forces(self, penetration_forces, sampled_visual_pts_joint_idxes, joint_penetration_forces):
        for cur_link in self.children:
            cur_link.set_penetration_forces(penetration_forces, sampled_visual_pts_joint_idxes, joint_penetration_forces)
    
    # set_actions_and_update_states(..., penetration_forces)
    def set_actions_and_update_states(self, actions, cur_timestep, time_cons, penetration_forces=None, sampled_visual_pts_joint_idxes=None, joint_name_to_penetration_forces_intermediates=None):
        # delta_glb_rot; delta_glb_trans # #
        delta_glb_rotation = actions['delta_glb_rot']
        delta_glb_trans = actions['delta_glb_trans']
        cur_glb_rot = self.glb_rotation.data.detach()
        cur_glb_trans = self.glb_trans.data.detach()
        nex_glb_rot = torch.matmul(delta_glb_rotation, cur_glb_rot) # 
        nex_glb_trans = torch.matmul(delta_glb_rotation, cur_glb_trans.unsqueeze(-1)).squeeze(-1) + delta_glb_trans
        link_actions = actions['link_actions']
        self.glb_rotation = nex_glb_rot
        self.glb_trans = nex_glb_trans
        for cur_link in self.children: # glb trans # # 
            cur_link.set_actions_and_update_states(link_actions, cur_timestep, time_cons, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes, joint_name_to_penetration_forces_intermediates=joint_name_to_penetration_forces_intermediates, children_penetration_torques=None)
            
    def set_and_update_states(self, states, cur_timestep, time_cons):
        delta_glb_rotation = states['delta_glb_rot'] # 
        delta_glb_trans = states['delta_glb_trans']
        cur_glb_rot = self.glb_rotation.data.detach()
        cur_glb_trans = self.glb_trans.data.detach()
        nex_glb_rot = torch.matmul(delta_glb_rotation, cur_glb_rot)
        nex_glb_trans = torch.matmul(delta_glb_rotation, cur_glb_trans.unsqueeze(-1)).squeeze(-1) + delta_glb_trans
        link_states = states['link_states']
        self.glb_rotation = nex_glb_rot
        self.glb_trans = nex_glb_trans
        for cur_link in self.children: # glb trans # 
            cur_link.set_and_update_states(link_states, cur_timestep, time_cons)
        
    
    def compute_transformation_via_current_state(self, visual_pts_list, link_name_to_transformations_and_transformed_pts, joint_idxes= None):
        # visual_pts_mass_list = []
        visual_pts_list = []
        visual_pts_mass_list = []
        # visual_pts_mass = []
        for cur_link in self.children:
            visual_pts_list, visual_pts_mass_list = cur_link.compute_transformation_via_current_state(self.glb_rotation.data, self.glb_trans.data, visual_pts_list, visual_pts_mass_list, link_name_to_transformations_and_transformed_pts, joint_idxes=joint_idxes)
        visual_pts_list = torch.cat(visual_pts_list, dim=0)
        visual_pts_mass_list= torch.cat(visual_pts_mass_list, dim=0)
        return visual_pts_list, visual_pts_mass_list
    
    # compute_expanded_visual_pts_transformation_via_current_state
    def compute_expanded_visual_pts_transformation_via_current_state(self, visual_pts_list):
        # visual_pts_mass_list = []
        visual_pts_list = []
        visual_pts_mass_list = []
        # visual_pts_mass = []
        for cur_link in self.children:
            visual_pts_list, visual_pts_mass_list = cur_link.compute_expanded_visual_pts_transformation_via_current_state(self.glb_rotation.data, self.glb_trans.data, visual_pts_list, visual_pts_mass_list)
        visual_pts_list = torch.cat(visual_pts_list, dim=0)
        visual_pts_mass_list= torch.cat(visual_pts_mass_list, dim=0)
        return visual_pts_list, visual_pts_mass_list
    
    def set_body_expanded_visual_pts(self, link_name_to_ragged_expanded_visual_pts):
        for cur_link in self.children:
            cur_link.set_body_expanded_visual_pts(link_name_to_ragged_expanded_visual_pts)
    


# robot manager # 
# set the initial state # 
# record optimizable actions #
# record optimizable time constants #
# and with the external forces? #


def parse_nparray_from_string(strr, args):
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
    
    


def parse_data_from_xml(xml_fn, args):
    
    tree = ElementTree()
    tree.parse(xml_fn)
    
    ### get total robots ###
    robots = tree.findall("./robot")
    i_robot = 0
    tot_robots = []
    for cur_robot in robots:
        print(f"Getting robot: {i_robot}")
        i_robot += 1
        cur_links = cur_robot.findall("./link")
        # i_link = 0
        cur_robot_links = []
        for cur_link in cur_links: ## child of the link ##
            ### a parse link util -> the child of the link is composed of (the joint; body; and children links (with children or with no child here))
            # cur_link_name = cur_link.attrib["name"]
            # print(f"Getting link: {i_link} with name: {cur_link_name}")
            # i_link += 1 ## 
            cur_robot_links.append(parse_link_data(cur_link, args=args))
        cur_robot_obj = Robot(cur_robot_links, args=args)
        tot_robots.append(cur_robot_obj)
    
    
    tot_actuators = []
    actuators = tree.findall("./actuator/motor")
    joint_nm_to_joint_idx = {}
    i_act = 0
    for cur_act in actuators:
        cur_act_joint_nm  = cur_act.attrib["joint"]
        joint_nm_to_joint_idx[cur_act_joint_nm] = i_act
        i_act += 1 ### add the act ###
    
    tot_robots[0].set_joint_idx(joint_nm_to_joint_idx) ### set joint idx here ### # tot robots #
    tot_robots[0].get_nn_pts()
    tot_robots[1].get_nn_pts()
    
    return tot_robots


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
    def __init__(self, xml_fn, args) -> None:
        self.xml_fn = xml_fn
        self.args = args
        
        ## 
        active_robot, passive_robot =  parse_data_from_xml(xml_fn, args)
        
        #### set and initialize the time constant ####
        self.time_constant = nn.Embedding(
            num_embeddings=3, embedding_dim=1
        ).cuda()
        torch.nn.init.ones_(self.time_constant.weight) #
        self.time_constant.weight.data = self.time_constant.weight.data * 0.2 ### time_constant data #
        
        #### set optimizable actions ####
        self.optimizable_actions = nn.Embedding(
            num_embeddings=100, embedding_dim=22,
        ).cuda()
        torch.nn.init.zeros_(self.optimizable_actions.weight) #
        
        self.learning_rate = 5e-4
    
        self.active_robot = active_robot
        
        
        self.set_init_states()
        init_visual_pts = self.get_init_state_visual_pts()
        self.init_visual_pts = init_visual_pts
        
        self.robot_visual_faces_list = []
        self.robot_visual_faces_list = self.active_robot.get_visual_faces_list(self.robot_visual_faces_list)
        self.robot_visual_pts_list = []
        self.robot_visual_pts_list = self.active_robot.get_visual_pts_list(self.robot_visual_pts_list)
        
        self.robot_pts, self.robot_faces = merge_meshes(self.robot_visual_pts_list, self.robot_visual_faces_list)
        
        print(f"robot_pts: {self.robot_pts.size()}, self.robot_faces: {self.robot_faces.size()}")
        cur_robot_mesh = trimesh.Trimesh(vertices=self.robot_pts.detach().cpu().numpy(), faces=self.robot_faces.detach().cpu().numpy())
        cur_robot_mesh.export(f'init_robot_mesh.ply')
        
    
    
    def get_timestep_to_states(self):
        joint_nm_to_ts_to_states = {}
        joint_nm_to_ts_to_states = self.active_robot.get_timestep_to_states(joint_nm_to_ts_to_states)
        return joint_nm_to_ts_to_states    
    
    # so for each joint; get the joint 
    def get_joint_nm_to_states(self):
        joint_nm_to_states = {}
        joint_nm_to_states = self.active_robot.get_joint_nm_to_states(joint_nm_to_states)
        return joint_nm_to_states
        
    def set_init_states_target_value(self, init_states):
        glb_rot = torch.eye(n=3, dtype=torch.float32).cuda()
        glb_trans = torch.zeros((3,), dtype=torch.float32).cuda() ### glb_trans #### and the rot 3##

        tot_init_states = {}
        tot_init_states['glb_rot'] = glb_rot;
        tot_init_states['glb_trans'] = glb_trans; 
        tot_init_states['links_init_states'] = init_states
        self.active_robot.set_init_states_target_value(tot_init_states)
    
    
    def set_penetration_forces(self, penetration_forces, sampled_visual_pts_joint_idxes, joint_penetration_forces):
        self.active_robot.set_penetration_forces(penetration_forces, sampled_visual_pts_joint_idxes, joint_penetration_forces)
    
    def set_init_states(self):
        glb_rot = torch.eye(n=3, dtype=torch.float32).cuda()
        glb_trans = torch.zeros((3,), dtype=torch.float32).cuda() ### glb_trans #### and the rot 3##
        
        ### random rotation ###
        # glb_rot_np = R.random().as_matrix()
        # glb_rot = torch.from_numpy(glb_rot_np).float().cuda()
        ### random rotation ###
        
        # glb_rot, glb_trans #
        init_states = {}
        init_states['glb_rot'] = glb_rot;
        init_states['glb_trans'] = glb_trans; 
        self.active_robot.set_init_states(init_states)
    
    def get_init_state_visual_pts(self, ret_link_name_to_tansformations=False, ret_joint_idxes=False):
        visual_pts_list = [] # compute the transformation via current state #
        link_name_to_transformations_and_transformed_pts = {}
        joint_idxes = []
        visual_pts_list, visual_pts_mass_list = self.active_robot.compute_transformation_via_current_state( visual_pts_list, link_name_to_transformations_and_transformed_pts, joint_idxes=joint_idxes)
        joint_idxes = torch.cat(joint_idxes, dim=0)
        init_visual_pts = visual_pts_list
        if ret_link_name_to_tansformations and ret_joint_idxes:
            return init_visual_pts, link_name_to_transformations_and_transformed_pts, joint_idxes
        elif ret_link_name_to_tansformations:
            return init_visual_pts, link_name_to_transformations_and_transformed_pts
        elif ret_joint_idxes:
            return init_visual_pts, joint_idxes
        else:
            return init_visual_pts
        
    # init_visual_pts, link_name_to_transformations_and_transformed_pts = get_init_state_visual_pts(ret_link_name_to_tansformations=True)
    # set_body_expanded_visual_pts
    # expanded_visual_pts = compute_expanded_visual_pts_transformation_via_current_state()
    
    # expanded_init_visual_pts = compute_expanded_visual_pts_transformation_via_current_state
    def compute_expanded_visual_pts_transformation_via_current_state(self,):
        visual_pts_list = [] # compute the transformation via current state #
        # link_name_to_transformations_and_transformed_pts = {}
        visual_pts_list, visual_pts_mass_list = self.active_robot.compute_expanded_visual_pts_transformation_via_current_state( visual_pts_list)
        # init_visual_pts = visual_pts_list
        # if ret_link_name_to_tansformations:
        #     return init_visual_pts, link_name_to_transformations_and_transformed_pts
        # else:
        return visual_pts_list
    
    def set_body_expanded_visual_pts(self, link_name_to_ragged_expanded_visual_pts):
        self.active_robot.set_body_expanded_visual_pts(link_name_to_ragged_expanded_visual_pts)
        # for cur_link in self.children:
        #     cur_link.set_body_expanded_visual_pts(link_name_to_ragged_expanded_visual_pts)
    
    
    def set_and_update_states(self, states, cur_timestep):
        time_cons = self.time_constant(torch.zeros((1,), dtype=torch.long).cuda()) #
        ## set and update the states ##
        self.active_robot.set_and_update_states(states, cur_timestep, time_cons)
        # for cur_link in self.children: # glb trans # 
        #     cur_link.set_and_update_states(link_actions, cur_timestep, time_cons)
    
    # set_actions_and_update_states(..., penetration_forces)
    def set_actions_and_update_states(self, actions, cur_timestep, penetration_forces=None, sampled_visual_pts_joint_idxes=None):
        # 
        joint_name_to_penetration_forces_intermediates = {}
        time_cons = self.time_constant(torch.zeros((1,), dtype=torch.long).cuda()) ### time constant of the system ##
        self.active_robot.set_actions_and_update_states(actions, cur_timestep, time_cons, penetration_forces=penetration_forces, sampled_visual_pts_joint_idxes=sampled_visual_pts_joint_idxes, joint_name_to_penetration_forces_intermediates=joint_name_to_penetration_forces_intermediates) ### 
        return joint_name_to_penetration_forces_intermediates
    
    
    
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
        self.n_timesteps = 19 # first 19-timesteps optimization #
        self.nn_tot_optimization_iters = 1000
        # self.nn_tot_optimization_iters = 57
        # TODO: load reference points #
        self.ts_to_reference_pts = np.load(reference_pts_dict, allow_pickle=True).item() #### 
        self.ts_to_reference_pts = {
            ts: torch.from_numpy(self.ts_to_reference_pts[ts]).float().cuda() for ts in self.ts_to_reference_pts
        }
    
    # optimize the glboal state and 
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
                actions = {}
                actions['delta_glb_rot']  = torch.eye(3, dtype=torch.float32).cuda()
                actions['delta_glb_trans'] = torch.zeros((3,), dtype=torch.float32).cuda()
                actions_link_actions = self.optimizable_actions(torch.zeros((1,), dtype=torch.long).cuda() + cur_ts).squeeze(0)
                # actions_link_actions = actions_link_actions * 0.2
                # actions_link_actions = actions_link_actions * -1. # 
                actions['link_actions'] = actions_link_actions
                self.set_actions_and_update_states(actions=actions, cur_timestep=cur_ts) # update the interaction # 
                
                cur_visual_pts = robot_agent.get_init_state_visual_pts()
                ts_to_robot_points[cur_ts + 1] = cur_visual_pts.clone()
                
                cur_reference_pts = self.ts_to_reference_pts[cur_ts + 1]
                diff = torch.sum((cur_visual_pts - cur_reference_pts) ** 2, dim=-1)
                diff = diff.mean()
                
                # diff.
                self.optimizer.zero_grad()
                diff.backward()
                self.optimizer.step()
                
                tot_losses.append(diff.item())
                
            
            # for ts in ts_to_robot_points:
            #     # print(f"ts: {ts}")
            #     if not ts in self.ts_to_reference_pts:
            #         continue
            #     cur_robot_pts = ts_to_robot_points[ts]
            #     cur_reference_pts = self.ts_to_reference_pts[ts]
            #     diff = torch.sum((cur_robot_pts - cur_reference_pts) ** 2, dim=-1)
            #     diff = torch.mean(diff)
            #     tot_losses.append(diff)
                
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




def create_zero_states():
    nn_joints = 17
    joint_name_to_state = {}
    for i_j in range(nn_joints):
        cur_joint_name = f"joint{i_j + 1}"
        joint_name_to_state[cur_joint_name] = 0.
    return joint_name_to_state

# [6.96331033e-17 3.54807679e-06 1.74046190e-15 2.66367417e-05
#  1.22444894e-05 3.38976792e-06 1.46917635e-15 2.66367383e-05
#  1.22444882e-05 3.38976786e-06 1.97778813e-15 2.66367383e-05
#  1.22444882e-05 3.38976786e-06 4.76033293e-16 1.26279884e-05
#  3.51189993e-06 0.00000000e+00 4.89999978e-03 0.00000000e+00]


def rotation_matrix_from_axis_angle_np(axis, angle): # rotation_matrix_from_axis_angle -> 
        sin_ = np.sin(angle) #  ti.math.sin(angle)
        cos_ = np.cos(angle) #  ti.math.cos(angle)
        # sin_ = torch.sin(angle) #  ti.math.sin(angle)
        # cos_ = torch.cos(angle) #  ti.math.cos(angle)
        u_x, u_y, u_z = axis[0], axis[1], axis[2]
        u_xx = u_x * u_x
        u_yy = u_y * u_y
        u_zz = u_z * u_z
        u_xy = u_x * u_y
        u_xz = u_x * u_z
        u_yz = u_y * u_z ## 
        
        
        row_a = np.stack(
            [cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], axis=0
        )
        # print(f"row_a: {row_a.size()}")
        row_b = np.stack(
            [u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], axis=0
        )
        # print(f"row_b: {row_b.size()}")
        row_c = np.stack(
            [u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], axis=0
        )
        # print(f"row_c: {row_c.size()}")
        
        ### rot_mtx for the rot_mtx ###
        rot_mtx = np.stack(
            [row_a, row_b, row_c], axis=-1 ### rot_matrix of he matrix ##
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


def get_camera_to_world_poses(n=10, ):
    ## sample from the upper half sphere ##
    # theta and phi for the 
    theta = np.random.uniform(low=0.0, high=1.0, size=(n,)) * np.pi * 2. # xz palne # 
    phi = np.random.uniform(low=-1.0, high=0.0, size=(n,)) * np.pi ## [-0.5 \pi, 0.5 \pi] ## negative pi to the original pi 
    # theta = torch.from_numpy(theta).float().cuda()
    tot_c2w_matrix = []
    for i_n in range(n):
        # y_rot_vec = torch.tensor([0., 1., 0.]).float().cuda(th_cuda_idx)
        # y_rot_mtx = load_utils.rotation_matrix_from_axis_angle(rot_vec, rot_angle) 
        
        
        z_axis_rot_axis = np.array([0, 0, 1.], dtype=np.float32)
        z_axis_rot_angle = np.pi - theta[i_n]
        z_axis_rot_matrix = rotation_matrix_from_axis_angle_np(z_axis_rot_axis, z_axis_rot_angle)
        rotated_plane_rot_axis_ori = np.array([1, -1, 0], dtype=np.float32)
        rotated_plane_rot_axis_ori = rotated_plane_rot_axis_ori / np.sqrt(np.sum(rotated_plane_rot_axis_ori ** 2))
        rotated_plane_rot_axis = np.matmul(z_axis_rot_matrix, rotated_plane_rot_axis_ori)
        
        plane_rot_angle = phi[i_n]
        plane_rot_matrix = rotation_matrix_from_axis_angle_np(rotated_plane_rot_axis, plane_rot_angle)
        
        c2w_matrix = np.matmul(plane_rot_matrix, z_axis_rot_matrix)
        c2w_trans_matrix = np.array(
            [np.cos(theta[i_n]) * np.sin(phi[i_n]), np.sin(theta[i_n]) * np.sin(phi[i_n]), np.cos(phi[i_n])], dtype=np.float32
        )
        c2w_matrix = np.concatenate(
            [c2w_matrix, c2w_trans_matrix.reshape(3, 1)], axis=-1
        ) ##c2w matrix
        tot_c2w_matrix.append(c2w_matrix)
    tot_c2w_matrix = np.stack(tot_c2w_matrix, axis=0)
    return tot_c2w_matrix
        

def get_camera_to_world_poses_th(n=10, th_cuda_idx=0):
    ## sample from the upper half sphere ##
    # theta and phi for the 
    theta = np.random.uniform(low=0.0, high=1.0, size=(n,)) * np.pi * 2. # xz palne # 
    phi = np.random.uniform(low=-1.0, high=0.0, size=(n,)) * np.pi ## [-0.5 \pi, 0.5 \pi] ## negative pi to the original pi 
    
    # n_total = 14
    # n_xz = 14
    # n_y = 7
    # theta = [i_xz * 1.0 / float(n_xz) * np.pi * 2. for i_xz in range(n_xz)]
    # phi = [i_y * (-1.0) / float(n_y) * np.pi for i_y in range(n_y)]
    
    
    theta = torch.from_numpy(theta).float().cuda(th_cuda_idx)
    phi = torch.from_numpy(phi).float().cuda(th_cuda_idx)
    
    tot_c2w_matrix = []
    for i_n in range(n): # if use veyr dense views like those 
        y_rot_angle = theta[i_n]
        y_rot_vec = torch.tensor([0., 1., 0.]).float().cuda(th_cuda_idx)
        y_rot_mtx = rotation_matrix_from_axis_angle(y_rot_vec, y_rot_angle) 
        
        x_axis = torch.tensor([1., 0., 0.]).float().cuda(th_cuda_idx)
        y_rot_x_axis = torch.matmul(y_rot_mtx, x_axis.unsqueeze(-1)).squeeze(-1) ### y_rot_x_axis # 
        
        x_rot_angle = phi[i_n]
        x_rot_mtx = rotation_matrix_from_axis_angle(y_rot_x_axis, x_rot_angle)
        
        rot_mtx = torch.matmul(x_rot_mtx, y_rot_mtx)
        xyz_offset = torch.tensor([0., 0., 1.5]).float().cuda(th_cuda_idx) 
        rot_xyz_offset = torch.matmul(rot_mtx, xyz_offset.unsqueeze(-1)).squeeze(-1).contiguous() + 0.5 ### 3 for the xyz offset 
        
        c2w_matrix = torch.cat(
            [rot_mtx, rot_xyz_offset.unsqueeze(-1)], dim=-1
        )
        tot_c2w_matrix.append(c2w_matrix)
        
        
        # z_axis_rot_axis = np.array([0, 0, 1.], dtype=np.float32)
        # z_axis_rot_angle = np.pi - theta[i_n]
        # z_axis_rot_matrix = rotation_matrix_from_axis_angle_np(z_axis_rot_axis, z_axis_rot_angle)
        # rotated_plane_rot_axis_ori = np.array([1, -1, 0], dtype=np.float32)
        # rotated_plane_rot_axis_ori = rotated_plane_rot_axis_ori / np.sqrt(np.sum(rotated_plane_rot_axis_ori ** 2))
        # rotated_plane_rot_axis = np.matmul(z_axis_rot_matrix, rotated_plane_rot_axis_ori)
        
        # plane_rot_angle = phi[i_n]
        # plane_rot_matrix = rotation_matrix_from_axis_angle_np(rotated_plane_rot_axis, plane_rot_angle)
        
        # c2w_matrix = np.matmul(plane_rot_matrix, z_axis_rot_matrix)
        # c2w_trans_matrix = np.array(
        #     [np.cos(theta[i_n]) * np.sin(phi[i_n]), np.sin(theta[i_n]) * np.sin(phi[i_n]), np.cos(phi[i_n])], dtype=np.float32
        # )
        # c2w_matrix = np.concatenate(
        #     [c2w_matrix, c2w_trans_matrix.reshape(3, 1)], axis=-1
        # ) ##c2w matrix
        # tot_c2w_matrix.append(c2w_matrix)
    # tot_c2w_matrix = np.stack(tot_c2w_matrix, axis=0)
    tot_c2w_matrix = torch.stack(tot_c2w_matrix, dim=0)
    return tot_c2w_matrix


def get_camera_to_world_poses_th_routine_1(n=7, th_cuda_idx=0):
    ## sample from the upper half sphere ##
    # theta and phi for the 
    
    # theta = np.random.uniform(low=0.0, high=1.0, size=(n,)) * np.pi * 2. # xz palne # 
    # phi = np.random.uniform(low=-1.0, high=0.0, size=(n,)) * np.pi ## [-0.5 \pi, 0.5 \pi] ## negative pi to the original pi 
    
    # n_total = 14
    n_xz = 2 * n #  14
    n_y = n # 7
    theta = [i_xz * 1.0 / float(n_xz) * np.pi * 2. for i_xz in range(n_xz)]
    phi = [i_y * (-1.0) / float(n_y) * np.pi for i_y in range(n_y)]
    
    theta = torch.tensor(theta).float().cuda(th_cuda_idx)
    phi = torch.tensor(phi).float().cuda(th_cuda_idx)
    # theta = torch.from_numpy(theta).float().cuda(th_cuda_idx)
    # phi = torch.from_numpy(phi).float().cuda(th_cuda_idx)
    
    tot_c2w_matrix = []
    
    for i_theta in range(theta.size(0)):
        for i_phi in range(phi.size(0)):
            y_rot_angle = theta[i_theta]
            y_rot_vec = torch.tensor([0., 1., 0.]).float().cuda(th_cuda_idx)
            y_rot_mtx = rotation_matrix_from_axis_angle(y_rot_vec, y_rot_angle) 
            
            x_axis = torch.tensor([1., 0., 0.]).float().cuda(th_cuda_idx)
            y_rot_x_axis = torch.matmul(y_rot_mtx, x_axis.unsqueeze(-1)).squeeze(-1) ### y_rot_x_axis # 
            
            x_rot_angle = phi[i_phi]
            x_rot_mtx = rotation_matrix_from_axis_angle(y_rot_x_axis, x_rot_angle)
            
            rot_mtx = torch.matmul(x_rot_mtx, y_rot_mtx)
            xyz_offset = torch.tensor([0., 0., 1.5]).float().cuda(th_cuda_idx) 
            rot_xyz_offset = torch.matmul(rot_mtx, xyz_offset.unsqueeze(-1)).squeeze(-1).contiguous() + 0.5 ### 3 for the xyz offset 
            
            c2w_matrix = torch.cat(
                [rot_mtx, rot_xyz_offset.unsqueeze(-1)], dim=-1
            )
            tot_c2w_matrix.append(c2w_matrix)
    
    tot_c2w_matrix = torch.stack(tot_c2w_matrix, dim=0)
    return tot_c2w_matrix


def get_camera_to_world_poses_th_routine_2(n=7, th_cuda_idx=0):
    ## sample from the upper half sphere ##
    # theta and phi for the 
    
    # theta = np.random.uniform(low=0.0, high=1.0, size=(n,)) * np.pi * 2. # xz palne # 
    # phi = np.random.uniform(low=-1.0, high=0.0, size=(n,)) * np.pi ## [-0.5 \pi, 0.5 \pi] ## negative pi to the original pi 
    
    # n_total = 14
    n_xz = 2 * n #  14
    n_y = 2 * n # 7
    theta = [i_xz * 1.0 / float(n_xz) * np.pi * 2. for i_xz in range(n_xz)]
    # phi = [i_y * (-1.0) / float(n_y) * np.pi for i_y in range(n_y)]
    phi = [i_y * (-1.0) / float(n_y) * np.pi * 2. for i_y in range(n_y)]
    
    theta = torch.tensor(theta).float().cuda(th_cuda_idx)
    phi = torch.tensor(phi).float().cuda(th_cuda_idx)
    # theta = torch.from_numpy(theta).float().cuda(th_cuda_idx)
    # phi = torch.from_numpy(phi).float().cuda(th_cuda_idx)
    
    tot_c2w_matrix = []
    
    for i_theta in range(theta.size(0)):
        for i_phi in range(phi.size(0)):
            y_rot_angle = theta[i_theta]
            y_rot_vec = torch.tensor([0., 1., 0.]).float().cuda(th_cuda_idx)
            y_rot_mtx = rotation_matrix_from_axis_angle(y_rot_vec, y_rot_angle) 
            
            x_axis = torch.tensor([1., 0., 0.]).float().cuda(th_cuda_idx)
            y_rot_x_axis = torch.matmul(y_rot_mtx, x_axis.unsqueeze(-1)).squeeze(-1) ### y_rot_x_axis # 
            
            x_rot_angle = phi[i_phi]
            x_rot_mtx = rotation_matrix_from_axis_angle(y_rot_x_axis, x_rot_angle)
            
            rot_mtx = torch.matmul(x_rot_mtx, y_rot_mtx)
            xyz_offset = torch.tensor([0., 0., 1.5]).float().cuda(th_cuda_idx) 
            rot_xyz_offset = torch.matmul(rot_mtx, xyz_offset.unsqueeze(-1)).squeeze(-1).contiguous() + 0.5 ### 3 for the xyz offset 
            
            c2w_matrix = torch.cat(
                [rot_mtx, rot_xyz_offset.unsqueeze(-1)], dim=-1
            )
            tot_c2w_matrix.append(c2w_matrix)
    
    tot_c2w_matrix = torch.stack(tot_c2w_matrix, dim=0)
    return tot_c2w_matrix



#### Big TODO: the external contact forces from the manipulated object to the robot ####

## optimize for actions from the redmax model ##

if __name__=='__main__': # agent of the 
    
    xml_fn = "/home/xueyi/diffsim/DiffHand/assets/hand_sphere.xml"
    robot_agent = RobotAgent(xml_fn=xml_fn, args=None)
    init_visual_pts = robot_agent.init_visual_pts.detach().cpu().numpy()
    
    exit(0)
    
    xml_fn = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.xml"
    robot_agent = RobotAgent(xml_fn=xml_fn, args=None)
    init_visual_pts = robot_agent.init_visual_pts.detach().cpu().numpy()
    
    robot_agent.forward_stepping_test()
    cur_visual_pts = robot_agent.get_init_state_visual_pts()
    cur_visual_pts = cur_visual_pts.detach().cpu().numpy()
    
    reference_pts_dict = "timestep_to_visual_pts.npy"
    robot_agent.initialize_optimization(reference_pts_dict=reference_pts_dict)
    optimized_ts_to_visual_pts, ts_to_ref_points = robot_agent.forward_stepping_optimization()
    
    timestep_to_visual_pts = robot_agent.forward_stepping_test()
    np.save(f"cur_visual_pts.npy", timestep_to_visual_pts) # cur_visual_pts # 
    np.save(f"timestep_to_visual_pts_opt.npy", timestep_to_visual_pts) 
    np.save(f"timestep_to_visual_pts_opt.npy", optimized_ts_to_visual_pts) 
    np.save(f"timestep_to_ref_pts.npy", ts_to_ref_points) 
    
    
    exit(0)
    
    
    xml_fn = "/home/xueyi/diffsim/DiffHand/assets/hand_sphere.xml"
    robot_agent = RobotAgent(xml_fn=xml_fn, args=None)
    init_visual_pts = robot_agent.init_visual_pts.detach().cpu().numpy()
    
    # np.save(f"init_visual_pts.npy", init_visual_pts) # 
    
    # robot_agent.forward_stepping_test()
    # cur_visual_pts = robot_agent.get_init_state_visual_pts()
    # cur_visual_pts = cur_visual_pts.detach().cpu().numpy()
    
    # reference_pts_dict = "timestep_to_visual_pts.npy"
    # robot_agent.initialize_optimization(reference_pts_dict=reference_pts_dict)
    # optimized_ts_to_visual_pts, ts_to_ref_points = robot_agent.forward_stepping_optimization()
    
    # timestep_to_visual_pts = robot_agent.forward_stepping_test()
    # np.save(f"cur_visual_pts.npy", timestep_to_visual_pts) # cur_visual_pts # 
    # np.save(f"timestep_to_visual_pts_opt.npy", timestep_to_visual_pts) 
    # np.save(f"timestep_to_visual_pts_opt.npy", optimized_ts_to_visual_pts) 
    # np.save(f"timestep_to_ref_pts.npy", ts_to_ref_points) 
    
    exit(0)
    
    
    xml_fn =   "/home/xueyi/diffsim/DiffHand/assets/hand_sphere.xml"
    tot_robots = parse_data_from_xml(xml_fn=xml_fn)
    # tot_robots = 
    
    active_optimized_states = """-0.00025872 -0.00025599 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025694 -0.00024656 -0.00025556 0. 0.0049 0."""
    active_optimized_states = """-1.10617972 -1.10742263 -1.06198363 -1.03212746 -1.05429142 -1.08617289 -1.05868192 -1.01624365 -1.04478191 -1.08260959 -1.06719107 -1.04082455 -1.05995886 -1.08674006 -1.09396691 -1.08965532 -1.10036577 -10.7117466 -3.62511998 1.49450353"""
    # active_goal_optimized_states = """-1.10617972 -1.10742263 -1.0614858 -1.03189609 -1.05404354 -1.08610468 -1.05863293 -1.0174248 -1.04576456 -1.08297396 -1.06719107 -1.04082455 -1.05995886 -1.08674006 -1.09396691 -1.08965532 -1.10036577 -10.73396897 -3.68095432 1.50679285"""
    active_optimized_states = """-0.42455298 -0.42570447 -0.40567708 -0.39798589 -0.40953955 -0.42025055 -0.37910662 -0.496165 -0.37664644 -0.41942727 -0.40596508 -0.3982109 -0.40959847 -0.42024905 -0.41835001 -0.41929961 -0.42365131 -1.18756073 -2.90337822 0.4224685"""
    active_optimized_states = """-0.42442816 -0.42557961 -0.40366201 -0.3977891 -0.40947627 -0.4201424 -0.3799285 -0.3808375 -0.37953552 -0.42039598 -0.4058405 -0.39808804 -0.40947487 -0.42012458 -0.41822534 -0.41917521 -0.4235266 -0.87189658 -1.42093761 0.21977979"""
    
    active_robot = tot_robots[0]
    zero_states = create_zero_states()
    active_robot.set_state(zero_states)
    active_robot.compute_transformation()
    name_to_visual_pts_surfaces = {}
    active_robot.get_name_to_visual_pts_faces(name_to_visual_pts_surfaces)
    print(len(name_to_visual_pts_surfaces))
    
    sv_res_rt = "/home/xueyi/diffsim/DiffHand/examples/save_res"
    sv_res_rt = os.path.join(sv_res_rt, "load_utils_test")
    os.makedirs(sv_res_rt, exist_ok=True)
    
    tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"res_with_zero_states.npy")
    np.save(tmp_visual_res_sv_fn, name_to_visual_pts_surfaces)
    print(f"tmp visual res saved to {tmp_visual_res_sv_fn}")
    
    
    optimized_states = get_name_to_state_from_str(active_optimized_states)
    active_robot.set_state(optimized_states)
    active_robot.compute_transformation()
    name_to_visual_pts_surfaces = {}
    active_robot.get_name_to_visual_pts_faces(name_to_visual_pts_surfaces)
    print(len(name_to_visual_pts_surfaces))
    # sv_res_rt = "/home/xueyi/diffsim/DiffHand/examples/save_res"
    # sv_res_rt = os.path.join(sv_res_rt, "load_utils_test")
    # os.makedirs(sv_res_rt, exist_ok=True)
    
    # tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"res_with_optimized_states.npy")
    tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"active_ngoal_res_with_optimized_states_goal_n3.npy")
    np.save(tmp_visual_res_sv_fn, name_to_visual_pts_surfaces)
    print(f"tmp visual res with optimized states saved to {tmp_visual_res_sv_fn}")
    
