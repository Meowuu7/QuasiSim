
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

## calculate transformation to the frame ##

## the joint idx ##
##### th_cuda_idx #####

# name to the main axis? # 

# def get_body_name_to_main_axis()
# another one is just getting the joint offset positions? 
# and after the revolute transformation # # all revolute joint points ####
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
    # col_a = torch.stack(
    #     [cos_, sin_], dim=0 ### col of the rotation matrix
    # )
    # col_b = torch.stack(
    #     [-1. * sin_, cos_], dim=0 ## cols of the rotation matrix
    # )
    # rot_mtx = torch.stack(
    #     [col_a, col_b], dim=-1 ### rotation matrix
    # )
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
        u_yz = u_y * u_z ## 
        # rot_mtx = np.stack(
        #     [np.array([cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dtype=np.float32), 
        #      np.array([u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dtype=np.float32), 
        #      np.array([u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dtype=np.float32)
        #     ], axis=-1 ### np stack 
        # ) ## a single
        
        # rot_mtx = torch.stack(
        #     [
        #         torch.tensor([cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dtype=torch.float32),
        #         torch.tensor([u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dtype=torch.float32), 
        #         torch.tensor([u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dtype=torch.float32)
        #     ], dim=-1 ## stack those torch tensors ##
        # )
        
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
        
        # rot_mtx = torch.stack(
        #     [
                
        #         torch.tensor([cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dtype=torch.float32),
        #         torch.tensor([u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dtype=torch.float32), 
        #         torch.tensor([u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dtype=torch.float32)
        #     ], dim=-1 ## stack those torch tensors ##
        # )
        
        # rot_mtx_numpy = rot_mtx.to_numpy()
        # rot_mtx_at_rot_mtx = rot_mtx @ rot_mtx.transpose()
        # print(rot_mtx_at_rot_mtx)
        return rot_mtx

## joint name = "joint3" ##
# <joint name="joint3" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e7"/>
class Joint:
    def __init__(self, name, joint_type, axis, pos, quat, frame, damping, args) -> None:
        self.name = name
        self.type = joint_type
        self.axis = axis
        self.pos = pos
        self.quat = quat
        self.frame = frame
        self.damping = damping
        
        self.args = args
        
        #### TODO: the dimension of the state vector ? ####
        # self.state = 0. ## parameter
        self.state = nn.Parameter(
            torch.zeros((1,), dtype=torch.float32, requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        )
        # self.rot_mtx = np.eye(3, dtypes=np.float32)
        # self.trans_vec = np.zeros((3,), dtype=np.float32) ## rot m
        self.rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32, requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True)
        self.trans_vec  = nn.Parameter(torch.zeros((3,), dtype=torch.float32,  requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True)
        # self.rot_mtx = np.eye(3, dtype=np.float32)
        # self.trans_vec = np.zeros((3,), dtype=np.float32)
        
        self.axis_rot_mtx = torch.tensor(
            [
                [1, 0, 0], [0, -1, 0], [0, 0, -1]
            ], dtype=torch.float32
        ).cuda(self.args.th_cuda_idx)
        
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
            ### rot_mtx ### ### rot_mtx ###
            rot_mtx = plane_rotation_matrix_from_angle_xz(cur_joint_rot_val)
            # rot_mtx = plane_rotation_matrix_from_angle(cur_joint_rot_val) ### 2 x 2 rot matrix # 
            # cur joint rot val #
            # rot mtx of the rotation 
            # xy_val = 
            # axis_rot_mtx
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

    def transform_joints_via_parent_rot_trans_infos(self, parent_rot_mtx, parent_trans_vec):
        # 
        # if self.type == "revolute" or self.type == "free2d":
        transformed_joint_pts = torch.matmul(parent_rot_mtx, self.pos.view(3 ,1).contiguous()).view(3).contiguous() + parent_trans_vec
        # else:
        self.transformed_joint_pts = transformed_joint_pts ### get self transformed joint pts here ###
        return transformed_joint_pts
        # if self.joint_idx >= 0:
        #     cur_joint_state = state_vals[self.joint_idx]
        # else:
        #     cur_joint_state = self.state # state #
        # # use the state to transform them # # transform ### transform the state ##
        # # use the state to transform them # # transform them for the state # transform for the state #
        # if self.type == "revolute":
        #     # print(f"computing transformation matrices with axis: {self.axis}, state: {self.state}")
        #     # rotation matrix from the axis angle # 
        #     rot_mtx = rotation_matrix_from_axis_angle(self.axis, cur_joint_state)
        #     # rot_mtx(p - p_v) + p_v -> rot_mtx p - rot_mtx p_v + p_v
        #     # trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
        #     # self.rot_mtx = np.copy(rot_mtx)
        #     # self.trans_vec = np.copy(trans_vec)
        #     trans_vec = self.pos - torch.matmul(rot_mtx, self.pos.view(3, 1)).view(3).contiguous()
        #     self.rot_mtx = rot_mtx
        #     self.trans_vec = trans_vec
        # elif self.type == "free2d":
        #     cur_joint_state = state_vals # still only for the current scene #
        #     # cur_joint_state
        #     cur_joint_rot_val = state_vals[2]
        #     ### rot_mtx ### ### rot_mtx ###
        #     rot_mtx = plane_rotation_matrix_from_angle_xz(cur_joint_rot_val)
        #     # rot_mtx = plane_rotation_matrix_from_angle(cur_joint_rot_val) ### 2 x 2 rot matrix # 
        #     # cur joint rot val #
        #     # rot mtx of the rotation 
        #     # xy_val = 
        #     # axis_rot_mtx
        #     # R_axis^T ( R R_axis (p) + trans (with the y-axis padded) )
        #     cur_trans_vec = torch.stack(
        #         [state_vals[0], torch.zeros_like(state_vals[0]), state_vals[1]], dim=0
        #     )
        #     # cur_trans_vec # 
        #     rot_mtx = torch.matmul(self.axis_rot_mtx.transpose(1, 0), torch.matmul(rot_mtx, self.axis_rot_mtx))
        #     trans_vec = torch.matmul(self.axis_rot_mtx.transpose(1, 0), cur_trans_vec.unsqueeze(-1).contiguous()).squeeze(-1).contiguous() + self.pos
            
        #     self.rot_mtx = rot_mtx
        #     self.trans_vec = trans_vec ## rot_mtx and trans_vec # 
        # else:
        #     ### TODO: implement transformations for joints in other types ###
        #     pass
        # return self.rot_mtx, self.trans_vec

## fixed joint -> translation and rotation ##
## revolute joint -> can be actuated ##
## set states and compute the transfromations in a top-to-down manner ##

## trnasform the robot -> a list of qs ##
## a list of qs ##
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
        
        ### for sphere object ###
        self.radius = radius
        ## or vertices here #
        ## pass them to the child and treat them as the parent transformation ##
        
        self.visual_pts_ref = None
        self.visual_faces_ref = None
        
        self.visual_pts = None ## visual pts and 
        
        self.body_name_to_main_axis = get_body_name_to_main_axis() ### get the body name to main axis here #
        
        self.get_visual_counterparts()
        
        
    def update_radius(self,):
        self.radius.data = self.radius.data - self.radius.grad.data
        
        self.radius.grad.data = self.radius.grad.data * 0.
        
    
    def update_xml_file(self,):
        xml_content_with_flexible_radius = f"""<redmax model="hand">
<!-- 1) change the damping value here? -->
<!-- 2) change the center of mass -->
    <option integrator="BDF2" timestep="0.01" gravity="0. 0. -0.000098"/>
	<ground pos="0 0 -10" normal="0 0 1"/>
	<default>
        <general_primitive_contact kn="1e6" kt="1e3" mu="0.8" damping="3e1" />
    </default>

    <robot>
        <link name="link0">
            <joint name="joint0" type="fixed" pos="0 0 0" quat="1 0 0 0" frame="WORLD"/>
            <body name="body0" type="mesh"  filename="hand/body0.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.700000 0.700000 0.700000 1"/>
            <link name="link1">
                <joint name="joint1" type="revolute" axis="0.000000 0.000000 -1.000000" pos="-3.300000 -5.689700 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                <body name="body1" type="mesh"  filename="hand/body1.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.600000 0.600000 0.600000 1"/>
                <link name="link2">
                    <joint name="joint2" type="revolute" axis="1.000000 0.000000 0.000000" pos="-3.300000 -7.680000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                    <body name="body2" type="mesh"  filename="hand/body2.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.500000 0.500000 0.500000 1"/>
                </link>
            </link>
            <link name="link3">
                <!-- revolute joint -->
                <joint name="joint3" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                <body name="body3" type="mesh"  filename="hand/body3.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.600000 0.600000 0.600000 1"/>
                <link name="link4">
                    <joint name="joint4" type="revolute" axis="0.000000 1.000000 0.000000" pos="6.680000 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                    <body name="body4" type="mesh"  filename="hand/body4.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.500000 0.500000 0.500000 1"/>
                        <link name="link5">
                            <joint name="joint5" type="revolute" axis="0.000000 1.000000 0.000000" pos="11.080000 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                            <body name="body5" type="mesh"  filename="hand/body5.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.400000 0.400000 0.400000 1"/>
                                <link name="link6">
                                    <joint name="joint6" type="revolute" axis="0.000000 1.000000 0.000000" pos="15.480000 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                                    <body name="body6" type="mesh"  filename="hand/body6.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.300000 0.300000 0.300000 1"/>
                                </link>
                        </link>
                </link>
            </link>
            <link name="link7">
                <joint name="joint7" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 -1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                <body name="body7" type="mesh"  filename="hand/body7.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.600000 0.600000 0.600000 1"/>
                <link name="link8">
                    <joint name="joint8" type="revolute" axis="0.000000 1.000000 0.000000" pos="6.680000 -1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                    <body name="body8" type="mesh"  filename="hand/body8.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.500000 0.500000 0.500000 1"/>
                        <link name="link9">
                            <joint name="joint9" type="revolute" axis="0.000000 1.000000 0.000000" pos="11.080000 -1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                            <body name="body9" type="mesh"  filename="hand/body9.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.400000 0.400000 0.400000 1"/>
                                <link name="link10">
                                    <joint name="joint10" type="revolute" axis="0.000000 1.000000 0.000000" pos="15.480000 -1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                                    <body name="body10" type="mesh"  filename="hand/body10.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.300000 0.300000 0.300000 1"/>
                                </link>
                        </link>
                </link>
            </link>
            <link name="link11">
                <joint name="joint11" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                <body name="body11" type="mesh"  filename="hand/body11.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.600000 0.600000 0.600000 1"/>
                <link name="link12">
                    <joint name="joint12" type="revolute" axis="0.000000 1.000000 0.000000" pos="6.680000 1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                    <body name="body12" type="mesh"  filename="hand/body12.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.500000 0.500000 0.500000 1"/>
                        <link name="link13">
                            <joint name="joint13" type="revolute" axis="0.000000 1.000000 0.000000" pos="11.080000 1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                            <body name="body13" type="mesh"  filename="hand/body13.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.400000 0.400000 0.400000 1"/>
                                <link name="link14">
                                    <joint name="joint14" type="revolute" axis="0.000000 1.000000 0.000000" pos="15.480000 1.475000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                                    <body name="body14" type="mesh"  filename="hand/body14.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.300000 0.300000 0.300000 1"/>
                                </link>
                        </link>
                </link>
            </link>
            <link name="link15">
                <joint name="joint15" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                <body name="body15" type="mesh"  filename="hand/body15.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.600000 0.600000 0.600000 1"/>
                <link name="link16">
                    <joint name="joint16" type="revolute" axis="0.000000 1.000000 0.000000" pos="6.680000 4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                    <body name="body16" type="mesh"  filename="hand/body16.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.500000 0.500000 0.500000 1"/>
                        <link name="link17">
                            <joint name="joint17" type="revolute" axis="0.000000 1.000000 0.000000" pos="11.080000 4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e4"/>
                            <body name="body17" type="mesh"  filename="hand/body17.obj"  pos="0 0 0"  quat="1 0 0 0"  transform_type="OBJ_TO_WORLD"  density="1"  mu="0" rgba="0.400000 0.400000 0.400000 1"/>
                        </link>
                </link>
            </link>
        </link>
    </robot>
	
	<robot>
		<link name="sphere">
			<joint name="sphere" type="free2d" pos = "10. 0.0 3.5" quat="1 -1 0 0" format="LOCAL" damping="0"/>
			<body name="sphere" type="sphere" radius="{self.radius[0].detach().cpu().item()}" pos="0 0 0" quat="1 0 0 0" density="0.5" mu="0" texture="resources/textures/sphere.jpg"/>
		</link>
	</robot>
	
	<contact>
		<ground_contact body="sphere" kn="1e6" kt="1e3" mu="0.8" damping="3e1"/>
		<general_primitive_contact general_body="body0" primitive_body="sphere"/>
		<general_primitive_contact general_body="body1" primitive_body="sphere"/>
		<general_primitive_contact general_body="body2" primitive_body="sphere"/>
		<general_primitive_contact general_body="body3" primitive_body="sphere"/>
		<general_primitive_contact general_body="body4" primitive_body="sphere"/>
		<general_primitive_contact general_body="body5" primitive_body="sphere"/>
		<general_primitive_contact general_body="body6" primitive_body="sphere"/>
		<general_primitive_contact general_body="body7" primitive_body="sphere"/>
		<general_primitive_contact general_body="body8" primitive_body="sphere"/>
		<general_primitive_contact general_body="body9" primitive_body="sphere"/>
		<general_primitive_contact general_body="body10" primitive_body="sphere"/>
		<general_primitive_contact general_body="body11" primitive_body="sphere"/>
		<general_primitive_contact general_body="body12" primitive_body="sphere"/>
		<general_primitive_contact general_body="body13" primitive_body="sphere"/>
		<general_primitive_contact general_body="body14" primitive_body="sphere"/>
		<general_primitive_contact general_body="body15" primitive_body="sphere"/>
		<general_primitive_contact general_body="body16" primitive_body="sphere"/>
		<general_primitive_contact general_body="body17" primitive_body="sphere"/>
	</contact>

    <actuator>
        <motor joint="joint1"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint2"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint3"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint4"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint5"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint6"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint7"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint8"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint9"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint10"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint11"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint12"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint13"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint14"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint15"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint16"  ctrl="force"  ctrl_range="-3e5 3e5"/>
        <motor joint="joint17"  ctrl="force"  ctrl_range="-3e5 3e5"/>
    </actuator>
</redmax>
"""
        xml_loading_fn = "/home/xueyi/diffsim/DiffHand/assets/hand_sphere_free_sphere_geo_test.xml"
        with open(xml_loading_fn, "w") as wf:
            wf.write(xml_content_with_flexible_radius)
            wf.close()
            
    ### get visual pts colorrs ### ### 
    def get_visual_pts_colors(self, ):
        tot_visual_pts_nn = self.visual_pts_ref.size(0)
        # self.pts_rgba = [torch.from_numpy(self.rgba).float().cuda(self.args.th_cuda_idx) for _ in range(tot_visual_pts_nn)] # total visual pts nn 
        self.pts_rgba = [torch.tensor(self.rgba.data).cuda(self.args.th_cuda_idx) for _ in range(tot_visual_pts_nn)] # total visual pts nn  skeletong
        self.pts_rgba = torch.stack(self.pts_rgba, dim=0) # 
        return self.pts_rgba
    
    def get_visual_counterparts(self,):
        ### TODO: implement this for visual counterparts ### mid line regression and name to body mapping relations --- for each body, how to calculate the midline and other properties?
        ######## get body type ########## get visual midline of the input mesh and the mesh vertices? ######## # skeleton of the hand -> 21 points ? retarget from this hand to the mano hand and use the mano hand priors?
        if self.body_type == "sphere":
            filename = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo/meshes/18.obj"
            if not os.path.exists(filename):
                filename = "/data/xueyi/diffsim/DiffHand/assets/18.obj"
        else:
            filename = self.filename
            rt_asset_path = "/home/xueyi/diffsim/DiffHand/assets" ### assets folder ###
            if not os.path.exists(rt_asset_path):
                rt_asset_path = "/data/xueyi/diffsim/DiffHand/assets"
            filename = os.path.join(rt_asset_path, filename)
        body_mesh = trimesh.load(filename, process=False) 
        # verts = np.array(body_mesh.vertices)
        # faces = np.array(body_mesh.faces, dtype=np.long)
        
        # self.visual_pts_ref = np.copy(verts) ## verts ##
        # self.visual_faces_ref = np.copy(faces) ## faces 
        # self.visual_pts_ref # 
        
        #### body_mesh.vertices ####
        # verts = torch.tensor(body_mesh.vertices, dtype=torch.float32).cuda(self.args.th_cuda_idx)
        # faces = torch.tensor(body_mesh.faces, dtype=torch.long).cuda(self.args.th_cuda_idx)
        #### body_mesh.vertices ####
        
        # self.pos = nn.Parameter(
        #     torch.tensor([0., 0., 0.], dtype=torch.float32,  requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        # )
        
        self.pos = nn.Parameter(
            torch.tensor(self.pos.detach().cpu().tolist(), dtype=torch.float32,  requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        )
        
        ### Step 1 ### -> set the pos to the correct initial pose ###
        
        self.radius = nn.Parameter(
            torch.tensor([self.args.initial_radius], dtype=torch.float32, requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        )
        ### visual pts ref ### ## body_mesh.vertices -> # 
        self.visual_pts_ref = torch.tensor(body_mesh.vertices, dtype=torch.float32).cuda(self.args.th_cuda_idx)
        
        # if self.name == "sphere":
        #     self.visual_pts_ref = self.visual_pts_ref  / 2. #  the initial radius
        #     self.visual_pts_ref = self.visual_pts_ref * self.radius ## multiple the initla radius # 
        
        # self.visual_pts_ref = nn.Parameter(
        #     torch.tensor(body_mesh.vertices, dtype=torch.float32, requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        # )
        # self.visual_faces_ref = nn.Parameter(
        #     torch.tensor(body_mesh.faces, dtype=torch.long, requires_grad=True).cuda(self.args.th_cuda_idx), requires_grad=True
        # )
        self.visual_faces_ref = torch.tensor(body_mesh.faces, dtype=torch.long).cuda(self.args.th_cuda_idx)
        
        # body_name_to_main_axis
        # body_name_to_main_axis for the body_name_to_main_axis # 
        # visual_faces_ref # 
        # visual_pts_ref #
        
        minn_pts, _ = torch.min(self.visual_pts_ref, dim=0) ### get the visual pts minn ###
        maxx_pts, _ = torch.max(self.visual_pts_ref, dim=0) ### visual pts maxx ###
        mean_pts = torch.mean(self.visual_pts_ref, dim=0) ### mean_pts of the mean_pts ###
        
        if self.name in self.body_name_to_main_axis:
            cur_main_axis = self.body_name_to_main_axis[self.name] ## get the body name ##
            
            if cur_main_axis == -2:
                main_axis_pts = minn_pts[1] # the main axis pts
                full_main_axis_pts = torch.tensor([mean_pts[0], main_axis_pts, mean_pts[2]], dtype=torch.float32).cuda(self.args.th_cuda_idx)
            elif cur_main_axis == 1:
                main_axis_pts = maxx_pts[0] # the maxx axis pts
                full_main_axis_pts = torch.tensor([main_axis_pts, mean_pts[1], mean_pts[2]], dtype=torch.float32).cuda(self.args.th_cuda_idx)
            self.full_main_axis_pts_ref = full_main_axis_pts
        else:
            self.full_main_axis_pts_ref = mean_pts.clone() ### get the mean pts ###
        # mean_pts
        # main_axis_pts = 
            
        
        # self.visual_pts_ref = verts #
        # self.visual_faces_ref = faces #
        # get visual points colors # the color should be an optimizable property # #  # or init visual point colors here ## # or init visual point colors here #
        # simulatoable assets ## for the
        
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
        
        # joint # parent_rot_mtx, parent_trans_vec
        self.parent_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True)
        self.parent_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True)
        self.curr_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True)
        self.curr_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True)
        # 
        self.tot_rot_mtx = nn.Parameter(torch.eye(n=3, dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True)
        self.tot_trans_vec = nn.Parameter(torch.zeros((3,), dtype=torch.float32).cuda(self.args.th_cuda_idx), requires_grad=True) ## torch zeros #
    
    def print_grads(self, ): ### print grads here ###
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
        
    
    def set_state(self, name_to_state):
        self.joint.set_state(name_to_state=name_to_state)
        for child_link in self.children:
            child_link.set_state(name_to_state)
            
            
    def set_state_via_vec(self, state_vec): # 
        self.joint.set_state_via_vec(state_vec)
        for child_link in self.children:
            child_link.set_state_via_vec(state_vec)
        # if self.joint_idx >= 0:
        #     self.state = state_vec[self.joint_idx]
    
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
            cur_link.get_visual_pts_list(visual_pts_list)
            
    
        
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
        # print(f"parent_rot_mtx: {self.parent_rot_mtx.grad}")
        # print(f"parent_trans_vec: {self.parent_trans_vec.grad}")
        # print(f"curr_rot_mtx: {self.curr_rot_mtx.grad}")
        # print(f"curr_trans_vec: {self.curr_trans_vec.grad}")
        # print(f"tot_rot_mtx: {self.tot_rot_mtx.grad}")
        # print(f"tot_trans_vec: {self.tot_trans_vec.grad}")
        # print(f"Joint")
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
            cur_link.get_visual_pts_list(visual_pts_list)
            
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
    def get_tot_transformed_joints(self, transformed_joints):
        for cur_link in self.children: # 
            transformed_joints = cur_link.get_tot_transformed_joints(transformed_joints)
        return transformed_joints
    
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
        # parent_rot_mtx, parent_trans_vec
        for cur_link in self.children:
            cur_link.compute_transformation_via_state_vecs(state_vals, cur_link.parent_rot_mtx, cur_link.parent_trans_vec, visual_pts_list)
        return visual_pts_list
    
    # get_visual_pts_rgba_values(self, pts_rgba_vals_list):
    def get_visual_pts_rgba_values(self, pts_rgba_vals_list):
        for cur_link in self.children:
            cur_link.get_visual_pts_rgba_values(pts_rgba_vals_list)
        return pts_rgba_vals_list ## compute pts rgba vals list ##

def parse_nparray_from_string(strr, args):
    vals = strr.split(" ")
    vals = [float(val) for val in vals]
    vals = np.array(vals, dtype=np.float32)
    vals = torch.from_numpy(vals).float()
    ## vals ##
    vals = nn.Parameter(vals.cuda(args.th_cuda_idx), requires_grad=True)
    
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






if __name__=='__main__':
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
    
