
import math
# import torch
# from ..utils import Timer
import numpy as np
# import torch.nn.functional as F
import os

import argparse

from xml.etree.ElementTree import ElementTree

import trimesh
# import List
# class link; joint; body
### 

## calculate transformation to the frame ##


def rotation_matrix_from_axis_angle(axis, angle): # rotation_matrix_from_axis_angle -> 
        sin_ = np.sin(angle ) #  ti.math.sin(angle)
        cos_ = np.cos(angle ) #  ti.math.cos(angle)
        
        # sin_ = np.sin(-np.pi / 4 ) #  ti.math.sin(angle) ## ti.math.sin ##
        # cos_ = np.cos(-np.pi / 4  ) #  ti.math.cos(angle) ## ti.math.sin ##
        u_x, u_y, u_z = axis[0], axis[1], axis[2]
        u_xx = u_x * u_x
        u_yy = u_y * u_y
        u_zz = u_z * u_z
        u_xy = u_x * u_y
        u_xz = u_x * u_z
        u_yz = u_y * u_z
        rot_mtx = np.stack(
            [np.array([cos_ + u_xx * (1 - cos_), u_xy * (1. - cos_) + u_z * sin_, u_xz * (1. - cos_) - u_y * sin_], dtype=np.float32), 
             np.array([u_xy * (1. - cos_) - u_z * sin_, cos_ + u_yy * (1. - cos_), u_yz * (1. - cos_) + u_x * sin_], dtype=np.float32), 
             np.array([u_xz * (1. - cos_) + u_y * sin_, u_yz * (1. - cos_) - u_x * sin_, cos_ + u_zz * (1. - cos_)], dtype=np.float32)
            ], axis=-1 ### np stack 
        )
        # rot_mtx_numpy = rot_mtx.to_numpy()
        # rot_mtx_at_rot_mtx = rot_mtx @ rot_mtx.transpose() ## 
        # print(rot_mtx_at_rot_mtx)
        return rot_mtx

## joint name = "joint3" ##
# <joint name="joint3" type="revolute" axis="0.000000 0.000000 -1.000000" pos="4.689700 -4.425000 0.000000" quat="1 0 0 0" frame="WORLD" damping="1e7"/>
class Joint:
    def __init__(self, name, joint_type, axis, pos, quat, frame, damping) -> None:
        self.name = name
        self.type = joint_type
        self.axis = axis
        print(f"joint_axis: {self.axis}, joint name: {self.name}")
        self.pos = pos
        self.quat = quat
        self.frame = frame
        self.damping = damping
        
        
        
        self.state = 0.
        self.rot_mtx = np.eye(3, dtype=np.float32)
        self.trans_vec = np.zeros((3,), dtype=np.float32)
        
        
    def compute_transformation(self,):
        # use the state to transform them # # transform # ## transform the state ##
        # use the state to transform them # # transform them for the state #
        if self.type == "revolute":
            # print(f"computing transformation matrices with axis: {self.axis}, state: {self.state}")
            # rotation matrix from the axis angle # 
            rot_mtx = rotation_matrix_from_axis_angle(self.axis, self.state)
            # rot_mtx(p - p_v) + p_v -> rot_mtx p - rot_mtx p_v + p_v
            trans_vec = self.pos - np.matmul(rot_mtx, self.pos.reshape(3, 1)).reshape(3)
            # trans_vec = self.pos - np.matmul(self.pos.reshape( 1, 3), rot_mtx).reshape(3)
            self.rot_mtx = np.copy(rot_mtx)
            self.trans_vec = np.copy(trans_vec)
        else:
            ### TODO: implement transformations for joints in other types ###
            pass
    
    def set_state(self, name_to_state):
        if self.name in name_to_state:
            # self.state = name_to_state["name"]
            self.state = name_to_state[self.name]




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
    def __init__(self, name, body_type, filename, pos, quat, transform_type, density, mu, rgba, radius) -> None:
        self.name = name
        self.body_type = body_type
        ### for mesh object ###
        self.filename = filename

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
        
        self.visual_pts = None
        
        self.get_visual_counterparts()
        
    
    def get_visual_counterparts(self,):
        ### TODO: implement this for visual counterparts 
        ######## get body type ###
        if self.body_type == "sphere":
            filename = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo/meshes/18.obj"
        else:
            filename = self.filename
            rt_asset_path = "/home/xueyi/diffsim/DiffHand/assets" ### assets folder ###
            filename = os.path.join(rt_asset_path, filename)
        body_mesh = trimesh.load(filename, process=False)
        verts = np.array(body_mesh.vertices)
        faces = np.array(body_mesh.faces, dtype=np.long)
        
        self.visual_pts_ref = np.copy(verts) ## verts ##
        self.visual_faces_ref = np.copy(faces) ## faces 
        # self.visual_pts_ref # 
        
        

    def transform_visual_pts(self, rot_mtx, trans_vec):
        # rot_mtx: 3 x 3 numpy array 
        # trans_vec: 3 numpy array
        # print(f"transforming body with rot_mtx: {rot_mtx} and trans_vec: {trans_vec}")
        self.visual_pts = np.matmul(rot_mtx, self.visual_pts_ref.T).T + trans_vec.reshape(1, 3)
        
        # self.visual_pts = np.matmul(self.visual_pts_ref, rot_mtx) + trans_vec.reshape(1, 3)
    
    
    # get the visual counterparts of the boyd mesh or elements #
    
    # xyz attribute ## ## xyz attribute #

# use get_name_to_visual_pts_faces to get the transformed visual pts and faces #
class Link: 
    def __init__(self, name, joint: Joint, body: Body, children) -> None:
        
        self.joint = joint
        self.body = body
        self.children = children
        self.name = name
        
        # joint #
        self.parent_rot_mtx = np.eye(3, dtype=np.float32)
        self.parent_trans_vec = np.zeros((3,), dtype=np.float32)
        self.curr_rot_mtx = np.eye(3, dtype=np.float32)
        self.curr_trans_vec = np.zeros((3,), dtype=np.float32) ### trans vec ##
        # 
        self.tot_rot_mtx = np.eye(3, dtype=np.float32)
        self.tot_trans_vec = np.zeros((3,), dtype=np.float32)
    
    def set_state(self, name_to_state):
        self.joint.set_state(name_to_state=name_to_state)
        for child_link in self.children:
            child_link.set_state(name_to_state)
    
    def compute_transformation(self,):
        self.joint.compute_transformation()
        self.curr_rot_mtx = np.copy(self.joint.rot_mtx)
        self.curr_trans_vec = np.copy(self.joint.trans_vec)
        
        ## (p rot_c + trans) rot_p + trans_p = p rot_c rot_p + trans rot_p
        # rot_p (rot_c p + trans_c) + trans_p # 
        # rot_p rot_c p + rot_p trans_c + trans_p #
        tot_parent_rot_mtx = np.matmul(self.parent_rot_mtx, self.curr_rot_mtx)
        tot_parent_trans_vec = np.matmul(self.parent_rot_mtx, self.curr_trans_vec.reshape(3, 1)).reshape(3) + self.parent_trans_vec
        
        # tot_parent_rot_mtx = np.matmul(self.curr_rot_mtx, self.parent_rot_mtx)
        # tot_parent_trans_vec = np.matmul(self.curr_trans_vec.reshape(1, 3), self.parent_rot_mtx).reshape(3) + self.parent_trans_vec
        
        self.tot_rot_mtx = np.copy(tot_parent_rot_mtx)
        self.tot_trans_vec = np.copy(tot_parent_trans_vec)
        
        for cur_link in self.children:
            cur_link.parent_rot_mtx = np.copy(tot_parent_rot_mtx) ### set children parent rot mtx and the trans vec #
            cur_link.parent_trans_vec = np.copy(tot_parent_trans_vec) ## 
            cur_link.compute_transformation() ## compute self's transformations
            
    def get_name_to_visual_pts_faces(self, name_to_visual_pts_faces):
        # transform_visual_pts #
        self.body.transform_visual_pts(rot_mtx=self.tot_rot_mtx, trans_vec=self.tot_trans_vec)
        name_to_visual_pts_faces[self.body.name] = {"pts": self.body.visual_pts, "faces": self.body.visual_faces_ref}
        for cur_link in self.children:
            cur_link.get_name_to_visual_pts_faces(name_to_visual_pts_faces) ## transform the pts faces 
            
    
        
    
    
    
        
class Robot:
    def __init__(self, children_links) -> None:
        self.children = children_links
        
    def set_state(self, name_to_state):
        for cur_link in self.children:
            cur_link.set_state(name_to_state)
    
    def compute_transformation(self,):
        for cur_link in self.children:
            cur_link.compute_transformation()
    
    def get_name_to_visual_pts_faces(self, name_to_visual_pts_faces):
        for cur_link in self.children:
            cur_link.get_name_to_visual_pts_faces(name_to_visual_pts_faces)
        

def parse_nparray_from_string(strr):
    vals = strr.split(" ")
    vals = [float(val) for val in vals]
    vals = np.array(vals, dtype=np.float32)
    return vals


def parse_link_data(link):
    
    link_name = link.attrib["name"]
    print(f"parsing link: {link_name}")
    
    joint = link.find("./joint")
    
    joint_name = joint.attrib["name"]
    joint_type = joint.attrib["type"]
    if joint_type in ["revolute"]: ## a general xml parser here? 
        axis = joint.attrib["axis"]
        axis = parse_nparray_from_string(axis)
    else:
        axis = None
    pos = joint.attrib["pos"] # 
    pos = parse_nparray_from_string(pos)
    quat = joint.attrib["quat"]
    quat = parse_nparray_from_string(quat)
    
    try:
        frame = joint.attrib["frame"]
    except:
        frame = "WORLD"
    
    if joint_type not in ["fixed"]:
        damping = joint.attrib["damping"]
        damping = float(damping)
    else:
        damping = 0.0
    
    cur_joint = Joint(joint_name, joint_type, axis, pos, quat, frame, damping)
    
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
    pos = parse_nparray_from_string(pos)
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
        rgba = parse_nparray_from_string(rgba)
    except:
        rgba = np.zeros((4,), dtype=np.float32)
    
    cur_body = Body(body_name, body_type, filename, pos, quat, transform_type, density, mu, rgba, radius)
    
    children_link = []
    links = link.findall("./link")
    for child_link in links:
        cur_child_link = parse_link_data(child_link)
        children_link.append(cur_child_link)
    
    link_name = link.attrib["name"]
    link_obj = Link(link_name, joint=cur_joint, body=cur_body, children=children_link)
    return link_obj
    
    

# (R_p ( (R_c p.T + t_c).T ).T + t_p).T
# 

def parse_data_from_xml(xml_fn):
    
    tree = ElementTree()
    tree.parse(xml_fn)
    
    robots = tree.findall("./robot")
    i_robot = 0
    tot_robots = []
    for cur_robot in robots:
        # print(f"Getting robot: {i_robot}")
        i_robot += 1
        cur_links = cur_robot.findall("./link")
        # i_link = 0
        cur_robot_links = []
        for cur_link in cur_links: ## child of the link ##
            ### a parse link util -> the child of the link is composed of (the joint; body; and children links (with children or with no child here))
            cur_link_name = cur_link.attrib["name"]
            # print(f"Getting link: {i_link} with name: {cur_link_name}")
            # i_link += 1 ## 
            cur_robot_links.append(parse_link_data(cur_link))
        cur_robot_obj = Robot(cur_robot_links)
        tot_robots.append(cur_robot_obj)
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


def get_name_to_state_from_vec(states_vec):
    # tot_states = states_str.split(" ")
    # tot_states = [float(cur_state) for cur_state in tot_states]
    joint_name_to_state = {}
    for i in range(states_vec.shape[0]):
        cur_joint_name = f"joint{i + 1}"
        cur_joint_state = states_vec[i].item()
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



if __name__=='__main__':
    xml_fn =   "/home/xueyi/diffsim/DiffHand/assets/hand_sphere.xml"
    tot_robots = parse_data_from_xml(xml_fn=xml_fn)
    # tot_robots = 
    
    active_optimized_states = """-0.00025872 -0.00025599 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025296 -0.00022881 -0.00024449 -0.0002549 -0.00025694 -0.00024656 -0.00025556 0. 0.0049 0."""
    active_optimized_states = """-1.10617972 -1.10742263 -1.06198363 -1.03212746 -1.05429142 -1.08617289 -1.05868192 -1.01624365 -1.04478191 -1.08260959 -1.06719107 -1.04082455 -1.05995886 -1.08674006 -1.09396691 -1.08965532 -1.10036577 -10.7117466 -3.62511998 1.49450353"""
    # active_goal_optimized_states = """-1.10617972 -1.10742263 -1.0614858 -1.03189609 -1.05404354 -1.08610468 -1.05863293 -1.0174248 -1.04576456 -1.08297396 -1.06719107 -1.04082455 -1.05995886 -1.08674006 -1.09396691 -1.08965532 -1.10036577 -10.73396897 -3.68095432 1.50679285"""
    active_optimized_states = """-0.42455298 -0.42570447 -0.40567708 -0.39798589 -0.40953955 -0.42025055 -0.37910662 -0.496165 -0.37664644 -0.41942727 -0.40596508 -0.3982109 -0.40959847 -0.42024905 -0.41835001 -0.41929961 -0.42365131 -1.18756073 -2.90337822 0.4224685"""
    active_optimized_states = """-0.42442816 -0.42557961 -0.40366201 -0.3977891 -0.40947627 -0.4201424 -0.3799285 -0.3808375 -0.37953552 -0.42039598 -0.4058405 -0.39808804 -0.40947487 -0.42012458 -0.41822534 -0.41917521 -0.4235266 -0.87189658 -1.42093761 0.21977979"""
    
    active_optimized_states = """-0.63532552 -0.63676849 -0.6095192 -0.59471904 -0.6111259 -0.62759359 -0.64914484 -0.865102 -0.71103336 -0.61231947 -0.61117582 -0.59593803 -0.61150941 -0.62759057 -0.62715831 -0.62690961 -0.63339176 -6.93218524 -7.56016252 1.4064949"""
    # ./save_res/hand_sphere_demo_target/rendered_images_optim_False_passive_False_nv_7_test.npy
    
    optimized_q_states = "./save_res/hand_sphere_demo_target/res_q_iter_79.npy"
    optimized_q_states = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target/res_q_iter_39.npy"
    optimized_q_states = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target/res_q_iter_75.npy"
    optimized_q_states = "./save_res/hand_sphere_demo_target/res_q_iter_49.npy"
    optimized_q_states = "./save_res/hand_sphere_demo_target/res_q_iter_78.npy"
    # optimized_q_states = "/home/xueyi/diffsim/DiffHand/examples/save_res/hand_sphere_demo_target/active_only_optim_res_q_iter_95.npy"
    optimized_q_states = np.load(optimized_q_states, allow_pickle=True)
    
    
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
    
    ## get name to state from str ##
    
    nn_active_states = 17
    state_idx = -1
    print(f"optimized_states: {optimized_q_states[state_idx]}")
    optimized_states = optimized_q_states[state_idx, :nn_active_states]
    
    
    # 
    # state idx 
    for i_state in range(optimized_q_states.shape[0]):
        cur_optimized_states = optimized_q_states[i_state, :nn_active_states]
        cur_optimized_states = get_name_to_state_from_vec(cur_optimized_states)
        
        ## set state ##
        active_robot.set_state(cur_optimized_states)
        active_robot.compute_transformation() ### compute transformation ## 
        name_to_visual_pts_surfaces = {}
        active_robot.get_name_to_visual_pts_faces(name_to_visual_pts_surfaces)
        
        ## optimized states ### optimized states ##
        tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"active_goal_res_with_optimized_states_goal_n8_istate_{i_state}.npy")
        np.save(tmp_visual_res_sv_fn, name_to_visual_pts_surfaces)
        print(f"tmp visual res with optimized states saved to {tmp_visual_res_sv_fn}")
    
    # optimized_states =  get_name_to_state_from_vec(optimized_states)
    
    # active_robot.set_state(optimized_states) ## set the state ##
    # active_robot.compute_transformation()
    # name_to_visual_pts_surfaces = {}
    # active_robot.get_name_to_visual_pts_faces(name_to_visual_pts_surfaces)
    # print(len(name_to_visual_pts_surfaces))
    # # sv_res_rt = "/home/xueyi/diffsim/DiffHand/examples/save_res"
    # # sv_res_rt = os.path.join(sv_res_rt, "load_utils_test") ## load utils test ##
    # # os.makedirs(sv_res_rt, exist_ok=True) ## load utils test ##
    
    # # tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"res_with_optimized_states.npy")
    # # tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"active_ngoal_res_with_optimized_states_goal_n8.npy")
    # tmp_visual_res_sv_fn = os.path.join(sv_res_rt, f"active_goal_res_with_optimized_states_goal_n8.npy")
    # np.save(tmp_visual_res_sv_fn, name_to_visual_pts_surfaces)
    # print(f"tmp visual res with optimized states saved to {tmp_visual_res_sv_fn}")
    
