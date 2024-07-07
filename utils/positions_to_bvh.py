
import numpy as np
import os
from data_loaders.humanml.common.quaternion import *
from utils.paramUtil import *
from data_loaders.humanml.common.skeleton import Skeleton

kinematic_chain_for_original = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
output_kinematic_chain = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 9, 12], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
chain_start_idx = [0, 0, 0, 9, 9]
reference_face_joint_idx=[1, 2, 16, 17] # lHip, rHip, lShoulder, rShoulder
appear_sequence_22 = [0, 2, 5, 8, 11, 1, 4, 7, 10, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
appear_sequence_20 = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 9, 12, 13, 16, 18, 20, 14, 17, 19, 21]
reference_offset = {  'RootJoint': [0,0,0],
                        'lHip':[0.100000, -0.051395, 0.000000], 
                        'rHip': [-0.100000, -0.051395, 0.000000],
                        'pelvis_lowerback': [0,0.093605,0],
                        'lKnee': [0.000000, -0.410000, 0.000000],
                        'rKnee': [0.000000, -0.410000, 0.000000],
                        'lowerback_upperwaist': [0,0.1,0],
                        'lAnkle': [0,-0.39,0],
                        'rAnkle': [0,-0.39,0],
                        'lowerback_torso': [0,0.1,0],
                        'lToeJoint': [0.000000, -0.050000, 0.130000],
                        'rToeJoint': [0.000000, -0.050000, 0.130000],
                        'torso_head': [0.000000, 0.282350, 0.000000],
                        'lTorso_Clavicle': [0.001000, 0.157500, 0.000000], 
                        'rTorso_Clavicle': [-0.001000, 0.157500, 0.000000],
                        'head': [0.000000, 0.192650, 0.000000], 
                        'lShoulder': [0.117647, 0.000000, 0.000000],
                        'rShoulder': [-0.117647, 0.000000, 0.000000],
                        'lElbow': [0.245000, 0.000000, 0.000000],
                        'rElbow': [-0.245000, 0.000000, 0.000000],
                        'lWrist':[0.24000, 0.000000, 0.000000],
                        'rWrist': [-0.24000, 0.000000, 0.000000],
                    }
end_point_offset = {
    "lToeJoint": [0.010000, 0.002000, 0.060000],
    "rToeJoint": [-0.010000, 0.002000, 0.060000],
    "lWrist": [0.116353, -0.002500, 0.000000],
    "rWrist": [-0.116353, -0.002500, 0.000000],
    "head": [0.000000, 0.190000, 0.000000],
    "torso_head": [0.000000, 0.190000, 0.000000],

}
reference_index_to_name = [
    'RootJoint',
    'lHip',
    'rHip',
    'pelvis_lowerback',
    'lKnee',
    'rKnee',
    'lowerback_upperwaist',
    'lAnkle',
    'rAnkle',
    'lowerback_torso',
    'lToeJoint',
    'rToeJoint',
    'torso_head',
    'lTorso_Clavicle',
    'rTorso_Clavicle',
    'head',
    'lShoulder',
    'rShoulder',
    'lElbow',
    'rElbow',
    'lWrist',
    'rWrist',
]
# 0: RootJoint
# 1: lHip
# 2: rHip
# 3: pelvis_lowerback
# 4: lKnee
# 5: rKnee
# 6: lowerback_upperwaist
# 7: lAnkle
# 8: rAnkle
# 9: lowerback_torso
# 10: lToeJoint
# 11: rToeJoint
# 12: torso_head
# 13: lTorso_Clavicle
# 14: rTorso_Clavicle
# 15: head
# 16: lShoulder
# 17: rShoulder
# 18: lElbow
# 19: rElbow
# 20: lWrist
# 21: rWrist

class Humanml2bvh:
    def __init__(self, device):
        self.device = device
        self.skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, self.device)

    def turn_263_to_bvh(self, data_263, save_dir):
        positions = recover_from_rot(data_263, 22, self.skeleton)
        pure_position_to_bvh(positions, save_dir)

def get_parent_positions(positions):
    # positions: (num_frames, num_joints, 3) or (num_joints, 3)
    # kinematic_chain: list of list of indices
    # root position is returned as (0, 0, 0)
    num_joints = positions.shape[-2]
    original_shape = positions.shape
    positions = positions.reshape(-1, num_joints, 3)
    parent_positions = torch.zeros_like(positions)

    for chain in kinematic_chain_for_original:
        parent_positions[:, chain[1:]] = positions[:, chain[:-1]]
    return parent_positions.reshape(original_shape)


def get_relative_positions(positions, kinematic_chain):
    # positions: (num_frames, num_joints, 3) or (num_joints, 3)
    # kinematic_chain: list of list of indices
    parent_positions = get_parent_positions(positions)
    return positions - parent_positions

def get_root_shoulder_rotation(positions, face_joint_idx):
    device = positions.device
    assert len(face_joint_idx) == 4
    '''Get Forward Direction'''
    num_joints = positions.shape[-2]
    original_shape = positions.shape
    positions = positions.reshape(-1, num_joints, 3)

    l_hip, r_hip, l_shoulder, r_shoulder = face_joint_idx
    across1 = positions[:, l_hip] - positions[:, r_hip]
    across2 = positions[:, l_shoulder] - positions[:, r_shoulder]
    across1 = across1 + across2
    across1 = across1 / torch.sqrt((across1**2).sum(axis=-1)).unsqueeze(1)
    across2 = across2 / torch.sqrt((across2**2).sum(axis=-1)).unsqueeze(1)

    forward_root = torch.cross(across1, torch.tensor([[0, 1.0, 0]]).to(device), axis=-1)
    forward_shoulder = torch.cross(across2, torch.tensor([[0, 1.0, 0]]).to(device), axis=-1)
    target = torch.tensor([[0, 0, 1.0]]).to(device).expand(len(forward_root), -1)

    root_quat = qbetween(target, forward_root)
    shoulder_quat = qbetween(target, forward_shoulder)

    return root_quat, shoulder_quat


def position_to_quat(positions, raw_offset, kinematic_chain, face_joint_idx, index_to_name, fps):
    device = positions.device
    num_frames = positions.shape[0]
    num_joints = positions.shape[-2]

    root_quat, shoulder_quat = get_root_shoulder_rotation(positions, face_joint_idx)
    relative_positions = get_relative_positions(positions, kinematic_chain)
    joints_quat = torch.zeros((num_frames, num_joints, 4)).to(device)
    joints_quat[:, 0] = root_quat

    for chain in kinematic_chain:
        R = joints_quat[:, chain[0]]
        for i in range(1, len(chain)-1):
            if i == 1 and chain[0] == 9:
                R = shoulder_quat

            joint_idx = chain[i]
            child_joint_idx = chain[i+1]
            child_joint_name = index_to_name[child_joint_idx]
            # the rotation of chain[i] is actually the rotation of of relative position of chain[i+1] and chain[i]
            u = torch.tensor(raw_offset[child_joint_name]).to(device).unsqueeze(0).expand(num_frames, -1)
            u = u / torch.sqrt((u**2).sum(axis=-1)).unsqueeze(1)

            v = relative_positions[:, chain[i+1]]
            v = v / torch.sqrt((v**2).sum(axis=-1)).unsqueeze(1)
    

            rot_u_v = qbetween(u, v) if joint_idx != 9 else shoulder_quat
            R_loc = qmul(qinv(R), rot_u_v)

            joints_quat[:, chain[i]] = R_loc
            R = qmul(R, R_loc)

            # find the rotation for head. Really ugly.
            if joint_idx == 9 and child_joint_idx == 12:
                joint_idx = 12
                child_joint_idx = 15
                u = torch.tensor(raw_offset[index_to_name[child_joint_idx]]).to(device).unsqueeze(0).expand(num_frames, -1)
                u = u / torch.sqrt((u**2).sum(axis=-1)).unsqueeze(1)

                v = relative_positions[:, child_joint_idx]
                v = v / torch.sqrt((v**2).sum(axis=-1)).unsqueeze(1)

                rot_u_v = qbetween(u, v)
                R_loc = qmul(qinv(R), rot_u_v)

                joints_quat[:, joint_idx] = R_loc
        
    return joints_quat

def position_to_euler(positions, raw_offset, kinematic_chain, face_joint_idx, index_to_name, fps):
    joint_quat = position_to_quat(positions, raw_offset, kinematic_chain, face_joint_idx, index_to_name, fps)
    joint_euler = qeuler(joint_quat, order='xyz')
    return joint_euler

def write_hiearchy(f, chain, kinematic_chain, names, offsets, level):
    joint_idx = chain[0]
    joint_name = names[joint_idx]
    offset = offsets[joint_name]
    f.write("\t" * level + f"JOINT {joint_name}\n")
    f.write("\t" * level + "{\n")
    f.write("\t" * (level + 1) + f"OFFSET {offset[0]} {offset[1]} {offset[2]}\n")
    f.write("\t" * (level + 1) + "CHANNELS 3 Xrotation Yrotation Zrotation\n")

    if len(chain) == 1:

        f.write("\t" * (level+1) + "End Site\n")
        f.write("\t" * (level+1) + "{\n")
        f.write("\t" * (level+2) + f"OFFSET {end_point_offset[joint_name][0]} {end_point_offset[joint_name][1]} {end_point_offset[joint_name][2]}\n")
        f.write("\t" * (level+1) + "}\n")

    elif len(chain) > 1:
        write_hiearchy(f, chain[1:], kinematic_chain, names, offsets, level+1)

    if joint_idx == 9:
        write_hiearchy(f, [13, 16, 18, 20], kinematic_chain, names, offsets, level+1)
        write_hiearchy(f, [14, 17, 19, 21], kinematic_chain, names, offsets, level+1)

    f.write("\t" * level + "}\n")
        
def write_bvh_head(f, kinematic_chain, names, offsets, fps, num_frames):
    f.write("HIERARCHY\n")
    f.write("ROOT RootJoint\n")
    f.write("{\n")
    f.write("\tOFFSET 0.0 0.0 0.0\n")
    f.write("\tCHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n")
    for chain in kinematic_chain[:3]:
        write_hiearchy(f, chain[1:], kinematic_chain, names, offsets, 1)
    f.write("}\n")
    f.write("MOTION\n")
    f.write(f"Frames: {num_frames}\n")
    f.write(f"Frame Time: {1/fps}\n")

def write_bvh_motion(f, appear_sequence, positions, raw_offset, kinematic_chain, face_joint_idx, index_to_name, fps):
    joint_euler = position_to_euler(positions, raw_offset, kinematic_chain, face_joint_idx, index_to_name, fps)
    num_frames = positions.shape[0]
    num_joints = positions.shape[-2]
    root_translation = positions[:, 0]

    for i in range(num_frames):
        f.write(f"{root_translation[i][0]} {root_translation[i][1]} {root_translation[i][2]} ")
        f.write(f"{joint_euler[i][0][0]} {joint_euler[i][0][1]} {joint_euler[i][0][2]} ")
        for j in appear_sequence:
            if j == 0:
                continue
            
            f.write(f"{joint_euler[i][j][0]} {joint_euler[i][j][1]} {joint_euler[i][j][2]} ")
        f.write("\n")
    
def dump_zero_motion(f, num_frames, num_joints):
    for i in range(num_frames):
        f.write(f"0.0 0.0 0.0 ")
        f.write(f"0.0 0.0 0.0 ")
        for j in range(num_joints-1):
            f.write(f"0.0 0.0 0.0 ")
        f.write("\n")

def dump_root_motion(f, num_frames, num_joints, positions, root_rot):
    for i in range(num_frames):
        f.write(f"{positions[i][0][0]} {positions[i][0][1]} {positions[i][0][2]} ")
        f.write(f"{root_rot[i][0]} {root_rot[i][1]} {root_rot[i][2]} ")
        for j in range(num_joints-1):
            f.write(f"0.0 0.0 0.0 ")
        f.write("\n")

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def pure_position_to_bvh(positions, save_dir="temp.bvh"):
    kinematic_chain = output_kinematic_chain
    sequence = appear_sequence_20
    root_rot, shoulder_rot = get_root_shoulder_rotation(positions, reference_face_joint_idx)
    root_euler = qeuler(root_rot, order='xyz')
    num_frames = positions.shape[0]
    with open(save_dir, "w") as f:
        write_bvh_head(f, kinematic_chain, reference_index_to_name, reference_offset, 50, num_frames)
        write_bvh_motion(f, sequence, positions, reference_offset, kinematic_chain, reference_face_joint_idx, reference_index_to_name, 20)