'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''
import numpy as np
import torch
import typing
from utils import pytorch_utils as ptu
# import VclSimuBackend
# try:
#     from VclSimuBackend.pymotionlib.Utils import quat_product
#     from VclSimuBackend.Common.MathHelper import MathHelper
#     from VclSimuBackend.ODESim import BodyInfoState
# except ImportError:
#     quat_product = VclSimuBackend.pymotionlib.Utils.quat_product
#     MathHelper = VclSimuBackend.Common.MathHelper
#     BodyInfoState = VclSimuBackend.ODESim.BodyInfoState
    
# from utils.diff_quat import *
# import utils.diff_quat as DiffRotation
# import DiffRotation as DiffRotation # a cuda speed up, but not significant...
#----------------------------------State Utils----------------------------------------------#

# @torch.jit.script
# def broadcast_quat_apply(q: torch.Tensor, vec3: torch.Tensor):
#     t = 2 * torch.cross(q[..., :3], vec3, dim=-1)
#     xyz: torch.Tensor = vec3 + q[..., 3, None] * t + torch.linalg.cross(q[..., :3], t, dim=-1)
#     return xyz
# # 
# # @torch.jit.script
# def broadcast_quat_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
#     """
#     multiply 2 quaternions. p.shape == q.shape
#     """
    
#     w: torch.Tensor = p[..., 3:4] * q[..., 3:4] - torch.sum(p[..., :3] * q[..., :3], dim=-1, keepdim=True)
#     xyz: torch.Tensor = (
#                 p[...,3,None] * q[..., :3] + q[..., 3, None] * p[..., :3] + torch.linalg.cross(p[..., :3], q[..., :3], dim=-1))

#     return torch.cat([xyz, w], dim=-1)

# @torch.jit.script
def broadcast_quat_apply(q: torch.Tensor, vec3: torch.Tensor):
    vec_1 = q[..., :3]
    vec_2 = vec3
    vec_1 = vec_1.repeat(1, vec_2.shape[1], 1)
    if vec_2.shape[0] < vec_1.shape[0]:
        vec_2 = vec_2.repeat(vec_1.shape[0]//vec_2.shape[0], 1, 1)
    t = 2 * torch.cross(vec_1, vec_2, dim=-1)
    # t = 2 * torch.cross(q[..., :3], vec3, dim=-1)
    xyz: torch.Tensor = vec3 + q[..., 3, None] * t + torch.cross(vec_1, t, dim=-1)
    # xyz: torch.Tensor = vec3 + q[..., 3, None] * t + torch.cross(q[..., :3], t, dim=-1)
    return xyz


# @torch.jit.script
def broadcast_quat_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    multiply 2 quaternions. p.shape == q.shape
    """
    
    w: torch.Tensor = p[..., 3:4] * q[..., 3:4] - torch.sum(p[..., :3] * q[..., :3], dim=-1, keepdim=True)
    
    vec_1 = p[..., :3]
    vec_2 = q[..., :3]
    vec_1 = vec_1.repeat(1, vec_2.shape[1], 1)
    if vec_2.shape[0] < vec_1.shape[0]:
        vec_2 = vec_2.repeat(vec_1.shape[0]//vec_2.shape[0], 1, 1)
    cross_vec1_vec2 = torch.cross(vec_1, vec_2, dim=-1)
    
    # xyz: torch.Tensor = (
    #             p[...,3,None] * q[..., :3] + q[..., 3, None] * p[..., :3] + torch.cross(p[..., :3], q[..., :3], dim=-1))
    xyz: torch.Tensor = (
                p[...,3,None] * q[..., :3] + q[..., 3, None] * p[..., :3] + cross_vec1_vec2)
    

    return torch.cat([xyz, w], dim=-1)



def resample(old_pos, old_rot, cur_pos, cur_rot, dt):
    '''
        resample velocity and angular velocity to dt using backward finite difference
    '''
    vel = (cur_pos - old_pos)/dt
    sign = np.sign((cur_rot * old_rot).sum(axis=-1, keepdims=True))
    qd = (cur_rot * sign - old_rot)/dt
    q_conj = old_rot.copy()
    q_conj[..., :3] *= -1
    avel = 2 * quat_product(qd.reshape(-1,4), q_conj)
    return vel, avel[...,:3]   

def character_pos_rot(character):
    pos = character.body_info.get_body_pos()
    rot = character.body_info.get_body_quat()
    rot = MathHelper.flip_quat_by_w(rot)
    return pos, rot

def character_state(character, old_state = None, dt = None):
    '''
        return full state: {pos, rot, vel, avel}
        :param old_state: if old state is not None, it will try to recompute avel 
        and vel according to dt, otherwise it will just return current vel and avel 
    '''
    pos, rot = character_pos_rot(character)
    if old_state is not None:
        # recompute vel
        assert dt is not None
        old_pos, old_rot = old_state[:,:3], old_state[:,3:7]
        vel, avel = resample(old_pos, old_rot, pos, rot, dt)
    else:
        vel = character.body_info.get_body_velo()
        avel = character.body_info.get_body_ang_velo()
    state = np.concatenate([pos, rot, vel, avel], axis=-1, dtype=np.float32)
    return state

def state_to_BodyInfoState(state):
    res = BodyInfoState.BodyInfoState()
    res.pos = np.ascontiguousarray(state[..., 0:3].flatten(), dtype=np.float64)
    res.quat = np.ascontiguousarray(state[..., 3:7].flatten(), dtype=np.float64)
    res.linear_vel = np.ascontiguousarray(state[..., 7:10].flatten(), dtype=np.float64)
    res.angular_vel = np.ascontiguousarray(state[..., 10:13].flatten(), dtype=np.float64)
    res.rot = np.ascontiguousarray(Rotation.from_quat(state[...,3:7].reshape(-1,4)).as_matrix().flatten(), dtype=np.float64)
    return res

def decompose_state(state):
    assert state.shape[-1] ==13
    
    return state[...,0:3], state[...,3:7], state[...,7:10], state[...,10:13]

#---------------------------Observation Utils--------------------------------------------#
def state2ob_old(states):
    '''
    :param states: full state
    :return: observation { local{pos, rot, vel, avel}, height, up_dir}
    '''

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"
    batch_size = states.shape[0]
    # assert states.shape == (batch_size, 20, 13)

    batch_size, num_body, _ = states.shape

    pos = states[..., 0:3].view(-1, 3)
    rot = states[..., 3:7].view(-1, 4)
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)

    root_pos = torch.tile(states[:, 0, 0:3], [1, 1, num_body]).view(-1, 3)
    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    root_rot_inv = quat_inv(root_rot)

    local_pos = quat_apply(root_rot_inv, pos - root_pos).view(batch_size, -1)
    local_vel = quat_apply(root_rot_inv, vel).view(batch_size, -1)
    local_rot = quat_to_matrix(flip_quat_by_w(quat_multiply(root_rot_inv, rot
                                             )))
    # Why here must be reshape ????
    local_rot = torch.transpose(local_rot, -2, -1)[:,:2,:].reshape(batch_size,-1)
    
    local_avel = quat_apply(root_rot_inv, avel).view(batch_size, -1)
    height = pos[..., 1].view(batch_size, -1)
    up_dir = torch.as_tensor([0, 1, 0]).view(-1, 3).tile([batch_size, 1]).float().to(states.device)
    local_up_dir = quat_apply(root_rot_inv[::num_body,:], up_dir).view(batch_size, -1)

    if batch_size == 1:
        local_pos = local_pos.flatten()
        local_rot = local_rot.flatten()
        local_vel = local_vel.flatten()
        local_avel = local_avel.flatten()
        height = height.flatten()
        local_up_dir = local_up_dir.flatten()
    return torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)

# @torch.jit.script
# def state2ob(states):
#     # needs pytorch >= 11.0
#     if len(states.shape) == 2:
#         states = states[None]
#     batch_size, num_body, _ = states.shape

    # pos = states[..., 0:3]
    # rot = states[..., 3:7]
    # vel = states[..., 7:10]
    # avel = states[..., 10:13]

    # root_pos = pos[:,0,:].view(-1,1,3)
    # root_rot_inv = quat_inv(rot[:,0,:].view(-1,1,4))
    # local_pos = broadcast_quat_apply(root_rot_inv, pos - root_pos ).view(batch_size, -1)
    # local_vel = broadcast_quat_apply(root_rot_inv, vel).view(batch_size, -1)
    # local_avel = broadcast_quat_apply(root_rot_inv, avel).view(batch_size, -1)
    # local_rot = flip_quat_by_w(broadcast_quat_multiply(root_rot_inv, rot
    #                                          )).view(-1,4)
    # local_rot = quat_to_vec6d(local_rot).view(batch_size,-1)
    # # Why here must be reshape ????
    
    # height = pos[..., 1].view(batch_size, -1)
    # up_dir = torch.as_tensor([0, 1, 0]).view(1,1, 3).float().to(states.device)
    # local_up_dir = broadcast_quat_apply(root_rot_inv, up_dir.view(1,1,3).float().to(root_rot_inv.device)).view(-1,3)

    # if batch_size == 1:
    #     local_pos = local_pos.flatten()
    #     local_rot = local_rot.flatten()
    #     local_vel = local_vel.flatten()
    #     local_avel = local_avel.flatten()
    #     height = height.flatten()
    #     local_up_dir = local_up_dir.flatten()
    # return torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)

# to the observations with more informatons? # 
def state2ob(states):
    # needs pytorch >= 11.0
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).to(ptu.device)
    flag = False
    frames, bs = 0, 0
    if len(states.shape) == 2:
        flag = True
        states = states[None]
    if len(states.shape) == 4:
        frames, bs, body_num, _ = states.shape
        states = states.reshape(frames * bs, body_num, -1)
    batch_size, num_body, _ = states.shape

    pos = states[..., 0:3]
    rot = states[..., 3:7]
    vel = states[..., 7:10]
    avel = states[..., 10:13]

    root_pos = pos[:,0,:].view(-1,1,3)
    root_rot_inv = quat_inv(rot[:,0,:].view(-1,1,4))
    local_pos = broadcast_quat_apply(root_rot_inv, pos - root_pos ).view(batch_size, -1)
    local_vel = broadcast_quat_apply(root_rot_inv, vel).view(batch_size, -1)
    local_avel = broadcast_quat_apply(root_rot_inv, avel).view(batch_size, -1)
    local_rot = flip_quat_by_w(broadcast_quat_multiply(root_rot_inv, rot
                                             )).view(-1,4)
    local_rot = quat_to_vec6d(local_rot).view(batch_size,-1)
    # Why here must be reshape ????
    
    height = pos[..., 1].view(batch_size, -1)
    up_dir = torch.as_tensor([0, 1, 0]).view(1,1, 3).float().to(states.device)
    local_up_dir = broadcast_quat_apply(root_rot_inv, up_dir.view(1,1,3).float().to(root_rot_inv.device)).view(-1,3)

    if flag:
        local_pos = local_pos.flatten()
        local_rot = local_rot.flatten()
        local_vel = local_vel.flatten()
        local_avel = local_avel.flatten()
        height = height.flatten()
        local_up_dir = local_up_dir.flatten()
    res = torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)
    if frames:
        res = res.reshape(frames, bs, -1)
    return res    
    
# add jit will be slower... why...?
# @torch.jit.script
def decompose_obs(obs):
    num_dim = obs.shape[-1]
    assert (num_dim - 3) % 16 == 0, "dim error"
    num_body = (num_dim - 3)//16
    pos = obs[...,0:3*num_body]
    rot = obs[...,3*num_body:9*num_body]
    vel = obs[...,9*num_body:12*num_body]
    avel = obs[...,12*num_body:15*num_body]
    height = obs[...,15*num_body:16*num_body]
    up_dir = obs[...,16*num_body:]
    return pos, rot, vel, avel, height, up_dir


def split_state(state):
    nn_finger_states = state.size(-1) - (3 + 3) - (3 + 4)
    # mano_trans = state[:, : 3]
    # mano_rot = state[:, 3: 3 + 3]
    # mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    # obj_rot = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 4]
    # obj_trans = state[:, 6 + nn_finger_states + 4: 6 + nn_finger_states + 4 + 3]
    
    mano_trans = state[:, : 3]
    mano_rot = state[:,  3: 3 + 3]
    mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    obj_trans = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 3]
    obj_rot = state[:, 6 + nn_finger_states + 3: 6 + nn_finger_states + 4 + 3] ## rotation quaternion matrix ##
    return mano_trans, mano_rot, mano_states, obj_trans, obj_rot




def split_state_twohands(state):
    # nn_finger_states = state.size(-1) - (3 + 3) * 2 - (3 + 4)
    # nn_finger_states = nn_finger_states // 2
    # nn_finger_states = state.size(-1) - (3 + 3) - (3 + 4)
    # mano_trans = state[:, : 3]
    # mano_rot = state[:, 3: 3 + 3]
    # mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    # obj_rot = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 4]
    # obj_trans = state[:, 6 + nn_finger_states + 4: 6 + nn_finger_states + 4 + 3]
    nn_mano_trans_dim = 3
    nn_mano_rot_dim = 3
    
    tot_mano_trans = state[:, : nn_mano_trans_dim * 2]
    tot_mano_rot = state[:, nn_mano_trans_dim * 2: (nn_mano_trans_dim + nn_mano_rot_dim) * 2]
    tot_mano_states = state[:, (nn_mano_trans_dim + nn_mano_rot_dim) * 2: -7]
    
    mano_trans, left_mano_trans = tot_mano_trans[:, : nn_mano_trans_dim], tot_mano_trans[:, nn_mano_trans_dim: ]
    mano_rot, left_mano_rot = tot_mano_rot[:, : nn_mano_rot_dim], tot_mano_rot[:, nn_mano_rot_dim: ]
    mano_states, left_mano_states = tot_mano_states[:, : tot_mano_states.shape[-1] // 2], tot_mano_states[:, tot_mano_states.shape[-1] // 2: ]
    
    obj_trans = state[:, -7: -4]
    obj_rot = state[:, -4:]

    return mano_trans, left_mano_trans, mano_rot, left_mano_rot, mano_states, left_mano_states, obj_trans, obj_rot ## rot, trans ##
    
    mano_trans = state[:, : 3]
    mano_rot = state[:,  3: 3 + 3]
    mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    obj_trans = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 3]
    obj_rot = state[:, 6 + nn_finger_states + 3: 6 + nn_finger_states + 4 + 3] ## rotation quaternion matrix ##
    return mano_trans, mano_rot, mano_states, obj_trans, obj_rot


    
    
def pose_err_ours(obs, target, weight:typing.Dict[str, float], weights=1, dt:float = 1/20, epsilon = 0.0, actor=None):
    ## get posees for the mano hand and the objects 
    ## for eachfo them respectively and useing corresponding weights #
    ## return them #
    ## rollout the values ##
    
    if len(obs.shape) == 1:
        obs = obs.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)
    
    ## delta mano rot of the rot vector ##
    ####### 
    # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot = split_state(obs - target)
    
    # trans, target #
    obs_mano_trans, obs_mano_rot, obs_mano_states, obs_obj_trans, obs_obj_rot = split_state(obs)
    tar_mano_trans, tar_mano_rot, tar_mano_states, tar_obj_trans, tar_obj_rot = split_state(target)
    
    delta_mano_trans = obs_mano_trans - tar_mano_trans
    delta_mano_rot = obs_mano_rot - tar_mano_rot
    delta_mano_states = obs_mano_states - tar_mano_states
    delta_obj_trans = obs_obj_trans - tar_obj_trans
    ####### 
    
    
    
    # if torch.any(torch.isnan(delta_mano_trans)): ## mano trans ## 
    #     print(f"has nan value delta_mano_trans")
    # if torch.any(torch.isnan(delta_mano_rot)):
    #     print(f"has nan value delta_mano_rot")
    # if torch.any(torch.isnan(delta_mano_states)):
    #     print(f"has nan value delta_mano_states")
    # if torch.any(torch.isnan(delta_obj_rot)):
    #     print(f"has nan value delta_obj_rot")
    # if torch.any(torch.isnan(delta_obj_trans)): 
    #     print(f"has nan value delta_obj_trans")

    weight_mano_trans, weight_mano_rot, weight_mano_states, weight_obj_rot, weight_obj_trans = weight['weight_mano_trans'], weight['weight_mano_rot'], weight['weight_mano_states'], weight['weight_obj_rot'], weight['weight_obj_trans']
        
    # print(f"delta_mano_trans: {delta_mano_trans.size()}, delta_mano_rot: {delta_mano_rot.size()}, delta_mano_states: {delta_mano_states.size()}, delta_obj_rot: {delta_obj_rot.size()}, delta_obj_trans: {delta_obj_trans.size()}")
        
    mano_trans_loss = weight_mano_trans * torch.norm(delta_mano_trans, p=1, dim=-1) 
    mano_rot_loss = weight_mano_rot * torch.norm(delta_mano_rot, p=1, dim=-1) 
    mano_states_loss = weight_mano_states * torch.norm(delta_mano_states, p=1, dim=-1) 
    
    
    if actor is not None:
        pred_transformed_obj_verts = actor.world_model.forward_kinematics_obj(obs_obj_rot[:, [3, 0, 1, 2]], obs_obj_trans)
        tar_transformed_obj_verts = actor.world_model.forward_kinematics_obj(tar_obj_rot[:, [3, 0, 1, 2]], tar_obj_trans)
        diff_verts = torch.sum(
            (pred_transformed_obj_verts - tar_transformed_obj_verts) ** 2, dim=-1)
        diff_verts = diff_verts.mean(dim=-1)
        # diff_verts = diff_verts.mean(dim=-1)
        obj_rot_loss = (weight_obj_rot * 1000) * diff_verts ## bsz 
        obj_trans_loss = (weight_obj_trans * 1000) * diff_verts
    else:
        cos_half_angle = torch.sum(obs_obj_rot * tar_obj_rot, dim=-1)
        # obj_rot_loss = weight_obj_rot * torch.norm(delta_obj_rot, p=1, dim=-1) # obj #
        obj_rot_loss = weight_obj_rot * (1. - cos_half_angle) ## negative half angle of the angle ## # negative cos 
        obj_trans_loss = weight_obj_trans * torch.norm(delta_obj_trans, p=1, dim=-1)
        
    mano_trans_loss = mano_trans_loss.mean()
    mano_rot_loss = mano_rot_loss.mean()
    mano_states_loss = mano_states_loss.mean()
    obj_rot_loss = obj_rot_loss.mean()
    obj_trans_loss = obj_trans_loss.mean()
    
    
    return mano_trans_loss, mano_rot_loss, mano_states_loss, obj_trans_loss, obj_rot_loss



def pose_err_ours_twohands(obs, target, weight:typing.Dict[str, float], weights=1, dt:float = 1/20, epsilon = 0.0):
    ## get posees for the mano hand and the objects 
    ## for eachfo them respectively and useing corresponding weights #
    ## return them #
    ## rollout the values ##
    
    if len(obs.shape) == 1:
        obs = obs.unsqueeze(0)
    if len(target.shape) == 1:
        target = target.unsqueeze(0)
    
    ## delta mano rot of the rot vector ##
    ####### 
    # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot = split_state(obs - target)
    
    if torch.any(torch.isnan(obs)):
        print(f"has nan value obs")
    if torch.any(torch.isnan(target)):
        print(f"has nan value target")
        
    
    # obs_mano_trans, obs_mano_rot, obs_mano_states, obs_obj_trans, obs_obj_rot = split_state(obs)
    # tar_mano_trans, tar_mano_rot, tar_mano_states, tar_obj_trans, tar_obj_rot = split_state(target)
    
    
    obs_mano_trans, obs_left_mano_trans, obs_mano_rot, obs_left_mano_rot, obs_mano_states, obs_left_mano_states, obs_obj_trans, obs_obj_rot = split_state_twohands(obs)
    tar_mano_trans, tar_left_mano_trans, tar_mano_rot, tar_left_mano_rot, tar_mano_states, tar_left_mano_states, tar_obj_trans, tar_obj_rot = split_state_twohands(target)
    
    
    delta_mano_trans = obs_mano_trans - tar_mano_trans
    delta_mano_rot = obs_mano_rot - tar_mano_rot
    delta_mano_states = obs_mano_states - tar_mano_states
    
    delta_left_mano_trans = obs_left_mano_trans - tar_left_mano_trans
    delta_left_mano_rot = obs_left_mano_rot - tar_left_mano_rot
    delta_left_mano_states = obs_left_mano_states - tar_left_mano_states
    
    delta_obj_trans = obs_obj_trans - tar_obj_trans
    ####### 
    cos_half_angle = torch.sum(obs_obj_rot * tar_obj_rot, dim=-1)
    
    
    # if torch.any(torch.isnan(delta_mano_trans)): ## mano trans ## 
    #     print(f"has nan value delta_mano_trans")
    # if torch.any(torch.isnan(delta_mano_rot)):
    #     print(f"has nan value delta_mano_rot")
    # if torch.any(torch.isnan(delta_mano_states)):
    #     print(f"has nan value delta_mano_states")
    # if torch.any(torch.isnan(delta_obj_rot)):
    #     print(f"has nan value delta_obj_rot")
    # if torch.any(torch.isnan(delta_obj_trans)): 
    #     print(f"has nan value delta_obj_trans")

    weight_mano_trans, weight_mano_rot, weight_mano_states, weight_obj_rot, weight_obj_trans = weight['weight_mano_trans'], weight['weight_mano_rot'], weight['weight_mano_states'], weight['weight_obj_rot'], weight['weight_obj_trans']
        
    # print(f"delta_mano_trans: {delta_mano_trans.size()}, delta_mano_rot: {delta_mano_rot.size()}, delta_mano_states: {delta_mano_states.size()}, delta_obj_rot: {delta_obj_rot.size()}, delta_obj_trans: {delta_obj_trans.size()}")
        
    mano_trans_loss = weight_mano_trans * (torch.norm(delta_mano_trans, p=1, dim=-1) + torch.norm(delta_left_mano_trans, p=1, dim=-1))
    mano_rot_loss = weight_mano_rot * (torch.norm(delta_mano_rot, p=1, dim=-1) + torch.norm(delta_left_mano_rot, p=1, dim=-1))
    mano_states_loss = weight_mano_states * (torch.norm(delta_mano_states, p=1, dim=-1) + torch.norm(delta_left_mano_states, p=1, dim=-1))
    # obj_rot_loss = weight_obj_rot * torch.norm(delta_obj_rot, p=1, dim=-1) # obj rotjloss #
    obj_rot_loss = weight_obj_rot * (1. - cos_half_angle) ## negative half angle of the angle ## # negative cos 
    obj_trans_loss = weight_obj_trans * torch.norm(delta_obj_trans, p=1, dim=-1)
    
    mano_trans_loss = mano_trans_loss.mean()
    mano_rot_loss = mano_rot_loss.mean()
    mano_states_loss = mano_states_loss.mean()
    obj_rot_loss = obj_rot_loss.mean()
    obj_trans_loss = obj_trans_loss.mean()
    
    
    return mano_trans_loss, mano_rot_loss, mano_states_loss, obj_trans_loss, obj_rot_loss
    

# @torch.jit.script
def pose_err(obs, target, weight:typing.Dict[str, float], weights=1, dt:float = 1/20, epsilon = 0.0):
    
    target = target.view(obs.shape) #in thejtorch formate ###
    assert obs.shape == target.shape

    delta_pos, delta_rot, delta_vel, delta_avel, delta_height, delta_up_dir = decompose_obs(obs - target)
    
    weight_pos, weight_vel, weight_rot, weight_avel = weight[
        "pos"], weight["vel"], weight["rot"], weight["avel"]

    weight_height, weight_up_dir, weight_l2, weight_l1 = weight[
        "height"], weight["up_dir"], weight["l2"], weight["l1"]
    
    pos_loss = weight_pos * \
        torch.mean(torch.norm(delta_pos, p=1, dim=-1) * weights)
    rot_loss = weight_rot * \
        torch.mean(torch.norm(delta_rot, p=1, dim=-1) * weights)
    vel_loss = weight_vel * \
        torch.mean(torch.norm(delta_vel, p=1, dim=-1) * weights)
    avel_loss = weight_avel * \
        torch.mean(torch.norm(delta_avel, p=1, dim=-1) * weights)
    height_loss = weight_height * \
        torch.mean(torch.norm(delta_height, p=1, dim=-1) * weights)
    up_dir_loss = weight_up_dir * \
        torch.mean(torch.norm(delta_up_dir, p=1, dim=-1) * weights)
    
    ## pos loss; rot loss; vel loss; ##
    return pos_loss, rot_loss, dt*vel_loss, \
        dt*avel_loss, height_loss, up_dir_loss

