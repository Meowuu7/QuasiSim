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
from torch import nn
import torch
import ControlVAECore.Utils.pytorch_utils as ptu
import ControlVAECore.Utils.diff_quat as DiffRotation
from ControlVAECore.Utils.motion_utils import *



def update_quaternion_batched(delta_angle, prev_quat):
    
    
    s1 = 0
    s2 = prev_quat[:, 0] # s2 
    v2 = prev_quat[:, 1:] # v2 
    v1 = delta_angle / 2
    new_v = s1 * v2 + s2.unsqueeze(-1) * v1 + torch.cross(v1, v2, dim=-1)
    new_s = s1 * s2 - torch.sum(v1 * v2, dim=-1) ## nb # 
    new_quat = torch.cat([new_s.unsqueeze(-1), new_v], dim=-1)
    return new_quat


# @torch.jit.script
def integrate_obs_vel(states, delta):
    shape_ori = states.shape

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"

    if len(delta.shape) == 2:
        delta = delta[None]
    assert len(delta.shape) == 3, "acc shape error"
    
    assert delta.shape[-1] == 6
    assert states.shape[-1] == 13

    batch_size, num_body, _ = states.shape

    
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)

    delta = delta.view(-1, 6)
    local_delta = delta[..., 0:3]
    local_ang_delta = delta[..., 3:6]

    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    true_delta = quat_apply(root_rot,local_delta)
    true_ang_delta = quat_apply(root_rot, local_ang_delta)

    next_vel = vel + true_delta
    next_avel = avel + true_ang_delta
    return next_vel, next_avel


def integrate_state(states, delta, dt):
    pos = states[..., 0:3].contiguous().view(-1, 3)
    rot = states[..., 3:7].contiguous().view(-1, 4)
    
    next_vel, next_avel = integrate_obs_vel(states, delta)
    next_pos = pos + next_vel * dt
    next_rot = quat_integrate(rot, next_avel, dt)
    
    batch_size, num_body, _ = states.shape
    
    next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                                 next_rot.view(batch_size, num_body, 4),
                                 next_vel.view(batch_size, num_body, 3),
                                 next_avel.view(batch_size, num_body, 3)
                                 ], dim=-1)

    return next_state.view(states.shape)


# def split_state(self, state):
#     mano_trans = state[:, : self.bullet_mano_num_trans_state]
#     mano_rot = state[:, self.bullet_mano_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
#     mano_states = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]
    
#     obj_rot = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state]
#     obj_trans = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state + self.obj_num_trans_state]
#     return mano_trans, mano_rot, mano_states, obj_rot, obj_trans

# def split_delta(self, delta):
#     delta_mano_trans = delta[:, :self.bullet_mano_num_trans_state] ## determined by the rot state ##
#     delta_mano_rot = delta[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state]
#     delta_mano_states = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints] ## delta mano  states ##
    
#     delta_obj_rot = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act]
#     delta_obj_trans = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act + self.obj_num_trans_act]
#     ## delta obj trans delta obj rot ##
#     return delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_rot, delta_obj_trans



## simpleworldmodel ##
def split_state(state):
    nn_finger_states = state.size(-1) - (3 + 3) - (3 + 4)
    mano_trans = state[:, : 3]
    mano_rot = state[:, 3: 3 + 3]
    mano_states = state[:, 3 + 3: 6 + nn_finger_states]
    
    obj_rot = state[:, 6 + nn_finger_states: 6 + nn_finger_states + 4]
    obj_trans = state[:, 6 + nn_finger_states + 4: 6 + nn_finger_states + 4 + 3]
    return mano_trans, mano_rot, mano_states, obj_rot, obj_trans
    
## simpleworldmodel ## 
class SimpleWorldModel(nn.Module):
    def __init__(self, ob_size, ac_size, delta_size, dt, statistics, **kargs) -> None:
        super(SimpleWorldModel, self).__init__()
        
        self.mlp = ptu.build_mlp(
            input_dim = ob_size + ac_size, # rotvec to 6d vector # delta size #
            output_dim= delta_size,
            hidden_layer_num=kargs['world_model_hidden_layer_num'],
            hidden_layer_size=kargs['world_model_hidden_layer_size'],
            activation=kargs['world_model_activation']
        )
        
        ## doens -> bsz x rollout length -> dones of the trajectory ##
        ## bullet_mano_num_joints, bullet_mano_finger_num_joints, bullet_mano_num_rot_state, bullet_mano_num_trans_state ###
        ## obj_num_rot_state, obj_num_rot_act, obj_num_trans_state, obj_num_trans_act ##
        self.bullet_mano_num_joints = kargs['bullet_mano_num_joints']
        self.bullet_mano_finger_num_joints = kargs['bullet_mano_finger_num_joints']
        self.bullet_mano_num_rot_state = 3
        self.bullet_mano_num_trans_state = 3 ## statistics ##
        self.obj_num_rot_state = 4
        self.obj_num_rot_act = 3
        self.obj_num_trans_state = 3
        self.obj_num_trans_act = 3

        self.obs_mean = nn.Parameter(ptu.from_numpy(statistics['obs_mean']), requires_grad= False)
        self.obs_std = nn.Parameter(ptu.from_numpy(statistics['obs_std']), requires_grad= False)
        
        self.delta_mean = nn.Parameter(ptu.from_numpy(statistics['delta_mean']), requires_grad= False)
        self.delta_std = nn.Parameter(ptu.from_numpy(statistics['delta_std']), requires_grad= False)
        
        ## dt ## 
        self.dt = dt
        
        self.weight = {}
        for key,value in kargs.items():
            if 'world_model_weight' in key:
                self.weight[key.replace('world_model_weight_','')] = value
        for k in ['mano_trans', 'mano_rot', 'mano_states', 'obj_rot', 'obj_trans']:
            self.weight[f'weight_{k}'] = 1.0
        
    def normalize_obs(self, observation):
        return observation
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        observation = ptu.normalize(observation, self.obs_mean, self.obs_std)
        return observation
    # 

    @staticmethod
    def integrate_state(states, delta, dt): ## 
        # needs pytorch >= 11.0 #
        
        pos = states[..., 0:3].contiguous().view(-1, 3)
        rot = states[..., 3:7].contiguous().view(-1, 4)
        batch_size, num_body, _ = states.shape

        vel = states[..., 7:10].contiguous().view(-1, 3)
        avel = states[..., 10:13].contiguous().view(-1, 3)

        root_rot = states[:, 0, 3:7].contiguous().view(-1,1,4)        
        delta = delta.contiguous().view(batch_size, -1, 3)
        
        true_delta = broadcast_quat_apply(root_rot, delta).contiguous().view(batch_size, -1, 6) # true delta #
        true_delta, true_ang_delta = true_delta[...,:3].contiguous().view(-1,3), true_delta[...,3:].contiguous().view(-1,3)
        
        next_vel = vel + true_delta
        next_avel = avel + true_ang_delta
        
        next_pos = pos + next_vel * dt
        next_rot = quat_integrate(rot, next_avel, dt)
        next_state = torch.cat([next_pos.contiguous().view(batch_size, num_body, 3),
                                    next_rot.contiguous().view(batch_size, num_body, 4),
                                    next_vel.contiguous().view(batch_size, num_body, 3),
                                    next_avel.contiguous().view(batch_size, num_body, 3)
                                    ], dim=-1)
        return next_state.contiguous().view(states.shape)
    
    
    
    def loss(self, pred, tar):
        pred_mano_trans, pred_mano_rot, pred_mano_states, pred_obj_trans,  pred_obj_rot = self.split_state(pred)
        tar_mano_trans, tar_mano_rot, tar_mano_states, tar_obj_trans, tar_obj_rot = self.split_state(tar)
        
        if torch.any(torch.isnan(tar_mano_trans))  or torch.any(torch.isnan(tar_mano_rot)) or torch.any(torch.isnan(tar_mano_states)) or torch.any(torch.isnan(tar_obj_rot)) or torch.any(torch.isnan(tar_obj_trans)):
            print(f"HAs nan values in target values ")
        
        
        if torch.any(torch.isnan(pred_mano_trans))  or torch.any(torch.isnan(pred_mano_rot)) or torch.any(torch.isnan(pred_mano_states)) or torch.any(torch.isnan(pred_obj_rot)) or torch.any(torch.isnan(pred_obj_trans)):
            print(f"HAs nan values in predicted values ")
        
        
        # if torch.any(torch.isnan(pred_mano_trans)): ## pred mano trans ##
        #     print(f"None value in pred_mano_trans!") ## pred mano trans ##
        # if torch.any(torch.isnan(pred_mano_rot)): ## 
        #     print(f"None value in pred_mano_rot!") # 
        # if torch.any(torch.isnan(pred_mano_states)): # 
        #     print(f"None value in pred_mano_states!") # 
        # if torch.any(torch.isnan(pred_obj_rot)): # 
        #     print(f"None value in pred_obj_rot!") # pred 
        # if torch.any(torch.isnan(pred_obj_trans)): ## 
        #     print(f"None value in pred_obj_trans!") # 
        ## weight mano trans ##
        
        weight_mano_trans, weight_mano_rot, weight_mano_states, weight_obj_rot, weight_obj_trans = self.weight['weight_mano_trans'], self.weight['weight_mano_rot'], self.weight['weight_mano_states'], self.weight['weight_obj_rot'], self.weight['weight_obj_trans']
        
        mano_trans_loss = weight_mano_trans * torch.mean(torch.norm(pred_mano_trans - tar_mano_trans, p=1, dim=-1))
        mano_rot_loss = weight_mano_rot * torch.mean(torch.norm(pred_mano_rot - tar_mano_rot, p=1, dim=-1))
        mano_states_loss = weight_mano_states * torch.mean(torch.norm(pred_mano_states - tar_mano_states, p=1, dim=-1))
        obj_rot_loss = weight_obj_rot * torch.mean(torch.norm(pred_obj_rot - tar_obj_rot, p=1, dim=-1))
        obj_trans_loss = weight_obj_trans * torch.mean(torch.norm(pred_obj_trans - tar_obj_trans, p=1, dim=-1))
        
        return mano_trans_loss, mano_rot_loss, mano_states_loss, obj_rot_loss, obj_trans_loss
        
        pred_pos, pred_rot, pred_vel, pred_avel = decompose_state(pred)
        tar_pos, tar_rot, tar_vel, tar_avel = decompose_state(tar)

        weight_pos, weight_vel, weight_rot, weight_avel = self.weight[
            "pos"], self.weight["vel"], self.weight["rot"], self.weight["avel"]

        batch_size = tar_pos.shape[0]

        pos_loss = weight_pos * \
            torch.mean(torch.norm(pred_pos - tar_pos, p=1, dim=-1))
        vel_loss = weight_vel * \
            torch.mean(torch.norm(pred_vel - tar_vel, p=1, dim=-1))
        avel_loss = weight_avel * \
            torch.mean(torch.norm(pred_avel - tar_avel, p=1, dim=-1))

        # special for rotation
        pred_rot_inv = quat_inv(pred_rot)
        tar_rot = DiffRotation.flip_quat_by_w(tar_rot.view(-1,4))
        pred_rot_inv = DiffRotation.flip_quat_by_w(pred_rot_inv.view(-1, 4))
        dot_pred_tar = torch.norm( DiffRotation.quat_to_rotvec( DiffRotation.quat_multiply(tar_rot,
                                              pred_rot_inv) ), p =2, dim=-1)
        rot_loss = weight_rot * \
            torch.mean(torch.abs(dot_pred_tar))
        return pos_loss, rot_loss, self.dt * vel_loss, self.dt * avel_loss
    
    def split_state(self, state): ## split 
        mano_trans = state[:, : self.bullet_mano_num_trans_state]
        mano_rot = state[:, self.bullet_mano_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state]
        mano_states = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints]
        
        obj_trans = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state]
        obj_rot = state[:, self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_trans_state: self.bullet_mano_num_trans_state + self.bullet_mano_num_rot_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_state + self.obj_num_trans_state]
        return mano_trans, mano_rot, mano_states, obj_trans, obj_rot
    
    ## split
    def split_delta(self, delta):
        delta_mano_trans = delta[:, :self.bullet_mano_num_trans_state] ## determined by the rot state ##
        delta_mano_rot = delta[:, self.bullet_mano_num_trans_state : self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state]
        delta_mano_states = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints] ## delta mano  states ##
        
        delta_obj_trans = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act]
        delta_obj_rot = delta[:, self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act: self.bullet_mano_num_rot_state + self.bullet_mano_num_trans_state + self.bullet_mano_finger_num_joints + self.obj_num_rot_act + self.obj_num_trans_act]
        ## delta obj trans delta obj rot ## ## split detla and jget the rot act and the transact e
        return delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot #gedeta mano rot # 
    
        
    def integrate_states_hoi(self, nex_mano_trans, nex_mano_rot, nex_mano_states, nex_obj_rot, nex_obj_trans):
        mano_states = torch.cat( # get states #
            [nex_mano_trans, nex_mano_rot, nex_mano_states], dim=-1
        )
        # obj_states = torch.cat(
        #     [nex_obj_rot, nex_obj_trans], dim=-1
        # )
        obj_states = torch.cat(
            [nex_obj_trans, nex_obj_rot], dim=-1

        )
        states = torch.cat(
            [mano_states, obj_states], dim=-1
        )
        return states 
    
    def forward(self, state, action, **obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                # observation = state2ob(state)
                observation = state # normalize obs 
                # observation1 = state2obfast(state)
            n_observation = self.normalize_obs(observation)
        
        ## the differentnceces
        # if torch.any(torch.isnan(n_observation)):
        #     print(f"Has NaN value in n_observation!!")
            
        # try:
        #     if torch.any(torch.isnan(observation)):
        #         print(f"Has NaN value in observation!!")
        # except:
        #     pass
        
        # # ### 
        
        # if torch.any(torch.isnan(action)):
        #     print(f"Has NaN value in action!!")
        
        # batch_size = 1 if len(action.shape)==1 else action.shape[0]
        # action = quat_to_vec6d(quat_from_rotvec(action.view(-1,3))).view(batch_size,-1)
        
        # action = flip_quat_by_w(quat_from_rotvec(action.view(-1,3)))#.view(batch_size, -1)
        # action = quat_to_vec6d(action).view(batch_size, -1)
        
        if torch.any(torch.isnan(n_observation)):
            print(f"Has NaN value in n_observation!!")
            
        if torch.any(torch.isnan(action)):
            print(f"Has NaN value in action!!")
        
        ## n_observation with actions ##  ## filed  ### filed ## 
        ## observtion with actions ## ## observation with actions ## ## actions -> delta pd targets here ## use that as the delta targets to predict the delta chagn ebetween the current stepped observation and the next real observation ## ##  and 
        n_delta = self.mlp( torch.cat([n_observation, action], dim = -1) )
        
        
        delta = n_delta
        
        # delta = ptu.unnormalize(n_delta, self.delta_mean, self.delta_std) 
        
        if torch.any(torch.isnan(n_delta)):
            print(f"Has NaN value in n_delta!!")
        # if torch.any(torch.isnan(delta)):
        #     print(f"Has NaN value in delta!!")
        
        ## bullet_mano_num_joints, bullet_mano_finger_nu
        # ## mano trans mano root, ## m_joints, bullet_mano_num_rot_state, bullet_mano_num_trans_state ###
        ## obj_num_rot_state, obj_num_rot_act, obj_num_trans_state, obj_num_trans_act ##
        ## just predict them as delta states here ##
        #### delta mano rot ##
        ## 
        # mano_trans, mano_rot, mano_states, obj_rot, obj_trans = self.split_state(state)
        mano_trans, mano_rot, mano_states, obj_trans, obj_rot = self.split_state(state)
        # delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_rot, delta_obj_trans = self.split_delta(delta)
        delta_mano_trans, delta_mano_rot, delta_mano_states, delta_obj_trans, delta_obj_rot = self.split_delta(delta)
        
        
        ### aggreagate them together ##
        nex_mano_trans = mano_trans + delta_mano_trans # delta mano trans 
        nex_mano_rot=  mano_rot + delta_mano_rot # delta mano rot 
        nex_mano_states = mano_states + delta_mano_states # delta mano states 
        
        # print(f"delta_obj_rot: {delta_obj_rot.size()}, obj_rot: {obj_rot.size()}")
        # update delta rot  # integrate for nex obj #  ##  
        obj_rot  = obj_rot[:, [3,0,1,2]] # obj rot
        nex_obj_rot = obj_rot + update_quaternion_batched(delta_obj_rot, obj_rot)
        ## get obj rot ##
        nex_obj_rot = nex_obj_rot / torch.clamp(torch.norm(nex_obj_rot, dim=-1, keepdim=True), min=1e-5)
        
        if torch.any(torch.isnan(nex_obj_rot)):
            print(f"Has NaN value in nex_obj_rot (after norm)!!")
        
        ## nex obj rot ##
        nex_obj_rot = nex_obj_rot[:, [1,2,3,0]]
        
        nex_obj_trans = obj_trans + delta_obj_trans
        
        ### get the nex t state s##
        state = self.integrate_states_hoi(
            nex_mano_trans, nex_mano_rot, nex_mano_states, nex_obj_rot, nex_obj_trans
        )
        # 
        
        
        # delta = n_delta
        # state = integrate_state(state, delta.view(batch_size, -1,6), self.dt)
        # state = self.integrate_state(state, delta, self.dt)
        return state
        
