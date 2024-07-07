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
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
import math
from ControlVAECore.Utils.pytorch_utils import *
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, var, **params) -> None:
        super(Encoder, self).__init__()
        self.activation = str_to_activation[params['activation']]
        self.hidden_size = params['hidden_layer_size']
        self.input_size = input_size
        self.condition_size = condition_size
        self.fc_layers = []
        self.fc_layers.append(nn.Linear(input_size + condition_size, self.hidden_size))
        for i in range(params['hidden_layer_num']):
            self.fc_layers.append(nn.Linear(input_size + self.hidden_size, self.hidden_size))    
        self.fc_layers = nn.ModuleList(self.fc_layers)
        
        for i_fc, layer in enumerate(self.fc_layers):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
        self.has_init = False
        
        self.mu = nn.Linear(input_size + self.hidden_size, output_size)
        self.logvar = nn.Linear(input_size + self.hidden_size, output_size)
        self.var = var # 
        if self.var is not None:
            self.log_var = math.log(var)*2
    
    def init_fc_layers(self, ):
        print(f"initing fc layers")
        for i_fc, layer in enumerate(self.fc_layers):
            if isinstance(layer, nn.Linear):
                # torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.has_init = True
    
    def encode(self, x, c):
        # print(f"has_init: {self.has_init}")
        # if not self.has_init:  
        #     self.init_fc_layers()
        
        res = c
        for i, layer in enumerate(self.fc_layers):
            if res is not None:
                res = layer(torch.cat([x,res], dim = -1))
            else:
                res = layer(x)
            if torch.any(torch.isnan(res)):
                maxx_x = torch.max(x).item()
                minn_x = torch.min(x).item()
                print(f"[Encoder - encode] layer {i} has NaN value in res; maxx_x: {maxx_x}, minn_x: {minn_x}, x: {x.size()}, init_fc_size: {self.input_size + self.condition_size}")
                raise ValueError("nan")
            res = self.activation(res)
            if torch.any(torch.isnan(res)):
                print(f"[Encoder - encode - act] layer {i} has NaN value in res")
                raise ValueError("nan")
            
        latent = torch.cat([x,res], dim = -1)
        mu = self.mu(latent)
        if self.var is not None:
            logvar = torch.ones_like(mu)*self.log_var
        else:
            logvar = self.logvar(latent)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(std)
        return mu + exp * std
    
    def forward(self, x, c):
        mu, logvar = self.encode(x,c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VisualFeatsLearnablePriorEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var = None, **kargs) -> None:
        super(VisualFeatsLearnablePriorEncoder, self).__init__()
        self.prior = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        self.post = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        self.var = fix_var
        
        ### Point type embedding dim ###
        self.point_type_embedding_dim = 128
        self.point_feat_hidden_dim = 128
        self.point_type_embedding_layer = nn.Embedding(
            num_embeddings=4, # 4 point types #
            embedding_dim=self.point_type_embedding_dim,
        ) # .to(self.device)
        
        ###### Get the point feature extraction layer ######
        self.point_feat_extraction_layer = nn.Sequential(
            nn.Linear(3 + self.point_type_embedding_dim, self.point_type_embedding_dim), nn.ReLU(),
            nn.Linear(self.point_type_embedding_dim, self.point_feat_hidden_dim), nn.ReLU(),
            nn.Linear(self.point_feat_hidden_dim, self.point_feat_hidden_dim)
        )
        self.glb_feat_layer = nn.Sequential(
            nn.Linear(self.point_feat_hidden_dim + self.point_feat_hidden_dim, self.point_feat_hidden_dim), nn.ReLU(),
            nn.Linear(self.point_feat_hidden_dim, self.point_feat_hidden_dim)
        )
        ### current state, desired actions, global feature embeddings -> modified delta aactions
        # self.in_delta_act_dim = 
        # self.delta_action_layer = nn.Sequential(
        #     nn.Linear(self.point_feat_hidden_dim + ac_size + ac_size, self.point_feat_hidden_dim), nn.ReLU(),
        #     nn.Linear(self.point_feat_hidden_dim, self.point_feat_hidden_dim // 2), nn.ReLU(),
        #     nn.Linear(self.point_feat_hidden_dim // 2, ac_size)
        # )
        # torch.nn.init.zeros_(self.delta_action_layer[-1].weight)
        # torch.nn.init.zeros_(self.delta_action_layer[-1].bias)
        
        
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, n_observation, encoded_feats):
        return self.prior(n_observation, encoded_feats)
    
    
    def encode_post(self, encoded_feats, n_target):
        return self.post(n_target, encoded_feats)
    
    def encode_visual_feats(self, observation, wm_wana):
        # ## encode visual feats ## #
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float().cuda()
        if len(observation.size()) == 1:
            observation = observation.unsqueeze(0)
        mano_trans, mano_rot, mano_states, obj_trans, obj_rot = wm_wana.split_state(observation)
        mano_trans = mano_trans.squeeze(0)
        mano_rot = mano_rot.squeeze(0)
        mano_states = mano_states.squeeze(0)
        obj_trans = obj_trans.squeeze(0)
        obj_rot = obj_rot.squeeze(0)
        
        mano_rot_quat_wxyz = wm_wana.euler_state_to_quat_wxyz(mano_rot)
        hand_visual_pts = wm_wana.forward_kinematics_v2(mano_trans, mano_rot_quat_wxyz, mano_states)
        
        ### obj visual pts ##
        obj_visual_pts = wm_wana.forward_kinematics_obj(obj_rot, obj_trans).squeeze(0)
        hand_pts_types = torch.zeros((hand_visual_pts.size(0), ), dtype=torch.long).cuda()
        obj_pts_types = torch.ones((obj_visual_pts.size(0), ), dtype=torch.long).cuda()
        hand_pts_embedding = self.point_type_embedding_layer(hand_pts_types)
        obj_pts_embedding = self.point_type_embedding_layer(obj_pts_types)
        hand_pts_in_feats = torch.cat(
            [hand_visual_pts, hand_pts_embedding], dim=-1
        )
        obj_pts_in_feats = torch.cat(
            [obj_visual_pts, obj_pts_embedding], dim=-1
        )
        hand_pts_feats = self.point_feat_extraction_layer(hand_pts_in_feats)
        obj_pts_feats = self.point_feat_extraction_layer(obj_pts_in_feats)
        hand_pts_feats, _ = torch.max(hand_pts_feats, dim=0, keepdim=True)
        obj_pts_feats, _ = torch.max(obj_pts_feats, dim=0, keepdim=True)
        hand_obj_pts_feats = torch.cat([hand_pts_feats, obj_pts_feats], dim=-1)
        hand_obj_pts_feats = self.glb_feat_layer(hand_obj_pts_feats)
        return hand_obj_pts_feats
        
        
    
    def forward(self, n_observation, n_target, observation, wm_wana):
        if torch.any(torch.isnan(n_observation)): # forward the action  
            print(f"[encoding] has NaN value in n_observation")
        if torch.any(torch.isnan(n_target)):
            print(f"[encoding] has NaN value in n_target")
        
        ##### Encode the visual features from the observation #####
        encoded_obs_feats = self.encode_visual_feats(observation, wm_wana) ## 
        
        _, mu_prior, logvar_prior = self.encode_prior(n_observation, encoded_obs_feats)
        latent_code, mu_post, logvar_post = self.encode_post(encoded_obs_feats, n_target)
        
        if torch.any(torch.isnan(latent_code)):
            print(f"[encoding] has NaN value in latent_code")
        if torch.any(torch.isnan(mu_post)):
            print(f"[encoding] has NaN value in mu_post")
        if torch.any(torch.isnan(mu_prior)):
            print(f"[encoding] has NaN value in mu_prior")
        
        
        return latent_code + mu_prior, mu_prior+mu_post, mu_prior
   


class SimpleLearnablePriorEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var = None, **kargs) -> None:
        super(SimpleLearnablePriorEncoder, self).__init__()
        self.prior = Encoder(
            input_size= condition_size,
            condition_size= 0,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        self.post = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        self.var = fix_var
        
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, n_observation):
        return self.prior(n_observation, None)
    def encode_post(self, n_observation, n_target):
        return self.post(n_target, n_observation)
    
    def forward(self, n_observation, n_target):
        if torch.any(torch.isnan(n_observation)): # forward the action  
            print(f"[encoding] has NaN value in n_observation")
        if torch.any(torch.isnan(n_target)):
            print(f"[encoding] has NaN value in n_target")
        _, mu_prior, logvar_prior = self.encode_prior(n_observation)
        latent_code, mu_post, logvar_post = self.encode_post(n_observation, n_target)
        
        if torch.any(torch.isnan(latent_code)):
            print(f"[encoding] has NaN value in latent_code")
        if torch.any(torch.isnan(mu_post)):
            print(f"[encoding] has NaN value in mu_post")
        if torch.any(torch.isnan(mu_prior)):
            print(f"[encoding] has NaN value in mu_prior")
        
        
        return latent_code + mu_prior, mu_prior+mu_post, mu_prior
    

class StandardVAEEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var, **kargs) -> None:
        super(StandardVAEEncoder, self).__init__()
        print('StandardVAEEncoder')
        self.encoder = Encoder(
            input_size= input_size,
            condition_size= condition_size,
            output_size= output_size,
            var = fix_var,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        self.var = fix_var
        self.latent_dim = output_size
        if self.var is not None:
            self.log_var = math.log(fix_var)*2
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, x):
        assert len(x.shape) == 2
        shape = (x.shape[0], self.latent_dim)
        mu = torch.zeros(shape, dtype = x.dtype, device= x.device)
        return torch.randn_like(mu)*self.var, mu, torch.ones_like(mu)*self.log_var
    
    def forward(self, normalized_obs, normalized_target):
        z, mu, logvar = self.encoder(normalized_target,normalized_obs)
        return z, mu, torch.zeros_like(mu)
    
    
class LearnablePriorEncoder(nn.Module):
    def __init__(self, input_size, condition_size, output_size, fix_var = None, **kargs) -> None:
        super(LearnablePriorEncoder, self).__init__()
        
        self.prior = build_mlp(
            input_dim = input_size,
            # output_dim = kargs['encoder_hidden_layer_size'],
            output_dim= output_size,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        self.posterior = build_mlp(
            input_dim = input_size + condition_size,
            # output_dim = kargs['encoder_hidden_layer_size'],
            output_dim= output_size,
            hidden_layer_num=kargs['encoder_hidden_layer_num'],
            hidden_layer_size=kargs['encoder_hidden_layer_size'],
            activation=kargs['encoder_activation']
        )
        
        # Maybe here should add an activation before it, just try it.... although I don't think it matters...
        self.mu = nn.Linear(kargs['encoder_hidden_layer_size'], output_size)
        self.var = fix_var
        if self.var is not None:
            self.log_var = math.log(self.var)*2
        else:
            self.log_var = nn.Linear(kargs['encoder_hidden_layer_size'], output_size)
    
    def feature2muvar(self, feature):
        mu = feature#self.mu(feature)
        if self.var is not None:
            logvar = torch.ones_like(mu) * self.log_var
        else:
            logvar = self.log_var(feature)
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.randn_like(mu)
        return mu + exp * std
    
    def kl_loss(self, mu_prior, mu_post):
        return 0.5*(mu_prior - mu_post)**2/(self.var**2)
    
    def encode_prior(self, normalized_observation):
        feature = self.prior(normalized_observation)
        mu, logvar = self.feature2muvar(feature)
        return mu, logvar
    
    def encode_posterior(self, normalized_observation, normalized_target):
        feature = self.posterior(torch.cat([normalized_observation, normalized_target], dim = -1))
        mu, logvar = self.feature2muvar(feature)
        return mu, logvar
    
    def forward(self, normalized_observation, normalized_target):
        """encode observation and target into posterior distribution

        Args:
            normalized_observation (tensor): observation
            normalized_target (tensor): target
        
        Returns:
            Tuple(tensor, tensor, tensor): latent code sampled from posterior distribution,
                mean of prior distribution, mean of posterior distribution
        """
        mu_prior, logvar_prior = self.encode_prior(normalized_observation)
        mu_post, logvar_post = self.encode_posterior(normalized_observation, normalized_target)
        
        latent_code = self.reparameterize(mu_prior+mu_post, logvar_post)
        return latent_code, mu_prior+mu_post, mu_prior




class GatingMixedDecoder(nn.Module):
    def __init__(self, latent_size, condition_size, output_size, **kargs):
        super(GatingMixedDecoder, self).__init__()

        input_size = latent_size+condition_size
        hidden_size = kargs['actor_hidden_layer_size']
        inter_size = latent_size + hidden_size
        num_experts = kargs['actor_num_experts']
        num_layer = kargs['actor_hidden_layer_num']
        self.activation = str_to_activation[kargs['actor_activation']]
        self.decoder_layers = []
        
        self.decoder_layers = [ ]
        for i_layer in range(num_layer + 1):
            cur_layer = nn.Linear(inter_size if i_layer != 0 else input_size, hidden_size if i_layer != num_layer else output_size)
            
            if i_layer < num_layer:
                torch.nn.init.xavier_uniform_(cur_layer.weight)
                torch.nn.init.zeros_(cur_layer.bias)
                self.decoder_layers.append(cur_layer)
                self.decoder_layers.append(self.activation)
            else:
                torch.nn.init.zeros_(cur_layer.weight)
                torch.nn.init.zeros_(cur_layer.bias)
                self.decoder_layers.append(cur_layer)
                
        
        self.decoder_layers = nn.Sequential(
            *self.decoder_layers
        )
        
        # for i_layer, cur_layer in enumerate(self.decoder_layers):
        #     if isinstance(cur_layer, nn.Linear) and i_layer < len(self.decoder_layers):
        #         torch.nn.init.xavier_uniform_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        #     elif isinstance(cur_layer, nn.Linear) and i_layer == len(self.decoder_layers):
        #         torch.nn.init.zeros_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        
        
        
        # put in list then initialize and register
        # for i in range(num_layer + 1):
        #     layer = (
        #         nn.Parameter(torch.empty(num_experts, inter_size if i!=0 else input_size, hidden_size if i!=num_layer else output_size)), # num experts #
        #         nn.Parameter(torch.empty(num_experts, hidden_size if i!=num_layer else output_size)),
        #         self.activation if i < num_layer else None 
        #     )
        #     self.decoder_layers.append(layer)

        # for index, (weight, bias, _) in enumerate(self.decoder_layers):
        #     index = str(index)
        #     stdv = 1. / math.sqrt(weight.size(1))
        #     weight.data.uniform_(-stdv, stdv)
        #     # bias.data.uniform_(-stdv, stdv)
        #     # bias.data.uniform_(-stdv, stdv)
        #     torch.nn.init.zeros_(bias)
        #     self.register_parameter("w" + index, weight)
        #     self.register_parameter("b" + index, bias)

        # gate_hsize = kargs['actor_gate_hidden_layer_size']
        # self.gate = nn.Sequential(
        #     nn.Linear(input_size, gate_hsize),
        #     nn.ELU(),
        #     nn.Linear(gate_hsize, gate_hsize),
        #     nn.ELU(),
        #     nn.Linear(gate_hsize, num_experts),
        # )
    
    def forward(self, z, c):
        assert len(c.shape) > 1
        
        ''' original mixture of experts '''
        # coefficients = F.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)
        # layer_out = c
        # for (weight, bias, activation) in self.decoder_layers:
        #     input = z if layer_out is None else torch.cat((z, layer_out), dim=-1)
        #     input = F.layer_norm(input, input.shape[1:])
        #     mixed_bias = contract('be,ek->bk',coefficients, bias)
        #     mixed_input = contract('be,bj,ejk->bk', coefficients, input, weight)
        #     out = mixed_input + mixed_bias
        #     layer_out = activation(out) if activation is not None else out
        ''' original mixture of experts '''

        # input = torch.cat([z,c], dim=-1)
        
        # print(f"input: {input.size()}, weight: {self.decoder_layers[0].weight.size()}, bias: {self.decoder_layers[0].bias.size()}")
        
        layer_out = c
        
        for i_layer, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Linear):
                input = torch.cat([z,layer_out], dim = -1)
                layer_out = layer(input)
            else:
                layer_out = layer(layer_out)
        
        # layer_out = self.decoder_layers(input)
        
        
        
        return layer_out
    



class GatingMixedDecoderV2(nn.Module):
    def __init__(self, latent_size, condition_size, output_size, is_delta=False, **kargs):
        super(GatingMixedDecoderV2, self).__init__()

        input_size = latent_size+condition_size
        hidden_size = kargs['actor_hidden_layer_size']
        inter_size = latent_size + hidden_size
        num_experts = kargs['actor_num_experts']
        num_layer = kargs['actor_hidden_layer_num']
        self.is_delta = is_delta
        
        # nn_frames x nn_dim
        self.nn_frames = kargs['nn_frames']
        self.nn_dim = kargs['nn_dim']
        
        self.frame_data = kargs['frame_data']
        
        self.frame_data_embedding_layer = nn.Embedding(
            num_embeddings=self.nn_frames,
            embedding_dim=self.nn_dim
        )
        
        self.frame_data_embedding_layer.weight.data[:, :] = self.frame_data.clone()
        
        # self.activation = str_to_activation[kargs['actor_activation']]
        # self.decoder_layers = []
        
        # self.decoder_layers = [ ]
        # for i_layer in range(num_layer + 1):
        #     cur_layer = nn.Linear(inter_size if i_layer != 0 else input_size, hidden_size if i_layer != num_layer else output_size)
            
        #     if i_layer < num_layer:
        #         torch.nn.init.xavier_uniform_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        #         self.decoder_layers.append(cur_layer)
        #         self.decoder_layers.append(self.activation)
        #     else:
        #         torch.nn.init.zeros_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        #         self.decoder_layers.append(cur_layer)
                
        
        # self.decoder_layers = nn.Sequential(
        #     *self.decoder_layers
        # )
        
        # for i_layer, cur_layer in enumerate(self.decoder_layers):
        #     if isinstance(cur_layer, nn.Linear) and i_layer < len(self.decoder_layers):
        #         torch.nn.init.xavier_uniform_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        #     elif isinstance(cur_layer, nn.Linear) and i_layer == len(self.decoder_layers):
        #         torch.nn.init.zeros_(cur_layer.weight)
        #         torch.nn.init.zeros_(cur_layer.bias)
        
        
        
        # put in list then initialize and register
        # for i in range(num_layer + 1):
        #     layer = (
        #         nn.Parameter(torch.empty(num_experts, inter_size if i!=0 else input_size, hidden_size if i!=num_layer else output_size)), # num experts #
        #         nn.Parameter(torch.empty(num_experts, hidden_size if i!=num_layer else output_size)),
        #         self.activation if i < num_layer else None 
        #     )
        #     self.decoder_layers.append(layer)

        # for index, (weight, bias, _) in enumerate(self.decoder_layers):
        #     index = str(index)
        #     stdv = 1. / math.sqrt(weight.size(1))
        #     weight.data.uniform_(-stdv, stdv)
        #     # bias.data.uniform_(-stdv, stdv)
        #     # bias.data.uniform_(-stdv, stdv)
        #     torch.nn.init.zeros_(bias)
        #     self.register_parameter("w" + index, weight)
        #     self.register_parameter("b" + index, bias)

        # gate_hsize = kargs['actor_gate_hidden_layer_size']
        # self.gate = nn.Sequential(
        #     nn.Linear(input_size, gate_hsize),
        #     nn.ELU(),
        #     nn.Linear(gate_hsize, gate_hsize),
        #     nn.ELU(),
        #     nn.Linear(gate_hsize, num_experts),
        # )
    
    def forward(self, ts):
        
        if not isinstance(ts, np.ndarray) and (not isinstance(ts, torch.Tensor)):
            ts = torch.ones(1, dtype = torch.long, device = self.frame_data_embedding_layer.weight.device)*ts
        
        # ts : nn_bsz 
        elif isinstance(ts, int):
            ts = torch.ones(1, dtype = torch.long, device = self.frame_data_embedding_layer.weight.device)*ts
        elif isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).long().to(self.frame_data_embedding_layer.weight.device)
        elif len(ts.size()) == 0:
            ts = ts.unsqueeze(0)
        
        layer_out = self.frame_data_embedding_layer(ts.long())
        
        # if self.is_delta:
        #     upd_layer_out = []
        #     for i_bsz in range(layer_out.size(0)):
        #         cur_ts = ts[i_bsz]
        #         if cur_ts == 0:
        #             upd_layer_out.append(layer_out[i_bsz])
        #         else:
        #             prev_acc_states = self.frame_data_embedding_layer(torch.arange(cur_ts).to(layer_out.device)).detach()
        #             prev_acc_states = prev_acc_states.sum(dim = 0)
        #             upd_layer_out.append(layer_out[i_bsz] + prev_acc_states)
        #     layer_out = torch.stack(upd_layer_out, dim = 0)
            
                    
        
        # assert len(c.shape) > 1
        
        
        
        # ''' original mixture of experts '''
        # # coefficients = F.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)
        # # layer_out = c
        # # for (weight, bias, activation) in self.decoder_layers:
        # #     input = z if layer_out is None else torch.cat((z, layer_out), dim=-1)
        # #     input = F.layer_norm(input, input.shape[1:])
        # #     mixed_bias = contract('be,ek->bk',coefficients, bias)
        # #     mixed_input = contract('be,bj,ejk->bk', coefficients, input, weight)
        # #     out = mixed_input + mixed_bias
        # #     layer_out = activation(out) if activation is not None else out
        # ''' original mixture of experts '''

        # # input = torch.cat([z,c], dim=-1)
        
        # # print(f"input: {input.size()}, weight: {self.decoder_layers[0].weight.size()}, bias: {self.decoder_layers[0].bias.size()}")
        
        # layer_out = c
        
        # for i_layer, layer in enumerate(self.decoder_layers):
        #     if isinstance(layer, nn.Linear):
        #         input = torch.cat([z,layer_out], dim = -1)
        #         layer_out = layer(input)
        #     else:
        #         layer_out = layer(layer_out)
        
        # # layer_out = self.decoder_layers(input)
        
        
        
        return layer_out
    
