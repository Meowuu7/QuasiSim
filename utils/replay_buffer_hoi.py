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
import torch as th

from utils.index_counter import index_counter


# 
class cycle_nd_queue():
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.content = None

    def append(self, item: np.ndarray, b_idx, e_idx) -> None:
        
        # item is tooooooo big
        if e_idx - b_idx > self.max_size:
            print(e_idx, b_idx, self.max_size)
            raise ValueError

        if self.content is None:
            shape = list(item.shape)
            shape[0] = self.max_size
            self.content = np.ones(shape, dtype= item.dtype) * -1


        b_idx = b_idx % self.max_size
        e_idx = e_idx % self.max_size
        if b_idx > e_idx:
            l = self.max_size - b_idx
            self.content[b_idx:] = item[:l]
            self.content[:e_idx] = item[l:]
        else:
            self.content[b_idx:e_idx] = item
    
    def __getitem__(self, idx):
        
        return self.content.__getitem__(idx%self.max_size)
    
    @property
    def shape(self):
        # added by jason huang. to make it easy for debugging
        return self.content.shape if self.content is not None else None

class ReplayBuffer(object):

    def __init__(self, keys, max_size=50000): ## replay buffer ##

        self.max_size = max_size
        self.content = {}
        for key in keys:
            self.content[key] = cycle_nd_queue(max_size)
        self.content['done'] = cycle_nd_queue(max_size)
        # for current training, the shape if of the following:
        # self.content['state']: (max_size, 20, 13)
        # self.content['action']: (max_size, 57)
        # self.content['target']: (max_size, 323)
        # self.content['done']: (max_size, )
        
        self.end_idx = 0
    
    def reset_max_size(self, max_size):
        #! this function can only be called before adding data
        self.max_size = max_size
    
    def clear(self):
        self.end_idx = 0
        self.terminals.contents = np.ones_like(self.terminals.content) * -1

    def add_trajectory(self, trajectory): ## add trajectory traj sampling ##
        num = trajectory['done'].shape[0] # done .shape 0
        e_idx = self.end_idx + num
        for key,value in trajectory.items():
            if key in self.content: ####### content[key] --- content done ########
                cur_content_val = self.content[key]
                print(f"key: {key}, val: {value.shape}, val_sample: {value[0]}")
                self.content[key].append(value, self.end_idx, e_idx)
        self.end_idx = e_idx % self.max_size
        
    
    def feasible_index(self, rollout_length):
        # calculate feasible index --- content done  ## calculate feasible index ##
        ## content don contnet rollout lenght ## 
        return index_counter.calculate_feasible_index(self.content['done'].content, rollout_length)
    
    def generate_data_loader(self, name_list, rollout_length, mini_batch_size, mini_batch_num):
        
        # sample rollout ## ## sam
        index = index_counter.sample_rollout(
                self.feasible_index(rollout_length), # ensure [i,i+rollout_length) is feasible 
                mini_batch_size* mini_batch_num, # total num of rollouts
                rollout_length 
                )
        
        res = []
        for name in name_list: ## batch_size x (rollout_lenght) x content_feat_dim ##
            res.append(  th.Tensor(self.content[name][index])) # shape (mini_batch_size* mini_batch_num, rollout_length, ...)
        dataset = th.utils.data.TensorDataset(*res)
        data_loader = th.utils.data.DataLoader(
            dataset, 
            mini_batch_size, 
            shuffle = False
            )
        list(data_loader)
        return data_loader



