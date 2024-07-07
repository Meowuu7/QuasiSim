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
import operator
from ..Utils import pytorch_utils as ptu

class TrajectorCollector():
    def __init__(self, **kargs) -> None:
        self.reset(**kargs)
    
    def reset(self, venv, **kargs):
        # set environment and actor
        self.env = venv # 
        
        # set property
        self.with_noise = kargs['runner_with_noise']
        
        try:
            self.use_ana = kargs['use_ana']
        except:
            self.use_ana = False    
    
    
    # @torch.no_grad
    def trajectory_sampling(self, sample_size, actor):
        ### fithe mpi index is zero then doen teh sampling here ##
        cnt = 0
        res = [] # sample one trajectory with sample_size # 
        print(f"trjaectory collector --- sampling trajectory with sample_size: {sample_size}") ## trajectory sampling ##
        while cnt < sample_size: # ## sample one trajectory ## # 
            trajectory = self.sample_one_trajectory(actor) ## sample one trajectory ##
            res.append(trajectory) # len highly related to the policy --- so, do action values affect the trajectory results? ##
            cnt+= len(trajectory['done'])
        
        # sample trajectory 
        # res = functools.reduce(operator.add, map(collections.Counter, res))
        res_dict = {} # res dict # 
        for key in res[0].keys(): # 
            # if self.use_ana:
            #     res_dict[key] = torch.cat( list(map(operator.itemgetter(key), res)) , dim = 0)
            # else:
            #     res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
            res_dict[key] = np.concatenate( list(map(operator.itemgetter(key), res)) , axis = 0)
        
        # states 
        
        # if self.use_ana:
        #     res_dict = {
        #         key: ptu.to_numpy(value) for key, value in res_dict.items()
        #     }
        return res_dict
    
    def eval_one_trajectory(self, actor):
        saver = self.env.get_bvh_saver()
        observation, info = self.env.reset()
        while True: 
            saver.append_no_root_to_buffer()

            # when eval, we do not hope noise... ## act deterministic ## # # act deter #
            # act deterministic # --- # 
            action = actor.act_determinastic(observation)
            action = ptu.to_numpy(action).flatten()
            new_observation, rwd, done, info = self.env.step(action)
            observation = new_observation
            if done:
                break ## break ##
        return saver
    
    ## 
    # st_fr_to_controls = self.eval_multi_traj_enum_st(actor) # expect that the world model should learn useful things fro mthe sampled trajectory 
    def eval_multi_traj_enum_st(self, actor):
        st_fr_to_controls = {}
        # for stfr in range(0, 40): ## evaluated ##
        for stfr in range(0, 1): ## evaluated ##
        # for stfr in range(30, 40):
            observation, info = self.env.reset(frame=stfr)
            states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)]
            robot_controls = []
            while True:
                action = actor.act_determinastic(observation)
                ## 
                ## obs actions steps ## # from actions to the observations ##
                if self.use_ana:
                    # print(f"evn.counter: {self.env.counter}") #  use the analytical model to get the states and actions #
                    # use the analytical # action optimization # -- not useful information for optimizing actions # optimizing actions #
                    # 
                    # forward step the actiion # 
                    new_observation, rwd, done, info = self.env.step_core(action.detach().squeeze(0))
                    action = ptu.to_numpy(action).flatten()
                    cur_step_pd_control = new_observation['state'] # states are controls 
                    cur_step_pd_control = ptu.to_numpy(cur_step_pd_control).flatten()
                else:
                
                    action = ptu.to_numpy(action).flatten() ## action to numpy ##
                    # forward step the actiion # 
                    # new_observation, rwd, done, info = self.env.step(action)
                    
                    # observation, cur_step_pd_control, reward, done, info  
                    new_observation, cur_step_pd_control, rwd, done, info   = self.env.step_core_new_wcontrol(action)

                # ## ptu actions and states ##3
                # action = ptu.to_numpy(action).flatten() ## action to numpy ##
                
                # states.append(observation['state']) ## state ##
                # actions.append(action) ## action ##
                # targets.append(observation['target']) ## target ##
                
                if self.use_ana:
                    states.append(observation['state'].detach().cpu().numpy()) ## state ##
                    actions.append(action) ## action ##
                    targets.append(observation['target'].detach().cpu().numpy()) ## target ##
                else:
                    states.append(observation['state']) ## state ##
                    actions.append(action) ## action ##
                    targets.append(observation['target']) ## target ##
                
                
                # done with the stpe counter #
                robot_controls.append(cur_step_pd_control)
                rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target']) ## target observation; 
                rwds.append(rwd)
                dones.append(done)
                frame_nums.append(observation['frame_num'])
                observation = new_observation
                if done:
                    break
            
            # robot_controls = np.stack(robot_controls, axis=0) ## robot controls; ##
            # if self.use_ana:
            #     robot_controls = torch.stack(robot_controls, dim=0).detach().cpu().numpy() ## robot controls; ##
            # else:
            robot_controls = np.stack(robot_controls, axis=0) ## robot controls; ##
        
            
            print(f"sampling one trajectory with length: {len(dones)}, rewards: {rwds}")
            
            sampled_dict = { ## can get the initial state and use that to reset the env ##
                'state': states,
                'action': actions,
                'target': targets,
                'done': dones,
                'rwd': rwds,
                'frame_num': frame_nums,
                'robot_controls': robot_controls, ## 
            }
            st_fr_to_controls[stfr] = sampled_dict
        return st_fr_to_controls
    
    
    def eval_one_traj(self, actor):
        ## reset 
        observation, info = self.env.reset(frame=0)  # env reset # # reset to frmae 0 ##

        # observation, info = self.env.reset(frame=30)  # env reset # 
        
        states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)] ## 
        
        robot_controls = []
         
        while True: ## with noise ##
            action = actor.act_determinastic(observation)
            
            # if np.random.choice([True, False], p = [0.4, 0.6]):
            #     action = actor.act_prior(observation) ### act prior ###
            #     action = action + torch.randn_like(action) * 0.05
            
            
            ## obs actions steps ## # from actions to the observations
            if self.use_ana:
                # print(f"evn.counter: {self.env.counter}")
                # forward step the actiion # 
                new_observation, rwd, done, info = self.env.step_core(action.detach().squeeze(0))
                action = ptu.to_numpy(action).flatten()
                cur_step_pd_control = new_observation['state'] # states are controls 
                # done = ptu.to_numpy(done).flatten()
                cur_step_pd_control = ptu.to_numpy(cur_step_pd_control).flatten()
            else:
            
                action = ptu.to_numpy(action).flatten() ## action to numpy ##
                # forward step the actiion # 
                # new_observation, rwd, done, info = self.env.step(action)
                
                # observation, cur_step_pd_control, reward, done, info   # step with the control ## 
                new_observation, cur_step_pd_control, rwd, done, info   = self.env.step_core_new_wcontrol(action)

            # ## ptu actions and states ##3
            # action = ptu.to_numpy(action).flatten() ## action to numpy ##
            
            # states.append(observation['state']) ## state ##
            # actions.append(action) ## action ##
            # targets.append(observation['target']) ## target ##
            
            if self.use_ana:
                states.append(observation['state'].detach().cpu().numpy()) ## state ## ## state at frame 0 
                actions.append(action.detach().clone()) ## action ## # action at frame 0 (gt state at frmae 1)
                targets.append(observation['target'].detach().cpu().numpy()) ## target ## ## target at frame 0 (gt state at frame 1)
            else:
                states.append(observation['state']) ## state ##
                actions.append(action) ## action ##
                targets.append(observation['target']) ## target ##
                
            
            # print(f"action: {action}")
            
            # forward step the actiion # 
            # new_observation, rwd, done, info = self.env.step(action)
            
            # # observation, cur_step_pd_control, reward, done, info  
            # new_observation, cur_step_pd_control, rwd, done, info   = self.env.step_core_new_wcontrol(action)
            
            # done with the stpe counter #
            robot_controls.append(cur_step_pd_control) ## control at frame 0 (gt state at frame 1)
            
            rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target']) ## target observation; 
            rwds.append(rwd)
            dones.append(done)
            
            
            frame_nums.append(observation['frame_num']) # 1 
            
            
            
            obs_obj_states = observation['state'][-7:]
            obs_obj_trans = obs_obj_states[:3]
            obs_obj_rot = obs_obj_states[3:]
            print(obs_obj_trans, obs_obj_rot)
            
            ## act jpripr ##
            ## 
            observation = new_observation
            if done:
                break
        
        # if self.use_ana:
        #     robot_controls = torch.stack(robot_controls, dim=0).detach().cpu().numpy() ## robot controls; ##
        # else: # 
        robot_controls = np.stack(robot_controls, axis=0) ## robot controls; ##
        
        print(f"sampling one trajectory with length: {len(dones)}, rewards: {rwds}")
        
        return {
            'state': states,
            'action': actions,
            'target': targets,
            'done': dones,
            'rwd': rwds,
            'frame_num': frame_nums,
            'robot_controls': robot_controls, ## then to the con
        }
    
    
    # @torch.no_grad ## sample one trajectory ##
    def sample_one_trajectory(self, actor):
        # sampling -> perturbation around one trajectory ? #
        if actor.train_policy_only:
            observation, info = self.env.reset(0)
        else:
            observation, info = self.env.reset() # act from one gt trajectory right? # 
        # observation, info = self.env.reset(frame=30)
        
        states, targets, actions, rwds, dones, frame_nums = [[] for i in range(6)]
         
        while True:  
            if self.with_noise: # one trajectory # # act from the observation -> 
                ### wnoise ---- distribution sample ###
                action_distribution = actor.act_distribution(observation)
                action = action_distribution.sample()
            else:
                action = actor.act_determinastic(observation)
            
            # if np.random.choice([True, False], p = [0.4, 0.6]):
            #     action = actor.act_prior(observation) ### act prior ###
            #     action = action + torch.randn_like(action) * 0.05
            
            # act deterministic # 
            ## obs actions steps ## # from actions to the observations
            if self.use_ana:
                # forward step the actiion # 
                # new_observation, rwd, done, info = self.env.step(action.detach().squeeze(0))
                # action = ptu.to_numpy(action).flatten()
                new_observation, rwd, done, info = self.env.step_core(action.detach().squeeze(0))
                action = ptu.to_numpy(action).flatten()
                # cur_step_pd_control = new_observation['state'] # states are controls 
            else:
            
                action = ptu.to_numpy(action).flatten() ## action to numpy ##
                # forward step the actiion # 
                # new_observation, rwd, done, info = self.env.step(action)
                # step_core_new_wcontrol
                new_observation, cur_step_pd_control, rwd, done, info   = self.env.step_core_new_wcontrol(action)
                # new_observation, rwd, done, info   = self.env.step_core(action, using_yield=True)
            
            if self.use_ana:
                states.append(observation['state'].detach().cpu().numpy()) ## state ##
                actions.append(action.detach().clone()) ## action ##
                targets.append(observation['target'].detach().cpu().numpy()) ## target ##
            else:
                states.append(observation['state']) ## state ##
                actions.append(action) ## action ##
                targets.append(observation['target']) ## target ##
                
            
            # done with the stpe counter #
            rwd = actor.cal_rwd(observation = new_observation['observation'], target = observation['target']) ## target observation; 
            rwds.append(rwd)
            dones.append(done)
            # print(f"cur_frame_num: {observation['frame_num']}")
            frame_nums.append(observation['frame_num'])
            ## act jpripr ##
            ## 
            observation = new_observation
            if done:
                break
        
        print(f"sampling one trajectory with length: {len(dones)}, rewards: {rwds}, frame_nums: {frame_nums}")
        
        
        
        return {
            'state': states,
            'action': actions,
            'target': targets,
            'done': dones,
            'rwd': rwds,
            'frame_num': frame_nums
        }
            
            
            
            