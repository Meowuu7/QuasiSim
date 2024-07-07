# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet
# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# import gym
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        self.device = device
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.Tanh(),
                nn.Linear(emb_size, emb_size),
                nn.Tanh(),
                # nn.Linear(emb_size, emb_size),
                # nn.Tanh(),
                nn.Linear(emb_size, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        distribution = MultivariateNormal(action_mean, cov_mat)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), distribution_entropy


## TODO: action std; # use the same actions here ##
class PPO:
    def __init__(self, args):
        self.args = args
        # self.env = env
        self.device = self.args.device

        # self.state_dim = self.env.observation_space.shape[0]
        # self.action_dim = self.env.action_space.shape[0]
        
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        
        
        self.policy = ActorCritic(self.device ,self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def evaluate(self, state, action):   
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        
        action_logprobs, state_value, distribution_entropy = self.policy_old.evaluate(state.unsqueeze(0), action.unsqueeze(0))
        
        
        # action_mean = self.policy_old.actor(state)
        
        # action_var = self.policy_old.action_var.expand_as(action_mean)
        # cov_mat = torch.diag_embed(action_var).to(self.device)
        
        # distribution = MultivariateNormal(action_mean, cov_mat)
        
        # action_logprobs = distribution.log_prob(action)
        # distribution_entropy = distribution.entropy()
        # state_value = self.policy_old.critic(state)
        
        return action_logprobs.detach().item(), state_value, distribution_entropy
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards = rewards.float().squeeze()
        else:
            rewards = rewards.float().squeeze()
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        
        # print()
        
        # Optimize policy for K epochs:
        for _ in range(self.args.K_epochs):
            # print(f"_ = {_}")
            # Evaluating old actions and values :
            logprobs, state_values, distribution_entropy = self.policy.evaluate(old_states, old_actions)
            # print(f"logprobs = {logprobs}, state_values = {state_values}, distribution_entropy = {distribution_entropy}")
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # print(f"ratios = {ratios}")
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            # print(f"reward = {rewards}, state_values = {state_values}, advantages = {advantages}")
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                    + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                    - self.args.loss_entropy_c*distribution_entropy
            
            # print(f"{torch.min(surr1, surr2)}, {self.MseLoss(state_values, rewards)}, {distribution_entropy}")
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())