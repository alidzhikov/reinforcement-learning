# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:38:55 2021

@author: Mustafa
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

import torch.optim as optim
env = gym.make('MountainCar-v0')
env.seed(0)

state = env.reset()

obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n
state, reward, done, _ = env.step(2)



class Q_value():
    def __init__(self, n_acts):
        self.weights = torch.randn(19).requires_grad_()
        self.actions = torch.arange(1, n_acts+1)
        
    def get_feature_param(self, state, action):
        s1 = state[0]
        s12 = pow(s1,2)
        s2 = state[1]
        s22 = pow(s2,2)
        a = action
        a2 = pow(a,2)
        return torch.tensor([1,s1,s2,a,s1*s2,s1*a,s2*a,s12,s22,a2,s12*s2,
                             s1*s22,s12*a,s22*a,a2*s2,a2*s1,s12*s22,s12*a2,a2*s22])
    
    def get_value(self, state):
        f1 = self.get_feature_param(state, self.actions[0])
        f2 = self.get_feature_param(state, self.actions[1])
        f3 = self.get_feature_param(state, self.actions[2])
        ret1 = f1*self.weights
        ret2 = f2*self.weights
        ret3 = f3*self.weights
        return torch.sum(ret1), torch.sum(ret2), torch.sum(ret3)
    
    def get_return(self, state):
        ret1, ret2, ret3 = self.get_value(state)
        if ret1>ret2 and ret1>ret3:
            return ret1
        elif ret2>ret3:
            return ret2
        else:
            return ret3
    
       
    def get_action(self, state):
        q = torch.tensor([self.get_value(state)])
        if torch.bernoulli(torch.tensor([0.9])) == 1:
            action = torch.argmax(q).item()
        else:
            if torch.bernoulli(torch.tensor([0.5])):
                action = 0
            else:
                action = 2
        return action
    
q_value = Q_value(n_acts)

def estimate(n=8, n_episodes=150, max_t=300, gamma=1, alpha=0.5):
    for i_episode in range(1, n_episodes+1):
        rewards = []
        saved_actions = []
        saved_states = []
        state = env.reset()
        action = q_value.get_action(state)
        saved_actions.append(action)
        state, reward, done, _ = env.step(action)
        saved_states.append(state)
        rewards.append(reward)
        T = 9999
        tau = 0
        for t in range(0, max_t):
            if i_episode>140:
                env.render();
            if t < T:    
                action = q_value.get_action(state)
                saved_actions.append(action)
                state_t, reward_t, done, _ = env.step(action)
                saved_states.append(state_t)
                rewards.append(reward_t)
                if done:
                    T = t + 1
                else:
                    action_t = q_value.get_action(state_t)
                    saved_actions.append(action_t)
            tau = t - n + 1
            if tau >= 0:
                last_reward_i = torch.argmin(torch.tensor([tau+n, T]))                
                G = rewards[tau+1:last_reward_i+1]
                discounts = [pow(gamma, i-1) for i in range(len(G))]
                G = sum([a*b for a,b in zip(discounts, G)])
                if tau+n < T:
                    value = q_value.get_return(saved_states[tau+n])
                    G = G + pow(gamma,n)*value
                    value_tau =  q_value.get_return(saved_states[tau])
                    value_tau.backward()
                    gradient_update = alpha*(G-value_tau)*q_value.weights.grad
                   
                    q_value.weights = (q_value.weights + alpha*(G-value_tau)*q_value.weights.grad).clone().detach().requires_grad_()
            if tau == T-1:
                break
        if i_episode%50 == 0:
            print("Episode {} and reward {}".format(i_episode, sum(rewards)))
    env.close()
estimate()