# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:02:02 2021

@author: Mustafa
"""

import numpy as np
import random
from collections import namedtuple, deque

from model import DPGNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.dpgnetwork_policy = DPGNetwork(state_size, action_size, seed).to(device)
        self.dpgnetwork_value = DPGNetwork(state_size, action_size, seed).to(device)
        self.optimizer_policy = optim.Adam(self.dpgnetwork_policy.parameters(), lr=LR)
        self.optimizer_value = optim.Adam(self.dpgnetwork_value.parameters(), lr=LR)

    
    def step(self, state, action, reward, next_state, done):
        next_state2 = torch.from_numpy(next_state).float().detach()
        state2 = torch.from_numpy(state).float().detach()
        value_t1 = self.dpgnetwork_value(next_state2)
        value_t = self.dpgnetwork_value(state2)
        
        
        delta = reward + GAMMA*value_t1 - value_t
        
        loss_value = delta.detach()*value_t1
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()
        
        policy = self.dpgnetwork_policy(state2)
        
        value_t_clone = self.dpgnetwork_value(state2)
        loss_policy = policy*value_t_clone
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.dpgnetwork_policy.eval()
        with torch.no_grad():
            action = self.dpgnetwork_policy(state)
        self.dpgnetwork_policy.train()
        print(action)
        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        prediction = self.qnetwork_local(states).gather(1, actions)
        target_prediction = self.qnetwork_target(next_states).max(1)[0].detach()
        target_prediction[torch.nonzero(dones,as_tuple=True)[0]] = 0
        expected_state_action_values = (target_prediction * gamma) + rewards.squeeze()
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(prediction, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

   