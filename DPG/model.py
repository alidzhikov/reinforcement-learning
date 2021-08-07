# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:00:59 2021

@author: Mustafa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DPGNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DPGNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_size)
            )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
        