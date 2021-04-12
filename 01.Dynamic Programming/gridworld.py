# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:29:44 2021

@author: Mustafa
"""

import numpy as np
states = np.arange(1, 15, 1)
state = 1
actions = [1, -1, 4, -4]
terminal = [0, 15]

def reset():
    state = np.random.randint(1,15)
    return state


def step(action):
    next_state = state + action
    if states.min() < next_state > states.max():
        reward = -1
    else if next_state == 0 or next_state == 15:
        reward = 1
    
    return 0,3,4,5 