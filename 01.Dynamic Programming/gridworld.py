# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:29:44 2021

@author: Mustafa
"""

import numpy as np
A = {'top': (0,-1), 'right': (1,0), 'down': (0,1), 'left': (-1, 0)}
terminal_s = [(0,0), (4,4)]
S = s = None
s = None

def reset():
    S = np.zeros((4, 4))
    s = S[1,1]
    return state


def step(action):
    s_prime_i = np.add(state, action)
    if states.min() < next_state > states.max():
        reward = -1
    elif next_state == 0 or next_state == 15:
        reward = 1
    
    return 0,3,4,5 