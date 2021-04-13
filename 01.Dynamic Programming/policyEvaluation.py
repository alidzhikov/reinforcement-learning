# -*- coding: utf-8 -*-

import numpy as np
from gridworld import Gridworld 

env = Gridworld()
env.reset()
policy = np.ones(env.S.shape + (4,)) * 0.25

def policy_eval(env, policy):
    theta = 0.0001
    V = np.zeros(env.S.shape)
    while True:
        delta = 0
        for xi, xv in enumerate(env.S):
            for yi, s in enumerate(xv):
                if env.is_terminal_s([xi,yi]):
                    continue
                v = 0
                for pi, pp in enumerate(policy[xi][yi]):
                    sp,r = env.step([xi, yi],env.A[pi])
                    next_v = V[sp[0]][sp[1]] if not sp is None else 0
                    v += (pp*(r + next_v))
                delta = max(delta, np.abs(v - V[xi][yi]))
                V[xi][yi] = v
        print(delta)
        if delta < theta:
            break;
    print(V)
    
policy_eval(env, policy)