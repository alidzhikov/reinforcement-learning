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
                    v += pp*(r + V[sp[0]][sp[1]])
                delta = max(delta, np.abs(v - V[xi][yi]))
                V[xi][yi] = v
        print(delta)
        if delta < theta:
            break;
    print(V)
    return V
    
V = policy_eval(env, policy)

def policy_improvement(env, policy, V):
    greedy_policy = np.zeros(env.S.shape + (4,))
    while True:
        policy_stable = True
        for xi, xv in enumerate(env.S):
            for yi, s in enumerate(xv):
                a = np.argmax(policy[xi][yi])
                a_vals = []
                for pi, pp in enumerate(policy[xi][yi]):
                    sp,r = env.step([xi, yi],env.A[pi])
                    a_vals.append(r + V[sp[0]][sp[1]])
                a_greedy = np.argmax(a_vals)
                greedy_policy[xi][yi][a_greedy] = 1
                if not a == a_greedy:
                    policy_stable = False
        if policy_stable:
            break;
        return greedy_policy
    
greedy_policy = policy_improvement(env, policy, V)
V = policy_eval(env, greedy_policy)