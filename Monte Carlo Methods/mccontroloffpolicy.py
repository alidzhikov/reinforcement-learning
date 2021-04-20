# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:33:17 2021

@author: Mustafa
"""
import numpy as np
import gym
from collections import defaultdict
import plotutils
import matplotlib

matplotlib.style.use('ggplot')

env = gym.make('Blackjack-v0')
action_space = 2
policyG = np.zeros((22, 11, 2, 2))
policyR = np.ones((22, 11, 2, 2)) * (1 / action_space)
Q = defaultdict(lambda: np.zeros(2))
C = defaultdict(lambda: np.zeros(2))


def update_policy(policy, s, A):
    policy[s][A] = 1
    policy[s][1-A] = 0

get_action = lambda s: round(np.random.choice([0,1], p=policyR[s]))
def gen_episode(policy):
    s = env.reset()
    episode = []
    while True:
        a = get_action(tuple(map(int, s)))
        (sp, r, done, _) = env.step(a)
        episode.append((tuple(map(int, s)), a, r))
        s = sp
        if done:
            break
    return episode


for x in range(500000):
    episode = gen_episode(policyR)
    first_occur = []
    G = 0
    W=1
    for i, (s, a, r) in enumerate(episode):
        if s not in first_occur and s[0] > 11:
            G = sum([x[2] for x in episode[i:]])
            first_occur.append(s)
            C[s][a] = C[s][a] + W
            Q[s][a] = Q[s][a] + W/C[s][a] * (G - Q[s][a])
            A = np.argmax(Q[s])
            update_policy(policyG, s, A)
            if not A == a:
                break
            W = 1/policyR[s][a]
            
# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotutils.plot_value_function(V, title="Optimal Value Function")