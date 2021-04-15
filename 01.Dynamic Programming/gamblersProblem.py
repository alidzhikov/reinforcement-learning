# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

GOAL = 100
S = np.arange(GOAL + 1)
V = np.zeros(GOAL + 1)
V[len(V) - 1] = 1
policy = np.zeros(GOAL + 1)
ph = 0.4
theta = 0.00000001
fig, axs = plt.subplots(2)
sweep = 0
while True:
    delta = 0
    sweep += 1
    for s in S[1:GOAL]:
        v = V[s]
        action_values = []
        actions = np.arange(min(s, GOAL-s)+1)
        for a in actions:
            action_values.append(
                ph * V[a + s] + (1 - ph) * V[s - a])
        V[s] = np.max(action_values)
        delta = max(delta, np.abs(v - V[s]))
    print(delta)
    
    fig.suptitle('value F when ph = ' + str(ph))
    axs[0].plot(S,V, label='sweep {}'.format(sweep))
    if delta < theta:
        print('sweep {}'.format(sweep))
        break;

for s in S[1:GOAL]:
    actions = np.arange(min(s, GOAL - s) + 1)
    action_returns = []
    for a in actions:
        action_returns.append(
            ph * V[s + a] + (1 - ph) * V[s - a])
    policy[s] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]


axs[1].plot(S, policy)
