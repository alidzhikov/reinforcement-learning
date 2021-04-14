# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

GOAL = 100
S = np.arange(GOAL)
V = np.zeros(GOAL)
V[len(V) - 1] = 1
policy = np.zeros(GOAL)
ph = 0.4
theta = 0.1
fig, axs = plt.subplots(2)
while True:
    delta = 0
    for i, s in enumerate(S):
        if not 0 < i < GOAL:
            continue
        v = V[i]
        action_values = []
        actions = np.arange(1, (s + 1 if s < 50 else GOAL - s))
        for a in actions:
            r = 0            
            action_values.append(
                ph * V[a + s] + (1 - ph) * V[s - a])
        # policy[s-1] = np.argmax(action_values)
        V[i] = np.max(action_values)
        delta = max(delta, np.abs(v - V[i]))
    print(delta)
    
    fig.suptitle('value F when ph = ' + str(ph))
    axs[0].plot(S,V)
    if delta < theta:
        break;
axs[1].plot(S, policy)
