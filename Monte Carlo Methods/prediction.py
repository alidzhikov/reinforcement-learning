import numpy as np
import gym
from collections import defaultdict
import plotutils
import matplotlib

matplotlib.style.use('ggplot')

env = gym.make('Blackjack-v0')

V = defaultdict(float)
ret_sum = defaultdict(float)
ret_count = defaultdict(float)
policy = np.ones((22, 11, 2))
policy[-2:] = policy[-2:] * 0

def gen_episode(policy):
    s = env.reset()
    episode = []
    while True:
        a = round(policy[tuple(map(int, s))])
        (sp, r, done, _) = env.step(a)
        episode.append((s, a, r))
        s = sp
        if done:
            break
    return episode


for x in range(10000):
    episode = gen_episode(policy)
    first_occur = []
    _, _, G = episode[len(episode)-1]
    for s, a, r in episode:
        if s not in first_occur and s[0] > 11:
            first_occur.append(s)
            ret_sum[s] += G
            ret_count[s] += 1
            V[s] = ret_sum[s]/ret_count[s]
 
plotutils.plot_value_function(V, title="10,000 Steps")