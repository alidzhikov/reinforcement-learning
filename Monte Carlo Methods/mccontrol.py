import numpy as np
import gym
from collections import defaultdict
import plotutils
import matplotlib

matplotlib.style.use('ggplot')

env = gym.make('Blackjack-v0')

Q = defaultdict(lambda: np.zeros(2))
ret_sum = defaultdict(float)
ret_count = defaultdict(float)
# policy = defaultdict(lambda: np.zeros(2))
epsilon = 0.1
policy = np.ones((22, 11, 2, 2))
policy[-2:] = policy[-2:] * 0
action_space = 2

def update_policy(policy, s, A):
    policy[s][A] = 1 - epsilon + epsilon/action_space
    policy[s][1-A] = epsilon/action_space
   
get_action = lambda s: round(np.random.choice([0,1], p=policy[s]))

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


for x in range(10000):
    episode = gen_episode(policy)
    first_occur = []
    _, _, G = episode[len(episode)-1]
    for s, a, r in episode:
        if s not in first_occur and s[0] > 11:
            first_occur.append(s)
            ret_sum[s] += G
            ret_count[s] += 1
            Q[s][a] = ret_sum[s]/ret_count[s]
            A = np.argmax(Q[s])
            update_policy(policy, s, A)
# plotutils.plot_value_function(Q, title="10,000 Steps")