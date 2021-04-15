import numpy as np
import gym
env = gym.make('Blackjack-v0')
obs = env.reset()

SHAPE = (22, 11, 2)
V = np.zeros(SHAPE)
policy = np.ones(SHAPE)
policy[-2:] = policy[-2:] * 0
R = {}

def gen_episode(policy):
    s = env.reset()
    episode = []
    while True:
        a = round(policy[tuple(map(int, s))])
        (sp, r, done, _) = env.step(a)
        episode.append((tuple(map(int, s)), a, r))
        s = sp
        if done:
            break
    return episode


for x in range(100000):
    episode = gen_episode(policy)
    print(episode)
    first_occur = []
    G = 0
    for s, a, r in episode:
        G += r
        if s not in first_occur:
            first_occur.append(s)
            if s in R:
                R[s].append(G)
            else:
                R[s] = [G]
            V[s] = sum(R[s])/len(R[s])
  