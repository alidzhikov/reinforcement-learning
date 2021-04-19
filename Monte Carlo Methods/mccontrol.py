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
epsilon = 0.1
policy = np.ones((22, 11, 2, 2))
policy = np.array([obs + [-1, 0] if i <21 else obs + [0, -1] for i, obs in enumerate(policy)])
action_space = 2


def update_policy(policy, s, A):
    policy[s][A] = 1 - epsilon + epsilon/action_space
    policy[s][1-A] = epsilon/action_space
   

def get_action(s):
    return round(np.random.choice([0,1], p=policy[s]))


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


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotutils.plot_value_function(V, title="Optimal Value Function")