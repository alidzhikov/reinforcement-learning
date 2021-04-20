# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from windygridworld import WindyGridworldEnv
import plotutils
import sys

env = WindyGridworldEnv()
# UP, RIGHT, DOWN, LEFT
actions = [0, 1, 2, 3]
def get_action(s, Q, epsilon):
    policy = np.ones(env.action_space.n) * epsilon / env.action_space.n
    A = np.argmax(Q[s])
    policy[A] += 1-epsilon
    return round(np.random.choice(actions, p=policy))

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotutils.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    for x in range(num_episodes):
        if(x+1) % 100 == 0:
            print("\rEpisode {}/{}".format(x+1,num_episodes), end="")
            sys.stdout.flush()
        s = env.reset()
        a = get_action(s, Q, epsilon)
        done = False
        ep_length = 0
        ep_reward = 0
        while not done:
            s_prm, r, done, _ = env.step(a)
            a_prm = get_action(s_prm, Q, epsilon)
            ep_length += 1
            ep_reward += r
            Q[s][a] += alpha * (r + discount_factor * Q[s_prm][a_prm] - Q[s][a])
            s = s_prm
            a = a_prm
            stats.episode_rewards[x] += r
        stats.episode_lengths[x] =  ep_length
        
    return Q, stats

Q, stats = sarsa(env, 200)

plotutils.plot_episode_stats(stats)
