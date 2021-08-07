# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:38:55 2021

@author: Mustafa
"""
import sys
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

import torch.optim as optim
env = gym.make('MountainCar-v0')
env.seed(0)


class Agent():
    def __init__(self, env, epsilon=0.3,
                 epsilon_decay_rate=0.2,
                 epsilon_min=0.1,
                 alpha=0.1):
        
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n
        self.env = env
        self.state_grid = self.create_uniform_grid(env.observation_space.low,
                                              env.observation_space.high)
        self.state_shape = tuple(len(dim)+1 for dim in self.state_grid)
        self.action_shape = env.action_space.n
        self.qv = np.zeros(shape=(self.state_shape+(self.action_shape,)))
        self.weights = torch.randn(19).requires_grad_()
        self.actions = torch.arange(1, n_acts+1)
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon = self.epsilon_initial = epsilon
        self.alpha = 0.1
        self.greedy_actions = 0
        self.exploration_actions = 0
        
    def discretize(self, sample):
        return tuple([np.digitize(sample[i], self.state_grid[i]) for i in range(len(sample))])
    
    def create_uniform_grid(self, low, high, bins=(10, 10)):
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
        print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
        return grid

    def get_feature_param(self, state, action):
        s1 = state[0]
        s12 = pow(s1,2)
        s2 = state[1]
        s22 = pow(s2,2)
        a = action
        a2 = pow(a,2)
        return torch.tensor([1,s1,s2,a,s1*s2,s1*a,s2*a,s12,s22,a2,s12*s2,
                             s1*s22,s12*a,s22*a,a2*s2,a2*s1,s12*s22,s12*a2,a2*s22])
    
    def get_value(self, state):
        f1 = self.get_feature_param(state, self.actions[0])
        f2 = self.get_feature_param(state, self.actions[1])
        f3 = self.get_feature_param(state, self.actions[2])
        ret1 = f1*self.weights
        ret2 = f2*self.weights
        ret3 = f3*self.weights
        return torch.sum(ret1), torch.sum(ret2), torch.sum(ret3)
    
    def get_return(self, state, action=None):
        state = self.discretize(state)
        if action:
            return self.qv[state][action]
        max_action = np.argmax(self.qv[state])
        return self.qv[state][max_action]
   
       
    def get_action(self, state):
        state = self.discretize(state)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay_rate
        if torch.bernoulli(torch.tensor([self.epsilon])) == 1:
            action = np.argmax(self.qv[state])
            self.greedy_actions += 1
        else:
            action = env.action_space.sample()
            self.exploration_actions += 1
        return action
    
    def update(self, state, action, G, value):
        state = self.discretize(state)
        self.qv[state][action] += self.alpha*(G - value)
    
    def reset_epsilon(self, i_episode):
        self.epsilon = self.epsilon_initial
        
agent = Agent(env)
# fe = agent.get_action(env.observation_space.sample())

def estimate(n=8, n_episodes=2000, max_t=300, gamma=0.99):
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        action = agent.get_action(state)
       
        
        tau = 0
        agent.reset_epsilon(i_episode)

        rewards = []
        saved_actions = [action]
        saved_states = [state]

        T = float('inf')
        for t in range(0, max_t):
            if i_episode>133140:
                env.render();
            if t < T:    
                saved_actions.append(action)
                state_t, reward, done, _ = env.step(action)
                saved_states.append(state_t)
                rewards.append(reward)
                if done:
                    T = t + 1
                else:
                    action_t = agent.get_action(state_t)
                    saved_actions.append(action_t)
            tau = t - n + 1
            if tau >= 0:
                last_reward_i = int(torch.min(torch.tensor([tau+n, T])).item())                
                G = rewards[tau+1:last_reward_i+1]
                discounts = [pow(gamma, i-1) for i in range(len(G))]
                G = sum([a*b for a,b in zip(discounts, G)])
                if tau+n < T:
                    value = agent.get_return(saved_states[tau+n], saved_actions[tau+n])
                    G = G + pow(gamma,n)*value
                    value_tau = agent.get_return(saved_states[tau])

                    agent.update(saved_states[tau], saved_actions[tau], G, value_tau)
                    
            if tau == T-1:
                break
            state = state_t
            action = action_t
        if i_episode%50 == 0:
            print("Episode {} and reward {}".format(i_episode, sum(rewards)))
    env.close()
estimate()

def discretize(sample,grid):
    return [np.digitize(sample[i], grid[i]) for i in range(len(sample))]
    
def create_uniform_grid(low, high, bins=(10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid

# state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))

class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)
        self.greedy_actions = 0
        self.exploration_actions = 0

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                self.exploration_actions += 1
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                self.greedy_actions += 1
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

    
# q_agent = QLearningAgent(env, state_grid)

def run(agent, env, num_episodes=2000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False
        
        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)
            
        # Save final score
        scores.append(total_reward)

        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()
    return scores

# scores = run(q_agent, env)



import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)
# Plot scores obtained per episode
# plt.plot(scores); plt.title("Scores");

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean

# rolling_mean = plot_scores(scores)


# Run in test mode and analyze scores obtained
# test_scores = run(q_agent, env, num_episodes=2, mode='test')
# print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
# _ = plot_scores(test_scores, rolling_window=10)


def plot_q_table(q_table):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(q_image, cmap='jet');
    cbar = fig.colorbar(cax)
    for x in range(q_image.shape[0]):
        for y in range(q_image.shape[1]):
            ax.text(x, y, q_actions[x, y], color='white',
                    horizontalalignment='center', verticalalignment='center')
    ax.grid(False)
    ax.set_title("Q-table, size: {}".format(q_table.shape))
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')


plot_q_table(agent.qv)

# TODO: Create a new agent with a different state space grid
# state_grid_new =  create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))
# q_agent_new = QLearningAgent(env, state_grid_new)
# q_agent_new.scores = []

# Train it over a desired number of episodes and analyze scores
# Note: This cell can be run multiple times, and scores will get accumulated
# q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
# rolling_mean_new = plot_scores(q_agent_new.scores)


# Run in test mode and analyze scores obtained
# test_scores = run(q_agent_new, env, num_episodes=100, mode='test')
# print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
# _ = plot_scores(test_scores)

# Visualize the learned Q-table
# plot_q_table(q_agent_new.q_table)

# state = env.reset()
# score = 0
# for t in range(200):
#     action = q_agent_new.act(state, mode='test')
#     env.render()
#     state, reward, done, _ = env.step(action)
#     score += reward
#     if done:
#         break 
# print('Final score:', score)
# env.close()