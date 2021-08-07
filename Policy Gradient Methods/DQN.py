# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 10:41:42 2021

@author: Mustafa
"""

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)