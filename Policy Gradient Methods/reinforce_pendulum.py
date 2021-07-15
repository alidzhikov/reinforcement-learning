import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import deque
import torch.optim as optim
env = gym.make('Pendulum-v0')



class PolicyRealAI(nn.Module):
    def __init__(self):
        super().__init__()
        common_layers =  [nn.Linear(3, 72), nn.ReLU()]
        mean = common_layers + [nn.Linear(72, 1)]
        std = common_layers + [nn.Linear(72, 1), nn.Softplus()]
        self.mean = nn.Sequential(*mean)
        self.std = nn.Sequential(*std)
        
    def forward(self, x):
        # logits = self.linear_relu_stack(x)
        return self.mean(x), self.std(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mean, std = self.forward(state)

        m = Normal(mean, std)
        # m = Normal(0, 1)
        action = m.sample()
        # action = mean + std*action
        return [action.item()], m.log_prob(action).requires_grad_()
    
    
    

policy = PolicyRealAI()
optimizer = optim.Adam(policy.parameters(), lr=80)



def reinforce(n_episodes=4, max_t=50, gamma=1.0, print_every=20, batch_size=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_batch in range(1, batch_size+1):
        policy_loss = []
        for i_episode in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = env.reset()
            for t in range(max_t):
                # env.render();
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break 
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
        # env.close()
            discounts = [gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])
        
        
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_batch % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_batch, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores


reinforce()



# state = env.reset()
# for t in range(100000):
#     action, _ = policy.act(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         print(t)
#         break 

# env.close()