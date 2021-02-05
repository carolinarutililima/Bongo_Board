import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from env_bong import Bong


#env = gym.make('Acrobot-v1')
env = Bong()


class Reinforce(nn.Module):
    
    def __init__(self):
        super(Reinforce, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    
    probs = model(state)
    
    m = Categorical(probs)
    
    action = m.sample()
    
    model.saved_log_probs.append(m.log_prob(action))
    
    return action.item()


def finish_episode():
    R = 0
    gamma = 0.9
    model_loss = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(model.saved_log_probs, returns):
        model_loss.append(-log_prob * R)
    optimizer.zero_grad()
    model_loss = torch.cat(model_loss).sum()
    model_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]




model = Reinforce()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()



def train():
    running_reward = 10


    for i_episode in range(1000):
        state, ep_reward = env.reset(), 0
        
        for t in range(1, 10000):  # Don't infinite loop while learning
            env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            
            model.rewards.append(reward)
            ep_reward += reward
            

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        finish_episode()
        

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > 250:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


train()


done = False
cnt = 0

observation = env.reset()

while not done:
    cnt += 1
    env.render()
    action = select_action(observation)
    observation, reward, done, _ = env.step(action)
    if done:
        break
print(f"Game lasted {cnt} moves")