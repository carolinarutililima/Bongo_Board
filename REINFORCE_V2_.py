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
  def __init__(self,input_shape,action_shape):
    super().__init__()
    

    
    self.model = nn.Sequential(
        nn.Linear(input_shape[0],64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,action_shape),
        nn.Softmax(dim = 1)
    )
    

    self.saved_log_probs = []
    self.rewards = []


  def forward(self,x):
    return self.model(x)  


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    
    probs = model(state)
    
    m = Categorical(probs)
    
    action = m.sample()
    
    model.saved_log_probs.append(m.log_prob(action))
    
    return action.item()


def finish_episode():
    R = 0
    model_loss = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + 0.9 * R
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


obs_shape = env.observation_space.shape
action_shape = env.action_space.n
model = Reinforce(obs_shape, action_shape)

#model = Reinforce()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()



def train():
    running_reward = 10


    for i_episode in range(500):
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
        if running_reward > 500:
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