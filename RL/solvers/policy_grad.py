import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 8)
        self.fc2 = nn.Linear(8, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)


def get_return(rewards, gamma=1):
    size = len(rewards)
    discounted_return = torch.zeros(size, dtype=torch.float32)
    g = 0
    for i in reversed(range(size)):
        g = rewards[i] + (gamma * g)
        discounted_return[i] = g
    return discounted_return


class MCPG(nn.Module):
    def __init__(self, env, batch_size=100, lr=0.02, gamma=1):
        super().__init__()
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.setup(n_inputs, n_outputs, lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.rewards = []
        self.reset_batch()

    def setup(self, n_inputs, n_outputs, lr):
        self.policy = Policy(n_inputs, n_outputs)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def reset_batch(self):
        self.returns = []
        self.probs = []

    def step(self, observation, reward=None):
        if reward != None:
            self.rewards.append(reward)
        observation = torch.tensor(observation, dtype=torch.float32)
        prob = self.policy(observation)
        m = Categorical(prob)
        action = m.sample().item()
        self.probs.append(prob[action])
        return action

    def update(self, reward):
        self.rewards.append(reward)
        total_reward = sum(self.rewards)
        self.returns.extend(get_return(self.rewards, self.gamma))
        self.rewards = []
        if len(self.returns) >= self.batch_size:
            self.update_weights()
            self.reset_batch()
        return total_reward

    def update_weights(self):
        self.opt.zero_grad()
        score = self.calculate_score()
        score.backward()
        self.opt.step()

    def calculate_score(self):
        G = torch.stack(self.returns)
        p = torch.stack(self.probs)
        print(p.shape, G.shape)
        return -(G * torch.log(p)).mean()


class Value(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MCPGBaseline(MCPG):
    def setup(self, n_inputs, n_outputs, lr):
        self.policy = Policy(n_inputs, n_outputs)
        self.value = Value(n_inputs)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

    def reset_batch(self):
        super().reset_batch()
        self.values = []

    def step(self, observation, reward=None):
        observation = torch.tensor(observation, dtype=torch.float32)
        self.values.append(self.value(observation))
        return super().step(observation, reward=reward)

    def calculate_score(self):
        G = torch.stack(self.returns)
        p = torch.stack(self.probs)
        V = torch.stack(self.values)
        error = G - V
        mse = (error ** 2).mean()
        return -(error.detach() * torch.log(p)).mean() + mse
