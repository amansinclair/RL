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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_return(rewards, gamma=1):
    discounted_return = torch.zeros(rewards.shape, dtype=torch.float32)
    g = 0
    for i in reversed(range(len(rewards))):
        g = rewards[i] + (gamma * g)
        discounted_return[i] = g
    return discounted_return


def play_single(policy, value, env, view=False):
    probs = torch.zeros((1000), dtype=torch.float32)
    rewards = torch.zeros((1000), dtype=torch.float32)
    observation = torch.tensor(env.reset(), dtype=torch.float32)
    values = torch.zeros((1000), dtype=torch.float32)
    done = False
    i = 0
    while not done and i < 1000:
        if view:
            env.render()
            time.sleep(0.1)
        p = policy(observation)
        v = value(observation)
        values[i] = v
        m = Categorical(p)
        action = m.sample().item()
        probs[i] = p[action]
        observation, reward, done, info = env.step(action)
        observation = torch.tensor(observation, dtype=torch.float32)
        rewards[i] = reward
        i += 1
    g = get_return(rewards[:i])
    return (probs[:i], values[:i], g, rewards[:i].sum())


def play(policy, value, env, n_episodes=5):
    probs = []
    gs = []
    vs = []
    score = 0
    for i in range(n_episodes):
        prob, v, g, r = play_single(policy, value, env)
        probs.append(prob)
        gs.append(g)
        vs.append(v)
        score += r
    return score / n_episodes, torch.cat(probs), torch.cat(vs), torch.cat(gs)


def train_batch(policy, value, opts, env):
    for opt in opts:
        opt.zero_grad()
    avg_return, probs, vs, gs = play(policy, value, env)
    error = gs - vs
    mse = (error ** 2).mean()
    mse.backward()
    score = -(error.detach() * torch.log(probs)).mean()
    # print(mean_error, score)
    # mean_error.backward()
    print(avg_return)
    score.backward()
    for opt in opts:
        opt.step()
    return avg_return


def view_round(policy, value, env):
    with torch.no_grad():
        play_single(policy, value, env, True)


policy = Policy()
value = Value()
opt_p = optim.Adam(policy.parameters(), lr=0.1)
opt_v = optim.Adam(value.parameters(), lr=0.1)
opts = [opt_p, opt_v]
env = gym.make("CartPole-v1")
scores = []
n_batches = 50
n_episodes = 5
for i in range(n_batches):
    score = train_batch(policy, value, opts, env)
    scores.append(score)
print(scores)
view_round(policy, value, env)
env.close()
plt.plot([i for i in range(0, n_batches * n_episodes, n_episodes)], scores)
plt.show()
