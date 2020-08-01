import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, x):
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            probs = F.softmax(x, dim=-1)
            m = Categorical(probs)
            return m.sample().item()


class CEMAgent:
    def __init__(self, env, gamma=0.99, lr=0.01, batch_size=256):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.gamma = gamma
        self.policy = Policy(self.n_inputs, self.n_outputs)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.crit = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.cut_off_score = 0
        self.n_roll_outs = 1
        self.best_obs = []
        self.best_actions = []
        self.reset()

    def reset(self):
        self.obs = []
        self.rewards = []
        self.actions = []

    def __str__(self):
        return "CEMAgent"

    def step(self, obs, reward=None, is_done=False):
        obs = torch.tensor(obs, dtype=torch.float32)
        if reward:
            self.rewards.append(reward)
        action = None
        if not is_done:
            self.obs.append(obs)
            action = self.policy.get_action(obs)
            self.actions.append(action)
        else:
            self.update()
            self.reset()
        return action

    def update(self):
        self.add_episode()
        obs, actions = self.get_batch()
        self.update_weights(obs, actions)

    def add_episode(self):
        score = self.get_return()
        if score > self.cut_off_score:
            self.best_obs += self.obs
            self.best_actions += self.actions
            self.cut_off_score += (score - self.cut_off_score) / self.n_roll_outs
            self.n_roll_outs += 1

    def get_batch(self):
        size = min(len(self.best_obs), self.batch_size)
        obs = torch.zeros((size, self.n_inputs), dtype=torch.float32)
        actions = torch.zeros(size, dtype=torch.long)
        idxs = random.sample([i for i in range(len(self.best_obs))], k=size)
        for i, idx in enumerate(idxs):
            obs[i] = self.best_obs[idx]
            actions[i] = self.best_actions[idx]
        return obs, actions

    def update_weights(self, best_obs, best_actions):
        self.opt.zero_grad()
        probs = self.policy(best_obs)
        loss = self.crit(probs, best_actions)
        loss.backward()
        self.opt.step()

    def get_return(self):
        g = 0
        for r in reversed(self.rewards):
            g = r + (self.gamma * g)
        return g

