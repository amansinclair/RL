import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random


class Q(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, env, batch_size=100, lr=0.01, gamma=1.0, e=0.01):
        n_inputs = n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.replays = deque(maxlen=1000)
        self.Q = Q(n_inputs, n_outputs)
        self.opt = optim.Adam(self.Q.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.base_probs = [e / n_outputs] * n_outputs
        self.max_prob = 1 - e
        self.actions = [a for a in range(n_outputs)]
        self.previous = None

    def step(self, observation, reward=None):
        observation = torch.tensor(observation, dtype=torch.float32)
        if reward != None:
            previous_obs, previous_a = self.previous
            self.replays.append((previous_obs, previous_a, reward, observation))
        with torch.no_grad():
            action_values = self.Q(observation)
        max_action = torch.argmin(action_values)
        probs = self.base_probs.copy()
        probs[max_action] = self.max_prob
        action = random.choices(self.actions, weights=probs)[0]
        self.previous = (observation, action)
        return action

    def update(self, observation, reward):
        previous_obs, previous_a = self.previous
        self.replays.append((previous_obs, previous_a, reward, None))
        batch_size = min(self.batch_size, len(self.replays))
        replays = random.choices(self.replays, k=batch_size)
        self.update_weights(replays)

    def update_weights(self, replays):
        self.opt.zero_grad()
        labels = self.get_labels(replays)
        qs = self.get_qs(replays)
        loss = self.calculate_loss(labels, qs)
        loss.backward()
        self.opt.step()

    def get_labels(self, replays):
        with torch.no_grad():
            labels = torch.zeros(len(replays))
            for i, replay in enumerate(replays):
                q = 0
                s, a, r, s2 = replay
                if s2 != None:
                    q = self.gamma * torch.max(self.Q(s2))
                labels[i] = r + q
        return labels

    def get_qs(self, replays):
        states = torch.stack([s for s, a, r, s2 in replays])
        actions = [a for s, a, r, s2 in replays]
        return self.Q(states)[:, actions]

    def calculate_loss(self, labels, qs):
        return ((qs - labels) ** 2).mean()

