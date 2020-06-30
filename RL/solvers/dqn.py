import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple
import random


class Q(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=64):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


Replay = namedtuple("Replay", "state action reward next_state is_done")


class Replays:
    def __init__(self, maxlen=1000):
        self.maxlen = maxlen
        self.count = 0
        self.replays = deque(maxlen=maxlen)

    def add(self, replay):
        self.replays.append(replay)
        self.count += 1

    def __len__(self):
        return min(self.count, self.maxlen)

    def get_batch(self, size):
        size = min(size, len(self))
        batch = random.sample(self.replays, size)
        states = []
        actions = []
        rewards = []
        next_states = []
        is_dones = []
        for state, action, reward, next_state, is_done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            is_dones.append(float(not (is_done)))
        return (
            torch.stack(states),
            torch.tensor(actions).view(-1, 1),
            torch.tensor(rewards),
            torch.stack(next_states),
            torch.tensor(is_dones),
        )


class DQN:
    def __init__(
        self, env, batch_size=128, lr=0.01, gamma=0.99, e=0.1, steps_per_update=10
    ):
        n_inputs = n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.replays = Replays()
        self.Q = Q(n_inputs, n_outputs)
        self.opt = optim.Adam(self.Q.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_per_update = steps_per_update
        self.base_probs = [e / (n_outputs - 1)] * n_outputs
        self.max_prob = 1 - e
        self.actions = [a for a in range(n_outputs)]
        self.previous = None
        self.steps = 0
        self.loss = nn.MSELoss()

    def step(self, observation, reward=None, is_done=False):
        self.steps += 1
        observation = torch.tensor(observation, dtype=torch.float32)
        action = self.get_action(observation)
        if reward != None:
            previous_obs, previous_a = self.previous
            self.replays.add(
                Replay(previous_obs, previous_a, reward, observation, is_done)
            )
            if (
                len(self.replays) >= self.batch_size
                and self.steps % self.steps_per_update == 0
            ):
                self.update()
        self.previous = (observation, action)
        return action

    def get_action(self, observation):
        with torch.no_grad():
            action_values = self.Q(observation)
        max_action = torch.argmax(action_values)
        probs = self.base_probs.copy()
        probs[max_action] = self.max_prob
        action = random.choices(self.actions, weights=probs)[0]
        return action

    def update(self):
        states, actions, rewards, next_states, is_dones = self.replays.get_batch(
            self.batch_size
        )
        self.opt.zero_grad()
        ys = self.get_labels(rewards, next_states, is_dones)
        qs = self.get_qs(states, actions)
        loss = self.loss(qs, ys)  # calculate_loss(qs, ys)
        loss.backward()
        self.opt.step()

    def get_labels(self, rewards, next_states, is_dones):
        with torch.no_grad():
            qmax, idxs = torch.max(self.Q(next_states), dim=1)
            y = (self.gamma * qmax * is_dones) + rewards
        return y

    def get_qs(self, states, actions):
        return torch.gather(self.Q(states), 1, actions).view(-1)

    # def calculate_loss(self, qs, ys):
    #    return ((qs - ys) ** 2).mean()

