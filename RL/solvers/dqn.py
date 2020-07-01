import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque, namedtuple
import random


class Q(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=128):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy:
    def __init__(self, n_actions, e=0.3, decay=0.995):
        self.actions = [a for a in range(n_actions)]
        self.e = e
        self.decay = decay

    def get_action(self, best_action):
        if random.choices([True, False], weights=[1 - self.e, self.e])[0]:
            action = best_action
        else:
            action = random.choice(self.actions)
        return action

    def update(self):
        self.e = max(self.decay * self.e, 0.01)


class DQN:
    def __init__(self, env, lr=0.01, gamma=0.99, e=0.1):
        n_inputs = n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.policy = Policy(n_outputs, e)
        self.Q = Q(n_inputs, n_outputs)
        self.opt = optim.Adam(self.Q.parameters(), lr=lr)
        self.crit = nn.MSELoss()
        self.previous = None

    def step(self, observation, reward=None, is_done=False):
        observation = torch.tensor(observation, dtype=torch.float32)
        best_action = self.get_best_action(observation)
        action = self.policy.get_action(best_action)
        if reward != None:
            previous_obs, previous_a = self.previous
            self.update_weights(previous_obs, previous_a, reward, observation, is_done)
        self.previous = (observation, action)
        if is_done:
            self.policy.update()
        return action

    def get_best_action(self, observation):
        with torch.no_grad():
            action_values = self.Q(observation)
        return torch.argmax(action_values).item()

    def update_weights(self, state, action, reward, next_state, is_done):
        self.opt.zero_grad()
        ys = self.get_labels(reward, next_state, is_done)
        qs = self.get_qs(state, action)
        loss = self.crit(qs, ys)
        loss.backward()
        self.opt.step()

    def get_labels(self, reward, next_state, is_done):
        is_done = 0 if is_done else 1
        with torch.no_grad():
            qmax = torch.max(self.Q(next_state))
            y = (self.gamma * qmax * is_done) + reward
        return y

    def get_qs(self, state, action):
        return self.Q(state)[action]


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


class DQNReplay:
    def __init__(
        self, env, batch_size=32, lr=0.01, gamma=0.99, e=0.1, steps_per_update=10
    ):
        n_inputs = n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.policy = Policy(n_outputs, e)
        self.replays = Replays()
        self.Q = Q(n_inputs, n_outputs)
        self.opt = optim.Adam(self.Q.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_per_update = steps_per_update
        self.previous = None
        self.steps = 0
        self.crit = nn.MSELoss()

    def step(self, observation, reward=None, is_done=False):
        self.steps += 1
        observation = torch.tensor(observation, dtype=torch.float32)
        best_action = self.get_best_action(observation)
        action = self.policy.get_action(best_action)
        if reward != None:
            previous_obs, previous_a = self.previous
            self.replays.add(
                Replay(previous_obs, previous_a, reward, observation, is_done)
            )
            if (
                len(self.replays) >= self.batch_size
                and self.steps % self.steps_per_update == 0
            ):
                self.update_weights()
        self.previous = (observation, action)
        if is_done:
            self.policy.update()
        return action

    def get_best_action(self, observation):
        with torch.no_grad():
            action_values = self.Q(observation)
        return torch.argmax(action_values).item()

    def update_weights(self):
        states, actions, rewards, next_states, is_dones = self.replays.get_batch(
            self.batch_size
        )
        self.opt.zero_grad()
        ys = self.get_labels(rewards, next_states, is_dones)
        qs = self.get_qs(states, actions)
        loss = self.crit(qs, ys)
        loss.backward()
        self.opt.step()

    def get_labels(self, rewards, next_states, is_dones):
        with torch.no_grad():
            qmax, idxs = torch.max(self.Q(next_states), dim=1)
            y = (self.gamma * qmax * is_dones) + rewards
        return y

    def get_qs(self, states, actions):
        return torch.gather(self.Q(states), 1, actions).view(-1)

