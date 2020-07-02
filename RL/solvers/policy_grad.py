from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=16):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

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
        return (-G * torch.log(p)).mean()

    @property
    def name(self):
        return str(self.__class__.__name__)


class Value(nn.Module):
    def __init__(self, n_inputs, size=16):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MCPGBaseline(MCPG):
    def setup(self, n_inputs, n_outputs, lr=0.01):
        self.policy = Policy(n_inputs, n_outputs)
        self.value = Value(n_inputs)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.vopt = optim.Adam(self.value.parameters(), lr=0.1)

    def reset_batch(self):
        super().reset_batch()
        self.values = []

    def step(self, observation, reward=None):
        observation = torch.tensor(observation, dtype=torch.float32)
        self.values.append(self.value(observation))
        return super().step(observation, reward=reward)

    def update_weights(self):
        self.opt.zero_grad()
        self.vopt.zero_grad()
        score = self.calculate_score()
        score.backward()
        self.opt.step()
        self.vopt.step()

    def calculate_score(self):
        G = torch.stack(self.returns)
        p = torch.stack(self.probs)
        V = torch.stack(self.values).view(-1)
        error = G - V
        mse = (error ** 2).mean()
        return (-error.detach() * torch.log(p)).mean() + mse


class A2C(nn.Module):
    def __init__(self, env, lr=0.01, gamma=0.99, vlr=0.1, tdlen=5, batch_size=100):
        super().__init__()
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.policy = Policy(n_inputs, n_outputs, size=16)
        self.value = Value(n_inputs, size=16)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.vopt = optim.Adam(self.value.parameters(), lr=vlr)
        self.gamma = gamma
        self.tdlen = tdlen
        self.batch_size = batch_size
        self.reset()
        self.episode_reset()

    def reset(self):
        self.obs = []
        self.probs = []
        self.returns = []

    def episode_reset(self):
        self.steps = 0
        self.rewards = deque(maxlen=self.tdlen + 1)

    def step(self, observation, reward=None):
        observation = torch.tensor(observation, dtype=torch.float32)
        prob = self.policy(observation)
        m = Categorical(prob)
        action = m.sample().item()
        if reward != None:
            self.rewards.append(reward)
        self.probs.append(prob[action])
        self.obs.append(observation)
        if self.steps > self.tdlen:
            self.returns.append(self.get_return())
        self.steps += 1
        return action

    def get_return(self):
        with torch.no_grad():
            V = self.value(self.obs[-1]).item() * self.gamma ** (self.tdlen + 1)
        R = self.get_summed_rs()
        return V + R

    def get_summed_rs(self):
        R = 0.0
        for r in reversed(self.rewards):
            R = r + (R * self.gamma)
        return R

    def update(self, reward):
        self.rewards.append(reward)
        self.update_tail_returns()
        if len(self.obs) >= self.batch_size:
            self.update_weights()
            self.reset()
        self.episode_reset()

    def update_tail_returns(self):
        while self.rewards:
            self.returns.append(self.get_summed_rs())
            self.rewards.popleft()

    def update_weights(self):
        self.opt.zero_grad()
        self.vopt.zero_grad()
        score = self.calculate_score()
        score.backward()
        self.opt.step()
        self.vopt.step()

    def calculate_score(self):
        p = torch.tensor(self.probs, dtype=torch.float32)
        obs = torch.stack(self.obs)
        td_returns = torch.tensor(self.returns, dtype=torch.float32)
        values = self.value(obs).view(-1)
        advantage = td_returns - values
        mse = (advantage ** 2).mean()
        return (-advantage.detach() * torch.log(p)).mean() + mse

    def calculate_td_values(self, values, rewards):
        n_iter = values.shape[0]
        next_values = torch.zeros(n_iter, dtype=torch.float32)
        for i in range(n_iter):
            td_idx = i + self.tdlen
            end_value = (
                0
                if td_idx >= n_iter
                else values[td_idx] * self.gamma ** (self.tdlen + 1)
            )
            end_idx = min(td_idx, n_iter)
            summed_rs = self.get_summed_rs(rewards[i:end_idx])
            next_values[i] = end_value + summed_rs
        return next_values

