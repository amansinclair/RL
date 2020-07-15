from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def get_return(rewards, gamma):
    size = len(rewards)
    discounted_return = torch.zeros(size, dtype=torch.float32)
    g = 0
    for i in reversed(range(size)):
        g = rewards[i] + (gamma * g)
        discounted_return[i] = g
    return discounted_return


class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=16):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class Agent:
    def __init__(self, env, gamma=1, plr=0.03, vlr=0.1, batch_size=100):
        self.gamma = gamma
        self.batch_size = batch_size
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
        self.policy = Policy(n_inputs, n_outputs)
        self.value = Value(n_inputs)
        self.opts = [
            optim.Adam(self.policy.parameters(), lr=plr),
            optim.Adam(self.value.parameters(), lr=vlr),
        ]
        self.batch_reset()
        self.episode_reset()

    def episode_reset(self):
        self.rewards = []

    def batch_reset(self):
        self.obs = []
        self.actions = []
        self.returns = []

    def step(self, observation, reward=None, is_done=False):
        observation = torch.tensor(observation, dtype=torch.float32)
        self.obs.append(observation)
        if reward != None:
            self.rewards.append(reward)
        action = None
        if not is_done:
            with torch.no_grad():
                prob = self.policy(observation)
            m = Categorical(prob)
            action = m.sample().item()
            self.actions.append(action)
        else:
            self.update(reward)
        return action

    def update(self, reward):
        self.rewards.append(reward)
        self.returns.extend(self.get_return())
        if len(self.returns) >= self.batch_size:
            self.update_weights()
            self.batch_reset()
        self.episode_reset()

    def update_weights(self):
        for opt in self.opts:
            opt.zero_grad()
        score = self.calculate_score()
        score.backward()
        for opt in self.opts:
            opt.step()

    def calculate_score(self):
        obs = torch.stack(self.obs)
        V = self.get_value(obs)
        G = torch.stack(self.returns)
        idxs = torch.tensor(self.actions).view(-1, 1)
        ps = self.policy(obs)
        P = ps.gather(1, idxs).view(-1)
        A = G - V
        mse = (A ** 2).mean()
        return (-A.detach() * torch.log(P)).mean() + mse

    def get_value(self, obs):
        return self.value(obs).view(-1)

    def get_return(self):
        size = len(self.rewards)
        discounted_return = torch.zeros(size, dtype=torch.float32)
        g = 0
        for i in reversed(range(size)):
            g = self.rewards[i] + (self.gamma * g)
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
        print(p)
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
        self.policy = Policy(n_inputs, n_outputs)
        self.value = Value(n_inputs)
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
        print("no effect")
        with torch.no_grad():
            V = self.value(self.obs[-1]).item() * (self.gamma ** (self.tdlen + 1))
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

