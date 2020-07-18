from collections import deque
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
        return F.softmax(x, dim=-1)


class Value(nn.Module):
    def __init__(self, n_inputs, size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MCAgent:
    def __init__(self, actor, critic, alr=0.01, clr=0.1, batch_size=1, normalize=False):
        self.actor = actor
        self.critic = critic
        self.opts = []
        self.opts.append(optim.Adam(self.actor.parameters(), lr=alr))
        if self.critic.has_params:
            self.opts.append(optim.Adam(self.critic.parameters(), lr=clr))
        self.batch_size = batch_size
        self.normalize = normalize
        self.reset()

    def reset(self):
        self.loss = 0
        self.n_steps = 0

    def __str__(self):
        return str(self.critic)

    def step(self, obs, reward=None, is_done=False):
        obs = torch.tensor(obs, dtype=torch.float32)
        self.critic.store(obs, reward, is_done)
        if is_done:
            self.update()
            self.actor.reset()
            self.critic.reset()
            action = None
        else:
            action = self.actor(obs)
        return action

    def update(self):
        self.n_steps += len(self.actor)
        self.loss += self.get_loss()
        if self.n_steps >= self.batch_size:
            self.update_weights()

    def get_loss(self):
        P = self.actor.get_probs()
        A = self.critic()
        if self.normalize:
            A = (A - A.mean()) / A.std()
        Aloss = (A ** 2).sum()
        Pscore = -(A.detach() * torch.log(P)).sum()
        return Pscore + Aloss

    def update_weights(self):
        for opt in self.opts:
            opt.zero_grad()
        self.loss.backward()
        for opt in self.opts:
            opt.step()
        self.reset()


class Actor(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=32):
        super().__init__()
        self.policy = Policy(n_inputs, n_outputs, size)
        self.reset()

    def reset(self):
        self.probs = []

    def __len__(self):
        return len(self.probs)

    def forward(self, obs):
        prob = self.policy(obs)
        m = Categorical(prob)
        action = m.sample().item()
        self.probs.append(prob[action])
        return action

    def get_probs(self):
        return torch.stack(self.probs)


class Critic(nn.Module):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.has_params = False
        self.reset()

    def reset(self):
        self.obs = []
        self.rewards = []

    def __len__(self):
        return len(self.rewards)

    def __str__(self):
        return self.__class__.__name__

    def store(self, obs, reward=None, is_done=False):
        if not is_done:
            self.obs.append(obs)
        if reward:
            self.rewards.append(reward)

    def forward(self):
        size = len(self.rewards)
        G = torch.zeros(size, dtype=torch.float32)
        g = 0
        for i in reversed(range(size)):
            g = self.rewards[i] + (self.gamma * g)
            G[i] = g
        return G


class CriticBaseline(Critic):
    def __init__(self, n_inputs, size=32, **kwargs):
        super().__init__(**kwargs)
        self.has_params = True
        self.value = Value(n_inputs, size)

    def forward(self):
        G = super().forward()
        obs = torch.stack(self.obs)
        V = self.value(obs).view(-1)
        return G - V


class CriticTD(CriticBaseline):
    def __init__(self, *args, td=5, **kwargs):
        super.__init__(*args, **kwargs)
        self.tdlen = td + 1

    def forward(self):
        end = self.tdlen
        returns = []
        with torch.no_grad():
            while end < len(self.rewards):
                r = self.rewards[end - self.tdlen : end]
                G = self.get_summed_rs(r)
                V = self.value(self.obs[end])
                returns.append(G + V.item())
                end += 1
        self.rewards = self.rewards[-min(self.tdlen, len(self.rewards)) :]
        end_returns = super().get_return()
        if returns:
            start_returns = torch.tensor(returns)
            return torch.cat((start_returns, end_returns))
        else:
            return end_returns

    def get_summed_rs(self, rewards):
        R = 0.0
        for r in reversed(rewards):
            R = r + (R * self.gamma)
        return R


class CriticGAE(CriticBaseline):
    def __init__(self, *args, gae=0.92, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae = gae

    def forward(self):
        obs = torch.stack(self.obs)
        V = self.value(obs).view(-1)
        size = len(V)
        next_V = torch.zeros(size)
        next_V[:-1] = V[1:]  # .detach()
        R = torch.tensor(self.rewards)
        td_error = R + (self.gamma * next_V) - V
        A = torch.zeros(size)
        a = 0
        for i in reversed(range(size)):
            a = td_error[i] + (self.gamma * self.gae * a)
            A[i] = a
        return A


"""
class TDAgent(Agent):
    def __init__(self, *args, td, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdlen = td + 1

    def get_return(self):
        end = self.tdlen
        returns = []
        with torch.no_grad():
            while end < len(self.rewards):
                r = self.rewards[end - self.tdlen : end]
                G = self.get_summed_rs(r)
                V = self.value(self.obs[end])
                returns.append(G + V.item())
                end += 1
        self.rewards = self.rewards[-min(self.tdlen, len(self.rewards)) :]
        end_returns = super().get_return()
        if returns:
            start_returns = torch.tensor(returns)
            return torch.cat((start_returns, end_returns))
        else:
            return end_returns

    def get_summed_rs(self, rewards):
        R = 0.0
        for r in reversed(rewards):
            R = r + (R * self.gamma)
        return R


class GAEAgent(Agent):
    def __init__(self, *args, gae=0.92, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae = gae

    def batch_reset(self):
        super().batch_reset()
        self.ep_start = 0

    def get_return(self):
        end = self.ep_start + len(self.rewards)
        V = self.get_value(torch.stack(self.obs[self.ep_start : end]))
        self.ep_start += len(self.rewards)
        Vstep = V[1:].detach()
        zero = torch.tensor([0.0])
        Vstep = torch.cat((Vstep, zero))
        delta = Vstep - V + torch.tensor(self.rewards)
        size = len(self.rewards)
        advantage = torch.zeros(size, dtype=torch.float32)
        A = 0
        for i in reversed(range(size)):
            A = delta[i] + (self.gamma * self.gae * A)
            advantage[i] = A
        advantages_2 = []
        advantage_1 = 0.0
        next_value = 0.0
        for r, v in zip(reversed(self.rewards), reversed(V)):
            td_error = r + next_value * self.gamma - v
            advantage_1 = td_error + advantage_1 * self.gamma * self.gae
            next_value = v
            advantages_2.insert(0, advantage_1)
        advantages_2 = torch.tensor(advantages_2)
        return advantages_2

    def calculate_score(self):
        obs = torch.stack(self.obs)
        A = torch.stack(self.returns)
        idxs = torch.tensor(self.actions).view(-1, 1)
        ps = self.policy(obs)
        P = ps.gather(1, idxs).view(-1)
        mse = (A ** 2).mean()
        return (-A.detach() * torch.log(P)).mean() + mse
"""
