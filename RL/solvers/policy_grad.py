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

    def get_batch_probs(self, obs):
        probs = self.policy(obs)
        m = Categorical(probs)
        actions = m.sample()
        return torch.gather(probs, 1, actions.view(-1, 1))


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
        """Call stand_return so that subclasses still have acess to stand_return."""
        return self.standard_return()

    def standard_return(self):
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
        G = self.standard_return()
        obs = torch.stack(self.obs)
        V = self.value(obs).view(-1)
        return G - V


class CriticTD(CriticBaseline):
    def __init__(self, *args, td=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdlen = td + 1

    def __str__(self):
        return self.__class__.__name__ + str(self.tdlen - 1)

    def forward(self):
        end = self.tdlen
        returns = []
        with torch.no_grad():
            while end < len(self.rewards):
                r = self.rewards[end - self.tdlen : end]
                R = self.get_summed_rs(r)
                V = self.value(self.obs[end])
                returns.append(R + V.item())
                end += 1
        end_returns = self.get_tail_rewards()
        if returns:
            start_returns = torch.tensor(returns)
            G = torch.cat((start_returns, end_returns))
        else:
            G = end_returns
        obs = torch.stack(self.obs)
        V = self.value(obs).view(-1)
        return G - V

    def get_summed_rs(self, rewards):
        R = 0.0
        for r in reversed(rewards):
            R = r + (R * self.gamma)
        return R

    def get_tail_rewards(self):
        self.rewards = self.rewards[-min(self.tdlen, len(self.rewards)) :]
        return self.standard_return()


class CriticGAE(CriticBaseline):
    def __init__(self, *args, gae=0.92, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae = gae

    def __str__(self):
        return self.__class__.__name__ + str(self.gae)

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


class PPOAgent(MCAgent):
    def __init__(self, *args, ppo=0.2, n_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo = 0.2
        self.n_epochs = n_epochs

    def reset(self):
        super().reset()
        self.p = None

    def update(self):
        self.update_weights()

    def get_loss(self):
        P = self.actor.get_batch_probs(torch.stack(self.critic.obs))
        R = self.get_surrogate(P)
        A = self.critic()
        if self.normalize:
            A = (A - A.mean()) / A.std()
        Aloss = (A ** 2).sum()
        Pscore = -(A.detach() * R).sum()
        return Pscore + Aloss

    def get_surrogate(self, P):
        if self.p != None:
            R = P / self.p.detach()
        else:
            R = torch.ones(P.shape)
        self.p = P
        return self.clip(R)

    def update_weights(self):
        for epoch in range(self.n_epochs):
            print("epoch", epoch + 1)
            for opt in self.opts:
                opt.zero_grad()
            self.loss = self.get_loss()
            self.loss.backward()
            for opt in self.opts:
                opt.step()
        self.reset()

    def clip(self, R):
        return torch.clamp(R, 1 - self.ppo, 1 + self.ppo)

