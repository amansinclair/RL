import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, policy_net):
        super().__init__()
        self.policy_net = policy_net

    def act(self, obs):
        with torch.no_grad():
            prob = self.policy_net(obs)
            m = Categorical(prob)
        return m.sample().item()

    def get_probs(self, obs, actions):
        probs = self.policy_net(obs)
        return torch.gather(probs, 1, actions.view(-1, 1)).squeeze()

    def get_log_probs(self, obs, actions):
        return self.get_probs(obs, actions).log()

    def get_entropy(self, obs):
        probs = self.policy_net(obs)
        m = Categorical(probs)
        return m.entropy()
