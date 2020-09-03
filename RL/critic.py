import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, value_net, adv_func):
        super().__init__()
        self.value_net = value_net
        self.adv_func = adv_func

    def get_values(self, obs):
        return self.value_net(obs)

    def get_advantage(self, obs, rewards, is_dones):
        with torch.no_grad():
            values = self.get_values(obs)
            values = values * (~is_dones)
            # for roll out in rollouts?
            advantages, targets = self.adv_func(values, rewards)
        return advantages, targets

