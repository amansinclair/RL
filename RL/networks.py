import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, sizes=None):
        super().__init__()
        if sizes:
            current_size = n_inputs
            layers = []
            for size in sizes:
                layers.append(nn.Linear(current_size, size))
                layers.append(nn.ReLU())
                current_size = size
            layers.append(nn.Linear(current_size, n_outputs))
        else:
            layers = [nn.Linear(n_inputs, n_outputs)]
        self.net = nn.Sequential(*layers)


class Policy(Net):
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class Value(Net):
    def forward(self, x):
        return self.net(x)

