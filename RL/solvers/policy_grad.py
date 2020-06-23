import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import time


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=0)


def get_return(rewards):
    g = torch.flip(rewards, [0]).cumsum(0)
    g = torch.flip(g, [0])
    return g


def play_single(net, env, view=False):
    probs = torch.zeros((200), dtype=torch.float32)
    rewards = torch.zeros((200), dtype=torch.float32)
    observation = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    i = 0
    while not done:
        if view:
            env.render()
            time.sleep(0.1)
        p = net(observation)
        action = np.random.choice([0, 1], p=p.detach().numpy())
        probs[i] = p[action]
        observation, reward, done, info = env.step(action)
        observation = torch.tensor(observation, dtype=torch.float32)
        rewards[i] = reward
        i += 1
    g = get_return(rewards[:i])
    return (probs[:i], g)


def play(net, env, batch_size=1000):
    probs = []
    gs = []
    current_size = 0
    i = 0
    while current_size < batch_size:
        prob, g = play_single(net, env)
        probs.append(prob)
        gs.append(g)
        current_size += prob.shape[0]
        i += 1
    return i, torch.cat(probs), torch.cat(gs)


def train_batch(net, opt, env):
    opt.zero_grad()
    i, probs, gs = play(net, env)
    loss = -(gs * torch.log(probs)).mean()
    # print(loss)
    loss.backward()
    opt.step()
    return i


def view_round(net, env):
    with torch.no_grad():
        play_single(net, env, True)


learning_rate = 0.01
net = Net()
opt = optim.Adam(net.parameters(), lr=learning_rate)
env = gym.make("CartPole-v1")
n = 0
for i in range(30):
    n += train_batch(net, opt, env)
    print(n)
view_round(net, env)
env.close()
