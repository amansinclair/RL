import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return F.softmax(x)


learning_rate = 0.01


def get_return(rewards):
    return rewards[::-1].cumsum()


def play(net, env):
    states = []
    actions = []
    rewards = []
    observation = torch.tensor(env.reset(), dtype=torch.float32)
    states.append(observation)
    done = False
    i = 0
    while i < 1000 and not done:
        env.render()
        time.sleep(0.1)
        i += 1
        probs = net(observation)
        print(probs)
        # probs = net(observation)
        action = np.random.choice([0, 1], p=probs.detach().numpy())
        # action = np.random.choice([0, 1])
        actions.append(action)
        observation, reward, done, info = env.step(action)
        observation = torch.tensor(observation, dtype=torch.float32)
        rewards.append(reward)
        states.append(observation)
    r = get_return(np.array(rewards, dtype="float32"))
    print(r)
    return (
        torch.stack(states),
        torch.tensor(actions, dtype=torch.double),
        torch.flip(torch.from_numpy(r)),
    )


net = Net()
env = gym.make("CartPole-v1")
s, a, r = play(net, env)
env.close()
