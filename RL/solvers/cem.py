import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Agent(nn.Module):
    def __init__(self, n_inputs=4, n_outputs=2, size=8):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        probs = F.softmax(x, dim=-1)
        m = Categorical(probs)
        return m.sample().item()

    def __lt__(self, x):
        return True


def train(env, n_agents, n_episodes, n_best=3):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    size = 8
    n_rounds = n_episodes // n_agents
    means = None
    stds = None
    for round in range(n_rounds):
        agents = get_agents(n_agents, input_size, output_size, size, means, stds)
        avg_score, results = train_round(env, agents)
        means, stds = get_statistics(results, n_best)
        print(round, avg_score)


def get_agents(n, input_size, output_size, size=8, means=None, stds=None):
    agents = [Agent(input_size, output_size, size=size) for i in range(n)]
    if means:
        i = 0
        for mean, std in zip(means, stds):
            for agent in agents:
                list(agent.parameters())[i] = torch.normal(mean, std)
    return agents


def train_round(env, agents):
    results = []
    end_rewards = []
    for agent in agents:
        obs = env.reset()
        is_done = False
        reward = None
        rewards = []
        while not is_done:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = agent(obs)
            obs, reward, is_done, info = env.step(action)
            rewards.append(reward)
        end_reward = sum(rewards)
        end_rewards.append(end_reward)
        results.append((end_reward, agent))
    return sum(end_rewards) / len(rewards), results


def get_statistics(results, n_best):
    results.sort()
    top_results = results[-n_best:]
    top_performers = [performer for result, performer in top_results]
    n_params = len(list(top_performers[0].parameters()))
    means = []
    stds = []
    for i in range(n_params):
        params = []
        for performer in top_performers:
            params.append(list(performer.parameters())[i])
        params = torch.stack(params)
        means.append(params.mean(dim=0))
        stds.append(params.std(dim=0))
    return means, stds

