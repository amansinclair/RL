import gym
import matplotlib.pyplot as plt
from RL.solvers import MCAgent, Actor, CriticGAE
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    critic = CriticGAE(n_inputs, size=32, gamma=0.99, gae=0.96)
    batch_size = 10
    agent = MCAgent(actor, critic, batch_size=batch_size, alr=0.01)
    return agent


results = training_loop(env_name, get_agent, n_episodes=100, n_repeats=5)
plot_means(results)

