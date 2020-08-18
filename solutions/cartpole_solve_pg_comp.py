import gym
import matplotlib.pyplot as plt
from RL.solvers import MCAgent, Actor, Critic, CriticBaseline
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    critic = CriticBaseline(n_inputs, size=32, gamma=0.99)
    # critic = Critic()
    batch_size = 50
    agent = MCAgent(actor, critic, batch_size=batch_size, alr=0.02)
    return agent


results = training_loop(env_name, get_agent, n_episodes=100, n_repeats=50)
plot_means(results)
np.save("cartpole_baseline_bs_50_lr_02.npy", results)

