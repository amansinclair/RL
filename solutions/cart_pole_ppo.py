import gym
import matplotlib.pyplot as plt
from RL.solvers import PPOAgent, PPOActor, CriticBaseline
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = PPOActor(n_inputs, n_outputs, size=32)
    critic = CriticBaseline(n_inputs, size=32, gamma=0.99)
    agent = PPOAgent(actor, critic, alr=0.06, clr=0.05)
    return agent


results = training_loop(env_name, get_agent, n_episodes=100, n_repeats=5)
plot_means(results)
# np.save("cartpole_gae_96_bs_20_lr_01.npy", results)
