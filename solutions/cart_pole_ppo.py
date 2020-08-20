import gym
import matplotlib.pyplot as plt
from RL.solvers import PPOAgent, PPOActor, CriticBaseline
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "MountainCar-v0"  # "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = PPOActor(n_inputs, n_outputs, size=32)
    critic = CriticBaseline(n_inputs, size=32, gamma=0.99)
    agent = PPOAgent(actor, critic, alr=0.01, clr=0.02)
    return agent


results = training_loop(env_name, get_agent, render=True, n_episodes=50, n_repeats=1)
# plot_means(results)
# np.save("cartpole_ppo_02_epochs_5_lr_01.npy", results)
