import gym
import matplotlib.pyplot as plt
from RL.solvers import Actor, Critic, Model
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "LunarLander-v2"  # "MountainCar-v0"  # "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    critic = Critic(n_inputs, size=32)
    agent = Model(actor, critic, rollout_length=50, n_rollouts=1)
    return agent


results = training_loop(
    env_name, get_agent, render=True, n_episodes=100, n_repeats=1, max_steps=500,
)
plot_means(results)

