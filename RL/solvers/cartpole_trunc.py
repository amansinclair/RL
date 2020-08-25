import gym
import matplotlib.pyplot as plt
from RL.solvers import Actor, Critic, Model, Rewarder
from RL.utils import training_loop, get_env_size, plot_means
import numpy as np

env_name = "MountainCar-v0"  # "LunarLander-v2"  # "MountainCar-v0"  # "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    critic = Critic(n_inputs, size=32)
    rewarder = Rewarder(n_inputs, size=32)
    agent = Model(actor, critic, rewarder, rollout_length=100, n_rollouts=3)
    return agent


results = training_loop(
    env_name, get_agent, render=False, n_episodes=60, n_repeats=5, max_steps=400,
)
plot_means(results)

