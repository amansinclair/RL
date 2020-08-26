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
    rewarder = Rewarder(n_inputs, size=2)
    agent = Model(
        actor, critic, rewarder, rollout_length=10, n_rollouts=1, rewarder_lr=0.1
    )
    return agent


results = training_loop(
    env_name, get_agent, render=False, n_episodes=500, n_repeats=1, max_steps=200,
)
plot_means(results)

