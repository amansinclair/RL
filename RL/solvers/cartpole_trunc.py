import gym
import matplotlib.pyplot as plt
from RL.solvers import Actor, Critic, Model, Rewarder, EmptyRewarder
from RL.utils import run, run_episode, EnvManager, get_env_size, plot_means, evaluate
import numpy as np

env_name = "MountainCar-v0"  # "LunarLander-v2"  # "MountainCar-v0"  # "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)

"""
def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    critic = Critic(n_inputs, size=32)
    rewarder = Rewarder(n_inputs, n_outputs=1, size=8)
    agent = Model(
        actor,
        critic,
        rewarder,
        rollout_length=200,
        n_rollouts=1,
        policy_lr=0.1,
        critic_lr=0.1,
        rewarder_lr=0.01,
    )
    return agent
"""


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=16)
    critic = Critic(n_inputs, size=16)
    rewarder = EmptyRewarder()  # Rewarder(n_inputs, n_outputs=1, size=16)
    agent = Model(
        actor,
        critic,
        rewarder,
        rollout_length=20,
        n_rollouts=1,
        critic_lr=0.01,
        rewarder_lr=0.01,
        norm_obs=False,
    )
    return agent


agent = get_agent()

with EnvManager(env_name, max_steps=200) as env:
    results = run(env, agent, render=True, n_episodes=20)
"""
with EnvManager(env_name, max_steps=1000) as env:
    results = evaluate(env, get_agent, render=False, n_episodes=60, n_repeats=10)


with EnvManager(env_name, max_steps=200) as env:
    run_episode(env, agent, render=True)

plot_means(results)
"""
