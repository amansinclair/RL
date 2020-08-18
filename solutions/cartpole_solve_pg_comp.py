import gym
import matplotlib.pyplot as plt
from RL.solvers import MCAgent, Actor, Critic, CriticBaseline
from RL.utils import training_loop, get_env_size
import numpy as np

env_name = "CartPole-v1"
n_inputs, n_outputs = get_env_size(env_name)


def get_agent():
    actor = Actor(n_inputs, n_outputs, size=32)
    # critic = CriticBaseline(n_inputs, size=32, gamma=0.99)
    critic = Critic()
    batch_size = 100
    agent = MCAgent(actor, critic, batch_size=batch_size, alr=0.03)
    return agent


results = training_loop(env_name, get_agent, n_episodes=100, n_repeats=20)
m = np.mean(results, axis=0)
std = np.std(results, axis=0)
fig = plt.figure()
plt.plot(m)
plt.fill_between(np.arange(len(m)), m - std, m + std, alpha=0.2, interpolate=True)
fig.savefig("mc_std.png")
plt.show()

