import gym
import matplotlib.pyplot as plt
from RL.solvers import PPOAgent, Actor, CriticBaseline
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
fig = plt.figure()
all_rewards = []

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
for b in range(1):
    actor = Actor(n_inputs, n_outputs)
    # critic = CriticGAE(n_inputs, gae=0.92)
    critic = CriticBaseline(n_inputs)
    agent = PPOAgent(actor, critic, alr=0.01, clr=0.01, ppo=0.2, normalize=False)
    total_rewards = []
    n_episodes = 1
    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        reward = None
        rewards = []
        while not done:
            action = agent.step(observation, reward)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
        agent.step(observation, reward, done)
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        print(episode + 1, total_reward)
    all_rewards.append(total_rewards)
mean_results = np.mean(np.stack(all_rewards), axis=0)
plt.plot(mean_results, label=str(agent))
plt.legend()
plt.show()
# fig.savefig("gae_comp.png")
env.close()
