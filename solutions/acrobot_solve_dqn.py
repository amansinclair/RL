import gym
import matplotlib.pyplot as plt
from RL.solvers import DQNReplay
import numpy as np
import torch


env = gym.make("Acrobot-v1")
# env._max_episode_steps = 500
n_episodes = 10
agent = DQNReplay(env, gamma=0.99, e=0.05)
total_rewards = []
for episode in range(n_episodes):
    observation = env.reset()
    done = False
    reward = None
    rewards = []
    while not done:
        env.render()
        action = agent.step(observation, reward, done)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
    agent.step(observation, reward, done)
    total_reward = sum(rewards)
    print("episode", episode + 1, total_reward, agent.policy.e)
    total_rewards.append(total_reward)
env.close()
print(total_rewards)
