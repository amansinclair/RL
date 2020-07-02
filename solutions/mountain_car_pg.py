import gym
import matplotlib.pyplot as plt
from RL.solvers import MCPGBaseline
import numpy as np
import torch


env = gym.make("MountainCar-v0")
# env._max_episode_steps = 500
n_episodes = 50
agent = MCPGBaseline(env, lr=0.01, gamma=0.99)
total_rewards = []
for episode in range(n_episodes):
    observation = env.reset()
    done = False
    reward = None
    rewards = []
    while not done:
        env.render()
        action = agent.step(observation, reward)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
    agent.update(reward)
    total_reward = sum(rewards)
    print("episode", episode + 1, total_reward)
    total_rewards.append(total_reward)
env.close()
print(total_rewards)
