import gym
import matplotlib.pyplot as plt
from RL.solvers import DQN
import numpy as np

env = gym.make("CartPole-v1")
lr = 0.01

agent = DQN(env, lr=lr, gamma=0.99)
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
    agent.update(observation, reward)
    total_reward = sum(rewards)
    total_rewards.append(total_reward)
    print(episode + 1, total_reward)
env.close()
plt.plot(total_rewards)
plt.show()
