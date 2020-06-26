import gym
import matplotlib.pyplot as plt
from RL.solvers import A2C
import numpy as np

env = gym.make("CartPole-v1")
lr = 0.01
all_rewards = []
for b in range(10):
    agent = A2C(env, lr=lr, gamma=0.99)
    total_rewards = []
    n_episodes = 200
    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        reward = None
        rewards = []
        while not done:
            action = agent.step(observation, reward, done)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
        agent.step(observation, reward, done)
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        print(episode + 1, total_reward)
    all_rewards.append(total_rewards)
env.close()
mean_results = np.mean(np.stack(all_rewards), axis=0)
plt.plot(mean_results)
plt.show()
