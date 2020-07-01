import gym
import matplotlib.pyplot as plt
from RL.solvers import DQN, DQNReplay
import numpy as np
import torch

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
lr = 0.001
fig = plt.figure()
for model, name, n_episodes in [(DQN, "DQN", 150), (DQNReplay, "DQNReplay", 50)]:
    all_results = []
    for i in range(20):
        agent = model(env, lr=lr, gamma=0.99, e=0.3)
        total_rewards = []
        for episode in range(n_episodes):
            observation = env.reset()
            done = False
            reward = None
            rewards = []
            last_actions = []
            while not done:
                action = agent.step(observation, reward, done)
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
            agent.step(observation, reward, done)
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            print(episode + 1, total_reward)
        all_results.append(total_rewards)
    mean_results = np.mean(np.stack(all_results), axis=0)
    plt.plot(mean_results, label=name)
env.close()
plt.legend()
plt.show()
fig.savefig("dqn.png")

