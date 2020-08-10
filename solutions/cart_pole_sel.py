import gym
import matplotlib.pyplot as plt
from RL.solvers import CEMAgent
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
fig = plt.figure()
all_rewards = []

for b in range(20):
    agent = CEMAgent(env)
    total_rewards = []
    n_episodes = 200
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
        print(episode + 1, total_reward, agent.cut_off_score)
    all_rewards.append(total_rewards)
mean_results = np.mean(np.stack(all_rewards), axis=0)
plt.plot(mean_results, label=str(agent))
plt.legend()
plt.show()
fig.savefig("cem_cartpole.png")
env.close()
