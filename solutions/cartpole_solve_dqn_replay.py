import gym
import matplotlib.pyplot as plt
from RL.solvers import DQN, DQNReplay
import numpy as np
import torch

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
lr = 0.001

# agent = DQN(env, lr=lr, gamma=0.99, e=0.3)
agent = DQNReplay(env, lr=lr, gamma=0.99, e=0.3)
total_rewards = []
n_episodes = 50
actions = []
for episode in range(n_episodes):
    observation = env.reset()
    done = False
    reward = None
    rewards = []
    last_actions = []
    while not done:
        action = agent.step(observation, reward, done)
        actions.append(action)
        last_actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
    agent.step(observation, reward, done)
    total_reward = sum(rewards)
    total_rewards.append(total_reward)
    print(episode + 1, total_reward)
print("FINAL Q", agent.Q(torch.tensor(env.reset(), dtype=torch.float32)))
print("e", agent.policy.e)
print("a", sum(actions) / len(actions))
print(last_actions)
env.close()
# mean_results = np.mean(np.stack(all_results), axis=0)
plt.plot(total_rewards, label="DQN_Replay")
plt.legend()
plt.show()

