import gym
import matplotlib.pyplot as plt
from RL.solvers import A2C

env = gym.make("CartPole-v1")
lr = 0.01
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
env.close()
plt.plot(total_rewards)
plt.show()
