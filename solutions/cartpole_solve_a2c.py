import gym
import matplotlib.pyplot as plt
from RL.solvers import A2C

env = gym.make("CartPole-v1")
lr = 0.01
agent = A2C(env, lr=lr, gamma=0.99)
total_rewards = []
n_episodes = 500
for episode in range(n_episodes):
    print(episode + 1)
    observation = env.reset()
    done = False
    reward = None
    rewards = []
    while not done:
        action = agent.step(observation, reward, done)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
    agent.step(observation, reward, done)
    total_rewards.append(sum(rewards))
env.close()
plt.plot(total_rewards)
plt.show()
