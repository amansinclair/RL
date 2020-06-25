import gym
import matplotlib.pyplot as plt
from RL.solvers import MCPG, MCPGBaseline


env = gym.make("CartPole-v1")
lr = 0.05
agent = MCPGBaseline(env, lr=lr)
total_rewards = []
n_episodes = 200
for episode in range(n_episodes):
    observation = env.reset()
    done = False
    reward = None
    while not done:
        action = agent.step(observation, reward)
        observation, reward, done, info = env.step(action)
    total_reward = agent.update(reward)
    total_rewards.append(total_reward)
    print(episode + 1, total_reward)
env.close()
plt.plot(total_rewards)
plt.show()
