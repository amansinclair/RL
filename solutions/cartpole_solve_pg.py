import gym
import matplotlib.pyplot as plt
from RL.solvers import MCPG, MCPGBaseline


env = gym.make("CartPole-v1")
lr = 0.05
agent = MCPGBaseline(env, lr=lr)
total_rewards = []
n_episodes = 200
for episode in range(n_episodes):
    print(episode + 1)
    observation = env.reset()
    done = False
    reward = None
    while not done:
        action = agent.step(observation, reward)
        observation, reward, done, info = env.step(action)
    total_rewards.append(agent.update(reward))
env.close()
plt.plot(total_rewards)
plt.show()
