import gym
import matplotlib.pyplot as plt
from RL.solvers import A2C, MCPGBaseline, MCPG, Agent, TDAgent
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
fig = plt.figure()
all_rewards = []
for td in [0, 3, 5, 10, 20, 500]:
    for b in range(20):
        # agent = Agent(env)
        agent = TDAgent(env, gamma=0.99, td=td)
        total_rewards = []
        n_episodes = 100
        for episode in range(n_episodes):
            observation = env.reset()
            done = False
            reward = None
            rewards = []
            while not done:
                action = agent.step(observation, reward)
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
            agent.update(reward)
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            print(episode + 1, total_reward)
        all_rewards.append(total_rewards)
    mean_results = np.mean(np.stack(all_rewards), axis=0)
    plt.plot(mean_results, label="TD_" + str(td))
plt.legend()
plt.show()
fig.savefig("pgtdcomp.png")
env.close()
