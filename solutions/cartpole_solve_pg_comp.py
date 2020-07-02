import gym
import matplotlib.pyplot as plt
from RL.solvers import MCPG, MCPGBaseline
import numpy as np


env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
fig = plt.figure()
for model, lr in [(MCPG, 0.03), (MCPGBaseline, 0.03)]:
    all_rewards = []
    for b in range(20):
        agent = model(env, lr=lr)
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
        all_rewards.append(total_rewards)
    env.close()
    mean_results = np.mean(np.stack(all_rewards), axis=0)
    plt.plot(mean_results, label=agent.name)
plt.legend()
plt.show()
fig.savefig("mcpg.png")
