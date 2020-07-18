import gym
import matplotlib.pyplot as plt
from RL.solvers import MCAgent, Actor, CriticBaseline, CriticGAE
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
fig = plt.figure()
all_rewards = []

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
for c in (CriticBaseline, CriticGAE):
    for b in range(60):
        actor = Actor(n_inputs, n_outputs)
        # critic = CriticGAE(n_inputs, gae=0.92)
        critic = c(n_inputs)
        agent = MCAgent(
            actor, critic, alr=0.03, clr=0.1, batch_size=100, normalize=False
        )
        total_rewards = []
        n_episodes = 120
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
            print(episode + 1, total_reward)
        all_rewards.append(total_rewards)
    mean_results = np.mean(np.stack(all_rewards), axis=0)
    plt.plot(mean_results, label=str(agent))
plt.legend()
plt.show()
fig.savefig("gae_bl_comp.png")
env.close()
