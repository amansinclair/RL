import gym
import numpy as np
import matplotlib.pyplot as plt


def get_env_size(env_name):
    env = gym.make(env_name)
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    env.close()
    return n_inputs, n_outputs


def training_loop(
    env_name,
    agent_func,
    n_repeats=1,
    n_episodes=200,
    render=False,
    max_steps=1000,
    seed=1337,
):
    env = gym.make(env_name)
    env.seed(seed)
    env._max_episode_steps = max_steps
    np.random.seed(seed)
    all_scores = []
    for repeat in range(n_repeats):
        print(f" ### Repeat {repeat + 1} of {n_repeats} ###")
        agent = agent_func()
        episode_scores = []
        n_episodes = n_episodes
        for episode in range(n_episodes):
            observation = env.reset()
            done = False
            reward = None
            episode_rewards = []
            while not done:
                if render:
                    env.render()
                action = agent.act(observation, reward, done)
                observation, reward, done, info = env.step(action)
                episode_rewards.append(reward)
            agent.act(observation, reward, done)
            episode_score = sum(episode_rewards)
            episode_scores.append(episode_score)
            print(episode + 1, episode_score)
        all_scores.append(episode_scores)
    env.close()
    return all_scores


def plot_means(results, show_std=True):
    m = np.mean(results, axis=0)
    plt.plot(m)
    if show_std:
        std = np.std(results, axis=0)
        plt.fill_between(
            np.arange(len(m)), m - std, m + std, alpha=0.2, interpolate=True
        )
    plt.show()
