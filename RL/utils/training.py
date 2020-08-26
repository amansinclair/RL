import gym
import numpy as np
import matplotlib.pyplot as plt


class EnvManager:
    def __init__(self, env_name, max_steps=200, seed=1337):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.env._max_episode_steps = max_steps
        np.random.seed(seed)

    def __enter__(self):
        return self.env

    def __exit__(self, *args):
        self.env.close()


def get_env_size(env_name):
    with EnvManager(env_name) as env:
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n
    return n_inputs, n_outputs


def run_episode(env, agent, render=False):
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
    return episode_score


def run(env, agent, n_episodes=200, render=False):
    episode_scores = []
    print(f"####### Training Environment {env}, episodes {n_episodes} #######")
    for episode in range(n_episodes):
        episode_score = run_episode(env, agent, render)
        episode_scores.append(episode_score)
        print("Episode:", episode + 1, "Score:", episode_score)
    return episode_scores


def evaluate(env, agent_func, n_repeats=1, n_episodes=200, render=False):
    all_scores = []
    print(
        f"####### Eval Environment {env}, repeats {n_repeats}, episodes {n_episodes} #######"
    )
    for repeat in range(n_repeats):
        print(f" ### Repeat {repeat + 1} of {n_repeats} ###")
        agent = agent_func()
        episode_scores = run(env, agent, n_episodes, render)
        all_scores.append(episode_scores)
    return all_scores


def plot_means(results, show_std=True):
    if isinstance(results[0], list):
        m = np.mean(results, axis=0)
    else:
        m = results
    plt.plot(m)
    if show_std:
        std = np.std(results, axis=0)
        plt.fill_between(
            np.arange(len(m)), m - std, m + std, alpha=0.2, interpolate=True
        )
    plt.show()
