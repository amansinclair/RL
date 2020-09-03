import gym
import numpy as np


class EnvManager:
    def __init__(self, env_name, max_steps=200):
        self.env_name = env_name
        self.max_steps = max_steps
        env = gym.make(self.env_name)
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        env.close()
        self.envs = []

    def get_env(self, seed=None):
        env = gym.make(self.env_name)
        env._max_episode_steps = self.max_steps
        if seed:
            env.seed(seed)
            np.random.seed(seed)
        self.envs.append(env)
        return env

    def __delete__(self):
        for env in self.envs:
            env.close()

