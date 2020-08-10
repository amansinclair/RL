import gym
import matplotlib.pyplot as plt
from RL.solvers import train
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v1")
env._max_episode_steps = 1000

train(env, 10, 100)
