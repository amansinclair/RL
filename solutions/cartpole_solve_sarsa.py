from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np
import gym


class Env:
    def __init__(self, cartpole):
        self.cartpole = cartpole

    def get_actions(self, s):
        return [0, 1]

    def step(self, a):
        s, r, is_terminal, _ = self.cartpole.step(a)
        return tuple(s), r, is_terminal

    def reset(self):
        return self.cartpole.reset()

    def close(self):
        self.cartpole.close()


n_tilings = 2
limits = [(-5, 5), (-10, 10), (-0.5, 0.5), (-10, 10)]
divisions = [8, 8, 8, 8]


tiling = UniformTiling(n_tilings, limits, divisions)
tiles_per_tiling, total_tiles = tiling.get_size()
env = Env(gym.make("CartPole-v0"))

W = np.zeros((4, total_tiles))
sarsa = Sarsa(W, tiling, env, alpha=0.125)
n_episodes = 2
for ep in range(n_episodes):
    s = env.reset()
    r = None
    is_terminal = False
    i = 0
    while not (is_terminal):
        i += 1
        a = int(sarsa.act(s, r, is_terminal))
        s, r, is_terminal = env.step(a)
        print(s)
    sarsa.act(s, r, is_terminal)
    print(i)
s = env.reset()
env.close()

