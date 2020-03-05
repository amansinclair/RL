from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np
import gym
import time


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

    def render(self):
        self.cartpole.render()


n_tilings = 2
limits = [(-5, 5), (-5, 5), (-0.5, 0.5), (-5, 5)]
divisions = [5, 5, 5, 5]


tiling = UniformTiling(n_tilings, limits, divisions)
tiles_per_tiling, total_tiles = tiling.get_size()
env = Env(gym.make("CartPole-v0"))

W = np.zeros((2, total_tiles))
sarsa = Sarsa(W, tiling, env, alpha=0.125)
n_episodes = 100
for ep in range(n_episodes):
    s = env.reset()
    r = None
    is_terminal = False
    i = 0
    while not (is_terminal):
        i += 1
        a = int(sarsa.act(s, r, is_terminal))
        s, r, is_terminal = env.step(a)
    sarsa.act(s, r, is_terminal)
    print(i)

s = env.reset()
r = None
is_terminal = False
i = 0
while not (is_terminal):
    env.render()
    time.sleep(0.1)
    a = int(sarsa.play(s, r, is_terminal))
    s, r, is_terminal = env.step(a)
    i += 1
env.close()
print("Greedy play result", i)

