from RL.envs import MountainCarEnv
from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np


env = MountainCarEnv()
n_tilings = 4
limits = [(-1.2, 0.5), (-0.07, 0.07)]
divisions = [5, 5]


tiling = UniformTiling(n_tilings, limits, divisions)
tiles_per_tiling, total_tiles = tiling.get_size()
# s = (0, 0)
# print(tiling.get_index(s))

W = np.zeros((3, total_tiles))
sarsa = Sarsa(W, tiling, env, alpha=1)
n_episodes = 5
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    i = 0
    while not is_terminal:
        i += 1
        a = sarsa.act(s, r, is_terminal)
        s, r, is_terminal = env.step(a)
    sarsa.act(s, r, is_terminal)
    print(i)
