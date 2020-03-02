from RL.envs import MountainCarEnv
from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np


env = MountainCarEnv()
n_tilings = 8
limits = [(-1.2, 0.5), (-0.07, 0.07)]
divisions = [8, 8]


tiling = UniformTiling(n_tilings, limits, divisions)
tiles_per_tiling, total_tiles = tiling.get_size()
# s = (0, 0)
# print(tiling.get_index(s))

W = np.zeros((3, total_tiles))
sarsa = Sarsa(W, tiling, env, alpha=0.125)
n_episodes = 200
n_trials = 100

its = np.zeros(n_trials)
for it in range(n_trials):
    print("performing iteration ", it + 1)
    for ep in range(n_episodes):
        s, r, is_terminal = env.reset()
        i = 0
        while not is_terminal:
            i += 1
            a = sarsa.act(s, r, is_terminal)
            st = s
            s, r, is_terminal = env.step(a)
        sarsa.act(s, r, is_terminal)
        its[ep] += i

its = its / n_trials


plt.plot(its)
plt.show()
