from RL.envs import MountainCarEnv
from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

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
n_episodes = 12
n_trials = 1

its = np.zeros(n_episodes)
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
Z = np.zeros((50, 50))
x_inc = (0.5 + 1.2) / 49
X = np.arange(-1.2, 0.5 + x_inc, x_inc)
y_inc = (0.07 + 0.07) / 49
Y = np.arange(-0.07, 0.07 + y_inc, y_inc)
actions = [0, 1, 2]
values = np.zeros(sarsa.W.shape[0])
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        s = (x, y)
        index = sarsa.get_index(s)
        for a in actions:
            values[a] = np.dot(sarsa.W[a], index)
        Z[i, j] = abs(np.max(values))

plt.imshow(Z)
# fig = plt.figure()
# ax = plt.axes(projection="3d")

# ax.plot_surface(X, Y, Z)
# plt.plot(its)
plt.show()

