from RL.envs import ContinuousEnv
from RL.solvers import SemiGradSarsa as Sarsa
from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np

env = ContinuousEnv()
n_tilings = 1
limits = [(-1.2, 0.5), (-0.07, 0.07)]
divisions = [20, 20]
tiling = UniformTiling(n_tilings, limits, divisions)
tiles_per_tiling, total_tiles = tiling.get_size()
n_episodes = 100
n_trials = 10

fig, ax = plt.subplots(2, 1)
its = np.zeros(n_episodes)
for it in range(n_trials):
    W = np.zeros((4, total_tiles))
    sarsa = Sarsa(W, tiling, env, n=4, alpha=0.25)
    print("performing iteration ", it + 1)
    for ep in range(n_episodes):
        s, r, is_terminal = env.reset()
        i = 0
        while not is_terminal:
            i += 1
            a = sarsa.act(s, r, is_terminal)
            s, r, is_terminal = env.step(a)
        sarsa.act(s, r, is_terminal)
        its[ep] += i


its = its / n_trials
ax[0].plot(its)
path = []
s, r, is_terminal = env.reset()
path.append(s)
while not is_terminal:
    a = sarsa.act(s, r, is_terminal)
    s, r, is_terminal = env.step(a)
    path.append(s)
emap = env.show_path(path)
print(len(path))

ax[1].imshow(emap)
plt.show()
"""
Z = np.zeros((50, 50))
x_inc = (0.5 + 1.2) / 49
X = np.arange(-1.2, 0.5 + x_inc, x_inc)
y_inc = (0.07 + 0.07) / 49
Y = np.arange(-0.07, 0.07 + y_inc, y_inc)
actions = [0, 1, 2, 3]
values = np.zeros(sarsa.W.shape[0])
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        s = (x, y)
        index = sarsa.get_index(s)
        for a in actions:
            values[a] = np.dot(sarsa.W[a], index)
        Z[i, j] = abs(np.max(values))
plt.imshow(Z)
plt.show()
"""
