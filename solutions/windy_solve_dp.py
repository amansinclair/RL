import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import WindyEnv
from solvers.dp import DP

V = np.zeros((10, 7))
env = WindyEnv()
dp = DP(V, env)


n_iter = 20
for i in range(n_iter):
    dp.update()


policy = dp.get_policy()
s, r, is_terminal = env.reset()
path = []
path.append(s)
while not (is_terminal):
    a = policy[s]
    s, r, is_terminal = env.step(a)
    path.append(s)

grid = np.zeros((10, 7))
non_terminal = env.get_states()
for state in path:
    grid[state] = 1

plt.imshow(grid)
plt.show()
