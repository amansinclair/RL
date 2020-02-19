import numpy as np
import matplotlib.pyplot as plt

from envs.walk import CliffWalk
from solvers.dp import DP

V = np.zeros((12, 4))
env = CliffWalk()
dp = DP(V, env)


n_iter = 20
for i in range(n_iter):
    dp.update()


policy = dp.get_policy()

s = env.reset()
path = []
path.append(s)
game_on = True
while game_on:
    a = policy[s]
    s, r, game_on = env.step(a)
    path.append(s)


grid = np.zeros((12, 4))
non_terminal = env.get_states()
for state in path:
    grid[state] = 1

plt.imshow(grid)
plt.show()

