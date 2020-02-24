import numpy as np
import matplotlib.pyplot as plt

from RL.envs import CliffWalk, get_path
from RL.solvers import Sarsa


Q = np.zeros((12, 4, 4))
env = CliffWalk()
sarsa = Sarsa(Q, env, alpha=0.5)


n_episodes = 500
steps = np.zeros(n_episodes)
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    a = sarsa.reset(s)
    s, r, is_terminal = env.step(a)
    r_total = 0
    while not (is_terminal):
        a = sarsa.act(s, r)
        s, r, is_terminal = env.step(a)
        r_total += r
    sarsa.act(s, r)
    r_total += r
    steps[ep] = r_total


fig, axs = plt.subplots(2, 1)

axs[0].plot([i for i in range(n_episodes)], steps)


grid = np.zeros((12, 4), dtype="int")
grid[11, 0] = 10

path = get_path(sarsa, env)

for p in path:
    grid[p] = 5
axs[1].imshow(grid.T)

plt.show()

