import numpy as np
import matplotlib.pyplot as plt

from envs.walk import CliffWalk
from solvers.sarsa import NStepSarsa


def get_path(solver, env):
    path = []
    game_on = True
    s = env.reset()
    while game_on:
        path.append(s)
        s, r, game_on = env.step(np.argmax(solver.Q[s]))
    return path


Q = np.zeros((12, 4, 4))
env = CliffWalk()
sarsa = NStepSarsa(Q, env, n=2, alpha=0.5)


n_episodes = 500
steps = np.zeros(n_episodes)
for ep in range(n_episodes):
    s = env.reset()
    sarsa.reset()
    game_on = True
    r = None
    r_total = 0
    while game_on:
        a = sarsa.act(s, r)
        s, r, game_on = env.step(a)
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

