import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import WindyEnv
from solvers.qlearn import QLearn


def get_path(solver, env):
    path = []
    game_on = True
    s = env.reset()
    while game_on:
        path.append(s)
        s, r, game_on = env.step(np.argmax(solver.Q[s]))
    return path


Q = np.zeros((10, 7, 4))
env = WindyEnv()
qlearn = QLearn(Q, env, alpha=0.5)


n_episodes = 500
steps = np.zeros(n_episodes)
total_steps = 0
for ep in range(n_episodes):
    s = env.reset()
    r = None
    qlearn.reset()
    game_on = True
    while game_on:
        a = qlearn.update(s, r)
        s, r, game_on = env.step(a)
        total_steps += 1
    steps[ep] = total_steps


ig, axs = plt.subplots(2, 1)

axs[0].plot(steps, [i for i in range(n_episodes)])


grid = np.zeros((10, 7), dtype="int")
grid[7, 3] = 10

path = get_path(qlearn, env)

for p in path:
    grid[p] = 5
axs[1].imshow(grid.T)
plt.show()


plt.show()

