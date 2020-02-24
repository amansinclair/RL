import numpy as np
import matplotlib.pyplot as plt

from RL.envs import WindyEnv, get_path
from RL.solvers import QLearn


Q = np.zeros((10, 7, 4))
env = WindyEnv()
qlearn = QLearn(Q, env, alpha=0.5)


n_episodes = 500
steps = np.zeros(n_episodes)
total_steps = 0
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    qlearn.reset()
    while not (is_terminal):
        a = qlearn.update(s, r)
        s, r, is_terminal = env.step(a)
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

