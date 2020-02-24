import numpy as np
import matplotlib.pyplot as plt


from RL.solvers import Sarsa
from RL.envs import GamblerEnv


Q = np.zeros((101, 101))
env = GamblerEnv()
sarsa = Sarsa(Q, env, alpha=0.1)

n_episodes = 20000
fig, axs = plt.subplots(2)
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    a = sarsa.reset(s)
    s, r, is_terminal = env.step(a)
    while not (is_terminal):
        a = sarsa.act(s, r)
        s, r, is_terminal = env.step(a)
    sarsa.act(s, r)
    if ep % 5000 == 0 and ep != 0:
        axs[0].plot(np.max(sarsa.Q, axis=1))
axs[0].plot(np.max(sarsa.Q, axis=1))
axs[1].plot(np.argmax(sarsa.Q, axis=1))
plt.show()

