import numpy as np
import matplotlib.pyplot as plt


from solvers.mc import MCOnPolicy
from envs.gambler import GamblerEnv


Q = np.zeros((101, 101))
env = GamblerEnv()
mc = MCOnPolicy(Q, env, e=0.2)

n_iter = 20000
fig, axs = plt.subplots(2)
for i in range(n_iter):
    game_on = True
    mc.reset()
    s = env.reset()
    r = None
    while game_on:
        a = mc.act(s, r)
        s, r, game_on = env.step(a)
    mc.update(r)
    if i % 5000 == 0 and i != 0:
        axs[0].plot(np.max(mc.Q, axis=1))
axs[1].plot(np.argmax(mc.Q, axis=1))
plt.show()
