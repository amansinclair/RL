import numpy as np
import matplotlib.pyplot as plt


from solvers.mc import MCOnPolicy
from envs.gambler import GamblerEnv


Q = np.zeros((101, 2))
env = GamblerEnv()
mc = MCOnPolicy(Q, env)

n_iter = 1
fig, axs = plt.subplots(2)
for i in range(n_iter):
    game_on = True
    mc.reset()
    s = env.reset()
    r = None
    while game_on:
        a = mc.act(s, r)
        print(s, a, r, game_on)
        s, r, game_on = env.step(a)
