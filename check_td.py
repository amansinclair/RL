import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import WindyEnv
from solvers.dp import DP
from solvers.td import TDV

V = np.zeros((10, 7))
V2 = V.copy()
env = WindyEnv()
dp = DP(V, env)

n_iter = 20
for i in range(n_iter):
    dp.update()

policy = dp.get_policy()
print(dp.V)
td = TDV(V2, env, policy)

n_episodes = 500
for ep in range(n_episodes):
    s = env.reset()
    td.reset()
    terminal = False
    r = None
    while not terminal:
        a = td.update(s, r, terminal)
        s, r, game_on = env.step(a)
        terminal = not (game_on)
    td.update(s, r)

print(td.V)
