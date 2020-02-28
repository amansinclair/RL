import numpy as np
import matplotlib.pyplot as plt

from RL.envs import TWalk
from RL.solvers import DP, MCGrad, TDGrad

V = np.zeros((1002))
W = np.zeros((10 + 2))
env = TWalk()
dp = DP(V, env)
mc = MCGrad(W)
td = TDGrad(W.copy())

"""
print("dp - ing")
n_iter = 50
for i in range(n_iter):
    dp.update()

plt.plot(dp.V[1:-2])


print("mc - ing")
n_episodes = 200000
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    while not is_terminal:
        mc.act(s, r, is_terminal)
        s, r, is_terminal = env.step()
    mc.act(s, r, is_terminal)
plt.plot(mc.V[1:-2])
"""
print("td - ing")
n_episodes = 10000
for ep in range(n_episodes):
    s, r, is_terminal = env.reset()
    while not is_terminal:
        td.act(s, r, is_terminal)
        s, r, is_terminal = env.step()
    td.act(s, r, is_terminal)
plt.plot(td.V[1:-2])
plt.show()

