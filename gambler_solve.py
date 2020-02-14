import numpy as np
import matplotlib.pyplot as plt


from solvers.dp import DP
from envs.gambler import GamblerEnv

V = np.zeros(101)
env = GamblerEnv()
dp = DP(V, env)

n_iter = 100
fig, axs = plt.subplots(2)
plot_idx = (1, 2, 3, 32)
for i in range(n_iter):
    dp.update()
    if i + 1 in plot_idx:
        axs[0].plot(dp.V[1:100], label=f"sweep {i + 1}")
axs[0].legend()

p = dp.get_policy()
axs[1].plot(p)
plt.show()

