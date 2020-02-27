import numpy as np
import matplotlib.pyplot as plt

from RL.envs import TWalk
from RL.solvers import DP

V = np.zeros((1002))
env = TWalk()
dp = DP(V, env)


n_iter = 100
for i in range(n_iter):
    dp.update()

plt.plot(dp.V[1:-2])
plt.show()

