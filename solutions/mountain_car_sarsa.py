from RL.envs import MountainCarEnv
import matplotlib.pyplot as plt
import numpy as np

mountain = MountainCarEnv()
a = 0
print(mountain.reset())
xs = np.zeros(100)
ys = np.zeros(100)
for i in range(100):
    s, r, t = mountain.step(a)
    xs[i] = s[0]
    ys[i] = mountain.y

plt.plot(ys, "o")
plt.show()
