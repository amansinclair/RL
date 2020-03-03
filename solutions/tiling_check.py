from RL.utils import UniformTiling
import matplotlib.pyplot as plt
import numpy as np

n_tilings = 3
limits = [(-1.2, 0.5), (-0.07, 0.07)]
divisions = [3, 3]


def ones_to_index(a):
    return np.where(a == 1)[0]


tiling = UniformTiling(n_tilings, limits, divisions)

print(tiling.get_size())
size = 1000
xs = np.random.uniform(-1.2, 0.5, size)
ys = np.random.uniform(-0.07, 0.07, size)

# print(ones_to_index(tiling.get_index((-0.7, -0.07))))

t1 = np.zeros(size)
t2 = np.zeros(size)
t3 = np.zeros(size)
for i in range(size):
    t1[i], t2[i], t3[i] = ones_to_index(tiling.get_index((xs[i], ys[i])))
fig, ax = plt.subplots(1, 2)
ax[0].plot(xs, ys, "o")
ax[1].plot(t1, "o")
ax[1].plot(t2, "o")
ax[1].plot(t3, "o")
plt.show()

