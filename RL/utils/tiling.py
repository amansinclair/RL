import numpy as np


class UniformTiling:
    def __init__(self, n_tilings, limits, n_divisions):
        self.n_tilings = n_tilings
        self.limits = limits
        self.n_divisions = n_divisions
        self.n_tiles, self.total_tiles = self.get_size()
        self.offsets, self.divisions = self.create_tilings()

    def get_size(self):
        total = 1
        for division in self.n_divisions:
            total *= division
        return total, total * self.n_tilings

    def get_index(self, s):
        dims = np.array(s)
        indexes = np.zeros(self.n_tilings, dtype="int")
        for tiling in range(self.n_tilings):
            index = (dims + (tiling * self.offsets)) // self.divisions
            index = index.astype("int")
            index = np.ravel_multi_index(index, self.n_divisions)
            indexes[tiling] = index + (self.n_tiles * tiling)
        return indexes

    def create_tilings(self):
        offsets = np.zeros(len(self.limits))
        divisions = np.zeros(len(self.limits))
        i = 0
        for limit, n_divs in zip(self.limits, self.n_divisions):
            x_min, x_max = limit
            w = (x_max - x_min) / n_divs
            divisions[i] = w
            offsets[i] = w / self.n_tilings
            i += 1
        return offsets, divisions


n_tilings = 4
limits = [(0, 10), (0, 5)]
divisions = [5, 5]

ut = UniformTiling(n_tilings, limits, divisions)

print(ut.get_index((0.49, 1)))

