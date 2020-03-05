import numpy as np


class UniformTiling:
    def __init__(self, n_tilings, limits, n_divisions):
        self.n_tilings = n_tilings
        self.limits = limits
        self.n_divisions = [1 + division for division in n_divisions]
        self.n_tiles, self.total_tiles = self.get_size()
        self.offsets, self.divisions = self.create_tilings(n_divisions)

    def get_size(self):
        total = 1
        for division in self.n_divisions:
            total *= division
        return total, total * self.n_tilings

    def get_index(self, s):
        dims = np.array(s)
        _, total_size = self.get_size()
        indexes = np.zeros(total_size)
        mins = [lim[0] for lim in self.limits]
        for tiling in range(self.n_tilings):
            index = (dims + (tiling * self.offsets) - mins) // self.divisions
            index = self.set_bounds(index)
            index = index.astype("int")
            index = np.ravel_multi_index(index, self.n_divisions)
            indexes[index + (self.n_tiles * tiling)] = 1
        return indexes

    def set_bounds(self, index):
        for i in range(index.shape[0]):
            index[i] = max(0, min(self.n_divisions[i], index[i]))
        return index

    def create_tilings(self, n_divisions):
        offsets = np.zeros(len(self.limits))
        divisions = np.zeros(len(self.limits))
        i = 0
        for limit, n_divs in zip(self.limits, n_divisions):
            x_min, x_max = limit
            w = (x_max - x_min) / n_divs
            divisions[i] = w
            offsets[i] = w / self.n_tilings
            i += 1
        return offsets, divisions

