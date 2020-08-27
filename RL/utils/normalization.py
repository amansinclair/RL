class Norm:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.std = 0

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(mean:{self.mean}, std:{self.std}, count:{self.count})"

    def __bool__(self):
        return bool(self.count)

    def update(self, x):
        batch_size = len(x)
        batch_mean = x.mean(dim=0)
        batch_var = x.std(dim=0, unbiased=False) ** 2
        var = self.std ** 2
        total_size = self.count + batch_size
        var = (
            ((self.count / total_size) * var)
            + ((batch_size / total_size) * batch_var)
            + (
                ((batch_size * self.count) / (total_size ** 2))
                * ((self.mean - batch_mean) ** 2)
            )
        )
        self.std = var ** 0.5
        self.mean = (self.mean * (self.count / total_size)) + (
            batch_mean * (batch_size / total_size)
        )
        self.count = total_size

    def normalize(self, x):
        return (x - self.mean) / self.std


class StdNorm(Norm):
    def normalize(self, x):
        return x / self.std

