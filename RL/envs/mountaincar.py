import math
import numpy as np
from .env import EnvReturn


class MountainCarEnv:
    def __init__(self):
        self.actions = [1, -1, 0]
        self.reset()

    @property
    def y(self):
        return math.sin(3 * self.state[0])

    def reset(self):
        x = np.random.uniform(-0.6, -0.4)
        v = 0
        self.state = (x, v)
        return EnvReturn(self.state, None, False)

    def get_actions(self, s):
        """FORWARD, REVERSE, NOTHING -> 1, -1, 0."""
        return [0, 1, 2]

    def step(self, a):
        a = self.actions[a]
        v = self.update_vel(*self.state, a)
        x = self.update_pos(self.state[0], v)
        r = -1
        is_terminal = False
        if x == 0.5:
            is_terminal = True
        elif x == -1.2:
            s, _, __ = self.reset()
            x, v = s
        self.state = (x, v)
        return EnvReturn(self.state, r, is_terminal)

    def update_pos(self, xt, vt_1):
        x = xt + vt_1
        x = min(max(x, -1.2), 0.5)
        return x

    def update_vel(self, xt, vt, a):
        v = vt + (a * 0.001) - (0.0025 * math.cos(3 * xt))
        v = min(max(v, -0.07), 0.07)
        return v

