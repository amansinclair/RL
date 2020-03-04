import numpy as np

from itertools import product
from .env import EnvReturn, Transition


class WindyEnv:
    def __init__(self):
        self.winds = np.zeros(10, dtype="int")
        self.winds[3:9] = 1
        self.winds[6:8] = 2
        self.goal = (7, 3)

    def get_states(self):
        xs = [x for x in range(10)]
        ys = [y for y in range(7)]
        states = set(xy for xy in product(xs, ys))
        return states - {self.goal}

    def get_actions(self, s):
        """UP, DOWN, RIGHT, LEFT -> 0, 1, 2, 3."""
        return [a for a in range(4)]

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s for all actions, a."""
        transitions = []
        for a in self.get_actions(s):
            self.state = s
            s_, r_, _ = self.step(a)
            t = Transition(a, [s_], [r_], [1])
            transitions.append(t)
        return transitions

    def reset(self):
        self.state = (0, 3)
        return EnvReturn(self.state, None, False)

    def step(self, a):
        wind = self.winds[self.state[0]]
        if a == 0:
            dx = 0
            dy = 1 + wind
        elif a == 1:
            dx = 0
            dy = wind - 1
        elif a == 2:
            dx = 1
            dy = wind
        else:
            dx = -1
            dy = wind
        y = self.check_y(self.state[1] + dy)
        x = self.check_x(self.state[0] + dx)
        self.state = (x, y)
        return EnvReturn(self.state, -1, (self.state == self.goal))

    def check_x(self, x):
        return max(min(9, x), 0)

    def check_y(self, y):
        return max(min(6, y), 0)


class ContinuousEnv:
    def __init__(self):
        self.x_min = -1.2
        self.x_max = 0.5
        self.y_min = -0.07
        self.y_max = 0.07
        self.holes = [(0.0, 0.0), (-1.0, -0.03), (0.3, 0.05), (0.3, -0.05)]

    def get_actions(self, s):
        """UP, DOWN, RIGHT, LEFT -> 0, 1, 2, 3."""
        return [a for a in range(4)]

    def reset(self):
        x = np.random.uniform(-1.1, -0.9)
        y = 0
        self.state = (x, y)
        return EnvReturn(self.state, None, False)

    def step(self, a):
        if a == 0:
            dx = 0
            dy = 0.007
        elif a == 1:
            dx = 0
            dy = -0.007
        elif a == 2:
            dx = 0.085
            dy = 0
        else:
            dx = -0.085
            dy = 0
        y = self.check_y(self.state[1] + dy)
        x = self.check_x(self.state[0] + dx)
        if self.is_in_hole(x, y):
            return self.reset()
        self.state = (x, y)
        return EnvReturn(self.state, -1, (x >= self.x_max))

    def check_x(self, x):
        return max(min(self.x_max, x), self.x_min)

    def check_y(self, y):
        return max(min(self.y_max, y), self.y_min)

    def is_in_hole(self, x, y):
        for hole_x, hole_y in self.holes:
            if x > hole_x - 0.1 and x < hole_x + 0.1:
                if y > hole_y - 0.01 and y < hole_y + 0.01:
                    return True
        return False

    def get_map(self):
        envmap = np.zeros((100, 100))
        dx = (self.x_max - self.x_min) / 100
        dy = (self.y_max - self.y_min) / 100
        for i in range(100):
            x = self.x_min + (i * dx)
            for j in range(100):
                y = self.y_min + (j * dy)
                if self.is_in_hole(x, y):
                    envmap[i, j] = 10
        return envmap

    def show_path(self, states):
        envmap = self.get_map()
        ds = 10 / len(states)
        shade = 1
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        for x, y in states:
            shade += ds
            x = int(((x - self.x_min) / x_range) * 99)
            y = int(((y - self.y_min) / y_range) * 99)
            envmap[x, y] = shade
        return envmap

