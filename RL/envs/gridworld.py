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
