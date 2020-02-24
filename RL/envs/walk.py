import numpy as np
from itertools import product
from .env import EnvReturn, Transition


class RandomWalk:
    def __init__(self, size=5):
        self.size = size
        self.right_terminal = self.size + 1
        self.state = None
        self.reset()

    def reset(self):
        self.state = int(np.ceil(self.size / 2))
        return EnvReturn(self.state, None, False)

    def get_states(self):
        return [s for s in range(1, self.size + 1)]

    def get_actions(self, s):
        return []

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s."""
        transitions = []
        right = s + 1
        if right == self.right_terminal:
            r_right = 1
        else:
            r_right = 0
        t = Transition(None, [s - 1, s + 1], [0, r_right], [0.5, 0.5])
        transitions.append(t)
        return transitions

    def choice(self):
        return np.random.choice([-1, 1])

    def step(self):
        self.state += self.choice()
        if self.state == 0:
            r = 0
            is_terminal = True
        elif self.state == self.right_terminal:
            r = 1
            is_terminal = True
        else:
            r = 0
            is_terminal = False
        return EnvReturn(self.state, r, is_terminal)

    def get_probs(self, s):
        right = s + 1
        if right == 6:
            r_right = 1
        else:
            r_right = 0
        return [(s - 1, 0.5, 0), (right, 0.5, r_right)]


class CliffWalk:  # 4 x 12
    def __init__(self):
        self.cliff = set((x, 0) for x in range(1, 11))
        self.start = (0, 0)
        self.goal = (11, 0)

    def reset(self):
        self.state = self.start
        return EnvReturn(self.state, None, False)

    def get_states(self):
        xs = [x for x in range(12)]
        ys = [y for y in range(4)]
        states = set(xy for xy in product(xs, ys))
        return states - ({self.goal} | self.cliff)

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

    def step(self, a):
        dx = 0
        dy = 0
        if a == 0:
            dy = 1
        elif a == 1:
            dy = -1
        elif a == 2:
            dx = 1
        else:
            dx = -1
        y = self.check_y(self.state[1] + dy)
        x = self.check_x(self.state[0] + dx)
        if (x, y) in self.cliff:
            r = -100
            self.reset()
        else:
            self.state = (x, y)
            r = -1
        return EnvReturn(self.state, r, (self.state == self.goal))

    def check_x(self, x):
        return max(min(11, x), 0)

    def check_y(self, y):
        return max(min(3, y), 0)
