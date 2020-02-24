import numpy as np

from .env import EnvReturn, Transition


class GamblerEnv:
    def __init__(self, ph=0.4, win_goal=100):
        self.ph = 0.4
        self.pt = 1 - ph
        self.win_goal = win_goal

    def reset(self):
        self.state = np.random.randint(1, self.win_goal)
        return EnvReturn(self.state, None, False)

    def get_states(self):
        return set(i for i in range(1, self.win_goal))

    def get_actions(self, s):
        return [i for i in range(1, min(s, self.win_goal - s) + 1)]

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s for all actions, a."""
        transitions = []
        for i in self.get_actions(s):
            win = s + i
            loss = s - i
            r = 0
            if win == 100:
                r = 1
            t = Transition(i, [win, loss], [r, 0], [self.ph, self.pt])
            transitions.append(t)
        return transitions

    def step(self, a):
        result = np.random.choice([0, 1], p=[self.pt, self.ph])
        r = 0
        is_terminal = False
        if result:
            self.state += a
            if self.state == self.win_goal:
                r = 1
                is_terminal = True
        else:
            self.state -= a
            if self.state == 0:
                is_terminal = False
        return EnvReturn(self.state, r, is_terminal)

