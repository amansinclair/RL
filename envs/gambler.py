import numpy as np


class GamblerEnv:
    def __init__(self, ph=0.4):
        self.ph = 0.4
        self.pt = 1 - ph

    def reset(self):
        self.state = np.random.randint(1, 100)
        return self.state

    def get_states(self):
        return set(i for i in range(1, 100))

    def get_actions(self, s):
        return [i for i in range(1, min(s, 100 - s) + 1)]

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s for all actions, a."""
        transitions = []
        for i in self.get_actions(s):
            win = s + i
            loss = s - i
            r = 0
            if win == 100:
                r = 1
            transitions.append((i, [win, loss], [r, 0], [self.ph, self.pt]))
        return transitions

    def step(self, a):
        result = np.random.choice([0, 1], p=[self.pt, self.ph])
        r = 0
        game_on = True
        if result:
            self.state += a
            if self.state == 100:
                r = 1
                game_on = False
        else:
            self.state -= a
            if self.state == 0:
                game_on = False
        return self.state, r, game_on

