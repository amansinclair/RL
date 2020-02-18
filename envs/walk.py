import numpy as np


class RandomWalk:
    def __init__(self):
        self.state = None
        self.reset()

    def reset(self):
        self.state = 3

    def get_states(self):
        return [s for s in range(1, 6)]

    def get_actions(self, s):
        return []

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s."""
        transitions = []
        right = s + 1
        if right == 6:
            r_right = 1
        else:
            r_right = 0
        return [(s - 1, 0.5, 0), (right, 0.5, r_right)]
        transitions.append((None, [s - 1, s + 1], [0, r_right], [0.5, 0.5]))
        return transitions

    def choice(self):
        return np.random.choice([-1, 1])

    def step(self):
        self.state += self.choice()
        if self.state == 0:
            r = 0
            game_on = False
        elif self.state == 6:
            r = 1
            game_on = False
        else:
            r = 0
            game_on = True
        return self.state, r, game_on

    def get_probs(self, s):
        right = s + 1
        if right == 6:
            r_right = 1
        else:
            r_right = 0
        return [(s - 1, 0.5, 0), (right, 0.5, r_right)]
