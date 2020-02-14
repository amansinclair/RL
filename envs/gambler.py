class GamblerEnv:
    def __init__(self, ph=0.4):
        self.ph = 0.4
        self.pt = 1 - ph

    def get_states(self):
        return set(i for i in range(1, 100))

    def get_transitions(self, s):
        """returns transition values (S, R, P) at s for all actions, a."""
        limit = min(s, 100 - s)
        transitions = []
        for i in range(1, limit + 1):
            win = s + i
            loss = s - i
            if win == 100:
                r = 1
            else:
                r = 0
            transitions.append((i, [win, loss], [r, 0], [self.ph, self.pt]))
        return transitions

