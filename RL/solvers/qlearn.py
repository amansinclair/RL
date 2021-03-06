import numpy as np


class QLearn:
    def __init__(self, Q, env, discount_rate=1, e=0.1, alpha=0.01):
        self.Q = Q
        self.env = env
        self.discount_rate = discount_rate
        self.e = e
        self.alpha = alpha

    def reset(self):
        self.previous = None

    def choose(self, s):
        actions = self.env.get_actions(s)
        random_a = np.random.choice(actions)
        values = np.take(self.Q[s], actions)
        best_a = actions[np.argmax(values)]
        return np.random.choice([random_a, best_a], p=[self.e, 1 - self.e])

    def update(self, s, r=None):
        if self.previous:
            q_old = self.Q[self.previous]
            self.Q[self.previous] = q_old + self.alpha * (
                r + (self.discount_rate * np.max(self.Q[s])) - q_old
            )
        a = self.choose(s)
        self.previous = self.get_index(s, a)
        return a

    def get_index(self, s, a):
        try:
            index = (*s, a)
        except TypeError:
            index = (s, a)
        return index

