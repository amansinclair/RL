import numpy as np


class Sarsa:
    def __init__(self, Q, env, discount_rate=1, e=0.1, alpha=0.01):
        self.Q = Q
        self.env = env
        self.e = e
        self.discount_rate = discount_rate
        self.alpha = alpha
        self.policy = np.zeros(Q.shape)

    def reset(self, s, a=None):
        self.current_state = s
        if a != None:
            self.current_action = a
        else:
            self.current_action = self.choose(s)
        return self.current_action

    def choose(self, s):
        actions = self.env.get_actions(s)
        ps = np.take(self.policy, actions)
        if np.sum(ps) == 0:
            a = np.random.choice(actions)
        else:
            a = np.random.choice(actions, p=ps)
        return a

    def act(self, s, r):
        current_index = self.get_index(self.current_state, self.current_action)
        current_value = self.Q[current_index]
        if s in self.env.get_states():
            next_action = self.choose(s)
            next_index = self.get_index(s, next_action)
            next_value = self.Q[next_index]
            self.reset(s, next_action)
        else:
            next_value = 0
            next_action = 0
        self.Q[current_index] = current_value + (
            self.alpha * (r + (self.discount_rate * next_value) - current_value)
        )
        return next_action

    def get_index(self, s, a):
        try:
            index = (*s, a)
        except TypeError:
            index = (s, a)
        return index
