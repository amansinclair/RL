import numpy as np


class DP:
    def __init__(self, V, env, discount_rate=1):
        self.V = V
        self.env = env
        self.discount_rate = discount_rate

    def get_action_value(self, S, R, P):
        v = 0
        for s, r, p in zip(S, R, P):
            v += p * (r + (self.discount_rate * self.V[s]))
        return v

    def get_values(self, transitions):
        V = np.zeros(len(transitions))
        A = np.zeros(len(transitions))
        for index, transition in enumerate(transitions):
            a, S, R, P = transition
            V[index] = self.get_action_value(S, R, P)
            A[index] = a
        return V, A

    def update(self):
        diff = 0
        for s in self.env.get_states():
            v_old = self.V[s]
            transitions = self.env.get_transitions(s)
            values, actions = self.get_values(transitions)
            self.V[s] = np.max(values)
            diff = max(diff, abs(self.V[s] - v_old))
        return diff

    def get_policy(self):
        states = self.env.get_states()
        policy = np.zeros(len(states) + 1)
        for s in states:
            transitions = self.env.get_transitions(s)
            values, actions = self.get_values(transitions)
            policy[s] = actions[np.argmax(values)]
        return policy

