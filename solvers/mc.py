import numpy as np


class MCOnPolicy:
    def __init__(self, Q, env, e=0.1):
        self.Q = Q
        self.env = env
        self.e = e
        self.N = np.zeros(Q.shape, dtype="int")
        self.n_actions = Q.shape[1]
        self.action_space = np.arange(self.n_actions)
        self.policy = np.zeros(Q.shape)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def choose(self, s):
        actions = self.env.get_actions(s)
        ps = np.take(self.policy, actions)
        if np.sum(ps) == 0:
            a = np.random.choice(actions)
        else:
            a = np.random.choice(actions, p=ps)
        return a

    def act(self, s, r=None):
        self.states.append(s)
        if r:
            self.rewards.append(r)
        a = self.choose(s)
        self.actions.append(a)
        return a

    def update(self, r):
        self.rewards.append(r)
        G = 0
        for i in range(len(self.states)):
            s = self.states.pop()
            a = self.actions.pop()
            r = self.rewards.pop()
            G = (G * self.discount_rate) + r
            if s not in self.states:
                index = (*s, a)
                self.N[index] += 1
                self.Q[index] += (G - self.Q[index]) / self.N[index]
                A = np.argmax(self.Q[index[:-1]])
                for action in self.env.get_actions(s):
                    index = (*s, action)
                    if action == A:
                        self.policy[index] = 1 - e + (e / 9)
                    else:
                        self.policy[index] = e / 9

