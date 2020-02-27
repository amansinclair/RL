import numpy as np


class MCOnPolicy:
    def __init__(self, Q, env, discount_rate=1, e=0.1):
        self.Q = Q
        self.env = env
        self.e = e
        self.discount_rate = discount_rate
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
        if r != None:
            self.rewards.append(r)
        a = self.choose(s)
        self.actions.append(a)
        return a

    def get_index(self, s, a):
        try:
            index = (*s, a)
        except TypeError:
            index = (s, a)
        return index

    def update(self, r):
        self.rewards.append(r)
        G = 0
        for i in range(len(self.states)):
            s = self.states.pop()
            a = self.actions.pop()
            r = self.rewards.pop()
            G = (G * self.discount_rate) + r
            if s not in self.states:
                index = self.get_index(s, a)
                self.N[index] += 1
                self.Q[index] += (G - self.Q[index]) / self.N[index]
                A = np.argmax(self.Q[index[:-1]])
                possible_actions = self.env.get_actions(s)
                for a in possible_actions:
                    index = self.get_index(s, a)
                    if a == A:
                        self.policy[index] = (
                            1 - self.e + (self.e / len(possible_actions))
                        )
                    else:
                        self.policy[index] = self.e / (len(possible_actions))


class MCGrad:
    """Only for 1000 random Walk testing."""

    def __init__(self, W, discount_rate=1, alpha=0.00002):
        self.W = W
        self.discount_rate = discount_rate
        self.alpha = alpha
        self.N = np.zeros(len(W))
        self.reset()

    def reset(self):
        self.states = []
        self.rewards = []

    def act(self, s, r=None, is_terminal=False, reset=True):
        if r != None:
            self.rewards.append(r)
        if is_terminal:
            self.update()
            if reset:
                self.reset()
        else:
            self.states.append(s)

    def update(self):
        G = 0
        for i in range(len(self.states)):
            s = self.states.pop()
            index = s // 100
            r = self.rewards.pop()
            G = r + (self.discount_rate * G)
            w_old = self.W[index]
            self.W[index] += self.alpha * (G - w_old)

    @property
    def V(self):
        size = 1002
        V = np.zeros(size)
        for i in range(1, size - 1):
            V[i] = self.W[(i // 100) + 1]
        return V

