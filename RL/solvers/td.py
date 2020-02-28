import numpy as np
from collections import deque


class TDV:
    def __init__(self, V, env, policy, n=3, discount_rate=1, e=0.1, alpha=0.1):
        self.V = V
        self.env = env
        self.policy = policy
        self.e = e
        self.n = n
        self.discount_rate = discount_rate
        self.alpha = alpha

    def reset(self):
        self.states = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)
        self.t = 0

    def sum_rewards(self):
        d = self.discount_rate
        total = 0
        for r in self.rewards:
            total += d * r
            d = d * self.discount_rate
        return total

    def update(self, s, r=None, terminal=False):
        a = None
        if r != None:
            self.rewards.append(r)
        if terminal:
            self.finish()
        elif self.t >= self.n:
            index = self.states[0]
            self.update_v(index, s)
        a = self.policy[s]
        self.states.append(s)
        self.t += 1
        return a

    def update_v(self, index, s=None):
        v_old = self.V[index]
        G = self.sum_rewards()
        if s:
            G += (self.discount_rate ** self.n) * self.V[s]
        self.V[index] = v_old + (self.alpha * (G - v_old))

    def finish(self):
        for s in self.states:
            self.rewards.popleft()
            self.update_v(s)


class TDGrad:
    """TD State Agg. Only for 1000 random Walk testing."""

    def __init__(self, W, n=4, discount_rate=1, alpha=0.4):
        self.W = W
        self.n = n
        self.discount_rate = discount_rate
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.t = 0
        self.states = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)

    def act(self, s, r=None, is_terminal=False):
        if r != None:
            self.rewards.append(r)
        if is_terminal:
            self.finish()
        elif self.t >= self.n:
            index = self.states[0] // 100
            self.update_v(index, s)
        self.states.append(s)
        self.t += 1

    def sum_rewards(self):
        d = self.discount_rate
        total = 0
        for r in self.rewards:
            total += d * r
            d = d * self.discount_rate
        return total

    def update_v(self, index, s=None):
        v_old = self.W[index]
        G = self.sum_rewards()
        if s:
            s = s // 100
            G += (self.discount_rate ** self.n) * self.V[s]
        self.W[index] = v_old + (self.alpha * (G - v_old))

    def finish(self):
        for s in self.states:
            self.rewards.popleft()
            self.update_v(s // 100)
        self.reset()

    @property
    def V(self):
        size = 1002
        V = np.zeros(size)
        for i in range(1, size - 1):
            V[i] = self.W[(i // 100) + 1]
        return V
