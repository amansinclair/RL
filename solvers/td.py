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
            self.r.popleft()
            self.update_v(s)
