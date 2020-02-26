import numpy as np
from collections import deque


class Sarsa:
    def __init__(self, Q, env, discount_rate=1, e=0.1, alpha=0.1):
        self.Q = Q
        self.env = env
        self.e = e
        self.discount_rate = discount_rate
        self.alpha = alpha

    def reset(self, s, a=None):
        self.current_state = s
        if a != None:
            self.current_action = a
        else:
            self.current_action = self.choose(s)
        return self.current_action

    def choose(self, s):
        actions = self.env.get_actions(s)
        values = np.take(self.Q[s], actions)
        best_a = actions[np.argmax(values)]
        c = np.random.choice([0, 1], p=[self.e, 1 - self.e])
        if c:
            return best_a
        else:
            return np.random.choice(actions)

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


class NStepSarsa(Sarsa):
    def __init__(self, *args, n=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def reset(self):
        self.states = deque(maxlen=self.n)
        self.actions = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)
        self.t = 0

    def sum_rewards(self):
        d = self.discount_rate
        total = 0
        for r in self.rewards:
            total += d * r
            d = d * self.discount_rate
        return total

    def act(self, s, r=None, terminal=False):
        a = self.choose(s)
        if r != None:
            self.rewards.append(r)
        if terminal:
            self.finish()
        elif self.t >= self.n:
            index = self.get_index(self.states[0], self.actions[0])
            self.update_q(index, s, a)
        self.states.append(s)
        self.actions.append(a)
        self.t += 1
        return a

    def update_q(self, index, s=None, a=None):
        q_old = self.Q[index]
        G = self.sum_rewards()
        if s:
            G += (self.discount_rate ** self.n) * self.Q[self.get_index(s, a)]
        self.Q[index] = q_old + (self.alpha * (G - q_old))

    def finish(self):
        for s, a in zip(self.states, self.actions):
            index = self.get_index(s, a)
            self.rewards.popleft()
            self.update_q(index)