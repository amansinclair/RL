import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


class QLearn:
    def __init__(self, net, e=0.1, discount_rate=0.8, alpha=0.01):
        self.net = net
        self.actions = [0.0, 1.0]
        self.e = e
        self.discount_rate = discount_rate
        self.alpha = alpha
        self.previous_state = None

    def get_index(self, s, a):
        return np.array((*s, a), dtype="float32")

    def act(self, s, r=None, is_terminal=False):
        if is_terminal:
            v_next = 0.0
            a_best = 0.0
        else:
            v_next, a_best = self.max_next(s)
        if r != None:
            r = float(r)
            a_best = self.update(s, r, v_next)
        a = self.choose(a_best)
        self.previous_state = self.get_index(s, a)
        return a

    def update(self, s, r, v_next):
        v_prev = self.net.predict(self.previous_state)
        v_new = v_prev + self.alpha * (r + (self.discount_rate * v_next) - v_prev)
        self.net.fit(self.previous_state, v_new)

    def choose(self, a_best):
        c = np.random.choice([0, 1], p=[self.e, 1 - self.e])
        if c:
            return a_best
        else:
            return np.random.choice(self.actions)

    def max_next(self, s):
        values = np.zeros(len(self.actions))
        for i, a in enumerate(self.actions):
            index = self.get_index(s, a)
            values[i] = self.net.predict(index)
        i_best = np.argmax(values)
        a_best = self.actions[i_best]
        v_max = np.max(values)
        return v_max, a_best


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(5, 5)
        self.out_layer = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.optim = optim.SGD(self.parameters(), lr=0.01)
        self.crit = nn.MSELoss()

    def forward(self, x):
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x

    def fit(self, x, target):
        x = torch.as_tensor(x)
        target = torch.tensor([target])
        self.optim.zero_grad()
        output = self(x)
        loss = self.crit(output, target)
        loss.backward()
        self.optim.step()

    def predict(self, x):
        x = torch.as_tensor(x)
        with torch.no_grad():
            output = self(x)
        return output.item()


if __name__ == "__main__":
    net = Net()
    qlearn = QLearn(net)
    env = gym.make("CartPole-v0")
    n_episodes = 5000
    for ep in range(n_episodes):
        s = env.reset()
        r = None
        is_terminal = False
        i = 0
        while not (is_terminal):
            i += 1
            a = int(qlearn.act(s))
            s, r, is_terminal, _ = env.step(a)
        print(i)
    s = env.reset()
    env.close()
    index = qlearn.get_index(s, 0.0)
    print("0.0 -> ", net.predict(index))
    index = qlearn.get_index(s, 1.0)
    print("1.0 -> ", net.predict(index))

