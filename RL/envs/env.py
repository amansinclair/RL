from collections import namedtuple
import numpy as np


EnvReturn = namedtuple("EnvReturn", "state reward is_terminal")
Transition = namedtuple("Transition", "action states rewards probabilities")


def get_path(solver, env):
    path = []
    game_on = True
    s, r, is_terminal = env.reset()
    while not (is_terminal):
        path.append(s)
        s, r, is_terminal = env.step(np.argmax(solver.Q[s]))
    return path
