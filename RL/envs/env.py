from collections import namedtuple


EnvReturn = namedtuple("EnvReturn", "state reward is_terminal")
Transition = namedtuple("Transition", "action states rewards probabilities")
