from RL.utils import EnvManager

env_m = EnvManager("CartPole-v0")

print(env_m.n_inputs)

with env_m as env_mm:
    print(env_mm.n_inputs)
