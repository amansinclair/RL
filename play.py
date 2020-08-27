import gym

env = gym.make("MountainCar-v0")
env._max_episode_steps = 400
obs = env.reset()
is_done = False
action = 0
while not is_done:
    env.render()
    obs, reward, is_done, info = env.step(action)
    print(obs[0])
    if obs[1] > 0:
        action = 1
    else:
        action = 0
