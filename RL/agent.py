import torch


class Agent:
    """Agent that returns after a fixed number of steps or at episode completion when mc is True."""

    def __init__(self, env_manager, actor, mc=False):
        self.env_manager = env_manager
        self.actor = actor
        self.env = None
        self.mc = mc
        self.current_index = 0

    def reset_buffer(self, n_steps):
        self.n_steps = n_steps
        self.obs = torch.zeros((n_steps + 1, self.env_manager.n_inputs))
        self.actions = torch.zeros(n_steps, dtype=torch.long)
        self.rewards = torch.zeros(n_steps)
        self.is_dones = torch.zeros(n_steps, dtype=torch.bool)
        self.current_index = 0

    def run(self, n_steps=None):
        if self.mc:
            steps_completed = self.run_episode()
        else:
            steps_completed = self.run_n_steps(n_steps)
        return steps_completed

    def run_episode(self):
        if not self.env:
            self.env = self.env_manager.get_env()
        obs, reward, is_done = self.reset_env()
        self.reset_buffer(self.env_manager.max_steps)
        while not is_done:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.actor.act(obs)
            self.store(obs, action, reward, is_done)
            self.current_index += 1
            obs, reward, is_done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32)
        self.store(obs, reward=reward, is_done=is_done)
        return self.current_index

    def run_n_steps(self, n_steps):
        if self.env:
            obs = self.obs[-1].tolist()
            self.reward = None
            self.is_done = False
        else:
            self.env = self.env_manager.get_env()
            obs, reward, is_done = self.reset_env()
        self.reset_buffer(n_steps)
        while self.current_index < self.n_steps:
            obs = torch.tensor(obs, dtype=torch.float32)
            action = self.actor.act(obs)
            self.store(obs, action, reward, is_done)
            self.current_index += 1
            obs, reward, is_done, info = self.env.step(action)
            if is_done:
                obs = torch.tensor(obs, dtype=torch.float32)
                self.store(obs, reward=reward, is_done=is_done)
                obs, reward, is_done = self.reset_env()
        return self.current_index

    def store(self, obs, action=None, reward=None, is_done=None):
        self.obs[self.current_index] = obs
        if action:
            self.actions[self.current_index] = action
        if reward:
            self.rewards[self.current_index - 1] = reward
        if is_done:
            self.is_dones[self.current_index - 1] = reward

    def reset_env(self):
        obs = self.env.reset()
        reward = None
        is_done = False
        return obs, reward, is_done

