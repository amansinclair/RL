import torch
import torch.optim as optim
from .agent import Agent


class A2C:
    """A2C (episodic)."""

    def __init__(self, env_manager, actor, critic, actor_lr=0.01, critic_lr=0.01):
        self.env_manager = env_manager
        self.agent = Agent(env_manager, actor, mc=True)
        self.critic = critic
        self.actor_opt = optim.Adam(self.agent.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train(self):
        n_steps = self.agent.run()
        obs = self.agent.obs[: n_steps + 1]
        actions = self.agent.actions[:n_steps]
        rewards = self.agent.rewards[:n_steps]
        is_dones = self.agent.is_dones[:n_steps]
        advantages, targets = self.critic.get_advantage(obs, rewards, is_dones)
        values = self.critic.get_values(obs[:-1])
        log_probs = self.agent.actor.get_log_probs(obs[:-1], actions)
        ploss = -((advantages * log_probs).mean())
        vloss = ((values - targets) ** 2).mean()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        vloss.backward()
        ploss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        return rewards.sum()


class PPO:
    """PPO (episodic)."""

    def __init__(
        self,
        env_manager,
        actor,
        critic,
        actor_lr=0.01,
        critic_lr=0.01,
        clip=0.2,
        n_epochs=5,
    ):
        self.env_manager = env_manager
        self.agent = Agent(env_manager, actor, mc=True)
        self.critic = critic
        self.clip = clip
        self.n_epochs = n_epochs
        self.actor_opt = optim.Adam(self.agent.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train(self):
        n_steps = self.agent.run()
        obs = self.agent.obs[: n_steps + 1]
        actions = self.agent.actions[:n_steps]
        rewards = self.agent.rewards[:n_steps]
        is_dones = self.agent.is_dones[:n_steps]
        advantages, targets = self.critic.get_advantage(obs, rewards, is_dones)
        old_probs = self.agent.actor.get_probs(obs[:-1], actions).detach()
        for epoch in range(self.n_epochs):
            probs = self.agent.actor.get_probs(obs[:-1], actions)
            prob_ratio = probs / old_probs
            clipped_prob_ratio = torch.clamp(
                prob_ratio, min=1.0 - self.clip, max=1.0 + self.clip
            )
            ploss = -(
                torch.min(advantages * prob_ratio, advantages * clipped_prob_ratio)
            ).mean()
            self.actor_opt.zero_grad()
            ploss.backward()
            self.actor_opt.step()
        self.critic_opt.zero_grad()
        values = self.critic.get_values(obs[:-1])
        vloss = ((values - targets) ** 2).mean()
        vloss.backward()
        self.critic_opt.step()
        return rewards.sum()

