import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ..utils import Norm, StdNorm, MeanNorm


class RollOut:
    def __init__(self):
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []
        self.intr_rewards = []
        self.is_not_dones = []

    def __len__(self):
        return len(self.obs)


class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class Value(nn.Module):
    def __init__(self, n_inputs, n_outputs=1, size=32):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, n_inputs, n_outputs, size=32):
        super().__init__()
        self.policy = Policy(n_inputs, n_outputs, size)

    def act(self, obs):
        with torch.no_grad():
            prob = self.policy(obs)
            m = Categorical(prob)
        return m.sample().item()

    def get_probs(self, obs, actions):
        probs = self.policy(obs)
        return torch.gather(probs, 1, actions).squeeze()

    def get_entropy(self, obs):
        probs = self.policy(obs)
        m = Categorical(probs)
        return m.entropy()


class Critic(nn.Module):
    def __init__(self, n_inputs, size=32):
        super().__init__()
        self.value = Value(n_inputs, 1, size)

    def get_values(self, obs):
        return self.value(obs).view(-1)


class Rewarder(nn.Module):
    def __init__(self, n_inputs, n_outputs=1, size=32):
        super().__init__()
        self.target = Value(n_inputs, n_outputs, size)
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        self.pred = Value(n_inputs, n_outputs, size)

    def forward(self, obs):
        targets = self.target(obs)
        preds = self.pred(obs)
        return ((preds - targets) ** 2).squeeze()

    def get_reward(self, obs):
        reward = self(obs).detach().sum().item()
        return reward


class EmptyRewarder:
    def __call__(self, obs):
        return obs

    def get_reward(self, obs):
        return obs


class Model:
    def __init__(
        self,
        actor,
        critic,
        rewarder,
        policy_lr=0.01,
        critic_lr=0.1,
        rewarder_lr=0.1,
        rollout_length=10,
        n_rollouts=5,
        discount_rate_e=0.99,
        discount_rate_i=0.99,
        gae_decay=0.95,
        ppo_clip=0.2,
        ppo_epochs=5,
        ent_coef=0.001,
        norm_obs=True,
        norm_ir=True,
    ):
        self.actor = actor
        self.critic = critic
        self.rewarder = rewarder
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # self.rewarder_opt = optim.Adam(self.rewarder.parameters(), lr=rewarder_lr)
        self.rollout_length = rollout_length
        self.n_rollouts = n_rollouts
        self.discount_rate_e = discount_rate_e
        self.discount_rate_i = discount_rate_i
        self.gae_decay = gae_decay
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ent_coef = ent_coef
        self.norm_obs = norm_obs
        if norm_obs:
            self.obs_normer = Norm()
        self.norm_ir = norm_ir
        if norm_ir:
            self.ir_normer = MeanNorm()
        self.reset()

    def reset(self):
        self.rollouts = []
        self.current_rollout = RollOut()

    def add_new_rollout(self):
        self.rollouts.append(self.current_rollout)
        self.current_rollout = RollOut()

    def act(self, obs, reward=None, is_done=False):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = None
        if reward:
            self.current_rollout.rewards.append(reward)
            self.current_rollout.next_obs.append(obs)
            self.current_rollout.is_not_dones.append(not (is_done))
            if self.norm_obs and self.obs_normer:
                nobs = self.obs_normer.normalize(obs)
            else:
                nobs = obs
            intr_reward = self.rewarder.get_reward(nobs)
            self.current_rollout.intr_rewards.append(intr_reward)
        if len(self.current_rollout) == self.rollout_length or is_done:
            if self.norm_ir:
                self.ir_normer.update(torch.stack(self.current_rollout.intr_rewards))
                # self.ir_normer.update(torch.tensor(self.current_rollout.intr_rewards))
            self.add_new_rollout()
            if len(self.rollouts) == self.n_rollouts:
                self.update()
                self.reset()
        if not is_done:
            action = self.actor.act(obs)
            self.current_rollout.obs.append(obs)
            self.current_rollout.actions.append(action)
        return action

    def update(self):
        all_obs = self.stack_obs()
        if self.norm_obs and self.obs_normer:
            norm_obs = self.obs_normer.normalize(all_obs)
        else:
            norm_obs = all_obs
        Aes, Te = self.get_advantages_targets()
        Ve = self.critic.get_values(all_obs)
        Veloss = ((Te - Ve) ** 2).mean()
        # Viloss = self.rewarder(norm_obs).mean()
        self.critic_opt.zero_grad()
        # self.rewarder_opt.zero_grad()
        Veloss.backward()
        # Viloss.backward()
        self.critic_opt.step()
        # self.rewarder_opt.step()
        actions = self.stack_actions()
        old_probs = self.actor.get_probs(all_obs, actions).detach()
        for epoch in range(self.ppo_epochs):
            self.actor_opt.zero_grad()
            probs = self.actor.get_probs(all_obs, actions)
            entropy = 0  # self.actor.get_entropy(all_obs).mean()
            prob_ratio = probs / old_probs
            clipped_prob_ratio = torch.clamp(
                prob_ratio, min=1.0 - self.ppo_clip, max=1.0 + self.ppo_clip
            )
            Pscore = -(torch.min(Aes * prob_ratio, Aes * clipped_prob_ratio)).mean() - (
                self.ent_coef * entropy
            )
            Pscore.backward()
            self.actor_opt.step()
        if self.norm_obs:
            self.obs_normer.update(all_obs)

    def get_advantages_targets(self):
        Aes = []
        Tes = []
        with torch.no_grad():
            for rollout in self.rollouts:
                Ve = self.critic.get_values(torch.stack(rollout.obs))
                Ve_next = self.critic.get_values(torch.stack(rollout.next_obs))
                mask = torch.tensor(rollout.is_not_dones, dtype=torch.float32)
                Ve_next = Ve_next * mask
                # Re = torch.tensor(rollout.intr_rewards)
                Re = torch.stack(rollout.intr_rewards)
                if self.norm_ir and self.ir_normer:
                    Re = self.ir_normer.normalize(Re)
                # Re += torch.tensor(rollout.rewards)
                td_error_e = Re + (self.discount_rate_e * Ve_next) - Ve
                size = len(Ve)
                Ae = torch.zeros(size)
                a_e = 0
                for i in reversed(range(size)):
                    a_e = td_error_e[i] + (self.discount_rate_e * self.gae_decay * a_e)
                    Ae[i] = a_e
                Aes.append(Ae)
                Tes.append(Ae + Ve)
        Aes = torch.cat(Aes)
        Tes = torch.cat(Tes)
        return Aes, Tes

    def stack_obs(self):
        obs = []
        for rollout in self.rollouts:
            obs += rollout.obs
        return torch.stack(obs)

    def stack_actions(self):
        acts = []
        for rollout in self.rollouts:
            acts += rollout.actions
        return torch.tensor(acts).view(-1, 1)
