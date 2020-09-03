from RL.models import A2C, PPO
from RL.networks import Policy, Value
from RL.actor import Actor
from RL.agent import Agent
from RL.critic import Critic
from RL.returns import BaselineAdvantage
from RL.em import EnvManager

env_manager = EnvManager("CartPole-v0")
n_inputs = env_manager.n_inputs
n_outputs = env_manager.n_outputs
policy_net = Policy(n_inputs, n_outputs, sizes=[16])
actor = Actor(policy_net)
value_net = Value(n_inputs, 1, sizes=[16])
adv_func = BaselineAdvantage(discount_rate=0.99)
critic = Critic(value_net, adv_func)
model = PPO(env_manager, actor, critic, actor_lr=0.01, critic_lr=0.1)

n_episodes = 50
for episode in range(n_episodes):
    total_reward = model.train()
    print(f"episode: {episode + 1} score: {total_reward}")
