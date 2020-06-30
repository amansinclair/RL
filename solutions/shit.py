import gym
import copy
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import time


def plot_res(values, title=""):
    """ Plot the reward curve and histogram of results over time."""
    # Update the window after each episode

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label="score per run")
    ax[0].axhline(195, c="red", ls="--", label="goal")
    ax[0].set_xlabel("Episodes")
    ax[0].set_ylabel("Reward")
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label="trend")
    except:
        print("")

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c="red", label="goal")
    ax[1].set_xlabel("Scores per Last 50 Episodes")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()
    plt.show()


class DQN:
    """ Deep Q Neural Network class. """

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim * 2, action_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def q_learning(
    env,
    model,
    episodes,
    gamma=0.9,
    epsilon=0.05,
    eps_decay=0.99,
    title="DQL",
    verbose=True,
):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    episode_i = 0
    for episode in range(episodes):
        episode_i += 1
        state = env.reset()
        done = False
        total = 0
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            q_values = model.predict(state).tolist()

            if done:
                q_values[action] = reward
                # Update network weights
                model.update(state, q_values)
                break

            q_values_next = model.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            model.update(state, q_values)

            state = next_state

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
    plot_res(final, title)
    return final


env = gym.make("CartPole-v1")
env._max_episode_steps = 1000


# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes = 150
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001


# Get DQN results
simple_dqn = DQN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, episodes, gamma=0.9, epsilon=0.3)

