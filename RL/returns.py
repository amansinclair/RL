import torch


class DiscountedReturn:
    def __init__(self, discount_rate=0.99):
        """Returns the standard discounted return."""
        self.discount_rate = discount_rate

    def __call__(self, rewards):
        size = len(rewards)
        returns = torch.zeros(size, dtype=torch.float32)
        current_return = 0.0
        for i in reversed(range(size)):
            current_return = rewards[i] + (self.discount_rate * current_return)
            returns[i] = current_return
        return returns


class BaselineAdvantage:
    """Returns advantage and value function target."""

    def __init__(self, discount_rate=0.99):
        self.dr = DiscountedReturn(discount_rate)

    def __call__(self, values, rewards):
        returns = self.dr(rewards)
        advantages = returns - values[:-1]
        return advantages, returns


class LambaReturn:
    def __init__(self, discount_rate=0.99, decay=0.96):
        """Returns td-lambda return. Size of rewards is one smaller than values."""
        self.discount_rate = discount_rate
        self.decay = decay

    def __call__(self, values, rewards):
        size = len(rewards)
        one_step_return = rewards + (self.discount_rate * values[1:])
        returns = torch.zeros(size, dtype=torch.float32)
        current_return = values[-1]
        for i in reversed(range(size)):
            current_return = one_step_return[i] + (
                self.discount_rate * self.decay * (current_return - values[i + 1])
            )
            returns[i] = current_return
        return returns


class GeneralAdvantageEstimation:
    """Returns advantage and td-lambda target. Size of rewards is one smaller than values."""

    def __init__(self, discount_rate=0.99, decay=0.96):
        self.discount_rate = discount_rate
        self.decay = decay

    def __call__(self, values, rewards):
        size = len(rewards)
        td_error = rewards + (self.discount_rate * values[1:]) - values[:-1]
        advantages = torch.zeros(size, dtype=torch.float32)
        current_advantage = 0.0
        for i in reversed(range(size)):
            current_advantage = td_error[i] + (
                self.discount_rate * self.decay * current_advantage
            )
            advantages[i] = current_advantage
        return advantages, advantages + values[:-1]

