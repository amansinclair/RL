import torch
import numpy as np
from RL.solvers import DiscountedReturn, LambaReturn, GeneralAdvantageEstimation
import scipy.signal


def approx_equal(a, b, perc=0.01):
    result = (a - b) / b
    return np.all(result < perc)


def discount_cumsum(x, discount):
    """taken from openai spinning up."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def test_discounted_return_no_discount():
    dr = DiscountedReturn(discount_rate=1)
    rewards = np.ones(100, dtype="float32")
    returns = np.flip(np.cumsum(rewards)).copy()
    returns_to_check = dr(torch.from_numpy(rewards)).numpy()
    assert np.all(returns == returns_to_check)


def test_discounted_return_general():
    size = 100
    discount_rate = 0.99
    returns = []
    for i in range(size):
        total = 0.0
        for j in range(size - i):
            total += discount_rate ** j
        returns.append(total)
    returns = np.array(returns)
    dr = DiscountedReturn(discount_rate=discount_rate)
    rewards = np.ones(size, dtype="float32")
    return_to_check = dr(torch.from_numpy(rewards)).numpy()
    assert approx_equal(return_to_check, returns)


def test_discounted_return_cmp():
    size = 100
    discount_rate = 0.95
    rewards = np.arange(size, dtype="float32")
    dr = DiscountedReturn(discount_rate=discount_rate)
    return_to_check = dr(torch.from_numpy(rewards)).numpy()
    returns = discount_cumsum(rewards, discount_rate)
    assert approx_equal(return_to_check, returns)


def nstep_return(nstep, rewards, values, discount_rate):
    """nstep==1 corresponds to TD(0)."""
    size = len(rewards)
    returns = torch.zeros(size)
    for i in range(size):
        n_rewards = min(nstep, size - i)
        if i + nstep <= size:
            v = values[i + nstep] * (discount_rate ** (nstep))
        else:
            v = 0.0
        for power, reward in enumerate(rewards[i : i + n_rewards]):
            v += reward * (discount_rate ** power)
        returns[i] = v
    return returns


def test_lambda_return():
    size = 10
    values = torch.from_numpy(np.arange(size + 1, dtype="float32"))
    rewards = torch.ones(size)
    discount_rate = 0.99
    decay = 0.96
    returns = torch.zeros(size)
    for nstep in range(1, size + 1):
        nstep_returns = nstep_return(nstep, rewards, values, discount_rate)
        mask = torch.zeros(size)
        if size + 1 - nstep > 0:
            mask[: size - nstep] = 1.0 - decay
        mask[size - nstep] = 1
        decay_factor = decay ** (nstep - 1)
        returns += nstep_returns * mask * decay_factor
    lr = LambaReturn(discount_rate, decay)
    result_to_check = lr(values, rewards)
    assert approx_equal(result_to_check.numpy(), returns.numpy())


def test_gae_lambda_return():
    size = 10
    values = torch.from_numpy(np.arange(size + 1, dtype="float32"))
    rewards = torch.ones(size)
    discount_rate = 0.99
    decay = 0.96
    returns = torch.zeros(size)
    for nstep in range(1, size + 1):
        nstep_returns = nstep_return(nstep, rewards, values, discount_rate)
        mask = torch.zeros(size)
        if size + 1 - nstep > 0:
            mask[: size - nstep] = 1.0 - decay
        mask[size - nstep] = 1
        decay_factor = decay ** (nstep - 1)
        returns += nstep_returns * mask * decay_factor
    gae = GeneralAdvantageEstimation(discount_rate, decay)
    A, result_to_check = gae(values, rewards)
    assert approx_equal(result_to_check.numpy(), returns.numpy())


def test_gae_lambda_advantage():
    size = 20
    discount_rate = 0.99
    decay = 0.96
    rewards = np.ones(size + 1, dtype="float32")
    t_rewards = torch.from_numpy(rewards[:-1])
    values = np.arange(size + 1, dtype="float32")
    t_values = torch.from_numpy(values)
    deltas = rewards[:-1] + discount_rate * values[1:] - values[:-1]
    advantage = discount_cumsum(deltas, discount_rate * decay)
    gae = GeneralAdvantageEstimation(discount_rate, decay)
    value_to_check = gae(t_values, t_rewards)[0].numpy()
    assert approx_equal(value_to_check, advantage)


test_discounted_return_no_discount()
test_discounted_return_general()
test_discounted_return_cmp()
test_lambda_return()
test_gae_lambda_advantage()
test_gae_lambda_return()

