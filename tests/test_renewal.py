import numpy as np
import pytest

from spkit.mc import assert_close, mc_estimate
from spkit.poisson import age_residual_at
from spkit.renewal import age_residual_process, renewal_count, renewal_reward, renewal_times


def _ones(n, rng):
    return np.ones(n)


def test_renewal_times_deterministic(rng):
    np.testing.assert_allclose(renewal_times(_ones, 10.5, rng), np.arange(1, 11, dtype=float))
    np.testing.assert_allclose(renewal_times(_ones, 10.0, rng), np.arange(1, 11, dtype=float))
    assert renewal_times(_ones, 0.5, rng).size == 0
    # horizon far beyond the first chunk exercises the chunk growth
    assert renewal_times(_ones, 1000.5, rng).size == 1000


def test_renewal_times_random(rng):
    T = 500.0
    times = renewal_times(lambda n, r: r.uniform(0.0, 2.0, n), T, rng)
    assert np.all(np.diff(times) >= 0) and times[0] > 0 and times[-1] <= T
    rate = [
        renewal_count(renewal_times(lambda n, r: r.uniform(0.0, 2.0, n), T, rng), T) / T
        for _ in range(200)
    ]
    assert_close(mc_estimate(rate), 1.0, name="renewal rate 1/E[X]", k=4.5)
    with pytest.raises(ValueError):
        renewal_times(lambda n, r: np.zeros(n), 5.0, rng)


def test_renewal_count_exact():
    times = np.array([1.0, 2.5, 4.0])
    np.testing.assert_array_equal(renewal_count(times, [0.0, 1.0, 2.0, 4.0, 9.0]), [0, 1, 1, 3, 3])
    assert int(renewal_count(times, 2.5)) == 2


def test_renewal_reward_deterministic(rng):
    total = renewal_reward(_ones, lambda x, r: 2.0 * x, 10.5, rng)
    assert total == 20.0
    assert renewal_reward(_ones, lambda x, r: x, 0.5, rng) == 0.0


def test_renewal_reward_rate(rng):
    T = 2000.0
    # cycle length U(0,2): E[X] = 1, reward X^2: E[R] = 4/3 -> long-run rate 4/3
    rates = [
        renewal_reward(lambda n, r: r.uniform(0.0, 2.0, n), lambda x, r: x**2, T, rng) / T
        for _ in range(100)
    ]
    assert_close(mc_estimate(rates), 4 / 3, name="reward rate", k=4.5)


def test_age_residual_process():
    times = np.array([1.0, 2.0])
    grid = np.array([0.5, 1.0, 1.5, 3.0])
    age, res = age_residual_process(times, grid)
    np.testing.assert_allclose(age, [0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(res, [0.5, 1.0, 0.5, np.inf])
    for t, a, r in zip(grid, age, res, strict=True):
        assert age_residual_at(times, t) == (a, r)
