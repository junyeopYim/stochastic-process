import numpy as np
import pytest

from spkit.mc import assert_close, mc_estimate, mc_proportion
from spkit.poisson import (
    age_residual_at,
    compound_poisson,
    count_in_bins,
    nhpp_thinning,
    poisson_arrivals,
    poisson_arrivals_conditional,
    thin,
)


def test_poisson_arrivals_basic(rng):
    rate, T = 3.0, 2000.0
    times = poisson_arrivals(rate, T, rng)
    assert np.all(np.diff(times) > 0)
    assert times[0] > 0 and times[-1] <= T
    counts = count_in_bins(times, T, 2000)
    assert counts.sum() == times.size
    assert_close(mc_estimate(counts), rate, name="bin mean")
    # Poisson: variance equals mean
    assert abs(counts.var(ddof=1) - rate) < 4 * rate * np.sqrt(2 / counts.size)
    assert poisson_arrivals(0.0, 5.0, rng).size == 0
    assert poisson_arrivals(2.0, 0.0, rng).size == 0
    with pytest.raises(ValueError):
        poisson_arrivals(-1.0, 1.0, rng)


def test_poisson_arrivals_conditional(rng):
    T, n = 5.0, 20_000
    times = poisson_arrivals_conditional(n, T, rng)
    assert times.shape == (n,)
    assert np.all(np.diff(times) >= 0)
    assert times[0] > 0 and times[-1] <= T
    assert_close(mc_estimate(times), T / 2, name="uniform mean")


def test_nhpp_thinning(rng):
    T = 2 * np.pi
    expected = 2 * T  # integral of 2 + 2 sin(t) over one period

    def rate_fn(t):
        return 2.0 + 2.0 * np.sin(t)

    counts = [nhpp_thinning(rate_fn, 4.0, T, rng).size for _ in range(500)]
    assert_close(mc_estimate(counts), expected, name="NHPP count")
    times = nhpp_thinning(rate_fn, 4.0, T, rng)
    assert np.all(np.diff(times) > 0) and times[-1] <= T
    with pytest.raises(ValueError):
        nhpp_thinning(rate_fn, 1.0, T, rng)
    assert nhpp_thinning(rate_fn, 4.0, 0.0, rng).size == 0


def test_count_in_bins_exact():
    times = np.array([0.1, 0.4, 1.2, 2.9, 3.0])
    np.testing.assert_array_equal(count_in_bins(times, 3.0, 3), [2, 1, 2])


def test_thin(rng):
    times = poisson_arrivals(5.0, 4000.0, rng)
    kept, removed = thin(times, 0.3, rng)
    np.testing.assert_array_equal(np.sort(np.concatenate([kept, removed])), times)
    assert_close(mc_proportion(kept.size, times.size), 0.3, name="kept fraction")


def test_compound_poisson(rng):
    rate, T = 3.0, 2.0
    X = compound_poisson(rate, T, lambda k, r: r.normal(1.0, 1.0, k), rng, n_paths=20_000)
    assert X.shape == (20_000,)
    assert_close(mc_estimate(X), rate * T * 1.0, name="mean")
    var = X.var(ddof=1)
    exact_var = rate * T * 2.0  # lambda T E[J^2]
    assert abs(var - exact_var) < 4 * exact_var * np.sqrt(2 / X.size) * 1.5
    np.testing.assert_array_equal(
        compound_poisson(0.0, T, lambda k, r: r.normal(size=k), rng, 3), 0.0
    )


def test_age_residual_at():
    times = np.array([1.0, 2.0, 3.0])
    assert age_residual_at(times, 2.5) == (0.5, 0.5)
    assert age_residual_at(times, 0.5) == (0.5, 0.5)
    assert age_residual_at(times, 2.0) == (0.0, 1.0)
    age, res = age_residual_at(times, 5.0)
    assert age == 2.0 and res == np.inf
