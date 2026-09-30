import numpy as np
import pytest

from spkit.brownian import (
    brownian_bridge,
    brownian_paths,
    coarsen,
    first_passage_index,
    gbm_paths,
    quadratic_variation,
    running_max,
    scaled_random_walk,
)
from spkit.mc import assert_close, mc_estimate


def test_brownian_paths(rng):
    n, m, T = 20_000, 100, 2.0
    t, B = brownian_paths(n, m, T, rng, x0=1.0)
    assert t.shape == (m + 1,) and B.shape == (n, m + 1)
    np.testing.assert_allclose(t[[0, -1]], [0.0, T])
    assert np.all(B[:, 0] == 1.0)
    inc = np.diff(B, axis=1)
    assert_close(mc_estimate(B[:, -1]), 1.0, name="mean")
    assert abs(inc.var(ddof=1) - T / m) < 4 * (T / m) * np.sqrt(2 / inc.size)
    assert abs((B[:, -1] - 1.0).var(ddof=1) - T) < 4 * T * np.sqrt(2 / n)


def test_coarsen(rng):
    t, B = brownian_paths(3, 12, 1.0, rng)
    tc, Bc = coarsen(t, B, 4)
    np.testing.assert_array_equal(tc, t[::4])
    np.testing.assert_array_equal(Bc, B[:, ::4])
    assert Bc.shape == (3, 4)
    with pytest.raises(ValueError):
        coarsen(t, B, 5)


def test_brownian_bridge(rng):
    n, m, T = 20_000, 40, 2.0
    t, X = brownian_bridge(n, m, T, rng, a=1.0, b=-1.0)
    np.testing.assert_allclose(X[:, 0], 1.0)
    np.testing.assert_allclose(X[:, -1], -1.0)
    k = m // 2  # t = T / 2: mean = (a + b) / 2, var = t (T - t) / T
    assert_close(mc_estimate(X[:, k]), 0.0, name="bridge mean")
    var_exact = t[k] * (T - t[k]) / T
    assert abs(X[:, k].var(ddof=1) - var_exact) < 4 * var_exact * np.sqrt(2 / n)


def test_gbm_paths(rng):
    S0, mu, sigma, T = 100.0, 0.05, 0.3, 1.0
    t, S = gbm_paths(S0, mu, sigma, 20_000, 50, T, rng)
    assert np.all(S[:, 0] == S0) and np.all(S > 0)
    assert_close(mc_estimate(S[:, -1]), S0 * np.exp(mu * T), name="E[S_T]")
    assert_close(mc_estimate(np.log(S[:, -1] / S0)), (mu - 0.5 * sigma**2) * T, name="E[log S_T]")


def test_scaled_random_walk(rng):
    n, m, T = 20_000, 64, 4.0
    t, W = scaled_random_walk(n, m, T, rng)
    assert t.shape == (m + 1,) and W.shape == (n, m + 1)
    assert np.all(W[:, 0] == 0.0)
    np.testing.assert_allclose(np.abs(np.diff(W, axis=1)), np.sqrt(T / m))
    assert abs(W[:, -1].var(ddof=1) - T) < 4 * T * np.sqrt(2 / n)


def test_running_max():
    X = np.array([[0.0, 1.0, 0.5, 2.0], [1.0, -1.0, -2.0, 0.0]])
    np.testing.assert_array_equal(running_max(X), [[0, 1, 1, 2], [1, 1, 1, 1]])


def test_first_passage_index():
    X = np.array([[0.0, 0.5, 1.2, 0.3], [0.0, -0.5, -1.5, 0.0], [2.0, 1.5, 0.9, 1.0]])
    np.testing.assert_array_equal(first_passage_index(X, 1.0), [2, -1, 2])
    np.testing.assert_array_equal(first_passage_index(X, -1.0), [-1, 2, -1])
    np.testing.assert_array_equal(first_passage_index(X, 0.0), [0, 0, -1])
    assert first_passage_index(X, 1.0).dtype.kind == "i"


def test_quadratic_variation(rng):
    X = np.array([[0.0, 1.0, 3.0, 2.0]])
    np.testing.assert_allclose(quadratic_variation(X), [[0.0, 1.0, 5.0, 6.0]])
    _, B = brownian_paths(2_000, 1_000, 1.5, rng)
    qv = quadratic_variation(B)
    assert qv.shape == B.shape
    assert_close(mc_estimate(qv[:, -1]), 1.5, name="[B]_T")
