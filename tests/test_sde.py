import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from spkit.mc import assert_close, mc_estimate
from spkit.sde import (
    black_scholes_call,
    black_scholes_put,
    euler_maruyama,
    mc_option_price,
    milstein,
    ou_exact,
)


def test_euler_maruyama_reduces_to_brownian_motion(rng):
    dW = rng.normal(0.0, np.sqrt(0.01), (7, 100))
    t, X = euler_maruyama(lambda t, x: 0.0 * x, lambda t, x: 2.0 + 0 * x, 1.0, 1.0, 100, rng, dW=dW)
    assert t.shape == (101,) and X.shape == (7, 101)
    np.testing.assert_allclose(X[:, 1:], 1.0 + 2.0 * np.cumsum(dW, axis=1))
    with pytest.raises(ValueError):
        euler_maruyama(lambda t, x: x, lambda t, x: x, 1.0, 1.0, 50, rng, dW=dW)
    # deterministic ODE: dX = X dt -> e^T with O(dt) error
    _, Y = euler_maruyama(lambda t, x: x, lambda t, x: 0 * x, 1.0, 1.0, 4000, rng, n_paths=2)
    np.testing.assert_allclose(Y[:, -1], np.e, rtol=2e-3)
    assert Y.shape == (2, 4001)


def test_milstein_correction_and_strong_error(rng):
    mu, sigma, x0, T, n = 0.1, 0.5, 1.0, 1.0, 64

    def a(t, x):
        return mu * x

    def b(t, x):
        return sigma * x

    def db(t, x):
        return sigma * np.ones_like(x)

    dt = T / n
    dW = rng.normal(0.0, np.sqrt(dt), (5_000, n))
    # one step: Milstein - Euler == 0.5 * b * b' * (dW^2 - dt), computed by hand
    _, Xe = euler_maruyama(a, b, x0, dt, 1, rng, dW=dW[:, :1])
    _, Xm = milstein(a, b, db, x0, dt, 1, rng, dW=dW[:, :1])
    np.testing.assert_allclose(Xm[:, 1] - Xe[:, 1], 0.5 * sigma * x0 * sigma * (dW[:, 0] ** 2 - dt))
    # full horizon against the exact GBM solution driven by the same Brownian path
    _, Xe = euler_maruyama(a, b, x0, T, n, rng, dW=dW)
    _, Xm = milstein(a, b, db, x0, T, n, rng, dW=dW)
    exact = x0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW.sum(axis=1))
    err_e = np.abs(Xe[:, -1] - exact).mean()
    err_m = np.abs(Xm[:, -1] - exact).mean()
    assert err_m < 0.5 * err_e
    assert Xm.shape == (5_000, n + 1)


def test_ou_exact_transition_moments(rng):
    theta, mu, sigma, x0, dt = 1.7, 0.5, 0.8, 3.0, 0.3
    n = 40_000
    t, X = ou_exact(theta, mu, sigma, x0, dt, 1, rng, n_paths=n)
    assert t.shape == (2,) and X.shape == (n, 2)
    assert np.all(X[:, 0] == x0)
    mean_exact = mu + (x0 - mu) * np.exp(-theta * dt)
    var_exact = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    # independent: Var of the stochastic integral int_0^dt sigma e^{-theta (dt - s)} dW_s
    var_ref = quad(lambda s: sigma**2 * np.exp(-2 * theta * (dt - s)), 0.0, dt)[0]
    np.testing.assert_allclose(var_exact, var_ref)
    assert_close(mc_estimate(X[:, 1]), mean_exact, name="OU mean")
    assert abs(X[:, 1].var(ddof=1) - var_exact) < 4 * var_exact * np.sqrt(2 / n)
    # long horizon: stationary variance sigma^2 / (2 theta)
    _, Y = ou_exact(theta, mu, sigma, x0, 10.0, 200, rng, n_paths=n)
    assert_close(mc_estimate(Y[:, -1]), mu, name="OU stationary mean")
    assert abs(Y[:, -1].var(ddof=1) - sigma**2 / (2 * theta)) < 4 * sigma**2 / (
        2 * theta
    ) * np.sqrt(2 / n)
    # theta = 0 degenerates to Brownian motion with variance sigma^2 dt
    _, Z = ou_exact(0.0, mu, sigma, x0, dt, 1, rng, n_paths=n)
    assert abs(Z[:, 1].var(ddof=1) - sigma**2 * dt) < 4 * sigma**2 * dt * np.sqrt(2 / n)


def _bs_by_quadrature(S0, K, r, sigma, T, kind):
    m = np.log(S0) + (r - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)

    def integrand(x):
        S = np.exp(x)
        pay = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
        return pay * norm.pdf(x, m, s)

    val, _ = quad(integrand, m - 12 * s, m + 12 * s, points=[np.log(K)], limit=200)
    return np.exp(-r * T) * val


@pytest.mark.parametrize(
    ("S0", "K", "r", "sigma", "T"),
    [(100, 100, 0.05, 0.2, 1.0), (50, 60, 0.01, 0.4, 0.5), (120, 90, 0.03, 0.25, 2.0)],
)
def test_black_scholes_against_quadrature(S0, K, r, sigma, T):
    call = black_scholes_call(S0, K, r, sigma, T)
    put = black_scholes_put(S0, K, r, sigma, T)
    np.testing.assert_allclose(call, _bs_by_quadrature(S0, K, r, sigma, T, "call"), rtol=1e-8)
    np.testing.assert_allclose(put, _bs_by_quadrature(S0, K, r, sigma, T, "put"), rtol=1e-8)
    np.testing.assert_allclose(call - put, S0 - K * np.exp(-r * T))  # put-call parity


def test_black_scholes_textbook_value_and_edge_cases():
    np.testing.assert_allclose(black_scholes_call(100, 100, 0.05, 0.2, 1.0), 10.4506, atol=5e-5)
    np.testing.assert_allclose(black_scholes_put(100, 100, 0.05, 0.2, 1.0), 5.5735, atol=5e-5)
    assert black_scholes_call(100, 90, 0.05, 0.2, 0.0) == 10.0
    assert black_scholes_put(100, 90, 0.05, 0.2, 0.0) == 0.0
    np.testing.assert_allclose(
        black_scholes_call(100, 90, 0.05, 0.0, 1.0), 100 - 90 * np.exp(-0.05)
    )


def test_mc_option_price(rng):
    S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 1.0

    def call(s):
        return np.maximum(s - K, 0.0)

    def put(s):
        return np.maximum(K - s, 0.0)

    res = mc_option_price(call, S0, r, sigma, T, 200_000, rng)
    assert res.n == 200_000
    assert_close(res, black_scholes_call(S0, K, r, sigma, T), name="MC call")
    anti = mc_option_price(put, S0, r, sigma, T, 200_000, rng, antithetic=True)
    assert anti.n == 100_000
    assert_close(anti, black_scholes_put(S0, K, r, sigma, T), name="MC put (antithetic)")
    assert anti.se < res.se
