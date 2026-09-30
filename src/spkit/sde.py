"""확률미분방정식(SDE) 수치해법과 옵션 가격 도구.

Euler-Maruyama and Milstein schemes for scalar SDEs (vectorised over paths,
optionally driven by a supplied Brownian increment array for strong-convergence
experiments), exact Ornstein-Uhlenbeck sampling, Black-Scholes formulas and a
Monte Carlo European option pricer.
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

from .mc import MCResult, mc_estimate

Coef = Callable[[float, np.ndarray], np.ndarray]


def _prepare(
    x0: float | np.ndarray,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    n_paths: int,
    dW: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dt = T / n_steps
    if dW is None:
        dW = rng.normal(0.0, np.sqrt(dt), (n_paths, n_steps))
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim != 2 or dW.shape[1] != n_steps:
            msg = f"dW must have shape (n_paths, {n_steps}), got {dW.shape}"
            raise ValueError(msg)
        n_paths = dW.shape[0]
    t = np.linspace(0.0, T, n_steps + 1)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    return t, X, dW, dt


def euler_maruyama(
    drift: Coef,
    diffusion: Coef,
    x0: float | np.ndarray,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    n_paths: int = 1,
    dW: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """오일러-마루야마 방법으로 dX = a(t, X) dt + b(t, X) dW 를 푼다.

    Parameters
    ----------
    drift, diffusion : callable
        ``a(t, x)`` and ``b(t, x)``, vectorised over the path array ``x``.
    x0 : float or numpy.ndarray
        Initial value (scalar or one value per path).
    T : float
        Horizon.
    n_steps : int
        Number of time steps.
    rng : numpy.random.Generator
        Random source (unused when ``dW`` is given).
    n_paths : int, default 1
        Number of paths (overridden by ``dW.shape[0]`` when ``dW`` is given).
    dW : numpy.ndarray, optional
        Brownian increments of shape ``(n_paths, n_steps)``; drawn as
        ``N(0, dt)`` if omitted.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, X)`` with ``X`` of shape ``(n_paths, n_steps + 1)``.
    """
    t, X, dW, dt = _prepare(x0, T, n_steps, rng, n_paths, dW)
    for k in range(n_steps):
        x = X[:, k]
        X[:, k + 1] = x + drift(t[k], x) * dt + diffusion(t[k], x) * dW[:, k]
    return t, X


def milstein(
    drift: Coef,
    diffusion: Coef,
    diffusion_dx: Coef,
    x0: float | np.ndarray,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    n_paths: int = 1,
    dW: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """스칼라 밀스타인 방법: 오일러 단계에 ``0.5 b b' (ΔW² - Δt)`` 를 더한다.

    Parameters
    ----------
    drift, diffusion, diffusion_dx : callable
        ``a(t, x)``, ``b(t, x)`` and ``∂b/∂x (t, x)``.
    x0, T, n_steps, rng, n_paths, dW
        As in :func:`euler_maruyama`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, X)`` with ``X`` of shape ``(n_paths, n_steps + 1)``.
    """
    t, X, dW, dt = _prepare(x0, T, n_steps, rng, n_paths, dW)
    for k in range(n_steps):
        x = X[:, k]
        b = diffusion(t[k], x)
        dw = dW[:, k]
        X[:, k + 1] = (
            x + drift(t[k], x) * dt + b * dw + 0.5 * b * diffusion_dx(t[k], x) * (dw**2 - dt)
        )
    return t, X


def ou_exact(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
    n_paths: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """OU 과정 dX = θ(μ - X) dt + σ dW 의 정확한 가우스 전이 표본.

    ``X_{t+Δ} | X_t ~ N(μ + (X_t - μ) e^{-θΔ}, σ² (1 - e^{-2θΔ}) / (2θ))``;
    for ``θ = 0`` the variance is ``σ² Δ``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, X)`` with ``X`` of shape ``(n_paths, n_steps + 1)``.
    """
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    decay = np.exp(-theta * dt)
    var = (
        sigma**2 * dt
        if theta == 0
        else sigma**2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
    )
    sd = np.sqrt(var)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        X[:, k + 1] = mu + (X[:, k] - mu) * decay + sd * Z[:, k]
    return t, X


def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    v = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / v
    return d1, d1 - v


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """블랙-숄즈 유럽형 콜옵션 가격 ``S0 N(d1) - K e^{-rT} N(d2)``."""
    if T <= 0 or sigma <= 0:
        return float(max(S0 - K * np.exp(-r * T), 0.0))
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """블랙-숄즈 유럽형 풋옵션 가격 ``K e^{-rT} N(-d2) - S0 N(-d1)``."""
    if T <= 0 or sigma <= 0:
        return float(max(K * np.exp(-r * T) - S0, 0.0))
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def mc_option_price(
    payoff: Callable[[np.ndarray], np.ndarray],
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> MCResult:
    """유럽형 옵션의 몬테카를로 가격 (S_T 의 정확한 로그정규 표본).

    Parameters
    ----------
    payoff : callable
        Vectorised payoff ``h(S_T)``.
    S0, r, sigma, T : float
        Spot, risk-free rate, volatility and maturity.
    n_paths : int
        Number of terminal draws. With ``antithetic=True`` the draws come in
        ``n_paths // 2`` pairs ``(Z, -Z)``; each pair average is one sample.
    rng : numpy.random.Generator
        Random source.
    antithetic : bool, default False
        Use antithetic variates.

    Returns
    -------
    MCResult
        Discounted mean payoff ``e^{-rT} E[h(S_T)]`` with its standard error.
    """

    def terminal(z: np.ndarray) -> np.ndarray:
        return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    disc = np.exp(-r * T)
    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        samples = 0.5 * disc * (payoff(terminal(Z)) + payoff(terminal(-Z)))
    else:
        samples = disc * payoff(terminal(rng.standard_normal(n_paths)))
    return mc_estimate(samples)
