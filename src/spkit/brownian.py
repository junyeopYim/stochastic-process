"""브라운 운동과 관련 경로 도구.

Brownian paths on a regular grid, bridges, geometric Brownian motion, the
Donsker scaled random walk and path functionals (running maximum, first
passage, quadratic variation). Arrays are ``(n_paths, n_steps + 1)``.
"""

import numpy as np


def brownian_paths(
    n_paths: int,
    n_steps: int,
    T: float,
    rng: np.random.Generator,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """[0, T] 격자 위의 표준 브라운 운동 경로들.

    Parameters
    ----------
    n_paths, n_steps : int
        Number of paths and time steps.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.
    x0 : float, default 0.0
        Starting value.

    Returns
    -------
    tuple of numpy.ndarray
        ``t`` of shape ``(n_steps + 1,)`` and ``B`` of shape ``(n_paths, n_steps + 1)``
        with independent ``N(0, dt)`` increments.
    """
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    B = np.empty((n_paths, n_steps + 1))
    B[:, 0] = x0
    B[:, 1:] = x0 + np.cumsum(rng.normal(0.0, np.sqrt(dt), (n_paths, n_steps)), axis=1)
    return t, B


def coarsen(t: np.ndarray, X: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """같은 경로를 factor 배 성긴 격자에서 본다.

    Parameters
    ----------
    t : numpy.ndarray, shape (n_steps + 1,)
        Time grid.
    X : numpy.ndarray, shape (..., n_steps + 1)
        Paths.
    factor : int
        Keep every ``factor``-th point; ``n_steps % factor`` must be 0.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t[::factor], X[..., ::factor])``.
    """
    t = np.asarray(t)
    X = np.asarray(X)
    n_steps = t.size - 1
    if factor < 1 or n_steps % factor != 0:
        msg = f"n_steps={n_steps} must be a multiple of factor={factor}"
        raise ValueError(msg)
    return t[::factor], X[..., ::factor]


def brownian_bridge(
    n_paths: int,
    n_steps: int,
    T: float,
    rng: np.random.Generator,
    a: float = 0.0,
    b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """a 에서 b 로 가는 브라운 다리 ``B_t - (t/T) B_T + a + (b - a) t/T``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, X)`` with ``X[:, 0] == a`` and ``X[:, -1] == b``.
    """
    t, B = brownian_paths(n_paths, n_steps, T, rng)
    frac = t / T
    X = B - frac * B[:, -1:] + a + (b - a) * frac
    return t, X


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    T: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """기하 브라운 운동 dS = μS dt + σS dW 의 정확한(로그정규) 경로.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, S)`` with ``S = S0 exp((μ - σ²/2) t + σ B_t)``.
    """
    t, B = brownian_paths(n_paths, n_steps, T, rng)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * B)
    return t, S


def scaled_random_walk(
    n_paths: int, n_steps: int, T: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """돈스커 척도의 랜덤워크 ``S_k sqrt(T / n)`` (브라운 운동 근사).

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, W)`` with ``t = linspace(0, T, n_steps + 1)`` and
        ``W[:, k] = S_k * sqrt(T) / sqrt(n_steps)`` for a ±1 walk ``S_k``.
    """
    steps = np.where(rng.random((n_paths, n_steps)) < 0.5, -1.0, 1.0)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(steps, axis=1) * np.sqrt(T / n_steps)
    return np.linspace(0.0, T, n_steps + 1), W


def running_max(X: np.ndarray) -> np.ndarray:
    """경로별 누적 최댓값 ``max_{s <= t} X_s`` (axis 1)."""
    return np.maximum.accumulate(np.asarray(X), axis=-1)


def first_passage_index(X: np.ndarray, level: float) -> np.ndarray:
    """각 경로가 수준 level 에 처음 닿는 시간 인덱스.

    A path starting below ``level`` hits when ``X >= level``; a path starting
    above hits when ``X <= level``.

    Returns
    -------
    numpy.ndarray, shape (n_paths,), dtype int
        First index of the hit, ``-1`` if the level is never reached.
    """
    X = np.atleast_2d(np.asarray(X))
    upward = level >= X[:, 0]
    hit = np.where(upward[:, None], X >= level, X <= level)
    idx = hit.argmax(axis=1)
    idx[~hit.any(axis=1)] = -1
    return idx


def quadratic_variation(X: np.ndarray) -> np.ndarray:
    """격자 위의 누적 이차변동 ``Σ (X_{k+1} - X_k)²`` (axis 1, 0 에서 시작)."""
    X = np.asarray(X)
    qv = np.zeros_like(X, dtype=float)
    qv[..., 1:] = np.cumsum(np.diff(X, axis=-1) ** 2, axis=-1)
    return qv
