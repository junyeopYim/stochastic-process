"""재생 과정 도구.

Renewal epochs from i.i.d. interarrivals, the counting process, renewal-reward
totals and the age / residual-life processes on a grid.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def _interarrivals_until(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw i.i.d. interarrivals in growing chunks; keep those whose partial sums are ``<= T``."""
    pieces: list[np.ndarray] = []
    total = 0.0
    chunk = 64
    while True:
        x = np.asarray(sampler(chunk, rng), dtype=float).ravel()
        if x.size != chunk or np.any(x < 0):
            msg = "sampler(n, rng) must return n non-negative interarrival times"
            raise ValueError(msg)
        s = total + np.cumsum(x)
        if s[-1] > T:
            k = int(np.searchsorted(s, T, side="right"))
            pieces.append(x[:k])
            break
        if s[-1] <= total:
            msg = "sampler produced only zero interarrivals; cannot reach T"
            raise ValueError(msg)
        pieces.append(x)
        total = float(s[-1])
        chunk = min(2 * chunk, 1 << 20)
    return np.concatenate(pieces)


def renewal_times(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(0, T] 위의 재생 시각들을 i.i.d. 간격으로부터 만든다.

    Parameters
    ----------
    sampler : callable
        ``sampler(n, rng)`` returns ``n`` i.i.d. non-negative interarrival times.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray
        Sorted renewal epochs ``S_1 < S_2 < ... <= T``.
    """
    return np.cumsum(_interarrivals_until(sampler, T, rng))


def renewal_count(times: np.ndarray, t: ArrayLike) -> np.ndarray:
    """계수 과정 N(t) = #{n : S_n <= t}.

    Returns
    -------
    numpy.ndarray
        Integer counts with the shape of ``t``.
    """
    return np.asarray(np.searchsorted(np.asarray(times, dtype=float), np.asarray(t), side="right"))


def renewal_reward(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    reward_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    T: float,
    rng: np.random.Generator,
) -> float:
    """시각 T 까지 완료된 주기들의 보상 총합.

    Parameters
    ----------
    sampler : callable
        Interarrival sampler, as in :func:`renewal_times`.
    reward_fn : callable
        ``reward_fn(interarrivals, rng)`` maps the completed cycle lengths to one
        reward per cycle (may use ``rng`` for random rewards).
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    float
        Sum of rewards over cycles completed by ``T``.
    """
    x = _interarrivals_until(sampler, T, rng)
    if x.size == 0:
        return 0.0
    return float(np.sum(np.asarray(reward_fn(x, rng), dtype=float)))


def age_residual_process(times: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """격자 위에서 나이 A(t) 와 잔여수명 R(t) 를 계산한다.

    Parameters
    ----------
    times : numpy.ndarray
        Sorted renewal epochs.
    t_grid : numpy.ndarray
        Observation times.

    Returns
    -------
    tuple of numpy.ndarray
        ``(age, residual)``: ``age = t - last epoch <= t`` (``t`` itself before the
        first epoch) and ``residual = next epoch > t - t`` (``np.inf`` after the
        last epoch).
    """
    times = np.asarray(times, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    idx = np.searchsorted(times, t_grid, side="right")
    padded = np.concatenate([[0.0], times, [np.inf]])
    age = t_grid - padded[idx]
    residual = padded[idx + 1] - t_grid
    return age, residual
