"""푸아송 과정 시뮬레이션 도구.

Homogeneous arrivals via exponential gaps, the order-statistics construction,
thinning for non-homogeneous processes, compound Poisson sums and the age /
residual life at a fixed time.
"""

from collections.abc import Callable

import numpy as np


def poisson_arrivals(rate: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """(0, T] 위의 비율 rate 푸아송 과정 도착 시각을 만든다.

    Exponential gaps are drawn in chunks and accumulated until the horizon is
    passed.

    Parameters
    ----------
    rate : float
        Arrival rate λ.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray
        Sorted arrival times in ``(0, T]``.
    """
    if rate < 0 or T < 0:
        msg = "rate and T must be non-negative"
        raise ValueError(msg)
    if rate == 0 or T == 0:
        return np.empty(0)
    chunk = max(16, int(1.5 * rate * T) + 16)
    pieces: list[np.ndarray] = []
    t_last = 0.0
    while True:
        arr = t_last + np.cumsum(rng.exponential(1.0 / rate, chunk))
        if arr[-1] > T:
            pieces.append(arr[arr <= T])
            break
        pieces.append(arr)
        t_last = float(arr[-1])
    return np.concatenate(pieces)


def poisson_arrivals_conditional(n: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """N(T) = n 이 주어졌을 때의 도착 시각: 순서통계량 구성.

    Parameters
    ----------
    n : int
        Number of arrivals in ``(0, T]``.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sorted i.i.d. uniforms on ``(0, T]``.
    """
    return np.sort(T * (1.0 - rng.random(n)))


def nhpp_thinning(
    rate_fn: Callable[[np.ndarray], np.ndarray],
    rate_max: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """비동질 푸아송 과정을 솎아내기(thinning)로 만든다.

    Candidates are homogeneous arrivals with rate ``rate_max``; each candidate at
    time ``s`` is kept with probability ``rate_fn(s) / rate_max``.

    Parameters
    ----------
    rate_fn : callable
        Vectorised intensity ``λ(t)``; must satisfy ``λ(t) <= rate_max``.
    rate_max : float
        Dominating constant rate.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray
        Sorted arrival times in ``(0, T]``.

    Raises
    ------
    ValueError
        If ``rate_fn`` exceeds ``rate_max`` at some candidate time.
    """
    cand = poisson_arrivals(rate_max, T, rng)
    if cand.size == 0:
        return cand
    lam = np.asarray(rate_fn(cand), dtype=float)
    if np.any(lam > rate_max * (1.0 + 1e-12)):
        msg = "rate_fn exceeds rate_max; thinning would be biased"
        raise ValueError(msg)
    return cand[rng.random(cand.size) * rate_max < lam]


def count_in_bins(times: np.ndarray, T: float, n_bins: int) -> np.ndarray:
    """[0, T] 를 등간격 구간으로 나눠 구간별 도착 수를 센다.

    Returns
    -------
    numpy.ndarray, shape (n_bins,), dtype int
    """
    counts, _ = np.histogram(np.asarray(times, dtype=float), bins=n_bins, range=(0.0, T))
    return counts


def thin(times: np.ndarray, p: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """각 도착을 독립적으로 확률 p 로 남긴다.

    Returns
    -------
    tuple of numpy.ndarray
        ``(kept, removed)``.
    """
    times = np.asarray(times)
    keep = rng.random(times.size) < p
    return times[keep], times[~keep]


def compound_poisson(
    rate: float,
    T: float,
    jump_sampler: Callable[[int, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    n_paths: int = 1,
) -> np.ndarray:
    """복합 푸아송 과정의 시각 T 값 X(T) = Σ_{i<=N(T)} J_i 를 만든다.

    Parameters
    ----------
    rate : float
        Arrival rate of jumps.
    T : float
        Horizon.
    jump_sampler : callable
        ``jump_sampler(k, rng)`` returns ``k`` i.i.d. jump sizes.
    rng : numpy.random.Generator
        Random source.
    n_paths : int, default 1
        Number of independent copies.

    Returns
    -------
    numpy.ndarray, shape (n_paths,)
        Total jump sum per path (``N ~ Poisson(rate T)`` per path; jumps for all
        paths are drawn in one call and summed with ``bincount``).
    """
    N = rng.poisson(rate * T, size=n_paths)
    total = int(N.sum())
    owner = np.repeat(np.arange(n_paths), N)
    if total == 0:
        return np.zeros(n_paths)
    jumps = np.asarray(jump_sampler(total, rng), dtype=float).ravel()
    if jumps.size != total:
        msg = f"jump_sampler returned {jumps.size} values, expected {total}"
        raise ValueError(msg)
    return np.bincount(owner, weights=jumps, minlength=n_paths)


def age_residual_at(times: np.ndarray, t: float) -> tuple[float, float]:
    """시각 t 에서의 나이 A(t) 와 잔여수명 R(t).

    Parameters
    ----------
    times : numpy.ndarray
        Sorted arrival times.
    t : float
        Observation time.

    Returns
    -------
    tuple of float
        ``A(t) = t - (last arrival <= t)`` (``t`` itself if none) and
        ``R(t) = (next arrival > t) - t`` (``np.inf`` if none).
    """
    times = np.asarray(times, dtype=float)
    idx = int(np.searchsorted(times, t, side="right"))
    age = t - times[idx - 1] if idx > 0 else t
    residual = times[idx] - t if idx < times.size else np.inf
    return float(age), float(residual)
