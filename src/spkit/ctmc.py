"""연속시간 마르코프 연쇄(CTMC) 도구.

A generator ``Q`` is an ``(n, n)`` matrix with non-negative off-diagonal entries
and zero row sums. Includes the Gillespie (direct) simulation, matrix
exponentials, uniformisation, birth-death chains and M/M/c queue formulas.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm


def _as_generator(Q: ArrayLike) -> np.ndarray:
    Q = np.asarray(Q, dtype=float)
    if not is_generator(Q):
        msg = "Q is not a valid generator matrix"
        raise ValueError(msg)
    return Q


def is_generator(Q: ArrayLike, atol: float = 1e-10) -> bool:
    """생성원 행렬(비대각 ≥ 0, 행합 0)인지 검사한다."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return False
    off = Q - np.diag(np.diag(Q))
    return bool(np.all(off >= -atol) and np.allclose(Q.sum(axis=1), 0.0, atol=atol))


def gillespie(
    Q: ArrayLike,
    x0: int,
    T: float,
    rng: np.random.Generator,
    max_events: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """길레스피 직접법으로 CTMC 궤적 하나를 시뮬레이션한다.

    Parameters
    ----------
    Q : array_like, shape (n, n)
        Generator matrix.
    x0 : int
        Initial state.
    T : float
        Horizon.
    rng : numpy.random.Generator
        Random source.
    max_events : int, default 1_000_000
        Cap on the number of jumps.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times, states)``: jump times starting with ``0.0`` and the states
        entered, starting with ``x0``. The simulation stops when the next jump
        would occur after ``T`` (the last state holds up to ``T``), when an
        absorbing state is reached, or after ``max_events`` jumps.
    """
    Q = _as_generator(Q)
    J, q = embedded_chain(Q)
    C = np.cumsum(J, axis=1)
    n = Q.shape[0]
    x = int(x0)
    t = 0.0
    times = [0.0]
    states = [x]
    for _ in range(max_events):
        rate = q[x]
        if rate <= 0.0:
            break
        t += rng.exponential(1.0 / rate)
        if t > T:
            break
        x = min(int(np.searchsorted(C[x], rng.random(), side="right")), n - 1)
        times.append(t)
        states.append(x)
    return np.asarray(times, dtype=float), np.asarray(states, dtype=np.int64)


def gillespie_paths(
    Q: ArrayLike,
    x0: int,
    T: float,
    rng: np.random.Generator,
    n_paths: int,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """여러 길레스피 궤적을 등간격 격자 위에서 표본화한다.

    Returns
    -------
    tuple of numpy.ndarray
        ``t_grid`` of shape ``(n_grid,)`` and integer ``X`` of shape
        ``(n_paths, n_grid)`` with ``X[i, k]`` the state of path ``i`` at
        ``t_grid[k]``.
    """
    t_grid = np.linspace(0.0, T, n_grid)
    X = np.empty((n_paths, n_grid), dtype=np.int64)
    for i in range(n_paths):
        times, states = gillespie(Q, x0, T, rng)
        X[i] = states[np.searchsorted(times, t_grid, side="right") - 1]
    return t_grid, X


def transition_matrix(Q: ArrayLike, t: float) -> np.ndarray:
    """전이확률행렬 P(t) = exp(Qt)."""
    return expm(np.asarray(Q, dtype=float) * t)


def stationary_ctmc(Q: ArrayLike) -> np.ndarray:
    """πQ = 0, Σπ = 1 을 풀어 정상분포를 구한다.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    Q = _as_generator(Q)
    n = Q.shape[0]
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def embedded_chain(Q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """내재 점프 연쇄와 체류율을 구한다.

    Returns
    -------
    tuple of numpy.ndarray
        ``(J, q)``: jump matrix ``J[i, j] = Q[i, j] / q_i`` (``j != i``, zero
        diagonal; an absorbing state gets ``J[i, i] = 1``) and holding rates
        ``q_i = -Q[i, i]``.
    """
    Q = np.asarray(Q, dtype=float)
    q = -np.diag(Q).copy()
    J = Q.copy()
    np.fill_diagonal(J, 0.0)
    active = q > 0
    J[active] /= q[active, None]
    absorbing = np.flatnonzero(~active)
    J[absorbing] = 0.0
    J[absorbing, absorbing] = 1.0
    return J, q


def uniformize(Q: ArrayLike, lam: float | None = None) -> tuple[np.ndarray, float]:
    """균일화: P = I + Q/λ 와 λ 를 돌려준다.

    Parameters
    ----------
    Q : array_like, shape (n, n)
        Generator.
    lam : float, optional
        Uniformisation rate; defaults to the largest holding rate.

    Returns
    -------
    tuple
        ``(P, lam)`` with ``P`` row-stochastic.
    """
    Q = _as_generator(Q)
    q_max = float(np.max(-np.diag(Q)))
    if lam is None:
        lam = q_max
    if lam <= 0 or lam < q_max * (1.0 - 1e-12):
        msg = f"lam must be >= max holding rate {q_max:g}"
        raise ValueError(msg)
    return np.eye(Q.shape[0]) + Q / lam, float(lam)


def _birth_death_rates(birth: ArrayLike, death: ArrayLike) -> tuple[np.ndarray, np.ndarray, int]:
    death = np.asarray(death, dtype=float)
    birth = np.asarray(birth, dtype=float)
    n = death.size - 1
    if birth.size < n:
        msg = f"birth needs at least {n} rates (one per state 0..{n - 1}), got {birth.size}"
        raise ValueError(msg)
    return birth[:n], death, n


def birth_death_generator(birth: ArrayLike, death: ArrayLike) -> np.ndarray:
    """출생-사망 연쇄의 생성원을 만든다.

    Parameters
    ----------
    birth : array_like
        ``birth[i]`` is the rate ``i -> i + 1`` for ``i = 0..n-1`` (extra
        trailing entries are ignored).
    death : array_like, length n + 1
        ``death[i]`` is the rate ``i -> i - 1`` for ``i = 1..n``; ``death[0]`` is
        ignored.

    Returns
    -------
    numpy.ndarray, shape (n + 1, n + 1)
    """
    birth, death, n = _birth_death_rates(birth, death)
    Q = np.zeros((n + 1, n + 1))
    i = np.arange(n)
    Q[i, i + 1] = birth
    j = np.arange(1, n + 1)
    Q[j, j - 1] = death[1:]
    Q[np.diag_indices(n + 1)] = -Q.sum(axis=1)
    return Q


def birth_death_stationary(birth: ArrayLike, death: ArrayLike) -> np.ndarray:
    """출생-사망 연쇄의 정상분포 π_n ∝ Π_{k<n} birth[k]/death[k+1].

    Computed through cumulative log-sums for numerical stability (a zero birth
    rate cuts the support; zero death rates are rejected).

    Returns
    -------
    numpy.ndarray, shape (n + 1,)
    """
    birth, death, _ = _birth_death_rates(birth, death)
    if np.any(death[1:] <= 0):
        msg = "death rates for states 1..n must be positive"
        raise ValueError(msg)
    with np.errstate(divide="ignore"):
        log_ratio = np.log(birth) - np.log(death[1:])
    log_pi = np.concatenate([[0.0], np.cumsum(log_ratio)])
    pi = np.exp(log_pi - log_pi.max())
    return pi / pi.sum()


def mm1_stationary(lam: float, mu: float, n_max: int) -> np.ndarray:
    """M/M/1 대기열의 정상분포 (1-ρ)ρ^n 을 0..n_max 로 절단·재정규화한다."""
    rho = lam / mu
    w = rho ** np.arange(n_max + 1, dtype=float)
    return w / w.sum()


def mmc_metrics(lam: float, mu: float, c: int) -> dict:
    """M/M/c 대기열의 정상상태 성능지표 (Erlang C).

    Parameters
    ----------
    lam : float
        Arrival rate.
    mu : float
        Service rate per server.
    c : int
        Number of servers.

    Returns
    -------
    dict
        ``rho = lam / (c mu)``, ``p_wait`` (Erlang C probability that an arrival
        waits), ``Lq``, ``L = Lq + lam / mu``, ``Wq = Lq / lam``, ``W = Wq + 1 / mu``.

    Raises
    ------
    ValueError
        If the queue is unstable (``rho >= 1``).
    """
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        msg = f"unstable queue: lam / (c mu) = {rho:g} >= 1"
        raise ValueError(msg)
    partial = 0.0
    term = 1.0  # a^k / k!
    for k in range(c):
        partial += term
        term *= a / (k + 1)
    tail = term / (1.0 - rho)  # a^c / c! * 1 / (1 - rho)
    p_wait = tail / (partial + tail)
    Lq = p_wait * rho / (1.0 - rho)
    L = Lq + a
    Wq = Lq / lam
    W = Wq + 1.0 / mu
    return {"rho": rho, "p_wait": p_wait, "Lq": Lq, "L": L, "Wq": Wq, "W": W}


def time_average_occupation(
    times: np.ndarray, states: np.ndarray, T: float, n_states: int
) -> np.ndarray:
    """길레스피 궤적이 [0, T] 동안 각 상태에 머문 시간 비율.

    Parameters
    ----------
    times, states : numpy.ndarray
        Jump times and states as returned by :func:`gillespie`.
    T : float
        Horizon (the last state holds until ``T``).
    n_states : int
        Number of states.

    Returns
    -------
    numpy.ndarray, shape (n_states,)
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states)
    keep = times <= T
    times, states = times[keep], states[keep]
    durations = np.diff(np.append(times, T))
    return np.bincount(states, weights=durations, minlength=n_states) / T
