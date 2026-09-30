"""이산시간 마르코프 연쇄(DTMC) 도구.

States are ``0..n-1`` and a transition matrix ``P`` is an ``(n, n)``
row-stochastic ndarray. Simulation is vectorised over paths and loops only over
time. Also contains the classic examples used in the notebooks (gambler's ruin,
Ehrenfest urn, PageRank, Metropolis chains, Galton-Watson branching).
"""

from collections import deque
from math import gcd

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components


def _as_matrix(P: ArrayLike) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        msg = f"expected a square matrix, got shape {P.shape}"
        raise ValueError(msg)
    return P


def is_stochastic(P: ArrayLike, atol: float = 1e-10) -> bool:
    """행 확률 행렬인지 검사한다.

    Parameters
    ----------
    P : array_like
        Candidate transition matrix.
    atol : float, default 1e-10
        Tolerance for non-negativity and row sums.

    Returns
    -------
    bool
        ``True`` iff ``P`` is square, entrywise ``>= -atol`` and every row sums to 1.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    return bool(np.all(P >= -atol) and np.allclose(P.sum(axis=1), 1.0, atol=atol))


def _sample_rows(C: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling: row ``i`` of ``C`` is a cumulative distribution for path ``i``."""
    return np.minimum((u[:, None] > C).sum(axis=1), C.shape[1] - 1)


def simulate_chain(
    P: ArrayLike,
    x0: int | ArrayLike,
    n_steps: int,
    rng: np.random.Generator,
    n_paths: int = 1,
) -> np.ndarray:
    """마르코프 연쇄 경로를 여러 개 동시에 시뮬레이션한다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Row-stochastic transition matrix.
    x0 : int or array_like
        Initial state for every path, or a probability vector over the ``n``
        states from which each path's initial state is drawn.
    n_steps : int
        Number of transitions.
    rng : numpy.random.Generator
        Random source.
    n_paths : int, default 1
        Number of independent paths.

    Returns
    -------
    numpy.ndarray, shape (n_paths, n_steps + 1), dtype int
        ``X[:, 0]`` is the initial state. Each step draws ``u = rng.random(n_paths)``
        and sets ``next = (u[:, None] > C[x]).sum(1)`` with ``C = P.cumsum(1)``.
    """
    P = _as_matrix(P)
    n = P.shape[0]
    C = np.cumsum(P, axis=1)
    X = np.empty((n_paths, n_steps + 1), dtype=np.int64)
    if np.ndim(x0) == 0:
        start = int(x0)
        if not 0 <= start < n:
            msg = f"x0={start} is not a state of a chain with {n} states"
            raise ValueError(msg)
        X[:, 0] = start
    else:
        p0 = np.asarray(x0, dtype=float)
        if p0.shape != (n,):
            msg = f"initial distribution must have shape ({n},), got {p0.shape}"
            raise ValueError(msg)
        X[:, 0] = _sample_rows(np.broadcast_to(np.cumsum(p0), (n_paths, n)), rng.random(n_paths))
    for k in range(n_steps):
        X[:, k + 1] = _sample_rows(C[X[:, k]], rng.random(n_paths))
    return X


def empirical_distribution(paths: np.ndarray, n_states: int, burn_in: int = 0) -> np.ndarray:
    """경로들이 각 상태에 머문 시간 비율을 구한다.

    Parameters
    ----------
    paths : numpy.ndarray, shape (n_paths, n_steps + 1)
        Integer state paths (a 1-D array is treated as one path).
    n_states : int
        Number of states.
    burn_in : int, default 0
        Time indices ``< burn_in`` are discarded.

    Returns
    -------
    numpy.ndarray, shape (n_states,)
        Fraction of (path, time) pairs spent in each state.
    """
    paths = np.atleast_2d(np.asarray(paths))
    sub = paths[:, burn_in:]
    if sub.size == 0:
        msg = "no observations left after burn-in"
        raise ValueError(msg)
    return np.bincount(sub.ravel(), minlength=n_states) / sub.size


def stationary(P: ArrayLike) -> np.ndarray:
    """선형 시스템 πP = π, Σπ = 1 을 풀어 정상분포를 구한다.

    One of the (redundant) balance equations is replaced by the normalisation
    constraint; the resulting system is non-singular for an irreducible chain.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Row-stochastic transition matrix.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Stationary distribution.

    Raises
    ------
    ValueError
        If ``P`` is not row-stochastic.
    """
    P = _as_matrix(P)
    if not is_stochastic(P):
        msg = "P is not a row-stochastic matrix"
        raise ValueError(msg)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def stationary_eig(P: ArrayLike) -> np.ndarray:
    """고유값 1의 왼쪽 고유벡터로 정상분포를 구한다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Row-stochastic transition matrix.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Real part of the left eigenvector for the eigenvalue closest to 1,
        normalised to sum to 1.
    """
    P = _as_matrix(P)
    w, v = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, idx])
    return pi / pi.sum()


def stationary_power(P: ArrayLike, n: int = 1000) -> np.ndarray:
    """행렬 거듭제곱 P^n 의 첫 행으로 정상분포를 근사한다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Row-stochastic transition matrix.
    n : int, default 1000
        Power.

    Returns
    -------
    numpy.ndarray, shape (n_states,)
        First row of ``matrix_power(P, n)``.
    """
    return np.linalg.matrix_power(_as_matrix(P), n)[0]


def communicating_classes(P: ArrayLike) -> list[list[int]]:
    """소통 클래스(강연결 성분)를 찾는다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Transition matrix; only the pattern ``P > 0`` matters.

    Returns
    -------
    list of list of int
        Classes sorted by their smallest state, each class sorted ascending.
    """
    P = _as_matrix(P)
    graph = csr_array((P > 0).astype(np.int8))
    _, labels = connected_components(graph, directed=True, connection="strong")
    classes: dict[int, list[int]] = {}
    for state, lab in enumerate(labels):
        classes.setdefault(int(lab), []).append(state)
    return sorted(classes.values(), key=lambda c: c[0])


def _class_period(P: np.ndarray, cls: list[int]) -> int:
    """gcd over intra-class edges ``u -> v`` of ``level[u] + 1 - level[v]`` (BFS levels)."""
    sub = P[np.ix_(cls, cls)] > 0
    level = np.full(len(cls), -1)
    level[0] = 0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(sub[u]):
            if level[v] < 0:
                level[v] = level[u] + 1
                queue.append(int(v))
    d = 0
    for u, v in zip(*np.nonzero(sub), strict=True):
        d = gcd(d, int(level[u] + 1 - level[v]))
    return d


def classify_states(P: ArrayLike) -> dict:
    """상태를 재귀/일시로 분류하고 각 클래스의 주기를 구한다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Transition matrix.

    Returns
    -------
    dict
        ``classes`` (list of lists, as :func:`communicating_classes`),
        ``recurrent`` (indices of closed classes), ``transient_states``,
        ``recurrent_states`` and ``period`` (class index -> gcd of cycle lengths
        inside the class, computed from BFS levels; ``0`` for a class without any
        cycle, i.e. a transient singleton without self-loop).
    """
    P = _as_matrix(P)
    n = P.shape[0]
    classes = communicating_classes(P)
    recurrent: list[int] = []
    period: dict[int, int] = {}
    for ci, cls in enumerate(classes):
        outside = np.ones(n, dtype=bool)
        outside[cls] = False
        if P[cls][:, outside].sum() <= 0.0:
            recurrent.append(ci)
        period[ci] = _class_period(P, cls)
    recurrent_states = sorted(s for ci in recurrent for s in classes[ci])
    transient_states = sorted(set(range(n)) - set(recurrent_states))
    return {
        "classes": classes,
        "recurrent": recurrent,
        "transient_states": transient_states,
        "recurrent_states": recurrent_states,
        "period": period,
    }


def absorption(P: ArrayLike, transient: list[int], absorbing: list[int]) -> dict:
    """흡수 연쇄의 기본 행렬, 흡수 확률, 기대 흡수 시간을 구한다.

    Parameters
    ----------
    P : array_like, shape (n, n)
        Transition matrix.
    transient : list of int
        Transient states ``T``.
    absorbing : list of int
        Absorbing states ``A``.

    Returns
    -------
    dict
        ``N = inv(I - Q)`` with ``Q = P[T, T]`` (fundamental matrix),
        ``B = N @ R`` with ``R = P[T, A]`` (absorption probabilities, shape
        ``(|T|, |A|)``) and ``t = N.sum(1)`` (expected steps to absorption).
    """
    P = _as_matrix(P)
    T = list(transient)
    A = list(absorbing)
    Q = P[np.ix_(T, T)]
    R = P[np.ix_(T, A)]
    N = np.linalg.inv(np.eye(len(T)) - Q)
    return {"N": N, "B": N @ R, "t": N.sum(axis=1)}


def hitting_times(paths: np.ndarray, target: int | set[int]) -> np.ndarray:
    """각 경로가 목표 상태 집합에 처음 도달하는 시각을 구한다.

    Parameters
    ----------
    paths : numpy.ndarray, shape (n_paths, n_steps + 1)
        Integer state paths.
    target : int or set of int
        Target state(s).

    Returns
    -------
    numpy.ndarray, shape (n_paths,), dtype float
        First time index at which the path is in ``target``; ``np.inf`` if never.
    """
    paths = np.atleast_2d(np.asarray(paths))
    targets = [int(target)] if isinstance(target, int | np.integer) else list(target)
    hit = np.isin(paths, targets)
    times = hit.argmax(axis=1).astype(float)
    times[~hit.any(axis=1)] = np.inf
    return times


def tv_distance(p: ArrayLike, q: ArrayLike) -> float:
    """두 분포 사이의 전변동 거리 ``0.5 * Σ|p - q|``."""
    return float(0.5 * np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())


def gamblers_ruin_matrix(N: int, p: float) -> np.ndarray:
    """도박꾼의 파산 연쇄(상태 0..N, 0과 N에서 흡수)의 전이행렬.

    Parameters
    ----------
    N : int
        Target fortune; states are ``0..N``.
    p : float
        Probability of winning one unit.

    Returns
    -------
    numpy.ndarray, shape (N + 1, N + 1)
    """
    P = np.zeros((N + 1, N + 1))
    i = np.arange(1, N)
    P[i, i + 1] = p
    P[i, i - 1] = 1.0 - p
    P[0, 0] = P[N, N] = 1.0
    return P


def gamblers_ruin_prob(i: int, N: int, p: float) -> float:
    """재산 i 에서 출발해 0 보다 N 에 먼저 도달할 확률.

    ``i / N`` for a fair game, otherwise ``(1 - r^i) / (1 - r^N)`` with ``r = q / p``.
    """
    if abs(p - 0.5) < 1e-12:
        return i / N
    r = (1.0 - p) / p
    return float((1.0 - r**i) / (1.0 - r**N))


def gamblers_ruin_duration(i: int, N: int, p: float) -> float:
    """재산 i 에서 출발한 게임의 기대 지속 시간.

    ``i (N - i)`` for a fair game, otherwise
    ``i / (q - p) - N / (q - p) * (1 - r^i) / (1 - r^N)`` with ``r = q / p``.
    """
    if abs(p - 0.5) < 1e-12:
        return float(i * (N - i))
    q = 1.0 - p
    r = q / p
    return float(i / (q - p) - (N / (q - p)) * (1.0 - r**i) / (1.0 - r**N))


def ehrenfest_matrix(n_balls: int) -> np.ndarray:
    """에렌페스트 항아리 모형(공 n 개)의 전이행렬.

    State ``i`` is the number of balls in urn A; a uniformly chosen ball moves
    to the other urn, so ``P[i, i - 1] = i / n`` and ``P[i, i + 1] = (n - i) / n``.

    Returns
    -------
    numpy.ndarray, shape (n_balls + 1, n_balls + 1)
    """
    n = n_balls
    P = np.zeros((n + 1, n + 1))
    i = np.arange(1, n + 1)
    P[i, i - 1] = i / n
    i = np.arange(0, n)
    P[i, i + 1] = (n - i) / n
    return P


def random_walk_paths(
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    p: float = 0.5,
    step: tuple = (-1, 1),
) -> np.ndarray:
    """0 에서 출발하는 단순 랜덤워크 경로들을 만든다.

    Parameters
    ----------
    n_paths, n_steps : int
        Number of paths and steps.
    rng : numpy.random.Generator
        Random source.
    p : float, default 0.5
        Probability of taking ``step[1]`` (the up step); ``step[0]`` otherwise.
    step : tuple, default (-1, 1)
        ``(down, up)`` increments.

    Returns
    -------
    numpy.ndarray, shape (n_paths, n_steps + 1), dtype int
        Cumulative sums, starting at 0.
    """
    down, up = step
    steps = np.where(rng.random((n_paths, n_steps)) < p, up, down)
    X = np.zeros((n_paths, n_steps + 1), dtype=steps.dtype)
    X[:, 1:] = np.cumsum(steps, axis=1)
    return X


def metropolis_chain(target: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    """유한 상태공간에서 메트로폴리스-헤이스팅스 전이행렬을 만든다.

    Parameters
    ----------
    target : numpy.ndarray, shape (n,)
        Target distribution π (need not be normalised).
    proposal : numpy.ndarray, shape (n, n)
        Row-stochastic proposal matrix ``q``.

    Returns
    -------
    numpy.ndarray, shape (n, n)
        ``P[x, y] = q(x, y) min(1, π(y) q(y, x) / (π(x) q(x, y)))`` for ``x != y``
        and ``P[x, x] = 1 - Σ_{y != x} P[x, y]`` (rejections stay put).
    """
    pi = np.asarray(target, dtype=float)
    q = _as_matrix(proposal)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (pi[None, :] * q.T) / (pi[:, None] * q)
    accept = np.minimum(1.0, np.nan_to_num(ratio, nan=0.0, posinf=1.0))
    P = q * accept
    np.fill_diagonal(P, 0.0)
    P[np.diag_indices_from(P)] = 1.0 - P.sum(axis=1)
    return P


def pagerank(A: np.ndarray, damping: float = 0.85) -> np.ndarray:
    """인접행렬로부터 구글 행렬을 만들고 페이지랭크(정상분포)를 구한다.

    Parameters
    ----------
    A : numpy.ndarray, shape (n, n)
        Adjacency (or weighted link) matrix; ``A[i, j] > 0`` is a link ``i -> j``.
    damping : float, default 0.85
        Probability of following a link rather than teleporting uniformly.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Stationary distribution of ``G = d P + (1 - d) / n`` where rows of ``P``
        are the normalised rows of ``A`` (dangling rows become uniform).
    """
    A = _as_matrix(A)
    n = A.shape[0]
    out = A.sum(axis=1)
    dangling = out <= 0
    P = np.where(dangling[:, None], 1.0 / n, A / np.where(dangling, 1.0, out)[:, None])
    G = damping * P + (1.0 - damping) / n
    return stationary(G)


def galton_watson(
    offspring_pmf: ArrayLike,
    n_generations: int,
    rng: np.random.Generator,
    n_paths: int = 1,
    z0: int = 1,
) -> np.ndarray:
    """골턴-왓슨 분기과정의 세대별 개체 수를 시뮬레이션한다.

    Each generation draws, per path, ``rng.multinomial(Z, pmf)`` (how many
    individuals have 0, 1, 2, ... children) and sums ``k * count_k``, so the cost
    per generation is ``O(n_paths * len(pmf))`` regardless of population size.

    Parameters
    ----------
    offspring_pmf : array_like
        ``pmf[k]`` is the probability of ``k`` children (normalised internally).
    n_generations : int
        Number of generations to simulate.
    rng : numpy.random.Generator
        Random source.
    n_paths : int, default 1
        Number of independent populations.
    z0 : int, default 1
        Initial population size.

    Returns
    -------
    numpy.ndarray, shape (n_paths, n_generations + 1), dtype int
    """
    pmf = np.asarray(offspring_pmf, dtype=float)
    pmf = pmf / pmf.sum()
    support = np.arange(pmf.size)
    Z = np.zeros((n_paths, n_generations + 1), dtype=np.int64)
    Z[:, 0] = z0
    for g in range(n_generations):
        if not Z[:, g].any():
            break
        counts = rng.multinomial(Z[:, g], pmf)
        Z[:, g + 1] = counts @ support
    return Z


def extinction_probability(
    offspring_pmf: ArrayLike, tol: float = 1e-12, max_iter: int = 10_000
) -> float:
    """자손 분포의 pgf 를 0 에서 반복해 최소 고정점(멸종 확률)을 구한다.

    Parameters
    ----------
    offspring_pmf : array_like
        Offspring distribution (normalised internally).
    tol : float, default 1e-12
        Stop when successive iterates differ by less than ``tol``.
    max_iter : int, default 10000
        Iteration cap.

    Returns
    -------
    float
        Smallest fixed point of ``s -> Σ p_k s^k`` in ``[0, 1]``; exactly ``1.0`` when the
        offspring mean is ``<= 1`` (subcritical/critical case, by the extinction theorem).
    """
    pmf = np.asarray(offspring_pmf, dtype=float)
    pmf = pmf / pmf.sum()
    mean = float(np.dot(np.arange(len(pmf)), pmf))
    # 정리: m <= 1 (그리고 p_1 < 1) 이면 멸종 확률은 정확히 1. 임계(m = 1) 경우 반복은
    # 2/(sigma^2 n) 속도로만 수렴하므로 반복 대신 정리를 그대로 쓴다.
    p_one = float(pmf[1]) if len(pmf) > 1 else 0.0
    if mean <= 1.0 and p_one < 1.0:
        return 1.0
    s = 0.0
    for _ in range(max_iter):
        s_new = float(np.polynomial.polynomial.polyval(s, pmf))
        if abs(s_new - s) < tol:
            return s_new
        s = s_new
    return s
