import numpy as np
import pytest
from scipy.stats import binom

from spkit.markov import (
    absorption,
    classify_states,
    communicating_classes,
    ehrenfest_matrix,
    empirical_distribution,
    extinction_probability,
    galton_watson,
    gamblers_ruin_duration,
    gamblers_ruin_matrix,
    gamblers_ruin_prob,
    hitting_times,
    is_stochastic,
    metropolis_chain,
    pagerank,
    random_walk_paths,
    simulate_chain,
    stationary,
    stationary_eig,
    stationary_power,
    tv_distance,
)
from spkit.mc import assert_close, mc_estimate, mc_proportion

P3 = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.7]])
PI3 = np.array([0.6, 0.3, 0.1])


def test_is_stochastic():
    assert is_stochastic(P3)
    assert not is_stochastic(np.array([[0.5, 0.6], [1.0, 0.0]]))
    assert not is_stochastic(np.array([[1.5, -0.5], [0.0, 1.0]]))
    assert not is_stochastic(np.ones((2, 3)))


def test_simulate_chain_shape_and_one_step_law(rng):
    X = simulate_chain(P3, 1, 1, rng, n_paths=20_000)
    assert X.shape == (20_000, 2)
    assert np.issubdtype(X.dtype, np.integer)
    assert np.all(X[:, 0] == 1)
    for j in range(3):
        assert_close(mc_proportion(X[:, 1] == j), P3[1, j], name=f"P[1,{j}]")


def test_simulate_chain_initial_distribution(rng):
    X = simulate_chain(P3, [0.2, 0.3, 0.5], 0, rng, n_paths=20_000)
    for j, p in enumerate([0.2, 0.3, 0.5]):
        assert_close(mc_proportion(X[:, 0] == j), p, name=f"x0={j}")
    with pytest.raises(ValueError):
        simulate_chain(P3, 5, 1, rng)


def test_empirical_distribution_exact():
    paths = np.array([[0, 1, 1, 2], [2, 2, 0, 0]])
    np.testing.assert_allclose(empirical_distribution(paths, 3), [3 / 8, 2 / 8, 3 / 8])
    np.testing.assert_allclose(empirical_distribution(paths, 3, burn_in=2), [0.5, 0.25, 0.25])


def test_empirical_matches_stationary_for_long_run(rng):
    X = simulate_chain(P3, 0, 5_000, rng, n_paths=20)
    np.testing.assert_allclose(empirical_distribution(X, 3, burn_in=500), PI3, atol=0.02)


def test_stationary_variants_agree():
    pi = stationary(P3)
    np.testing.assert_allclose(pi, PI3)
    np.testing.assert_allclose(pi @ P3, pi)
    np.testing.assert_allclose(stationary_eig(P3), PI3)
    np.testing.assert_allclose(stationary_power(P3), PI3)
    with pytest.raises(ValueError):
        stationary(np.array([[0.5, 0.6], [1.0, 0.0]]))


def test_communicating_classes_and_classification():
    P = gamblers_ruin_matrix(4, 0.3)
    assert communicating_classes(P) == [[0], [1, 2, 3], [4]]
    info = classify_states(P)
    assert info["classes"] == [[0], [1, 2, 3], [4]]
    assert info["recurrent"] == [0, 2]
    assert info["transient_states"] == [1, 2, 3]
    assert info["recurrent_states"] == [0, 4]
    assert info["period"] == {0: 1, 1: 2, 2: 1}
    assert classify_states(ehrenfest_matrix(5))["period"] == {0: 2}
    assert classify_states(P3)["period"] == {0: 1}
    assert classify_states(P3)["recurrent_states"] == [0, 1, 2]
    # transient singleton without any cycle -> gcd of the empty set (0)
    chain = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert classify_states(chain)["period"] == {0: 0, 1: 1}


def _ruin_duration_by_linear_system(N, p):
    """Independent check: solve D_i = 1 + p D_{i+1} + q D_{i-1}, D_0 = D_N = 0."""
    q = 1.0 - p
    m = N - 1
    A = np.eye(m)
    for row in range(m):
        if row + 1 < m:
            A[row, row + 1] = -p
        if row - 1 >= 0:
            A[row, row - 1] = -q
    return np.linalg.solve(A, np.ones(m))


@pytest.mark.parametrize("p", [0.3, 0.5, 0.65])
def test_absorption_matches_gamblers_ruin_closed_forms(p):
    N = 10
    P = gamblers_ruin_matrix(N, p)
    assert is_stochastic(P)
    assert P[0, 0] == 1.0 and P[N, N] == 1.0
    transient = list(range(1, N))
    res = absorption(P, transient, [0, N])
    assert res["N"].shape == (N - 1, N - 1) and res["B"].shape == (N - 1, 2)
    np.testing.assert_allclose(res["B"].sum(axis=1), 1.0)
    for k, i in enumerate(transient):
        np.testing.assert_allclose(res["B"][k, 1], gamblers_ruin_prob(i, N, p))
        np.testing.assert_allclose(res["t"][k], gamblers_ruin_duration(i, N, p))
    np.testing.assert_allclose(res["t"], _ruin_duration_by_linear_system(N, p))


def test_gamblers_ruin_monte_carlo(rng):
    N, p, i = 8, 0.4, 3
    X = simulate_chain(gamblers_ruin_matrix(N, p), i, 400, rng, n_paths=20_000)
    tau = hitting_times(X, {0, N})
    assert np.all(np.isfinite(tau))
    won = X[np.arange(X.shape[0]), tau.astype(int)] == N
    assert_close(mc_proportion(won), gamblers_ruin_prob(i, N, p), name="win prob")
    assert_close(mc_estimate(tau), gamblers_ruin_duration(i, N, p), name="duration")


def test_hitting_times_exact():
    paths = np.array([[0, 1, 2, 2], [1, 1, 1, 1], [2, 0, 0, 0]])
    np.testing.assert_array_equal(hitting_times(paths, 2), [2.0, np.inf, 0.0])
    np.testing.assert_array_equal(hitting_times(paths, {0, 2}), [0.0, np.inf, 0.0])
    assert hitting_times(paths, 2).dtype == float


def test_tv_distance():
    np.testing.assert_allclose(tv_distance([0.5, 0.5], [0.5, 0.5]), 0.0)
    np.testing.assert_allclose(tv_distance([1.0, 0.0], [0.0, 1.0]), 1.0)
    np.testing.assert_allclose(tv_distance([0.2, 0.8], [0.5, 0.5]), 0.3)


def test_ehrenfest_matrix():
    n = 6
    P = ehrenfest_matrix(n)
    assert P.shape == (n + 1, n + 1) and is_stochastic(P)
    np.testing.assert_allclose(P[2, 1], 2 / 6)
    np.testing.assert_allclose(P[2, 3], 4 / 6)
    np.testing.assert_allclose(stationary(P), binom.pmf(np.arange(n + 1), n, 0.5))


def test_random_walk_paths(rng):
    X = random_walk_paths(20_000, 50, rng, p=0.6)
    assert X.shape == (20_000, 51) and np.issubdtype(X.dtype, np.integer)
    assert np.all(X[:, 0] == 0)
    assert set(np.unique(np.diff(X, axis=1))) == {-1, 1}
    assert_close(mc_estimate(X[:, -1]), 50 * (2 * 0.6 - 1), name="drift")
    Y = random_walk_paths(3, 4, rng, step=(0, 2))
    assert set(np.unique(np.diff(Y, axis=1))) <= {0, 2}


def test_metropolis_chain_detailed_balance():
    target = np.array([1.0, 2.0, 3.0, 4.0])  # unnormalised
    n = target.size
    proposal = np.full((n, n), 1.0 / n)
    P = metropolis_chain(target, proposal)
    assert is_stochastic(P)
    pi = target / target.sum()
    np.testing.assert_allclose(pi[:, None] * P, (pi[:, None] * P).T)
    np.testing.assert_allclose(stationary(P), pi)
    # nearest-neighbour proposal with an explicit hand computation
    q = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    P2 = metropolis_chain(np.array([0.5, 0.25, 0.25]), q)
    np.testing.assert_allclose(P2[0], [0.75, 0.25, 0.0])
    np.testing.assert_allclose(P2[1], [0.5, 0.0, 0.5])


def test_pagerank():
    A = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    pr = pagerank(A, damping=0.85)
    np.testing.assert_allclose(pr.sum(), 1.0)
    # independent power iteration on the same Google matrix
    n = 4
    P = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    G = 0.85 * P + 0.15 / n
    v = np.full(n, 1.0 / n)
    for _ in range(500):
        v = v @ G
    np.testing.assert_allclose(pr, v, atol=1e-10)
    assert pr[2] > pr[3]


def test_galton_watson_shape_and_extinction(rng):
    pmf = [0.2, 0.3, 0.5]  # mean 1.3, extinction probability 0.4
    q = extinction_probability(pmf)
    np.testing.assert_allclose(q, 0.4, atol=1e-9)
    Z = galton_watson(pmf, 30, rng, n_paths=20_000)
    assert Z.shape == (20_000, 31) and np.issubdtype(Z.dtype, np.integer)
    assert np.all(Z[:, 0] == 1)
    assert_close(mc_proportion(Z[:, -1] == 0), q, name="extinction")


def test_galton_watson_large_population(rng):
    pmf = np.array([0.2, 0.3, 0.5])
    Z = galton_watson(pmf, 5, rng, n_paths=200, z0=100_000)
    assert Z.shape == (200, 6)
    mean = 100_000 * 1.3**5
    # each Z_5 has sd of order sqrt(Z) * const, well below 1% of the mean
    np.testing.assert_allclose(Z[:, -1].mean(), mean, rtol=1e-3)
    dead = galton_watson([1.0], 3, rng, n_paths=4)
    np.testing.assert_array_equal(dead[:, 1:], 0)


def test_extinction_probability_cases():
    np.testing.assert_allclose(extinction_probability([0.5, 0.5]), 1.0, atol=1e-6)
    np.testing.assert_allclose(extinction_probability([0.25, 0.25, 0.5]), 0.5, atol=1e-9)
    np.testing.assert_allclose(extinction_probability([0.0, 0.0, 1.0]), 0.0)


def test_extinction_probability_critical_and_subcritical_return_one() -> None:
    # 임계(m = 1): 반복만으로는 1 에 도달하지 못하므로 정리로 처리해야 한다
    assert extinction_probability([0.25, 0.5, 0.25]) == 1.0
    from scipy.stats import poisson

    assert extinction_probability(poisson.pmf(np.arange(41), 1.0)) == 1.0
    # 아임계
    assert extinction_probability([0.5, 0.3, 0.2]) == 1.0
    # 초임계: pgf 고정점 (0.25 + 0.25 s + 0.5 s^2 = s -> s = 1/2)
    np.testing.assert_allclose(extinction_probability([0.25, 0.25, 0.5]), 0.5, atol=1e-10)
