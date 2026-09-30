from math import factorial

import numpy as np
import pytest

from spkit.ctmc import (
    birth_death_generator,
    birth_death_stationary,
    embedded_chain,
    gillespie,
    gillespie_paths,
    is_generator,
    mm1_stationary,
    mmc_metrics,
    stationary_ctmc,
    time_average_occupation,
    transition_matrix,
    uniformize,
)
from spkit.markov import is_stochastic
from spkit.mc import assert_close, mc_proportion

Q2 = np.array([[-1.0, 1.0], [2.0, -2.0]])  # 0 -> 1 at rate 1, 1 -> 0 at rate 2


def test_is_generator():
    assert is_generator(Q2)
    assert not is_generator(np.array([[-1.0, 0.5], [2.0, -2.0]]))
    assert not is_generator(np.array([[1.0, -1.0], [2.0, -2.0]]))
    assert not is_generator(np.ones((2, 3)))


def test_gillespie_trajectory(rng):
    times, states = gillespie(Q2, 0, 50.0, rng)
    assert times[0] == 0.0 and states[0] == 0
    assert np.all(np.diff(times) > 0) and times[-1] <= 50.0
    assert np.all(np.diff(states) != 0)  # every jump changes state
    assert np.issubdtype(states.dtype, np.integer)
    times, states = gillespie(Q2, 0, 50.0, rng, max_events=5)
    assert times.size == 6
    # absorbing state stops the simulation
    Qa = np.array([[-1.0, 1.0], [0.0, 0.0]])
    times, states = gillespie(Qa, 0, 1e9, rng)
    assert states.tolist() == [0, 1] and times.size == 2


def test_gillespie_paths_match_transition_matrix(rng):
    T = 3.0
    t_grid, X = gillespie_paths(Q2, 0, T, rng, n_paths=5_000, n_grid=31)
    assert t_grid.shape == (31,) and X.shape == (5_000, 31)
    assert np.all(X[:, 0] == 0)
    assert_close(mc_proportion(X[:, -1] == 1), transition_matrix(Q2, T)[0, 1], name="P01(T)")
    k = 10  # intermediate grid point
    assert_close(mc_proportion(X[:, k] == 1), transition_matrix(Q2, t_grid[k])[0, 1], name="P01")


def test_transition_matrix_closed_form():
    a, b, t = 1.0, 2.0, 0.7
    Pt = transition_matrix(Q2, t)
    np.testing.assert_allclose(Pt.sum(axis=1), 1.0)
    np.testing.assert_allclose(transition_matrix(Q2, 0.0), np.eye(2))
    np.testing.assert_allclose(Pt[0, 1], a / (a + b) * (1 - np.exp(-(a + b) * t)))
    np.testing.assert_allclose(Pt[1, 0], b / (a + b) * (1 - np.exp(-(a + b) * t)))


def test_stationary_ctmc():
    pi = stationary_ctmc(Q2)
    np.testing.assert_allclose(pi, [2 / 3, 1 / 3])
    np.testing.assert_allclose(pi @ Q2, 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        stationary_ctmc(np.array([[1.0, -1.0], [2.0, -2.0]]))


def test_embedded_chain():
    Q = np.array([[-3.0, 1.0, 2.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    J, q = embedded_chain(Q)
    np.testing.assert_allclose(q, [3.0, 1.0, 0.0])
    np.testing.assert_allclose(J, [[0, 1 / 3, 2 / 3], [1, 0, 0], [0, 0, 1]])
    assert is_stochastic(J)


def test_uniformize():
    P, lam = uniformize(Q2)
    assert lam == 2.0
    assert is_stochastic(P)
    np.testing.assert_allclose(P, np.eye(2) + Q2 / lam)
    P5, lam5 = uniformize(Q2, 5.0)
    assert lam5 == 5.0 and is_stochastic(P5)
    np.testing.assert_allclose(P5, np.eye(2) + Q2 / 5.0)
    with pytest.raises(ValueError):
        uniformize(Q2, 1.0)


def test_birth_death_generator():
    birth = [1.0, 2.0, 3.0]
    death = [0.0, 4.0, 5.0, 6.0]
    Q = birth_death_generator(birth, death)
    assert Q.shape == (4, 4) and is_generator(Q)
    np.testing.assert_allclose(np.diag(Q, 1), birth)
    np.testing.assert_allclose(np.diag(Q, -1), death[1:])
    # a longer birth vector is truncated to the n needed rates
    np.testing.assert_allclose(birth_death_generator(birth + [99.0], death), Q)
    with pytest.raises(ValueError):
        birth_death_generator([1.0], death)


def test_birth_death_stationary_independent_product():
    birth = np.array([2.0, 1.5, 1.0, 0.5])
    death = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pi = birth_death_stationary(birth, death)
    ref = np.array([np.prod(birth[:k] / death[1 : k + 1]) for k in range(5)])
    np.testing.assert_allclose(pi, ref / ref.sum())
    np.testing.assert_allclose(pi, stationary_ctmc(birth_death_generator(birth, death)))
    np.testing.assert_allclose(pi @ birth_death_generator(birth, death), 0.0, atol=1e-12)
    # zero birth rate cuts the support
    pi0 = birth_death_stationary([1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(pi0, [0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError):
        birth_death_stationary([1.0, 1.0], [0.0, 0.0, 1.0])


def test_mm1_stationary():
    lam, mu, n_max = 1.0, 2.0, 30
    pi = mm1_stationary(lam, mu, n_max)
    assert pi.shape == (n_max + 1,)
    np.testing.assert_allclose(pi.sum(), 1.0)
    np.testing.assert_allclose(pi[1:] / pi[:-1], lam / mu)
    np.testing.assert_allclose(pi, (1 - 0.5) * 0.5 ** np.arange(n_max + 1), atol=1e-9)
    np.testing.assert_allclose(
        pi, birth_death_stationary(np.full(n_max, lam), np.full(n_max + 1, mu))
    )


def _erlang_c_reference(lam, mu, c):
    a = lam / mu
    rho = a / c
    top = a**c / factorial(c) / (1 - rho)
    return top / (sum(a**k / factorial(k) for k in range(c)) + top)


@pytest.mark.parametrize(
    ("lam", "mu", "c"), [(1.0, 1.0, 2), (3.0, 1.2, 4), (0.8, 1.0, 1), (10, 3, 5)]
)
def test_mmc_metrics_erlang_c(lam, mu, c):
    m = mmc_metrics(lam, mu, c)
    assert set(m) == {"rho", "p_wait", "Lq", "L", "Wq", "W"}
    rho = lam / (c * mu)
    np.testing.assert_allclose(m["rho"], rho)
    np.testing.assert_allclose(m["p_wait"], _erlang_c_reference(lam, mu, c))
    np.testing.assert_allclose(m["Lq"], m["p_wait"] * rho / (1 - rho))
    np.testing.assert_allclose(m["L"], m["Lq"] + lam / mu)
    np.testing.assert_allclose(m["Wq"], m["Lq"] / lam)  # Little
    np.testing.assert_allclose(m["W"], m["L"] / lam)  # Little
    if c == 1:  # reduces to M/M/1
        np.testing.assert_allclose(m["p_wait"], rho)
        np.testing.assert_allclose(m["L"], rho / (1 - rho))
        np.testing.assert_allclose(m["W"], 1 / (mu - lam))
    if (lam, mu, c) == (1.0, 1.0, 2):  # textbook M/M/2 value Lq = 2 rho^3 / (1 - rho^2)
        np.testing.assert_allclose(m["Lq"], 1 / 3)


def test_mmc_metrics_unstable():
    with pytest.raises(ValueError):
        mmc_metrics(2.0, 1.0, 2)


def test_time_average_occupation_exact():
    times = np.array([0.0, 1.0, 3.0])
    states = np.array([0, 1, 0])
    np.testing.assert_allclose(time_average_occupation(times, states, 4.0, 3), [0.5, 0.5, 0.0])


def test_time_average_occupation_matches_stationary(rng):
    T = 3000.0
    times, states = gillespie(Q2, 0, T, rng)
    np.testing.assert_allclose(
        time_average_occupation(times, states, T, 2), [2 / 3, 1 / 3], atol=0.03
    )
