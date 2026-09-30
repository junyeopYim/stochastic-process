import numpy as np
import pytest

import spkit
from spkit import fast, rng_for, scale, seed_for


@pytest.mark.parametrize(
    ("nb_id", "seed"),
    [("00a", 1), ("01d", 104), ("01r", 118), ("01x", 124), ("07e", 705), ("C1", 901), ("C5", 905)],
)
def test_seed_for(nb_id, seed):
    assert seed_for(nb_id) == seed


@pytest.mark.parametrize("bad", ["", "abc", "1", "C", "01", "x01"])
def test_seed_for_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        seed_for(bad)


def test_fast_reads_env(monkeypatch):
    monkeypatch.delenv("SPKIT_FAST", raising=False)
    assert fast() is False
    monkeypatch.setenv("SPKIT_FAST", "")
    assert fast() is False
    monkeypatch.setenv("SPKIT_FAST", "1")
    assert fast() is True


def test_scale(monkeypatch):
    monkeypatch.delenv("SPKIT_FAST", raising=False)
    assert scale(50_000) == 50_000
    monkeypatch.setenv("SPKIT_FAST", "1")
    assert scale(50_000) == 5_000
    assert scale(50_000, divisor=100) == 500
    assert scale(1_000) == 200
    assert scale(1_000, minimum=10) == 100


def test_rng_for_is_reproducible():
    seed, gen = rng_for("01d")
    assert seed == 104
    assert isinstance(gen, np.random.Generator)
    _, gen2 = rng_for("01d")
    np.testing.assert_array_equal(gen.random(5), gen2.random(5))


def test_package_reexports():
    assert spkit.seed_for is spkit.rng.seed_for
    assert spkit.rng_for is spkit.rng.rng_for
    assert spkit.fast is spkit.rng.fast
    assert spkit.scale is spkit.rng.scale
    assert spkit.setup_plots is spkit.plots.setup_plots
    for name in ("mc", "markov", "poisson", "ctmc", "renewal", "brownian", "sde", "plots", "rng"):
        assert hasattr(spkit, name)
        assert name in spkit.__all__
