import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed generator shared by every test module."""
    return np.random.default_rng(12345)
