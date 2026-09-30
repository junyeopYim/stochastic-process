"""spkit: 확률 과정 과정용 공용 시뮬레이션 헬퍼.

Shared helpers for the stochastic-processes notebook course. Every simulation
takes an explicit :class:`numpy.random.Generator` (``rng``); use
:func:`rng_for` to obtain the notebook's canonical one.

Submodules
----------
mc, markov, poisson, ctmc, renewal, brownian, sde, plots, rng
"""

from . import brownian, ctmc, markov, mc, plots, poisson, renewal, rng, sde
from .plots import setup_plots
from .rng import fast, rng_for, scale, seed_for

__all__ = [
    "brownian",
    "ctmc",
    "fast",
    "markov",
    "mc",
    "plots",
    "poisson",
    "renewal",
    "rng",
    "rng_for",
    "scale",
    "sde",
    "seed_for",
    "setup_plots",
]
