import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from scipy.stats import norm, poisson

from spkit.brownian import brownian_paths
from spkit.plots import plot_hist_vs_pdf, plot_hist_vs_pmf, plot_paths, setup_plots


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_setup_plots_is_idempotent():
    setup_plots()
    setup_plots()
    assert tuple(mpl.rcParams["figure.figsize"]) == (7.0, 4.0)
    assert mpl.rcParams["figure.dpi"] == 110
    assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["grid.alpha"] == 0.3
    assert mpl.rcParams["font.family"] == ["DejaVu Sans"]
    assert mpl.rcParams["axes.unicode_minus"] is False
    assert mpl.rcParams["legend.frameon"] is False


def test_plot_paths(rng):
    t, B = brownian_paths(12, 20, 1.0, rng)
    ax = plot_paths(t, B, n_show=5, color="k")
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 5
    assert ax.lines[0].get_color() == "k"
    fig, ax2 = plt.subplots()
    out = plot_paths(t, B, ax=ax2, step=True)
    assert out is ax2 and len(ax2.lines) == 10
    assert ax2.lines[0].get_drawstyle() == "steps-post"
    ax3 = plot_paths(t, B[0])
    assert len(ax3.lines) == 1


def test_plot_hist_vs_pmf(rng):
    samples = rng.poisson(3.0, 500)
    support = np.arange(0, 12)
    ax = plot_hist_vs_pmf(samples, support, poisson.pmf(support, 3.0), label_pmf="Poisson(3)")
    assert isinstance(ax, Axes)
    assert len(ax.patches) >= 1
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Poisson(3)" in labels and "simulation" in labels


def test_plot_hist_vs_pdf(rng):
    samples = rng.standard_normal(500)
    x = np.linspace(-4, 4, 200)
    fig, ax0 = plt.subplots()
    ax = plot_hist_vs_pdf(samples, x, norm.pdf(x), ax=ax0, bins=30)
    assert ax is ax0
    assert len(ax.patches) == 30
    assert len(ax.lines) == 1
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "exact pdf" in labels
