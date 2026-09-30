"""노트북 공용 그리기 헬퍼 (matplotlib).

Consistent rcParams (:func:`setup_plots`) and three small plotting helpers:
sample paths, histogram versus an exact pmf, histogram versus an exact pdf.
Every helper returns the :class:`matplotlib.axes.Axes` it drew on.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

_RC = {
    "figure.figsize": (7, 4),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "legend.frameon": False,
}


def setup_plots() -> None:
    """노트북 공통 matplotlib 스타일을 적용한다 (여러 번 불러도 안전).

    Sets ``figure.figsize=(7, 4)``, ``figure.dpi=110``, light grid, DejaVu Sans,
    ASCII minus sign and frameless legends.
    """
    mpl.rcParams.update(_RC)


def plot_paths(
    t: np.ndarray,
    X: np.ndarray,
    ax: Axes | None = None,
    n_show: int = 10,
    alpha: float = 0.8,
    lw: float = 1.0,
    **kw,
) -> Axes:
    """표본 경로 몇 개를 t 에 대해 그린다.

    Parameters
    ----------
    t : numpy.ndarray, shape (n_steps + 1,)
        Time grid.
    X : numpy.ndarray, shape (n_paths, n_steps + 1)
        Paths (a 1-D array is one path).
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    n_show : int, default 10
        Number of rows of ``X`` to draw.
    alpha, lw : float
        Line transparency and width.
    **kw
        Extra keyword arguments for ``ax.plot``. ``step=True`` draws with
        ``ax.step(..., where="post")`` for jump processes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    step = bool(kw.pop("step", False))
    t = np.asarray(t)
    X = np.atleast_2d(np.asarray(X))
    if ax is None:
        _, ax = plt.subplots()
    for row in X[:n_show]:
        if step:
            ax.step(t, row, where="post", alpha=alpha, lw=lw, **kw)
        else:
            ax.plot(t, row, alpha=alpha, lw=lw, **kw)
    ax.set_xlabel("t")
    return ax


def plot_hist_vs_pmf(
    samples: np.ndarray,
    pmf_support: np.ndarray,
    pmf_values: np.ndarray,
    ax: Axes | None = None,
    label_pmf: str = "exact pmf",
) -> Axes:
    """정수 표본의 히스토그램 위에 정확한 pmf 를 겹쳐 그린다.

    Parameters
    ----------
    samples : array_like
        Integer-valued samples.
    pmf_support, pmf_values : numpy.ndarray
        Support points and probabilities of the exact pmf.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    label_pmf : str
        Legend label for the pmf stems.

    Returns
    -------
    matplotlib.axes.Axes
    """
    samples = np.asarray(samples).ravel()
    support = np.asarray(pmf_support)
    values = np.asarray(pmf_values, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    lo = min(samples.min(), support.min())
    hi = max(samples.max(), support.max())
    bins = np.arange(lo - 0.5, hi + 1.5)
    ax.hist(samples, bins=bins, density=True, alpha=0.5, label="simulation")
    ax.stem(support, values, linefmt="C1-", markerfmt="C1o", basefmt=" ", label=label_pmf)
    ax.legend()
    return ax


def plot_hist_vs_pdf(
    samples: np.ndarray,
    x: np.ndarray,
    pdf: np.ndarray,
    ax: Axes | None = None,
    bins: int = 50,
    label_pdf: str = "exact pdf",
) -> Axes:
    """연속 표본의 밀도 히스토그램 위에 정확한 pdf 곡선을 겹쳐 그린다.

    Parameters
    ----------
    samples : array_like
        Samples.
    x, pdf : numpy.ndarray
        Curve of the exact density.
    ax : matplotlib.axes.Axes, optional
        Target axes.
    bins : int, default 50
        Histogram bins.
    label_pdf : str
        Legend label for the curve.

    Returns
    -------
    matplotlib.axes.Axes
    """
    samples = np.asarray(samples).ravel()
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(samples, bins=bins, density=True, alpha=0.5, label="simulation")
    ax.plot(np.asarray(x), np.asarray(pdf), color="C1", lw=2, label=label_pdf)
    ax.legend()
    return ax
