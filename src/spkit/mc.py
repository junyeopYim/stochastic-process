"""몬테카를로 추정치의 기록·비교·검증 도구.

Bookkeeping helpers used throughout the notebooks: a result record carrying a
standard error and 95% CI (:class:`MCResult`), estimators from raw samples or
Bernoulli outcomes, a Markdown comparison table and an assertion helper.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class MCResult:
    """몬테카를로 추정치(평균, 표준오차, 표본 수).

    Attributes
    ----------
    mean : float
        Sample mean.
    se : float
        Standard error of the mean.
    n : int
        Number of independent samples behind the estimate.
    """

    mean: float
    se: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        """95% 신뢰구간 ``mean ± 1.96 se``."""
        half = 1.96 * self.se
        return (self.mean - half, self.mean + half)

    def __format__(self, spec: str) -> str:
        spec = spec or ".4f"
        return f"{self.mean:{spec}} ± {self.se:{spec}} (n={self.n})"

    def __str__(self) -> str:
        return format(self)


def mc_estimate(samples: ArrayLike) -> MCResult:
    """표본 평균과 표준오차를 계산한다.

    Parameters
    ----------
    samples : array_like
        Independent samples (flattened).

    Returns
    -------
    MCResult
        ``mean`` and ``se = std(ddof=1) / sqrt(n)``.
    """
    x = np.asarray(samples, dtype=float).ravel()
    n = x.size
    if n < 2:
        msg = "mc_estimate needs at least two samples"
        raise ValueError(msg)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n)


def mc_proportion(successes: int | ArrayLike, n: int | None = None) -> MCResult:
    """베르누이 성공 비율과 표준오차를 계산한다.

    Parameters
    ----------
    successes : int or array_like of bool
        Either the number of successes (then ``n`` is required) or a boolean
        array of outcomes.
    n : int, optional
        Number of trials when ``successes`` is a count.

    Returns
    -------
    MCResult
        ``mean = p`` and ``se = max(sqrt(p (1 - p) / n), 1 / n)``; the floor keeps
        the error bar informative when ``p`` is 0 or 1.
    """
    if n is None:
        if np.ndim(successes) == 0:
            msg = "n is required when successes is a count"
            raise ValueError(msg)
        outcomes = np.asarray(successes, dtype=bool).ravel()
        n = outcomes.size
        k = int(outcomes.sum())
    else:
        k = int(np.asarray(successes).sum()) if np.ndim(successes) else int(successes)
    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    p = k / n
    se = max(float(np.sqrt(p * (1.0 - p) / n)), 1.0 / n)
    return MCResult(p, se, n)


def _verdict(est: float, exact: float | None, se: float | None) -> str:
    if exact is None:
        return "—"
    if se is None or se <= 0:
        return "ok" if np.isclose(est, exact) else "investigate (no SE)"
    z = abs(est - exact) / se
    return "ok" if z <= 3.0 else f"investigate ({z:.1f} SE)"


def compare_table(rows: list[dict]) -> str:
    """예측·MC 추정·닫힌 형식을 나란히 놓는 Markdown 표를 만든다.

    Parameters
    ----------
    rows : list of dict
        Each row has ``name`` (str), ``predicted`` (float or None), ``estimate``
        (:class:`MCResult` or float), ``exact`` (float or None) and, when
        ``estimate`` is a float, optionally ``se``.

    Returns
    -------
    str
        Markdown table with columns ``항목 | 내 예측 | MC 추정 (95% CI) | 닫힌 형식 | 판정``.
        판정 is ``"—"`` without an exact value, ``"ok"`` when the estimate is
        within 3 SE of it and ``"investigate (z SE)"`` otherwise. Nothing is
        printed; wrap the result in ``IPython.display.Markdown``.
    """
    lines = [
        "| 항목 | 내 예측 | MC 추정 (95% CI) | 닫힌 형식 | 판정 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        estimate = row["estimate"]
        if isinstance(estimate, MCResult):
            est, se = estimate.mean, estimate.se
        else:
            est, se = float(estimate), row.get("se")
        predicted = row.get("predicted")
        exact = row.get("exact")
        pred_s = "—" if predicted is None else f"{predicted:.4f}"
        if se is None:
            est_s = f"{est:.4f}"
        else:
            est_s = f"{est:.4f} [{est - 1.96 * se:.4f}, {est + 1.96 * se:.4f}]"
        exact_s = "—" if exact is None else f"{exact:.4f}"
        lines.append(
            f"| {row['name']} | {pred_s} | {est_s} | {exact_s} | {_verdict(est, exact, se)} |"
        )
    return "\n".join(lines)


def assert_close(
    estimate: MCResult | float,
    exact: float,
    se: float | None = None,
    k: float = 4.0,
    name: str = "",
) -> None:
    """추정치가 닫힌 형식과 k·SE 안에 있는지 검사한다.

    Parameters
    ----------
    estimate : MCResult or float
        Monte Carlo estimate. For an :class:`MCResult`, ``se`` defaults to its
        standard error.
    exact : float
        Reference value.
    se : float, optional
        Standard error to use; required when ``estimate`` is a plain float.
    k : float, default 4.0
        Tolerance in units of SE.
    name : str, default ""
        Label used in the error message.

    Raises
    ------
    AssertionError
        If ``|estimate - exact| > k * se``.
    """
    if isinstance(estimate, MCResult):
        est = estimate.mean
        if se is None:
            se = estimate.se
    else:
        est = float(estimate)
    if se is None or se <= 0:
        msg = "assert_close needs a positive standard error"
        raise ValueError(msg)
    z = abs(est - exact) / se
    if z > k:
        label = f"{name}: " if name else ""
        msg = f"{label}estimate {est:.6g} vs exact {exact:.6g} differs by {z:.2f} SE (> {k:g} SE, se={se:.3g})"
        raise AssertionError(msg)
