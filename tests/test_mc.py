import numpy as np
import pytest

from spkit.mc import MCResult, assert_close, compare_table, mc_estimate, mc_proportion


def test_mcresult_ci_and_format():
    r = MCResult(0.2503, 0.0021, 20000)
    np.testing.assert_allclose(r.ci95, (0.2503 - 1.96 * 0.0021, 0.2503 + 1.96 * 0.0021))
    assert str(r) == "0.2503 ± 0.0021 (n=20000)"
    assert f"{r}" == "0.2503 ± 0.0021 (n=20000)"
    assert f"{r:.2f}" == "0.25 ± 0.00 (n=20000)"


def test_mc_estimate_exact():
    r = mc_estimate([1.0, 2.0, 3.0, 4.0])
    assert r.n == 4
    np.testing.assert_allclose(r.mean, 2.5)
    np.testing.assert_allclose(r.se, np.sqrt(5.0 / 3.0) / 2.0)
    with pytest.raises(ValueError):
        mc_estimate([1.0])


def test_mc_proportion():
    outcomes = np.tile([True, False, True, True], 100)
    r = mc_proportion(outcomes)
    assert (r.mean, r.n) == (0.75, 400)
    np.testing.assert_allclose(r.se, np.sqrt(0.75 * 0.25 / 400))
    r2 = mc_proportion(300, 400)
    assert (r2.mean, r2.se, r2.n) == (r.mean, r.se, r.n)
    # tiny n: the 1/n floor dominates sqrt(p (1 - p) / n)
    assert mc_proportion(3, 4).se == 0.25
    r0 = mc_proportion(0, 100)
    assert r0.mean == 0.0
    assert r0.se == 0.01  # floor 1/n keeps the error bar informative
    with pytest.raises(ValueError):
        mc_proportion(3)


def test_compare_table():
    rows = [
        {"name": "P(A)", "predicted": 0.5, "estimate": MCResult(0.51, 0.01, 100), "exact": 0.5},
        {"name": "E[X]", "predicted": None, "estimate": 0.7, "se": 0.01, "exact": 0.5},
        {"name": "raw", "predicted": 1.0, "estimate": 0.9, "exact": None},
    ]
    table = compare_table(rows)
    lines = table.splitlines()
    assert lines[0] == "| 항목 | 내 예측 | MC 추정 (95% CI) | 닫힌 형식 | 판정 |"
    assert len(lines) == 5
    assert "| P(A) | 0.5000 | 0.5100 [0.4904, 0.5296] | 0.5000 | ok |" in lines
    assert "investigate (20.0 SE)" in lines[3]
    assert lines[3].startswith("| E[X] | — |")
    assert lines[4] == "| raw | 1.0000 | 0.9000 | — | — |"


def test_assert_close():
    assert_close(MCResult(0.52, 0.01, 100), 0.5)
    assert_close(0.52, 0.5, se=0.01)
    with pytest.raises(AssertionError, match="hit prob"):
        assert_close(MCResult(0.6, 0.01, 100), 0.5, name="hit prob")
    with pytest.raises(AssertionError):
        assert_close(0.6, 0.5, se=0.01, k=2.0)
    with pytest.raises(ValueError):
        assert_close(0.6, 0.5)
