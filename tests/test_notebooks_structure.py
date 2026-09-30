"""노트북 템플릿 준수 검사 (course linter).

모든 노트북이 같은 골격을 따르는지 확인한다. 실행 검사는 CI의 `pytest --nbmake`가 맡는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))
NOTEBOOKS = [p for p in NOTEBOOKS if ".ipynb_checkpoints" not in p.parts]

ID_RE = re.compile(r"^(?P<id>\d\d[a-z]|C\d)_[a-z0-9_]+\.ipynb$")

REQUIRED_HEADINGS = {
    "core": ["Recall", "이론", "Predict", "Simulate", "Compare", "연습문제", "정리"],
    "review": ["닫힌 책", "재유도", "자기채점", "정리"],
    "capstone": ["합격 기준", "단계", "Compare", "정리"],
}

FORBIDDEN_CODE = [
    (re.compile(r"np\.random\.seed\("), "np.random.seed 대신 rng_for()를 쓰세요"),
    (re.compile(r"RandomState\("), "RandomState 대신 numpy Generator를 쓰세요"),
    (
        re.compile(r"np\.random\.(rand|randn|randint|choice|normal|uniform)\("),
        "레거시 np.random.* 대신 rng.* 를 쓰세요",
    ),
]


def _kind(nb_id: str) -> str:
    if nb_id.startswith("C"):
        return "capstone"
    if nb_id.endswith("r"):
        return "review"
    return "core"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(nb: dict, cell_type: str) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == cell_type]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_follows_template(path: Path) -> None:
    m = ID_RE.match(path.name)
    assert m, f"파일명 규칙 위반: {path.name} (예: 01d_stationary.ipynb, C1_pagerank.ipynb)"
    nb_id = m.group("id")
    nb = _load(path)

    md = _cells(nb, "markdown")
    code = _cells(nb, "code")
    assert md and code, "마크다운 셀과 코드 셀이 모두 있어야 합니다"

    first = md[0].lstrip()
    assert first.startswith(f"# {nb_id}"), f"첫 셀은 '# {nb_id} · 제목' 헤더여야 합니다"

    headings = "\n".join(
        line for cell in md for line in cell.splitlines() if line.startswith("## ")
    )
    for key in REQUIRED_HEADINGS[_kind(nb_id)]:
        assert key in headings, f"필수 섹션 헤딩 누락: '{key}'"

    all_code = "\n".join(code)
    assert f'rng_for("{nb_id}")' in all_code, f'설정 셀에 rng_for("{nb_id}") 가 있어야 합니다'
    for pattern, msg in FORBIDDEN_CODE:
        assert not pattern.search(all_code), msg

    if _kind(nb_id) == "core":
        assert "predictions" in all_code, "Predict 섹션의 predictions 딕셔너리가 없습니다"
        assert "compare_table(" in all_code, "Compare 섹션에서 compare_table을 써야 합니다"
        assert "assert_close(" in all_code, "Compare 섹션 끝에 assert_close가 있어야 합니다"
        assert "<details>" in "\n".join(md), "Recall/연습문제 정답은 <details>로 숨겨야 합니다"


def test_seed_rule_matches_filename() -> None:
    from spkit import seed_for

    assert seed_for("00a") == 1
    assert seed_for("01d") == 104
    assert seed_for("01r") == 118
    assert seed_for("01x") == 124
    assert seed_for("C1") == 901
