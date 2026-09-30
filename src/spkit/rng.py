"""난수 시드 규칙과 빠른 실행 모드 스위치.

Seed conventions shared by all notebooks (:func:`seed_for`), the ``SPKIT_FAST``
switch used to shrink Monte Carlo sizes in CI (:func:`fast`, :func:`scale`) and
the canonical way to obtain a :class:`numpy.random.Generator` (:func:`rng_for`).
"""

import os
import re

import numpy as np

_MODULE_ID = re.compile(r"^(\d+)([A-Za-z])$")
_CAPSTONE_ID = re.compile(r"^[Cc](\d+)$")


def seed_for(nb_id: str) -> int:
    """노트북 id를 정수 시드로 바꾸는 규칙.

    Module notebooks ``"MMx"`` (module number, one letter) map to
    ``100 * MM + (letter index, a = 1)``: ``"00a" -> 1``, ``"01d" -> 104``,
    ``"01x" -> 124``, ``"07e" -> 705``. Capstones ``"C<k>"`` map to ``900 + k``
    (``"C1" -> 901``).

    Parameters
    ----------
    nb_id : str
        Notebook identifier such as ``"01r"`` or ``"C5"``.

    Returns
    -------
    int
        Deterministic seed for that notebook.

    Raises
    ------
    ValueError
        If ``nb_id`` follows neither pattern.
    """
    nb_id = nb_id.strip()
    if m := _CAPSTONE_ID.match(nb_id):
        return 900 + int(m.group(1))
    if m := _MODULE_ID.match(nb_id):
        module, letter = m.groups()
        return 100 * int(module) + (ord(letter.lower()) - ord("a") + 1)
    msg = f"unrecognised notebook id {nb_id!r}; expected e.g. '01d' or 'C1'"
    raise ValueError(msg)


def fast() -> bool:
    """SPKIT_FAST 환경변수가 켜져 있는지 확인.

    Returns
    -------
    bool
        ``True`` iff the environment variable ``SPKIT_FAST`` is set to a
        non-empty value.
    """
    return bool(os.environ.get("SPKIT_FAST", ""))


def scale(n: int, divisor: int = 10, minimum: int = 200) -> int:
    """빠른 모드에서 표본 수를 줄이는 헬퍼.

    Parameters
    ----------
    n : int
        Sample size the notebook wants in full mode.
    divisor : int, default 10
        Reduction factor applied in fast mode.
    minimum : int, default 200
        Floor for the reduced size.

    Returns
    -------
    int
        ``n`` when :func:`fast` is ``False``, otherwise ``max(minimum, n // divisor)``.
    """
    if not fast():
        return n
    return max(minimum, n // divisor)


def rng_for(nb_id: str) -> tuple[int, np.random.Generator]:
    """노트북 id로부터 (시드, Generator) 쌍을 만든다.

    Parameters
    ----------
    nb_id : str
        Notebook identifier, see :func:`seed_for`.

    Returns
    -------
    tuple[int, numpy.random.Generator]
        ``(SEED, np.random.default_rng(SEED))``.
    """
    seed = seed_for(nb_id)
    return seed, np.random.default_rng(seed)
