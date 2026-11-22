"""Guard: every JSON present in BOTH models/ and results/ must be byte-identical.

`models/*.json` are the canonical pipeline outputs; `results/` holds a curated,
committed copy of the thesis-relevant subset. A copy must never drift from its
source (see results/README.md). This fails loudly if one does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PAIRS = sorted(
    (p.name for p in (_ROOT / "results").glob("*.json")
     if (_ROOT / "models" / p.name).exists())
)


def test_some_curated_copies_exist():
    """Sanity: there is at least one results/ copy to check (catches a broken glob)."""
    assert _PAIRS, "no JSON present in both models/ and results/ -- check the folders"


@pytest.mark.parametrize("name", _PAIRS)
def test_results_copy_is_byte_identical_to_models(name):
    """results/<name> must equal models/<name> byte-for-byte."""
    results_bytes = (_ROOT / "results" / name).read_bytes()
    models_bytes = (_ROOT / "models" / name).read_bytes()
    assert results_bytes == models_bytes, (
        f"results/{name} differs from models/{name}. The curated copy drifted; "
        f"re-copy with `cp models/{name} results/{name}` (never hand-edit results/)."
    )
