"""Shared Stage 3 annotation-failure primitives.

Pulled out of ``explain.py`` so both ``explain.py`` (which produces parse
failures) and ``batch.py`` (which handles a whole-chunk ``generate()``
failure with an individual retry) can share them without either importing
the other.
"""

from __future__ import annotations

from typing import Any

PARSE_FAILURE: dict[str, Any] = {
    "mitigating_grounds": [],
    "aggravating_grounds": [],
    "planned_return": None,
    "clinical_justification": "",
    "decision_model": None,
    "decision_rule": None,
    "all_quotes_verified": None,
    "annotation_failed": True,
}


class BatchGenerationError(RuntimeError):
    """A whole-batch ``generate()`` call failed (OOM, kernel/transient serving
    error) -- raised by :func:`src.stage3.explain.call_llm_batch` only when
    ``raise_on_generate_error=True``, so the batch caller can retry each
    patient in the chunk individually (a smaller batch may fit) before
    recording any of them as a permanent annotation failure. The default
    (single-patient / API) path keeps returning a parse-failure dict instead,
    so its contract is unchanged."""


def generation_failure_annotation(detail: str) -> dict[str, Any]:
    """A fresh annotation dict marking a patient's audit as failed after an
    individual generate() retry also failed -- same shape as a parse failure,
    so it flows through _assemble_result into a normal annotation_failed row
    (rather than being silently dropped)."""
    return {**PARSE_FAILURE,
            "clinical_justification": f"[HF generate error after individual retry: {detail}]"}
