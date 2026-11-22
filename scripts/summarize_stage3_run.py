"""Aggregate-only summary of a completed Stage 3 batch run.

Reads ``models/stage3_batch_results.csv`` and its ``.raw.jsonl`` sidecar and
writes ``models/stage3_run_summary.json`` -- counts and shares only, never a
hadm_id/subject_id or any note text, so the summary is safe to commit under the
MIMIC-IV-Note DUA. (The CSV and the sidecar themselves carry identifiers/notes
and stay gitignored; this script only emits aggregates derived from them.)

Reports:

* ``decision_model`` and ``decision_rule`` value counts;
* agreement between ``decision_model`` and ``decision_rule`` among admissions
  the model actually decided (uphold/override);
* share of decided admissions with ``all_quotes_verified``;
* ``note_truncated`` count and share;
* ``planned_return`` distribution;
* counts of invalid LLM answers, insufficient_evidence, and prep failures;
* from the raw sidecar, how many raw answers have JSON keys out of the schema
  field order (``_generation_schema()``), and which field was out of place;
* optional ``--logs GLOB``: how many batches failed with "CUDA out of memory"
  and were retried per patient (counted from job logs; the matching lines carry
  hadm_ids, so only the COUNT is kept, never the lines).

CPU only, no model load. Usage::

    python -m scripts.summarize_stage3_run
    python -m scripts.summarize_stage3_run --logs 'logs/stage3_*.log'
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

from src.config import get_model_dir
from src.stage3.explain import _generation_schema

_DECIDED = ("uphold", "override")
# One retried-batch log line from _process_chunk, failing with CUDA OOM. The
# line also contains hadm_ids, so it is only ever COUNTED, never stored.
_OOM_RETRY_RE = re.compile(
    r"FAILED:.*(?:CUDA out of memory|out of memory).*retry.*individually", re.IGNORECASE
)


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV column to bool: True/'True' -> True, everything else
    (False, NaN, '') -> False."""
    return series.astype(str).str.lower().eq("true")


def _counts(series: pd.Series) -> dict[str, int]:
    """value_counts as a plain str->int dict, missing values keyed 'missing'."""
    vc = series.value_counts(dropna=False)
    return {("missing" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}


def schema_field_order() -> list[str]:
    """The expected generation-schema key order (N1/N3 forced order)."""
    return list(_generation_schema()["properties"].keys())


def summarize_csv(df: pd.DataFrame) -> dict:
    """All CSV-derived aggregates (no identifiers emitted)."""
    n = len(df)
    decided_mask = df["decision_model"].isin(_DECIDED)
    n_decided = int(decided_mask.sum())
    agree = df.loc[decided_mask, "decision_model"].eq(df.loc[decided_mask, "decision_rule"])
    quotes = _as_bool(df.loc[decided_mask, "all_quotes_verified"]) if n_decided else pd.Series([])
    truncated = _as_bool(df["note_truncated"])
    is_prep_failed = df.get("discordance_mode", pd.Series(dtype=object)).eq("PREP_FAILED")
    annotation_failed = _as_bool(df["annotation_failed"])
    return {
        "n_rows": n,
        "decision_model_counts": _counts(df["decision_model"]),
        "decision_rule_counts": _counts(df["decision_rule"]),
        "n_decided": n_decided,
        "model_rule_agreement_among_decided": round(float(agree.mean()), 5) if n_decided else None,
        "all_quotes_verified_share_among_decided":
            round(float(quotes.mean()), 5) if n_decided else None,
        "note_truncated_count": int(truncated.sum()),
        "note_truncated_share": round(float(truncated.mean()), 5) if n else None,
        "planned_return_distribution": _counts(df["planned_return"]),
        "n_insufficient_evidence": int(df["decision_model"].eq("insufficient_evidence").sum()),
        "n_prep_failed": int(is_prep_failed.sum()),
        "n_invalid_llm_answers": int((annotation_failed & ~is_prep_failed).sum()),
    }


def _key_order_issue(keys: list[str], expected: list[str]) -> tuple[str | None, bool]:
    """For one raw answer's key list: return (first_out_of_place_field,
    has_unexpected_key). The out-of-place field is the first position where the
    schema-ordered keys present disagree with the raw order."""
    unexpected = any(k not in expected for k in keys)
    raw_known = [k for k in keys if k in expected]
    want = [k for k in expected if k in raw_known]
    out_of_place = next((w for w, r in zip(want, raw_known) if w != r), None)
    return out_of_place, unexpected


def summarize_raw(raw_path: Path, expected: list[str]) -> dict:
    """Raw-sidecar key-order analysis (counts only)."""
    if not raw_path.exists():
        return {"raw_sidecar_present": False}
    n_lines = n_parse_fail = n_out_of_order = n_unexpected_keys = 0
    out_of_place_field_counts: dict[str, int] = {}
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                answer = json.loads(json.loads(line)["raw_response"])
                keys = list(answer.keys())
            except (ValueError, KeyError, TypeError, AttributeError):
                n_parse_fail += 1
                continue
            field, unexpected = _key_order_issue(keys, expected)
            if unexpected:
                n_unexpected_keys += 1
            if field is not None:
                n_out_of_order += 1
                out_of_place_field_counts[field] = out_of_place_field_counts.get(field, 0) + 1
    return {
        "raw_sidecar_present": True,
        "n_raw_lines": n_lines,
        "n_raw_unparseable": n_parse_fail,
        "n_keys_out_of_schema_order": n_out_of_order,
        "out_of_place_field_counts": out_of_place_field_counts,
        "n_answers_with_unexpected_keys": n_unexpected_keys,
    }


def count_oom_retries(log_glob: str) -> dict:
    """Count batches that failed with CUDA OOM and were retried per patient,
    across every log file matching ``log_glob``. Only the count is kept -- the
    matching lines contain hadm_ids."""
    paths = sorted(glob.glob(log_glob))
    n_oom = 0
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _OOM_RETRY_RE.search(line):
                    n_oom += 1
    return {"log_files_scanned": len(paths), "oom_retried_batches": n_oom}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Aggregate summary of a Stage 3 batch run")
    parser.add_argument("--results", type=Path, default=None,
                        help="stage3_batch_results.csv (default: models/)")
    parser.add_argument("--logs", type=str, default=None,
                        help="glob of job logs to count CUDA-OOM per-patient retries")
    args = parser.parse_args()

    model_dir = get_model_dir()
    results_path = args.results or (model_dir / "stage3_batch_results.csv")
    df = pd.read_csv(results_path)
    raw_path = results_path.with_name(results_path.stem + ".raw.jsonl")

    summary = summarize_csv(df)
    summary["raw"] = summarize_raw(raw_path, schema_field_order())
    if args.logs:
        summary["oom_retries"] = count_oom_retries(args.logs)

    out_path = model_dir / "stage3_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[summarize_stage3_run] Saved -> {out_path}")


if __name__ == "__main__":
    main()
