"""Batch Stage 3 audit runner.

``src/stage3/pipeline.py``'s ``explain_patient`` is on-demand, single-patient
only (built for the API). This module runs it over every Stage 1-flagged,
note-covered admission in ``models/stage2_results.csv``, preloading the
Stage 1 artifact, Stage 2 results, and feature matrix once (instead of per
request), and writing results incrementally so a slow or interrupted Ollama
run doesn't lose completed work.

This is required, not optional, for two things the on-demand path cannot
produce on its own:

- RQ2 metrics at scale (net reclassification improvement of the auditor
  over Stage 1 alone) — needs a decision for every flagged+noted admission,
  not one at a time.
- ``evaluate_pipeline.py``'s final-prediction column, which per
  docs/ARCHITECTURE.md should be Layer 3's decision, not Stage 2's
  threshold, once this file exists.

Also runs the discordance-threshold sensitivity sweep (``--sweep``) over an
existing batch result — see :func:`src.stage3.explain.sweep_discordance_thresholds`.

Usage::

    python -m src.stage3.batch                  # full flagged+noted cohort
    python -m src.stage3.batch --limit 10        # smoke test
    python -m src.stage3.batch --resume          # skip hadm_ids already written
    python -m src.stage3.batch --sweep           # sensitivity sweep only, no LLM calls
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
from src.stage3.explain import sweep_discordance_thresholds
from src.stage3.pipeline import explain_patient

_OUTPUT_FIELDS = [
    "hadm_id", "stage1_score", "stage1_threshold", "stage2_score",
    "stage2_confirmed", "r1", "r2", "displacement", "discordance_mode",
    "decision", "primary_clinical_domain", "supporting_quote",
    "quote_verified", "planned_return", "clinical_justification",
    "annotation_failed",
]


def _load_artifact(cfg: AppConfig) -> dict:
    """Load the Stage 1 XGBoost artifact — before any torch import (see
    src/stage3/pipeline.py for why this ordering matters on macOS)."""
    model_dir = get_model_dir()
    path = model_dir / f"stage1_{cfg.stage1.model}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 1 artifact not found at {path}. Run python -m src.model.train first."
        )
    return joblib.load(path)


def _load_results(model_dir: Path) -> pd.DataFrame:
    """Load Stage 2's flagged-cohort results (the cascade population)."""
    path = model_dir / "stage2_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run python -m src.stage2.predict first."
        )
    return pd.read_csv(path)


def _already_done(out_path: Path) -> set[int]:
    """Return hadm_ids already present in an existing output CSV, if resuming."""
    if not out_path.exists():
        return set()
    existing = pd.read_csv(out_path, usecols=["hadm_id"])
    return set(existing["hadm_id"].astype(int).tolist())


def run_batch_audit(
    cfg: AppConfig,
    *,
    hadm_ids: list[int] | None = None,
    limit: int | None = None,
    resume: bool = False,
    out_path: Path | None = None,
) -> Path:
    """Run Stage 3 over every Stage 1-flagged, note-covered admission.

    Args:
        cfg:      validated project config.
        hadm_ids: explicit list of admissions to audit. If ``None``, audits
                  every admission in ``models/stage2_results.csv``.
        limit:    audit only the first N admissions (smoke-testing).
        resume:   skip hadm_ids already present in ``out_path`` and append
                  rather than overwrite.
        out_path: output CSV path. Defaults to
                  ``models/stage3_batch_results.csv``.

    Returns:
        The output path written to.
    """
    # Artifact MUST load before torch — see src/stage3/pipeline.py.
    artifact = _load_artifact(cfg)
    model_dir = get_model_dir()
    results_df = _load_results(model_dir)
    feature_matrix = load_feature_matrix(cfg, "full")

    out_path = out_path or (model_dir / "stage3_batch_results.csv")

    targets = hadm_ids if hadm_ids is not None else results_df["hadm_id"].astype(int).tolist()
    if limit is not None:
        targets = targets[:limit]

    done = _already_done(out_path) if resume else set()
    pending = [h for h in targets if h not in done]
    print(
        f"[stage3/batch] {len(targets):,} target admissions, "
        f"{len(done):,} already done, {len(pending):,} pending"
    )

    write_header = not (resume and out_path.exists())
    mode = "a" if resume and out_path.exists() else "w"
    n_ok = n_failed = n_annotation_failed = 0

    with open(out_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        for i, hadm_id in enumerate(pending, start=1):
            try:
                result = explain_patient(
                    hadm_id, cfg,
                    results_df=results_df, artifact=artifact,
                    feature_matrix=feature_matrix,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # One bad admission (missing note, parse failure upstream,
                # transient Ollama error) must not kill the whole batch.
                print(f"[stage3/batch] [{i}/{len(pending)}] hadm_id={hadm_id} FAILED: {exc}")
                n_failed += 1
                continue

            row = result.model_dump()
            writer.writerow({k: row[k] for k in _OUTPUT_FIELDS})
            fh.flush()
            n_ok += 1
            if row["annotation_failed"]:
                n_annotation_failed += 1
            if i % 25 == 0 or i == len(pending):
                print(f"[stage3/batch] [{i}/{len(pending)}] "
                      f"ok={n_ok} failed={n_failed} annotation_failed={n_annotation_failed}")

    print(f"[stage3/batch] Done. Saved -> {out_path} "
          f"(ok={n_ok}, failed={n_failed}, annotation_failed={n_annotation_failed})")
    return out_path


def run_sensitivity_sweep(
    model_dir: Path | None = None,
    thresholds_pp: list[float] | None = None,
) -> dict:
    """Sweep the discordance-mode threshold over an existing batch result.

    Requires ``stage3_batch_results.csv`` (i.e. :func:`run_batch_audit` must
    have been run first) — makes no LLM calls itself. Saves
    ``models/discordance_sensitivity.json``.
    """
    model_dir = model_dir or get_model_dir()
    results_path = model_dir / "stage3_batch_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run `python -m src.stage3.batch` first."
        )
    df = pd.read_csv(results_path)
    sweep = sweep_discordance_thresholds(df["displacement"].to_numpy(), thresholds_pp)

    out_path = model_dir / "discordance_sensitivity.json"
    out_path.write_text(json.dumps({
        "n_patients": int(len(df)),
        "current_config_value_pp": 20.0,
        "sweep": sweep,
    }, indent=2))
    print(f"[stage3/batch] Sensitivity sweep saved -> {out_path}")
    for thr, dist in sweep.items():
        print(f"  ±{thr}pp: MITIGATES={dist['NOTE_MITIGATES']:.1%}  "
              f"AMPLIFIES={dist['NOTE_AMPLIFIES']:.1%}  "
              f"CONCORDANT={dist['CONCORDANT']:.1%}")
    return sweep


def main() -> None:
    """CLI entry point for the batch Stage 3 audit."""
    parser = argparse.ArgumentParser(
        description="Batch Stage 3 audit over flagged+noted admissions"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Audit only the first N admissions"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip hadm_ids already in the output CSV"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the discordance-threshold sensitivity sweep over an "
             "existing batch result instead of auditing (no LLM calls)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.sweep:
        run_sensitivity_sweep()
    else:
        run_batch_audit(cfg, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
