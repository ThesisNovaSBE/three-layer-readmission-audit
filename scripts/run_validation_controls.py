"""Stage 3 validation controls (D1 blind-note, D2 no-Stage-2, D3
self-agreement) on a deliberate sample of the completed batch run.

Session 19 added ``run_blind_note_control`` / ``run_no_stage2_control`` /
``check_self_agreement`` (``src/stage3/batch.py``) as diagnostics, callable
directly but never wired to a CLI entry point or actually run at scale --
this is that entry point. Per those functions' own docstrings, this is a
diagnostic on a deliberate sample (tens to low hundreds), not a second
full batch run.

D1 (blind-note) is described in the code itself as "the most important
single check we run": if ``decision_model`` barely changes with the note
withheld, the auditor isn't actually reading it.

Runs every control through the SAME batched generation path as the main
run (``src.stage3.batch._process_chunk``, chunked by
``cfg.stage3.generation_batch_size`` with left padding) -- not one patient
at a time. An earlier version called ``explain_patient`` per admission in
a loop; confirmed 2026-11 that this is itself a confound, since bfloat16
batched vs. single-sequence generation can produce slightly different
logits, so a single-patient control's decision is not guaranteed identical
to what the same patient would get inside the real batched run. Batching
the controls makes the manipulated input (note withheld / Stage 2
withheld / nothing withheld) the only difference from the main run.

Writes incrementally, crash-safe and resumable, same pattern as
``run_batch_audit``. There is no --force: before rerunning these controls
against a fresh batch run (e.g. after a methodology fix upstream),
manually move the old stage3_control_*.csv files out of models/, or every
hadm_id will be "already done" and nothing recomputes.

Usage::

    python -m scripts.run_validation_controls --n 150
    python -m scripts.run_validation_controls --n 150 --controls d1
    python -m scripts.run_validation_controls --arm-test 8   # see below
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import get_model_dir, load_config
from src.data.features import load_feature_matrix
from src.stage3.batch import (
    _OUTPUT_FIELDS,
    _BatchSetup,
    _already_done,
    _load_artifact,
    _load_results,
    _preload_notes,
    _process_chunk,
)

# Imported from batch.py, not duplicated -- a second copy of this list
# drifting out of sync with the real one is exactly how the controls'
# output schema could silently miss a field (e.g. top_shap_features/
# attention_sentences) that the main run has.


def _select_sample(batch_results: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Reproducible random sample of admissions the real batch run
    actually scored (excludes annotation_failed -- only those have a real
    decision to compare controls against)."""
    valid = batch_results[~batch_results["annotation_failed"]]
    n = min(n, len(valid))
    return valid.sample(n=n, random_state=seed)["hadm_id"].astype(int).tolist()


def _run_batched_control(
    name: str,
    hadm_ids: list[int],
    cfg,
    out_path: Path,
    setup: _BatchSetup,
    *,
    suppress_note: bool = False,
    suppress_stage2: bool = False,
) -> pd.DataFrame:
    """Run one control over hadm_ids through the main run's own batched
    chunk processor (``_process_chunk``), writing incrementally (crash-safe,
    resumable). ``setup`` is the same bundle the real run passes."""
    done = _already_done(out_path)
    pending = [h for h in hadm_ids if h not in done]
    print(f"[validation_controls] {name}: {len(hadm_ids)} target, "
          f"{len(done)} already done, {len(pending)} pending", flush=True)

    if pending:
        import csv  # pylint: disable=import-outside-toplevel
        chunk_size = max(1, cfg.stage3.generation_batch_size)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
            if write_header:
                writer.writeheader()
            for i in range(0, len(pending), chunk_size):
                chunk = pending[i:i + chunk_size]
                _process_chunk(
                    chunk, cfg, setup, writer=writer, fh=fh,
                    suppress_note=suppress_note, suppress_stage2=suppress_stage2,
                )
                done_n = min(i + chunk_size, len(pending))
                print(f"[validation_controls] {name} [{done_n}/{len(pending)}]", flush=True)

    return pd.read_csv(out_path)


def _agreement_report(real: pd.DataFrame, control: pd.DataFrame, label: str) -> dict:
    """Compare a control run's decision_model against the real batch
    decision for the same admissions."""
    merged = real[["hadm_id", "decision_model"]].merge(
        control[["hadm_id", "decision_model"]], on="hadm_id",
        suffixes=("_real", f"_{label}"),
    )
    agree = (merged["decision_model_real"] == merged[f"decision_model_{label}"]).mean()
    return {
        "n": int(len(merged)),
        "decision_model_agreement_vs_real": float(agree),
        "real_distribution": merged["decision_model_real"].value_counts().to_dict(),
        f"{label}_distribution": merged[f"decision_model_{label}"].value_counts().to_dict(),
    }


def _run_arm_test(n: int, cfg, model_dir: Path) -> None:
    """Acceptance test: run the FIRST n admissions, in the main run's own
    row order, through the batched reference arm (no manipulation), and
    assert the decisions match the main run exactly.

    Must use the first n in file order, not a random sample: each
    admission in the main run was generated together with the specific
    neighbours that landed in its batch (left-padded to the longest
    prompt in that batch), and bf16 batched generation is sensitive to
    those batch partners. A random sample would pair each admission with
    different neighbours than the real run used, so a legitimate bf16
    rounding difference could fail this test for a reason that has
    nothing to do with whether the batched control path is correct. The
    first n rows of stage3_batch_results.csv are exactly the first
    n // generation_batch_size batches of the real run's first segment
    (before any --resume could shift batch boundaries), so this
    reproduces the real batch grouping exactly.

    Also run this on the same GPU type as the main run (A100, not the
    H100 partition) -- a different GPU can change bf16 rounding too, and
    that would be yet another reason this could fail that has nothing to
    do with the code being tested.

    This is the live, GPU-needed check that the batched control path
    genuinely reproduces the main run when given identical input -- run
    this once, on a small n, before trusting a full control run's
    agreement numbers. Does not touch any of the real control output
    files (writes to its own throwaway path).
    """
    real = pd.read_csv(model_dir / "stage3_batch_results.csv")
    sample = real.head(n)["hadm_id"].astype(int).tolist()
    print(f"[arm_test] Testing the first {len(sample)} admissions (real run's row order, "
          "same batch grouping) against the main run", flush=True)

    results_df = _load_results(model_dir)
    setup = _BatchSetup(
        artifact=_load_artifact(cfg), results_df=results_df,
        feature_matrix=load_feature_matrix(cfg, "full"),
        notes_lookup=_preload_notes(cfg, sample, results_df), pending=[],
    )

    out_path = model_dir / "stage3_arm_test_reference.csv"
    if out_path.exists():
        out_path.unlink()  # always fresh -- this is a quick live check, never resumed
    control = _run_batched_control("arm_test", sample, cfg, out_path, setup)

    merged = real.merge(control, on="hadm_id", suffixes=("_real", "_arm"))
    mismatches = merged[merged["decision_model_real"] != merged["decision_model_arm"]]
    print(f"\n[arm_test] {len(merged) - len(mismatches)}/{len(merged)} decisions match "
          "the main run exactly.", flush=True)
    if len(mismatches):
        print("[arm_test] MISMATCHES (batched reference arm must reproduce the main run "
              "when input is identical -- do not trust the full controls until this is "
              "0/n):", flush=True)
        print(mismatches[["hadm_id", "decision_model_real", "decision_model_arm"]]
              .to_string(index=False), flush=True)
        # Non-zero exit so a chained `--arm-test N && --n ...` invocation
        # (see scripts/slurm_stage3_validation.sh) halts before spending
        # GPU hours on controls built on a broken batched path.
        sys.exit(1)
    print("[arm_test] PASS -- safe to proceed with the full D1/D2/D3 controls.", flush=True)


def main() -> None:
    """CLI entry point for the D1/D2/D3 validation-control sample run."""
    parser = argparse.ArgumentParser(description="Stage 3 validation controls (D1/D2/D3)")
    parser.add_argument("--n", type=int, default=150, help="Sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controls", default="d1,d2,d3",
        help="Comma-separated subset: d1 (blind-note), d2 (no-stage2), d3 (self-agreement)",
    )
    parser.add_argument(
        "--arm-test", type=int, default=None, metavar="N",
        help="Run the live acceptance test instead: N admissions through the "
             "unmanipulated batched reference arm, checked against the main run's "
             "actual decisions. Run this first, on a small N, before trusting a "
             "full control run.",
    )
    args = parser.parse_args()

    cfg = load_config()
    model_dir = get_model_dir()

    if args.arm_test is not None:
        _run_arm_test(args.arm_test, cfg, model_dir)
        return

    controls = set(args.controls.split(","))

    real_results = _load_results(model_dir)
    batch_df = pd.read_csv(model_dir / "stage3_batch_results.csv")
    sample = _select_sample(batch_df, args.n, args.seed)
    print(f"[validation_controls] Sample size: {len(sample)} (requested {args.n})", flush=True)

    # Artifact MUST load before torch -- see src/stage3/pipeline.py.
    setup = _BatchSetup(
        artifact=_load_artifact(cfg), results_df=real_results,
        feature_matrix=load_feature_matrix(cfg, "full"),
        notes_lookup=_preload_notes(cfg, sample, real_results), pending=[],
    )
    report: dict = {"n_sample": len(sample), "sample_hadm_ids": sample}

    if "d1" in controls:
        out = model_dir / "stage3_control_blind_note.csv"
        control_df = _run_batched_control(
            "D1 blind-note", sample, cfg, out, setup, suppress_note=True,
        )
        report["d1_blind_note"] = _agreement_report(batch_df, control_df, "blind_note")
        print(f"[validation_controls] D1 agreement vs real: "
              f"{report['d1_blind_note']['decision_model_agreement_vs_real']:.1%}", flush=True)

    if "d2" in controls:
        out = model_dir / "stage3_control_no_stage2.csv"
        control_df = _run_batched_control(
            "D2 no-Stage-2", sample, cfg, out, setup, suppress_stage2=True,
        )
        report["d2_no_stage2"] = _agreement_report(batch_df, control_df, "no_stage2")
        print(f"[validation_controls] D2 agreement vs real: "
              f"{report['d2_no_stage2']['decision_model_agreement_vs_real']:.1%}", flush=True)

    if "d3" in controls:
        out_a = model_dir / "stage3_control_self_agreement_run1.csv"
        out_b = model_dir / "stage3_control_self_agreement_run2.csv"
        run1 = _run_batched_control("D3 self-agreement run 1", sample, cfg, out_a, setup)
        run2 = _run_batched_control("D3 self-agreement run 2", sample, cfg, out_b, setup)
        merged = run1.merge(run2, on="hadm_id", suffixes=("_1", "_2"))
        report["d3_self_agreement"] = {
            "n": int(len(merged)),
            "decision_model_agreement": float(
                (merged["decision_model_1"] == merged["decision_model_2"]).mean()
            ),
            "decision_rule_agreement": float(
                (merged["decision_rule_1"] == merged["decision_rule_2"]).mean()
            ),
        }
        print(f"[validation_controls] D3 self-agreement (decision_model): "
              f"{report['d3_self_agreement']['decision_model_agreement']:.1%}", flush=True)

    out_path = model_dir / "stage3_validation_controls_summary.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[validation_controls] Saved summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
