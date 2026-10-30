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

Writes incrementally to one CSV per control, crash-safe and resumable --
same pattern as ``src/stage3/batch.py``'s ``run_batch_audit``, since this
calls the LLM once per patient per control (not batched like the main run)
and can run for a while. Resuming is automatic: rerun the same command and
already-written hadm_ids are skipped.

Usage::

    python -m scripts.run_validation_controls --n 150
    python -m scripts.run_validation_controls --n 150 --controls d1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import get_model_dir, load_config
from src.data.features import load_feature_matrix
from src.stage3.pipeline import _load_artifact, _load_results, explain_patient

_FIELDS = [
    "hadm_id", "decision_model", "decision_rule", "all_quotes_verified",
    "mitigating_grounds", "aggravating_grounds", "clinical_justification",
    "annotation_failed",
]


def _select_sample(batch_results: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Reproducible random sample of admissions the real batch run
    actually scored (excludes annotation_failed -- only those have a real
    decision to compare controls against)."""
    valid = batch_results[~batch_results["annotation_failed"]]
    n = min(n, len(valid))
    return valid.sample(n=n, random_state=seed)["hadm_id"].astype(int).tolist()


def _already_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["hadm_id"])["hadm_id"].astype(int))


def _run_single_control(
    name: str,
    hadm_ids: list[int],
    cfg,
    out_path: Path,
    *,
    results_df: pd.DataFrame,
    artifact: dict,
    feature_matrix: pd.DataFrame,
    **explain_kwargs,
) -> None:
    """Run explain_patient over hadm_ids with the given control kwargs,
    writing incrementally so a crash/timeout loses at most one patient."""
    done = _already_done(out_path)
    pending = [h for h in hadm_ids if h not in done]
    print(f"[validation_controls] {name}: {len(hadm_ids)} target, "
          f"{len(done)} already done, {len(pending)} pending", flush=True)
    if not pending:
        return

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        import csv  # pylint: disable=import-outside-toplevel
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        for i, hadm_id in enumerate(pending, 1):
            try:
                result = explain_patient(
                    hadm_id, cfg, results_df=results_df, artifact=artifact,
                    feature_matrix=feature_matrix, **explain_kwargs,
                )
                row = result.model_dump()
                row["mitigating_grounds"] = json.dumps(row["mitigating_grounds"])
                row["aggravating_grounds"] = json.dumps(row["aggravating_grounds"])
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[validation_controls] {name} hadm_id={hadm_id} FAILED: {exc}", flush=True)
                row = {
                    "hadm_id": hadm_id, "decision_model": None, "decision_rule": None,
                    "all_quotes_verified": None, "mitigating_grounds": "[]",
                    "aggravating_grounds": "[]",
                    "clinical_justification": f"[CONTROL FAILED: {exc}]",
                    "annotation_failed": True,
                }
            writer.writerow({k: row[k] for k in _FIELDS})
            fh.flush()
            if i % 10 == 0 or i == len(pending):
                print(f"[validation_controls] {name} [{i}/{len(pending)}]", flush=True)


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


def main() -> None:
    """CLI entry point for the D1/D2/D3 validation-control sample run."""
    parser = argparse.ArgumentParser(description="Stage 3 validation controls (D1/D2/D3)")
    parser.add_argument("--n", type=int, default=150, help="Sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controls", default="d1,d2,d3",
        help="Comma-separated subset: d1 (blind-note), d2 (no-stage2), d3 (self-agreement)",
    )
    args = parser.parse_args()
    controls = set(args.controls.split(","))

    cfg = load_config()
    model_dir = get_model_dir()
    # Artifact MUST load before torch -- see src/stage3/pipeline.py.
    artifact = _load_artifact(cfg)
    real_results = _load_results(model_dir)
    feature_matrix = load_feature_matrix(cfg, "full")

    batch_df = pd.read_csv(model_dir / "stage3_batch_results.csv")
    sample = _select_sample(batch_df, args.n, args.seed)
    print(f"[validation_controls] Sample size: {len(sample)} (requested {args.n})", flush=True)

    report: dict = {"n_sample": len(sample), "sample_hadm_ids": sample}

    if "d1" in controls:
        out = model_dir / "stage3_control_blind_note.csv"
        _run_single_control(
            "D1 blind-note", sample, cfg, out,
            results_df=real_results, artifact=artifact, feature_matrix=feature_matrix,
            suppress_note=True,
        )
        control_df = pd.read_csv(out)
        report["d1_blind_note"] = _agreement_report(batch_df, control_df, "blind_note")
        print(f"[validation_controls] D1 agreement vs real: "
              f"{report['d1_blind_note']['decision_model_agreement_vs_real']:.1%}", flush=True)

    if "d2" in controls:
        out = model_dir / "stage3_control_no_stage2.csv"
        _run_single_control(
            "D2 no-Stage-2", sample, cfg, out,
            results_df=real_results, artifact=artifact, feature_matrix=feature_matrix,
            suppress_stage2=True,
        )
        control_df = pd.read_csv(out)
        report["d2_no_stage2"] = _agreement_report(batch_df, control_df, "no_stage2")
        print(f"[validation_controls] D2 agreement vs real: "
              f"{report['d2_no_stage2']['decision_model_agreement_vs_real']:.1%}", flush=True)

    if "d3" in controls:
        out_a = model_dir / "stage3_control_self_agreement_run1.csv"
        out_b = model_dir / "stage3_control_self_agreement_run2.csv"
        _run_single_control(
            "D3 self-agreement run 1", sample, cfg, out_a,
            results_df=real_results, artifact=artifact, feature_matrix=feature_matrix,
        )
        _run_single_control(
            "D3 self-agreement run 2", sample, cfg, out_b,
            results_df=real_results, artifact=artifact, feature_matrix=feature_matrix,
        )
        run1, run2 = pd.read_csv(out_a), pd.read_csv(out_b)
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
