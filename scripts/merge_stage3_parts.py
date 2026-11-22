"""Merge the main sequential Stage 3 result with parallel range-job results.

The full Stage 3 run can be executed as one sequential ``--resume`` job plus
several parallel range jobs (``scripts/slurm_stage3_range.sh``, batch-aligned
``--range-start``/``--range-end`` slices). This script stitches them back into
the single canonical ``models/stage3_batch_results.csv`` the rest of the
pipeline (``evaluate_pipeline.py``, ``slurm_stage3_validation.sh``, the
sensitivity sweep) expects.

What it guarantees before writing anything:

* every input CSV has a header identical to the main file's;
* the union of all hadm_ids equals the FULL target set
  (``models/stage2_results.csv``) exactly -- same completeness check as
  ``scripts/slurm_stage3_validation.sh`` -- so neither a missing nor an extra
  admission slips through;
* on a hadm_id present in more than one file, the row from the main file wins
  (then earlier ``--parts`` win over later ones); the number of dropped
  duplicate rows is reported.

Only after all checks pass does it write, in the original target order, to the
main path via a temporary file, having first copied the unmerged main file to
``models/stage3_batch_results.premerge.csv``. The ``.raw.jsonl`` sidecars are
merged the same way. Any failed check exits 1 with a clear message and leaves
every file untouched.

CPU only -- no model, no torch. Safe to run on a login node.

Usage::

    python -m scripts.merge_stage3_parts \\
        --parts models/stage3_batch_results.range5200-9600.csv
    # or several:
    python -m scripts.merge_stage3_parts \\
        --parts models/stage3_batch_results.range5200-7600.csv \\
                models/stage3_batch_results.range7600-9899.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.config import get_model_dir


def _fail(msg: str) -> None:
    """Print a clear error to stderr and exit 1 (a failed merge check)."""
    print(f"[merge_stage3_parts] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _raw_sidecar(csv_path: Path) -> Path:
    """The ``.raw.jsonl`` sidecar path for a result CSV -- mirrors
    ``run_batch_audit``'s ``out_path.with_name(out_path.stem + '.raw.jsonl')``."""
    return csv_path.with_name(csv_path.stem + ".raw.jsonl")


def _load_target_order(stage2_path: Path) -> list[int]:
    """The full ordered target hadm_id list (the order a single sequential run
    would audit in), read from stage2_results.csv."""
    if not stage2_path.exists():
        _fail(f"{stage2_path} not found -- cannot determine the target set/order.")
    df = pd.read_csv(stage2_path, usecols=["hadm_id"])
    return [int(h) for h in df["hadm_id"].tolist()]


def _read_checked(path: Path, reference_header: list[str] | None) -> pd.DataFrame:
    """Read one result CSV; if ``reference_header`` is given, require an
    identical header (order included), else exit 1."""
    if not path.exists():
        _fail(f"{path} not found.")
    df = pd.read_csv(path)
    if reference_header is not None and list(df.columns) != reference_header:
        _fail(
            f"{path} header does not match the main file's.\n"
            f"  main:  {reference_header}\n"
            f"  this:  {list(df.columns)}"
        )
    return df


def _merge_raw_sidecars(
    main_path: Path, part_paths: list[Path], target_order: list[int]
) -> None:
    """Merge the ``.raw.jsonl`` sidecars the same way as the CSVs: dedup by
    hadm_id (main first, then parts in order), order by the target order, write
    to the main sidecar via a temp file after backing it up to ``.premerge``.

    Unlike the CSV merge there is no completeness check -- prep-failure rows get
    a CSV row but no raw line, so the sidecars legitimately cover fewer
    admissions than the results. A missing sidecar is reported and skipped, not
    treated as a failure (the CSV merge is the critical deliverable)."""
    main_raw = _raw_sidecar(main_path)
    ordered_sources = [main_raw] + [_raw_sidecar(p) for p in part_paths]
    present_sources = [p for p in ordered_sources if p.exists()]
    for p in ordered_sources:
        if not p.exists():
            print(f"[merge_stage3_parts] raw sidecar {p} absent -- skipping it.")
    if not main_raw.exists() or not present_sources:
        print("[merge_stage3_parts] no raw sidecar to merge into "
              f"{main_raw} -- leaving raw output untouched.")
        return

    by_hadm: dict[int, str] = {}
    for src in present_sources:
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    hadm = int(json.loads(line)["hadm_id"])
                except (ValueError, KeyError, TypeError):
                    continue  # a malformed/partial line -- skip, don't abort
                by_hadm.setdefault(hadm, line)  # first file wins (main precedence)

    premerge = main_raw.with_name(main_raw.name.replace(".raw.jsonl", ".premerge.raw.jsonl"))
    shutil.copy2(main_raw, premerge)
    tmp = main_raw.with_name(main_raw.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        for hadm in target_order:
            if hadm in by_hadm:
                out.write(by_hadm[hadm] + "\n")
    tmp.replace(main_raw)
    print(f"[merge_stage3_parts] merged raw sidecars -> {main_raw} "
          f"(backup {premerge}, {len(by_hadm):,} lines).")


def merge_stage3_parts(
    main_path: Path, part_paths: list[Path], stage2_path: Path
) -> dict:
    """Validate and merge; write the canonical results in target order. Exits 1
    on any failed check (missing input, header mismatch, incomplete union).
    Returns a small report dict on success."""
    if not part_paths:
        _fail("no --parts given; nothing to merge.")
    target_order = _load_target_order(stage2_path)
    target_set = set(target_order)

    main_df = _read_checked(main_path, reference_header=None)
    header = list(main_df.columns)
    part_dfs = [_read_checked(p, reference_header=header) for p in part_paths]

    combined = pd.concat([main_df, *part_dfs], ignore_index=True)
    combined["hadm_id"] = combined["hadm_id"].astype(int)
    n_before = len(combined)
    # keep="first" + main_df concatenated first => main's row wins on conflict,
    # then earlier parts win over later ones.
    merged = combined.drop_duplicates(subset="hadm_id", keep="first")
    n_dropped = n_before - len(merged)

    merged_set = set(merged["hadm_id"].tolist())
    missing = [h for h in target_order if h not in merged_set]
    extra = sorted(merged_set - target_set)
    if missing or extra:
        _fail(
            "merged hadm_ids do not equal the Stage 2 target set.\n"
            f"  target admissions (stage2_results.csv): {len(target_set):,}\n"
            f"  merged admissions:                      {len(merged_set):,}\n"
            f"  in target but not merged: {len(missing):,} (e.g. {missing[:5]})\n"
            f"  in merged but not target: {len(extra):,} (e.g. {extra[:5]})"
        )

    ordered = merged.set_index("hadm_id").reindex(target_order).reset_index()
    ordered = ordered[header]

    premerge = main_path.with_name(main_path.stem + ".premerge.csv")
    shutil.copy2(main_path, premerge)
    tmp = main_path.with_name(main_path.name + ".tmp")
    ordered.to_csv(tmp, index=False)
    tmp.replace(main_path)
    print(f"[merge_stage3_parts] merged {len(part_paths)} range file(s) into "
          f"{main_path}: {len(ordered):,} rows in target order, "
          f"{n_dropped:,} duplicate row(s) dropped (main file took precedence). "
          f"Unmerged main backed up to {premerge}.")

    _merge_raw_sidecars(main_path, part_paths, target_order)
    return {
        "n_merged": int(len(ordered)),
        "n_dropped_duplicates": int(n_dropped),
        "n_parts": len(part_paths),
    }


def main() -> None:
    """CLI entry point."""
    model_dir = get_model_dir()
    parser = argparse.ArgumentParser(
        description="Merge the main sequential Stage 3 result with range-job results."
    )
    parser.add_argument(
        "--main", type=Path, default=model_dir / "stage3_batch_results.csv",
        help="Canonical/authoritative results file (default: "
             "models/stage3_batch_results.csv). Rows here win on any hadm_id conflict.",
    )
    parser.add_argument(
        "--parts", type=Path, nargs="+", required=True,
        help="One or more range-job result CSVs to merge into --main.",
    )
    parser.add_argument(
        "--stage2", type=Path, default=model_dir / "stage2_results.csv",
        help="Stage 2 results, used for the full target set and order "
             "(default: models/stage2_results.csv).",
    )
    args = parser.parse_args()
    merge_stage3_parts(args.main, list(args.parts), args.stage2)


if __name__ == "__main__":
    main()
