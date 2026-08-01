"""One-command Stage 3 runner: cross-modal discordance analysis + explanations.

For every patient flagged by Stage 1 (both confirmed AND rejected by Stage 2),
generates:

  1. Cross-modal discordance annotation — CONCORDANT, NOTE_MITIGATES, or
     NOTE_AMPLIFIES — explaining whether the discharge note aligns with or
     contradicts the structured EHR risk signal.
  2. Primary clinical category driving any discordance (social_support,
     discharge_planning, frailty_markers, …).
  3. One-sentence clinician-facing explanation synthesising Stages 1 and 2.

Population-level aggregation across all processed patients produces the core
research finding: which clinical domains in discharge notes systematically
explain the predictive gap between structured EHR data and clinical text.

Requires:
  - Stage 2 results:  models/stage2_results.csv  (run setup_stage2.py first)
  - Stage 1 artifact: models/stage1_xgboost.joblib
  - MIMIC-IV-Note:    MIMIC_IV_NOTE_DIR in .env  (needed for note text)
  - Ollama running:   https://ollama.com
  - Model pulled:     ollama pull phi4-mini

Usage:
    python setup_stage3.py
    python setup_stage3.py --limit 500   # process 500 sampled patients
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import get_model_dir, load_config


def check_prerequisites(cfg, results_path: Path) -> None:
    """Verify Stage 2 results and Ollama are available before starting."""
    if not results_path.exists():
        print(f"[setup_stage3] ERROR: Stage 2 results not found at {results_path}")
        print("  Run setup_stage2.py first.")
        sys.exit(1)

    try:
        import ollama
        ollama.list()
    except Exception:  # pylint: disable=broad-exception-caught
        print("[setup_stage3] ERROR: Cannot connect to Ollama.")
        print("  Make sure Ollama is installed and running: https://ollama.com")
        sys.exit(1)

    model = cfg.stage3.ollama_model
    try:
        import ollama as _ol
        available = [m.model for m in _ol.list().models]
        if not any(m.startswith(model.split(":")[0]) for m in available):
            print(f"[setup_stage3] ERROR: Model '{model}' not found in Ollama.")
            print(f"  Run:  ollama pull {model}")
            sys.exit(1)
    except Exception:  # pylint: disable=broad-exception-caught
        print("[setup_stage3] WARNING: Could not verify model availability — continuing.")


def _print_discordance_summary(analysis: dict) -> None:
    """Print a readable summary of the population-level discordance analysis."""
    if not analysis:
        return

    n = analysis.get("n_patients", 0)
    if n == 0:
        return

    modes = analysis.get("mode_distribution", {})
    counts = modes.get("counts", {})
    pcts = modes.get("percentages", {})

    print("\n  Discordance mode distribution:")
    for mode in ("CONCORDANT", "NOTE_MITIGATES", "NOTE_AMPLIFIES"):
        c = counts.get(mode, 0)
        p = pcts.get(mode, 0.0)
        print(f"    {mode:<20}: {c:>5,}  ({p:>5.1f}%)")

    mitig = analysis.get("note_mitigates", {})
    if mitig.get("n", 0) > 0:
        print(f"\n  NOTE_MITIGATES top categories (n={mitig['n']:,}, "
              f"{mitig.get('pct_of_rejected', 0):.1f}% of rejected flags):")
        for cat, cnt in list(mitig.get("top_categories", {}).items())[:5]:
            print(f"    {cat:<28}: {cnt:>4,}")

    ampli = analysis.get("note_amplifies", {})
    if ampli.get("n", 0) > 0:
        print(f"\n  NOTE_AMPLIFIES top categories (n={ampli['n']:,}, "
              f"{ampli.get('pct_of_confirmed', 0):.1f}% of confirmed flags):")
        for cat, cnt in list(ampli.get("top_categories", {}).items())[:5]:
            print(f"    {cat:<28}: {cnt:>4,}")


def main() -> None:
    """CLI entry point for Stage 3."""
    parser = argparse.ArgumentParser(description="Run Stage 3 end-to-end")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to Stage 2 results CSV (default: models/stage2_results.csv)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process this many patients (stratified sample; default: all). "
             "500 patients ≈ 30-60 min with phi4-mini."
    )
    parser.add_argument(
        "--no-attention", action="store_true",
        help="Skip Longformer attention extraction (required on CPU/Mac — "
             "loading 2048-token attention matrices OOMs without a GPU)."
    )
    args = parser.parse_args()

    cfg = load_config()
    cfg.stage3.enabled = True  # force-enable for this script
    if args.no_attention:
        cfg.stage3.attention_extraction = False

    model_dir = get_model_dir()
    results_path = args.input or model_dir / "stage2_results.csv"

    print("=" * 64)
    print("  Readmission Prediction — Stage 3 (Cross-Modal Analysis)")
    print("=" * 64)
    print(f"  Model:  {cfg.stage3.ollama_model}")
    print(f"  Input:  {results_path}")
    print(f"  Limit:  {args.limit or 'all patients'}")
    print(f"  Attention extraction: {cfg.stage3.attention_extraction}")
    print()

    check_prerequisites(cfg, results_path)

    stage2_df = pd.read_csv(results_path)
    n_total = len(stage2_df)
    n_confirmed = int((stage2_df["stage2_confirmed"] == 1).sum())
    n_rejected = n_total - n_confirmed
    print(f"[setup_stage3] {n_total:,} Stage 1 flagged patients in results "
          f"({n_confirmed:,} confirmed, {n_rejected:,} rejected by Stage 2).")
    if args.limit and args.limit < n_total:
        print(f"[setup_stage3] --limit {args.limit}: analysing a stratified sample.\n")
    else:
        print()

    from src.stage3.run import run_stage3
    out_df = run_stage3(cfg, results_path=results_path, limit=args.limit)

    if out_df.empty:
        print("[setup_stage3] No results produced.")
        return

    # ── Load and print population analysis ────────────────────────────────────
    analysis_path = model_dir / "stage3_discordance_analysis.json"
    analysis = {}
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

    # ── Print example explanations (3 confirmed, 3 rejected) ─────────────────
    print("\n" + "=" * 64)
    print("  Stage 3 complete — example annotations")
    print("=" * 64)

    for label, flag_val in [("CONFIRMED by Stage 2", 1), ("REJECTED by Stage 2", 0)]:
        subset = out_df[out_df["stage2_confirmed"] == flag_val].head(3)
        if subset.empty:
            continue
        print(f"\n  ── {label} ──")
        for _, row in subset.iterrows():
            print(f"\n  hadm_id={int(row['hadm_id'])}")
            print(f"    Mode:      {row.get('discordance_mode', 'N/A')}")
            print(f"    Category:  {row.get('primary_category', 'N/A')}")
            exp = str(row.get("explanation", ""))
            print(f"    Explain:   {exp[:150]}{'...' if len(exp) > 150 else ''}")

    _print_discordance_summary(analysis)

    print(f"\n  Per-patient results:   models/stage3_discordance.csv")
    print(f"  Population analysis:   models/stage3_discordance_analysis.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
