"""One-command Stage 3 runner: generate plain-language risk explanations via Ollama.

Takes the Stage 2 confirmed high-risk patients and generates a 3-5 sentence
plain-language explanation for each one explaining why they are at elevated
readmission risk.

Requires:
  - Stage 2 results: models/stage2_results.csv
                     (run setup_stage2.py first)
  - Ollama running:  https://ollama.com  (install and start the app)
  - Model pulled:    ollama pull phi4-mini

Usage:
    python setup_stage3.py
    python setup_stage3.py --input models/stage2_results.csv
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config, get_model_dir


def check_prerequisites(cfg: dict, results_path: Path) -> None:
    if not results_path.exists():
        print(f"[setup_stage3] ERROR: Stage 2 results not found at {results_path}")
        print("  Run setup_stage2.py first.")
        sys.exit(1)

    try:
        import ollama
        ollama.list()  # will raise if Ollama is not running
    except Exception:
        print("[setup_stage3] ERROR: Cannot connect to Ollama.")
        print("  Make sure Ollama is installed and running: https://ollama.com")
        sys.exit(1)

    model = cfg["stage3"]["ollama_model"]
    try:
        import ollama
        available = [m.model for m in ollama.list().models]
        # check if any available model starts with the name (handles tag variants)
        if not any(m.startswith(model.split(":")[0]) for m in available):
            print(f"[setup_stage3] ERROR: Model '{model}' not found in Ollama.")
            print(f"  Run:  ollama pull {model}")
            sys.exit(1)
    except Exception as e:
        print(f"[setup_stage3] WARNING: Could not verify model availability: {e}")
        print("  Continuing anyway — will fail at generation if model is missing.")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 end-to-end")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to Stage 2 results CSV (default: models/stage2_results.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only generate explanations for N confirmed patients (random sample). "
                             "Useful for demos and thesis examples.")
    args = parser.parse_args()

    cfg = load_config()

    # Force Stage 3 enabled for this script regardless of config
    cfg["stage3"]["enabled"] = True

    model_dir = get_model_dir()
    results_path = args.input or model_dir / "stage2_results.csv"

    print("=" * 64)
    print("  Readmission Prediction — Stage 3 (Explanation)")
    print("=" * 64)
    print(f"  Model:  {cfg['stage3']['ollama_model']}")
    print(f"  Input:  {results_path}")
    print()

    check_prerequisites(cfg, results_path)

    import pandas as pd
    stage2_df = pd.read_csv(results_path)
    n_confirmed = int((stage2_df["stage2_confirmed"] == 1).sum())
    print(f"[setup_stage3] {n_confirmed:,} confirmed high-risk patients to explain.")
    if args.limit and args.limit < n_confirmed:
        print(f"[setup_stage3] --limit {args.limit}: generating explanations for {args.limit} sampled patients.\n")
    else:
        print()

    from src.stage3.run import run_stage3
    out_df = run_stage3(cfg, results_path=results_path, limit=args.limit)

    # Print a few examples
    confirmed = out_df[out_df["stage2_confirmed"] == 1].reset_index(drop=True)
    n_show = min(3, len(confirmed))

    print("\n" + "=" * 64)
    print("  Stage 3 complete — example explanations")
    print("=" * 64)
    for i in range(n_show):
        row = confirmed.iloc[i]
        print(f"\n  Patient hadm_id={int(row['hadm_id'])}:")
        for line in str(row.get("explanation", "")).split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
    print(f"\n  Full output saved to: models/stage3_explanations.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
