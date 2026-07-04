"""Entry point for Stage 3: generate plain-language explanations for confirmed high-risk patients.

Reads Stage 2 results (models/stage2_results.csv), merges in the feature matrix
for prompt context, and calls Ollama to generate one explanation per patient.

Output saved to models/stage3_explanations.csv.

Prerequisites:
  - Stage 2 results: run `python -m src.stage2.predict` first
  - Ollama running with model pulled: `ollama pull phi4-mini`
  - stage3.enabled: true in config.yaml

Usage:
    python -m src.stage3.run
    python -m src.stage3.run --input models/stage2_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix
from src.schemas import ID_COLS, TARGET_COL
from src.stage3.explain import generate_explanations_batch


def run_stage3(cfg: dict, results_path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Generate explanations for Stage 2 confirmed patients.

    Args:
        cfg: loaded config dict.
        results_path: path to Stage 2 results CSV. Defaults to models/stage2_results.csv.
        limit: if set, only generate explanations for this many randomly sampled patients.

    Returns:
        DataFrame of confirmed patients with an added 'explanation' column.
    """
    model_dir = get_model_dir()
    results_path = results_path or model_dir / "stage2_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Stage 2 results not found at {results_path}. "
            "Run `python -m src.stage2.predict` first."
        )

    stage2_df = pd.read_csv(results_path)
    confirmed_df = stage2_df[stage2_df["stage2_confirmed"] == 1].reset_index(drop=True)
    print(f"[stage3] {len(confirmed_df):,} confirmed high-risk patients to explain.")

    if limit and len(confirmed_df) > limit:
        confirmed_df = confirmed_df.sample(n=limit, random_state=42).reset_index(drop=True)
        print(f"[stage3] Sampled {limit} patients for explanation generation.")

    if len(confirmed_df) == 0:
        print("[stage3] No confirmed patients — nothing to do.")
        return confirmed_df

    # ── Merge feature matrix for richer prompt context ──
    # Always load full matrix so all hadm_ids from Stage 2 (trained in full mode)
    # get feature context; quick mode would miss most rows since they're subsampled.
    matrix = load_feature_matrix(cfg, "full")
    exclude = {TARGET_COL, "gender", "age_band"}
    feat_cols = [c for c in matrix.columns if c not in exclude]
    confirmed_df = confirmed_df.merge(matrix[feat_cols], on="hadm_id", how="left")

    patients = confirmed_df.to_dict(orient="records")
    results = generate_explanations_batch(patients, cfg)
    out_df = pd.DataFrame(results)

    out_path = model_dir / "stage3_explanations.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[stage3] Saved explanations -> {out_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: generate readmission risk explanations")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to Stage 2 results CSV (default: models/stage2_results.csv)"
    )
    args = parser.parse_args()
    cfg = load_config()
    run_stage3(cfg, results_path=args.input)
