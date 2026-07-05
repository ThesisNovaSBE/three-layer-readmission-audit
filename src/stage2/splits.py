"""Build patient-level train / validation / calibration splits for Stage 2.

Splits at the subject_id (patient) level — all admissions from one patient
always land in the same split, preventing data leakage. Stratification is by
age_band so each split has similar demographic composition.

Why patient-level?
  One patient can have multiple admissions. If we split at the admission level,
  a patient's earlier note lands in training and a later note in validation —
  the model has effectively seen that patient, inflating validation metrics.

Output (saved to data/processed/):
  stage2_finetune_hadm_ids.csv   — fine-tune set (60% of training patients)
  stage2_val_hadm_ids.csv        — validation set (20%)
  stage2_cal_hadm_ids.csv        — calibration set (20%)

Each file contains: hadm_id, subject_id, age_band, readmission_30d

Usage:
    python -m src.stage2.splits
    python -m src.stage2.splits --force   # re-build even if splits already exist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, get_model_dir, get_data_dir
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL


def build_splits(cfg: dict, artifact: dict | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    """Build and save patient-level Stage 2 splits.

    Args:
        cfg: loaded config dict.
        artifact: pre-loaded Stage 1 artifact. Loaded from disk if None.
        force: if False, return cached splits when output files already exist.

    Returns:
        dict with keys "finetune", "val", "cal" — each a DataFrame of
        (hadm_id, subject_id, age_band, readmission_30d).
    """
    processed_dir = get_data_dir() / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    ft_path  = processed_dir / "stage2_finetune_hadm_ids.csv"
    val_path = processed_dir / "stage2_val_hadm_ids.csv"
    cal_path = processed_dir / "stage2_cal_hadm_ids.csv"

    if not force and ft_path.exists() and val_path.exists() and cal_path.exists():
        print("[stage2/splits] Loading cached splits from data/processed/")
        return {
            "finetune": pd.read_csv(ft_path),
            "val":      pd.read_csv(val_path),
            "cal":      pd.read_csv(cal_path),
        }

    seed = cfg["run"]["random_state"]
    val_frac = cfg["stage2"]["val_fraction"]
    cal_frac = cfg["stage2"]["calibration_fraction"]

    # ── Load Stage 1 artifact ──
    if artifact is None:
        model_dir = get_model_dir()
        stage1_name = cfg["stage1"]["model"]
        artifact = joblib.load(model_dir / f"stage1_{stage1_name}.joblib")

    mode = artifact["mode"]
    train_idx = artifact["train_idx"]

    # ── Get all training admissions with age_band ──
    matrix = load_feature_matrix(cfg, mode)
    train_df = matrix.iloc[train_idx][
        ["subject_id", "hadm_id", "age_band", TARGET_COL]
    ].reset_index(drop=True)

    print(f"[stage2/splits] Training admissions: {len(train_df):,} "
          f"across {train_df['subject_id'].nunique():,} patients")

    # ── Patient-level split (stratified by age_band) ──
    # One row per patient; age_band is constant per patient.
    patient_df = (train_df
                  .groupby("subject_id", as_index=False)
                  .agg(age_band=("age_band", "first")))

    # First split off calibration set (patient level)
    patients_temp, patients_cal = train_test_split(
        patient_df,
        test_size=cal_frac,
        stratify=patient_df["age_band"],
        random_state=seed,
    )

    # Split remainder into fine-tune and validation
    val_frac_adj = val_frac / (1.0 - cal_frac)  # adjust fraction for the remaining pool
    patients_ft, patients_val = train_test_split(
        patients_temp,
        test_size=val_frac_adj,
        stratify=patients_temp["age_band"],
        random_state=seed,
    )

    # ── Map back to admission level ──
    def _get_hadm_ids(patient_split: pd.DataFrame) -> pd.DataFrame:
        ids = set(patient_split["subject_id"])
        return train_df[train_df["subject_id"].isin(ids)].reset_index(drop=True)

    ft_df  = _get_hadm_ids(patients_ft)
    val_df = _get_hadm_ids(patients_val)
    cal_df = _get_hadm_ids(patients_cal)

    # ── Save ──
    ft_df.to_csv(ft_path,  index=False)
    val_df.to_csv(val_path, index=False)
    cal_df.to_csv(cal_path, index=False)

    _report(ft_df, val_df, cal_df)
    return {"finetune": ft_df, "val": val_df, "cal": cal_df}


def _report(ft_df: pd.DataFrame, val_df: pd.DataFrame, cal_df: pd.DataFrame) -> None:
    print("\n[stage2/splits] Split summary (admission level):")
    print(f"  {'Split':<12} {'N admissions':>14} {'N patients':>12} {'Pos rate':>10}")
    print("  " + "-" * 52)
    for name, df in [("finetune", ft_df), ("val", val_df), ("cal", cal_df)]:
        print(f"  {name:<12} {len(df):>14,} {df['subject_id'].nunique():>12,} "
              f"{df[TARGET_COL].mean():>10.1%}")
    print()

    print("  Age band distribution (% of admissions per split):")
    all_bands = sorted(ft_df["age_band"].unique(), key=str)
    header = f"  {'Band':<12}" + "".join(f"{'finetune':>12}{'val':>8}{'cal':>8}")
    print(f"  {'Band':<12} {'finetune':>10} {'val':>8} {'cal':>8}")
    print("  " + "-" * 42)
    for band in all_bands:
        ft_pct  = (ft_df["age_band"] == band).mean()
        val_pct = (val_df["age_band"] == band).mean()
        cal_pct = (cal_df["age_band"] == band).mean()
        print(f"  {str(band):<12} {ft_pct:>10.1%} {val_pct:>8.1%} {cal_pct:>8.1%}")
    print()
    print(f"[stage2/splits] Saved to data/processed/stage2_*_hadm_ids.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Stage 2 patient-level splits")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild splits even if output files exist")
    args = parser.parse_args()
    cfg = load_config()
    build_splits(cfg, force=args.force)
