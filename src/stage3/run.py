"""Stage 3 pipeline: cross-modal discordance analysis + clinician explanations.

Processes ALL Stage 1 flagged patients (confirmed AND rejected by Stage 2),
not just confirmed ones.  The rejected cases are the most scientifically
interesting: they show where discharge note content contradicts a structured
EHR risk signal, which is the core research finding.

Pipeline
--------
1. Load Stage 2 results (``models/stage2_results.csv``).
2. Merge feature matrix to recover age_band and Stage 1 features per patient.
3. Load Stage 1 artifact; compute SHAP features per patient.
4. Load discharge notes from MIMIC-IV-Note.
5. Optionally extract Clinical-Longformer attention spans.
6. Call phi4-mini for each patient: discordance_mode + primary_category +
   explanation.
7. Save per-patient results to ``models/stage3_discordance.csv``.
8. Aggregate population-level discordance statistics and save to
   ``models/stage3_discordance_analysis.json``.

Usage::

    python -m src.stage3.run
    python -m src.stage3.run --limit 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL
from src.stage3.attention import extract_attention_spans
from src.stage3.categorize import aggregate_discordance
from src.stage3.explain import annotate_batch
from src.stage3.shap_extract import extract_shap_features


def _load_results(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "stage2_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 2 results not found at {path}.  Run setup_stage2.py first."
        )
    return pd.read_csv(path)


def _merge_age_band(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Attach age_band to results via the feature matrix."""
    matrix = load_feature_matrix(cfg, "full")
    age_cols = [c for c in ["hadm_id", "age_band"] if c in matrix.columns]
    if len(age_cols) == 2:
        df = df.merge(matrix[age_cols], on="hadm_id", how="left")
        df["age_band"] = df["age_band"].astype(str)
    return df


def _load_artifact(cfg: AppConfig) -> dict:
    model_dir = get_model_dir()
    artifact_path = model_dir / f"stage1_{cfg.stage1.model}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Stage 1 artifact not found at {artifact_path}.  "
            "Run setup_demo.py then python -m src.model.train first."
        )
    return joblib.load(artifact_path)


def _get_patient_features(
    df: pd.DataFrame, artifact: dict, cfg: AppConfig
) -> pd.DataFrame:
    """Return feature rows aligned to df's hadm_ids for SHAP extraction."""
    matrix = load_feature_matrix(cfg, artifact.get("mode", "full"))
    feat_cols = artifact["feature_cols"]
    available_feat = [c for c in feat_cols if c in matrix.columns]
    m = matrix[["hadm_id"] + available_feat]
    merged = df[["hadm_id"]].merge(m, on="hadm_id", how="left")
    return merged[available_feat].reset_index(drop=True)


def _sample_stratified(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Stratified sample keeping confirmed/rejected representation."""
    confirmed = df[df["stage2_confirmed"] == 1]
    rejected = df[df["stage2_confirmed"] == 0]
    n_conf = min(limit // 2, len(confirmed))
    n_rej = min(limit - n_conf, len(rejected))
    sampled = pd.concat([
        confirmed.sample(n=n_conf, random_state=42),
        rejected.sample(n=n_rej, random_state=42),
    ]).reset_index(drop=True)
    print(f"[stage3] Sampled {len(sampled):,} patients ({n_conf:,} confirmed, {n_rej:,} rejected).")
    return sampled


def _load_notes_map(df: pd.DataFrame, cfg: AppConfig) -> dict[int, str]:
    """Return hadm_id -> discharge note text (empty dict on failure)."""
    # pylint: disable=import-outside-toplevel  # deferred: importing dataset.py
    # initialises PyTorch/MPS, which must happen AFTER joblib XGBoost load
    from src.stage2.dataset import build_notes_dataframe, load_notes
    hadm_id_set = set(df["hadm_id"].astype(int).tolist())
    try:
        notes_raw = load_notes(cfg, hadm_ids=hadm_id_set)
        notes_df = build_notes_dataframe(
            notes_raw, df[["hadm_id", "subject_id", TARGET_COL]]
        )
        mapping = dict(zip(notes_df["hadm_id"].astype(int), notes_df["text"]))
        print(f"[stage3] Discharge notes loaded for {len(mapping):,} patients.")
        return mapping
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[stage3] WARNING: Could not load discharge notes: {exc}")
        return {}


def _get_attention_spans(
    df: pd.DataFrame,
    hadm_to_text: dict[int, str],
    model_dir: Path,
    cfg: AppConfig,
) -> dict[int, list[str]]:
    """Extract attention spans; returns empty dict if disabled or model missing."""
    if not cfg.stage3.attention_extraction or not hadm_to_text:
        return {}
    ordered_ids = [int(h) for h in df["hadm_id"]]
    ordered_texts = [hadm_to_text.get(h, "") for h in ordered_ids]
    ids_with_notes = [h for h, t in zip(ordered_ids, ordered_texts) if t]
    texts_with_notes = [t for t in ordered_texts if t]
    if not ids_with_notes:
        return {}
    return extract_attention_spans(
        ids_with_notes,
        texts_with_notes,
        model_dir,
        top_n=cfg.stage3.top_attention_sentences,
        max_length=cfg.stage2.max_seq_length,
    )


def run_stage3(
    cfg: AppConfig,
    results_path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the Stage 3 cross-modal discordance analysis.

    Args:
        cfg:          validated project config.
        results_path: path to Stage 2 results CSV.  Defaults to
                      ``models/stage2_results.csv``.
        limit:        if set, process this many patients (stratified random
                      sample across confirmed / rejected and age groups).

    Returns:
        DataFrame of annotated patients with columns added:
        ``discordance_mode``, ``primary_category``, ``explanation``.
    """
    if not cfg.stage3.enabled:
        print("[stage3] Disabled in config (stage3.enabled: false). Skipping.")
        return pd.DataFrame()

    model_dir = get_model_dir()
    results_path = results_path or model_dir / "stage2_results.csv"

    # ── 1. Load Stage 1 artifact FIRST (must precede any large numpy allocation)
    # On macOS ARM, joblib-deserialising an XGBoost pickle after a large
    # pandas DataFrame is already in memory triggers a SIGSEGV in XGBoost's
    # C++ layer.  Loading the artifact before the feature matrix avoids this.
    try:
        artifact = _load_artifact(cfg)
    except FileNotFoundError as exc:
        print(f"[stage3] WARNING: {exc}  — proceeding without SHAP features.")
        artifact = None

    # ── 2. Load + optionally sample Stage 2 results ──────────────────────────
    df = _load_results(model_dir)
    print(
        f"[stage3] {len(df):,} Stage 1 flagged patients "
        f"({int(df['stage2_confirmed'].sum()):,} confirmed, "
        f"{int((df['stage2_confirmed'] == 0).sum()):,} rejected by Stage 2)"
    )
    df = _merge_age_band(df, cfg)
    if limit and len(df) > limit:
        df = _sample_stratified(df, limit)

    # ── 3. SHAP features ─────────────────────────────────────────────────────
    if artifact is not None:
        shap_features_list = extract_shap_features(
            artifact, _get_patient_features(df, artifact, cfg),
            top_k=cfg.stage3.top_shap_features,
        )
        print(f"[stage3] SHAP features extracted for {len(df):,} patients.")
    else:
        shap_features_list = [[] for _ in range(len(df))]

    # ── 4. Discharge notes + attention spans ─────────────────────────────────
    hadm_to_text = _load_notes_map(df, cfg)
    attention_spans = _get_attention_spans(df, hadm_to_text, model_dir, cfg)

    # ── 5. phi4-mini annotation ───────────────────────────────────────────────
    patients = df.to_dict(orient="records")
    for p in patients:
        p["note_text"] = hadm_to_text.get(int(p.get("hadm_id", 0)), "")

    annotated = annotate_batch(patients, shap_features_list, attention_spans, cfg)
    out_df = pd.DataFrame(annotated)
    if "note_text" in out_df.columns:
        out_df = out_df.drop(columns=["note_text"])

    out_path = model_dir / "stage3_discordance.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[stage3] Saved per-patient results -> {out_path}")

    # ── 6. Population-level aggregation ──────────────────────────────────────
    if cfg.stage3.discordance_analysis:
        analysis = aggregate_discordance(out_df)
        analysis_path = model_dir / "stage3_discordance_analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2))
        print(f"[stage3] Saved discordance analysis -> {analysis_path}")

    return out_df


def main() -> None:
    """CLI entry point for Stage 3."""
    parser = argparse.ArgumentParser(description="Stage 3: discordance analysis + explanations")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to Stage 2 results CSV (default: models/stage2_results.csv)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of patients to process (stratified sample; default: all)",
    )
    args = parser.parse_args()
    cfg = load_config()
    run_stage3(cfg, results_path=args.input, limit=args.limit)


if __name__ == "__main__":
    main()
