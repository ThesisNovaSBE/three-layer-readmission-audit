"""Stage 3 on-demand explanation for one patient.

This is the sole entry point for Stage 3.  It is called by the API when a
clinician requests an explanation for a specific patient — never in batch.

The function synthesises three layers:

* **Stage 1** (XGBoost on structured EHR): what the numbers say and why.
* **Stage 2** (Clinical-Longformer on the discharge note): whether the note
  confirmed or contradicted the structured risk, and by how much.
* **Stage 3** (phi4-mini via Ollama): *why* the note agreed or disagreed —
  a clinical narrative that connects both signals.

Usage::

    from src.stage3.pipeline import explain_patient
    from src.config import load_config

    cfg = load_config()
    result = explain_patient(hadm_id=<hadm_id>, cfg=cfg)
    print(result.narrative)
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL
from src.stage2._utils import band_key, get_stage2_model_path
from src.stage3.attention import extract_attention_spans
from src.stage3.explain import build_prompt, call_phi4mini
from src.stage3.models import ExplanationResult
from src.stage3.shap_extract import extract_shap_for_patient


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_artifact(cfg: AppConfig) -> dict:
    """Load the Stage 1 XGBoost artifact from disk."""
    model_dir = get_model_dir()
    path = model_dir / f"stage1_{cfg.stage1.model}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 1 artifact not found at {path}. "
            "Run setup_demo.py then python -m src.model.train first."
        )
    return joblib.load(path)


def _load_results(model_dir: object) -> pd.DataFrame:
    """Load Stage 2 results CSV from the models directory."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    path = Path(str(model_dir)) / "stage2_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 2 results not found at {path}. "
            "Run setup_stage2.py first."
        )
    return pd.read_csv(path)


def _lookup_patient(hadm_id: int, results_df: pd.DataFrame) -> dict:
    """Extract one patient's Stage 1 + Stage 2 data from the results DataFrame."""
    row = results_df[results_df["hadm_id"] == hadm_id]
    if row.empty:
        raise KeyError(
            f"hadm_id {hadm_id} not found in Stage 2 results. "
            "The patient may not have been flagged by Stage 1."
        )
    r = row.iloc[0]
    return {
        "hadm_id": int(r["hadm_id"]),
        "subject_id": int(r.get("subject_id", 0)),
        "stage1_score": float(r["stage1_score"]),
        "stage1_threshold": float(r["stage1_threshold"]),
        "stage2_score": float(r["stage2_score"]),
        "stage2_confirmed": bool(int(r["stage2_confirmed"])),
        "age_band": str(r.get("age_band", "")),
    }


def _get_stage2_threshold(patient: dict) -> float:
    """Return the Stage 2 per-age-group threshold from calibration if available."""
    model_dir = get_model_dir()
    cal_path = model_dir / "stage2_calibration.json"
    if cal_path.exists():
        import json  # pylint: disable=import-outside-toplevel
        cal = json.loads(cal_path.read_text())
        key = band_key(patient["age_band"])
        thresholds: dict = cal.get("thresholds", {})
        if key in thresholds:
            return float(thresholds[key])
    return float(load_config().stage2.threshold)


def _get_patient_feature_row(
    hadm_id: int,
    artifact: dict,
    cfg: AppConfig,
    feature_matrix: pd.DataFrame | None,
) -> pd.Series:
    """Return the feature row for a patient from the feature matrix."""
    matrix = feature_matrix if feature_matrix is not None else load_feature_matrix(cfg, "full")
    feat_cols = artifact["feature_cols"]
    available = [c for c in feat_cols if c in matrix.columns]
    match = matrix[matrix["hadm_id"] == hadm_id]
    if match.empty:
        return pd.Series({c: 0.0 for c in available})
    return match.iloc[0][available]


def _load_note_text(hadm_id: int, subject_id: int, cfg: AppConfig) -> str:
    """Load the discharge note text for one patient. Returns '' on failure."""
    # pylint: disable=import-outside-toplevel  # deferred: importing dataset.py
    # initialises PyTorch/MPS; must happen AFTER the XGBoost artifact is loaded
    from src.stage2.dataset import build_notes_dataframe, load_notes
    try:
        notes_raw = load_notes(cfg, hadm_ids={hadm_id})
        label_df = pd.DataFrame([{
            "hadm_id": hadm_id,
            "subject_id": subject_id,
            TARGET_COL: 0,  # label not needed for explanation
        }])
        notes_df = build_notes_dataframe(notes_raw, label_df)
        if notes_df.empty:
            return ""
        return str(notes_df.iloc[0]["text"])
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


def _get_attention(
    hadm_id: int,
    note_text: str,
    cfg: AppConfig,
) -> list[str]:
    """Extract attention sentences for one patient. Returns [] on failure."""
    if not cfg.stage3.attention_extraction or not note_text.strip():
        return []
    model_dir = get_model_dir()
    try:
        get_stage2_model_path(model_dir)
    except FileNotFoundError:
        return []
    spans = extract_attention_spans(
        [hadm_id],
        [note_text],
        model_dir,
        top_n=cfg.stage3.top_attention_sentences,
        max_length=cfg.stage2.max_seq_length,
    )
    return spans.get(hadm_id, [])


# ── Public API ─────────────────────────────────────────────────────────────────

def explain_patient(
    hadm_id: int,
    cfg: AppConfig,
    *,
    results_df: pd.DataFrame | None = None,
    artifact: dict | None = None,
    feature_matrix: pd.DataFrame | None = None,
) -> ExplanationResult:
    """Generate a Stage 3 explanation for one patient on demand.

    All heavy objects (artifact, results_df, feature_matrix) can be pre-loaded
    by the caller and passed in as keyword arguments — the API server does this
    at startup so each explanation request does not pay the load cost again.
    When ``None``, each is loaded fresh from disk.

    Args:
        hadm_id:        Hospital admission ID to explain.
        cfg:            Validated project configuration.
        results_df:     Pre-loaded Stage 2 results DataFrame (optional).
        artifact:       Pre-loaded Stage 1 XGBoost artifact dict (optional).
        feature_matrix: Pre-loaded full feature matrix (optional).

    Returns:
        :class:`ExplanationResult` with all fields populated.

    Raises:
        FileNotFoundError: if required model files are missing.
        KeyError:          if the patient is not in Stage 2 results.
    """
    # Load XGBoost artifact FIRST — importing torch after joblib on macOS ARM
    # causes a MPS/XGBoost C-extension conflict (SIGSEGV).
    loaded_artifact = artifact if artifact is not None else _load_artifact(cfg)

    df = results_df if results_df is not None else _load_results(get_model_dir())
    patient = _lookup_patient(hadm_id, df)

    feature_row = _get_patient_feature_row(
        hadm_id, loaded_artifact, cfg, feature_matrix
    )
    shap_strings = extract_shap_for_patient(
        loaded_artifact, feature_row, top_k=cfg.stage3.top_shap_features
    )

    note_text = _load_note_text(hadm_id, patient["subject_id"], cfg)
    attention_sentences = _get_attention(hadm_id, note_text, cfg)

    stage2_threshold = _get_stage2_threshold(patient)
    prompt = build_prompt(
        stage1_score=patient["stage1_score"],
        stage1_threshold=patient["stage1_threshold"],
        stage2_score=patient["stage2_score"],
        stage2_threshold=stage2_threshold,
        stage2_confirmed=patient["stage2_confirmed"],
        shap_feature_strings=shap_strings,
        attention_sentences=attention_sentences,
        note_text=note_text,
    )
    annotation = call_phi4mini(prompt, cfg)

    return ExplanationResult(
        hadm_id=patient["hadm_id"],
        stage1_score=patient["stage1_score"],
        stage1_threshold=patient["stage1_threshold"],
        stage2_score=patient["stage2_score"],
        stage2_confirmed=patient["stage2_confirmed"],
        score_delta=patient["stage2_score"] - patient["stage1_score"],
        top_shap_features=shap_strings,
        attention_sentences=attention_sentences,
        discordance_mode=annotation["discordance_mode"],
        primary_category=annotation["primary_category"],
        narrative=annotation["narrative"],
        annotation_failed=annotation["annotation_failed"],
    )


def main() -> None:
    """CLI entry point: explain one patient by hadm_id."""
    import argparse  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description="Stage 3: explain one patient")
    parser.add_argument("hadm_id", type=int, help="Hospital admission ID")
    args = parser.parse_args()

    cfg = load_config()
    result = explain_patient(args.hadm_id, cfg)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
