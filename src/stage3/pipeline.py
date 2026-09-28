"""Stage 3 on-demand audit for one patient.

This is the sole entry point for Stage 3.  It is called by the API when a
clinician requests review of a specific flagged patient — never in batch.

The function assembles evidence from all three layers and lets MedGemma-27B
reach its own decision:

* **Stage 1** (XGBoost on structured EHR): the risk score and SHAP-ranked
  reasons for the flag.
* **Stage 2** (Clinical-Longformer on the discharge note): an
  independently-derived, note-only risk estimate — evidence for the auditor,
  not a decision it explains.
* **Stage 3** (MedGemma-27B via HF ``transformers.generate()`` +
  lm-format-enforcer): reads the discharge note plus both scores, extracts
  mitigating/aggravating grounds with quotes, and returns its own
  uphold/override/insufficient_evidence judgment with a justification —
  alongside ``decision_rule``, the same decision recomputed deterministically
  in code from the extracted grounds.

Usage::

    from src.stage3.pipeline import explain_patient
    from src.config import load_config

    cfg = load_config()
    result = explain_patient(hadm_id=20185601, cfg=cfg)
    print(result.decision_model, result.clinical_justification)
"""

from __future__ import annotations

from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
from src.schemas import MODEL_TARGET_COL
from src.stage2._utils import get_stage2_model_path
from src.stage3.attention import extract_attention_spans
from src.stage3.discordance import compute_discordance
from src.stage3.explain import build_prompt, call_llm, is_note_truncated
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


def _cohort_scores(results_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (stage1_scores, stage2_scores) for the flagged+noted cohort.

    This is the reference population percentile ranks are computed against —
    every admission that reached Stage 2 (has a ``stage2_score``), i.e. the
    same population :func:`src.model.evaluate_pipeline._eval_stage2` calls the
    "notes cohort".
    """
    cohort = results_df.dropna(subset=["stage2_score"])
    return (
        cohort["stage1_score"].to_numpy(dtype=float),
        cohort["stage2_score"].to_numpy(dtype=float),
    )


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
            MODEL_TARGET_COL: 0,  # label not needed for explanation
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

class _PreparedPatient(NamedTuple):
    """Everything :func:`explain_patient` computes before calling the LLM.

    Split out 2026-09-15 so :mod:`src.stage3.batch` can prepare a whole
    chunk of patients, submit all their prompts to
    :func:`src.stage3.explain.call_llm_batch` together in one batched
    ``generate()`` call, then assemble each result -- instead of one
    patient at a time. See :func:`_prepare_patient` / :func:`_assemble_result`.
    """

    patient: dict
    discordance: dict
    shap_strings: list[str]
    attention_sentences: list[str]
    note_text: str
    prompt: str
    prompt_note_text: str
    resolved_model_name: str


def _prepare_patient(
    hadm_id: int,
    cfg: AppConfig,
    *,
    results_df: pd.DataFrame | None = None,
    artifact: dict | None = None,
    feature_matrix: pd.DataFrame | None = None,
    note_text: str | None = None,
    model_name: str | None = None,
    suppress_note: bool = False,
    suppress_stage2: bool = False,
) -> _PreparedPatient:
    """Assemble one patient's evidence and build their prompt -- everything
    :func:`explain_patient` does up to (not including) the LLM call.

    All heavy objects (artifact, results_df, feature_matrix, note_text) can be
    pre-loaded by the caller and passed in as keyword arguments — the API
    server does this at startup so each explanation request does not pay the
    load cost again. When ``None``, each is loaded fresh from disk.
    ``note_text`` specifically matters for batch use: it defaults to loading
    via ``load_notes(cfg, hadm_ids={hadm_id})``, a full chunked scan of the
    MIMIC-IV-Note file for a single admission — fine for one-at-a-time API
    requests, but re-scanning the whole file per admission in a loop (as
    `batch.py` was doing before this parameter existed) does not scale to a
    ~9,800-admission batch. `batch.py` now loads all needed notes once
    upfront and passes each one in here directly.

    Args:
        hadm_id:        Hospital admission ID to explain.
        cfg:            Validated project configuration.
        results_df:     Pre-loaded Stage 2 results DataFrame (optional).
        artifact:       Pre-loaded Stage 1 XGBoost artifact dict (optional).
        feature_matrix: Pre-loaded full feature matrix (optional).
        note_text:      Pre-loaded discharge note text for this admission
                        (optional) — see above.
        model_name:     local model path to audit with. Defaults to
                        ``cfg.stage3.model_name``. Pass
                        ``cfg.stage3.robustness_model`` to run the same
                        patient through the scale-robustness arm instead.
        suppress_note:  blind-note validation control (session 19 Phase D1)
                        — withhold the discharge note (and attention spans)
                        from the prompt. The returned result still reports
                        the real stage1_score/stage2_score/etc. for
                        comparison against the real audit of the same
                        patient; only what the LLM was shown differs.
        suppress_stage2: no-Stage-2 validation control (Phase D2) — withhold
                        Stage 2's score and the discordance section from the
                        prompt. Same real-values-reported caveat as above.

    Returns:
        :class:`_PreparedPatient` ready for :func:`call_llm`/
        :func:`~src.stage3.explain.call_llm_batch` and then
        :func:`_assemble_result`.

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

    note_text = (
        note_text if note_text is not None
        else _load_note_text(hadm_id, patient["subject_id"], cfg)
    )
    attention_sentences = _get_attention(hadm_id, note_text, cfg)

    cohort_s1, cohort_s2 = _cohort_scores(df)
    discordance = compute_discordance(
        patient["stage1_score"], patient["stage2_score"],
        cohort_s1, cohort_s2,
        displacement_pp=cfg.stage3.discordance_displacement_pp,
    )

    # Validation controls (session 19): what the LLM is SHOWN differs; what
    # gets reported (stage1/stage2 scores, discordance) stays the real
    # values, so the control's output is comparable to the real audit of the
    # same patient. suppress_note also blanks note_text end to end (prompt
    # AND quote-verification/decision_rule input) -- simulates "this
    # admission had no note", not "the model saw a note but pretend it didn't".
    prompt_note_text = "" if suppress_note else note_text
    prompt_attention = [] if suppress_note else attention_sentences

    prompt = build_prompt(
        stage1_score=patient["stage1_score"],
        stage1_threshold=patient["stage1_threshold"],
        stage2_score=patient["stage2_score"],
        discordance=discordance,
        shap_feature_strings=shap_strings,
        note_text=prompt_note_text,
        attention_sentences=prompt_attention,
        hide_stage2=suppress_stage2,
    )
    resolved_model_name = model_name or cfg.stage3.model_name

    return _PreparedPatient(
        patient=patient,
        discordance=discordance,
        shap_strings=shap_strings,
        attention_sentences=attention_sentences,
        note_text=note_text,
        prompt=prompt,
        prompt_note_text=prompt_note_text,
        resolved_model_name=resolved_model_name,
    )


def _assemble_result(
    prepared: _PreparedPatient, annotation: dict
) -> ExplanationResult:
    """Build the final :class:`ExplanationResult` from a prepared patient
    plus the LLM's parsed annotation. The second half of what
    :func:`explain_patient` does in one shot — see :func:`_prepare_patient`.
    """
    patient = prepared.patient
    discordance = prepared.discordance
    return ExplanationResult(
        hadm_id=patient["hadm_id"],
        stage1_score=patient["stage1_score"],
        stage1_threshold=patient["stage1_threshold"],
        stage2_score=patient["stage2_score"],
        stage2_confirmed=patient["stage2_confirmed"],
        r1=discordance["r1"],
        r2=discordance["r2"],
        displacement=discordance["displacement"],
        discordance_mode=str(discordance["mode"]),
        top_shap_features=prepared.shap_strings,
        attention_sentences=prepared.attention_sentences,
        mitigating_grounds=annotation["mitigating_grounds"],
        aggravating_grounds=annotation["aggravating_grounds"],
        all_quotes_verified=annotation["all_quotes_verified"],
        planned_return=annotation["planned_return"],
        clinical_justification=annotation["clinical_justification"],
        decision_model=annotation["decision_model"],
        decision_rule=annotation["decision_rule"],
        note_truncated=is_note_truncated(prepared.note_text),
        model_name=prepared.resolved_model_name,
        annotation_failed=annotation["annotation_failed"],
    )


def explain_patient(
    hadm_id: int,
    cfg: AppConfig,
    *,
    results_df: pd.DataFrame | None = None,
    artifact: dict | None = None,
    feature_matrix: pd.DataFrame | None = None,
    note_text: str | None = None,
    model_name: str | None = None,
    suppress_note: bool = False,
    suppress_stage2: bool = False,
) -> ExplanationResult:
    """Generate a Stage 3 explanation for one patient on demand.

    Does prepare -> call -> assemble in sequence for this single patient —
    unchanged behavior/signature from before the 2026-09-15 batching split
    (see :func:`_prepare_patient`). ``batch.py`` calls the two halves
    directly instead, so it can batch the LLM call across many patients.

    Args:
        hadm_id:        Hospital admission ID to explain.
        cfg:            Validated project configuration.
        results_df:     Pre-loaded Stage 2 results DataFrame (optional).
        artifact:       Pre-loaded Stage 1 XGBoost artifact dict (optional).
        feature_matrix: Pre-loaded full feature matrix (optional).
        note_text:      Pre-loaded discharge note text for this admission
                        (optional).
        model_name:     local model path to audit with. Defaults to
                        ``cfg.stage3.model_name``.
        suppress_note:  blind-note validation control (session 19 Phase D1).
        suppress_stage2: no-Stage-2 validation control (Phase D2).

    Returns:
        :class:`ExplanationResult` with all fields populated.

    Raises:
        FileNotFoundError: if required model files are missing.
        KeyError:          if the patient is not in Stage 2 results.
    """
    prepared = _prepare_patient(
        hadm_id, cfg,
        results_df=results_df, artifact=artifact, feature_matrix=feature_matrix,
        note_text=note_text, model_name=model_name,
        suppress_note=suppress_note, suppress_stage2=suppress_stage2,
    )
    annotation = call_llm(
        prepared.prompt, cfg,
        model_name=prepared.resolved_model_name, note_text=prepared.prompt_note_text,
    )
    return _assemble_result(prepared, annotation)


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
