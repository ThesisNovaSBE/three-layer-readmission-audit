"""FastAPI backend for the readmission prediction dashboard.

Start the server::

    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /api/health                         → server status and loaded patient count
GET  /api/patients[?confirmed_only=true] → Stage-2-flagged patients
GET  /api/patients/{hadm_id}             → one patient's full record
POST /api/patients/{hadm_id}/explain     → on-demand Stage 3 clinical explanation

Demo mode
---------
If ``models/stage2_results.csv`` is absent (e.g. Stage 2 not yet run), the
server falls back to deriving patient data from Stage 1 predictions so the
frontend can be tested end-to-end with synthetic data.  ``demo_mode: true``
is set in the ``/api/health`` response so the frontend can display a banner.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
from src.stage2._utils import band_key
from src.stage3.models import ExplanationResult
from src.stage3.pipeline import explain_patient

log = logging.getLogger(__name__)

_CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:4173",  # Vite preview
    "http://localhost:3000",
]



# ── Response models ───────────────────────────────────────────────────────────

class PatientOut(BaseModel):
    """Patient record returned by the list and detail endpoints."""

    hadm_id: int
    age_band: str
    stage1_score: float
    stage1_threshold: float
    stage2_score: float
    stage2_threshold: float
    stage2_confirmed: bool
    readmitted: bool | None = None
    age: float | None = None
    gender: str | None = None
    los_days: float | None = None
    charlson_index: float | None = None
    n_comorbidities: int | None = None
    n_prior_admissions: int | None = None
    days_since_last_discharge: float | None = None
    admission_type: str | None = None
    came_via_ed: bool | None = None
    discharged_to_facility: bool | None = None
    creatinine_last: float | None = None
    white_blood_cells_last: float | None = None


class HealthOut(BaseModel):
    """Health check response."""

    status: str
    patients_loaded: int
    artifact_loaded: bool
    demo_mode: bool


# ── NaN-to-None helpers ───────────────────────────────────────────────────────

def _opt_float(val: object) -> float | None:
    """Return float or None for pandas NaN / Python None."""
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _opt_int(val: object) -> int | None:
    """Return int or None for pandas NaN / Python None."""
    f = _opt_float(val)
    return None if f is None else int(f)


def _opt_bool(val: object) -> bool | None:
    """Return bool or None for pandas NaN / Python None."""
    i = _opt_int(val)
    return None if i is None else bool(i)


def _opt_str(val: object) -> str | None:
    """Return str or None for pandas NaN / Python None / empty string."""
    if val is None:
        return None
    try:
        s = str(val)
        return None if s in ("nan", "None", "") else s
    except (TypeError, ValueError):
        return None


# ── Startup helpers ───────────────────────────────────────────────────────────

def _load_artifact(model_dir: Path) -> dict | None:
    """Load the Stage 1 artifact (tries xgboost, hgb, lr in order).

    Returns None if no artifact is found — server starts in degraded mode.
    Must be called BEFORE any function that imports torch to avoid the macOS
    ARM MPS + XGBoost joblib SIGSEGV.
    """
    for stem in ("stage1_xgboost", "stage1_histgradientboosting",
                 "stage1_logistic_regression"):
        path = model_dir / f"{stem}.joblib"
        if path.exists():
            log.info("[api] Loaded artifact: %s", path.name)
            return joblib.load(path)  # type: ignore[no-any-return]
    log.warning("[api] No Stage 1 artifact found in %s — explain unavailable", model_dir)
    return None


def _load_results(model_dir: Path) -> pd.DataFrame | None:
    """Load Stage 2 results CSV, or None if not yet generated."""
    path = model_dir / "stage2_results.csv"
    if not path.exists():
        log.warning("[api] stage2_results.csv not found — activating demo mode")
        return None
    df = pd.read_csv(path)
    log.info("[api] Loaded %d Stage 2 results", len(df))
    return df


def _load_thresholds(model_dir: Path) -> dict[str, float]:
    """Load per-age-band Stage 2 decision thresholds from calibration JSON."""
    path = model_dir / "stage2_calibration.json"
    if path.exists():
        return json.loads(path.read_text()).get("thresholds", {})
    return {}


def _load_feature_matrix_safe(cfg: AppConfig) -> pd.DataFrame | None:
    """Load the feature matrix, returning None on any failure."""
    try:
        df = load_feature_matrix(cfg, "full")
        log.info("[api] Feature matrix loaded: %d rows", len(df))
        return df
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning("[api] Feature matrix unavailable: %s", exc)
        return None


def _demo_results(
    artifact: dict,
    feature_matrix: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Derive synthetic Stage 2 results from Stage 1 predictions.

    Used when stage2_results.csv is absent.  Stage 2 score is set equal to
    Stage 1 score so the delta is always 0 — this is honest (no real note
    model ran) and still lets the full UI flow be tested with synthetic data.
    """
    feat_cols = artifact["feature_cols"]
    available = [c for c in feat_cols if c in feature_matrix.columns]
    x_mat = feature_matrix[available].fillna(0).to_numpy()
    scores = artifact["estimator"].predict_proba(x_mat)[:, 1]
    thr = float(artifact.get("threshold", cfg.stage1.target_recall))
    if "age_band" in feature_matrix.columns:
        age_band_col = feature_matrix["age_band"].astype(str)
    else:
        age_band_col = pd.Series(["unknown"] * len(feature_matrix), index=feature_matrix.index)
    result = pd.DataFrame({
        "hadm_id": feature_matrix["hadm_id"],
        "subject_id": feature_matrix["subject_id"] if "subject_id" in feature_matrix.columns else 0,
        "age_band": age_band_col,
        "stage1_score": scores,
        "stage1_threshold": thr,
        "stage2_score": scores,
        "stage2_confirmed": (scores >= thr).astype(int),
    })
    return result[result["stage2_confirmed"] == 1].reset_index(drop=True)


def _get_threshold(age_band: str, thresholds: dict[str, float], cfg: AppConfig) -> float:
    """Return the Stage 2 decision threshold for a given age band."""
    key = band_key(age_band)
    return float(thresholds.get(key, cfg.stage2.threshold))


def _build_patients_df(
    results_df: pd.DataFrame,
    feature_matrix: pd.DataFrame | None,
    thresholds: dict[str, float],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Merge Stage 2 results with feature-matrix columns for the patient list."""
    if feature_matrix is None or feature_matrix.empty:
        out = results_df.copy()
    else:
        existing = set(results_df.columns)
        feat_cols = ["hadm_id"] + [
            c for c in feature_matrix.columns if c not in existing and c != "hadm_id"
        ]
        out = results_df.merge(feature_matrix[feat_cols], on="hadm_id", how="left")
    if "age_band" not in out.columns:
        if "age" in out.columns:
            out["age_band"] = pd.cut(
                out["age"], bins=[17, 40, 55, 70, 120],
                labels=["18-40", "41-55", "56-70", "70+"],
            ).astype(str)
        else:
            out["age_band"] = "unknown"
    out["stage2_threshold"] = out["age_band"].astype(str).apply(
        lambda ab: _get_threshold(ab, thresholds, cfg)
    )
    return out


def _row_to_patient(row: pd.Series) -> PatientOut:
    """Convert one merged DataFrame row to a PatientOut Pydantic model."""
    age_band = _opt_str(row.get("age_band")) or ""
    admission_type = "EMERGENCY" if _opt_bool(row.get("admission_type_emergency")) else "OTHER"
    return PatientOut(
        hadm_id=int(row["hadm_id"]),
        age_band=age_band,
        stage1_score=float(row["stage1_score"]),
        stage1_threshold=float(row["stage1_threshold"]),
        stage2_score=float(row["stage2_score"]),
        stage2_threshold=float(row.get("stage2_threshold", 0.30)),
        stage2_confirmed=bool(int(row["stage2_confirmed"])),
        readmitted=_opt_bool(row.get("readmission_30d")),
        age=_opt_float(row.get("age")),
        gender=_opt_str(row.get("gender")),
        los_days=_opt_float(row.get("los_days")),
        charlson_index=_opt_float(row.get("charlson_index")),
        n_comorbidities=_opt_int(row.get("n_comorbidities")),
        n_prior_admissions=_opt_int(row.get("n_prior_admissions")),
        days_since_last_discharge=_opt_float(row.get("days_since_last_discharge")),
        admission_type=admission_type,
        came_via_ed=_opt_bool(row.get("came_via_ed")),
        discharged_to_facility=_opt_bool(row.get("discharged_to_facility")),
        creatinine_last=_opt_float(row.get("creatinine_last")),
        white_blood_cells_last=_opt_float(row.get("white_blood_cells_last")),
    )


# ── App & startup ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load heavy objects once at startup and store them on app.state."""
    cfg = load_config()
    model_dir = get_model_dir()

    # Artifact MUST load before any torch import — macOS ARM MPS + XGBoost SIGSEGV.
    artifact = _load_artifact(model_dir)
    results_df = _load_results(model_dir)
    thresholds = _load_thresholds(model_dir)
    feature_matrix = _load_feature_matrix_safe(cfg)

    demo_mode = False
    if results_df is None:
        if artifact is not None and feature_matrix is not None:
            results_df = _demo_results(artifact, feature_matrix, cfg)
            demo_mode = True
        else:
            results_df = pd.DataFrame()

    patients_df = (
        _build_patients_df(results_df, feature_matrix, thresholds, cfg)
        if not results_df.empty
        else pd.DataFrame()
    )

    _app.state.cfg = cfg
    _app.state.patients_df = patients_df
    _app.state.artifact = artifact
    _app.state.feature_matrix = feature_matrix if feature_matrix is not None else pd.DataFrame()
    _app.state.thresholds = thresholds
    _app.state.demo_mode = demo_mode

    log.info(
        "[api] Ready — %d patients loaded (demo_mode=%s)", len(patients_df), demo_mode
    )
    yield


app = FastAPI(title="Readmission Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthOut)
def get_health(request: Request) -> HealthOut:
    """Return server status and basic diagnostics."""
    return HealthOut(
        status="ok",
        patients_loaded=len(request.app.state.patients_df),
        artifact_loaded=request.app.state.artifact is not None,
        demo_mode=request.app.state.demo_mode,
    )


@app.get("/api/patients", response_model=list[PatientOut])
def list_patients(request: Request, confirmed_only: bool = True) -> list[PatientOut]:
    """Return Stage-2-processed patients, optionally filtered to confirmed only.

    Args:
        confirmed_only: When ``True`` (default), return only patients where
                        Stage 2 confirmed the Stage 1 flag.  Pass ``false``
                        to see all Stage-2-scored patients.
    """
    df: pd.DataFrame = request.app.state.patients_df
    if df.empty:
        return []
    if confirmed_only and "stage2_confirmed" in df.columns:
        df = df[df["stage2_confirmed"].astype(int) == 1]
    return [_row_to_patient(row) for _, row in df.iterrows()]


@app.get("/api/patients/{hadm_id}", response_model=PatientOut)
def get_patient(hadm_id: int, request: Request) -> PatientOut:
    """Return one patient's full record by hospital admission ID."""
    df: pd.DataFrame = request.app.state.patients_df
    match = df[df["hadm_id"] == hadm_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Patient {hadm_id} not found.")
    return _row_to_patient(match.iloc[0])


@app.post("/api/patients/{hadm_id}/explain", response_model=ExplanationResult)
def explain(hadm_id: int, request: Request) -> ExplanationResult:
    """Generate a Stage 3 on-demand clinical explanation for one patient.

    Calls phi4-mini via Ollama — the first request may take 10–30 s while
    the model loads.  Subsequent requests on a warm model are faster.

    The ``results_df``, ``artifact``, and ``feature_matrix`` pre-loaded at
    startup are passed in to avoid per-request load costs.
    """
    cfg: AppConfig = request.app.state.cfg
    results_df: pd.DataFrame = request.app.state.patients_df
    artifact: dict | None = request.app.state.artifact
    feature_matrix: pd.DataFrame = request.app.state.feature_matrix

    if results_df.empty:
        raise HTTPException(status_code=503, detail="No patient data loaded.")
    if artifact is None:
        raise HTTPException(status_code=503, detail="Stage 1 artifact not loaded.")

    fm_arg = feature_matrix if not feature_matrix.empty else None
    try:
        return explain_patient(
            hadm_id,
            cfg,
            results_df=results_df,
            artifact=artifact,
            feature_matrix=fm_arg,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
