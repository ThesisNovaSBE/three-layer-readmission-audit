"""Pydantic v2 models for the project configuration (config.yaml).

All fields mirror the YAML structure exactly. Pydantic validates types and
provides attribute access instead of raw dict key lookups, which catches
misconfiguration at load time rather than at runtime.

Usage::

    from src.config import load_config
    cfg = load_config()
    print(cfg.stage2.model_name)   # "yikuan8/Clinical-Longformer"
    print(cfg.run.mode)            # "quick" | "full"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DataConfig(BaseModel):
    """Paths to MIMIC-IV data and synthetic-data parameters."""

    model_config = ConfigDict(frozen=False)

    mimic_iv_dir: str = ""
    mimic_iv_note_dir: str = ""
    synthetic_n_patients: int = 2000
    synthetic_seed: int = 42


class RunModeConfig(BaseModel):
    """Parameters that differ between quick (dev) and full (production) runs."""

    model_config = ConfigDict(frozen=False)

    n_rows: int = 0
    optuna_trials: int = 15
    cv_folds: int = 3


class RunConfig(BaseModel):
    """Top-level run settings."""

    model_config = ConfigDict(frozen=False)

    mode: str = "quick"
    random_state: int = 42
    quick: RunModeConfig = RunModeConfig()
    full: RunModeConfig = RunModeConfig(n_rows=0, optuna_trials=150, cv_folds=5)

    def active(self, mode: str | None = None) -> RunModeConfig:
        """Return the mode-specific config for the given (or current) mode."""
        effective = mode or self.mode
        return self.quick if effective == "quick" else self.full


class CohortConfig(BaseModel):
    """Inclusion / exclusion criteria for the patient cohort."""

    model_config = ConfigDict(frozen=False)

    readmission_window_days: int = 30
    min_age: int = 18
    exclude_in_hospital_death: bool = True
    exclude_elective: bool = True


class FeaturesConfig(BaseModel):
    """Feature engineering settings."""

    model_config = ConfigDict(frozen=False)

    lab_items: list[str] = []
    vital_items: list[str] = []
    lab_aggregates: list[str] = ["last", "mean", "min", "max"]
    comorbidity_scheme: str = "charlson"


class Stage1Config(BaseModel):
    """Stage 1 classical ML settings."""

    model_config = ConfigDict(frozen=False)

    model: str = "xgboost"
    models: list[str] = ["logistic_regression", "xgboost", "histgradientboosting"]
    target_recall: float = 0.85
    recall_report_points: list[float] = [0.80, 0.85, 0.90]
    test_size: float = 0.2
    use_smote: bool = False
    early_stopping_rounds: int = 50
    n_estimators_cap: int = 2000


class Stage2Config(BaseModel):
    """Stage 2 Clinical-Longformer fine-tuning settings."""

    model_config = ConfigDict(frozen=False)

    model_name: str = "yikuan8/Clinical-Longformer"
    max_seq_length: int = 2048
    val_fraction: float = 0.20
    calibration_fraction: float = 0.20
    age_group_train_targets: dict[str, int] = {}
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    early_stopping_patience: int = 3
    learning_rate: float = 1.0e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    save_steps: int = 500
    focal_loss: bool = True
    focal_gamma: float = 2.0
    age_group_loss_weights: dict[str, float] = {}
    calibration_method: str = "platt"
    threshold_strategy: str = "recall_floor"
    recall_floor: float = 0.65
    threshold: float = 0.30


class Stage3Config(BaseModel):
    """Stage 3 LLM explanation settings."""

    model_config = ConfigDict(frozen=False)

    enabled: bool = False
    ollama_model: str = "phi4-mini"
    temperature: float = 0.3
    discordance_analysis: bool = True       # cross-modal concordance / discordance analysis
    attention_extraction: bool = True       # extract Longformer attention spans (needs trained model)
    top_attention_sentences: int = 5        # sentences to extract per patient
    top_shap_features: int = 5              # Stage 1 SHAP features to include per patient


class OutputConfig(BaseModel):
    """Output directory configuration."""

    model_config = ConfigDict(frozen=False)

    model_dir: str = "models"
    log_dir: str = "logs"
    processed_dir: str = "data/processed"


class AppConfig(BaseModel):
    """Root configuration object validated from config.yaml."""

    model_config = ConfigDict(frozen=False)

    data: DataConfig = DataConfig()
    run: RunConfig = RunConfig()
    cohort: CohortConfig = CohortConfig()
    features: FeaturesConfig = FeaturesConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()
    output: OutputConfig = OutputConfig()
