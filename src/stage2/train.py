"""Fine-tune Clinical-Longformer for Stage 2 false-positive pruning.

Key features
------------
- 2048-token sequences with **head+tail truncation** (first 1024 + last 1024
  tokens) — preserves both the admission summary and the discharge plan.
- **Structured feature fusion**: Stage 1 probability score + top structured
  features (age, LOS, Charlson, etc.) fed through a small MLP, concatenated
  with the Longformer [CLS] embedding before the classifier head.
- Age × label stratified training sample — 70+ oversampled.
- Focal loss (gamma=2.0) — upweights hard examples (elderly ambiguous notes).
- Per-age-group loss multipliers — direct fairness pressure during training.
- Proper patient-level splits: finetune / validation / calibration.
- Gradient checkpointing support for smaller GPUs.
- bf16 / fp16 switches for GPU precision.
- Automatic crash recovery: resumes from the latest checkpoint if the job is
  restarted.

Requires:

- Stage 1 artifact: ``models/stage1_<model>.joblib``
- Stage 2 splits:   ``data/processed/stage2_*_hadm_ids.csv`` (run splits.py first)
- MIMIC-IV-Note:    ``MIMIC_IV_NOTE_DIR`` set in ``.env``

Usage::

    python -m src.stage2.train
    python setup_stage2.py --mode full   (recommended)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score
    from transformers import (
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError as _torch_err:
    raise ImportError(
        "Stage 2 requires PyTorch and Transformers. "
        "Install with: pip install torch transformers"
    ) from _torch_err

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.schemas import TARGET_COL
from src.stage2._utils import STRUCT_FEATURE_COLS, band_key
from src.stage2.dataset import (
    ClinicalNotesDataset,
    build_notes_dataframe,
    load_notes,
    normalize_struct_features,
)
from src.stage2.model import FusionLongformer
from src.stage2.splits import build_splits


# ── Hyperparameter bundle ─────────────────────────────────────────────────────

@dataclass
class _TrainHparams:
    """All Stage 2 training hyperparameters extracted from config."""

    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float
    grad_accum: int
    max_length: int
    use_focal: bool
    focal_gamma: float
    group_lw: dict
    age_targets: dict
    patience: int
    warmup_ratio: float
    weight_decay: float
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    num_workers: int
    save_steps: int


def _parse_hparams(cfg: AppConfig) -> _TrainHparams:
    """Extract all Stage 2 hyperparameters from config into a typed bundle."""
    s2 = cfg.stage2
    return _TrainHparams(
        model_name=s2.model_name,
        batch_size=s2.batch_size,
        epochs=s2.epochs,
        learning_rate=s2.learning_rate,
        grad_accum=s2.gradient_accumulation_steps,
        max_length=s2.max_seq_length,
        use_focal=s2.focal_loss,
        focal_gamma=s2.focal_gamma,
        group_lw=s2.age_group_loss_weights,
        age_targets=s2.age_group_train_targets,
        patience=s2.early_stopping_patience,
        warmup_ratio=s2.warmup_ratio,
        weight_decay=s2.weight_decay,
        gradient_checkpointing=s2.gradient_checkpointing,
        fp16=s2.fp16,
        bf16=s2.bf16,
        num_workers=s2.dataloader_num_workers,
        save_steps=s2.save_steps,
    )


# ── Structured feature helpers ────────────────────────────────────────────────

class _StructData(NamedTuple):
    """Normalised structured feature arrays + scaler statistics."""

    train: np.ndarray
    val: np.ndarray
    scaler: dict


def _make_struct_data(
    ft_notes: pd.DataFrame,
    train_df: pd.DataFrame,
    val_notes: pd.DataFrame,
) -> _StructData:
    """Fit z-score scaler on the fine-tune set and transform train + val.

    Args:
        ft_notes:  full fine-tune notes DataFrame (pre-sampling; scaler is fit
                   on the full distribution, not the sampled subset).
        train_df:  age-stratified sampled fine-tune notes (passed to training).
        val_notes: validation notes DataFrame.

    Returns:
        :class:`_StructData` with normalised arrays and the scaler dict.
    """
    _, means, stds = normalize_struct_features(ft_notes)
    train_arr, _, _ = normalize_struct_features(train_df, means, stds)
    val_arr, _, _ = normalize_struct_features(val_notes, means, stds)
    return _StructData(train=train_arr, val=val_arr, scaler={"means": means, "stds": stds})


# ── Data helpers ──────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred: tuple) -> dict:
    """Compute AUPRC and AUROC from Trainer eval predictions."""
    logits, labels = eval_pred
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    )[:, 1].numpy()
    auprc = float(average_precision_score(labels, probs))
    try:
        auroc_val = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc_val = 0.0
    return {"auprc": auprc, "auroc": auroc_val}


def _build_age_stratified_sample(
    notes_df: pd.DataFrame,
    targets: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample notes_df by age_band × label cell according to target counts.

    Args:
        notes_df: DataFrame with ``age_band`` and ``TARGET_COL`` columns.
        targets:  dict mapping ``"<band>_<label>"`` keys to sample counts.
        seed:     random seed for reproducibility.

    Returns:
        Shuffled DataFrame of sampled notes (including all original columns).
    """
    parts = []
    for key, n_target in targets.items():
        band_str, label_str = key.rsplit("_", 1)
        label = int(label_str)
        mask = (
            (notes_df["age_band"].apply(band_key) == band_str)
            & (notes_df[TARGET_COL] == label)
        )
        cell = notes_df[mask]
        if len(cell) == 0:
            print(f"  [stage2/train] WARNING: no notes for cell {key}")
            continue
        n_take = min(n_target, len(cell))
        if n_take < n_target:
            print(
                f"  [stage2/train] Cell {key}: {len(cell):,} available "
                f"(target {n_target:,}) — using all"
            )
        parts.append(cell.sample(n=n_take, random_state=seed))

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    print(
        f"[stage2/train] Age-stratified sample: {len(result):,} notes "
        f"(readmission rate: {result[TARGET_COL].mean():.1%})"
    )
    for bk in ["18-40", "41-55", "56-70", "70+"]:
        sub = result[result["age_band"].apply(band_key) == bk]
        if len(sub):
            print(f"  {bk:<8} {len(sub):>6,}  ({sub[TARGET_COL].mean():.1%} readmit)")
    return result


def _load_split_notes(
    cfg: AppConfig,
    artifact: dict,
    ft_hadm_ids: set,
    val_hadm_ids: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split discharge notes, augmented with Stage 1 scores + struct features.

    Computes Stage 1 probability scores for the training partition and joins
    them together with the structured feature columns from the feature matrix
    so that both are available for :class:`FusionLongformer` training.

    Args:
        cfg:          validated project config.
        artifact:     pre-loaded Stage 1 artifact.
        ft_hadm_ids:  hadm_ids belonging to the finetune split.
        val_hadm_ids: hadm_ids belonging to the validation split.

    Returns:
        Tuple ``(ft_notes, val_notes)`` — each a notes DataFrame with
        ``age_band``, ``text``, ``TARGET_COL``, ``stage1_score``, and any
        available structured feature columns.
    """
    matrix = load_feature_matrix(cfg, artifact["mode"])
    all_feats = split_xy(matrix)[0]
    train_idx = artifact["train_idx"]

    # Compute Stage 1 probability scores for the training partition
    x_train = all_feats.iloc[train_idx][artifact["feature_cols"]]
    s1_scores = artifact["estimator"].predict_proba(x_train)[:, 1]

    struct_cols = [c for c in STRUCT_FEATURE_COLS if c in matrix.columns]
    labels_df = matrix.iloc[train_idx][
        ["hadm_id", "subject_id", "age_band", TARGET_COL] + struct_cols
    ].copy().reset_index(drop=True)
    labels_df["stage1_score"] = s1_scores

    all_needed = ft_hadm_ids | val_hadm_ids
    print(f"[stage2/train] Loading notes for {len(all_needed):,} admissions ...")
    notes = load_notes(cfg, hadm_ids=all_needed)

    merge_cols = ["hadm_id", "age_band"] + struct_cols + ["stage1_score"]

    def _build(hadm_ids: set) -> pd.DataFrame:
        sub = labels_df[labels_df["hadm_id"].isin(hadm_ids)]
        df = build_notes_dataframe(notes, sub[["hadm_id", "subject_id", TARGET_COL]])
        return df.merge(sub[merge_cols], on="hadm_id", how="left")

    return _build(ft_hadm_ids), _build(val_hadm_ids)


# ── Training setup helpers ────────────────────────────────────────────────────

def _detect_device() -> str:
    """Return the best available device identifier string."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _find_resume_checkpoint(checkpoint_dir) -> str | None:
    """Return the path of the latest checkpoint, or None if none exist.

    Called before ``trainer.train()`` to enable automatic crash recovery —
    if training is interrupted, restarting the script resumes from the last
    saved checkpoint rather than starting from scratch.
    """
    if checkpoint_dir.exists():
        existing = sorted(checkpoint_dir.glob("checkpoint-*"))
        if existing:
            print(f"[stage2/train] Resuming from checkpoint: {existing[-1].name}")
            return str(existing[-1])
    print("\n[stage2/train] Starting fine-tuning from scratch ...")
    return None


def _compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Return a [neg_weight, pos_weight] tensor for cross-entropy loss."""
    pos = int((train_df[TARGET_COL] == 1).sum())
    neg = int((train_df[TARGET_COL] == 0).sum())
    print(f"[stage2/train] Class weights: neg=1.0  pos={neg / pos:.2f}")
    return torch.tensor([1.0, neg / pos], dtype=torch.float32)


def _build_training_args(
    hp: _TrainHparams,
    model_dir,
    device: str,
    seed: int,
) -> TrainingArguments:
    """Construct HuggingFace TrainingArguments from hyperparameters.

    Uses step-level checkpointing (every ``save_steps`` optimizer steps) so
    that a crash mid-epoch loses at most ``save_steps`` steps of progress.
    The last 3 checkpoints are kept; the best is restored at the end.
    """
    use_fp16 = hp.fp16 and device == "cuda"
    use_bf16 = hp.bf16 and device == "cuda"
    print(
        f"[stage2/train] Device: {device}  fp16={use_fp16}  bf16={use_bf16}  "
        f"checkpoint every {hp.save_steps} steps"
    )
    return TrainingArguments(
        output_dir=str(model_dir / "stage2_checkpoints"),
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=hp.batch_size * 2,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=hp.learning_rate,
        warmup_ratio=hp.warmup_ratio,
        weight_decay=hp.weight_decay,
        evaluation_strategy="steps",
        eval_steps=hp.save_steps,
        save_strategy="steps",
        save_steps=hp.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=(device == "cpu"),
        seed=seed,
        report_to="none",
        logging_steps=50,
        dataloader_num_workers=hp.num_workers,
    )


def _save_results(
    trainer: Trainer,
    val_dataset: ClinicalNotesDataset,
    model_dir,
    tokenizer: object,
    scaler_stats: dict,
) -> None:
    """Save the best model, tokenizer, struct scaler, fusion config, and val metrics."""
    save_path = model_dir / "stage2_longformer_best"

    # Save Longformer backbone (config + weights) for standalone loading
    trainer.model.longformer.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]

    # Full model state dict (fine-tuned Longformer + struct MLP + classifier head)
    torch.save(trainer.model.state_dict(), str(save_path / "fusion_weights.pt"))

    # Fusion config needed to reconstruct the model at inference time
    fusion_cfg = {
        "n_struct_features": trainer.model.n_struct_features,
        "num_labels": trainer.model.classifier.out_features,
    }
    (save_path / "stage2_fusion_config.json").write_text(json.dumps(fusion_cfg))

    # Normalisation stats for reproducing the same scaling at inference
    (model_dir / "stage2_struct_scaler.json").write_text(json.dumps(scaler_stats))

    print(f"[stage2/train] Saved fusion model -> {save_path}")

    results = trainer.evaluate(val_dataset)
    auprc_val = results.get("eval_auprc", float("nan"))
    auroc_val = results.get("eval_auroc", float("nan"))
    print(f"[stage2/train] Val AUPRC={auprc_val:.4f}  AUROC={auroc_val:.4f}")

    metrics_path = model_dir / "stage2_train_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"[stage2/train] Saved metrics -> {metrics_path}")


# ── Main training function ────────────────────────────────────────────────────

def train_stage2(cfg: AppConfig, artifact: dict | None = None) -> None:
    """Fine-tune FusionLongformer for Stage 2.

    Args:
        cfg:      validated project config.
        artifact: pre-loaded Stage 1 artifact. If ``None``, loaded from disk.
                  Pass a pre-loaded artifact to avoid the joblib/torch conflict
                  on macOS.
    """
    hp = _parse_hparams(cfg)
    seed = cfg.run.random_state
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    if artifact is None:
        artifact_path = model_dir / f"stage1_{cfg.stage1.model}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Stage 1 artifact not found at {artifact_path}. "
                "Run `python -m src.model.train` first."
            )
        artifact = joblib.load(artifact_path)

    print("[stage2/train] Loading patient-level splits ...")
    splits = build_splits(cfg, artifact=artifact)
    ft_hadm_ids = set(splits["finetune"]["hadm_id"].astype(int))
    val_hadm_ids = set(splits["val"]["hadm_id"].astype(int))

    ft_notes, val_notes = _load_split_notes(cfg, artifact, ft_hadm_ids, val_hadm_ids)

    train_df = (
        _build_age_stratified_sample(ft_notes, hp.age_targets, seed)
        if hp.age_targets else ft_notes
    )
    if not hp.age_targets:
        print(f"[stage2/train] No age targets — using all {len(train_df):,} fine-tune notes")

    # Normalise structured features: fit on full fine-tune set, apply to train + val
    struct_data = _make_struct_data(ft_notes, train_df, val_notes)

    print(f"[stage2/train] Train: {len(train_df):,}  |  Val: {len(val_notes):,}")
    print(f"[stage2/train] Struct features: {struct_data.train.shape[1]} columns")

    train_gw = (
        [hp.group_lw.get(band_key(b), 1.0) for b in train_df["age_band"]]
        if hp.group_lw else [1.0] * len(train_df)
    )

    print(f"[stage2/train] Loading tokenizer from '{hp.model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)

    train_dataset = ClinicalNotesDataset(
        train_df["text"].tolist(), train_df[TARGET_COL].tolist(),
        tokenizer, hp.max_length,
        struct_features=struct_data.train,
        group_weights=train_gw,
    )
    val_dataset = ClinicalNotesDataset(
        val_notes["text"].tolist(), val_notes[TARGET_COL].tolist(),
        tokenizer, hp.max_length,
        struct_features=struct_data.val,
    )

    print(f"[stage2/train] Loading FusionLongformer from '{hp.model_name}' ...")
    model = FusionLongformer(hp.model_name, n_struct_features=struct_data.train.shape[1])
    if hp.gradient_checkpointing:
        model.longformer.gradient_checkpointing_enable()
        print("[stage2/train] Gradient checkpointing enabled")

    class_weights = _compute_class_weights(train_df)
    print(
        f"[stage2/train]   focal={'yes' if hp.use_focal else 'no'}  "
        f"gamma={hp.focal_gamma}"
    )

    device = _detect_device()
    training_args = _build_training_args(hp, model_dir, device, seed)

    class FocalGroupWeightedTrainer(Trainer):
        """Custom Trainer applying focal loss and per-group loss weights."""

        def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
            """Compute focal loss with optional per-sample group weighting."""
            labels = inputs.pop("labels")
            group_weight = inputs.pop("group_weight", None)
            outputs = model(**inputs)
            logits = outputs.logits

            ce = F.cross_entropy(
                logits, labels,
                weight=class_weights.to(logits.device),
                reduction="none",
            )
            if hp.use_focal:
                pt = torch.exp(-ce)
                ce = ((1.0 - pt) ** hp.focal_gamma) * ce
            if group_weight is not None:
                ce = ce * group_weight.to(logits.device)

            loss = ce.mean()
            return (loss, outputs) if return_outputs else loss

    trainer = FocalGroupWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp.patience)],
    )

    trainer.train(
        resume_from_checkpoint=_find_resume_checkpoint(model_dir / "stage2_checkpoints")
    )

    _save_results(trainer, val_dataset, model_dir, tokenizer, struct_data.scaler)
    print("[stage2/train] Done. Run calibrate.py next.")


def main() -> None:
    """CLI entry point for Stage 2 fine-tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()
    _cfg = load_config()
    if args.mode:
        _cfg.run.mode = args.mode
    train_stage2(_cfg)


if __name__ == "__main__":
    main()
