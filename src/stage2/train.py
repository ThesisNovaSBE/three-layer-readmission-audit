"""Fine-tune Clinical-Longformer for Stage 2 false-positive pruning.

Key improvements over v1:

- 2048-token sequences (was 512) — captures full average discharge note
- Age × label stratified training sample — 70+ oversampled
- Focal loss (gamma=2.0) — upweights hard examples (elderly ambiguous notes)
- Per-age-group loss multipliers — direct fairness pressure during training
- Proper patient-level splits: finetune / validation / calibration
- Gradient checkpointing support for smaller GPUs
- bf16 / fp16 switches for GPU precision

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
import inspect
import json
from dataclasses import dataclass

import joblib
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score
    from transformers import (
        AutoTokenizer,
        EarlyStoppingCallback,
        LongformerForSequenceClassification,
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
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL
from src.stage2._utils import band_key
from src.stage2.dataset import ClinicalNotesDataset, build_notes_dataframe, load_notes
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
        Shuffled DataFrame of sampled notes.
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
    """Load and split discharge notes for the finetune and validation sets.

    Args:
        cfg:          validated project config.
        artifact:     pre-loaded Stage 1 artifact.
        ft_hadm_ids:  hadm_ids belonging to the finetune split.
        val_hadm_ids: hadm_ids belonging to the validation split.

    Returns:
        Tuple ``(ft_notes, val_notes)`` — each a notes DataFrame with
        ``age_band``, ``text``, and ``TARGET_COL`` columns.
    """
    matrix = load_feature_matrix(cfg, artifact["mode"])
    labels_df = matrix[["hadm_id", "subject_id", "age_band", TARGET_COL]]

    all_needed = ft_hadm_ids | val_hadm_ids
    print(f"[stage2/train] Loading notes for {len(all_needed):,} admissions ...")
    notes = load_notes(cfg, hadm_ids=all_needed)

    def _build_split_df(hadm_ids: set) -> pd.DataFrame:
        sub = labels_df[labels_df["hadm_id"].isin(hadm_ids)]
        df = build_notes_dataframe(notes, sub[["hadm_id", "subject_id", TARGET_COL]])
        return df.merge(labels_df[["hadm_id", "age_band"]], on="hadm_id", how="left")

    return _build_split_df(ft_hadm_ids), _build_split_df(val_hadm_ids)


# ── Training setup helpers ────────────────────────────────────────────────────

def _detect_device() -> str:
    """Return the best available device identifier string."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _checkpoint_step(path) -> int:
    """Return the global step number encoded in a ``checkpoint-<step>`` dir name."""
    return int(path.name.rsplit("-", 1)[-1])


def _find_resume_checkpoint(checkpoint_dir) -> str | None:
    """Return the path of the latest checkpoint, or None if none exist.

    Called before ``trainer.train()`` to enable automatic crash recovery —
    if training is interrupted, restarting the script resumes from the last
    saved checkpoint rather than starting from scratch.

    Sorts by the numeric step (``_checkpoint_step``), not lexicographically —
    HuggingFace names checkpoints ``checkpoint-<step>`` with no zero-padding,
    so a plain string sort orders "checkpoint-1500" before "checkpoint-500"
    and would silently resume from an older checkpoint than the true latest.
    """
    if checkpoint_dir.exists():
        existing = sorted(checkpoint_dir.glob("checkpoint-*"), key=_checkpoint_step)
        if existing:
            print(f"[stage2/train] Resuming from checkpoint: {existing[-1].name}")
            return str(existing[-1])
    print("\n[stage2/train] Starting fine-tuning from scratch ...")
    return None


def _compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Return a [neg_weight, pos_weight] tensor for cross-entropy loss."""
    pos = int((train_df[TARGET_COL] == 1).sum())
    neg = int((train_df[TARGET_COL] == 0).sum())
    print(
        f"[stage2/train] Class weights: neg=1.0  pos={neg / pos:.2f}"
    )
    return torch.tensor([1.0, neg / pos], dtype=torch.float32)


def _eval_strategy_kwarg() -> str:
    """Return whichever eval-strategy kwarg name this installed Transformers
    version actually accepts.

    The name changed from ``evaluation_strategy`` (<4.41) to
    ``eval_strategy`` (>=4.41), and ``evaluation_strategy`` was REMOVED
    entirely in Transformers 5.x -- hardcoding either name breaks on some
    installed version, and requirements.txt's unpinned upper bound
    (``transformers>=4.36``) means a fresh install can land on either side
    of that line. Detected at runtime instead of guessed, so this can't
    silently break again regardless of which version ends up installed.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


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
        # Step-level eval + save so a crash never loses more than save_steps steps.
        **{_eval_strategy_kwarg(): "steps"},
        eval_steps=hp.save_steps,
        save_strategy="steps",
        save_steps=hp.save_steps,
        save_total_limit=3,          # keep only the 3 most recent checkpoints
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
) -> None:
    """Save the best model, tokenizer, and validation metrics."""
    save_path = model_dir / "stage2_longformer_best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))  # type: ignore[union-attr]
    print(f"[stage2/train] Saved fine-tuned model -> {save_path}")

    results = trainer.evaluate(val_dataset)
    auprc_val = results.get("eval_auprc", float("nan"))
    auroc_val = results.get("eval_auroc", float("nan"))
    print(f"[stage2/train] Val AUPRC={auprc_val:.4f}  AUROC={auroc_val:.4f}")

    metrics_path = model_dir / "stage2_train_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"[stage2/train] Saved metrics -> {metrics_path}")


# ── Main training function ────────────────────────────────────────────────────

def train_stage2(cfg: AppConfig, artifact: dict | None = None) -> None:
    """Fine-tune Clinical-Longformer for Stage 2.

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

    print(f"[stage2/train] Train: {len(train_df):,}  |  Val: {len(val_notes):,}")

    train_gw = (
        [hp.group_lw.get(band_key(b), 1.0) for b in train_df["age_band"]]
        if hp.group_lw else [1.0] * len(train_df)
    )

    print(f"[stage2/train] Loading tokenizer from '{hp.model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(hp.model_name)

    train_dataset = ClinicalNotesDataset(
        train_df["text"].tolist(), train_df[TARGET_COL].tolist(),
        tokenizer, hp.max_length, group_weights=train_gw,
    )
    val_dataset = ClinicalNotesDataset(
        val_notes["text"].tolist(), val_notes[TARGET_COL].tolist(),
        tokenizer, hp.max_length,
    )

    print(f"[stage2/train] Loading model from '{hp.model_name}' ...")
    model = LongformerForSequenceClassification.from_pretrained(
        hp.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    if hp.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[stage2/train] Gradient checkpointing enabled")

    class_weights = _compute_class_weights(train_df)
    print(
        f"[stage2/train]   focal={'yes' if hp.use_focal else 'no'}  "
        f"gamma={hp.focal_gamma}"
    )

    device = _detect_device()
    training_args = _build_training_args(hp, model_dir, device, seed)

    # Capture loop-scoped variables for use inside the Trainer closure.
    _use_focal = hp.use_focal
    _focal_gamma = hp.focal_gamma
    _class_weights = class_weights

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
                weight=_class_weights.to(logits.device),
                reduction="none",
            )
            if _use_focal:
                pt = torch.exp(-ce)
                ce = ((1.0 - pt) ** _focal_gamma) * ce
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

    _save_results(trainer, val_dataset, model_dir, tokenizer)
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
