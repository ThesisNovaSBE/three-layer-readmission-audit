"""Fine-tune Clinical-Longformer for Stage 2 false-positive pruning.

Key improvements over v1:
  - 2048-token sequences (was 512) — captures full average discharge note
  - Age x label stratified training sample — 70+ oversampled
  - Focal loss (gamma=2.0) — upweights hard examples (elderly ambiguous notes)
  - Per-age-group loss multipliers — direct fairness pressure during training
  - Proper patient-level splits: finetune / validation / calibration
  - Gradient checkpointing support for smaller GPUs
  - bf16 / fp16 switches for GPU precision

Requires:
  - Stage 1 artifact: models/stage1_<model>.joblib
  - Stage 2 splits:   data/processed/stage2_*_hadm_ids.csv  (run splits.py first)
  - MIMIC-IV-Note:    MIMIC_IV_NOTE_DIR set in .env

Usage:
    python -m src.stage2.train
    python setup_stage2.py --mode full   (recommended)
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix
from src.stage2.dataset import load_notes, build_notes_dataframe, ClinicalNotesDataset
from src.stage2.splits import build_splits
from src.schemas import TARGET_COL


def _band_key(age_band) -> str:
    """Map a pandas Interval age_band to config key string."""
    mapping = {
        "(17, 40]":  "18-40",
        "(40, 55]":  "41-55",
        "(55, 70]":  "56-70",
        "(70, 120]": "70+",
    }
    return mapping.get(str(age_band), str(age_band))


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    auprc = float(average_precision_score(labels, probs))
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = 0.0
    return {"auprc": auprc, "auroc": auroc}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_age_stratified_sample(
    notes_df: pd.DataFrame,
    targets: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Sample notes_df by age_band x label cell according to target counts."""
    parts = []
    for key, n_target in targets.items():
        band_str, label_str = key.rsplit("_", 1)
        label = int(label_str)
        mask = (notes_df["age_band"].apply(_band_key) == band_str) & \
               (notes_df[TARGET_COL] == label)
        cell = notes_df[mask]
        if len(cell) == 0:
            print(f"  [stage2/train] WARNING: no notes for cell {key}")
            continue
        n_take = min(n_target, len(cell))
        if n_take < n_target:
            print(f"  [stage2/train] Cell {key}: {len(cell):,} available "
                  f"(target {n_target:,}) — using all")
        parts.append(cell.sample(n=n_take, random_state=seed))

    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    print(f"[stage2/train] Age-stratified sample: {len(result):,} notes "
          f"(readmission rate: {result[TARGET_COL].mean():.1%})")
    for bk in ["18-40", "41-55", "56-70", "70+"]:
        sub = result[result["age_band"].apply(_band_key) == bk]
        if len(sub):
            print(f"  {bk:<8} {len(sub):>6,}  ({sub[TARGET_COL].mean():.1%} readmit)")
    return result


def train_stage2(cfg: dict, artifact: dict | None = None) -> None:
    """Fine-tune Clinical-Longformer for Stage 2.

    Args:
        cfg: loaded config dict.
        artifact: pre-loaded Stage 1 artifact. If None, loaded from disk.
                  Pass pre-loaded to avoid joblib/torch conflict on macOS.
    """
    seed        = cfg["run"]["random_state"]
    s2          = cfg["stage2"]
    model_name  = s2["model_name"]
    batch_size  = s2["batch_size"]
    epochs      = s2["epochs"]
    lr          = s2["learning_rate"]
    grad_accum  = s2.get("gradient_accumulation_steps", 4)
    max_length  = s2["max_seq_length"]
    use_focal   = s2.get("focal_loss", True)
    focal_gamma = s2.get("focal_gamma", 2.0)
    group_lw    = s2.get("age_group_loss_weights", {})
    age_targets = s2.get("age_group_train_targets", {})

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 artifact
    if artifact is None:
        stage1_name   = cfg["stage1"]["model"]
        artifact_path = model_dir / f"stage1_{stage1_name}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Stage 1 artifact not found at {artifact_path}. "
                "Run `python -m src.model.train` first."
            )
        artifact = joblib.load(artifact_path)

    mode = artifact["mode"]

    # Build / load patient-level splits
    print("[stage2/train] Loading patient-level splits ...")
    splits = build_splits(cfg, artifact=artifact)
    ft_hadm_ids  = set(splits["finetune"]["hadm_id"].astype(int))
    val_hadm_ids = set(splits["val"]["hadm_id"].astype(int))

    # Feature matrix (for age_band)
    matrix    = load_feature_matrix(cfg, mode)
    labels_df = matrix[["hadm_id", "subject_id", "age_band", TARGET_COL]]

    # Load notes
    all_needed = ft_hadm_ids | val_hadm_ids
    print(f"[stage2/train] Loading notes for {len(all_needed):,} admissions ...")
    notes = load_notes(cfg, hadm_ids=all_needed)

    def _build_split_df(hadm_ids: set) -> pd.DataFrame:
        sub = labels_df[labels_df["hadm_id"].isin(hadm_ids)]
        df  = build_notes_dataframe(notes, sub[["hadm_id", "subject_id", TARGET_COL]])
        df  = df.merge(labels_df[["hadm_id", "age_band"]], on="hadm_id", how="left")
        return df

    ft_notes  = _build_split_df(ft_hadm_ids)
    val_notes = _build_split_df(val_hadm_ids)

    # Age-stratified fine-tune sample
    if age_targets:
        train_df = _build_age_stratified_sample(ft_notes, age_targets, seed)
    else:
        train_df = ft_notes
        print(f"[stage2/train] No age targets — using all {len(train_df):,} fine-tune notes")

    print(f"[stage2/train] Train: {len(train_df):,}  |  Val: {len(val_notes):,}")

    # Per-sample group weights for training set
    train_gw = [group_lw.get(_band_key(b), 1.0) for b in train_df["age_band"]] \
               if group_lw else [1.0] * len(train_df)

    # Tokenise
    print(f"[stage2/train] Loading tokenizer from '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = ClinicalNotesDataset(
        train_df["text"].tolist(),
        train_df[TARGET_COL].tolist(),
        tokenizer,
        max_length,
        group_weights=train_gw,
    )
    val_dataset = ClinicalNotesDataset(
        val_notes["text"].tolist(),
        val_notes[TARGET_COL].tolist(),
        tokenizer,
        max_length,
    )

    # Model
    print(f"[stage2/train] Loading model from '{model_name}' ...")
    model = LongformerForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    if s2.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("[stage2/train] Gradient checkpointing enabled")

    # Class weights
    pos = int((train_df[TARGET_COL] == 1).sum())
    neg = int((train_df[TARGET_COL] == 0).sum())
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float32)
    print(f"[stage2/train] Class weights: neg=1.0  pos={neg/pos:.2f}  "
          f"focal={'yes' if use_focal else 'no'}  gamma={focal_gamma}")

    device   = _detect_device()
    use_fp16 = s2.get("fp16", False) and device == "cuda"
    use_bf16 = s2.get("bf16", False) and device == "cuda"
    print(f"[stage2/train] Device: {device}  fp16={use_fp16}  bf16={use_bf16}")

    training_args = TrainingArguments(
        output_dir=str(model_dir / "stage2_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=s2.get("warmup_ratio", 0.10),
        weight_decay=s2.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=(device == "cpu"),
        seed=seed,
        report_to="none",
        logging_steps=50,
        dataloader_num_workers=s2.get("dataloader_num_workers", 0),
    )

    # Focal loss + group-weighted custom trainer
    class FocalGroupWeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            gw     = inputs.pop("group_weight", None)
            outputs = model(**inputs)
            logits  = outputs.logits

            ce = F.cross_entropy(
                logits, labels,
                weight=class_weights.to(logits.device),
                reduction="none",
            )
            if use_focal:
                pt = torch.exp(-ce)
                ce = ((1.0 - pt) ** focal_gamma) * ce
            if gw is not None:
                ce = ce * gw.to(logits.device)

            loss = ce.mean()
            return (loss, outputs) if return_outputs else loss

    trainer = FocalGroupWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=s2.get("early_stopping_patience", 3)
        )],
    )

    print("\n[stage2/train] Starting fine-tuning ...")
    trainer.train()

    # Save best model
    save_path = model_dir / "stage2_longformer_best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[stage2/train] Saved fine-tuned model -> {save_path}")

    results      = trainer.evaluate(val_dataset)
    auprc_val    = results.get("eval_auprc", float("nan"))
    auroc_val    = results.get("eval_auroc", float("nan"))
    print(f"[stage2/train] Val AUPRC={auprc_val:.4f}  AUROC={auroc_val:.4f}")

    metrics_path = model_dir / "stage2_train_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"[stage2/train] Saved metrics -> {metrics_path}")
    print("[stage2/train] Done. Run calibrate.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()
    cfg  = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode
    train_stage2(cfg)
