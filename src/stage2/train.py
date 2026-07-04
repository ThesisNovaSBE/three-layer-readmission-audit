"""Fine-tune Clinical-Longformer for Stage 2 false-positive pruning.

Takes the patient-level train split established by Stage 1 and fine-tunes
yikuan8/Clinical-Longformer as a binary classifier on discharge notes.
Patients flagged by Stage 1 that are NOT readmitted (false positives) should
be pruned here, increasing precision while preserving Stage 1's high recall.

Requires:
  - Stage 1 artifact: models/stage1_<model>.joblib (run src.model.train first)
  - MIMIC-IV-Note:    MIMIC_IV_NOTE_DIR set in .env

Usage:
    python -m src.stage2.train
    python -m src.stage2.train --mode full
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
from src.schemas import TARGET_COL


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
    # MPS (Apple Silicon GPU) has known incompatibilities with Longformer's
    # sliding-window attention — causes segfaults. Force CPU on macOS.
    return "cpu"


def train_stage2(cfg: dict, artifact: dict | None = None) -> None:
    """Fine-tune Clinical-Longformer for Stage 2.

    Args:
        cfg: loaded config dict.
        artifact: pre-loaded Stage 1 artifact dict. If None, it is loaded from
                  disk here. Pass a pre-loaded artifact when calling from a
                  context where torch is already imported (joblib.load of an
                  XGBoost model crashes on macOS if called after torch).
    """
    seed = cfg["run"]["random_state"]
    s2 = cfg["stage2"]
    model_name = s2["model_name"]
    batch_size = s2["batch_size"]
    epochs = s2["epochs"]
    lr = s2["learning_rate"]
    grad_accum = s2.get("gradient_accumulation_steps", 8)

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Reuse Stage 1's patient-level split ──
    if artifact is None:
        stage1_name = cfg["stage1"]["model"]
        artifact_path = model_dir / f"stage1_{stage1_name}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Stage 1 artifact not found at {artifact_path}. "
                "Run `python -m src.model.train` first."
            )
        artifact = joblib.load(artifact_path)
    train_idx, test_idx = artifact["train_idx"], artifact["test_idx"]

    # Use the mode the Stage 1 model was trained on so train_idx / test_idx
    # are valid positions in the reconstructed feature matrix.
    mode = artifact["mode"]
    max_length = s2["max_seq_length"]
    matrix = load_feature_matrix(cfg, mode)
    train_hadm_ids = set(matrix.iloc[train_idx]["hadm_id"].astype(int))
    test_hadm_ids = set(matrix.iloc[test_idx]["hadm_id"].astype(int))

    # ── Load and split notes — pass hadm_ids so only needed rows are read ──
    all_hadm_ids = train_hadm_ids | test_hadm_ids
    notes = load_notes(cfg, hadm_ids=all_hadm_ids)
    labels_df = matrix[["hadm_id", "subject_id", TARGET_COL]]
    notes_df = build_notes_dataframe(notes, labels_df)

    train_df = notes_df[notes_df["hadm_id"].isin(train_hadm_ids)].reset_index(drop=True)
    test_df = notes_df[notes_df["hadm_id"].isin(test_hadm_ids)].reset_index(drop=True)

    # Subsample training notes if cap is set. Stratified by label so class ratio is preserved.
    # Fine-tuning on all 251k notes would take ~46 days on CPU; 15k ≈ 9 hours train + 3h eval.
    max_train_notes = s2.get("max_train_notes", 0)
    if max_train_notes and len(train_df) > max_train_notes:
        _, train_df = train_test_split(
            train_df, test_size=max_train_notes,
            stratify=train_df[TARGET_COL], random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)
        print(f"[stage2/train] Subsampled to {len(train_df):,} training notes "
              f"(stratified, readmission rate: {train_df[TARGET_COL].mean():.1%})")

    print(f"[stage2/train] Train: {len(train_df):,} notes | Test: {len(test_df):,} notes")

    # ── Tokenise ──
    print(f"[stage2/train] Loading tokenizer from '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = ClinicalNotesDataset(
        train_df["text"].tolist(), train_df[TARGET_COL].tolist(), tokenizer, max_length)

    # For checkpoint selection during training, cap the eval set so periodic evaluation
    # (every epoch) doesn't dominate wall time. The full test set is used for final metrics.
    max_eval_notes = s2.get("max_eval_notes", 0)
    eval_df = test_df
    if max_eval_notes and len(test_df) > max_eval_notes:
        _, eval_df = train_test_split(
            test_df, test_size=max_eval_notes,
            stratify=test_df[TARGET_COL], random_state=seed,
        )
        eval_df = eval_df.reset_index(drop=True)
        print(f"[stage2/train] Eval set capped to {len(eval_df):,} notes for checkpoint selection "
              f"(full {len(test_df):,}-note test set used for final metrics)")

    eval_dataset = ClinicalNotesDataset(
        eval_df["text"].tolist(), eval_df[TARGET_COL].tolist(), tokenizer, max_length)

    # ── Model with class-weighted loss ──
    print(f"[stage2/train] Loading model from '{model_name}' ...")
    model = LongformerForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True)

    pos = int((train_df[TARGET_COL] == 1).sum())
    neg = int((train_df[TARGET_COL] == 0).sum())
    class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float32)
    print(f"[stage2/train] Class weights: neg=1.0  pos={neg / pos:.2f}")

    device = _detect_device()
    print(f"[stage2/train] Device: {device}")

    use_fp16 = device == "cuda"

    output_dir = str(model_dir / "stage2_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        fp16=use_fp16,
        use_cpu=(device == "cpu"),
        seed=seed,
        report_to="none",
        logging_steps=50,
        dataloader_num_workers=0,  # safer on macOS
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,   # capped for fast per-epoch checkpoint selection
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[stage2/train] Starting fine-tuning ...")
    trainer.train()

    # ── Save best model ──
    save_path = model_dir / "stage2_longformer_best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[stage2/train] Saved fine-tuned model -> {save_path}")

    # ── Final evaluation (uses same capped eval set for consistency + speed) ──
    # Full pipeline precision/recall is reported by predict_stage2, not here.
    results = trainer.evaluate(eval_dataset)
    auprc = results.get("eval_auprc", float("nan"))
    auroc = results.get("eval_auroc", float("nan"))
    print(f"[stage2/train] Test AUPRC={auprc:.4f}  AUROC={auroc:.4f}")

    metrics_path = model_dir / "stage2_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"[stage2/train] Saved metrics -> {metrics_path}")
    print("[stage2/train] Done. Run `python -m src.stage2.predict` to score flagged patients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stage 2 Clinical-Longformer")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode
    train_stage2(cfg)
