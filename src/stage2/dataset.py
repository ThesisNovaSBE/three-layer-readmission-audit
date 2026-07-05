"""Load and preprocess clinical notes for Stage 2 fine-tuning.

MIMIC-IV-Note is required. Set MIMIC_IV_NOTE_DIR in .env.
Notes are joined with the Stage 1 feature matrix via (subject_id, hadm_id).

One discharge note per admission is kept (the last if duplicates exist).
Stage 2 cannot run on synthetic data — clinical notes are real only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import load_config
from src.schemas import TARGET_COL


def load_notes(cfg: dict, hadm_ids: set | None = None) -> pd.DataFrame:
    """Load discharge notes from MIMIC-IV-Note.

    Reads in chunks and filters to only the requested hadm_ids so the full
    multi-GB file is never held in memory at once.

    Args:
        cfg: loaded config dict.
        hadm_ids: if provided, only rows matching these admission IDs are kept.
                  Pass the union of train + test hadm_ids to minimise memory use.

    Returns DataFrame with columns: hadm_id, subject_id, text.
    """
    note_dir = cfg["data"].get("mimic_iv_note_dir", "")
    if not note_dir or not Path(note_dir).exists():
        raise FileNotFoundError(
            "MIMIC-IV-Note not found. Set MIMIC_IV_NOTE_DIR in .env. "
            "Stage 2 requires real clinical notes and cannot run on synthetic data."
        )

    note_path = Path(note_dir) / "note" / "discharge.csv.gz"
    if not note_path.exists():
        note_path = Path(note_dir) / "discharge.csv.gz"
    if not note_path.exists():
        raise FileNotFoundError(f"Discharge notes file not found. Tried: {note_path}")

    print(f"[stage2/dataset] Loading notes from {note_path} (chunked) ...")

    # Peek at columns to handle different MIMIC-IV-Note schema versions
    peek = pd.read_csv(note_path, nrows=0)
    available_cols = set(peek.columns)
    use_cols = [c for c in ["subject_id", "hadm_id", "text", "note_type"]
                if c in available_cols]

    chunks = []
    for chunk in pd.read_csv(note_path, usecols=use_cols, chunksize=50_000,
                              dtype={"subject_id": "int64"}):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

        # Filter to needed admissions early to keep memory low
        if hadm_ids is not None:
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

        # Keep only discharge summaries if the note_type column exists.
        # MIMIC-IV-Note uses 'DS' as the note_type code for discharge summaries.
        if "note_type" in chunk.columns:
            chunk = chunk[
                chunk["note_type"].str.upper().isin({"DS", "DISCHARGE", "DISCHARGE SUMMARY"})
            ]

        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "No discharge notes found after filtering. "
            "Check MIMIC_IV_NOTE_DIR path and hadm_id alignment."
        )

    notes = pd.concat(chunks, ignore_index=True)

    # One note per admission — keep the last if multiple exist
    notes = notes.sort_values("hadm_id").groupby("hadm_id", as_index=False).last()

    keep_cols = [c for c in ["hadm_id", "subject_id", "text"] if c in notes.columns]
    notes = notes[keep_cols]

    print(f"[stage2/dataset] Loaded {len(notes):,} discharge notes.")
    return notes


def build_notes_dataframe(
    notes: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join notes with readmission labels.

    Args:
        notes: output of load_notes() — columns: hadm_id, subject_id, text.
        labels: DataFrame with columns: hadm_id, subject_id, readmission_30d.

    Returns:
        Merged DataFrame keeping only rows that have both a note and a label.
    """
    merged = notes.merge(
        labels[["hadm_id", "subject_id", TARGET_COL]],
        on=["hadm_id", "subject_id"],
        how="inner",
    )
    merged = merged.dropna(subset=["text", TARGET_COL])
    merged[TARGET_COL] = merged[TARGET_COL].astype(int)
    print(
        f"[stage2/dataset] Notes matched to labels: {len(merged):,} | "
        f"readmission rate: {merged[TARGET_COL].mean():.1%}"
    )
    return merged.reset_index(drop=True)


class ClinicalNotesDataset(Dataset):
    """PyTorch Dataset: tokenized clinical note + binary readmission label.

    Tokenizes lazily (one note per __getitem__ call) rather than all at once
    in __init__. Eager tokenization of thousands of notes padded to 1024 tokens
    creates a tensor too large for Apple Silicon unified memory and causes a
    segmentation fault before training even starts.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        group_weights: list[float] | None = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_weights = group_weights  # per-sample age-group loss multiplier

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = self.labels[idx]
        if self.group_weights is not None:
            import torch
            item["group_weight"] = torch.tensor(self.group_weights[idx], dtype=torch.float32)
        return item
