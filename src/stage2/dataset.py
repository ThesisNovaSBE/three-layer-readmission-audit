"""Load and preprocess clinical notes for Stage 2 fine-tuning.

MIMIC-IV-Note is required. Set MIMIC_IV_NOTE_DIR in .env.
Notes are joined with the Stage 1 feature matrix via (subject_id, hadm_id).

One discharge note per admission is kept (the last if duplicates exist).
Stage 2 cannot run on synthetic data — clinical notes are real only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config_schema import AppConfig
from src.schemas import TARGET_COL

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from transformers import PreTrainedTokenizerBase
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore[misc,assignment]  # sentinel base class


def load_notes(cfg: AppConfig, hadm_ids: set | None = None) -> pd.DataFrame:
    """Load discharge notes from MIMIC-IV-Note.

    Reads in chunks and filters to only the requested hadm_ids so the full
    multi-GB file is never held in memory at once.

    Args:
        cfg:      validated project config.
        hadm_ids: if provided, only rows matching these admission IDs are kept.
                  Pass the union of train + test hadm_ids to minimise memory use.

    Returns:
        DataFrame with columns: ``hadm_id``, ``subject_id``, ``text``.

    Raises:
        FileNotFoundError: if MIMIC-IV-Note is not configured or not found.
        ValueError: if no discharge notes match the requested hadm_ids.
    """
    note_dir = cfg.data.mimic_iv_note_dir
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

    peek = pd.read_csv(note_path, nrows=0)
    available_cols = set(peek.columns)
    use_cols = [c for c in ["subject_id", "hadm_id", "text", "note_type"]
                if c in available_cols]

    chunks = []
    for chunk in pd.read_csv(
        note_path, usecols=use_cols, chunksize=50_000,
        dtype={"subject_id": "int64"},
    ):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

        if hadm_ids is not None:
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

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
        notes:  output of :func:`load_notes` — columns: hadm_id, subject_id, text.
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


class ClinicalNotesDataset(_TorchDataset):  # type: ignore[valid-type]
    """PyTorch Dataset: tokenized clinical note + binary readmission label.

    Tokenizes lazily (one note per ``__getitem__`` call) rather than all at
    once in ``__init__``. Eager tokenization of thousands of notes padded to
    2048 tokens creates a tensor too large for Apple Silicon unified memory and
    causes a segmentation fault before training even starts.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 1024,
        group_weights: list[float] | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            texts:         list of raw note strings.
            labels:        list of binary readmission labels (0 or 1).
            tokenizer:     HuggingFace tokenizer compatible with the model.
            max_length:    maximum tokenization length (tokens).
            group_weights: optional per-sample age-group loss multipliers.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "Stage 2 requires PyTorch and Transformers. "
                "Install with: pip install torch transformers"
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_weights = group_weights

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Return a tokenized sample dict for the given index."""
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
            item["group_weight"] = torch.tensor(
                self.group_weights[idx], dtype=torch.float32
            )
        return item
