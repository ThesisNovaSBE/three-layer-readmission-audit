"""Load and preprocess clinical notes for Stage 2 fine-tuning.

MIMIC-IV-Note is required. Set MIMIC_IV_NOTE_DIR in .env.
Notes are joined with the Stage 1 feature matrix via (subject_id, hadm_id).

One discharge note per admission is kept (the last if duplicates exist).
Stage 2 cannot run on synthetic data — clinical notes are real only.

Tokenization uses head+tail truncation: the first ``max_length // 2`` tokens
and the last ``max_length // 2`` tokens are kept, preserving both the
admission reason (head of note) and the discharge plan (tail of note).
For notes shorter than ``max_length`` the sequence is padded normally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config_schema import AppConfig
from src.schemas import TARGET_COL
from src.stage2._utils import CONTINUOUS_STRUCT_COLS, STRUCT_FEATURE_COLS

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from transformers import PreTrainedTokenizerBase
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore[misc,assignment]


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


def normalize_struct_features(
    df: pd.DataFrame,
    means: dict | None = None,
    stds: dict | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Z-score normalise continuous structured feature columns.

    Binary features (0/1) are left unchanged.
    NaN values (e.g. ``days_since_last_discharge`` for first
    admissions) are filled with 0.0 after scaling.

    Only columns that are present in ``df`` and listed in
    :data:`~src.stage2._utils.STRUCT_FEATURE_COLS` are included.

    Args:
        df:    DataFrame whose columns include a subset of
               :data:`~src.stage2._utils.STRUCT_FEATURE_COLS`.
        means: pre-computed column means from the training set.
               Computed from ``df`` when ``None`` (use training set only).
        stds:  pre-computed column standard deviations.
               Computed from ``df`` when ``None``.

    Returns:
        Tuple ``(feature_array, means_dict, stds_dict)`` where
        ``feature_array`` has shape ``(len(df), n_present_cols)`` and dtype
        ``float32``.
    """
    present = [c for c in STRUCT_FEATURE_COLS if c in df.columns]
    out = df[present].copy().astype(float)

    if means is None:
        means = {
            c: float(out[c].mean())
            for c in CONTINUOUS_STRUCT_COLS if c in present
        }
    if stds is None:
        stds = {
            c: max(float(out[c].std(ddof=0)), 1e-8)
            for c in CONTINUOUS_STRUCT_COLS if c in present
        }

    for col in CONTINUOUS_STRUCT_COLS:
        if col in present:
            out[col] = (out[col] - means[col]) / stds[col]

    return out.fillna(0.0).values.astype("float32"), means, stds


class ClinicalNotesDataset(_TorchDataset):  # type: ignore[valid-type]
    """PyTorch Dataset: tokenized clinical note + binary readmission label.

    Tokenization uses **head+tail truncation**: if the note exceeds
    ``max_length`` tokens, the first ``max_length // 2`` and last
    ``max_length // 2`` tokens are kept.  This preserves both the admission
    reason (front of note) and the discharge plan / follow-up instructions
    (end of note), which are jointly predictive of 30-day readmission.

    Tokenization is lazy (one note per ``__getitem__`` call) to avoid OOM
    on large datasets.

    Args:
        texts:           list of raw note strings.
        labels:          list of binary readmission labels (0 or 1).
        tokenizer:       HuggingFace tokenizer compatible with the model.
        max_length:      maximum sequence length in tokens.
        struct_features: optional ``(n_samples, n_features)`` float32 array of
                         normalised structured features.  When provided, each
                         ``__getitem__`` call adds a ``"struct_features"`` key
                         to the returned dict (used by :class:`FusionLongformer`).
        group_weights:   optional per-sample age-group loss multipliers.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int = 2048,
        struct_features: np.ndarray | None = None,
        group_weights: list[float] | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "Stage 2 requires PyTorch and Transformers. "
                "Install with: pip install torch transformers"
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.struct_features = struct_features
        self.group_weights = group_weights

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Return a tokenized sample dict with head+tail truncation applied."""
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)

        if len(input_ids) > self.max_length:
            # Head+tail: keep first half and last half of the token sequence.
            # This preserves the admission summary (head) and the discharge
            # plan / follow-up instructions (tail) — both predictive of
            # 30-day readmission.
            half = self.max_length // 2
            input_ids = torch.cat([input_ids[:half], input_ids[-half:]])
            attn_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            pad_len = self.max_length - len(input_ids)
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            pad_tensor = torch.full((pad_len,), fill_value=pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_tensor])
            orig_mask = encoding["attention_mask"].squeeze(0)
            pad_zeros = torch.zeros(pad_len, dtype=torch.long)
            attn_mask = torch.cat([orig_mask, pad_zeros])

        item = {"input_ids": input_ids, "attention_mask": attn_mask}
        item["labels"] = self.labels[idx]

        if self.struct_features is not None:
            item["struct_features"] = torch.tensor(
                self.struct_features[idx], dtype=torch.float32
            )
        if self.group_weights is not None:
            item["group_weight"] = torch.tensor(
                self.group_weights[idx], dtype=torch.float32
            )
        return item
