"""Fusion model: Clinical-Longformer backbone + structured feature MLP.

Architecture
------------
  Longformer backbone  →  [CLS] pooled output          (hidden_size-dim, e.g. 768)
  Structured feature MLP  →  Linear → LayerNorm → ReLU → Linear  (64-dim)
  Concatenate            →  (hidden_size + 64)-dim
  Dropout + Linear       →  2 logits

The model is HuggingFace Trainer-compatible: ``forward()`` returns a
``SequenceClassifierOutput`` so the Trainer can read ``.loss`` and ``.logits``
and checkpoints are saved / restored via ``model.state_dict()``.

Global attention is set on the [CLS] token inside ``forward()``; the dataset
does not need to produce ``global_attention_mask``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers import LongformerModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.stage2._utils import STRUCT_FEATURE_COLS


class FusionLongformer(nn.Module):
    """Clinical-Longformer + structured feature MLP fusion classifier.

    Args:
        longformer_name:  HuggingFace model name or local path to a saved
                          Longformer backbone (supports ``from_pretrained``).
        n_struct_features: number of structured input features (must match
                           the length of the feature columns actually present;
                           see :data:`STRUCT_FEATURE_COLS`).
        num_labels:       number of output classes (default 2).
        dropout:          dropout probability before the classifier head.
    """

    def __init__(
        self,
        longformer_name: str,
        n_struct_features: int,
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_struct_features = n_struct_features
        self.longformer = LongformerModel.from_pretrained(
            longformer_name, ignore_mismatched_sizes=True
        )
        hidden = self.longformer.config.hidden_size  # 768 for Longformer-base
        self.struct_mlp = nn.Sequential(
            nn.Linear(n_struct_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + 64, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        struct_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_kwargs,
    ) -> SequenceClassifierOutput:
        """Forward pass.

        Args:
            input_ids:       (B, seq_len) token indices.
            attention_mask:  (B, seq_len) 1 for real tokens, 0 for padding.
            struct_features: (B, n_struct_features) normalised structured features.
            labels:          (B,) integer class labels; if provided, loss is returned.

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput`
            with ``.loss`` (if labels given) and ``.logits``.
        """
        # Longformer requires global attention on at least one token;
        # the [CLS] token (position 0) is the standard choice for classification.
        global_attn = torch.zeros_like(attention_mask)
        global_attn[:, 0] = 1

        lf_out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attn,
        )
        cls_emb = lf_out.last_hidden_state[:, 0, :]       # (B, hidden)
        struct_emb = self.struct_mlp(struct_features.float())  # (B, 64)
        fused = self.drop(torch.cat([cls_emb, struct_emb], dim=-1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


__all__ = ["FusionLongformer", "STRUCT_FEATURE_COLS"]
