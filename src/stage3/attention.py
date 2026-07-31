"""Extract top-attended text spans from the fine-tuned Clinical-Longformer.

For each patient, runs one forward pass with output_attentions=True and
identifies the sentences that received the highest average attention weight
from the [CLS] (classification) token.  These spans represent what Stage 2
"looked at" when making its confirm / reject decision, and form the core of
the cross-modal discordance analysis in Stage 3.

Caveat (Jain & Wallace 2019 / Wiegreffe & Pinter 2019): raw attention weights
are not guaranteed to be faithful explanations of model decisions.  They
correlate with gradient-based attribution in many practical cases but should
be treated as *indicative* rather than *causal* in the thesis.  The attention
spans are used here only to ground the phi4-mini prompt with salient note
context, not as standalone proofs of which sentences caused the prediction.

If the fine-tuned model has not been saved yet (training in progress) or
PyTorch is unavailable, returns empty lists so the rest of Stage 3 can still
run without attention context.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.stage2._utils import get_stage2_model_path

_TORCH_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, LongformerForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _split_sentences(text: str) -> list[str]:
    """Split a clinical note into sentence-length fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _sentences_by_attention(
    text: str,
    token_weights: list[float],
    tokenizer,
    top_n: int,
) -> list[str]:
    """Return the top-n sentences ranked by summed token attention weights.

    Maps token-level attention scores back to sentences by re-tokenizing each
    sentence and summing the weights for the tokens it covers.

    Args:
        text:          original note text.
        token_weights: per-token weight from [CLS] token (length == seq_len - 1,
                       i.e., the CLS token itself is excluded at index 0).
        tokenizer:     Longformer tokenizer instance.
        top_n:         number of top sentences to return.

    Returns:
        Up to top_n sentence strings, sorted by attention score descending.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    scored: list[tuple[str, float]] = []
    offset = 0
    for sent in sentences:
        toks = tokenizer.tokenize(sent)
        n_tok = len(toks)
        if not toks or offset + n_tok > len(token_weights):
            break
        avg = sum(token_weights[offset: offset + n_tok]) / n_tok
        scored.append((sent, avg))
        offset += n_tok

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]]


def extract_attention_spans(
    hadm_ids: list[int],
    texts: list[str],
    model_dir: Path,
    top_n: int = 5,
    max_length: int = 2048,
) -> dict[int, list[str]]:
    """Extract top-attended sentences from the fine-tuned Clinical-Longformer.

    Runs inference in batch_size=1 on CPU to stay memory-safe on a Mac.

    Args:
        hadm_ids:   admission IDs (must match texts 1-to-1).
        texts:      discharge note strings.
        model_dir:  project models/ directory (used to locate stage2 checkpoint).
        top_n:      number of top sentences to return per patient.
        max_length: tokenization max length — must match Stage 2 training config.

    Returns:
        Dict mapping hadm_id -> list[str] of top-n sentences.
        Returns empty lists per patient if the model is not found.
    """
    empty: dict[int, list[str]] = {h: [] for h in hadm_ids}

    if not _TORCH_AVAILABLE:
        print("[stage3/attention] PyTorch not available — skipping.")
        return empty

    try:
        stage2_path = get_stage2_model_path(model_dir)
    except FileNotFoundError as exc:
        print(f"[stage3/attention] {exc} — skipping attention extraction.")
        return empty

    print(f"[stage3/attention] Loading model from {stage2_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    model = LongformerForSequenceClassification.from_pretrained(str(stage2_path))
    model.eval()

    result: dict[int, list[str]] = {}
    print(f"[stage3/attention] Extracting spans for {len(hadm_ids):,} notes ...")

    for i, (hadm_id, text) in enumerate(zip(hadm_ids, texts)):
        if not text:
            result[hadm_id] = []
            continue

        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Ensure [CLS] at position 0 receives global attention
        global_attention_mask = torch.zeros_like(enc["input_ids"])
        global_attention_mask[0, 0] = 1
        enc["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            out = model(**enc, output_attentions=True)

        # global_attentions: tuple per layer; last layer is most informative.
        # Shape varies: (batch, heads, num_global, seq_len) — CLS attending to others.
        if out.global_attentions:
            ga = out.global_attentions[-1]  # (1, heads, 1, seq_len)
            weights = ga.squeeze(0).mean(0).squeeze(0).tolist()  # (seq_len,)
        elif out.attentions:
            # Fallback: local attention from position 0 (CLS)
            la = out.attentions[-1]  # (1, heads, seq_len, seq_len)
            weights = la.squeeze(0).mean(0)[0].tolist()  # (seq_len,)
        else:
            result[hadm_id] = []
            continue

        # Skip position 0 (CLS token itself)
        token_weights = weights[1:]
        result[hadm_id] = _sentences_by_attention(text, token_weights, tokenizer, top_n)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(hadm_ids)}")

    print("[stage3/attention] Done.")
    return result
