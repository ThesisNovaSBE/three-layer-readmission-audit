"""Stop-token helpers for Stage 3 batched generation.

Extracted from explain.py (2026-11) to keep that module under pylint's
1000-line limit, the same reason _notecap.py and discordance.py were split
out. No behavioural change; explain.py re-exports both names, so existing
imports (``from src.stage3.explain import _detect_truncated``) keep working.
"""

from __future__ import annotations


def _detect_truncated(generated_ids, stop_ids) -> list[bool]:
    """Per-sequence truncation flag: True if a generated row contains NONE of
    the model's stop tokens (cut off by max_new_tokens, not stopped
    naturally). Takes the full SET of stop ids (see _stop_token_ids), not
    just the tokenizer's single ``eos_token_id``: a row that ended cleanly
    on a secondary stop like ``<end_of_turn>`` must not be over-reported as
    truncated just because it lacks the primary EOS.

    Pure function (2026-09-18, audit finding 2.3) so it is unit-testable
    with hand-built token-ID tensors, no real model/tokenizer/GPU -- only
    needs ``generated_ids`` row iteration + a ``.tolist()`` per row.

    Args:
        generated_ids: 2D tensor/array, one row per sequence in the batch,
                        already sliced to exclude the prompt (see
                        call_llm_batch).
        stop_ids:       collection of stop-token ids. Empty/None means
                        "unknown", in which case nothing is flagged truncated.

    Returns:
        One bool per row, True meaning "likely truncated by the token cap".
    """
    stop = {s for s in (stop_ids or []) if s is not None}
    if not stop:
        return [False for _ in generated_ids]
    return [stop.isdisjoint(row.tolist()) for row in generated_ids]


def _stop_token_ids(tokenizer, model) -> set[int]:
    """Collect every token id that ends generation cleanly for this model:
    the tokenizer's ``eos_token_id``, the id of ``<end_of_turn>`` (a chat
    model's real turn terminator), and every id in
    ``model.generation_config.eos_token_id`` (which may be an int or a list).
    Ids that don't exist for this tokenizer (``<end_of_turn>`` resolving to
    ``unk``) are dropped."""
    stop: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tokenizer.unk_token_id:
        stop.add(eot)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        stop.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        stop.update(int(x) for x in gen_eos)
    return stop
