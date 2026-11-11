"""Discharge-note length capping for the Stage 3 prompt.

Extracted from explain.py (2026-11) to keep that module under pylint's
line-count limit, the same reason src/stage3/discordance.py was split out.
No behavioral change -- same functions, same callers (explain.build_prompt
renders _note_block; explain._finalize_annotation verifies quotes against
_shown_note_text; pipeline.py logs is_note_truncated).
"""

from __future__ import annotations

# Generous safety cap, not a routine truncation. Session 14 measured median
# MIMIC discharge notes at ~2,649 tokens; at Stage 2's 4096-token window
# (~4-5 chars/token in clinical text) that is comfortably under 20,000 chars.
# Over the cap, the note is shown head + tail (not head-only): a discharge
# summary's end carries the disposition, discharge instructions and
# follow-up, exactly the content an auditor needs and exactly what a
# head-only cut removes. Same total budget as before.
_NOTE_HEAD_CHARS = 12_000
_NOTE_TAIL_CHARS = 8_000
_NOTE_MAX_CHARS = _NOTE_HEAD_CHARS + _NOTE_TAIL_CHARS  # 20_000

_NOTE_OMITTED_MARKER = "[... middle of the discharge note omitted: {n:,} characters ...]"


def _shown_note_text(note_text: str) -> str:
    """Return the exact note text shown to the model.

    Within the cap: the whole (stripped) note. Over the cap: the first
    ``_NOTE_HEAD_CHARS`` and the last ``_NOTE_TAIL_CHARS``, joined by a
    marker line naming how many characters were omitted. The head is trimmed
    back to the last newline in its final 500 chars (so it doesn't end
    mid-line); the tail is trimmed forward to the first newline in its first
    500 chars (so it doesn't start mid-line). The combined body is at most
    ``_NOTE_MAX_CHARS`` plus the marker.

    This is the single source of truth for "what the model actually saw":
    :func:`_note_block` renders it into the prompt, and
    :func:`src.stage3.explain._finalize_annotation` verifies quotes against
    it. A quote matching only the omitted middle therefore counts as
    unverified -- it was never shown, so it cannot be real evidence.
    """
    text = note_text.strip()
    if len(text) <= _NOTE_MAX_CHARS:
        return text
    head = text[:_NOTE_HEAD_CHARS]
    head_nl = head.rfind("\n", _NOTE_HEAD_CHARS - 500)
    if head_nl != -1:
        head = head[:head_nl]
    tail = text[-_NOTE_TAIL_CHARS:]
    tail_nl = tail.find("\n", 0, 500)
    if tail_nl != -1:
        tail = tail[tail_nl + 1:]
    omitted = len(text) - len(head) - len(tail)
    marker = _NOTE_OMITTED_MARKER.format(n=omitted)
    return f"{head}\n{marker}\n{tail}"


def is_note_truncated(note_text: str) -> bool:
    """Whether ``note_text`` exceeds the safety cap applied in the prompt.

    Unchanged meaning: the note exceeded ``_NOTE_MAX_CHARS`` and so its
    middle was omitted from the prompt. Exposed separately from
    :func:`_note_block` so callers (pipeline.py, batch.py) can log it as
    structured data, not just prose inside the prompt string.
    """
    return len(note_text.strip()) > _NOTE_MAX_CHARS


def _note_block(note_text: str) -> str:
    """Return the discharge note for the prompt, head+tail-capped if long."""
    text = note_text.strip()
    if not text:
        return "  (discharge note not available)"
    shown = _shown_note_text(note_text)
    if len(text) > _NOTE_MAX_CHARS:
        return (f"(discharge note, {len(text):,} chars; middle section omitted "
                f"to fit the {_NOTE_MAX_CHARS:,}-character limit)\n  {shown}")
    return f"(full discharge note, {len(text):,} chars)\n  {shown}"
