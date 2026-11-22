"""Parallel range-job selection for the Stage 3 batch audit.

Extracted from src/stage3/batch.py (2026-11-19) purely to keep that module
under pylint's line-count limit -- same pattern as _stopping.py / _notecap.py /
discordance.py, no behavioural change. Pure functions, no I/O: given the
ordered target list they decide which admissions one parallel range job audits
(src.stage3.batch._setup_batch is the only caller).
"""

from __future__ import annotations


def select_range(
    targets: list[int], range_start: int, range_end: int, done: set[int]
) -> list[int]:
    """Pure range selection for a parallel range job: the admissions in
    ``targets[range_start:range_end]`` not already in the output file, in the
    original target order.

    A fresh range's pending list is exactly ``targets[S:E]``; because
    :func:`validate_range` requires S/E to be multiples of the generation batch
    size, chunking it from index 0 reproduces the single sequential run's chunk
    boundaries -- identical bfloat16 batch composition over the same slice.
    ``done`` filtering (a --resume within one range) preserves order but,
    exactly like the sequential --resume path, compacts the pending list (so a
    mid-range resume no longer guarantees aligned chunks -- the same trade-off
    the sequential run already makes).
    """
    return [h for h in targets[range_start:range_end] if h not in done]


def validate_range(
    range_start: int | None, range_end: int | None, n_targets: int, batch_size: int
) -> None:
    """Validate --range-start/--range-end, or raise ``SystemExit`` (parallel
    range jobs). Both must be given together; ``0 <= start < end <= n_targets``;
    and each must be a multiple of the generation batch size, except ``end`` may
    equal ``n_targets`` (the final range reaches the end of the target list,
    which need not itself be batch-aligned). The batch-alignment is what keeps
    each range's chunk composition identical to the single sequential run's."""
    if (range_start is None) != (range_end is None):
        raise SystemExit(
            "[stage3/batch] --range-start and --range-end must be given together."
        )
    if range_start is None:
        return
    if not 0 <= range_start < range_end <= n_targets:
        raise SystemExit(
            f"[stage3/batch] Invalid range [{range_start}, {range_end}): require "
            f"0 <= start < end <= {n_targets} (total target admissions)."
        )
    if range_start % batch_size != 0:
        raise SystemExit(
            f"[stage3/batch] --range-start {range_start} must be a multiple of the "
            f"generation batch size ({batch_size}) so the range's chunk composition "
            "matches the sequential run."
        )
    if range_end % batch_size != 0 and range_end != n_targets:
        raise SystemExit(
            f"[stage3/batch] --range-end {range_end} must be a multiple of the "
            f"generation batch size ({batch_size}), or equal the total target "
            f"count ({n_targets})."
        )
