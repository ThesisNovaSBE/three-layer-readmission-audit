"""Disclosure-control helpers for the EDA -- every number that leaves memory
goes through here.

MIMIC-IV / MIMIC-IV-Note are credentialed data under the PhysioNet
Credentialed Health Data Use Agreement. The DUA allows publishing
*aggregate* results but forbids sharing anything that could re-identify a
patient or redistribute the data. The EDA therefore follows five rules:

1. **Aggregates only.** No row-level records, no ``subject_id`` / ``hadm_id``,
   no exact (shifted) dates leave this package -- only counts, rates,
   quantiles and binned histograms.
2. **Small-cell suppression.** Any count below :data:`MIN_CELL` (11) is
   reported as ``"<11"`` and any rate whose numerator *or* denominator is
   below it is reported as ``None``. Threshold follows the CMS cell-size
   suppression policy (cells of 1-10 are never published), the strictest
   commonly cited convention for US claims/EHR aggregates; PhysioNet's DUA
   itself names no number.
3. **No extremes.** Raw ``min`` / ``max`` describe a single individual, so
   distributions are summarised by the p1..p99 quantiles and histograms are
   clipped to [p1, p99].
4. **No free text.** Note statistics are computed per note *inside* the
   notes module and only per-note numbers/booleans are kept; the text itself
   is discarded chunk by chunk and no excerpts, keyword-in-context snippets,
   or example notes are ever written out.
5. **Outputs live in ``results/eda/``** as JSON/PNG/Markdown (``*.csv`` is
   gitignored repo-wide, so a careless table dump cannot be committed by
   accident either).

:func:`assert_no_identifiers` is a last-line guard called before any JSON
is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_CELL = 11
SUPPRESSED = "<11"
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_FORBIDDEN_KEYS = {"subject_id", "hadm_id", "stay_id", "note_id", "text"}


def safe_count(n: int | float) -> int | str:
    """Return ``n`` as an int, or ``"<11"`` if 0 < n < MIN_CELL.

    Zero is kept as 0: "no admissions in this cell" discloses nothing about
    any individual, and suppressing it would hide structural zeros (e.g. no
    ICD-9 codes after 2015) that are themselves informative.
    """
    n = int(n)
    return SUPPRESSED if 0 < n < MIN_CELL else n


def safe_rate(numerator: int | float, denominator: int | float, digits: int = 4) -> float | None:
    """Return numerator/denominator, or ``None`` if either side is a small cell.

    A rate with a tiny numerator (e.g. 3/5000) or a tiny denominator
    (e.g. 2/3) lets a reader back out a small count, so both are suppressed.
    """
    numerator, denominator = int(numerator), int(denominator)
    if denominator < MIN_CELL or 0 < numerator < MIN_CELL:
        return None
    return round(numerator / denominator, digits)


def safe_quantiles(values: pd.Series | np.ndarray, digits: int = 3) -> dict[str, Any]:
    """Summarise a numeric distribution without exposing individual extremes.

    Returns n (non-missing), n_missing, mean, sd and the p1..p99 quantiles.
    Returns only the counts if fewer than ``MIN_CELL`` non-missing values exist.
    """
    s = pd.Series(values, dtype="float64")
    vals = s.dropna()
    out: dict[str, Any] = {
        "n": safe_count(len(vals)),
        "n_missing": safe_count(s.isna().sum()),
        "missing_rate": safe_rate(s.isna().sum(), len(s)),
    }
    if len(vals) < MIN_CELL:
        return out
    out["mean"] = round(float(vals.mean()), digits)
    out["sd"] = round(float(vals.std()), digits)
    for q in QUANTILES:
        out[f"p{int(q * 100)}"] = round(float(vals.quantile(q)), digits)
    return out


def safe_value_counts(series: pd.Series, top: int | None = None,
                      other_label: str = "Other") -> list[dict[str, Any]]:
    """Category counts with small cells folded into ``other_label``.

    Folding (rather than just masking) keeps the totals honest: the column
    still sums to N, but no single small category is identifiable.
    """
    counts = series.fillna("MISSING").astype(str).value_counts()
    small = counts[counts < MIN_CELL]
    keep = counts[counts >= MIN_CELL]
    if top is not None and len(keep) > top:
        small = pd.concat([small, keep.iloc[top:]])
        keep = keep.iloc[:top]
    rows = [{"category": k, "n": int(v), "share": round(v / counts.sum(), 4)}
            for k, v in keep.items()]
    if small.sum() > 0:
        rows.append({"category": other_label, "n": safe_count(small.sum()),
                     "share": safe_rate(small.sum(), counts.sum())})
    return rows


def rate_by_group(df: pd.DataFrame, group_col: str, target_col: str,
                  top: int | None = None) -> list[dict[str, Any]]:
    """Outcome rate per category of ``group_col``, small groups folded to Other."""
    g = df[group_col].fillna("MISSING").astype(str)
    sizes = g.value_counts()
    keep = sizes[sizes >= MIN_CELL].index
    if top is not None:
        keep = keep[:top]
    g = g.where(g.isin(keep), "Other")
    rows = []
    for cat, sub in df.groupby(g, observed=True)[target_col]:
        rows.append({
            "category": cat,
            "n": safe_count(len(sub)),
            "n_positive": safe_count(sub.sum()),
            "rate": safe_rate(sub.sum(), len(sub)),
        })
    return sorted(rows, key=lambda r: -(r["n"] if isinstance(r["n"], int) else 0))


def safe_histogram(values: pd.Series | np.ndarray, bins: int = 40,
                   clip: tuple[float, float] = (0.01, 0.99)) -> dict[str, list]:
    """Histogram clipped to [p1, p99] with small bins zeroed out.

    The returned ``counts`` are what the figure is drawn from, so a plotted
    bar can never represent fewer than ``MIN_CELL`` admissions.
    """
    vals = pd.Series(values, dtype="float64").dropna()
    if len(vals) < MIN_CELL:
        return {"edges": [], "counts": []}
    lo, hi = vals.quantile(clip[0]), vals.quantile(clip[1])
    if lo == hi:
        hi = lo + 1
    # Drop (not clip) values outside [p1, p99]: clipping would pile every
    # outlier into the two edge bins and draw artificial spikes.
    counts, edges = np.histogram(vals[(vals >= lo) & (vals <= hi)], bins=bins, range=(lo, hi))
    counts = np.where(counts < MIN_CELL, 0, counts)
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def smd(a: pd.Series, b: pd.Series) -> float | None:
    """Standardized mean difference (pooled SD), Austin (2009). |SMD|>=0.1 = imbalance."""
    a, b = pd.Series(a, dtype="float64").dropna(), pd.Series(b, dtype="float64").dropna()
    if len(a) < MIN_CELL or len(b) < MIN_CELL:
        return None
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return None if pooled == 0 else round(float((a.mean() - b.mean()) / pooled), 4)


def assert_no_identifiers(obj: Any, path: str = "") -> None:
    """Raise if a nested output object contains an identifier/text key.

    A structural guard, not a content scan: aggregate dicts built by this
    package never use these keys, so their presence means a row-level frame
    slipped through.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"Identifier/text key {k!r} at {path or '<root>'} "
                                 "-- EDA outputs must be aggregate-only.")
            assert_no_identifiers(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_identifiers(v, f"{path}[{i}]")
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Raw pandas object at {path or '<root>'} -- convert to "
                        "aggregates before writing.")


def write_json(obj: Any, path: Path) -> None:
    """Validate with :func:`assert_no_identifiers`, then write pretty JSON."""
    assert_no_identifiers(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    """JSON fallback for numpy scalars."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")
