"""EDA of MIMIC-IV-Note, focused on what Stage 2 and Stage 3 actually read.

Privacy design (see ``src/eda/privacy.py``): the discharge file is streamed
in chunks; each chunk's text is reduced to per-note *numbers and booleans*
(length, token count, section offsets, keyword flags) and then dropped. No
text, excerpt, or keyword-in-context snippet is ever kept or written, and
only suppressed aggregates reach ``results/eda/notes_summary.json``.

==========================  ==============================================
Section                     Thesis question it answers
==========================  ==============================================
inventory                   How many notes, of which type, linked to what?
coverage                    How many cohort admissions have a note
                            (duplicates, radiology as a fallback)?
length                      How long are notes in Longformer tokens; how
                            many are truncated at the 4096 window?
sections                    Which sections exist, and which sit *beyond*
                            the window (i.e. Stage 2 never reads them)?
placeholder_sections        Are Social History / Follow-up sections mostly
                            de-identified away (limits Stage 3 grounds)?
keywords                    How common are Stage 3 grounds concepts, and
                            do they shift the readmission base rate?
timing                      When is the note stored relative to discharge
                            (deployment realism for a discharge-time tool)?
deid                        How much of a note is de-id placeholders?
==========================  ==============================================

Usage::

    python -m src.eda.notes            # all notes in the cohort (~30-60 min)
    python -m src.eda.notes --sample 20000   # quick look on a random subset
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.features import _apply_cohort_filters, compute_readmission_label
from src.eda import loaders as L
from src.eda import plots
from src.eda.privacy import (
    MIN_CELL,
    safe_count,
    safe_histogram,
    safe_quantiles,
    safe_rate,
    safe_value_counts,
    write_json,
)
from src.schemas import TARGET_COL_UNPLANNED as TARGET

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "eda"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

SECTIONS = [
    "Chief Complaint", "History of Present Illness", "Past Medical History",
    "Social History", "Family History", "Physical Exam", "Pertinent Results",
    "Brief Hospital Course", "Medications on Admission", "Discharge Medications",
    "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition",
    "Discharge Instructions", "Followup Instructions",
]
_SECTION_RE = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(s) for s in SECTIONS) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE)
_CANON = {s.lower(): s for s in SECTIONS}

# Concepts behind Stage 3's grounds taxonomy (MODEL_CARD.md). Deliberately
# simple lexical proxies -- this measures how often the *words* occur, not
# whether the clinical fact holds (negation is not handled).
CONCEPTS = {
    "palliative / hospice": r"\bhospice\b|\bcomfort (?:measures|care)\b|\bpalliative\b",
    "lives alone": r"\blives? alone\b",
    "home services (VNA)": r"\bhome with services\b|\bVNA\b|\bvisiting nurse",
    "nonadherence / AMA": r"non-?complian|non-?adheren|against medical advice",
    "cognitive impairment": r"\bdementia\b|\bdelirium\b|cognitive impairment",
    "functional dependence": r"\bwheelchair\b|\bbed-?bound\b|\bwalker\b|dependent for",
    "substance use": r"alcohol (?:use|abuse|withdrawal)|\betoh\b|opioid use disorder|\bivdu\b",
    "follow-up appointment": r"\bappointment\b",
}
_CONCEPT_RE = {k: re.compile(v, re.IGNORECASE) for k, v in CONCEPTS.items()}
_PLACEHOLDER_ONLY = re.compile(r"^[\s_.,:;-]*$")
PLACEHOLDER_SECTIONS = ["Social History", "Followup Instructions", "Discharge Disposition",
                        "Discharge Instructions", "Family History"]


def _note_stats(texts: list[str], tokenizer, max_len: int) -> pd.DataFrame:
    """Reduce a batch of note texts to per-note numeric features (text dropped)."""
    enc = tokenizer(texts, add_special_tokens=True, return_offsets_mapping=True,
                    truncation=False, verbose=False)
    rows = []
    for text, ids, offs in zip(texts, enc["input_ids"], enc["offset_mapping"]):
        n_tok = len(ids)
        # Char offset where Stage 2's window ends: last content token before </s>.
        cutoff = offs[max_len - 2][1] if n_tok > max_len else len(text)
        row = {"n_chars": len(text), "n_words": len(text.split()), "n_tokens": n_tok,
               "truncated": n_tok > max_len, "n_placeholders": text.count("___")}
        matches = list(_SECTION_RE.finditer(text))
        seen = {}
        for i, mt in enumerate(matches):
            name = _CANON[mt.group(1).lower()]
            if name in seen:
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seen[name] = (mt.start(), text[mt.end():end])
        for s in SECTIONS:
            if s in seen:
                start, body = seen[s]
                row[f"sec::{s}"] = True
                row[f"pos::{s}"] = start / max(len(text), 1)
                row[f"beyond::{s}"] = start >= cutoff
                if s in PLACEHOLDER_SECTIONS:
                    row[f"ph::{s}"] = bool(_PLACEHOLDER_ONLY.match(body))
            else:
                row[f"sec::{s}"] = False
        for k, rx in _CONCEPT_RE.items():
            row[f"kw::{k}"] = bool(rx.search(text))
        rows.append(row)
    return pd.DataFrame(rows)


def scan_discharge_notes(note_path: Path, keep_hadm: set, tokenizer, max_len: int,
                         sample: int | None = None) -> pd.DataFrame:
    """Stream discharge.csv.gz -> one row of numbers per cohort note.

    Duplicate notes per admission are resolved like ``src/stage2/dataset.py``
    (last occurrence in file order wins). ``hadm_id`` stays in memory only to
    join the label; it is never written out.
    """
    if sample:
        rng = np.random.default_rng(42)
        keep_hadm = set(rng.choice(sorted(keep_hadm), size=min(sample, len(keep_hadm)),
                                   replace=False).tolist())
    parts = []
    for i, chunk in enumerate(pd.read_csv(
            note_path, chunksize=20_000,
            usecols=["hadm_id", "note_seq", "charttime", "storetime", "text"],
            parse_dates=["charttime", "storetime"])):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk = chunk[chunk["hadm_id"].astype("int64").isin(keep_hadm)]
        if chunk.empty:
            continue
        texts = chunk["text"].tolist()
        # Sub-batches: offset mappings for 20k long notes would need several GB.
        stats = pd.concat([_note_stats(texts[j:j + 500], tokenizer, max_len)
                           for j in range(0, len(texts), 500)], ignore_index=True)
        stats.insert(0, "hadm_id", chunk["hadm_id"].astype("int64").to_numpy())
        stats["charttime"] = chunk["charttime"].to_numpy()
        stats["storetime"] = chunk["storetime"].to_numpy()
        parts.append(stats)
        del texts, chunk
        print(f"[eda/notes] chunk {i}: {sum(len(p) for p in parts):,} cohort notes so far",
              flush=True)
    notes = pd.concat(parts, ignore_index=True)
    notes["dup_rank"] = notes.groupby("hadm_id").cumcount(ascending=False)
    return notes


def _cohort(cfg) -> pd.DataFrame:
    """Model cohort (hadm_id, subject_id, dischtime, target, age_band) -- fast path."""
    adm = L.load_admissions(cfg)
    pts = L.load_patients(cfg)
    lab = compute_readmission_label(adm.drop(columns=["deathtime", "language"]), cfg)
    coh = _apply_cohort_filters(lab, pts, cfg)
    coh["age_band"] = pd.cut(coh["anchor_age"], bins=[17, 40, 55, 70, 120],
                             labels=["18-40", "41-55", "56-70", "70+"]).astype(str)
    return coh[["hadm_id", "subject_id", "dischtime", "age_band", TARGET]]


def summarise(notes_all: pd.DataFrame, coh: pd.DataFrame, max_len: int,
              note_root: Path, raw_counts: dict) -> dict:
    """Turn the per-note frame into the aggregate-only summary."""
    dup_admissions = int((notes_all.groupby("hadm_id").size() > 1).sum())
    n = notes_all[notes_all["dup_rank"] == 0].merge(coh, on="hadm_id", how="left")
    pos = n[TARGET] == 1

    def by_outcome(col):
        return {"readmitted": safe_quantiles(n.loc[pos, col]),
                "not_readmitted": safe_quantiles(n.loc[~pos, col])}

    sections = []
    for s in SECTIONS:
        present = n[f"sec::{s}"]
        row = {"section": s, "present_share": safe_rate(present.sum(), len(n))}
        if present.sum() >= MIN_CELL:
            beyond = n.loc[present, f"beyond::{s}"].astype(bool)
            row["median_relative_position"] = round(float(n.loc[present, f"pos::{s}"].median()), 3)
            row["beyond_window_share_of_present"] = safe_rate(beyond.sum(), present.sum())
            if s in PLACEHOLDER_SECTIONS:
                ph = n.loc[present, f"ph::{s}"].astype(bool)
                row["placeholder_only_share_of_present"] = safe_rate(ph.sum(), present.sum())
        sections.append(row)

    kws = []
    for k in CONCEPTS:
        hit = n[f"kw::{k}"].astype(bool)
        kws.append({"concept": k, "mention_share": safe_rate(hit.sum(), len(n)),
                    "rate_if_mentioned": safe_rate(n.loc[hit, TARGET].sum(), hit.sum()),
                    "rate_if_absent": safe_rate(n.loc[~hit, TARGET].sum(), (~hit).sum()),
                    "mention_share_readmitted": safe_rate(hit[pos].sum(), pos.sum()),
                    "mention_share_not_readmitted": safe_rate(hit[~pos].sum(), (~pos).sum())})

    lag_h = (n["storetime"] - n["dischtime"]).dt.total_seconds() / 3600
    chart_lag_d = (n["charttime"].dt.normalize() - n["dischtime"].dt.normalize()).dt.days
    ph_bins = pd.cut(n["n_placeholders"], bins=[-1, 0, 10, 25, 50, 100, np.inf],
                     labels=["0", "1-10", "11-25", "26-50", "51-100", "100+"])
    ph_rows = [{"bucket": str(b), "n": safe_count(c), "share": safe_rate(c, len(n))}
               for b, c in ph_bins.value_counts(sort=False).items()]

    trunc_by_age = []
    for ab, sub in n.groupby("age_band"):
        trunc_by_age.append({"age_band": ab, "n": safe_count(len(sub)),
                             "truncated_share": safe_rate(sub["truncated"].sum(), len(sub))})

    return {
        "target": TARGET, "max_seq_length": max_len,
        "privacy": {"min_cell": MIN_CELL, "note": "per-note numbers only; no text retained"},
        "inventory": {**raw_counts, "note_version": note_root.name},
        "coverage": {
            "cohort_admissions": safe_count(len(coh)),
            "cohort_admissions_with_discharge_note": raw_counts["cohort_with_discharge_note"],
            "share": safe_rate(_num(raw_counts["cohort_with_discharge_note"]), len(coh)),
            "notes_analysed": safe_count(len(n)),
            "admissions_with_multiple_discharge_notes": safe_count(dup_admissions),
            **raw_counts.get("radiology_coverage", {}),
        },
        "length": {
            "chars": safe_quantiles(n["n_chars"], digits=0),
            "words": safe_quantiles(n["n_words"], digits=0),
            "tokens": safe_quantiles(n["n_tokens"], digits=0),
            "tokens_by_outcome": by_outcome("n_tokens"),
            "chars_per_token": round(float(n["n_chars"].sum() / n["n_tokens"].sum()), 3),
            "truncated_share": safe_rate(n["truncated"].sum(), len(n)),
            "truncated_share_readmitted": safe_rate(n.loc[pos, "truncated"].sum(), pos.sum()),
            "truncated_share_not_readmitted": safe_rate(
                n.loc[~pos, "truncated"].sum(), (~pos).sum()),
            "truncated_by_age_band": trunc_by_age,
            "tokens_hist_readmitted": safe_histogram(n.loc[pos, "n_tokens"], bins=50),
            "tokens_hist_not_readmitted": safe_histogram(n.loc[~pos, "n_tokens"], bins=50),
        },
        "sections": {"per_section": sections,
                     "n_sections_found": safe_quantiles(
                         n[[f"sec::{s}" for s in SECTIONS]].sum(axis=1))},
        "keywords": {"caveat": "lexical match, no negation handling",
                     "per_concept": kws},
        "timing": {
            "hours_dischtime_to_storetime": safe_quantiles(lag_h, digits=1),
            "hours_dischtime_to_storetime_hist": safe_histogram(lag_h, bins=40),
            "stored_before_discharge_share": safe_rate((lag_h < 0).sum(), lag_h.notna().sum()),
            "stored_within_24h_share": safe_rate(((lag_h >= 0) & (lag_h <= 24)).sum(),
                                                 lag_h.notna().sum()),
            "charttime_minus_discharge_date_days": safe_value_counts(
                chart_lag_d.clip(-2, 3).astype("Int64").astype(str)),
        },
        "deid": {"placeholders_per_note": safe_quantiles(n["n_placeholders"], digits=0),
                 "placeholder_hist_rows": ph_rows},
    }


def _num(x) -> int:
    """Suppressed counts are strings; treat them as 0 for derived rates."""
    return x if isinstance(x, int) else 0


def raw_inventory(note_root: Path, coh_ids: set) -> dict:
    """Counts that need only ID / type columns (no text)."""
    note_dir = note_root / "note"
    d = pd.read_csv(note_dir / "discharge.csv.gz", usecols=["hadm_id", "note_type"])
    r = pd.read_csv(note_dir / "radiology.csv.gz", usecols=["hadm_id", "note_type"])
    dd = pd.read_csv(note_dir / "discharge_detail.csv.gz", usecols=["field_name"])
    r_coh = r.dropna(subset=["hadm_id"])
    r_coh = r_coh[r_coh["hadm_id"].astype("int64").isin(coh_ids)]
    d_ids = set(d["hadm_id"].dropna().astype("int64"))
    per_adm = r_coh.groupby("hadm_id").size()
    no_ds = coh_ids - d_ids
    return {
        "cohort_with_discharge_note": safe_count(len(coh_ids & d_ids)),
        "discharge_notes": safe_count(len(d)),
        "discharge_note_types": safe_value_counts(d["note_type"]),
        "discharge_notes_without_hadm": safe_count(d["hadm_id"].isna().sum()),
        "radiology_reports": safe_count(len(r)),
        "radiology_note_types": safe_value_counts(r["note_type"]),
        "radiology_without_hadm_share": safe_rate(r["hadm_id"].isna().sum(), len(r)),
        "discharge_detail_fields": safe_value_counts(dd["field_name"]),
        "radiology_coverage": {
            "cohort_admissions_with_radiology": safe_count(per_adm.index.nunique()),
            "radiology_reports_per_admission": safe_quantiles(per_adm),
            "admissions_without_discharge_note_but_with_radiology": safe_count(
                len(no_ds & set(per_adm.index.astype("int64")))),
        },
    }


def run(sample: int | None = None) -> dict:
    """Compute the notes summary + figures."""
    # pylint: disable=import-outside-toplevel  # heavy import only when needed
    from transformers import AutoTokenizer

    cfg = load_config()
    _, note_root = L.mimic_paths(cfg)
    max_len = cfg.stage2.max_seq_length
    tok = AutoTokenizer.from_pretrained(cfg.stage2.model_name)
    tok.model_max_length = 10**9  # we measure full length; silence the warning
    coh = _cohort(cfg)
    coh_ids = set(coh["hadm_id"])

    cache = L.CACHE_DIR / f"note_stats{'_s' + str(sample) if sample else ''}.pkl"
    if cache.exists():
        notes = pd.read_pickle(cache)
    else:
        notes = scan_discharge_notes(note_root / "note" / "discharge.csv.gz", coh_ids,
                                     tok, max_len, sample)
        L.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        notes.to_pickle(cache)

    summary = summarise(notes, coh, max_len, note_root, raw_inventory(note_root, coh_ids))
    summary["sample"] = sample
    name = "notes_summary.json" if not sample else f"notes_summary_sample{sample}.json"
    write_json(summary, RESULTS_DIR / name)
    plots.notes_figures(summary, RESULTS_DIR / "figures", max_len)
    print(f"[eda/notes] wrote {RESULTS_DIR / name}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MIMIC-IV-Note EDA (aggregate-only)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random subset of cohort admissions (quick look)")
    run(ap.parse_args().sample)
