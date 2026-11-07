"""Figures for the EDA -- drawn ONLY from the aggregate summary dicts.

Plot functions never receive a row-level frame, so a figure cannot show
anything the (suppressed) JSON does not already contain.

Style: one categorical order (blue = not readmitted / single series,
orange = readmitted), thin marks, recessive grid, direct labels only where
they carry the point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _title(fig, title: str) -> None:
    """Figure-anchored, left-aligned title (never clipped by wide y-labels)."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)


def _num(x):
    """Suppressed counts ('<11') plot as 0."""
    return x if isinstance(x, (int, float)) and x is not None else 0


def hbar(rows: list[dict], label_key: str, value_key: str, title: str, path: Path,
         xlabel: str = "", pct: bool = False, ref: float | None = None,
         color: str = BLUE) -> None:
    """Horizontal bar chart, sorted as given, values direct-labelled."""
    rows = [r for r in rows if r.get(value_key) is not None]
    if not rows:
        return
    labels = [str(r[label_key]) for r in rows][::-1]
    vals = [_num(r[value_key]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    ax.barh(labels, vals, color=color, height=0.62, edgecolor=SURFACE, linewidth=1)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.1%}" if pct else f" {v:,.0f}", va="center",
                fontsize=7.5, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(ref, len(vals) - 0.4, f" overall {ref:.1%}", fontsize=7.5, color=INK2)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(xlabel)
    _title(fig, title)
    ax.margins(x=0.15)
    _save(fig, path)


def hist_pair(h_pos: dict, h_neg: dict, title: str, path: Path, xlabel: str,
              vline: float | None = None, vlabel: str = "") -> None:
    """Two density-normalised histograms (readmitted vs not) as step lines."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for h, color, lab in [(h_neg, BLUE, "Not readmitted"), (h_pos, ORANGE, "Readmitted")]:
        if not h.get("counts"):
            continue
        edges, counts = np.array(h["edges"]), np.array(h["counts"], dtype=float)
        dens = counts / max(counts.sum(), 1)
        ax.step(edges[:-1], dens, where="post", color=color, linewidth=2, label=lab)
    if vline is not None:
        ax.axvline(vline, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(vline, ax.get_ylim()[1] * 0.92, f" {vlabel}", fontsize=7.5, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of admissions")
    _title(fig, title)
    ax.legend(loc="upper right")
    _save(fig, path)


def hist_single(h: dict, title: str, path: Path, xlabel: str, ylabel: str = "Admissions",
                vline: float | None = None, vlabel: str = "") -> None:
    """Single-series histogram as bars."""
    if not h.get("counts"):
        return
    edges, counts = np.array(h["edges"]), np.array(h["counts"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(edges[:-1], counts, width=np.diff(edges) * 0.9, align="edge", color=BLUE)
    if vline is not None:
        ax.axvline(vline, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(vline, max(counts) * 0.95, f" {vlabel}", fontsize=7.5, color=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _title(fig, title)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def grouped_hbar(rows: list[dict], label_key: str, series: list[tuple[str, str, str]],
                 title: str, path: Path, pct: bool = True) -> None:
    """Grouped horizontal bars; ``series`` = [(value_key, legend, color)]."""
    # Keep a row if any series has a value; a suppressed cell plots as 0.
    rows = [r for r in rows if any(r.get(k) is not None for k, _, _ in series)]
    if not rows:
        return
    labels = [str(r[label_key]) for r in rows][::-1]
    y = np.arange(len(rows))
    h = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.3))
    for i, (key, lab, color) in enumerate(series):
        vals = [_num(r[key]) for r in rows][::-1]
        ax.barh(y + (i - (len(series) - 1) / 2) * h, vals, height=h * 0.9,
                color=color, label=lab)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _title(fig, title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(series))
    _save(fig, path)


# ──────────────────────────────────────────────────────────────────────────────
# Figure sets
# ──────────────────────────────────────────────────────────────────────────────

def structured_figures(s: dict, out: Path) -> None:
    """All structured-data figures (s01..s12)."""
    hbar(s["cohort_flow"]["steps"], "step", "n_admissions",
         "Cohort flow: admissions retained at each step",
         out / "s01_cohort_flow.png", xlabel="Admissions")

    o = s["outcome"]
    hist_single(o["days_to_readmission"]["histogram_daily"],
                "Days from discharge to readmission (all-cause, within 30 days)",
                out / "s02_days_to_readmission.png", xlabel="Days after discharge",
                ylabel="Readmissions")

    overall = o["unplanned_30d (model target)"]["rate"]
    titles = {"age_band": "age band", "race_group": "race group", "insurance": "insurance",
              "discharge_location": "discharge location", "year_group": "admission years",
              "admission_type": "admission type"}
    for i, (col, name) in enumerate(titles.items(), start=3):
        hbar(s["subgroups"][col], "category", "rate",
             f"Unplanned 30-day readmission rate by {name}",
             out / f"s{i:02d}_rate_by_{col}.png", pct=True, ref=overall)

    # Lab aggregates (_mean/_min/_max) share their _last missingness: show one row each.
    miss = [r for r in s["missingness"]["features_with_missing"]
            if not r["feature"].endswith(("_mean", "_min", "_max"))]
    grouped_hbar(miss, "feature", [("missing_rate_non_icu", "No ICU stay", BLUE),
                                   ("missing_rate_icu", "ICU stay", ORANGE)],
                 "Feature missingness: vitals exist only for ICU admissions",
                 out / "s09_missingness_icu.png")
    grouped_hbar(miss, "feature", [("readmission_rate_if_present", "Value present", BLUE),
                                   ("readmission_rate_if_missing", "Value missing", ORANGE)],
                 "Readmission rate when a feature is present vs. missing",
                 out / "s10_missingness_informative.png")

    hbar(s["univariate_auroc"][:25], "feature", "auroc_oriented",
         "Univariate AUROC per feature (top 25, direction-agnostic)",
         out / "s11_univariate_auroc.png", xlabel="AUROC (0.5 = no signal)")

    util = s["utilisation"]["by_bucket"]
    grouped_hbar(util, "admissions_per_patient",
                 [("share_of_admissions", "Share of admissions", BLUE),
                  ("share_of_positives", "Share of readmissions", ORANGE)],
                 "Admissions per patient: frequent users carry most readmissions",
                 out / "s12_utilisation.png")

    fd = s["feature_distributions"]
    for col, xl in [("los_days", "Length of stay (days)"),
                    ("n_prior_admissions", "Prior admissions"),
                    ("charlson_index", "Charlson index")]:
        hist_pair(fd[col]["hist_readmitted"], fd[col]["hist_not_readmitted"],
                  f"{xl}: readmitted vs. not (clipped to p1-p99)",
                  out / f"s13_dist_{col}.png", xlabel=xl)


def notes_figures(s: dict, out: Path, max_tokens: int) -> None:
    """All note figures (n01..n06)."""
    ln = s["length"]
    hist_pair(ln["tokens_hist_readmitted"], ln["tokens_hist_not_readmitted"],
              "Discharge-note length in Clinical-Longformer tokens",
              out / "n01_token_length.png", xlabel="Tokens",
              vline=max_tokens, vlabel=f"Stage 2 window ({max_tokens})")

    sec = s["sections"]["per_section"]
    grouped_hbar(sec, "section", [("present_share", "Section present", BLUE),
                                  ("beyond_window_share_of_present",
                                   f"...but starts after token {max_tokens}", ORANGE)],
                 "Note sections: presence and truncation by the Stage 2 window",
                 out / "n02_sections_truncation.png")

    kw = s["keywords"]["per_concept"]
    grouped_hbar(kw, "concept", [("rate_if_absent", "Not mentioned", BLUE),
                                 ("rate_if_mentioned", "Mentioned", ORANGE)],
                 "Readmission rate by mention of Stage 3 grounds concepts",
                 out / "n03_keywords_rate.png")
    hbar(kw, "concept", "mention_share", "Share of discharge notes mentioning each concept",
         out / "n04_keywords_prevalence.png", pct=True)

    lag = s["timing"].get("hours_dischtime_to_storetime_hist")
    if lag:
        hist_single(lag, "Hours from discharge to note being stored (p1-p99)",
                    out / "n05_note_store_lag.png", xlabel="Hours after discharge",
                    ylabel="Notes", vline=0, vlabel="discharge")

    hbar(s["deid"]["placeholder_hist_rows"], "bucket", "share",
         "De-identification placeholders ('___') per note", out / "n06_deid_placeholders.png",
         pct=True)
