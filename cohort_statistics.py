#!/usr/bin/env python3
"""
cohort_statistics.py
====================
Clinical association analyses for the 14-patient kidney cohort.

Analyses
--------
1. eGFR correlation
   Spearman ρ between patient eGFR and per-metabolite interaction score.
   Outputs: ranked bar chart (with source→sink cell-type labels),
            scatter plots for top hits (title carries source→sink).

2. Fibrosis association
   Spearman ρ between fibrosis % and metabolite score.
   Outputs: ranked bar chart, scatter plots, box plots by tertile.

3. Sex association  (Male vs Female)
   Mann-Whitney U per metabolite.
   Outputs: volcano plot.

4. Hypertension association  (yes vs no)
   Mann-Whitney U per metabolite.
   Outputs: volcano plot.

5. Diabetes association  (yes vs no)
   Mann-Whitney U per metabolite.
   Outputs: volcano plot.

6. Multi-variable OLS regression
   score ~ eGFR + fibrosis + age + sex_num + is_DKD + is_HKD
   Requires: pip install statsmodels
   Outputs: β-coefficient heatmap, regression_results.csv.

7. Clinical summary heatmap
   Rows = top metabolites (labelled as met  [src→snk]), columns = patients.
   Top annotation strips: group colour, sex colour, eGFR (RdYlGn),
   fibrosis (Oranges).
   All significant hits saved to all_significant_hits.csv.

Cell-type annotation additions (NEW)
-------------------------------------
* build_celltype_map()          — builds a (metabolite → source, sink) lookup
                                   from patient_results across the full cohort.
* _met_label()                  — returns "met  [src→snk]" for axis / tick labels.
* All bar charts, scatter titles, volcano labels, heatmap row labels, and
  regression heatmap row labels now carry source→sink cell-type context.
* NEW plot: plot_egfr_celltype_strip()
  A supplementary strip chart showing, for each significantly eGFR-correlated
  metabolite, which cell-type pair drives the exchange and the direction of flux.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec

from run_cohort_pipeline import COHORT_METADATA, COHORT_META_BY_ID

try:
    from plot_publication_figures import (
        plot_regression_summary,
        plot_clinical_correlation_panel,
        plot_detection_dotplot,
        GROUP_COLORS as _PUB_GROUP_COLORS,
    )
    HAS_PUB_PLOTS = True
except ImportError:
    HAS_PUB_PLOTS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StatConfig:
    base_out_dir:      str   = "cohort_output"
    min_score:         float = 50.0    # exclude metabolites below this mean score
    p_thresh:          float = 0.05
    top_n:             int   = 20      # metabolites shown in summary plots
    fibrosis_tertile_pcts: List[float] = field(
        default_factory=lambda: [33.3, 66.7]
    )   # percentile cut-points for low/medium/high
    verbose:           bool  = True


# =============================================================================
# Cell-type map  (NEW)
# =============================================================================

def build_celltype_map(patient_results: Dict) -> Dict[str, Tuple[str, str]]:
    """
    Build a majority-vote (metabolite → (source_cell_type, sink_cell_type)) map
    by scanning all patients' per_region_data interactions.

    This is the same logic as build_source_sink_map() in cohort_comparison but
    exposed here so cohort_statistics can annotate plots without importing the
    full comparison module.

    Returns
    -------
    dict  metabolite_name → (source_str, sink_str)
          Falls back to ("?", "?") for metabolites with no interaction data.
    """
    from collections import Counter
    votes: Dict[str, Counter] = {}

    for pid, results in patient_results.items():
        per_region = getattr(results, "per_region_data", {}) or {}
        for key, data in per_region.items():
            for met, pairs in (data.get("interactions", {}) or {}).items():
                if not pairs:
                    continue
                src = str(pairs[0].get("source", "?"))
                snk = str(pairs[0].get("sink",   "?"))
                votes.setdefault(met, Counter())[(src, snk)] += 1

    return {m: c.most_common(1)[0][0] for m, c in votes.items() if c}


def _met_label(met: str, ct_map: Dict[str, Tuple[str, str]],
               max_ct_len: int = 10) -> str:
    """
    Return a compact label:  "met  [src→snk]"

    Cell-type names are truncated to max_ct_len characters each so that long
    names (e.g. "Endo_Peritubular") do not dominate the tick labels.
    Cell-type part is omitted entirely when source/sink are both "?".
    """
    src, snk = ct_map.get(met, ("?", "?"))
    if src == "?" and snk == "?":
        return met

    def _trunc(s: str) -> str:
        return s if len(s) <= max_ct_len else s[:max_ct_len - 1] + "…"

    return f"{met}  [{_trunc(src)}→{_trunc(snk)}]"


# =============================================================================
# Build score matrix
# =============================================================================

def build_score_matrix(patient_results: Dict) -> pd.DataFrame:
    """
    Build a (metabolite × patient) score matrix from per-patient results.

    Delegates to build_full_score_matrix() from cohort_comparison, which
    sums ALL pairs per metabolite (not just top-1) and covers the full
    cohort-wide metabolite universe.  NaN (not detected in a patient) is
    filled with 0.0 so the matrix is fully numeric for statistics here.

    Returns
    -------
    pd.DataFrame  rows = metabolites, columns = patient IDs,
                  values = total interaction score (0 if not detected)
    """
    from cohort_comparison import build_full_score_matrix
    return build_full_score_matrix(patient_results).fillna(0.0)


# =============================================================================
# Internal helpers
# =============================================================================

def _clinical_df() -> pd.DataFrame:
    """
    Return clinical metadata as a tidy DataFrame indexed by patient_id.
    Adds numeric encodings for regression.
    """
    rows = []
    for m in COHORT_METADATA:
        rows.append({
            "patient_id":   m["id"],
            "age":          float(m["age"]),
            "sex":          m["sex"],
            "sex_num":      1.0 if m["sex"] == "Male" else 0.0,
            "race":         m["race"],
            "hypertension": float(int(m["hypertension"])),
            "diabetes":     float(int(m["diabetes"])),
            "eGFR":         float(m["eGFR"]),
            "fibrosis":     float(m["fibrosis"]),
            "group":        m["group"],
            "is_DKD":       float(int(m["group"] == "DKD")),
            "is_HKD":       float(int(m["group"] == "HKD")),
            "is_disease":   float(int(m["group"] != "Control")),
        })
    return pd.DataFrame(rows).set_index("patient_id")


def _filter_score_matrix(sm: pd.DataFrame, cfg: StatConfig) -> pd.DataFrame:
    """Keep only metabolites with mean score ≥ min_score."""
    from plotting import _should_exclude_metabolite
    keep = [
        m for m in sm.index
        if not _should_exclude_metabolite(m)
        and float(sm.loc[m].mean()) >= cfg.min_score
    ]
    return sm.loc[keep]


def _group_colors() -> Dict[str, str]:
    return {"Control": "#27ae60", "DKD": "#e74c3c", "HKD": "#e67e22"}


# =============================================================================
# Scatter helper (reused by eGFR and fibrosis)
# =============================================================================

def _winsorise(arr: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Winsorise at pct/100-pct percentiles to reduce outlier influence."""
    lo = np.nanpercentile(arr, pct)
    hi = np.nanpercentile(arr, 100.0 - pct)
    return np.clip(arr, lo, hi)


def _scatter_clinical(
    sm: pd.DataFrame,
    clinical_vals: np.ndarray,
    patient_ids: List[str],
    result_df: pd.DataFrame,
    x_label: str,
    out_dir: str,
    cfg: StatConfig,
    n_top: int = 4,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    """
    Scatter plots of interaction score vs a continuous clinical variable.

    Systematic improvements:
    1. LOG TRANSFORM  — scores are log10(score+1) transformed before plotting.
    2. WINSORISATION  — OLS line fitted on Winsorised (1st/99th pct) log-scores.
    3. PATIENT LABELS — each dot labelled with last 4 chars of patient ID.
    4. SPEARMAN ON LOG SCORES — rho recomputed on log-transformed scores.
    5. Y-AXIS HEADROOM — 15% headroom above maximum.
    6. CELL-TYPE CONTEXT (NEW) — panel title now includes "source→sink" cell-type
       pair derived from ct_map, so each scatter makes clear which exchange axis
       drives the association.
    """
    clin = _clinical_df()
    top  = result_df.head(n_top)
    if top.empty:
        return

    fig, axes = plt.subplots(1, len(top),
                              figsize=(5.0 * len(top), 5.2),
                              sharey=False)
    if len(top) == 1:
        axes = [axes]

    gc = _group_colors()

    for ax, (_, row) in zip(axes, top.iterrows()):
        met = row["metabolite"]
        if met not in sm.index:
            ax.set_visible(False)
            continue

        raw_scores = sm.loc[met].values.astype(float)
        log_scores = np.log10(raw_scores + 1.0)
        y_label    = "log10(score + 1)"
        colors     = [gc.get(clin.loc[p, "group"], "#aaaaaa")
                      for p in patient_ids]

        ax.scatter(clinical_vals, log_scores, c=colors, s=70,
                   edgecolors="white", linewidths=0.6, zorder=4)

        for xi, yi, pid in zip(clinical_vals, log_scores, patient_ids):
            ax.annotate(
                pid[-4:],
                (xi, yi),
                textcoords="offset points", xytext=(4, 3),
                fontsize=6.5, color="#444444", alpha=0.85, zorder=5,
            )

        mask = np.isfinite(clinical_vals) & np.isfinite(log_scores)
        if mask.sum() >= 3:
            w_scores = _winsorise(log_scores[mask])
            m_, b, *_ = stats.linregress(clinical_vals[mask], w_scores)
            xl = np.linspace(clinical_vals[mask].min(),
                             clinical_vals[mask].max(), 200)
            ax.plot(xl, m_ * xl + b, color="#333333",
                    linewidth=1.6, linestyle="--", alpha=0.80)
            w_lo = np.nanpercentile(log_scores[mask], 1.0)
            w_hi = np.nanpercentile(log_scores[mask], 99.0)
            for yb in (w_lo, w_hi):
                ax.axhline(yb, color="#aaaaaa", linewidth=0.7,
                           linestyle=":", alpha=0.5)

        if mask.sum() >= 3:
            rho_log, p_log = stats.spearmanr(
                clinical_vals[mask], log_scores[mask])
        else:
            rho_log, p_log = np.nan, np.nan
        rho_raw = row.get("rho", np.nan)
        p_raw   = row.get("p_value", np.nan)

        # ── Cell-type context (NEW) ──────────────────────────────────────
        if ct_map is not None:
            src_lbl, snk_lbl = ct_map.get(met, ("?", "?"))
        else:
            src_lbl = str(row.get("source", "?")) if "source" in row.index else "?"
            snk_lbl = str(row.get("sink",   "?")) if "sink"   in row.index else "?"

        t1 = f"{met}  [{src_lbl}→{snk_lbl}]"
        t2 = (f"ρ_log={round(rho_log, 2)}  p={round(p_log, 3)}"
              f"  |  ρ_raw={round(rho_raw, 2)}  p={round(p_raw, 3)}")
        ax.set_title(t1 + "\n" + t2, fontsize=8.0, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=9, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=9)

        ymax = np.nanmax(log_scores) if np.any(np.isfinite(log_scores)) else 1.0
        ax.set_ylim(bottom=-0.05 * ymax, top=ymax * 1.18)
        ax.grid(alpha=0.18, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    legend_els = [mpatches.Patch(color=c, label=g) for g, c in gc.items()]
    fig.legend(handles=legend_els, loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=8,
               title="Group", title_fontsize=8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fname = x_label.replace(" ", "_").replace("%", "pct").lower()
    plt.savefig(os.path.join(out_dir, f"{fname}_scatter.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# Binary volcano helper (sex, hypertension, diabetes)
# =============================================================================

def _plot_binary_volcano(
    result: pd.DataFrame,
    label_a: str,
    label_b: str,
    fc_col: str,
    out_dir: str,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    """
    Small volcano for a binary clinical variable association.

    Cell-type annotation (NEW): significant dots are labelled as
    "met [src→snk]" when ct_map is provided, making it immediately clear
    which exchange axis is highlighted.
    """
    if result.empty:
        return
    fc   = result[fc_col].values
    logp = -np.log10(result["p_value"].clip(lower=1e-300))
    sig  = result["significant"].values

    colors = np.where(sig & (fc > 0), "#c0392b",
                      np.where(sig & (fc < 0), "#2980b9", "#aaaaaa"))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(fc, logp, c=colors,
               s=np.where(sig, 60, 22),
               alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.axhline(-np.log10(cfg.p_thresh), color="#555555",
               linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(0, color="#555555", linewidth=0.8, alpha=0.4)
    ax.set_ylim(bottom=0)

    for _, row in result[result["significant"]].head(10).iterrows():
        met = row["metabolite"]
        # ── Cell-type label (NEW) ─────────────────────────────────────────
        lbl = _met_label(met, ct_map) if ct_map is not None else met
        ax.annotate(
            lbl,
            (row[fc_col], -np.log10(max(row["p_value"], 1e-300))),
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow",
                      alpha=0.4, edgecolor="none"),
        )

    ax.set_xlabel(f"Log₂FC  ({label_a} / {label_b})",
                  fontsize=10, fontweight="bold")
    ax.set_ylabel("–log₁₀(p)", fontsize=10, fontweight="bold")
    ax.set_title(f"{label_a} vs {label_b}",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    safe = (f"{label_a}_{label_b}"
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("+", "pos").replace("-", "neg"))
    plt.savefig(os.path.join(out_dir, f"volcano_{safe}.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# NEW: eGFR cell-type strip chart
# =============================================================================

def plot_egfr_celltype_strip(
    result_df: pd.DataFrame,
    ct_map: Dict[str, Tuple[str, str]],
    out_dir: str,
    cfg: StatConfig,
    n_top: int = 30,
) -> Optional[str]:
    """
    Supplementary strip chart: for each of the top eGFR-correlated metabolites
    (by |ρ|), show:
      • A horizontal bar whose length = Spearman ρ and colour = direction.
      • A text annotation giving the source→sink cell-type pair.
      • Dot-size = detection rate (fraction of patients with score > 0).

    This plot answers directly: "which cell-type exchange axis drives the
    eGFR association?" without cluttering the main scatter or bar chart.

    Parameters
    ----------
    result_df : pd.DataFrame  — output of analyze_egfr_correlation()
                                must have columns: metabolite, rho, p_value.
    ct_map    : dict           — metabolite → (source, sink)
    out_dir   : str
    cfg       : StatConfig
    n_top     : int            — number of metabolites to show (sorted by |ρ|)

    Returns
    -------
    str  path to saved figure, or None.
    """
    if result_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    plot_df = (result_df
               .assign(_abs_rho=lambda d: d["rho"].abs())
               .nlargest(n_top, "_abs_rho")
               .sort_values("rho", ascending=True)
               .reset_index(drop=True))

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.45)))
    ax.set_facecolor("#f8f9fa")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        met   = row["metabolite"]
        rho   = float(row["rho"])
        pv    = float(row["p_value"])
        sig   = pv < cfg.p_thresh
        src, snk = ct_map.get(met, ("?", "?"))

        bar_col  = "#c0392b" if rho > 0 else "#2980b9"
        bar_alpha = 0.90 if sig else 0.45

        # Bar
        ax.barh(i, rho, color=bar_col, alpha=bar_alpha,
                edgecolor="white", linewidth=0.4, height=0.72)

        # Cell-type annotation (right of bar or at zero)
        ct_txt = f"[{src}→{snk}]"
        x_ann  = rho + (0.02 if rho >= 0 else -0.02)
        ha_ann = "left"  if rho >= 0 else "right"
        ax.text(x_ann, i, ct_txt,
                va="center", ha=ha_ann,
                fontsize=7.2, color="#333333",
                fontstyle="italic")

        # Star for significance
        if sig:
            star_x = rho + (0.01 if rho >= 0 else -0.01)
            ax.text(star_x - (0.015 if rho < 0 else 0), i + 0.38,
                    "  *" if rho >= 0 else "*  ",
                    va="bottom", ha=ha_ann,
                    fontsize=9, color=bar_col, fontweight="bold")

    # Tick labels: met name only (cell-type is shown inline)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(plot_df["metabolite"].tolist(), fontsize=8.5)

    ax.axvline(0, color="#444444", linewidth=0.9, alpha=0.6, zorder=1)
    for rv in (-0.5, -0.3, 0.3, 0.5):
        ax.axvline(rv, color="#dddddd", linewidth=0.6,
                   linestyle=":", zorder=0)

    rho_max = max(plot_df["rho"].abs().max(), 0.5)
    ax.set_xlim(-(rho_max + 0.35), rho_max + 0.35)
    ax.set_xlabel("Spearman ρ  (score ~ eGFR)",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        "eGFR–Exchange Associations with Cell-Type Context\n"
        "Italic labels show [source_cell → sink_cell]  ·  * = p < 0.05  ·  "
        f"top {n} by |ρ|",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    # Colour legend
    leg = [
        mpatches.Patch(color="#c0392b", alpha=0.90, label="Positive ρ (↑ score with eGFR)"),
        mpatches.Patch(color="#2980b9", alpha=0.90, label="Negative ρ (↓ score with eGFR)"),
        mpatches.Patch(color="#888888", alpha=0.45, label="p ≥ 0.05 (not significant)"),
    ]
    ax.legend(handles=leg, fontsize=8, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    fp = os.path.join(out_dir, "egfr_celltype_strip.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


# =============================================================================
# NEW: per-metabolite cell-type context table
# =============================================================================

def save_celltype_context_table(
    result_df: pd.DataFrame,
    ct_map: Dict[str, Tuple[str, str]],
    out_path: str,
) -> str:
    """
    Attach source/sink cell-type columns to any result DataFrame and save as CSV.

    Adds columns:
      source_cell   — majority-vote source cell type
      sink_cell     — majority-vote sink cell type
      exchange_axis — "source_cell→sink_cell"

    Parameters
    ----------
    result_df : pd.DataFrame  — any result table that has a "metabolite" column
    ct_map    : dict
    out_path  : str

    Returns
    -------
    str  path to saved CSV
    """
    df = result_df.copy()
    df["source_cell"]   = df["metabolite"].map(
        lambda m: ct_map.get(m, ("?", "?"))[0])
    df["sink_cell"]     = df["metabolite"].map(
        lambda m: ct_map.get(m, ("?", "?"))[1])
    df["exchange_axis"] = df["source_cell"] + "→" + df["sink_cell"]
    df.to_csv(out_path, index=False)
    return out_path


# =============================================================================
# 1. eGFR correlation
# =============================================================================

def analyze_egfr_correlation(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Spearman ρ between per-patient eGFR and each metabolite's score.

    Changes vs. original
    --------------------
    * result DataFrame now carries source_cell / sink_cell / exchange_axis columns
      (populated from ct_map when available).
    * Ranked bar chart y-tick labels show "met  [src→snk]" when ct_map available.
    * Scatter panel titles carry "[src→snk]" cell-type context.
    * NEW: calls plot_egfr_celltype_strip() to produce the supplementary
      cell-type context strip chart.
    * Saves annotated CSV to egfr_correlation_with_celltypes.csv.

    Returns
    -------
    pd.DataFrame  columns: metabolite, source_cell, sink_cell, exchange_axis,
                           rho, p_value, significant
                  sorted by |rho| descending
    """
    out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics",
                           "egfr_correlation")
    os.makedirs(out_dir, exist_ok=True)

    clin   = _clinical_df()
    sm     = _filter_score_matrix(score_matrix, cfg)
    common = [p for p in sm.columns if p in clin.index]
    sm     = sm[common]
    egfr   = clin.loc[common, "eGFR"].values.astype(float)

    rows = []
    for met in sm.index:
        scores = sm.loc[met].values.astype(float)
        try:
            rho, p = stats.spearmanr(scores, egfr)
        except Exception:
            rho, p = 0.0, 1.0
        src, snk = (ct_map.get(met, ("?", "?"))
                    if ct_map else ("?", "?"))
        rows.append({
            "metabolite":    met,
            "source_cell":   src,
            "sink_cell":     snk,
            "exchange_axis": f"{src}→{snk}",
            "rho":           float(rho),
            "p_value":       float(p),
            "significant":   bool(p < cfg.p_thresh),
        })

    result = (pd.DataFrame(rows)
              .sort_values("rho", key=abs, ascending=False)
              .reset_index(drop=True))

    # Save CSV with cell-type columns
    result.to_csv(
        os.path.join(out_dir, "egfr_correlation.csv"), index=False)
    if ct_map:
        result.to_csv(
            os.path.join(out_dir, "egfr_correlation_with_celltypes.csv"),
            index=False)

    # ── Ranked bar chart with cell-type tick labels ───────────────────────────
    top = result.nlargest(cfg.top_n, "rho", keep="all").iloc[:cfg.top_n]
    bot = result.nsmallest(cfg.top_n, "rho", keep="all").iloc[:cfg.top_n]
    bar_df = (pd.concat([top, bot])
              .drop_duplicates("metabolite")
              .sort_values("rho", ascending=True))

    fig, ax = plt.subplots(figsize=(12, max(6, len(bar_df) * 0.42)))
    bar_colors = ["#2980b9" if r > 0 else "#c0392b" for r in bar_df["rho"]]

    # NEW: y-tick labels include source→sink when ct_map available
    if ct_map:
        labels = [
            _met_label(row["metabolite"], ct_map)
            + ("*" if row["significant"] else "")
            for _, row in bar_df.iterrows()
        ]
    else:
        labels = [
            f"{row['metabolite']}{'*' if row['significant'] else ''}"
            for _, row in bar_df.iterrows()
        ]

    ax.barh(np.arange(len(bar_df)), bar_df["rho"].values,
            color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_yticks(np.arange(len(bar_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Spearman ρ  (score ~ eGFR)", fontsize=10, fontweight="bold")
    ax.set_title(
        "eGFR Correlation with Metabolite Exchange Scores\n"
        "Labels: met  [source_cell→sink_cell]  ·  Blue = positive  ·  * = p < 0.05",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "egfr_rho_bar.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── Scatter for top hits (with ct_map passed through) ────────────────────
    _scatter_clinical(sm, egfr, common, result, "eGFR",
                      out_dir, cfg, n_top=min(4, len(result)),
                      ct_map=ct_map)

    # ── NEW: cell-type strip chart ────────────────────────────────────────────
    if ct_map:
        fp_strip = plot_egfr_celltype_strip(result, ct_map, out_dir, cfg)
        if cfg.verbose and fp_strip:
            print(f"  eGFR cell-type strip → {fp_strip}")

    # ── Publication-quality clinical correlation panel ────────────────────────
    if HAS_PUB_PLOTS:
        try:
            patient_meta_pub = clin.loc[common, ["group", "eGFR"]]
            plot_clinical_correlation_panel(
                score_matrix = sm,
                patient_meta = patient_meta_pub,
                corr_df      = result.head(4),
                clinical_col = "eGFR",
                out_path     = os.path.join(out_dir, "egfr_scatter_pub.png"),
                title        = "Top eGFR–exchange associations (log-transformed, Winsorised OLS)",
            )
        except Exception as _e:
            if cfg.verbose:
                print(f"  eGFR pub scatter skipped: {_e}")

    if cfg.verbose:
        n_sig = result["significant"].sum()
        top0  = result.iloc[0]
        print(f"  eGFR: {n_sig} significant metabolites  "
              f"(top: {top0['metabolite']}  "
              f"[{top0['source_cell']}→{top0['sink_cell']}]  "
              f"ρ={top0['rho']:.3f}  p={top0['p_value']:.3g})")

    return result


# =============================================================================
# 2. Fibrosis association
# =============================================================================

def analyze_fibrosis_association(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Spearman ρ between fibrosis % and metabolite score.
    Also produces stratified box plots by low / medium / high tertile.

    Changes vs. original
    --------------------
    * result DataFrame carries source_cell / sink_cell / exchange_axis.
    * Scatter titles carry "[src→snk]" cell-type context.
    * Box plot panel titles carry "[src→snk]".
    * Saves annotated CSV.
    """
    out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics", "fibrosis")
    os.makedirs(out_dir, exist_ok=True)

    clin   = _clinical_df()
    sm     = _filter_score_matrix(score_matrix, cfg)
    common = [p for p in sm.columns if p in clin.index]
    sm     = sm[common]
    fib    = clin.loc[common, "fibrosis"].values.astype(float)

    rows = []
    for met in sm.index:
        scores = sm.loc[met].values.astype(float)
        try:
            rho, p = stats.spearmanr(scores, fib)
        except Exception:
            rho, p = 0.0, 1.0
        src, snk = (ct_map.get(met, ("?", "?"))
                    if ct_map else ("?", "?"))
        rows.append({
            "metabolite":    met,
            "source_cell":   src,
            "sink_cell":     snk,
            "exchange_axis": f"{src}→{snk}",
            "rho":           float(rho),
            "p_value":       float(p),
            "significant":   bool(p < cfg.p_thresh),
        })

    result = (pd.DataFrame(rows)
              .sort_values("p_value")
              .reset_index(drop=True))
    result.to_csv(os.path.join(out_dir, "fibrosis_correlation.csv"), index=False)
    if ct_map:
        result.to_csv(
            os.path.join(out_dir, "fibrosis_correlation_with_celltypes.csv"),
            index=False)

    # Scatter for top hits (ct_map passed through)
    _scatter_clinical(sm, fib, common, result, "Fibrosis (%)",
                      out_dir, cfg, n_top=min(4, len(result)),
                      ct_map=ct_map)

    # Box plots by tertile — titles now include source→sink
    tertile_bounds = np.percentile(fib, cfg.fibrosis_tertile_pcts)
    tertiles       = np.digitize(fib, tertile_bounds)   # 0=low,1=mid,2=high
    t_labels       = ["Low", "Medium", "High"]

    top_sig = result[result["significant"]].head(6)
    if not top_sig.empty:
        n_m   = len(top_sig)
        fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 5), sharey=False)
        if n_m == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, top_sig.iterrows()):
            met    = row["metabolite"]
            scores = sm.loc[met].values.astype(float)
            groups = [scores[tertiles == t] for t in range(3)]
            ax.boxplot(groups, labels=t_labels, patch_artist=True,
                       boxprops=dict(facecolor="#a8d4f5", color="#2980b9"),
                       medianprops=dict(color="#c0392b", linewidth=2))
            # NEW: include source→sink in title
            src_t = row.get("source_cell", "?")
            snk_t = row.get("sink_cell",   "?")
            ax.set_title(
                f"{met}  [{src_t}→{snk_t}]\nρ={row['rho']:.2f}  "
                f"p={row['p_value']:.3g}",
                fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Fibrosis Tertile", fontsize=9)
            ax.set_ylabel("Interaction Score", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Metabolite Scores by Fibrosis Tertile",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fibrosis_boxplots.png"),
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    if cfg.verbose:
        n_sig = result["significant"].sum()
        print(f"  Fibrosis: {n_sig} significant metabolites")

    return result


# =============================================================================
# 3. Sex association
# =============================================================================

def analyze_sex_association(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Mann-Whitney U: Male vs Female metabolite scores.
    Volcano labels now carry [src→snk] cell-type context.
    """
    out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics", "sex")
    os.makedirs(out_dir, exist_ok=True)

    clin       = _clinical_df()
    sm         = _filter_score_matrix(score_matrix, cfg)
    common     = [p for p in sm.columns if p in clin.index]
    sm         = sm[common]
    male_ids   = [p for p in common if clin.loc[p, "sex"] == "Male"]
    female_ids = [p for p in common if clin.loc[p, "sex"] == "Female"]

    rows = []
    for met in sm.index:
        sc_m  = sm.loc[met, male_ids].values.astype(float)
        sc_f  = sm.loc[met, female_ids].values.astype(float)
        m_m   = float(np.mean(sc_m))
        m_f   = float(np.mean(sc_f))
        fc    = float(np.log2((m_m + 1e-6) / (m_f + 1e-6)))
        try:
            _, p = stats.mannwhitneyu(sc_m, sc_f, alternative="two-sided")
        except Exception:
            p = 1.0
        src, snk = (ct_map.get(met, ("?", "?"))
                    if ct_map else ("?", "?"))
        rows.append({
            "metabolite":    met,
            "source_cell":   src,
            "sink_cell":     snk,
            "exchange_axis": f"{src}→{snk}",
            "mean_Male":     m_m,
            "mean_Female":   m_f,
            "log2fc_M_vs_F": fc,
            "p_value":       float(p),
            "significant":   bool(p < cfg.p_thresh),
        })

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    result.to_csv(os.path.join(out_dir, "sex_association.csv"), index=False)

    _plot_binary_volcano(result, "Male", "Female", "log2fc_M_vs_F",
                         out_dir, cfg, ct_map=ct_map)

    if cfg.verbose:
        print(f"  Sex: {result['significant'].sum()} significant metabolites")
    return result


# =============================================================================
# 4 & 5. Hypertension / Diabetes co-morbidity
# =============================================================================

def analyze_comorbidity(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Mann-Whitney U for each binary co-morbidity flag.

    Changes vs. original
    --------------------
    * result DataFrames carry source_cell / sink_cell / exchange_axis.
    * Volcano labels carry [src→snk] cell-type context.

    Returns
    -------
    dict  "hypertension" → DataFrame,  "diabetes" → DataFrame
    """
    results = {}
    for flag, label in [("hypertension", "Hypertension"),
                        ("diabetes",     "Diabetes")]:
        out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics", flag)
        os.makedirs(out_dir, exist_ok=True)

        clin    = _clinical_df()
        sm      = _filter_score_matrix(score_matrix, cfg)
        common  = [p for p in sm.columns if p in clin.index]
        sm      = sm[common]
        pos_ids = [p for p in common if clin.loc[p, flag] == 1.0]
        neg_ids = [p for p in common if clin.loc[p, flag] == 0.0]

        rows = []
        for met in sm.index:
            sc_p  = sm.loc[met, pos_ids].values.astype(float)
            sc_n  = sm.loc[met, neg_ids].values.astype(float)
            m_p   = float(np.mean(sc_p))
            m_n   = float(np.mean(sc_n))
            fc    = float(np.log2((m_p + 1e-6) / (m_n + 1e-6)))
            try:
                _, p = stats.mannwhitneyu(sc_p, sc_n, alternative="two-sided")
            except Exception:
                p = 1.0
            src, snk = (ct_map.get(met, ("?", "?"))
                        if ct_map else ("?", "?"))
            rows.append({
                "metabolite":          met,
                "source_cell":         src,
                "sink_cell":           snk,
                "exchange_axis":       f"{src}→{snk}",
                f"mean_{flag}_yes":    m_p,
                f"mean_{flag}_no":     m_n,
                "log2fc":              fc,
                "p_value":             float(p),
                "significant":         bool(p < cfg.p_thresh),
            })

        df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, f"{flag}_association.csv"), index=False)
        _plot_binary_volcano(df, f"{label} (+)", f"{label} (–)",
                             "log2fc", out_dir, cfg, ct_map=ct_map)
        results[flag] = df

        if cfg.verbose:
            print(f"  {label}: {df['significant'].sum()} significant metabolites")

    return results


# =============================================================================
# 6. Multi-variable OLS regression
# =============================================================================

def analyze_multivariable_regression(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Per-metabolite OLS:
        score ~ eGFR + fibrosis + age + sex_num + is_DKD + is_HKD

    Changes vs. original
    --------------------
    * result DataFrame carries source_cell / sink_cell / exchange_axis.
    * β-coefficient heatmap row labels show "met  [src→snk]" when ct_map
      is provided, so each row identifies its exchange axis at a glance.
    * Saves annotated CSV: regression_results_with_celltypes.csv.

    Returns β coefficients and p-values for each predictor × metabolite.
    Requires: pip install statsmodels
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [SKIP] statsmodels not installed — "
              "install with: pip install statsmodels")
        return pd.DataFrame()

    out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics", "regression")
    os.makedirs(out_dir, exist_ok=True)

    clin       = _clinical_df()
    sm         = _filter_score_matrix(score_matrix, cfg)
    common     = [p for p in sm.columns if p in clin.index]
    sm         = sm[common]
    clin_s     = clin.loc[common]
    predictors = ["eGFR", "fibrosis", "age", "sex_num", "is_DKD", "is_HKD"]

    rows = []
    for met in sm.index:
        df_reg           = clin_s[predictors].copy()
        df_reg["score"]  = sm.loc[met].values.astype(float)
        src, snk = (ct_map.get(met, ("?", "?"))
                    if ct_map else ("?", "?"))
        try:
            model = smf.ols(
                "score ~ " + " + ".join(predictors), data=df_reg
            ).fit()
            for pred in predictors:
                rows.append({
                    "metabolite":    met,
                    "source_cell":   src,
                    "sink_cell":     snk,
                    "exchange_axis": f"{src}→{snk}",
                    "predictor":     pred,
                    "beta":          float(model.params.get(pred, np.nan)),
                    "p_value":       float(model.pvalues.get(pred, np.nan)),
                    "significant":   bool(model.pvalues.get(pred, 1.0)
                                          < cfg.p_thresh),
                    "r_squared":     float(model.rsquared),
                })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.to_csv(
        os.path.join(out_dir, "regression_results.csv"), index=False)
    if ct_map:
        result.to_csv(
            os.path.join(out_dir, "regression_results_with_celltypes.csv"),
            index=False)

    # ── β-coefficient heatmap with source→sink row labels ────────────────────
    pivot    = result.pivot_table(
        index="metabolite", columns="predictor",
        values="beta", aggfunc="first"
    )
    top_mets = pivot.abs().mean(axis=1).nlargest(cfg.top_n).index
    pivot    = pivot.loc[top_mets]

    # NEW: build annotated row labels
    if ct_map:
        row_labels = [_met_label(m, ct_map) for m in pivot.index]
    else:
        row_labels = list(pivot.index)

    fig, ax = plt.subplots(figsize=(len(predictors) * 1.6 + 2,
                                    max(6, len(top_mets) * 0.45)))
    sns.heatmap(
        pivot, cmap="coolwarm", center=0, ax=ax,
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        linewidths=0.4, linecolor="#eeeeee",
        cbar_kws={"label": "β coefficient", "shrink": 0.6},
        yticklabels=row_labels,
    )
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title(
        f"Regression Coefficients: Score ~ Clinical Predictors\n"
        f"Top {len(top_mets)} metabolites by mean |β|  ·  "
        "rows labelled as met  [source_cell→sink_cell]",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regression_beta_heatmap.png"),
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # ── Publication regression dot plot ──────────────────────────────────────
    if HAS_PUB_PLOTS:
        try:
            plot_regression_summary(
                reg_df   = result,
                out_path = os.path.join(out_dir, "regression_summary_pub.png"),
                top_n    = min(12, int(result.significant.sum() or 12)),
            )
        except Exception as _e:
            if cfg.verbose:
                print(f"  Regression pub plot skipped: {_e}")

    if cfg.verbose:
        n_sig = result["significant"].sum()
        print(f"  Regression: {n_sig} significant predictor-metabolite pairs")

    return result


# =============================================================================
# 7. Clinical summary heatmap
# =============================================================================

def plot_clinical_summary_heatmap(
    score_matrix: pd.DataFrame,
    cfg: StatConfig,
    ct_map: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Optional[str]:
    """
    Combined figure:
      Top strip  : group colour bar, sex colour bar, eGFR (RdYlGn),
                   fibrosis (Oranges)
      Main body  : log10(score+1) for top metabolites × patients

    Changes vs. original
    --------------------
    * Row labels on the left now show "met  [src→snk]" when ct_map is
      provided, giving each row immediate cell-type context.

    Patients are sorted by (group, eGFR).
    """
    out_dir = os.path.join(cfg.base_out_dir, "clinical_statistics")
    os.makedirs(out_dir, exist_ok=True)

    clin   = _clinical_df()
    sm     = _filter_score_matrix(score_matrix, cfg)
    common = sorted(
        [p for p in sm.columns if p in clin.index],
        key=lambda p: (clin.loc[p, "group"], clin.loc[p, "eGFR"])
    )
    sm     = sm[common]

    top_mets  = sm.mean(axis=1).nlargest(cfg.top_n).index
    data_log  = np.log10(sm.loc[top_mets] + 1).values

    n_pat  = len(common)
    n_met  = len(top_mets)
    fig_w  = max(14, n_pat * 0.9 + 4)
    fig_h  = max(9,  n_met * 0.38 + 3.5)

    fig    = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # GridSpec: 4 annotation rows + n_met heatmap rows, n_pat+2 columns
    gs     = GridSpec(
        n_met + 4, n_pat + 2,
        figure=fig, hspace=0.02, wspace=0.02,
        left=0.22, right=0.90, top=0.90, bottom=0.10,
    )

    gc          = _group_colors()
    sex_col     = {"Male": "#3498db", "Female": "#e91e8c"}
    egfr_norm   = mcolors.Normalize(
        vmin=clin.loc[common, "eGFR"].min(),
        vmax=clin.loc[common, "eGFR"].max()
    )
    fib_norm    = mcolors.Normalize(
        vmin=0, vmax=max(clin.loc[common, "fibrosis"].max(), 1)
    )
    egfr_cmap   = plt.get_cmap("RdYlGn")
    fib_cmap    = plt.get_cmap("Oranges")

    def _strip(row_idx, values, cmap_or_map, title):
        ax = fig.add_subplot(gs[row_idx, :n_pat])
        for j, (pid, v) in enumerate(zip(common, values)):
            col = (cmap_or_map(v) if callable(cmap_or_map)
                   else cmap_or_map.get(v, "#aaaaaa"))
            ax.add_patch(plt.Rectangle((j, 0), 1, 1,
                facecolor=col, edgecolor="white", lw=0.4))
        ax.set_xlim(0, n_pat); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(-0.5, 0.5, title, transform=ax.transData,
                fontsize=7, va="center", ha="right", fontweight="bold")

    _strip(0, [clin.loc[p, "group"] for p in common], gc,     "Group")
    _strip(1, [clin.loc[p, "sex"]   for p in common], sex_col,"Sex")
    _strip(2, [egfr_cmap(egfr_norm(clin.loc[p, "eGFR"])) for p in common],
           lambda x: x, "eGFR")
    _strip(3, [fib_cmap(fib_norm(clin.loc[p, "fibrosis"])) for p in common],
           lambda x: x, "Fibrosis")

    # Main heatmap — row labels carry [src→snk] when ct_map available
    main_cmap = plt.get_cmap("YlOrRd")
    vmin      = float(data_log.min())
    vmax      = float(data_log.max())
    hm_norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, met in enumerate(top_mets):
        for j, pid in enumerate(common):
            v   = float(data_log[i, j])
            ax_c = fig.add_subplot(gs[i + 4, j])
            ax_c.add_patch(plt.Rectangle((0, 0), 1, 1,
                facecolor=main_cmap(hm_norm(v)),
                edgecolor="white", lw=0.25))
            ax_c.axis("off")
            if j == 0:
                # NEW: use [src→snk] label when ct_map available
                row_lbl = (_met_label(met, ct_map, max_ct_len=9)
                           if ct_map else met)
                ax_c.text(-0.12, 0.5, row_lbl,
                          transform=ax_c.transAxes,
                          fontsize=6.8, va="center", ha="right",
                          fontweight="bold")
            if i == n_met - 1:
                ax_c.text(0.5, -0.30,
                          f"{pid}\n({clin.loc[pid,'group']})",
                          transform=ax_c.transAxes,
                          fontsize=6, va="top", ha="center", rotation=45)

    # Colourbars
    for cmap_, norm_, label_, pos_ in [
        (main_cmap,  hm_norm,   "log₁₀(Score+1)",  [0.91, 0.20, 0.012, 0.50]),
        (egfr_cmap,  egfr_norm, "eGFR (mL/min)",   [0.91, 0.73, 0.012, 0.10]),
        (fib_cmap,   fib_norm,  "Fibrosis (%)",     [0.91, 0.61, 0.012, 0.10]),
    ]:
        sm_cb = plt.cm.ScalarMappable(cmap=cmap_, norm=norm_)
        sm_cb.set_array([])
        cb = fig.colorbar(sm_cb, ax=None, cax=fig.add_axes(pos_))
        cb.set_label(label_, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # Legend
    legend_els = (
        [mpatches.Patch(color=gc[g], label=g)     for g in ["Control","DKD","HKD"]] +
        [mpatches.Patch(color=sex_col[s], label=s) for s in ["Male","Female"]]
    )
    fig.legend(handles=legend_els, loc="lower right",
               bbox_to_anchor=(0.90, 0.01), fontsize=7, ncol=2,
               framealpha=0.9, title="Legend", title_fontsize=7)

    fig.suptitle(
        f"Clinical Summary Heatmap — Top {n_met} Metabolites by Mean Score\n"
        "Row labels: met  [source_cell→sink_cell]  ·  "
        "Patients ordered by group then eGFR",
        fontsize=13, fontweight="bold", y=0.96,
    )

    fp = os.path.join(out_dir, "clinical_summary_heatmap.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    if cfg.verbose:
        print(f"  Clinical summary heatmap → {fp}")
    return fp


# =============================================================================
# Master runner
# =============================================================================

def run_all_clinical_analyses(
    score_matrix: pd.DataFrame,
    cfg: Optional[StatConfig] = None,
    patient_results: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run all clinical analyses and return a dict of result DataFrames.

    Parameters
    ----------
    score_matrix    : pd.DataFrame
        rows = metabolites, columns = patient IDs.
        Build with:  score_matrix = build_score_matrix(patient_results)
    cfg             : StatConfig
    patient_results : dict, optional
        If provided, build_celltype_map() is called to annotate all plots
        with source→sink cell-type context.  If None, ct_map is empty and
        cell-type annotations are omitted gracefully.

    Returns
    -------
    dict with keys:
        egfr, fibrosis, sex, hypertension, diabetes, regression
    """
    if cfg is None:
        cfg = StatConfig()

    # ── Build cell-type map if patient_results supplied (NEW) ────────────────
    ct_map: Dict[str, Tuple[str, str]] = {}
    if patient_results is not None:
        if cfg.verbose:
            print("Building cell-type map from patient results...")
        ct_map = build_celltype_map(patient_results)
        if cfg.verbose:
            print(f"  Cell-type map: {len(ct_map)} metabolites annotated")

    print("=" * 70)
    print("CLINICAL ASSOCIATION ANALYSES")
    print("=" * 70)
    print(f"Patients in matrix : {score_matrix.shape[1]}")
    print(f"Metabolites        : {score_matrix.shape[0]}")
    print(f"min_score filter   : {cfg.min_score}")
    print(f"p threshold        : {cfg.p_thresh}")
    print(f"Cell-type map      : {'yes (' + str(len(ct_map)) + ' entries)' if ct_map else 'not available'}")
    sm_filt = _filter_score_matrix(score_matrix, cfg)
    print(f"Metabolites passing filter: {len(sm_filt)}")
    print()

    out = {}

    print("1. eGFR correlation...")
    out["egfr"]    = analyze_egfr_correlation(score_matrix, cfg, ct_map=ct_map or None)

    print("2. Fibrosis association...")
    out["fibrosis"]= analyze_fibrosis_association(score_matrix, cfg, ct_map=ct_map or None)

    print("3. Sex association...")
    out["sex"]     = analyze_sex_association(score_matrix, cfg, ct_map=ct_map or None)

    print("4. Hypertension / Diabetes associations...")
    comorbidity    = analyze_comorbidity(score_matrix, cfg, ct_map=ct_map or None)
    out.update(comorbidity)

    print("5. Multi-variable regression...")
    out["regression"] = analyze_multivariable_regression(
        score_matrix, cfg, ct_map=ct_map or None)

    print("6. Clinical summary heatmap...")
    plot_clinical_summary_heatmap(score_matrix, cfg, ct_map=ct_map or None)

    # Collect all significant hits into one CSV (with cell-type columns)
    stat_dir  = os.path.join(cfg.base_out_dir, "clinical_statistics")
    sig_rows  = []
    for name, df in out.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "significant" not in df.columns:
            continue
        sig = df[df["significant"]].copy()
        sig["analysis"] = name
        sig_rows.append(sig)

    if sig_rows:
        combined_sig = pd.concat(sig_rows, ignore_index=True)
        sig_path     = os.path.join(stat_dir, "all_significant_hits.csv")
        combined_sig.to_csv(sig_path, index=False)
        print(f"\nAll significant hits → {sig_path}")
        print(f"Total significant associations: {len(combined_sig)}")

    print("\n" + "=" * 70)
    print(f"Clinical analyses complete → {stat_dir}")
    return out


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(
        description="Run clinical statistics for the 14-patient kidney cohort."
    )
    p.add_argument("--out-dir",   default="cohort_output")
    p.add_argument("--min-score", type=float, default=50.0)
    p.add_argument("--p-thresh",  type=float, default=0.05)
    p.add_argument("--top-n",     type=int,   default=20)
    p.add_argument("--quiet",     action="store_true")
    args = p.parse_args()

    from run_cohort_pipeline import load_patient_results
    pr = load_patient_results(args.out_dir)
    if not pr:
        print("ERROR: No results.pkl found. Run run_cohort_pipeline.py first.")
        sys.exit(1)

    sm = build_score_matrix(pr)
    run_all_clinical_analyses(
        sm,
        cfg=StatConfig(
            base_out_dir = args.out_dir,
            min_score    = args.min_score,
            p_thresh     = args.p_thresh,
            top_n        = args.top_n,
            verbose      = not args.quiet,
        ),
        patient_results=pr,   # pass pr so cell-type map is built automatically
    )