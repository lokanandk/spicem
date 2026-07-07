#!/usr/bin/env python3
"""
plot_figures.py
===========================
Multi-panel summary figures for the SPICEM kidney cohort.

Generates the following figures, compatible with Nature Methods style:

  plot_cohort_summary_figure()
      4-panel figure combining: folate/one-carbon detection rates (a),
      DKD fibrosis+iron signatures (b), DKD-vs-HKD effect sizes (c),
      multivariate regression R² summary (d).

  plot_signature_detection_bars()
      Grouped bar chart for any set of metabolites and groups.

  plot_effect_size_lollipop()
      Lollipop/dot plot of Cohen's d for a single comparison —
      cleaner alternative to bar charts for many metabolites.

  plot_regression_summary()
      Horizontal dot plot showing regression R² and significant
      predictors for each metabolite model.

  plot_volcano_publication()
      Publication-quality volcano (effect size vs –log10 p) with
      clean per-quadrant colouring and non-overlapping labels.

  plot_detection_dotplot()
      Dot plot of per-patient detection across groups — one row per
      metabolite, one column per patient, dot size = log score.

  plot_clinical_correlation_panel()
      Multi-panel scatter of top clinical associations with
      log-transformed y-axis, Winsorised regression line, and
      patient-ID annotations.

All functions accept pre-loaded DataFrames so they can be called from
both the pipeline and interactively in a Jupyter notebook.

Usage
-----
  from plot_publication_figures import (
      plot_cohort_summary_figure,
      plot_signature_detection_bars,
      plot_effect_size_lollipop,
      plot_regression_summary,
      plot_volcano_publication,
      plot_detection_dotplot,
      plot_clinical_correlation_panel,
  )
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# =============================================================================
# Shared style constants  (Nature Methods / eLife palette)
# =============================================================================

# Group colours — consistent across all figures
GROUP_COLORS: Dict[str, str] = {
    "DKD":      "#C0392B",
    "HKD":      "#E67E22",
    "Control":  "#27AE60",
    "Diseased": "#8E44AD",
}

# Direction colours for effect-size plots
UP_COLOR   = "#C0392B"   # higher in group A
DOWN_COLOR = "#2980B9"   # higher in group B
NS_COLOR   = "#BDC3C7"   # not significant

# Significance star thresholds
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "")]

# Base font size
FS = 9.5


def _sig_stars(p: float) -> str:
    for thresh, stars in SIG_LEVELS:
        if p <= thresh:
            return stars
    return ""


def _apply_nature_style(ax: plt.Axes, grid_axis: str = "x") -> None:
    """Apply clean Nature-paper spine/grid style to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=FS - 1.5, width=0.7, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.22, linestyle=":", linewidth=0.6,
                color="#888888", zorder=0)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    """Add bold panel label (a, b, c …) in the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            color="#111111")


def _stat_annotation(ax: plt.Axes, x: float, y: float, stars: str,
                     fontsize: float = 9.5) -> None:
    if stars:
        ax.text(x, y, stars, ha="center", va="bottom",
                fontsize=fontsize, color="#222222", fontweight="bold")


def _legend(ax: plt.Axes, groups: List[str],
            loc: str = "upper right", ncol: int = 1) -> None:
    handles = [mpatches.Patch(facecolor=GROUP_COLORS.get(g, "#888888"),
                               label=g, linewidth=0)
               for g in groups]
    ax.legend(handles=handles, fontsize=FS - 1, loc=loc,
              frameon=True, framealpha=0.92, edgecolor="none",
              ncol=ncol, handlelength=1.0, handleheight=0.9)


# =============================================================================
# 1.  plot_cohort_summary_figure  (the 4-panel figure from the report)
# =============================================================================

def plot_cohort_summary_figure(
    comp_dkd_ctrl:  pd.DataFrame,
    comp_hkd_ctrl:  pd.DataFrame,
    comp_dkd_hkd:   pd.DataFrame,
    reg_df:         pd.DataFrame,
    out_dir:        str,
    filename:       str = "summary_figure.png",
    dpi:            int = 300,
) -> str:
    """
    4-panel publication summary figure.

    Parameters
    ----------
    comp_dkd_ctrl : comparison_DKD_vs_Control.csv DataFrame
    comp_hkd_ctrl : comparison_HKD_vs_Control.csv DataFrame
    comp_dkd_hkd  : comparison_DKD_vs_HKD.csv DataFrame
    reg_df        : regression_results.csv DataFrame
    out_dir       : output directory
    filename      : output filename

    Returns
    -------
    str  path to saved figure
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(14.5, 11.0), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.52, wspace=0.40,
        left=0.08, right=0.98, top=0.95, bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── a: Folate/one-carbon detection rates ─────────────────────────────────
    _plot_folate_panel(ax_a, comp_dkd_ctrl, comp_hkd_ctrl, comp_dkd_hkd)
    _panel_label(ax_a, "a")

    # ── b: DKD fibrosis + iron signatures ────────────────────────────────────
    _plot_dkd_sig_panel(ax_b, comp_dkd_ctrl, reg_df)
    _panel_label(ax_b, "b")

    # ── c: DKD vs HKD effect sizes ────────────────────────────────────────────
    _plot_effect_size_panel(ax_c, comp_dkd_hkd)
    _panel_label(ax_c, "c")

    # ── d: Regression R² summary ──────────────────────────────────────────────
    _plot_regression_panel(ax_d, reg_df)
    _panel_label(ax_d, "d")

    fp = os.path.join(out_dir, filename)
    fig.savefig(fp, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fp


def _plot_folate_panel(ax, comp_dkd_ctrl, comp_hkd_ctrl, comp_dkd_hkd):
    """Panel a: grouped bars for folate/one-carbon detection rates."""
    folate_mets = [
        ("thf",        "THF\n(C_TAL→MyoFib)"),
        ("5mthf",      "5-mTHF\n(MyoFib→Fib)"),
        ("6thf",       "5-fTHF\n(Fib→PT_S1)"),
        ("6dhf",       "6,7-dhFol\n(CNT→PT_S3)"),
        ("fol",        "Folate\n(PT_S2→Podo)"),
    ]

    groups = [("DKD", comp_dkd_ctrl, "rate_DKD", "rate_Control"),
              ("HKD", comp_hkd_ctrl, "rate_HKD", "rate_Control")]

    n_met = len(folate_mets)
    x     = np.arange(n_met)
    n_grp = 3   # DKD, HKD, Control
    w     = 0.24
    offsets = [-w, 0, w]

    # Gather data
    data = {"DKD": [], "HKD": [], "Control": []}
    for met_id, _ in folate_mets:
        for (grp, df, ra, rb) in groups:
            r = df[df.metabolite == met_id]
            data[grp].append(float(r[ra].iloc[0]) * 100 if not r.empty else 0.0)
        # Control from DKD comparison
        r = comp_dkd_ctrl[comp_dkd_ctrl.metabolite == met_id]
        data["Control"].append(float(r["rate_Control"].iloc[0]) * 100 if not r.empty else 0.0)

    for i, grp in enumerate(["DKD", "HKD", "Control"]):
        col = GROUP_COLORS[grp]
        bars = ax.bar(x + offsets[i], data[grp], w,
                      color=col, alpha=0.88, edgecolor="white",
                      linewidth=0.5, label=grp, zorder=3)

    # Stat annotation: THF DKD vs HKD (perm p=0.005, Fisher p=0.002)
    ax.text(x[0] + offsets[0], data["DKD"][0] + 3, "***",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(x[0] + offsets[1], data["HKD"][0] + 3, "***",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.01, 0.98,
            "Permutation p\u2009<\u20090.05 (DKD vs HKD, Fisher p\u2009=\u20090.002)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            color="#666666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in folate_mets], fontsize=FS - 1.5)
    ax.set_ylabel("Patients with detected exchange (%)", fontsize=FS)
    ax.set_ylim(0, 122)
    ax.set_title("Folate/one-carbon exchange:\nabsent in DKD",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")
    _legend(ax, ["DKD", "HKD", "Control"], loc="upper right")
    _apply_nature_style(ax, "y")


def _plot_dkd_sig_panel(ax, comp_dkd_ctrl, reg_df):
    """Panel b: DKD vs Control detection rates for fibrosis/iron signatures."""
    sigs = [
        ("pro_L",  "pro_L\n(Fib→MyoFib)"),
        ("gthrd",  "gthrd\n(C_TAL→PC)"),
        ("gmp",    "gmp\n(MyoFib→PT_S3)"),
        ("dopa",   "dopa\n(MyoFib→Plasma)"),
        ("fe2",    "Fe\u00b2\u207a/Fe\u00b3\u207a\n(PC→C_TAL)"),
    ]
    n_sig = len(sigs)
    x     = np.arange(n_sig)
    w     = 0.32
    dkd_rates  = []
    ctrl_rates = []
    r2_map = {}

    # Get regression R² per metabolite
    for met, _ in sigs:
        sub = reg_df[reg_df.metabolite == met]
        if not sub.empty:
            r2_map[met] = float(sub["r_squared"].iloc[0])

    for met, _ in sigs:
        r = comp_dkd_ctrl[comp_dkd_ctrl.metabolite == met]
        dkd_rates.append(float(r["rate_DKD"].iloc[0]) * 100 if not r.empty else 0.0)
        ctrl_rates.append(float(r["rate_Control"].iloc[0]) * 100 if not r.empty else 0.0)

    ax.bar(x - w / 2, dkd_rates,  w, color=GROUP_COLORS["DKD"],
           alpha=0.88, edgecolor="white", linewidth=0.5, label="DKD", zorder=3)
    ax.bar(x + w / 2, ctrl_rates, w, color=GROUP_COLORS["Control"],
           alpha=0.88, edgecolor="white", linewidth=0.5, label="Control", zorder=3)

    # R² annotation on relevant bars
    for i, (met, _) in enumerate(sigs):
        if met in r2_map:
            r2 = r2_map[met]
            ypos = max(dkd_rates[i], ctrl_rates[i]) + 3
            ax.text(x[i], ypos, f"R\u00b2={r2:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in sigs], fontsize=FS - 1.5)
    ax.set_ylabel("Patients with detected exchange (%)", fontsize=FS)
    ax.set_ylim(0, 122)
    ax.set_title("DKD fibrosis and\niron-oxidative signatures",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")
    _legend(ax, ["DKD", "Control"], loc="upper right")
    _apply_nature_style(ax, "y")


def _plot_effect_size_panel(ax, comp_dkd_hkd):
    """Panel c: horizontal lollipop chart of Cohen's d for DKD vs HKD."""
    df = comp_dkd_hkd.copy()
    # Take top 18 by |evidence_score| preserving direction
    top_up   = df[df.cohens_d > 0].nlargest(9,  "evidence_score")
    top_down = df[df.cohens_d < 0].nlargest(9,  "evidence_score")
    plot_df  = pd.concat([top_up, top_down]).drop_duplicates("metabolite")
    plot_df  = plot_df.sort_values("cohens_d", ascending=True).reset_index(drop=True)

    y      = np.arange(len(plot_df))
    colors = [UP_COLOR if d > 0 else DOWN_COLOR for d in plot_df.cohens_d]

    # Lollipop stems
    ax.hlines(y, 0, plot_df.cohens_d, colors=colors,
              linewidth=1.4, alpha=0.6, zorder=2)
    # Dots
    ax.scatter(plot_df.cohens_d, y, color=colors, s=52,
               zorder=4, edgecolors="white", linewidths=0.6)

    # Significance stars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        stars = _sig_stars(row.get("perm_p", 1.0))
        if stars:
            xpos  = row.cohens_d
            xoff  = 0.12 if xpos >= 0 else -0.12
            ax.text(xpos + xoff, i, stars, va="center",
                    ha="left" if xpos >= 0 else "right",
                    fontsize=9, fontweight="bold", color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.metabolite, fontsize=FS - 1.5)
    ax.axvline(0, color="#444444", linewidth=0.9, alpha=0.7, zorder=1)
    for dv in (0.5, 0.8, -0.5, -0.8):
        ax.axvline(dv, color="#CCCCCC", linewidth=0.5,
                   linestyle=":", zorder=0, alpha=0.8)

    ax.set_xlabel("Cohen's d  (positive\u2009=\u2009higher in DKD)", fontsize=FS)
    ax.set_title("DKD vs HKD effect sizes\n(* perm p\u2009<\u20090.05)",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")

    leg = [mpatches.Patch(facecolor=UP_COLOR,   label="Higher DKD", linewidth=0),
           mpatches.Patch(facecolor=DOWN_COLOR, label="Higher HKD", linewidth=0)]
    ax.legend(handles=leg, fontsize=FS - 1, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="none")
    _apply_nature_style(ax, "x")


def _plot_regression_panel(ax, reg_df):
    """Panel d: horizontal bar chart of model R² with dominant predictor label."""
    # Best R² per metabolite; dominant = lowest-p significant predictor
    sig_df  = reg_df[reg_df.significant].copy()
    top_r2  = (sig_df.sort_values("r_squared", ascending=False)
                .drop_duplicates("metabolite").head(9))
    dom_p   = (sig_df.sort_values("p_value")
                .drop_duplicates("metabolite")[["metabolite","predictor"]]
                .rename(columns={"predictor":"dom_predictor"}))
    best    = top_r2.merge(dom_p, on="metabolite", how="left")
    best["predictor"] = best["dom_predictor"].fillna(best["predictor"])
    best    = best.sort_values("r_squared", ascending=True).reset_index(drop=True)

    pred_color = {
        "is_DKD":   GROUP_COLORS["DKD"],
        "is_HKD":   GROUP_COLORS["HKD"],
        "eGFR":     "#2980B9",
        "fibrosis": "#7F8C8D",
        "age":      "#95A5A6",
        "sex_num":  "#BDC3C7",
    }

    def _label_and_color(predictor):
        lbl_map = {"is_DKD": "DKD", "is_HKD": "HKD", "eGFR": "eGFR*",
                   "fibrosis": "Fibrosis", "age": "Age",
                   "sex_num": "Sex"}
        return (lbl_map.get(predictor, predictor),
                pred_color.get(predictor, "#888888"))

    y = np.arange(len(best))
    bars = ax.barh(y, best.r_squared, color="#BDC3C7",
                   edgecolor="white", linewidth=0.5, alpha=0.5, zorder=2)

    # Colour segment to highlight significant predictor contribution
    for i, (_, row) in enumerate(best.iterrows()):
        lbl, col = _label_and_color(row.predictor)
        ax.barh(i, row.r_squared, color=col,
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        # Label at end of bar
        ax.text(row.r_squared + 0.008, i, lbl,
                va="center", fontsize=FS - 1.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(best.metabolite, fontsize=FS - 1.5)
    ax.set_xlabel("Model R²  (multivariate OLS)", fontsize=FS)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.6, color="#BBBBBB", linewidth=0.6, linestyle=":", zorder=0)
    ax.axvline(0.7, color="#BBBBBB", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_title("Multivariate regression model fit\n(dominant predictor labelled)",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")

    # Custom legend
    legend_items = [
        mpatches.Patch(facecolor=GROUP_COLORS["DKD"],     label="DKD-specific",  linewidth=0),
        mpatches.Patch(facecolor=GROUP_COLORS["HKD"],     label="HKD-specific",  linewidth=0),
        mpatches.Patch(facecolor="#9B59B6",               label="Both diseases", linewidth=0),
        mpatches.Patch(facecolor="#7F8C8D",               label="Clinical var.", linewidth=0),
    ]
    ax.legend(handles=legend_items, fontsize=FS - 1.5, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="none",
              ncol=2, handlelength=0.9)
    _apply_nature_style(ax, "x")


# =============================================================================
# 2.  plot_signature_detection_bars  (flexible grouped bar)
# =============================================================================

def plot_signature_detection_bars(
    comparisons:    Dict[str, pd.DataFrame],
    metabolites:    List[Tuple[str, str]],
    group_order:    List[str],
    out_path:       str,
    title:          str = "",
    ylabel:         str = "Patients with detected exchange (%)",
    figsize:        Tuple[float, float] = (10, 5),
    dpi:            int = 300,
    stat_pairs:     Optional[List[Tuple[str, str, float, str]]] = None,
) -> str:
    """
    Grouped bar chart of detection rates for a custom set of metabolites.

    Parameters
    ----------
    comparisons  : dict mapping group name -> comparison DataFrame.
                   Each DataFrame must have columns rate_<group> and rate_<ref>.
    metabolites  : list of (metabolite_id, display_label) tuples.
    group_order  : list of group names to plot in order.
    out_path     : full output file path (.png).
    stat_pairs   : optional list of (group_a, group_b, p_value, metabolite_id)
                   tuples to annotate significance brackets.

    Returns
    -------
    str  path to saved figure
    """
    n_met = len(metabolites)
    n_grp = len(group_order)
    width = 0.8 / n_grp
    x     = np.arange(n_met)
    offsets = np.linspace(-(n_grp - 1) / 2 * width,
                           (n_grp - 1) / 2 * width, n_grp)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    for gi, grp in enumerate(group_order):
        col  = GROUP_COLORS.get(grp, "#888888")
        vals = []
        for met_id, _ in metabolites:
            # Find this group's rate from whichever comparison DataFrame has it
            rate_col = f"rate_{grp}"
            found    = 0.0
            for df in comparisons.values():
                if rate_col in df.columns:
                    r = df[df.metabolite == met_id]
                    if not r.empty:
                        found = float(r[rate_col].iloc[0]) * 100
                        break
            vals.append(found)
        ax.bar(x + offsets[gi], vals, width,
               color=col, alpha=0.88, edgecolor="white",
               linewidth=0.5, label=grp, zorder=3)

    # Significance annotations
    if stat_pairs:
        for (ga, gb, pval, met_id) in stat_pairs:
            stars = _sig_stars(pval)
            if not stars:
                continue
            mi = [m for m, _ in metabolites].index(met_id) if met_id in [m for m, _ in metabolites] else -1
            if mi < 0:
                continue
            gi_a = group_order.index(ga) if ga in group_order else -1
            gi_b = group_order.index(gb) if gb in group_order else -1
            if gi_a < 0 or gi_b < 0:
                continue
            xa  = x[mi] + offsets[gi_a]
            xb  = x[mi] + offsets[gi_b]
            yhi = max(
                [float(df[df.metabolite == met_id][f"rate_{g}"].iloc[0]) * 100
                 for g in group_order
                 for df in comparisons.values()
                 if f"rate_{g}" in df.columns and not df[df.metabolite == met_id].empty] or [0]
            ) + 6
            ax.plot([xa, xa, xb, xb], [yhi, yhi + 3, yhi + 3, yhi],
                    lw=0.9, color="#444444")
            ax.text((xa + xb) / 2, yhi + 3.5, stars, ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metabolites], fontsize=FS - 1)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_ylim(0, 125)
    if title:
        ax.set_title(title, fontsize=FS + 1, fontweight="bold", loc="left")
    _legend(ax, group_order, loc="upper right", ncol=min(3, n_grp))
    _apply_nature_style(ax, "y")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# =============================================================================
# 3.  plot_effect_size_lollipop
# =============================================================================

def plot_effect_size_lollipop(
    comp_df:    pd.DataFrame,
    group_a:    str,
    group_b:    str,
    out_path:   str,
    n_top:      int = 20,
    title:      str = "",
    figsize:    Tuple[float, float] = (8, 7),
    dpi:        int = 300,
) -> str:
    """
    Lollipop (stem + dot) plot of Cohen's d for the top n_top metabolites
    by composite evidence score (n_top/2 per direction).

    Cleaner than bar charts for many metabolites: stems show direction clearly,
    dots avoid the false-precision of bar width.
    """
    top_up   = comp_df[comp_df.cohens_d > 0].nlargest(n_top // 2, "evidence_score")
    top_down = comp_df[comp_df.cohens_d < 0].nlargest(n_top // 2, "evidence_score")
    plot_df  = (pd.concat([top_up, top_down])
                .drop_duplicates("metabolite")
                .sort_values("cohens_d", ascending=True)
                .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    y = np.arange(len(plot_df))

    # Colour by significance
    sig = plot_df.get("sig_perm", pd.Series([False] * len(plot_df)))
    col_arr = []
    for d, s in zip(plot_df.cohens_d, sig):
        if bool(s):
            col_arr.append(UP_COLOR if d > 0 else DOWN_COLOR)
        else:
            col_arr.append("#E8A89C" if d > 0 else "#A8C8E8")

    ax.hlines(y, 0, plot_df.cohens_d.values,
              colors=col_arr, linewidth=1.5, alpha=0.75, zorder=2)
    ax.scatter(plot_df.cohens_d, y, color=col_arr, s=58,
               zorder=4, edgecolors="white", linewidths=0.7)

    # Detection rate as secondary dot size channel
    ra_col = f"rate_{group_a}"
    rb_col = f"rate_{group_b}"
    if ra_col in plot_df.columns and rb_col in plot_df.columns:
        det_max = np.maximum(plot_df[ra_col].values, plot_df[rb_col].values)
        ax.scatter(plot_df.cohens_d, y, s=det_max * 120 + 10,
                   color="none", edgecolors=col_arr,
                   linewidths=1.0, alpha=0.50, zorder=3)

    # Stars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        stars = _sig_stars(float(row.get("perm_p", 1.0)))
        if not stars:
            stars = _sig_stars(float(row.get("fisher_p", 1.0)))
        if stars:
            xpos = float(row.cohens_d)
            xoff = 0.13 if xpos >= 0 else -0.13
            ax.text(xpos + xoff, i, stars, va="center",
                    ha="left" if xpos >= 0 else "right",
                    fontsize=9, fontweight="bold", color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.metabolite, fontsize=FS - 1)
    ax.axvline(0, color="#444444", linewidth=0.9, alpha=0.75, zorder=1)
    for dv in (0.5, 0.8, -0.5, -0.8):
        ax.axvline(dv, color="#CCCCCC", linewidth=0.5, linestyle=":", zorder=0)

    xlim = max(abs(plot_df.cohens_d.max()), abs(plot_df.cohens_d.min())) * 1.25
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel(f"Cohen's d  (+ve\u2009=\u2009higher in {group_a}  |\u2009"
                  f"\u2013ve\u2009=\u2009higher in {group_b})", fontsize=FS)
    ax.set_title(title or f"{group_a} vs {group_b} — top exchanges by effect size",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")

    det_legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#999999", markersize=4, label="det. rate 33%"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#999999", markersize=8, label="det. rate 67%"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#999999", markersize=11, label="det. rate 100%"),
        mpatches.Patch(facecolor=UP_COLOR,   label=f"↑ {group_a} (sig.)", linewidth=0),
        mpatches.Patch(facecolor=DOWN_COLOR, label=f"↑ {group_b} (sig.)", linewidth=0),
        mpatches.Patch(facecolor="#E8A89C",  label=f"↑ {group_a} (n.s.)", linewidth=0),
        mpatches.Patch(facecolor="#A8C8E8",  label=f"↑ {group_b} (n.s.)", linewidth=0),
    ]
    ax.legend(handles=det_legend, fontsize=FS - 2, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="none",
              ncol=2, handlelength=1.0)
    _apply_nature_style(ax, "x")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# =============================================================================
# 4.  plot_regression_summary
# =============================================================================

def plot_regression_summary(
    reg_df:    pd.DataFrame,
    out_path:  str,
    top_n:     int = 12,
    figsize:   Tuple[float, float] = (10, 6),
    dpi:       int = 300,
) -> str:
    """
    Dot plot of regression model R² (x) vs metabolite (y).
    Each dot is coloured by the dominant (lowest-p) significant predictor.
    Error-bar-style horizontal line shows the R² range across predictors.
    """
    pred_color = {
        "is_DKD":   GROUP_COLORS["DKD"],
        "is_HKD":   GROUP_COLORS["HKD"],
        "eGFR":     "#2980B9",
        "fibrosis": "#7F8C8D",
        "age":      "#95A5A6",
        "sex_num":  "#BDC3C7",
    }
    pred_label = {
        "is_DKD": "DKD", "is_HKD": "HKD", "eGFR": "eGFR",
        "fibrosis": "Fibrosis", "age": "Age", "sex_num": "Sex",
    }

    # Best R² per metabolite; dominant = lowest-p significant predictor
    sig     = reg_df[reg_df.significant].copy()
    top_r2  = (sig.sort_values("r_squared", ascending=False)
                .drop_duplicates("metabolite").nlargest(top_n, "r_squared"))
    dom_p   = (sig.sort_values("p_value")
                .drop_duplicates("metabolite")[["metabolite","predictor"]]
                .rename(columns={"predictor":"dom_predictor"}))
    best    = top_r2.merge(dom_p, on="metabolite", how="left")
    best["predictor"] = best["dom_predictor"].fillna(best["predictor"])
    best    = best.sort_values("r_squared", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    y = np.arange(len(best))

    for i, (_, row) in enumerate(best.iterrows()):
        col = pred_color.get(row.predictor, "#888888")
        # R² dot
        ax.scatter(row.r_squared, i, color=col, s=70,
                   zorder=4, edgecolors="white", linewidths=0.8)
        # Thin bar from 0 to R²
        ax.hlines(i, 0, row.r_squared, color=col, linewidth=2.5,
                  alpha=0.35, zorder=2)
        # Predictor label
        ax.text(row.r_squared + 0.012, i,
                pred_label.get(row.predictor, row.predictor),
                va="center", fontsize=FS - 1.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(best.metabolite, fontsize=FS - 1)
    ax.set_xlabel("Model R²  (multivariate OLS)", fontsize=FS)
    ax.set_xlim(0, 1.05)
    for v in (0.5, 0.6, 0.7, 0.8):
        ax.axvline(v, color="#DDDDDD", linewidth=0.6, linestyle=":", zorder=0)
    ax.set_title(f"Top {top_n} regression models  (dot colour = dominant predictor)",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")

    leg_items = [
        mpatches.Patch(facecolor=GROUP_COLORS["DKD"], label="DKD",      linewidth=0),
        mpatches.Patch(facecolor=GROUP_COLORS["HKD"], label="HKD",      linewidth=0),
        mpatches.Patch(facecolor="#2980B9",            label="eGFR",     linewidth=0),
        mpatches.Patch(facecolor="#7F8C8D",            label="Fibrosis", linewidth=0),
        mpatches.Patch(facecolor="#95A5A6",            label="Age",      linewidth=0),
    ]
    ax.legend(handles=leg_items, fontsize=FS - 1.5, loc="lower right",
              frameon=True, framealpha=0.92, edgecolor="none",
              ncol=2, handlelength=0.9)
    _apply_nature_style(ax, "x")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# =============================================================================
# 5.  plot_volcano_publication
# =============================================================================

def plot_volcano_publication(
    comp_df:   pd.DataFrame,
    group_a:   str,
    group_b:   str,
    out_path:  str,
    p_thresh:  float = 0.05,
    n_label:   int   = 10,
    figsize:   Tuple[float, float] = (8.5, 7),
    dpi:       int   = 300,
) -> str:
    """
    Publication-quality volcano: Cohen's d vs –log10(perm_p).
    Dot size encodes max detection rate; quadrant shading separates regions.
    Labels placed with adjustText; statistical thresholds annotated.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    d    = comp_df["cohens_d"].values
    logp = -np.log10(comp_df["perm_p"].clip(lower=1e-300).values)

    ra_col = f"rate_{group_a}"
    rb_col = f"rate_{group_b}"
    det    = np.maximum(
        comp_df.get(ra_col, pd.Series(np.zeros(len(comp_df)))).values,
        comp_df.get(rb_col, pd.Series(np.zeros(len(comp_df)))).values,
    )
    sizes = (det * 200 + 12).clip(12, 280)

    sig_p  = comp_df["perm_p"].values < p_thresh
    sig_f  = comp_df.get("fisher_p", pd.Series(np.ones(len(comp_df)))).values < p_thresh
    sig    = sig_p | sig_f

    colors = np.where(
        sig & (d > 0), UP_COLOR,
        np.where(sig & (d < 0), DOWN_COLOR, NS_COLOR)
    )
    alphas = np.where(sig, 0.88, 0.50)

    # Background quadrant tint
    ymax_est = max(logp.max() * 1.15, 2.5)
    xmax_est = max(abs(d).max() * 1.15, 1.0)
    ax.axvspan(0, xmax_est * 1.2,  ymin=0, ymax=1,
               facecolor="#FDECEA", alpha=0.18, zorder=0)
    ax.axvspan(-xmax_est * 1.2, 0, ymin=0, ymax=1,
               facecolor="#EBF5FB", alpha=0.18, zorder=0)

    for xi, yi, ci, si, ai in zip(d, logp, colors, sizes, alphas):
        ax.scatter(xi, yi, c=ci, s=si, alpha=ai,
                   edgecolors="white", linewidths=0.4, zorder=3)

    ax.axhline(-np.log10(p_thresh), color="#555555",
               linestyle="--", linewidth=0.9, alpha=0.7,
               label=f"p\u2009=\u2009{p_thresh}")
    ax.axvline(0,   color="#444444", linewidth=0.8, alpha=0.6, zorder=1)
    for dv in (0.5, 0.8, -0.5, -0.8):
        ax.axvline(dv, color="#CCCCCC", linewidth=0.5, linestyle=":", zorder=0)

    # Labels: top by evidence score, split by direction
    top_up   = comp_df[comp_df.cohens_d > 0].nlargest(n_label // 2, "evidence_score")
    top_down = comp_df[comp_df.cohens_d < 0].nlargest(n_label // 2, "evidence_score")
    to_label = pd.concat([top_up, top_down]).drop_duplicates("metabolite")

    texts = []
    for _, row in to_label.iterrows():
        xi = float(row.cohens_d)
        yi = float(-np.log10(max(float(row.get("perm_p", 1.0)), 1e-300)))
        texts.append(ax.text(
            xi, yi, row.metabolite,
            fontsize=FS - 1.5, fontweight="bold",
            color=UP_COLOR if xi > 0 else DOWN_COLOR,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      alpha=0.80, edgecolor="none"),
            zorder=6,
        ))
    if HAS_ADJUSTTEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                    lw=0.6, alpha=0.7),
                    expand=(1.2, 1.3))

    ax.set_xlim(-xmax_est * 1.15, xmax_est * 1.15)
    ax.set_ylim(bottom=-0.05)
    ax.set_xlabel(f"Cohen's d  (+ve\u2009=\u2009higher in {group_a})", fontsize=FS)
    ax.set_ylabel("\u2013log\u2081\u2080(permutation p)", fontsize=FS)
    n_sig = int(sig.sum())
    ax.set_title(
        f"{group_a} vs {group_b}\u2009|\u2009"
        f"{n_sig} nominally significant (perm or Fisher p\u2009<\u2009{p_thresh})\n"
        f"dot size\u2009=\u2009max detection rate",
        fontsize=FS, fontweight="bold", loc="left",
    )

    # ── Legend placement: emptiest quadrant (data-space, prefer lower half) ──
    ymid_data = float(np.nanpercentile(logp, 40))   # below 40th pctile counts as lower
    q_cnt = {
        "upper left":  int(((d < 0) & (logp > ymid_data)).sum()),
        "upper right": int(((d > 0) & (logp > ymid_data)).sum()),
        "lower left":  int(((d < 0) & (logp <= ymid_data)).sum()),
        "lower right": int(((d > 0) & (logp <= ymid_data)).sum()),
    }
    # Prefer lower; break ties by least populated
    lower_counts = {"lower left": q_cnt["lower left"], "lower right": q_cnt["lower right"]}
    best_loc = min(lower_counts, key=lower_counts.get)

    leg = [
        mpatches.Patch(facecolor=UP_COLOR,   label=f"\u2191 {group_a}  (p<{p_thresh})", linewidth=0),
        mpatches.Patch(facecolor=DOWN_COLOR, label=f"\u2191 {group_b}  (p<{p_thresh})", linewidth=0),
        mpatches.Patch(facecolor=NS_COLOR,   label="Not significant", linewidth=0),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markersize=4,  label="det. rate 33%"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markersize=8,  label="det. rate 67%"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markersize=11, label="det. rate 100%"),
    ]
    ax.legend(handles=leg, fontsize=FS - 1.5, loc=best_loc,
              frameon=True, framealpha=0.94, edgecolor="#DDDDDD",
              ncol=1, columnspacing=0.6, handlelength=1.0,
              borderpad=0.6, labelspacing=0.35)
    _apply_nature_style(ax, "")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# =============================================================================
# 6.  plot_detection_dotplot
# =============================================================================

def plot_detection_dotplot(
    score_matrix:  pd.DataFrame,
    metabolites:   List[str],
    patient_meta:  pd.DataFrame,
    group_order:   List[str],
    out_path:      str,
    title:         str = "",
    figsize:       Tuple[float, float] = (11, 6),
    dpi:           int  = 300,
) -> str:
    """
    Per-patient detection dot plot.

    Each row = one metabolite; each column = one patient.
    Dot colour = disease group; dot size = log10(score + 1).
    Patients sorted by group then eGFR within group.
    Group strips shown as coloured bars above the grid.

    Parameters
    ----------
    score_matrix  : metabolites × patients DataFrame (from build_full_score_matrix)
    metabolites   : metabolite IDs to include (rows)
    patient_meta  : DataFrame indexed by patient_id with columns 'group', 'eGFR'
    group_order   : list of group names for ordering columns
    out_path      : full output path
    """
    # Sort patients by group then eGFR
    order = []
    for g in group_order:
        sub = patient_meta[patient_meta.group == g].sort_values("eGFR",
                                                                  ascending=False)
        order.extend(sub.index.tolist())
    order = [p for p in order if p in score_matrix.columns]
    mets  = [m for m in metabolites if m in score_matrix.index]

    if not order or not mets:
        return out_path

    sm    = score_matrix.loc[mets, order].fillna(0.0)
    log_s = np.log10(sm + 1).values   # shape: n_mets x n_pat

    n_met = len(mets)
    n_pat = len(order)

    fig_h = max(3.5, n_met * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(max(figsize[0], n_pat * 0.65 + 2),
                                     fig_h), facecolor="white")

    # Draw dots
    for mi, met in enumerate(mets):
        for pi, pid in enumerate(order):
            val  = float(log_s[mi, pi])
            grp  = patient_meta.loc[pid, "group"] if pid in patient_meta.index else "?"
            col  = GROUP_COLORS.get(grp, "#888888")
            size = (val * 60 + 4) if val > 0 else 4
            fc   = col if val > 0 else "none"
            ax.scatter(pi, mi, s=size, facecolors=fc, edgecolors=col,
                       linewidths=0.9, alpha=0.88, zorder=3)

    # Group strip above
    strip_y = n_met + 0.3
    prev_g, start = None, 0
    for pi, pid in enumerate(order):
        grp = patient_meta.loc[pid, "group"] if pid in patient_meta.index else "?"
        if grp != prev_g:
            if prev_g is not None:
                ax.barh(strip_y, pi - start, 0.35, left=start - 0.5,
                        color=GROUP_COLORS.get(prev_g, "#888"), alpha=0.85,
                        edgecolor="white", linewidth=0.4, zorder=2)
                ax.text((start + pi) / 2 - 0.5, strip_y, prev_g,
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold", zorder=4)
            start = pi
            prev_g = grp
    # Last group
    ax.barh(strip_y, n_pat - start, 0.35, left=start - 0.5,
            color=GROUP_COLORS.get(prev_g, "#888"), alpha=0.85,
            edgecolor="white", linewidth=0.4, zorder=2)
    ax.text((start + n_pat) / 2 - 0.5, strip_y, prev_g,
            ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold", zorder=4)

    ax.set_yticks(range(n_met))
    ax.set_yticklabels(mets, fontsize=FS - 1)
    ax.set_xticks(range(n_pat))
    ax.set_xticklabels([p[-4:] for p in order], fontsize=7.5,
                        rotation=45, ha="right")
    ax.set_xlim(-0.8, n_pat - 0.2)
    ax.set_ylim(-0.6, n_met + 0.7)
    ax.set_xlabel("Patient ID (last 4 digits)", fontsize=FS)
    ax.set_title(title or "Per-patient exchange detection",
                 fontsize=FS + 0.5, fontweight="bold", loc="left")

    # Size legend
    for lscore, lbl in [(0, "not detected"), (1, "log=1"), (2, "log=2"), (3, "log=3")]:
        ax.scatter([], [], s=lscore * 60 + 4 if lscore > 0 else 4,
                   facecolors="#888888" if lscore > 0 else "none",
                   edgecolors="#888888", linewidths=0.9,
                   label=lbl, alpha=0.85)
    ax.legend(fontsize=FS - 2, loc="lower right", frameon=True,
              framealpha=0.92, edgecolor="none", title="log\u2081\u2080(score+1)",
              title_fontsize=FS - 2)
    _apply_nature_style(ax, "")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# =============================================================================
# 7.  plot_clinical_correlation_panel
# =============================================================================

def plot_clinical_correlation_panel(
    score_matrix:   pd.DataFrame,
    patient_meta:   pd.DataFrame,
    corr_df:        pd.DataFrame,
    clinical_col:   str,
    out_path:       str,
    n_top:          int  = 4,
    title:          str  = "",
    figsize:        Tuple[float, float] = (13, 4.5),
    dpi:            int  = 300,
) -> str:
    """
    n_top-panel scatter of top clinical associations.

    Each panel: x = clinical variable, y = log10(score+1).
    Regression line fitted on Winsorised scores.
    Points coloured by group; labelled with patient ID suffix.
    Both raw and log-score Spearman rho annotated.
    """
    common_pids = [p for p in patient_meta.index if p in score_matrix.columns]
    clin_vals   = patient_meta.loc[common_pids, clinical_col].values.astype(float)
    groups      = patient_meta.loc[common_pids, "group"].values

    top = corr_df.head(n_top)
    if top.empty:
        return out_path

    fig, axes = plt.subplots(1, n_top, figsize=figsize, facecolor="white")
    if n_top == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top.iterrows()):
        met = row["metabolite"]
        if met not in score_matrix.index:
            ax.set_visible(False)
            continue

        raw   = score_matrix.loc[met, common_pids].fillna(0).values.astype(float)
        log_s = np.log10(raw + 1.0)
        colors = [GROUP_COLORS.get(g, "#888") for g in groups]

        ax.scatter(clin_vals, log_s, c=colors, s=65,
                   edgecolors="white", linewidths=0.6, zorder=4, alpha=0.90)

        # Patient labels
        for xi, yi, pid in zip(clin_vals, log_s, common_pids):
            ax.annotate(pid[-4:], (xi, yi),
                        xytext=(3.5, 3), textcoords="offset points",
                        fontsize=6.5, color="#555555", alpha=0.9, zorder=5)

        # Regression on Winsorised log scores
        mask = np.isfinite(clin_vals) & np.isfinite(log_s)
        if mask.sum() >= 3:
            lo = np.nanpercentile(log_s[mask], 1)
            hi = np.nanpercentile(log_s[mask], 99)
            w_s = np.clip(log_s[mask], lo, hi)
            m, b, *_ = stats.linregress(clin_vals[mask], w_s)
            xl = np.linspace(clin_vals[mask].min(), clin_vals[mask].max(), 200)
            ax.plot(xl, m * xl + b, color="#333333",
                    linewidth=1.5, linestyle="--", alpha=0.80, zorder=3)
            # Winsorisation bounds
            for yb in (lo, hi):
                ax.axhline(yb, color="#CCCCCC", linewidth=0.6,
                           linestyle=":", alpha=0.6)

            rho_log, p_log = stats.spearmanr(clin_vals[mask], log_s[mask])
        else:
            rho_log, p_log = np.nan, np.nan

        rho_raw = float(row.get("rho", np.nan))
        p_raw   = float(row.get("p_value", np.nan))
        stars   = _sig_stars(p_log) or _sig_stars(p_raw)

        # Source/sink from corr_df or fallback
        # Source/sink: only show if both exist and are not placeholder strings
        def _clean(v):
            if v is None: return ""
            try:
                if isinstance(v, float) and np.isnan(v): return ""
            except Exception: pass
            s = str(v).strip()
            return "" if s in ("?", "nan", "None", "") else s

        src = _clean(row.get("source") if "source" in row.index else "")
        snk = _clean(row.get("sink")   if "sink"   in row.index else "")
        axis_str = f"({src}\u2192{snk})" if (src and snk) else ""
        title_l1 = f"{met}" + (f"  {axis_str}" if axis_str else "")

        ax.set_xlabel(clinical_col, fontsize=FS)
        ax.set_ylabel("log\u2081\u2080(score\u2009+\u20091)", fontsize=FS)
        p_str   = f"{p_log:.3g}"   if not np.isnan(p_log)   else "n/a"
        rho_str = f"{rho_log:.2f}" if not np.isnan(rho_log) else "n/a"
        ax.set_title(
            f"{title_l1}\n\u03c1 = {rho_str},  p = {p_str}  {stars}",
            fontsize=FS - 0.5, fontweight="bold",
        )
        ymax = np.nanmax(log_s) if np.any(np.isfinite(log_s)) else 1.0
        ax.set_ylim(bottom=-0.05 * ymax, top=ymax * 1.18)
        _apply_nature_style(ax, "")
        ax.grid(alpha=0.18, linestyle=":", linewidth=0.6, color="#AAAAAA")

    # Group legend in last subplot
    handles = [mpatches.Patch(facecolor=GROUP_COLORS.get(g, "#888"),
                               label=g, linewidth=0)
               for g in sorted(set(groups))]
    axes[-1].legend(handles=handles, fontsize=FS - 1.5, loc="upper right",
                    frameon=True, framealpha=0.92, edgecolor="none")

    if title:
        fig.suptitle(title, fontsize=FS + 1, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
