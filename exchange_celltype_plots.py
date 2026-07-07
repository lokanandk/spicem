#!/usr/bin/env python3
"""
EXCHANGE FLUX - PER-CELL-TYPE SECRETION/UPTAKE PLOTS
=======================================================

Figures for exchange_celltype_analysis.py. Two families:

1) plot_celltype_secretion_uptake - for ONE cell type: its top-N secreted and
   top-N taken-up metabolites, as grouped bars across clinical groups
   (DKD/HKD/Control, etc.) - "how does this cell type's exchange profile
   shift with disease?"
2) plot_top_metabolites_celltype_heatmap - cohort-wide top-N INDIVIDUAL
   metabolites (not collapsed to one score) by secretion/uptake magnitude,
   broken out across every cell type - "which metabolites dominate exchange,
   and who is doing it?"

Design: matplotlib, dpi=300 PNGs (matches the rest of the exchange pipeline's
plotting.py conventions). Group colours reuse
consensus_exchange_network.GROUP_COLORS ({"Control": green, "HKD": orange,
"DKD": red}) so these figures read as part of the same cohort deliverable as
the WCEG plots. Magnitude (|flux|) uses sequential magma; NaN = grey (not
detected). A recessive footer states sample size / normalisation on every
figure, matching the intracellular-flux plotting module's convention.
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

warnings.filterwarnings("ignore")

try:
    from exchange_celltype_analysis import GROUP_COLORS
except ImportError:
    GROUP_COLORS = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}

_NAN_GREY = "#d9d9d9"
_FALLBACK_COLORS = ["#3a6ea5", "#c1443f", "#7f8c8d", "#8e44ad", "#16a085"]


def _save(fig, out_path: Optional[str]):
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def _footer(fig, text: str):
    fig.text(0.01, -0.02, text, fontsize=7.5, color="#666666", ha="left", va="top")


def _group_color(g: str, i: int) -> str:
    return GROUP_COLORS.get(g, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


# =============================================================================
# 1) Per-cell-type: top-N secretion / uptake across clinical groups
# =============================================================================

def plot_celltype_secretion_uptake(
    matrix_sec: pd.DataFrame,
    matrix_upt: pd.DataFrame,
    cell_type: str,
    group_order: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
):
    """
    Two-panel grouped horizontal bar chart for one cell type: left =
    top secreted metabolites, right = top taken-up metabolites, bars grouped
    by clinical group. matrix_sec / matrix_upt come from
    exchange_celltype_analysis.celltype_metabolite_by_group (metabolite x
    group, values = mean |flux|; NaN = not detected in that group -> no bar).
    """
    if (matrix_sec is None or matrix_sec.empty) and (matrix_upt is None or matrix_upt.empty):
        return None

    groups = list(group_order) if group_order else sorted(
        set((matrix_sec.columns if matrix_sec is not None else [])) |
        set((matrix_upt.columns if matrix_upt is not None else []))
    )
    n_g = max(len(groups), 1)
    bar_h = 0.8 / n_g

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.55 * max(
        len(matrix_sec) if matrix_sec is not None else 0,
        len(matrix_upt) if matrix_upt is not None else 0, 1))))

    for ax, mat, panel_title in [
        (axes[0], matrix_sec, f"Secretion — top {len(matrix_sec) if matrix_sec is not None else 0}"),
        (axes[1], matrix_upt, f"Uptake — top {len(matrix_upt) if matrix_upt is not None else 0}"),
    ]:
        if mat is None or mat.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue
        mets = mat.index.tolist()
        y = np.arange(len(mets))
        for gi, g in enumerate(groups):
            vals = mat[g].values if g in mat.columns else np.full(len(mets), np.nan)
            offset = (gi - (n_g - 1) / 2) * bar_h
            ax.barh(y + offset, np.nan_to_num(vals, nan=0.0), height=bar_h * 0.9,
                   color=_group_color(g, gi), edgecolor="white", linewidth=0.4, label=g)
        ax.set_yticks(y)
        ax.set_yticklabels(mets, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("mean |flux| (biomass units)")
        ax.set_title(panel_title, fontsize=12, weight="bold")
        ax.grid(axis="x", color="#eee", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_group_color(g, i)) for i, g in enumerate(groups)]
    fig.legend(handles, groups, loc="lower center", ncol=len(groups), frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Exchange profile — {cell_type}", fontsize=14, weight="bold")
    _footer(fig, "Bars = mean |flux| per clinical group, pooled across patients/regions. "
                 "A missing bar means that metabolite was not detected in that group.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, out_path)


# =============================================================================
# 2) Cohort-wide: top-N individual metabolites across ALL cell types
# =============================================================================

def plot_top_metabolites_celltype_heatmap(
    matrix_sec: pd.DataFrame,
    matrix_upt: pd.DataFrame,
    out_path: Optional[str] = None,
    title_suffix: str = "",
):
    """
    Two heatmaps side by side (Secretion / Uptake): rows = top-N individual
    metabolites (kept separate, not collapsed into one score), columns =
    cell type. Sequential magma for magnitude; NaN (not detected) = grey.
    """
    if (matrix_sec is None or matrix_sec.empty) and (matrix_upt is None or matrix_upt.empty):
        return None

    n_panels = int(matrix_sec is not None and not matrix_sec.empty) + \
               int(matrix_upt is not None and not matrix_upt.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, max(5, 0.34 * max(
        len(matrix_sec) if matrix_sec is not None else 0,
        len(matrix_upt) if matrix_upt is not None else 0, 1))))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(_NAN_GREY)

    panels = []
    if matrix_sec is not None and not matrix_sec.empty:
        panels.append((matrix_sec, "Secretion"))
    if matrix_upt is not None and not matrix_upt.empty:
        panels.append((matrix_upt, "Uptake"))

    for ax, (mat, label) in zip(axes, panels):
        col_order = mat.notna().sum(axis=0).sort_values(ascending=False).index
        m = mat[col_order]
        data = np.ma.masked_invalid(m.values)
        vmax = float(np.nanpercentile(m.values, 98)) if np.isfinite(m.values).any() else 1.0
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax or 1.0))
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(m.index, fontsize=8)
        ax.set_title(f"{label} — top {len(m)} metabolites{title_suffix}", fontsize=12, weight="bold")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("mean |flux|")

    axes[0].plot([], [], marker="s", linestyle="", color=_NAN_GREY, label="not detected")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.14), frameon=False, fontsize=8)
    _footer(fig, "Rows = individual metabolites ranked by aggregate |flux| across cell types "
                 "(kept separate - not collapsed into one interaction score). "
                 "Columns ordered by detection breadth.")
    fig.tight_layout()
    return _save(fig, out_path)


# =============================================================================
# 3) Differential: (metabolite, cell_type) flux between two clinical groups
# =============================================================================

def _robust_sym_limit(values, pct: float = 98.0) -> float:
    v = values[np.isfinite(values)]
    v = np.abs(v)
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, pct))
    return lim if lim > 0 else 1.0


def plot_differential_celltype_heatmap(
    log2fc_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
    role: str,
    group0: str,
    group1: str,
    alpha: float = 0.05,
    out_path: Optional[str] = None,
):
    """
    (metabolite x cell_type) heatmap of log2 fold-change between two
    clinical groups (diverging, 0-centred: red = higher in group1, blue =
    higher in group0), with a black dot marking cells significant at
    p < alpha (Mann-Whitney). NaN (not enough detections in one group) =
    grey, distinct from the diverging midpoint.
    """
    if log2fc_matrix is None or log2fc_matrix.empty:
        return None

    col_order = log2fc_matrix.notna().sum(axis=0).sort_values(ascending=False).index
    fc = log2fc_matrix[col_order]
    pv = pval_matrix.reindex(index=fc.index, columns=fc.columns) if pval_matrix is not None else None

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(_NAN_GREY)
    lim = _robust_sym_limit(fc.values)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    data = np.ma.masked_invalid(fc.values)
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * fc.shape[1]), max(5, 0.34 * fc.shape[0])))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    if pv is not None:
        sig_y, sig_x = np.where(pv.values < alpha)
        ax.scatter(sig_x, sig_y, marker="*", s=60, color="black", zorder=3,
                  label=f"p < {alpha}")

    ax.set_xticks(range(fc.shape[1]))
    ax.set_xticklabels(fc.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(fc.shape[0]))
    ax.set_yticklabels(fc.index, fontsize=8)
    ax.set_title(f"Differential {role} — {group1} vs {group0}", fontsize=12, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"log2 fold-change ({group1} / {group0})")

    ax.plot([], [], marker="s", linestyle="", color=_NAN_GREY, label="not enough detections")
    ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1.0), frameon=False, fontsize=8)
    n_sig = int((pv.values < alpha).sum()) if pv is not None else 0
    _footer(fig, f"Red = higher in {group1}; blue = higher in {group0}. "
                 f"Mann-Whitney U per (metabolite, cell type); {n_sig} cells significant at p<{alpha} "
                 f"(marked with *). Grey = fewer than the minimum detections in one group.")
    fig.tight_layout()
    return _save(fig, out_path)


if __name__ == "__main__":
    print("Exchange cell-type plotting helpers loaded.")
