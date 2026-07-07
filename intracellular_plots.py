#!/usr/bin/env python3
"""
INTRACELLULAR FLUX - PLOTTING HELPERS
=====================================

Figures for the intracellular-flux analysis (companion to intracellular.py).

Design notes (colour is chosen by the job it does):
- Magnitude / one-signed quantities  -> single-hue SEQUENTIAL (viridis / magma)
- Signed quantities (log2FC, signed flux, z-score) -> two-hue DIVERGING
  (RdBu_r) with the neutral midpoint pinned at 0.
- Missing values (NaN) are rendered as a distinct light GREY so "not observed"
  never reads as the diverging white midpoint.
- No rainbow colormaps; recessive grids/spines; direct labels where they help.

Every function optionally saves a PNG (out_path) and returns the Figure, so the
plots persist to disk like the rest of the pipeline even if inline display is
finicky.
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, PowerNorm

warnings.filterwarnings("ignore")

_NAN_GREY = "#d9d9d9"
_BG_CELL = "#e8e8e8"

# Same hand-curated, perceptually-distinct cell-type palette as
# plotting._make_celltype_palette, copied locally so this module never needs
# plotting.py's heavier deps (networkx/seaborn) just to colour cell types
# consistently with the rest of the pipeline's spatial figures.
_CT_PALETTE_PRIMARY = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
    "#a9a9a9", "#ffffff",
]
_CT_PALETTE_EXTENDED = [
    "#e6beff", "#ff6961", "#77dd77", "#fdfd96", "#84b6f4", "#fdcae1",
    "#b5ead7", "#c7b8ea", "#ffb347", "#779ecb", "#966fd6", "#03c03c",
]
_CT_FULL_PALETTE = _CT_PALETTE_PRIMARY + _CT_PALETTE_EXTENDED


def _celltype_palette(cell_types) -> Dict[str, str]:
    """Deterministic cell_type -> hex colour, alphabetically assigned so the
    same type gets the same colour across every figure (matches
    plotting._make_celltype_palette's convention)."""
    unique_sorted = sorted(set(str(c) for c in cell_types))
    return {ct: _CT_FULL_PALETTE[i % len(_CT_FULL_PALETTE)] for i, ct in enumerate(unique_sorted)}


def _save(fig, out_path: Optional[str]):
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _robust_sym_limit(values: np.ndarray, pct: float = 98.0) -> float:
    """Symmetric colour limit from a robust percentile (ignores NaN)."""
    v = np.abs(values[np.isfinite(values)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, pct))
    return lim if lim > 0 else 1.0


def _footer(fig, text: str):
    """A small recessive caption pinned to the figure bottom - context that
    keeps a figure interpretable without the surrounding notebook prose
    (sample size, normalisation, what grey/colour mean)."""
    fig.text(0.01, -0.02, text, fontsize=7.5, color="#666666", ha="left", va="top")


# =============================================================================
# 8.0 / 8.1  Cell-type phenotype
# =============================================================================

def plot_enrichment_bar(
    enriched_df: pd.DataFrame,
    top_n: int = 20,
    out_path: Optional[str] = None,
    title: str = "Most cell-type-specific reactions",
):
    """
    Horizontal bar of the most cell-type-specific reactions by significance
    (-log10 p). Single sequential hue for magnitude; each bar is directly
    labelled with the cell type that dominates the reaction.
    """
    if enriched_df is None or enriched_df.empty:
        return None
    d = enriched_df.copy()
    d = d.sort_values("P").head(top_n).iloc[::-1]  # most significant on top
    nlp = -np.log10(np.clip(d["P"].values, 1e-300, 1.0))

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(nlp.min()), vmax=float(nlp.max()) if nlp.max() > 0 else 1.0)
    colors = [cmap(norm(v)) for v in nlp]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, nlp, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Reaction"].values, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}\, p$  (Kruskal-Wallis across cell types)")
    ax.set_title(title, fontsize=12, weight="bold")

    for yi, (val, ct) in enumerate(zip(nlp, d["Best_CellType"].values)):
        ax.text(val + nlp.max() * 0.01, yi, str(ct), va="center", fontsize=8, color="#333")

    ax.grid(axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.18)
    n_ct = int(enriched_df["N_CellTypes"].max()) if "N_CellTypes" in enriched_df.columns else None
    _footer(fig, f"Label = cell type with the largest |mean flux| for that reaction. "
                 f"n = {len(enriched_df)} reactions tested" +
                 (f" across up to {n_ct} cell types." if n_ct else "."))
    fig.tight_layout()
    return _save(fig, out_path)


def plot_celltype_reaction_heatmap(
    enriched_df: pd.DataFrame,
    celltype_matrix: pd.DataFrame,
    top_n: int = 30,
    out_path: Optional[str] = None,
    row_zscore: bool = True,
    title: str = "Intracellular flux phenotype",
):
    """
    Heatmap of the top cell-type-enriched reactions (rows) across cell types
    (columns). By default each reaction is z-scored across cell types so
    specialisation is visible even when absolute fluxes are tiny (this is why
    the naive heatmap looked blank). NaN cells are masked to grey.
    """
    if enriched_df is None or enriched_df.empty or celltype_matrix is None or celltype_matrix.empty:
        return None

    top_rxns = enriched_df.sort_values("P").head(top_n)["Reaction"].tolist()
    sub = celltype_matrix.reindex(top_rxns).dropna(how="all")
    if sub.empty:
        return None

    if row_zscore:
        mat = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
        cmap = plt.get_cmap("RdBu_r").copy()
        norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)
        cbar_label = "flux z-score across cell types"
    else:
        mat = sub
        cmap = plt.get_cmap("RdBu_r").copy()
        lim = _robust_sym_limit(mat.values)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cbar_label = "flux (biomass-normalised)"
    cmap.set_bad(_NAN_GREY)

    # Order columns by overall activity so related cell types sit together.
    col_order = sub.notna().sum(axis=0).sort_values(ascending=False).index
    mat = mat[col_order]

    data = np.ma.masked_invalid(mat.values)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * mat.shape[1]),
                                    max(4.5, 0.34 * mat.shape[0])))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title, fontsize=12, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label)

    # grey swatch so readers know grey == not observed
    ax.plot([], [], marker="s", linestyle="", color=_NAN_GREY, label="not observed")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=8)
    _footer(fig, f"Rows = top {len(mat)} cell-type-enriched reactions (Kruskal-Wallis p, ranked). "
                 "Flux is biomass-normalised; z-score is computed per row across cell types.")
    fig.tight_layout()
    return _save(fig, out_path)


# =============================================================================
# 8.2  Mechanistic cross-link (exchange -> intracellular)
# =============================================================================

def plot_mechanistic_links(
    link_df: pd.DataFrame,
    metabolites: Optional[Sequence[str]] = None,
    max_metabolites: int = 6,
    out_path: Optional[str] = None,
):
    """
    For each exchange interaction, a diverging horizontal bar: internal
    reactions that PRODUCE the metabolite in the source cell type (right, warm)
    vs. those that CONSUME it in the sink cell type (left, cool). Makes the
    'why' behind a secretion->uptake coupling legible.
    """
    if link_df is None or link_df.empty:
        return None

    mets = list(metabolites) if metabolites is not None else \
        list(pd.unique(link_df["Metabolite"]))[:max_metabolites]
    mets = mets[:max_metabolites]
    if not mets:
        return None

    n = len(mets)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.2 * n, 2.4)), squeeze=False)
    axes = axes[:, 0]

    prod_color, cons_color = "#c1443f", "#3a6ea5"  # warm produce / cool consume

    for ax, met in zip(axes, mets):
        sub = link_df[link_df["Metabolite"] == met]
        prod = sub[sub["Role"] == "produces_in_source"].sort_values("TotalFlux")
        cons = sub[sub["Role"] == "consumes_in_sink"].sort_values("TotalFlux")

        labels, vals, colors = [], [], []
        for _, r in cons.iterrows():
            labels.append(f"{r['Reaction']}  ({r['CellType']})")
            vals.append(-float(r["TotalFlux"]))
            colors.append(cons_color)
        for _, r in prod.iterrows():
            labels.append(f"{r['Reaction']}  ({r['CellType']})")
            vals.append(float(r["TotalFlux"]))
            colors.append(prod_color)

        y = np.arange(len(vals))
        ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="#888", linewidth=0.8)

        src = sub["Source"].iloc[0] if not sub.empty else "?"
        snk = sub["Sink"].iloc[0] if not sub.empty else "?"
        ax.set_title(f"{met}:  {src}  →  {snk}", fontsize=10, weight="bold", loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#eee", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("← consumed in sink        produced in source →   (total flux)")
    fig.suptitle("Mechanistic basis of exchange interactions", fontsize=12, weight="bold")
    _footer(fig, "Each panel: internal reactions in the SOURCE cell type that produce the "
                 "exchanged metabolite (warm, right) vs. reactions in the SINK cell type that "
                 "consume it (cool, left) - the mechanistic basis for that exchange interaction.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


# =============================================================================
# 8.3  Between-condition comparison
# =============================================================================

def plot_condition_volcano(
    comparison_df: pd.DataFrame,
    cond0: str,
    cond1: str,
    alpha: float = 0.05,
    label_top: int = 12,
    out_path: Optional[str] = None,
    *,
    label_col: str = "Reaction",
):
    """
    Volcano of between-group differences: log2FC (x) vs -log10 p (y).
    Diverging hue for significant up/down; neutral grey for non-significant.
    Top hits are directly labelled. Works for reaction- or subsystem-level
    comparisons - pass label_col='Subsystem' for the latter
    (compare_subsystems_between_conditions output).
    """
    if comparison_df is None or comparison_df.empty:
        return None
    d = comparison_df.copy()
    d["nlp"] = -np.log10(np.clip(d["P"].values, 1e-300, 1.0))
    sig = d["P"] < alpha
    up = sig & (d["Log2FC"] > 0)
    dn = sig & (d["Log2FC"] < 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d.loc[~sig, "Log2FC"], d.loc[~sig, "nlp"], s=12,
               color=_NAN_GREY, alpha=0.7, label="n.s.", edgecolor="none")
    ax.scatter(d.loc[up, "Log2FC"], d.loc[up, "nlp"], s=18,
               color="#c1443f", alpha=0.85, label=f"up in {cond1}", edgecolor="none")
    ax.scatter(d.loc[dn, "Log2FC"], d.loc[dn, "nlp"], s=18,
               color="#3a6ea5", alpha=0.85, label=f"up in {cond0}", edgecolor="none")

    ax.axhline(-np.log10(alpha), color="#888", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#bbb", linewidth=0.8)

    top = d[sig].reindex(d[sig]["nlp"].sort_values(ascending=False).index).head(label_top)
    for _, r in top.iterrows():
        ax.annotate(str(r[label_col]), (r["Log2FC"], r["nlp"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points", color="#333")

    ax.set_xlabel(rf"$\log_2$ fold-change  ({cond1} / {cond0})")
    ax.set_ylabel(r"$-\log_{10}\, p$")
    ax.set_title(f"Differential intracellular {label_col.lower()} activity between groups",
                 fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#f0f0f0", linewidth=0.8)
    ax.set_axisbelow(True)
    n_sig = int(sig.sum())
    _footer(fig, f"{len(d)} {label_col.lower()}s tested, {n_sig} significant at p<{alpha} "
                 f"(dashed line). Mann-Whitney U on biomass-normalised flux, pooled across cell types.")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_condition_top_reactions(
    comparison_df: pd.DataFrame,
    top_n: int = 20,
    out_path: Optional[str] = None,
    title: str = "Top differential reactions",
    *,
    label_col: str = "Reaction",
):
    """
    Horizontal diverging bar of the most significant differential
    reactions/subsystems, coloured by direction of change (log2FC sign).
    Pass label_col='Subsystem' for pathway-level comparisons.
    """
    if comparison_df is None or comparison_df.empty:
        return None
    d = comparison_df.copy().sort_values("P").head(top_n)
    d = d.iloc[::-1]
    vals = d["Log2FC"].values
    colors = ["#c1443f" if v > 0 else "#3a6ea5" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.42 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col].values, fontsize=9)
    ax.set_xlabel(r"$\log_2$ fold-change")
    ax.set_title(title, fontsize=12, weight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, "Red = higher in the second condition; blue = higher in the first "
                 "(see title). Ranked by Mann-Whitney p-value.")
    fig.tight_layout()
    return _save(fig, out_path)


# =============================================================================
# Spatial maps (per-cell intracellular flux in tissue space)
# =============================================================================
#
# Two-panel, annotated design: LEFT = cell-type reference map (same palette/
# legend convention as plotting.plot_spatial_celltype_map, so a reader can
# cross-reference which cell type sits where), RIGHT = the activity map
# itself. The activity panel uses a PowerNorm (gamma<1) rather than a linear
# Normalize: intracellular flux is heavily right-skewed (most cells near
# zero, a handful very high), so a linear scale crushes nearly everything to
# the bottom of the colormap - PowerNorm expands the low/mid range so real
# spatial structure becomes visible instead of a mostly-black plot. Marker
# size also encodes magnitude (redundant coding = easier to read at a
# glance), and the top-activity cells get a black outline ring so hotspots
# are annotated directly on the figure, not just inferable from colour.

def _celltype_reference_panel(ax, all_cells: pd.DataFrame, invert_y: bool = True):
    """Left-panel: every cell coloured by cell_type, with an external legend.
    Mirrors plotting.plot_spatial_celltype_map's conventions (abundant types
    drawn first/underneath, alphabetical legend) at a smaller footprint."""
    if all_cells is None or all_cells.empty or "cell_type" not in all_cells.columns:
        ax.axis("off")
        return
    pts = all_cells.drop_duplicates("barcode")
    palette = _celltype_palette(pts["cell_type"])
    ct_counts = pts["cell_type"].value_counts()

    for ct in ct_counts.index:  # abundant first (background), rare on top
        sub_ct = pts[pts["cell_type"] == ct]
        ax.scatter(sub_ct["px_x"], sub_ct["px_y"], s=14, color=palette[ct],
                   label=str(ct), edgecolor="white", linewidth=0.25,
                   alpha=0.88, zorder=2)

    ax.set_facecolor("#f7f7f7")
    ax.set_title("Cell types", fontsize=11, weight="bold")
    ax.set_xlabel("px_x"); ax.set_ylabel("px_y")
    ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles, labels_leg = ax.get_legend_handles_labels()
    order = sorted(range(len(labels_leg)), key=lambda i: labels_leg[i])
    n_types = len(order)
    fs = 6.5 if n_types > 15 else 7.5
    ax.legend([handles[i] for i in order], [labels_leg[i] for i in order],
              title="Cell type", title_fontsize=fs + 1, fontsize=fs,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              framealpha=0.95, edgecolor="#cccccc")


def _plot_spatial_annotated(
    all_cells: pd.DataFrame,
    value_df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    out_path: Optional[str],
    invert_y: bool = True,
    diverging: bool = False,
    patient_id: Optional[str] = None,
    condition: Optional[str] = None,
    highlight_top_pct: float = 5.0,
    footer_extra: str = "",
    show_celltype_panel: bool = True,
    gamma: float = 0.45,
):
    """Shared two-panel renderer used by every spatial intracellular figure."""
    if value_df is None or value_df.empty:
        return None

    vals = value_df[value_col].astype(float).values

    if diverging:
        lim = _robust_sym_limit(vals)
        cmap, norm = plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        mag = np.abs(vals) / (lim + 1e-12)
    else:
        vmax = float(np.percentile(vals, 98)) if len(vals) else 1.0
        vmax = vmax if vmax > 0 else (float(vals.max()) or 1.0)
        cmap, norm = plt.get_cmap("magma"), PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
        mag = np.clip(vals / (vmax + 1e-12), 0, 1)

    sizes = 14 + 46 * np.clip(mag, 0, 1)

    if show_celltype_panel and all_cells is not None and "cell_type" in (all_cells.columns if all_cells is not None else []):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 6.8))
        _celltype_reference_panel(ax0, all_cells, invert_y=invert_y)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6.8))

    if all_cells is not None and not all_cells.empty:
        bg = all_cells.drop_duplicates("barcode")
        ax1.scatter(bg["px_x"], bg["px_y"], s=6, color=_BG_CELL, alpha=0.35,
                   edgecolor="none", zorder=1)
        ax1.scatter([], [], s=20, color=_BG_CELL, alpha=0.8, label="all cells (context)")

    sc = ax1.scatter(value_df["px_x"], value_df["px_y"], c=vals, s=sizes,
                     cmap=cmap, norm=norm, edgecolor="white", linewidth=0.3, zorder=2)

    # Annotate hotspots directly: ring the top-activity cells (by |value|).
    best_ct = None
    if highlight_top_pct and len(vals):
        rank_vals = np.abs(vals) if diverging else vals
        thresh = np.percentile(rank_vals, 100 - highlight_top_pct)
        hi_mask = rank_vals >= thresh
        if hi_mask.any():
            hi = value_df.loc[hi_mask]
            ax1.scatter(hi["px_x"], hi["px_y"], s=sizes[hi_mask] * 1.7, facecolor="none",
                       edgecolor="black", linewidth=1.0, zorder=3,
                       label=f"top {highlight_top_pct:.0f}%")
        if "cell_type" in value_df.columns:
            means = value_df.groupby("cell_type")[value_col].mean()
            if len(means):
                best_ct = means.abs().idxmax() if diverging else means.idxmax()

    cb = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label)

    ctx_bits = [b for b in (patient_id, condition) if b]
    full_title = title + ("\n" + "  |  ".join(ctx_bits) if ctx_bits else "")
    ax1.set_title(full_title, fontsize=12, weight="bold")
    ax1.set_xlabel("px_x"); ax1.set_ylabel("px_y")
    ax1.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax1.invert_yaxis()
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    footer = f"n = {len(value_df)} cells"
    if best_ct is not None:
        footer += f"; highest mean activity in {best_ct}"
    if footer_extra:
        footer += f". {footer_extra}"
    _footer(fig, footer)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_spatial_reaction(
    frame: pd.DataFrame,
    reaction: str,
    out_path: Optional[str] = None,
    invert_y: bool = True,
    patient_id: Optional[str] = None,
    condition: Optional[str] = None,
    show_celltype_panel: bool = True,
    highlight_top_pct: float = 5.0,
):
    """
    Map a single reaction's per-cell flux in tissue space, with a cell-type
    reference panel alongside. Signed flux -> diverging colormap; one-signed
    -> PowerNorm magma (see _plot_spatial_annotated).
    """
    if frame is None or frame.empty:
        return None
    sub = frame[frame["reaction"] == reaction]
    if sub.empty:
        return None
    signed = (sub["flux"].min() < 0) and (sub["flux"].max() > 0)
    cbar_label = "flux (biomass-normalised, signed)" if signed else "flux (biomass-normalised)"

    return _plot_spatial_annotated(
        all_cells=frame, value_df=sub, value_col="flux",
        title=f"Spatial intracellular flux — {reaction}",
        cbar_label=cbar_label, out_path=out_path, invert_y=invert_y,
        diverging=signed, patient_id=patient_id, condition=condition,
        highlight_top_pct=highlight_top_pct, show_celltype_panel=show_celltype_panel,
        footer_extra="Center-cell attribution: one value per cell from its own "
                     "community-centered solve.",
    )


def plot_spatial_pathway(
    frame: pd.DataFrame,
    reactions: Sequence[str],
    label: str,
    agg: str = "sum_abs",
    out_path: Optional[str] = None,
    invert_y: bool = True,
    patient_id: Optional[str] = None,
    condition: Optional[str] = None,
    show_celltype_panel: bool = True,
    highlight_top_pct: float = 5.0,
):
    """
    Map a *pathway/module* activity per cell (e.g. OXPHOS) by aggregating the
    given reactions within each cell. agg: 'sum_abs' (total throughput, one-
    signed -> PowerNorm magma) or 'sum' (net, signed -> diverging).
    """
    if frame is None or frame.empty:
        return None
    sub = frame[frame["reaction"].isin(list(reactions))].copy()
    if sub.empty:
        return None
    sub["v"] = sub["flux"].abs() if agg == "sum_abs" else sub["flux"]
    per_cell = sub.groupby(["barcode", "px_x", "px_y", "cell_type"], as_index=False)["v"].sum()
    if per_cell.empty:
        return None

    return _plot_spatial_annotated(
        all_cells=frame, value_df=per_cell, value_col="v",
        title=f"Spatial pathway activity — {label}\n({len(reactions)} reactions)",
        cbar_label=f"{label} activity ({agg})", out_path=out_path, invert_y=invert_y,
        diverging=(agg != "sum_abs"), patient_id=patient_id, condition=condition,
        highlight_top_pct=highlight_top_pct, show_celltype_panel=show_celltype_panel,
        footer_extra=f"module = {list(reactions)[:6]}{'...' if len(reactions) > 6 else ''}; "
                     f"agg='{agg}' per cell (center-cell attribution).",
    )


# =============================================================================
# Subsystem / pathway-level figures (real model subsystems)
# =============================================================================

def plot_subsystem_enrichment_dot(
    enriched_df: pd.DataFrame,
    celltype_matrix: pd.DataFrame,
    prevalence_matrix: pd.DataFrame,
    top_n: int = 25,
    out_path: Optional[str] = None,
    title: str = "Cell-type-specific pathway enrichment",
    label_col: str = "Subsystem",
):
    """
    Classic pathway-enrichment dot plot (as in single-cell GO/pathway
    enrichment): rows = subsystem (or metabolite, via label_col), columns =
    cell type.
      - dot COLOUR = z-scored mean activity for that cell type (diverging,
        0-centred) -> "is this cell type unusually active here relative to
        other cell types?"
      - dot SIZE   = prevalence: fraction of that cell type's samples with
        nonzero flux -> distinguishes a strong-but-rare signal from a
        weak-but-near-universal one.
    Requires find_enriched_subsystems/find_enriched_metabolite_activity +
    the matching build_*_celltype_matrix/build_subsystem_prevalence_matrix
    from intracellular.py. Pass label_col="Metabolite" to reuse this for
    metabolite-activity data instead of subsystems.
    """
    if (enriched_df is None or enriched_df.empty or
            celltype_matrix is None or celltype_matrix.empty):
        return None

    top_subs = enriched_df.sort_values("P").head(top_n)[label_col].tolist()
    mat = celltype_matrix.reindex(top_subs).dropna(how="all")
    if mat.empty:
        return None
    prev = prevalence_matrix.reindex(index=mat.index, columns=mat.columns).fillna(0.0)

    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1) + 1e-9, axis=0)
    col_order = mat.notna().sum(axis=0).sort_values(ascending=False).index
    z, prev = z[col_order], prev[col_order]

    n_rows, n_cols = z.shape
    yy, xx = np.meshgrid(range(n_rows), range(n_cols), indexing="ij")
    cvals = z.values.flatten()
    svals = np.nan_to_num(prev.values.flatten(), nan=0.0)
    valid = np.isfinite(cvals)

    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n_cols + 2), max(4.5, 0.32 * n_rows)))
    sc = ax.scatter(xx.flatten()[valid], yy.flatten()[valid], c=cvals[valid],
                    s=20 + 260 * svals[valid], cmap=cmap, norm=norm,
                    edgecolor="#444", linewidth=0.4, zorder=2)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(z.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(z.index, fontsize=8)
    ax.set_xlim(-0.6, n_cols - 0.4); ax.set_ylim(n_rows - 0.4, -0.6)
    ax.grid(color="#f0f0f0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=12, weight="bold")

    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("activity z-score across cell types")

    # size legend (prevalence)
    for frac in (0.25, 0.5, 1.0):
        ax.scatter([], [], s=20 + 260 * frac, color="#888", edgecolor="#444",
                   linewidth=0.4, label=f"{int(frac*100)}% active")
    ax.legend(title="prevalence", loc="upper left", bbox_to_anchor=(1.16, 1.0),
              frameon=False, fontsize=8, title_fontsize=8)

    _footer(fig, f"Rows = top {n_rows} cell-type-enriched subsystems (Kruskal-Wallis p, ranked). "
                 "Colour = per-subsystem z-score of mean |flux| across cell types (diverging, 0 = average). "
                 "Size = fraction of that cell type's community-instances with nonzero flux through the subsystem.")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_subsystem_heatmap(
    enriched_df: pd.DataFrame,
    celltype_matrix: pd.DataFrame,
    top_n: int = 30,
    out_path: Optional[str] = None,
    title: str = "Pathway-level metabolic phenotype",
    label_col: str = "Subsystem",
):
    """Subsystem-level analogue of plot_celltype_reaction_heatmap. Pathway
    aggregation is far less noisy than any single reaction (many alternate
    -optima routings cancel out in the sum), so this is the recommended
    default phenotype figure. Pass label_col="Metabolite" (with
    find_enriched_metabolite_activity / build_metabolite_activity_celltype_
    matrix) to reuse this for metabolite-activity data instead."""
    if (enriched_df is None or enriched_df.empty or
            celltype_matrix is None or celltype_matrix.empty):
        return None

    top_subs = enriched_df.sort_values("P").head(top_n)[label_col].tolist()
    sub = celltype_matrix.reindex(top_subs).dropna(how="all")
    if sub.empty:
        return None

    mat = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(_NAN_GREY)
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5)

    col_order = sub.notna().sum(axis=0).sort_values(ascending=False).index
    mat = mat[col_order]

    data = np.ma.masked_invalid(mat.values)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * mat.shape[1]),
                                    max(4.5, 0.34 * mat.shape[0])))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title, fontsize=12, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("activity z-score across cell types")
    ax.plot([], [], marker="s", linestyle="", color=_NAN_GREY, label="not observed")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=8)
    _footer(fig, f"Rows = top {len(mat)} cell-type-enriched {label_col.lower()}s. Activity = sum of "
                 "|flux| over every reaction, per community member, biomass-normalised.")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_subsystem_signature_bars(
    celltype_matrix: pd.DataFrame,
    cell_types: Optional[Sequence[str]] = None,
    top_n: int = 6,
    n_cols: int = 3,
    out_path: Optional[str] = None,
):
    """
    Small multiples: one panel per cell type, showing its top-N most
    DISTINCTIVE subsystems (highest z-score, i.e. specifically enriched in
    that cell type relative to the others) - a per-cell-type metabolic
    'signature' panel.
    """
    if celltype_matrix is None or celltype_matrix.empty:
        return None
    z = celltype_matrix.sub(celltype_matrix.mean(axis=1), axis=0).div(
        celltype_matrix.std(axis=1) + 1e-9, axis=0)

    cts = list(cell_types) if cell_types is not None else list(celltype_matrix.columns)
    n = len(cts)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), squeeze=False)

    cmap = plt.get_cmap("magma")
    for i, ct in enumerate(cts):
        ax = axes[i // n_cols][i % n_cols]
        if ct not in z.columns:
            ax.axis("off")
            continue
        top = z[ct].dropna().sort_values(ascending=False).head(top_n).iloc[::-1]
        if top.empty:
            ax.axis("off")
            continue
        vmin = min(0.0, float(top.min()))
        vmax = float(top.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(norm(v)) for v in top.values]
        y = np.arange(len(top))
        ax.barh(y, top.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=7.5)
        ax.set_title(str(ct), fontsize=10, weight="bold")
        ax.set_xlabel("z-score", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color="#eee", linewidth=0.7)
        ax.set_axisbelow(True)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("Cell-type metabolic signatures (top distinctive subsystems)",
                 fontsize=13, weight="bold")
    _footer(fig, "Per panel: subsystems ranked by activity z-score computed for that cell type "
                 "relative to all other cell types (higher = more specifically enriched here).")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, out_path)


def plot_spatial_subsystem(
    spatial_frame: pd.DataFrame,
    subsystem: str,
    out_path: Optional[str] = None,
    invert_y: bool = True,
    agg_label: str = "mean",
    patient_id: Optional[str] = None,
    condition: Optional[str] = None,
    show_celltype_panel: bool = True,
    highlight_top_pct: float = 5.0,
):
    """
    Map one subsystem's per-cell activity (from build_spatial_subsystem_frame)
    in tissue space, with a cell-type reference panel alongside. Activity is
    a sum of |flux| (one-signed magnitude) -> PowerNorm magma (see
    _plot_spatial_annotated for why a plain linear scale under-uses contrast
    here).
    """
    if spatial_frame is None or spatial_frame.empty:
        return None
    sub = spatial_frame[spatial_frame["subsystem"] == subsystem]
    if sub.empty:
        return None

    return _plot_spatial_annotated(
        all_cells=spatial_frame, value_df=sub, value_col="value",
        title=f"Spatial pathway activity — {subsystem}",
        cbar_label=f"{subsystem} activity ({agg_label} |flux|, biomass-normalised)",
        out_path=out_path, invert_y=invert_y, diverging=False,
        patient_id=patient_id, condition=condition,
        highlight_top_pct=highlight_top_pct, show_celltype_panel=show_celltype_panel,
        footer_extra=f"value = {agg_label} activity across every community this cell "
                     "participated in (as center or neighbour).",
    )


# =============================================================================
# Differential pathway activity, stratified by cell type (disease context)
# =============================================================================

def plot_subsystem_differential_heatmap(
    log2fc_matrix: pd.DataFrame,
    pval_matrix: pd.DataFrame,
    cond0: str,
    cond1: str,
    alpha: float = 0.05,
    out_path: Optional[str] = None,
    entity_label: str = "pathway",
):
    """
    (entity x cell_type) heatmap of log2 fold-change between two conditions
    (diverging, 0-centred), with a black star marking cells significant at
    p < alpha (Mann-Whitney). This is the cell-type-STRATIFIED disease-
    differential view - it complements plot_subsystem_heatmap (single-
    condition phenotype) and the pooled cell-type-agnostic bar/volcano from
    compare_subsystems_between_conditions by showing WHICH cell type drives
    each pathway's shift between groups. NaN (not enough samples in one
    condition) = grey. Works for either pathway or metabolite differential
    matrices (from build_subsystem_differential_matrices or
    build_metabolite_activity_differential_matrices) - pass
    entity_label="metabolite" for the latter.
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
    ax.set_title(f"Differential {entity_label} activity — {cond1} vs {cond0}", fontsize=12, weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"log2 fold-change ({cond1} / {cond0})")

    ax.plot([], [], marker="s", linestyle="", color=_NAN_GREY, label="not enough samples")
    ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1.0), frameon=False, fontsize=8)
    n_sig = int((pv.values < alpha).sum()) if pv is not None else 0
    _footer(fig, f"Red = higher in {cond1}; blue = higher in {cond0}. Mann-Whitney U per "
                 f"({entity_label}, cell type); {n_sig} cells significant at p<{alpha} (marked with *). "
                 "Activity = sum |flux| per community member, biomass-normalised.")
    fig.tight_layout()
    return _save(fig, out_path)


# =============================================================================
# COMBINED: pathway x metabolite allocation (emergent insight plots)
# =============================================================================

def plot_metabolite_pathway_allocation(
    allocation_table: pd.DataFrame,
    metabolite: str,
    conditions: Sequence[str],
    out_path: Optional[str] = None,
):
    """
    Stacked bar of one metabolite's pathway-flux ALLOCATION (fraction of its
    total turnover contributed by each subsystem), one bar per condition.
    This is the direct visual of the emergent insight: a metabolite's total
    activity can look unchanged while the bar's internal composition
    reorganises completely between conditions - that reorganisation is
    invisible to both plot_subsystem_heatmap (no metabolite axis) and
    plot_metabolite_activity-style views (no pathway axis) alone.

    allocation_table: output of intracellular.compute_pathway_allocation
    (with group_cols=["condition"]), or intracellular.
    build_metabolite_pathway_allocation_table directly (fractions computed
    on the fly per condition in that case).
    """
    if allocation_table is None or allocation_table.empty:
        return None
    d = allocation_table[allocation_table["metabolite"] == metabolite]
    if d.empty:
        return None

    if "fraction" not in d.columns:
        # raw joint table: compute fractions per condition on the fly
        tot = d.groupby("condition")["activity"].transform("sum")
        d = d.assign(fraction=d["activity"] / tot.replace(0, np.nan))
        value_col = "fraction"
    else:
        value_col = "fraction"

    conds = [c for c in conditions if c in d["condition"].unique()]
    subs = sorted(d["subsystem"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {s: cmap(i / max(len(subs) - 1, 1)) for i, s in enumerate(subs)}

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(conds)), 5.5))
    bottoms = np.zeros(len(conds))
    for s in subs:
        vals = np.array([
            float(d.loc[(d["condition"] == c) & (d["subsystem"] == s), value_col].sum())
            for c in conds
        ])
        ax.bar(conds, vals, bottom=bottoms, color=colors[s], edgecolor="white",
              linewidth=0.8, label=s, width=0.6)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.06:
                ax.text(i, b + v / 2, f"{v:.0%}", ha="center", va="center",
                       fontsize=8.5, color="white" if v > 0.12 else "black")
        bottoms += vals

    ax.set_ylabel("fraction of total turnover")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Pathway allocation of {metabolite}", fontsize=12, weight="bold")
    ax.legend(title="Pathway", loc="upper left", bbox_to_anchor=(1.02, 1.0),
             frameon=False, fontsize=8, title_fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _footer(fig, f"Each bar = 100% of {metabolite}'s mean total turnover flux in that condition, "
                 "split by which pathway's reactions produced/consumed it.")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_allocation_shift_ranking(
    shift_df: pd.DataFrame,
    cond0: str,
    cond1: str,
    top_n: int = 15,
    out_path: Optional[str] = None,
):
    """
    Ranked horizontal bar of the metabolites whose pathway ALLOCATION shifts
    most between two conditions (intracellular.
    rank_metabolites_by_allocation_shift's Allocation_Shift score, 0-1 total
    variation distance) - the emergent "most-rerouted metabolites" summary.
    Each bar is annotated with its dominant pathway in each condition, so a
    reader sees not just THAT it rerouted but WHERE from/to.
    """
    if shift_df is None or shift_df.empty:
        return None
    d = shift_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(d))))
    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=0.0, vmax=float(d["Allocation_Shift"].max()) or 1.0)
    colors = [cmap(norm(v)) for v in d["Allocation_Shift"]]
    y = np.arange(len(d))
    ax.barh(y, d["Allocation_Shift"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Metabolite"], fontsize=9)

    top0_col = f"Top_Pathway_{cond0}"
    top1_col = f"Top_Pathway_{cond1}"
    for yi, (_, r) in enumerate(d.iterrows()):
        label = f"{r.get(top0_col, '?')} → {r.get(top1_col, '?')}"
        ax.text(r["Allocation_Shift"] + 0.01, yi, label, va="center", fontsize=7.5, color="#333")

    ax.set_xlabel("allocation shift (total variation distance, 0-1)")
    ax.set_title(f"Most-rerouted metabolites — {cond0} → {cond1}", fontsize=12, weight="bold")
    ax.set_xlim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, "0 = identical pathway usage in both conditions; 1 = completely disjoint "
                 "(all flux moved to different pathways). Annotation = dominant pathway "
                 f"in {cond0} → dominant pathway in {cond1}.")
    fig.tight_layout()
    return _save(fig, out_path)


if __name__ == "__main__":
    print("Intracellular plotting helpers loaded.")
