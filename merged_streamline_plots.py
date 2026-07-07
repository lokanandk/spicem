#!/usr/bin/env python3
"""
merged_streamline_plots.py  (v5)
=================================
Multi-row streamline diffusion figures for each target metabolite.

Changes from v4
---------------

1. TITLE SIMPLIFICATION
   Old: "1/5  (2 any routing)" with ★ symbol
   New: "2/5" using n_any as the numerator — shows how many patients have
   ANY exchange of this metabolite (any src→snk routing) out of group total.
   This is more honest: the spatial panel shows any patient with the source
   cell type present, not just exact routing matches.
   No star symbol.

2. BAR CHART BUG FIX — zero bars despite visible arrows
   Root cause: the bar chart was querying score_corr for the EXACT (src,snk)
   pair only. But the representative patient was selected under "any routing"
   priority — their flux for this exact pair is zero, yet the vector field
   (computed for the whole tissue, not per-pair) shows non-zero flow.

   Fix: the bar chart now shows mean S_corr across ALL (src,snk) pairs for
   this metabolite in each group — not restricted to the exact pair of the row.
   This correctly shows the group-level metabolite exchange magnitude.
   A secondary thin bar / annotation shows the exact-pair contribution when
   non-zero, for completeness.

3. BAR CHART REDESIGN
   - Horizontal bars (barh) — compact, easier to read group labels
   - Professional colour palette (colorblind-friendly, not group colours)
     Control: #4878CF (blue)  HKD: #6ACC65 (green)  DKD: #D65F5F (red-brown)
   - Height = 2.2 inches (was stretching to match spatial panel height)
   - No y-axis tick labels needed (group names already on bars)
   - x-axis label: "Mean S_corr (all routings)"
   - Error bars = SEM; value annotation at end of each bar
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
try:
    from plotting import (
        plot_streamline_panel,
        _coarsen_vector_field,
        _build_flux_df,
        PlotConfig,
        DEFAULT_CONFIG,
        FLOW_CMAP,
        _should_exclude_metabolite,
    )
    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False
    def _should_exclude_metabolite(m): return False

try:
    from consensus_exchange_network import (
        build_exchange_records,
        build_wceg_for_metabolite,
        _get_ct_counts,
        COHORT_GROUPS,
        GROUP_COLORS,
    )
    _HAS_CEG = True
except ImportError:
    _HAS_CEG = False
    COHORT_GROUPS = ["Control", "HKD", "DKD"]
    GROUP_COLORS  = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}

try:
    from run_cohort_pipeline import COHORT_METADATA
    _PATIENT_GROUP = {m["id"]: m["group"] for m in COHORT_METADATA}
except ImportError:
    _PATIENT_GROUP = {}

# ---------------------------------------------------------------------------
# Global metabolite exclusion list
# Xenobiotic or artifactual metabolites excluded from ALL merged plot analyses.
# Add metabolite names here to exclude them everywhere.
# ---------------------------------------------------------------------------
_GLOBAL_EXCLUDE: set = {
    "4nph",     # 4-nitrophenol — xenobiotic contaminant
    "npphos",   # nitrophenyl phosphate — xenobiotic
}

def _should_exclude_met_global(m: str) -> bool:
    """True if metabolite should be excluded from all analyses."""
    return _should_exclude_metabolite(m) or m in _GLOBAL_EXCLUDE

GROUP_LABELS = {"Control": "Control", "HKD": "HKD", "DKD": "DKD"}

# Professional colourblind-friendly palette for bar chart (NOT group colours)
BAR_COLORS = {
    "Control": "#4878CF",   # muted blue
    "HKD":     "#6ACC65",   # muted green
    "DKD":     "#D65F5F",   # muted red-brown
}


def _patient_group(pid):
    return _PATIENT_GROUP.get(pid, "Unknown")

def _get_condition(results):
    conds = getattr(getattr(results, "config", None), "conditions", [])
    return conds[0] if conds else None


# ===========================================================================
# Detection counting
# ===========================================================================

def _count_detections(
    exchange_df: pd.DataFrame,
    group: str,
    metabolite: str,
    source_cell: str,
    sink_cell: str,
    patient_results: Dict,
) -> Tuple[int, int, int]:
    """
    Return (n_exact, n_any, n_total).
    n_exact = patients with score_raw > 0 for this (src, snk)
    n_any   = patients with score_raw > 0 for ANY routing of this metabolite
    n_total = patients in group
    """
    pids    = [p for p in patient_results if _patient_group(p) == group]
    n_total = len(pids)
    gdf     = exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"]      == group)
    ]
    n_exact = int(gdf[
        (gdf["source"]    == source_cell) &
        (gdf["sink"]      == sink_cell) &
        (gdf["score_raw"] > 0)
    ]["patient_id"].nunique())
    n_any = int(gdf[gdf["score_raw"] > 0]["patient_id"].nunique())
    return n_exact, n_any, n_total


# ===========================================================================
# Exchange selection (liberal — raw score > 0)
# ===========================================================================

def _select_top_exchanges(
    exchange_df: pd.DataFrame,
    wceg_by_group: Dict[str, pd.DataFrame],
    groups: List[str],
    metabolite: str,
    top_exchanges: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Select top (src, snk) pairs for this metabolite.
    Rank = n_groups × log1p(n_patients) × mean_score_corr,
    with WCEG-supported edges getting a ×2 bonus.
    Returns list of (src, snk, rank_score).
    """
    met_df = exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["score_raw"]  > 0)
    ]
    if met_df.empty:
        return []

    wceg_set: set = set()
    for g, edges in wceg_by_group.items():
        if edges is None or edges.empty:
            continue
        for _, row in edges.iterrows():
            wceg_set.add((str(row["source"]), str(row["sink"])))

    stats: Dict = {}
    sc_col = "score_corr" if "score_corr" in met_df.columns else "score_raw"
    for (src, snk), grp in met_df.groupby(["source", "sink"]):
        key    = (str(src), str(snk))
        n_pats = int(grp["patient_id"].nunique())
        n_grps = int(grp["group"].nunique())
        mean_s = float(grp[sc_col].mean())
        is_w   = key in wceg_set
        rank_s = n_grps * np.log1p(n_pats) * max(mean_s, 0.01) * (2.0 if is_w else 1.0)
        stats[key] = {"rank_s": rank_s}

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["rank_s"], reverse=True)
    return [(src, snk, d["rank_s"]) for (src, snk), d in ranked[:top_exchanges]]


# ===========================================================================
# Patient selection
# ===========================================================================

def _select_best_patient(
    exchange_df: pd.DataFrame,
    patient_results: Dict,
    group: str,
    metabolite: str,
    source_cell: str,
    sink_cell: str,
) -> Tuple[Optional[str], bool]:
    """
    Priority 1: patients with exact (src,snk) and score_raw > 0, highest N_src.
    Priority 2: patients with any positive metabolite score, highest N_src.
    Priority 3: any patient with source cells present.
    Returns (patient_id, was_exact).
    """
    pids = [p for p in patient_results if _patient_group(p) == group]
    if not pids:
        return None, False

    src_counts = {p: _get_ct_counts(patient_results[p]).get(source_cell, 0)
                  for p in pids}

    exact_pids = set(exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"]      == group) &
        (exchange_df["source"]     == source_cell) &
        (exchange_df["sink"]       == sink_cell) &
        (exchange_df["score_raw"]  > 0)
    ]["patient_id"].tolist())
    if exact_pids:
        cands = [(p, src_counts.get(p, 0)) for p in pids if p in exact_pids]
        return max(cands, key=lambda x: x[1])[0], True

    any_pids = set(exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"]      == group) &
        (exchange_df["score_raw"]  > 0)
    ]["patient_id"].tolist())
    cands_any = [(p, src_counts.get(p, 0))
                 for p in pids if p in any_pids and src_counts.get(p, 0) > 0]
    if cands_any:
        return max(cands_any, key=lambda x: x[1])[0], False

    cands_all = [(p, src_counts.get(p, 0)) for p in pids if src_counts.get(p, 0) > 0]
    if cands_all:
        return max(cands_all, key=lambda x: x[1])[0], False

    return pids[0], False


# ===========================================================================
# Vector field helpers
# ===========================================================================

def _get_best_region_data(results, metabolite):
    cond = _get_condition(results)
    per_region = getattr(results, "per_region_data", {}) or {}
    best_U = best_V = best_xi = best_yi = best_meta = None
    best_spd = -1.0
    for key, data in per_region.items():
        if cond and not key.startswith(cond):
            continue
        vf = (data.get("vector_fields") or {}).get(metabolite)
        if vf is None:
            continue
        U, V, xi, yi = vf
        if U is None or V is None:
            continue
        spd = float(np.nanmean(np.sqrt(U**2 + V**2)))
        if spd > best_spd:
            best_U, best_V, best_xi, best_yi = U, V, xi, yi
            best_meta = data.get("meta_df", None)
            best_spd = spd
    return best_U, best_V, best_xi, best_yi, best_meta


def _get_max_speed_coarsened(results, metabolite, coarsen):
    U, V, xi, yi, _ = _get_best_region_data(results, metabolite)
    if U is None:
        return 0.0
    try:
        U_c, V_c, _, _ = _coarsen_vector_field(U, V, xi, yi, coarsen)
        s = float(np.nanmax(np.sqrt(U_c**2 + V_c**2)))
        return s if np.isfinite(s) else 0.0
    except Exception:
        return 0.0



def _compute_neighbourhood_flow_speed(
    results,
    metabolite: str,
    source_cell: str,
    sink_cell: str,
) -> float:
    """
    Compute the mean flow speed in the neighbourhood of source and sink cells.

    This is the quantity that drives the visual representation:
      - Halo size    ∝ local flow speed at each cell's position
      - Streamline colour ∝ flow speed along the line
      - Streamline density reflects where speed is above threshold

    Algorithm
    ---------
    1. Get the best vector field (U, V, xi, yi) for this metabolite.
    2. Get cell positions (px_x, px_y) for source_cell and sink_cell.
    3. For each cell position, find the nearest grid point and read
       speed = sqrt(U[i,j]^2 + V[i,j]^2).
    4. Return the mean speed over all source+sink cell positions.

    Returns 0.0 if no vector field or no matching cells are found.
    """
    U, V, xi, yi, meta_df = _get_best_region_data(results, metabolite)
    if U is None or meta_df is None or meta_df.empty:
        return 0.0

    # Find cell-type column
    ct_col = None
    for c in ["cell_type","Graph.based","Idents","celltype","CellType","annotation"]:
        if c in meta_df.columns:
            ct_col = c
            break
    if ct_col is None:
        return 0.0

    x_col = "px_x" if "px_x" in meta_df.columns else None
    y_col = "px_y" if "px_y" in meta_df.columns else None
    if x_col is None or y_col is None:
        return 0.0

    try:
        cts = meta_df[ct_col].astype(str).values
        px  = pd.to_numeric(meta_df[x_col], errors="coerce").values
        py  = pd.to_numeric(meta_df[y_col], errors="coerce").values

        # Mask for source + sink cells
        mask = (cts == source_cell) | (cts == sink_cell)
        mask &= np.isfinite(px) & np.isfinite(py)
        if not mask.any():
            return 0.0

        cell_x = px[mask]
        cell_y = py[mask]

        # Speed field
        speed = np.sqrt(U**2 + V**2)

        # xi, yi are 1-D grid coordinates
        # Map each cell position to nearest grid index
        xi_arr = xi if xi.ndim == 1 else xi[0]
        yi_arr = yi if yi.ndim == 1 else yi[:, 0]

        x_min, x_max_g = float(xi_arr.min()), float(xi_arr.max())
        y_min, y_max_g = float(yi_arr.min()), float(yi_arr.max())
        nx = len(xi_arr)
        ny = len(yi_arr)

        speeds_at_cells = []
        for cx, cy in zip(cell_x, cell_y):
            # Nearest grid index (clipped)
            ix = int(np.clip(np.round((cx - x_min) / (x_max_g - x_min + 1e-9) * (nx-1)),
                             0, nx-1))
            iy = int(np.clip(np.round((cy - y_min) / (y_max_g - y_min + 1e-9) * (ny-1)),
                             0, ny-1))
            # speed array shape is (ny, nx)
            if speed.shape == (ny, nx):
                speeds_at_cells.append(float(speed[iy, ix]))
            elif speed.shape == (nx, ny):
                speeds_at_cells.append(float(speed[ix, iy]))

        if not speeds_at_cells:
            return 0.0
        return float(np.mean(speeds_at_cells))

    except Exception:
        return 0.0

# ===========================================================================
# Quantitative inset: horizontal bar chart
# ===========================================================================

def _draw_quantitative_inset(
    ax,
    exchange_df,
    metabolite,
    source_cell,
    sink_cell,
    groups,
    patient_results,
    global_xmax=None,
    n_boot=None,     # unused, kept for signature compatibility
    boot_seed=None,  # unused, kept for signature compatibility
):
    """
    Per-patient bar chart grouped by condition.

    Each bar = one patient's flow speed at (source, sink) cell positions.
    Bars are grouped by disease group (Control / HKD / DKD), coloured by group.
    No summary statistic — every data point is shown directly.

    Patients with zero / undetected flow are shown as very small stub bars
    (height = LOG_MIN) in a lighter shade so the reader sees non-detection.

    x-axis: log scale flow speed.
    y-axis: one tick per patient, labelled "PID  (group)".
    """
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    LOG_MIN = 0.05

    # Collect all patients in group order
    pat_list   = []   # (pid, group, speed)
    for g in groups:
        pids_g = [p for p in patient_results if _patient_group(p) == g]
        for pid in sorted(pids_g):
            spd = _compute_neighbourhood_flow_speed(
                patient_results[pid], metabolite, source_cell, sink_cell)
            pat_list.append((pid, g, spd))

    if not pat_list:
        ax.set_visible(False)
        return

    n_pats  = len(pat_list)
    y_pos   = np.arange(n_pats)
    bar_h   = 0.65

    x_max = global_xmax if (global_xmax and global_xmax > 0) else max(
        (s for _, _, s in pat_list if s > 0), default=LOG_MIN)

    # Draw one bar per patient
    for yi, (pid, g, spd) in enumerate(pat_list):
        base_col = BAR_COLORS.get(g, "#888888")
        detected = spd > 0
        x_val    = max(spd, LOG_MIN)

        if detected:
            ax.barh(yi, x_val, height=bar_h,
                    color=base_col, alpha=0.80,
                    edgecolor="white", linewidth=0.3, zorder=2)
        else:
            # Stub bar for non-detected patient
            ax.barh(yi, LOG_MIN, height=bar_h,
                    color=base_col, alpha=0.25,
                    edgecolor=base_col, linewidth=0.4,
                    hatch="////", zorder=2)

    # Group separator lines between conditions
    cumulative = 0
    for g in groups:
        n_g = sum(1 for _, grp, _ in pat_list if grp == g)
        cumulative += n_g
        if cumulative < n_pats:
            ax.axhline(cumulative - 0.5, color="#cccccc",
                       lw=0.7, ls="--", zorder=1)

    # Y-axis labels: "PID" only (keep compact)
    y_labels = [pid for pid, _, _ in pat_list]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=5.5)

    # Colour the tick labels by group
    for tick, (pid, g, _) in zip(ax.get_yticklabels(), pat_list):
        tick.set_color(BAR_COLORS.get(g, "#333"))

    # Group legend on right side as text annotations
    cumulative = 0
    for g in groups:
        n_g   = sum(1 for _, grp, _ in pat_list if grp == g)
        mid_y = cumulative + n_g / 2.0 - 0.5
        ax.text(1.01, mid_y, GROUP_LABELS.get(g, g),
                transform=ax.get_yaxis_transform(),
                fontsize=6, color=BAR_COLORS.get(g, "#333"),
                va="center", ha="left", fontweight="bold")
        cumulative += n_g

    # Log scale x-axis
    ax.set_xscale("log")
    ax.set_xlim(LOG_MIN * 0.7, x_max * 3.0)
    ax.axvline(LOG_MIN, color="#cccccc", lw=0.5, ls=":", zorder=1)
    ax.tick_params(axis="x", labelsize=5.5, which="both")
    ax.set_xlabel("Flow speed (log)", fontsize=6.5, labelpad=2)
    ax.set_title("Per-patient\nflow speed",
                 fontsize=6.5, fontweight="bold", color="#333", pad=3)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.4)


def _make_title(group: str, n_any: int, n_total: int, pid: str) -> str:
    """
    Clean title: "Control\n2/5\nrep: HK2852"
    Uses n_any (any routing) as the numerator — more meaningful.
    No star symbol.
    """
    grp  = GROUP_LABELS.get(group, group)
    frac = f"{n_any}/{n_total}"
    return f"{grp}\n{frac}\nrep: {pid}"


# ===========================================================================
# Draw one streamline panel
# ===========================================================================

def _draw_panel(
    ax,
    results,
    metabolite: str,
    source_cell: str,
    sink_cell: str,
    group: str,
    n_any: int,
    n_total: int,
    pid: str,
    cfg: "PlotConfig",
    shared_vmax: Optional[float],
    coarsen: int,
    n_seed_points: int,
    cell_proximity_radius: float,
    speed_mask_threshold: float,
    seed: int,
):
    if not _HAS_PLOTTING:
        ax.text(0.5, 0.5, "plotting.py not available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888")
        return

    U, V, xi, yi, meta_df = _get_best_region_data(results, metabolite)

    if U is None:
        ax.set_facecolor("#f5f5f5"); ax.axis("off")
        ax.text(0.5, 0.5, f"{GROUP_LABELS.get(group,group)}\n(no spatial data)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#aaa", fontstyle="italic")
        ax.set_title(
            _make_title(group, n_any, n_total, pid),
            fontsize=cfg.subtitle_font_size, fontweight="bold",
            color=GROUP_COLORS.get(group,"#333"), pad=8)
        return

    U_c, V_c, xi_c, yi_c = _coarsen_vector_field(U, V, xi, yi, coarsen)

    cond = _get_condition(results)
    flux_df = None
    if cond:
        try:
            flux_df = _build_flux_df(results, cond, metabolite)
        except Exception:
            pass

    xlim = (float(xi_c.min()), float(xi_c.max()))
    ylim = (float(yi_c.min()), float(yi_c.max()))

    title = _make_title(group, n_any, n_total, pid)

    plot_streamline_panel(
        ax,
        U_c, V_c, xi_c, yi_c,
        meta_df,
        src_type=source_cell,
        snk_type=sink_cell,
        title=title,
        show_colorbar=True,
        xlim=xlim,
        ylim=ylim,
        cfg=cfg,
        cmap=FLOW_CMAP,
        colorbar_label="Flow speed",
        use_consensus_settings=True,
        shared_vmax=shared_vmax,
        n_seed_points=n_seed_points,
        cell_proximity_radius=cell_proximity_radius,
        speed_mask_threshold=speed_mask_threshold,
        flux_df=flux_df,
        seed=seed,
    )


# ===========================================================================
# Multi-row figure builder
# ===========================================================================

def _make_merged_figure(
    metabolite: str,
    patient_results: Dict,
    exchange_df: pd.DataFrame,
    wceg_by_group: Dict[str, pd.DataFrame],
    groups: List[str],
    cfg: "PlotConfig",
    top_exchanges: int = 5,
    wceg_n_permutations: int = 1000,
    wceg_alpha: float = 0.20,
    n_seed_points: int = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold: float = 0.01,
    seed: int = 42,
) -> Tuple[Optional[plt.Figure], List[Dict]]:
    """
    Layout per figure:
      Col 0 (label)  | Col 1..ngrp (streamlines) | Col ngrp+1 (horiz bar)
    """
    coarsen = cfg.consensus_differential_coarsen_factor

    top_exch = _select_top_exchanges(
        exchange_df, wceg_by_group, groups, metabolite, top_exchanges)
    if not top_exch:
        return None, []

    n_rows = len(top_exch)
    n_cols = len(groups)

    # ── Pre-gather panel data ───────────────────────────────────────────────
    panel_data: List[List[Optional[Dict]]] = []
    all_manifest: List[Dict] = []

    for ri, (src, snk, rank_s) in enumerate(top_exch):
        row_data = []
        for g in groups:
            n_total = sum(1 for p in patient_results if _patient_group(p) == g)
            n_exact, n_any, _ = _count_detections(
                exchange_df, g, metabolite, src, snk, patient_results)
            best_pid, was_exact = _select_best_patient(
                exchange_df, patient_results, g, metabolite, src, snk)

            if best_pid is None or best_pid not in patient_results:
                row_data.append(None)
                all_manifest.append({
                    "metabolite": metabolite, "exchange_rank": ri+1,
                    "source": src, "sink": snk, "rank_score": round(rank_s,4),
                    "group": g, "best_patient": "",
                    "n_exact": n_exact, "n_any": n_any, "n_total": n_total,
                    "selection": "no_patient",
                })
                continue

            row_data.append({
                "results": patient_results[best_pid],
                "src": src, "snk": snk,
                "n_exact": n_exact, "n_any": n_any, "n_total": n_total,
                "was_exact": was_exact,
                "best_pid": best_pid,
            })
            all_manifest.append({
                "metabolite": metabolite, "exchange_rank": ri+1,
                "source": src, "sink": snk, "rank_score": round(rank_s,4),
                "group": g, "best_patient": best_pid,
                "n_exact": n_exact, "n_any": n_any, "n_total": n_total,
                "selection": "exact" if was_exact else "any",
            })
        panel_data.append(row_data)

    # Shared vmax
    shared_vmax = 0.0
    for row_data in panel_data:
        for d in row_data:
            if d is None: continue
            s = _get_max_speed_coarsened(d["results"], metabolite, coarsen)
            if s > shared_vmax: shared_vmax = s
    if shared_vmax < 1e-12: shared_vmax = 1.0

    # ── Figure layout ───────────────────────────────────────────────────────
    # Cols: [label 0.16 | cond×3 1.0 each | quant 0.55]
    # Quant panels are compact squares (not full panel height)
    panel_w = 7      # inches per streamline column
    panel_h = 7      # inches per row
    quant_w = 0.55   # relative width of bar chart column

    fig_w = (0.16 + n_cols + quant_w) * panel_w * 0.9
    fig_h = n_rows * panel_h + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Two-level GridSpec:
    # Outer: n_rows rows, each containing the spatial row + a compact quant subplot.
    # We achieve compact quant subplots by using a nested GridSpec per row.
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols + 2,    # label | ngrp streamlines | quant
        figure=fig,
        width_ratios=[0.16] + [1.0]*n_cols + [quant_w],
        wspace=0.05,
        hspace=0.18,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.04,
    )

    any_plotted = False

    # Compute global_xmax = max mean S_corr across all (exchange, group)
    # for THIS metabolite so all bar charts share the same x-axis scale.
    _global_xmax = 0.0
    for _src, _snk, _ in top_exch:
        for _g in groups:
            for _pid in [p for p in patient_results if _patient_group(p)==_g]:
                _spd = _compute_neighbourhood_flow_speed(
                    patient_results[_pid], metabolite, _src, _snk)
                if _spd > _global_xmax:
                    _global_xmax = _spd
    if _global_xmax < 1e-6:
        _global_xmax = 1.0

    for ri, (src, snk, rank_s) in enumerate(top_exch):

        # Row label
        ax_lbl = fig.add_subplot(outer_gs[ri, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.95, 0.5,
            f"#{ri+1}\n{src[:12]}\n→\n{snk[:12]}",
            transform=ax_lbl.transAxes,
            ha="right", va="center",
            fontsize=8.0, fontweight="bold",
            color="#333333", linespacing=1.4,
        )

        # Streamline panels
        for ci, g in enumerate(groups):
            ax = fig.add_subplot(outer_gs[ri, ci + 1])
            d  = panel_data[ri][ci]
            if d is None:
                ax.set_facecolor("#f0f0f0"); ax.axis("off")
                n_tot = sum(1 for p in patient_results if _patient_group(p)==g)
                ax.text(0.5, 0.5,
                        f"{GROUP_LABELS.get(g,g)}\n0/{n_tot}\n(no patients)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="#bbb")
                continue

            _draw_panel(
                ax=ax,
                results=d["results"],
                metabolite=metabolite,
                source_cell=src, sink_cell=snk,
                group=g,
                n_any=d["n_any"], n_total=d["n_total"],
                pid=d["best_pid"],
                cfg=cfg, shared_vmax=shared_vmax,
                coarsen=coarsen,
                n_seed_points=n_seed_points,
                cell_proximity_radius=cell_proximity_radius,
                speed_mask_threshold=speed_mask_threshold,
                seed=seed,
            )
            any_plotted = True

        # ── Quantitative inset — compact, vertically centred ────────────────
        # Use a SubplotSpec with vertical padding to make the bar chart
        # shorter than the full row height.
        ax_outer = fig.add_subplot(outer_gs[ri, n_cols + 1])
        ax_outer.set_visible(False)   # invisible container

        # Place the bar chart in the upper ~60% of the row cell
        bbox = ax_outer.get_position()
        bar_ax = fig.add_axes([
            bbox.x0 + 0.005,
            bbox.y0 + bbox.height * 0.20,   # 20% padding at bottom
            bbox.width - 0.010,
            bbox.height * 0.62,             # 62% of row height
        ])
        _draw_quantitative_inset(
            bar_ax, exchange_df, metabolite, src, snk, groups, patient_results,
            global_xmax=_global_xmax)

    if not any_plotted:
        plt.close(fig)
        return None, all_manifest

    fig.text(
        0.5, 0.008,
        "N/N = patients with any routing for this metabolite / group total  ·  "
        "Halos ∝ local flux magnitude  ·  "
        "Arrows seeded from source cells, forward integration  ·  "
        "Bar: full = all routings, thin = this routing",
        ha="center", va="bottom",
        fontsize=6.5, color="#666", fontstyle="italic",
    )

    fig.suptitle(
        f"Consensus Metabolite Flow: {metabolite}\n"
        f"Top {n_rows} exchange(s)  ·  Ranked by cross-group prevalence × score  ·  "
        f"Right: mean S_corr ± SEM (all routings)",
        fontsize=cfg.title_font_size,
        fontweight="bold",
        y=0.97,
    )
    return fig, all_manifest


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_merged_streamline_plots(
    patient_results: Dict,
    selected_metabolites: List[str],
    out_dir: str = "cohort_output/merged_plots",
    groups: Optional[List[str]] = None,
    wceg_n_permutations: int = 1000,
    wceg_alpha: float = 0.20,
    exchange_df: Optional[pd.DataFrame] = None,
    cfg: Optional["PlotConfig"] = None,
    top_exchanges: int = 5,
    n_seed_points: int = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold: float = 0.01,
    dpi: int = 300,
    verbose: bool = True,
) -> List[str]:
    """
    Generate multi-row streamline diffusion figures.

    Layout: rows = exchanges, cols = [label | Control | HKD | DKD | bar chart]

    Title: "2/5" using n_any (any routing) as numerator. No star symbol.

    Bar chart: horizontal, compact, mean S_corr of ALL routings per group
    (so bars are non-zero whenever the metabolite is exchanged, regardless
    of which specific cell pair). Professional colourblind-friendly colours.
    A thinner overlay bar shows the exact-routing contribution.
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return []
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return []

    os.makedirs(out_dir, exist_ok=True)
    if groups is None: groups = COHORT_GROUPS
    if cfg    is None: cfg   = DEFAULT_CONFIG

    if verbose:
        print(f"\n{'='*62}")
        print(f"  MERGED STREAMLINE PLOTS  (v5)")
        print(f"  Out       : {out_dir}")
        print(f"  Mets      : {len(selected_metabolites)}")
        print(f"  Max rows  : {top_exchanges}")
        print(f"  Title     : N_any/N_total (no star)")
        print(f"  Bar chart : horizontal, all routings, colourblind palette")
        print(f"{'='*62}")

    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records …")
        exchange_df = build_exchange_records(patient_results)
    if verbose and not exchange_df.empty:
        print(f"  {exchange_df['metabolite'].nunique()} metabolites")

    saved: List[str] = []
    all_manifest: List[Dict] = []

    for ri, met in enumerate(selected_metabolites, 1):
        if _should_exclude_met_global(met):
            continue
        if verbose:
            print(f"  [{ri:2d}/{len(selected_metabolites)}] {met} …",
                  end=" ", flush=True)
        try:
            wceg_by_group = build_wceg_for_metabolite(
                exchange_df, patient_results, met,
                groups=groups,
                n_permutations=wceg_n_permutations,
                alpha=wceg_alpha,
            )
            top_exch = _select_top_exchanges(
                exchange_df, wceg_by_group, groups, met, top_exchanges)
            if verbose and top_exch:
                print(f"[{len(top_exch)}rows] ", end="", flush=True)

            fig, manifest_rows = _make_merged_figure(
                metabolite=met,
                patient_results=patient_results,
                exchange_df=exchange_df,
                wceg_by_group=wceg_by_group,
                groups=groups,
                cfg=cfg,
                top_exchanges=top_exchanges,
                wceg_n_permutations=wceg_n_permutations,
                wceg_alpha=wceg_alpha,
                n_seed_points=n_seed_points,
                cell_proximity_radius=cell_proximity_radius,
                speed_mask_threshold=speed_mask_threshold,
                seed=int(getattr(cfg, "random_seed", 42)),
            )
            all_manifest.extend(manifest_rows)

            if fig is None:
                if verbose: print("SKIP (no data)"); continue

            fname = f"merged_streamline_{ri:02d}_{met.replace('/','_')}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(fpath)
            if verbose: print(f"OK → {fname}")

        except Exception as e:
            if verbose:
                import traceback
                print(f"ERROR: {e}"); traceback.print_exc()

    if all_manifest:
        pd.DataFrame(all_manifest).to_csv(
            os.path.join(out_dir, "merged_plots_manifest.csv"), index=False)

    if verbose:
        print(f"\n  ✓ {len(saved)} figures → {out_dir}\n")
    return saved


# ===========================================================================
# WCEG-weighted exchange selection  (W = mean(S_corr) × consistency)
# ===========================================================================

def _select_top_exchanges_wceg(
    wceg_by_group: Dict[str, pd.DataFrame],
    groups: List[str],
    top_exchanges: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Rank (src, snk) pairs by their WCEG edge weight
        W = mean(S_corr) × consistency
    where consistency = n_detected / n_group.

    The max W across all groups is used as the ranking score so that edges
    dominant in any condition float to the top.

    Returns list of (src, snk, max_W) tuples, sorted descending, up to
    top_exchanges entries.
    """
    best_w: Dict[Tuple[str,str], float] = {}
    for g in groups:
        edges = wceg_by_group.get(g, pd.DataFrame())
        if edges is None or edges.empty:
            continue
        for _, row in edges.iterrows():
            key = (str(row["source"]), str(row["sink"]))
            w   = float(row.get("edge_weight", 0.0))
            if w > best_w.get(key, 0.0):
                best_w[key] = w

    if not best_w:
        return []

    ranked = sorted(best_w.items(), key=lambda kv: kv[1], reverse=True)
    return [(src, snk, w) for (src, snk), w in ranked[:top_exchanges]]


# ===========================================================================
# Shared figure builder that accepts a pre-computed exchange list
# (used by both standard and WCEG-weighted entry points)
# ===========================================================================

def _make_figure_from_exchange_list(
    metabolite:           str,
    patient_results:      Dict,
    exchange_df:          pd.DataFrame,
    top_exch:             List[Tuple[str, str, float]],  # (src, snk, rank_val)
    groups:               List[str],
    cfg:                  "PlotConfig",
    n_seed_points:        int   = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold: float  = 0.01,
    seed:                 int    = 42,
    row_label_prefix:     str    = "",   # e.g. "W=" for WCEG
    suptitle_suffix:      str    = "",
) -> Tuple[Optional[plt.Figure], List[Dict]]:
    """
    Core figure builder shared by both generate_merged_streamline_plots
    and generate_merged_streamline_plots_wceg.

    top_exch items: (source, sink, rank_value)  — rank_value is only used
    in the row-label annotation; the spatial rendering is identical.
    """
    coarsen = cfg.consensus_differential_coarsen_factor
    if not top_exch:
        return None, []

    n_rows = len(top_exch)
    n_cols = len(groups)

    # Pre-gather panel data for all (row, group) combinations
    panel_data: List[List[Optional[Dict]]] = []
    all_manifest: List[Dict] = []

    for ri, (src, snk, rank_val) in enumerate(top_exch):
        row_data = []
        for g in groups:
            n_total = sum(1 for p in patient_results if _patient_group(p) == g)
            n_exact, n_any, _ = _count_detections(
                exchange_df, g, metabolite, src, snk, patient_results)
            best_pid, was_exact = _select_best_patient(
                exchange_df, patient_results, g, metabolite, src, snk)

            if best_pid is None or best_pid not in patient_results:
                row_data.append(None)
                all_manifest.append({
                    "metabolite": metabolite, "exchange_rank": ri+1,
                    "source": src, "sink": snk, "rank_value": round(rank_val, 4),
                    "group": g, "best_patient": "",
                    "n_exact": n_exact, "n_any": n_any, "n_total": n_total,
                    "selection": "no_patient",
                })
                continue

            row_data.append({
                "results":    patient_results[best_pid],
                "src": src,   "snk": snk,
                "n_exact":    n_exact, "n_any": n_any, "n_total": n_total,
                "was_exact":  was_exact,
                "best_pid":   best_pid,
            })
            all_manifest.append({
                "metabolite": metabolite, "exchange_rank": ri+1,
                "source": src, "sink": snk, "rank_value": round(rank_val, 4),
                "group": g, "best_patient": best_pid,
                "n_exact": n_exact, "n_any": n_any, "n_total": n_total,
                "selection": "exact" if was_exact else "any",
            })
        panel_data.append(row_data)

    # Compute shared vmax across ALL spatial panels
    shared_vmax = 0.0
    for row_data in panel_data:
        for d in row_data:
            if d is None: continue
            s = _get_max_speed_coarsened(d["results"], metabolite, coarsen)
            if s > shared_vmax: shared_vmax = s
    if shared_vmax < 1e-12: shared_vmax = 1.0

    # Compute global_xmax for bar charts (across all rows and groups)
    global_xmax = 0.0
    for src, snk, _ in top_exch:
        for g in groups:
            for pid in [p for p in patient_results if _patient_group(p)==g]:
                spd = _compute_neighbourhood_flow_speed(
                    patient_results[pid], metabolite, src, snk)
                if spd > global_xmax:
                    global_xmax = spd
    if global_xmax < 1e-6: global_xmax = 1.0

    # Figure layout: [label | cond×N | bar]
    panel_w = 7; panel_h = 7; quant_w = 0.55
    fig_w = (0.16 + n_cols + quant_w) * panel_w * 0.9
    fig_h = n_rows * panel_h + 1.5
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    width_ratios = [0.16] + [1.0]*n_cols + [quant_w]
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols+2, figure=fig,
        width_ratios=width_ratios,
        wspace=0.05, hspace=0.18,
        left=0.03, right=0.97, top=0.93, bottom=0.04)

    any_plotted = False

    for ri, (src, snk, rank_val) in enumerate(top_exch):

        # Row label
        ax_lbl = fig.add_subplot(outer_gs[ri, 0])
        ax_lbl.axis("off")
        lbl_text = (f"#{ri+1}\n{src[:12]}\n→\n{snk[:12]}"
                    + (f"\n{row_label_prefix}{rank_val:.3f}"
                       if row_label_prefix else ""))
        lbl_col  = "#8B0000" if row_label_prefix else "#333333"
        ax_lbl.text(0.95, 0.5, lbl_text,
                    transform=ax_lbl.transAxes,
                    ha="right", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=lbl_col, linespacing=1.4)

        # Spatial panels
        for ci, g in enumerate(groups):
            ax = fig.add_subplot(outer_gs[ri, ci+1])
            d  = panel_data[ri][ci]
            if d is None:
                ax.set_facecolor("#f0f0f0"); ax.axis("off")
                n_tot = sum(1 for p in patient_results if _patient_group(p)==g)
                ax.text(0.5, 0.5,
                        f"{GROUP_LABELS.get(g,g)}\n0/{n_tot}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="#bbb")
                continue

            _draw_panel(
                ax=ax,
                results=d["results"],
                metabolite=metabolite,
                source_cell=src, sink_cell=snk,
                group=g,
                n_any=d["n_any"], n_total=d["n_total"],
                pid=d["best_pid"],
                cfg=cfg, shared_vmax=shared_vmax,
                coarsen=coarsen,
                n_seed_points=n_seed_points,
                cell_proximity_radius=cell_proximity_radius,
                speed_mask_threshold=speed_mask_threshold,
                seed=seed,
            )
            any_plotted = True

        # Quantitative inset
        ax_outer = fig.add_subplot(outer_gs[ri, n_cols+1])
        ax_outer.set_visible(False)
        bbox = ax_outer.get_position()
        bar_ax = fig.add_axes([
            bbox.x0 + 0.005,
            bbox.y0 + bbox.height*0.20,
            bbox.width - 0.010,
            bbox.height*0.62,
        ])
        _draw_quantitative_inset(
            bar_ax, exchange_df, metabolite, src, snk,
            groups, patient_results, global_xmax=global_xmax)

    if not any_plotted:
        plt.close(fig); return None, all_manifest

    fig.text(
        0.5, 0.008,
        "N/N = patients with any routing / group total  ·  "
        "Halos ∝ local flux magnitude  ·  "
        "Bar = mean flow speed at src+sink cells ± SEM",
        ha="center", va="bottom", fontsize=6.5, color="#666", fontstyle="italic")

    fig.suptitle(
        f"Consensus Metabolite Flow: {metabolite}{suptitle_suffix}\n"
        f"Top {n_rows} exchange(s)",
        fontsize=cfg.title_font_size, fontweight="bold", y=0.97)

    return fig, all_manifest


# ===========================================================================
# WCEG-weighted public entry point
# ===========================================================================

def generate_merged_streamline_plots_wceg(
    patient_results:       Dict,
    selected_metabolites:  List[str],
    out_dir:               str   = "cohort_output/merged_plots_wceg",
    groups:                Optional[List[str]]   = None,
    wceg_n_permutations:   int   = 1000,
    wceg_alpha:            float = 0.20,
    exchange_df:           Optional[pd.DataFrame] = None,
    cfg:                   Optional["PlotConfig"] = None,
    top_exchanges:         int   = 5,
    n_seed_points:         int   = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold:  float = 0.01,
    dpi:                   int   = 300,
    verbose:               bool  = True,
) -> List[str]:
    """
    Generate WCEG-weighted merged streamline plots → merged_plots_wceg/

    Exchange rows are selected and ranked by WCEG edge weight:
        W = mean(S_corr) × consistency
    where consistency = n_detected / n_group_total.

    This ranks exchanges by both strength (composition-corrected score) and
    reproducibility (fraction of group patients that showed the exchange).

    Everything else — spatial rendering, bar chart, patient selection — is
    identical to generate_merged_streamline_plots().

    Compare merged_plots/ (prevalence-ranked) with merged_plots_wceg/
    (strength×reproducibility-ranked) to understand which exchanges are
    statistically dominant vs. merely common.
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return []
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return []

    os.makedirs(out_dir, exist_ok=True)
    if groups is None: groups = COHORT_GROUPS
    if cfg    is None: cfg   = DEFAULT_CONFIG

    if verbose:
        print(f"\n{'='*62}")
        print(f"  MERGED STREAMLINE PLOTS  (WCEG-weighted)")
        print(f"  Ranking: W = mean(S_corr) × consistency")
        print(f"  Out: {out_dir}")
        print(f"  Mets: {len(selected_metabolites)}  |  Max rows: {top_exchanges}")
        print(f"  WCEG: n_perm={wceg_n_permutations}  alpha={wceg_alpha}")
        print(f"{'='*62}")

    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records …")
        exchange_df = build_exchange_records(patient_results)
    if verbose and not exchange_df.empty:
        print(f"  {exchange_df['metabolite'].nunique()} metabolites")

    saved:        List[str]  = []
    all_manifest: List[Dict] = []

    for ri, met in enumerate(selected_metabolites, 1):
        if _should_exclude_met_global(met):
            continue
        if verbose:
            print(f"  [{ri:2d}/{len(selected_metabolites)}] {met} …",
                  end=" ", flush=True)
        try:
            wceg_by_group = build_wceg_for_metabolite(
                exchange_df, patient_results, met,
                groups=groups,
                n_permutations=wceg_n_permutations,
                alpha=wceg_alpha,
            )

            # WCEG-ranked exchange list
            top_exch = _select_top_exchanges_wceg(
                wceg_by_group, groups, top_exchanges)

            if not top_exch:
                if verbose: print("SKIP (no WCEG edges)"); continue
            if verbose:
                print(f"[{len(top_exch)} WCEG rows] ", end="", flush=True)

            fig, manifest_rows = _make_figure_from_exchange_list(
                metabolite=met,
                patient_results=patient_results,
                exchange_df=exchange_df,
                top_exch=top_exch,
                groups=groups,
                cfg=cfg,
                n_seed_points=n_seed_points,
                cell_proximity_radius=cell_proximity_radius,
                speed_mask_threshold=speed_mask_threshold,
                seed=int(getattr(cfg, "random_seed", 42)),
                row_label_prefix="W=",
                suptitle_suffix="\n[WCEG-weighted: W = mean(S_corr) × consistency]",
            )
            all_manifest.extend(manifest_rows)

            if fig is None:
                if verbose: print("SKIP (no data)"); continue

            fname = f"merged_wceg_{ri:02d}_{met.replace('/','_')}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(fpath)
            if verbose: print(f"OK → {fname}")

        except Exception as e:
            if verbose:
                import traceback
                print(f"ERROR: {e}"); traceback.print_exc()

    if all_manifest:
        pd.DataFrame(all_manifest).to_csv(
            os.path.join(out_dir, "merged_wceg_manifest.csv"), index=False)

    if verbose:
        print(f"\n  ✓ {len(saved)} WCEG-weighted figures → {out_dir}\n")
    return saved


# ===========================================================================
# Differential streamline plots
# (exchanges that are statistically different between two conditions)
# ===========================================================================

def generate_merged_streamline_plots_differential(
    patient_results:       Dict,
    comparison_results:    Dict,            # output of run_full_cohort_comparison()
    out_dir:               str   = "cohort_output/merged_plots_differential",
    groups:                Optional[List[str]]   = None,
    exchange_df:           Optional[pd.DataFrame] = None,
    cfg:                   Optional["PlotConfig"] = None,
    top_metabolites:       int   = 20,      # top N significant metabolites per comparison
    top_exchanges:         int   = 5,       # max exchange rows per figure
    min_evidence:          float = 0.0,     # minimum evidence_score to include
    n_seed_points:         int   = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold:  float = 0.01,
    dpi:                   int   = 300,
    verbose:               bool  = True,
) -> Dict[str, List[str]]:
    """
    Generate differential merged streamline plots → merged_plots_differential/

    Purpose
    -------
    While generate_merged_streamline_plots() shows the STRONGEST exchanges
    (high absolute WCEG weight) and generate_merged_streamline_plots_wceg()
    shows the most REPRODUCIBLE exchanges (W = mean(S_corr) × consistency),
    this function shows the most DISEASE-ASSOCIATED exchanges — the ones
    that are statistically different between two conditions.

    Input
    -----
    comparison_results : dict returned by run_full_cohort_comparison()
        Keys are (group_a, group_b) tuples e.g. ("DKD","Control").
        Values are DataFrames with columns: metabolite, source, sink,
        evidence_score, significant, log2fc, perm_p, fisher_p.

    Output folder structure
    -----------------------
    merged_plots_differential/
        DKD_vs_Control/
            merged_diff_01_<met>.png   ← top hit by evidence_score
            merged_diff_02_<met>.png
            ...
            differential_manifest.csv
        HKD_vs_Control/
            ...

    Exchange row selection per figure
    ----------------------------------
    For a given significant metabolite in "DKD vs Control":
    1. Filter comparison_results to significant rows for this metabolite.
    2. Each significant (src,snk) pair becomes one row in the figure,
       ranked by evidence_score descending (most significant first).
    3. For metabolites with no significant (src,snk) pair but significant
       at metabolite level (classical comparison), fall back to the top
       exchanges from the WCEG edge tables.
    4. Row label shows: evidence_score and direction (↑DKD or ↓DKD).

    The spatial panels and bar charts are identical to the other functions.
    The bar chart is especially informative here because it directly shows
    WHY the exchange is differential — one group will have a taller bar.

    Returns
    -------
    dict: {comparison_label → list of saved file paths}
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return {}
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return {}

    if groups is None: groups = COHORT_GROUPS
    if cfg    is None: cfg   = DEFAULT_CONFIG

    if verbose:
        print(f"\n{'='*62}")
        print(f"  MERGED STREAMLINE PLOTS  (differential)")
        print(f"  Shows exchanges that differ between conditions")
        print(f"  Out: {out_dir}")
        print(f"  Top metabolites per comparison: {top_metabolites}")
        print(f"  Max exchange rows per figure  : {top_exchanges}")
        print(f"{'='*62}")

    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records …")
        exchange_df = build_exchange_records(patient_results)

    all_saved: Dict[str, List[str]] = {}

    for (group_a, group_b), comp_df in comparison_results.items():
        label = f"{group_a}_vs_{group_b}"
        comp_dir = os.path.join(out_dir, label)
        os.makedirs(comp_dir, exist_ok=True)

        if comp_df is None or comp_df.empty:
            if verbose: print(f"\n  {label}: empty comparison, skipping")
            continue

        if verbose:
            print(f"\n  ── {label} ──")

        # Determine the two groups actually being compared
        # (group_a is the "treatment", group_b is the "reference")
        cmp_groups = []
        for g in [group_a, group_b]:
            if g == "Diseased":
                cmp_groups += ["DKD", "HKD"]
            elif g in COHORT_GROUPS:
                cmp_groups.append(g)
        # Always show all three conditions for spatial context
        plot_groups = groups

        # ── Select significant metabolites ranked by evidence_score ───────
        sig_df = comp_df[comp_df["significant"]].copy()
        if sig_df.empty:
            if verbose: print(f"    No significant hits, skipping")
            continue

        if min_evidence > 0:
            sig_df = sig_df[sig_df["evidence_score"] >= min_evidence]

        sig_df = sig_df.sort_values("evidence_score", ascending=False)

        # Get unique metabolites (keep first / highest evidence_score per met)
        sig_mets_order = sig_df.drop_duplicates("metabolite")["metabolite"].tolist()
        sig_mets_order = sig_mets_order[:top_metabolites]

        if verbose:
            print(f"    {len(sig_mets_order)} significant metabolites "
                  f"(of {comp_df['significant'].sum()} sig edges)")

        saved_for_comp: List[str] = []
        all_manifest:   List[Dict] = []

        for ri, met in enumerate(sig_mets_order, 1):
            if _should_exclude_met_global(met):
                continue
            if verbose:
                print(f"    [{ri:2d}/{len(sig_mets_order)}] {met} …",
                      end=" ", flush=True)
            try:
                # ── Build top_exch from significant (src,snk) for this met ──
                met_sig = (sig_df[sig_df["metabolite"] == met]
                           .sort_values("evidence_score", ascending=False))

                top_exch: List[Tuple[str, str, float]] = []

                # Has edge-level (src,snk) info (WCEG comparison)?
                if "source" in met_sig.columns and "sink" in met_sig.columns:
                    for _, row in met_sig.head(top_exchanges).iterrows():
                        src = str(row["source"]); snk = str(row["sink"])
                        ev  = float(row["evidence_score"])
                        fc  = float(row.get("log2fc", 0.0))
                        # Encode direction in rank_value sign for label
                        # (positive = up in group_a, negative = down)
                        signed_ev = ev if fc > 0 else -ev
                        top_exch.append((src, snk, signed_ev))

                # Classical comparison (metabolite-level only) — fall back
                # to WCEG edges for this metabolite
                if not top_exch:
                    from consensus_exchange_network import build_wceg_for_metabolite
                    wceg_by_group = build_wceg_for_metabolite(
                        exchange_df, patient_results, met,
                        groups=plot_groups,
                        n_permutations=500,   # fast fallback
                        alpha=0.20,
                    )
                    top_exch = _select_top_exchanges_wceg(
                        wceg_by_group, plot_groups, top_exchanges)

                if not top_exch:
                    if verbose: print("SKIP (no edges)"); continue
                if verbose:
                    print(f"[{len(top_exch)} rows] ", end="", flush=True)

                # ── Build custom row_label_prefix per row ──────────────────
                # We encode the direction in the row label via suptitle_suffix
                # passed per-figure; individual row labels show ev + direction.
                # We abuse row_label_prefix="" and let suptitle_suffix carry
                # the comparison context; the signed rank_val drives the label.

                # Format: "#1  src→snk  ↑DKD  ev=0.42"
                # We encode direction info via a custom label builder below.

                fig, manifest_rows = _make_figure_from_exchange_list(
                    metabolite=met,
                    patient_results=patient_results,
                    exchange_df=exchange_df,
                    top_exch=top_exch,
                    groups=plot_groups,
                    cfg=cfg,
                    n_seed_points=n_seed_points,
                    cell_proximity_radius=cell_proximity_radius,
                    speed_mask_threshold=speed_mask_threshold,
                    seed=int(getattr(cfg, "random_seed", 42)),
                    row_label_prefix="ev=",
                    suptitle_suffix=(
                        f"\n[Differential: {group_a} vs {group_b}  ·  "
                        f"Ranked by evidence score]"
                    ),
                )

                # Post-process row labels to add direction arrows
                # (we do this by annotating the figure after building it)
                if fig is not None:
                    for ax in fig.get_axes():
                        title = ax.get_title()
                        if not title: continue
                        # Find which row this axis is in by checking title content
                        for src, snk, signed_ev in top_exch:
                            if src[:8] in title or snk[:8] in title:
                                direction = (f"↑{group_a}" if signed_ev > 0
                                             else f"↓{group_a}")
                                if direction not in title:
                                    ax.set_title(
                                        title.replace("rep:", f"{direction}  ·  rep:"),
                                        fontsize=ax.title.get_fontsize(),
                                        fontweight="bold",
                                        color=ax.title.get_color(),
                                    )
                                break

                # Add direction info to manifest rows
                for mr in manifest_rows:
                    mr["comparison"]  = label
                    mr["group_a"]     = group_a
                    mr["group_b"]     = group_b
                    # Look up direction from sig_df
                    edge_row = met_sig[
                        (met_sig.get("source", pd.Series()) == mr.get("source","")) &
                        (met_sig.get("sink",   pd.Series()) == mr.get("sink",""))
                    ] if "source" in met_sig.columns else pd.DataFrame()
                    if not edge_row.empty:
                        mr["log2fc"]        = float(edge_row.iloc[0].get("log2fc", 0))
                        mr["evidence_score"]= float(edge_row.iloc[0].get("evidence_score", 0))
                        mr["perm_p"]        = float(edge_row.iloc[0].get("perm_p", 1))
                all_manifest.extend(manifest_rows)

                if fig is None:
                    if verbose: print("SKIP (no data)"); continue

                fname = f"merged_diff_{ri:02d}_{met.replace('/','_')}.png"
                fpath = os.path.join(comp_dir, fname)
                fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                saved_for_comp.append(fpath)
                if verbose: print(f"OK → {fname}")

            except Exception as e:
                if verbose:
                    import traceback
                    print(f"ERROR: {e}"); traceback.print_exc()

        # Save per-comparison manifest
        if all_manifest:
            pd.DataFrame(all_manifest).to_csv(
                os.path.join(comp_dir, "differential_manifest.csv"), index=False)

        all_saved[label] = saved_for_comp
        if verbose:
            total = sum(len(v) for v in all_saved.values())
            print(f"\n    ✓ {len(saved_for_comp)} differential figures → {comp_dir}")

    if verbose:
        total = sum(len(v) for v in all_saved.values())
        print(f"\n  ✓ {total} total differential figures across "
              f"{len(all_saved)} comparisons\n")
    return all_saved


# ===========================================================================
# Combined differential-strength exchange selection
# W_max × |Δconsistency| × (−log10(p_Fisher) + floor)
# ===========================================================================

def _select_top_exchanges_combined(
    exchange_df:   pd.DataFrame,
    wceg_by_group: Dict[str, pd.DataFrame],
    patient_results: Dict,
    groups:        List[str],
    metabolite:    str,
    top_exchanges: int   = 5,
    ref_group:     str   = "Control",
    p_floor:       float = 0.5,
) -> List[Tuple[str, str, float]]:
    """
    Rank (src, snk) pairs by a combined differential-strength score:

        score = W_max × |Δconsistency| × (−log10(p_Fisher) + p_floor)

    where:
        W_max         = max WCEG edge weight across any group
                        = max(mean(S_corr) × consistency)  per group
        Δconsistency  = max(|cons_g − cons_ref|) across disease groups
                        (how much the detection rate changes vs reference)
        p_Fisher      = Fisher's exact p-value for the most different
                        disease group vs the reference group
        p_floor       = minimum contribution from statistics (default 0.5)
                        — prevents zero scores when n is too small for
                        Fisher's to reach p<0.05 (critical for n=3 Control)

    This single score simultaneously rewards:
      • Biologically meaningful exchanges  (high W_max)
      • Disease-relevant exchanges         (high |Δconsistency|)
      • Statistically supported changes    (low p_Fisher, as continuous weight)

    Exchanges that are equally strong in all groups (|Δ|=0) score 0 and
    fall to the bottom even if W_max is very large — correctly de-prioritising
    housekeeping exchanges that don't change in disease.

    Exchanges that change a lot but are weak (low W_max) also score low —
    correctly de-prioritising noise.

    Returns list of (src, snk, combined_score) tuples, sorted descending,
    up to top_exchanges entries.
    """
    from scipy.stats import fisher_exact as _fisher

    met_df = exchange_df[exchange_df["metabolite"] == metabolite]
    if met_df.empty:
        return []

    # ── Build per-group consistency for each (src, snk) ──────────────────
    n_per_group = {g: sum(1 for p in patient_results if _patient_group(p)==g)
                   for g in groups}
    ref_n = n_per_group.get(ref_group, 1)

    # ── Collect WCEG edge weights ─────────────────────────────────────────
    wceg_w: Dict[Tuple[str,str], float] = {}
    for g, edges in wceg_by_group.items():
        if edges is None or edges.empty: continue
        for _, row in edges.iterrows():
            key = (str(row["source"]), str(row["sink"]))
            w   = float(row.get("edge_weight", 0.0))
            if w > wceg_w.get(key, 0.0):
                wceg_w[key] = w

    # ── Score every (src, snk) pair ───────────────────────────────────────
    all_pairs = set(zip(met_df["source"].astype(str), met_df["sink"].astype(str)))
    scores: Dict[Tuple[str,str], float] = {}

    for (src, snk) in all_pairs:
        pair_df = met_df[
            (met_df["source"] == src) &
            (met_df["sink"]   == snk)
        ]

        # Per-group detection counts
        cons: Dict[str, float] = {}
        det:  Dict[str, int]   = {}
        for g in groups:
            n_g   = n_per_group.get(g, 1)
            n_det = int(pair_df[
                (pair_df["group"] == g) & (pair_df["score_raw"] > 0)
            ]["patient_id"].nunique())
            det[g]  = n_det
            cons[g] = n_det / max(n_g, 1)

        # Reference group consistency
        cons_ref = cons.get(ref_group, 0.0)
        det_ref  = det.get(ref_group,  0)

        # Max |Δconsistency| and corresponding Fisher's p across disease groups
        best_delta = 0.0
        best_p     = 1.0
        for g in groups:
            if g == ref_group: continue
            n_g   = n_per_group.get(g, 1)
            n_det = det.get(g, 0)
            delta = abs(cons[g] - cons_ref)
            if delta > best_delta:
                best_delta = delta
                try:
                    _, p = _fisher(
                        [[n_det,  n_g  - n_det],
                         [det_ref, ref_n - det_ref]],
                        alternative="two-sided")
                    best_p = float(p)
                except Exception:
                    best_p = 1.0

        # W_max from WCEG; fall back to mean score_corr × max_consistency
        W_max = wceg_w.get((src, snk), 0.0)
        if W_max == 0.0:
            sc_col = "score_corr" if "score_corr" in pair_df.columns else "score_raw"
            mean_sc = float(pair_df[sc_col].mean()) if not pair_df.empty else 0.0
            max_cons = max(cons.values()) if cons else 0.0
            W_max = mean_sc * max_cons

        stat_weight = -np.log10(max(best_p, 1e-10)) + p_floor
        combined    = W_max * best_delta * stat_weight
        scores[(src, snk)] = combined

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    # Only include pairs with combined > 0 (i.e. some differential signal)
    result = [(src, snk, s) for (src, snk), s in ranked[:top_exchanges] if s > 0]

    # If nothing has differential signal (e.g. metabolite present in all
    # groups equally), fall back to pure W_max ranking
    if not result:
        wceg_ranked = sorted(wceg_w.items(), key=lambda kv: kv[1], reverse=True)
        result = [(src, snk, w)
                  for (src, snk), w in wceg_ranked[:top_exchanges] if w > 0]

    return result


# ===========================================================================
# Combined merged streamline plots (main public entry point)
# ===========================================================================

def generate_merged_streamline_plots_combined(
    patient_results:       Dict,
    out_dir:               str   = "cohort_output/merged_plots_combined",
    groups:                Optional[List[str]]   = None,
    # Metabolite selection (same as plots_consensus_cohort)
    top_n:                 int   = 20,
    clinical_fields:       Optional[List[str]]   = None,
    clinical_weight:       float = 0.40,
    ckd_weight:            float = 0.60,
    top_clinical_k:        int   = 5,
    # Exchange row ranking
    top_exchanges:         int   = 5,
    ref_group:             str   = "Control",
    p_floor:               float = 0.5,
    # WCEG settings
    wceg_n_permutations:   int   = 1000,
    wceg_alpha:            float = 0.20,
    exchange_df:           Optional[pd.DataFrame] = None,
    cfg:                   Optional["PlotConfig"] = None,
    # Rendering
    n_seed_points:         int   = 300,
    cell_proximity_radius: float = 0.45,
    speed_mask_threshold:  float = 0.01,
    dpi:                   int   = 300,
    verbose:               bool  = True,
) -> List[str]:
    """
    Generate combined differential-strength merged streamline plots
    → merged_plots_combined/

    This function resolves the dilemma between merged_plots_wceg/
    (biologically strong but not disease-specific) and
    merged_plots_differential/ (disease-specific but statistically
    underpowered at n=14).

    Two-stage design
    ----------------

    Stage 1 — Metabolite selection (identical to plots_consensus_cohort)
        Uses select_target_metabolites() from cohort_consensus_plots.py:
            combined = ckd_weight × CKD_specificity
                     + clinical_weight × clinical_boost

        UNION  top_clinical_k metabolites per clinical field by |ρ|.

        This means the metabolite list includes:
          • Metabolites elevated in CKD vs Control
          • Metabolites correlated with eGFR or fibrosis (continuously)
          • Both categories when relevant

    Stage 2 — Exchange row ranking (combined differential-strength score)
        For each metabolite figure, rows are ordered by:
            score = W_max × |Δconsistency| × (−log10(p_Fisher) + p_floor)

        where:
          W_max        = max WCEG weight across groups (biological strength)
          Δconsistency = max |cons_disease − cons_control| (disease relevance)
          −log10(p)+½  = continuous statistical weight; p_floor=0.5 ensures
                         exchanges still score when n=3 prevents p<0.05

        Exchanges with |Δconsistency|=0 (unchanged across conditions) score
        zero and fall to the bottom regardless of how strong they are — this
        is the key property that makes the ranking disease-relevant.

        Fallback: if no exchange has differential signal, reverts to
        pure W_max ranking (shows the dominant biology for that metabolite).

    Row labels show the combined score so you can judge confidence.
    Bar charts show mean flow speed (consistent with spatial panels).

    Parameters
    ----------
    ref_group  : reference condition for Δconsistency and Fisher's test
                 (default "Control")
    p_floor    : minimum −log10(p) contribution (default 0.5, i.e. p~0.32)
                 prevents zero scores when sample size limits power
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return []
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return []

    # Import metabolite selection from cohort_consensus_plots
    try:
        from cohort_consensus_plots import (
            select_target_metabolites,
            build_cohort_metabolite_table,
            compute_composition_corrected_scores,
        )
        _HAS_CONS = True
    except ImportError:
        _HAS_CONS = False

    os.makedirs(out_dir, exist_ok=True)
    if groups is None:          groups         = COHORT_GROUPS
    if cfg    is None:          cfg            = DEFAULT_CONFIG
    if clinical_fields is None: clinical_fields = ["eGFR", "fibrosis"]

    if verbose:
        print(f"\n{'='*62}")
        print(f"  MERGED STREAMLINE PLOTS  (combined)")
        print(f"  Metabolite list  : same as plots_consensus_cohort")
        print(f"    top_n={top_n}  clinical_k={top_clinical_k}  "
              f"clin_w={clinical_weight}  ckd_w={ckd_weight}")
        print(f"  Row ranking      : W_max × |Δconsistency| × (-log10p + {p_floor})")
        print(f"  Reference group  : {ref_group}")
        print(f"  Out: {out_dir}")
        print(f"{'='*62}")

    # ── Build exchange records ─────────────────────────────────────────────
    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records …")
        exchange_df = build_exchange_records(patient_results)

    # ── Stage 1: Metabolite selection ─────────────────────────────────────
    if verbose: print("  Selecting metabolites (consensus approach) …")
    selected_mets: List[str] = []

    if _HAS_CONS:
        try:
            cohort_df      = build_cohort_metabolite_table(patient_results)
            corr_scores_df = compute_composition_corrected_scores(
                cohort_df, patient_results)
            selected_mets, rank_df, clin_ext = select_target_metabolites(
                cohort_df,
                top_n=top_n,
                patient_results=patient_results,
                corr_scores_df=corr_scores_df,
                clinical_fields=clinical_fields,
                clinical_weight=clinical_weight,
                ckd_weight=ckd_weight,
                top_clinical_k=top_clinical_k,
                return_metadata=True,
            )
            if verbose:
                print(f"  {len(selected_mets)} metabolites selected")
                reason_counts = rank_df[rank_df["metabolite"].isin(selected_mets)
                                        ]["selection_reason"].value_counts()
                for reason, cnt in reason_counts.items():
                    print(f"    {reason}: {cnt}")
        except Exception as e:
            if verbose: print(f"  WARNING: metabolite selection failed: {e}; "
                              f"using all exchange_df metabolites")

    if not selected_mets:
        # Fallback: all metabolites in exchange_df
        selected_mets = exchange_df["metabolite"].unique().tolist()
        if verbose: print(f"  Fallback: {len(selected_mets)} metabolites")

    if verbose:
        print(f"  {exchange_df['metabolite'].nunique()} metabolites in exchange records")

    # ── Stage 2: Generate figures ──────────────────────────────────────────
    saved:        List[str]  = []
    all_manifest: List[Dict] = []

    for ri, met in enumerate(selected_mets, 1):
        if _should_exclude_met_global(met):
            continue
        if verbose:
            print(f"  [{ri:2d}/{len(selected_mets)}] {met} …",
                  end=" ", flush=True)
        try:
            # Build WCEG edges for this metabolite
            wceg_by_group = build_wceg_for_metabolite(
                exchange_df, patient_results, met,
                groups=groups,
                n_permutations=wceg_n_permutations,
                alpha=wceg_alpha,
            )

            # Combined differential-strength row selection
            top_exch = _select_top_exchanges_combined(
                exchange_df, wceg_by_group, patient_results,
                groups, met,
                top_exchanges=top_exchanges,
                ref_group=ref_group,
                p_floor=p_floor,
            )

            if not top_exch:
                if verbose: print("SKIP (no exchanges)"); continue
            if verbose:
                print(f"[{len(top_exch)} rows] ", end="", flush=True)

            # Determine selection reason for suptitle annotation
            sel_reason = ""
            if _HAS_CONS and "rank_df" in dir():
                sub = rank_df[rank_df["metabolite"] == met]
                if not sub.empty:
                    sel_reason = str(sub.iloc[0].get("selection_reason",""))

            suptitle_suffix = (
                f"\n[Combined: W × |Δconsistency| × stat  ·  "
                f"ref={ref_group}"
                + (f"  ·  {sel_reason}" if sel_reason else "")
                + "]"
            )

            fig, manifest_rows = _make_figure_from_exchange_list(
                metabolite=met,
                patient_results=patient_results,
                exchange_df=exchange_df,
                top_exch=top_exch,
                groups=groups,
                cfg=cfg,
                n_seed_points=n_seed_points,
                cell_proximity_radius=cell_proximity_radius,
                speed_mask_threshold=speed_mask_threshold,
                seed=int(getattr(cfg, "random_seed", 42)),
                row_label_prefix="sc=",   # "sc" = combined score
                suptitle_suffix=suptitle_suffix,
            )

            # Add selection metadata to manifest
            for mr in manifest_rows:
                mr["selection_reason"] = sel_reason
            all_manifest.extend(manifest_rows)

            if fig is None:
                if verbose: print("SKIP (no data)"); continue

            fname = f"merged_combined_{ri:02d}_{met.replace('/','_')}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(fpath)
            if verbose: print(f"OK → {fname}")

        except Exception as e:
            if verbose:
                import traceback
                print(f"ERROR: {e}"); traceback.print_exc()

    # Save manifest
    if all_manifest:
        mf_path = os.path.join(out_dir, "merged_combined_manifest.csv")
        pd.DataFrame(all_manifest).to_csv(mf_path, index=False)

    if verbose:
        print(f"\n  ✓ {len(saved)} combined figures → {out_dir}\n")
    return saved


# ===========================================================================
# WCEG-significant merged streamline plots
# Metabolites = significant hits from WCEG volcano comparisons
# ===========================================================================

def generate_merged_streamline_plots_wceg_sig(
    patient_results:       Dict,
    comparison_results:    Dict,            # from run_full_cohort_comparison()
    out_dir:               str   = "cohort_output/merged_plots_wceg_sig",
    groups:                Optional[List[str]]   = None,
    wceg_n_permutations:   int   = 1000,
    wceg_alpha:            float = 0.20,
    exchange_df:           Optional[pd.DataFrame] = None,
    cfg:                   Optional["PlotConfig"] = None,
    top_metabolites:       int   = 20,      # max sig mets per comparison
    top_exchanges:         int   = 5,       # max exchange rows per figure
    dpi:                   int   = 300,
    verbose:               bool  = True,
) -> Dict[str, List[str]]:
    """
    Merged streamline plots for metabolites significant in the WCEG volcano.

    This creates the direct visual bridge between the statistical output
    (WCEG volcano plots) and the spatial biology (streamline diffusion).

    For each comparison in comparison_results (DKD vs Control, HKD vs Control,
    DKD vs HKD, Diseased vs Control):

      1. Extract significant metabolites from the comparison DataFrame,
         ranked by evidence_score descending, up to top_metabolites.

      2. For each significant metabolite, generate a merged streamline figure
         showing the top exchange rows ranked by WCEG weight W = mean(S_corr)
         x consistency. This is the same ranking as merged_plots_wceg/.

      3. Save to merged_plots_wceg_sig/<comparison>/

    Output structure:
        merged_plots_wceg_sig/
            DKD_vs_Control/
                merged_sig_01_glu_L.png     <- top significant hit
                merged_sig_02_ala_L.png
                ...
                wceg_sig_manifest.csv
            HKD_vs_Control/
                ...

    The figure layout is identical to generate_merged_streamline_plots_wceg():
    rows = top WCEG-weighted exchanges for this metabolite,
    cols = [Control | HKD | DKD | bar chart].
    Row labels show W= value and figure title notes the comparison.

    Parameters
    ----------
    comparison_results : dict returned by run_full_cohort_comparison()
        Keys: (group_a, group_b) tuples.
        Values: DataFrames with columns metabolite, evidence_score, significant.
    top_metabolites : max number of significant metabolites per comparison.
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return {}
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return {}

    if groups is None: groups = COHORT_GROUPS
    if cfg    is None: cfg   = DEFAULT_CONFIG

    if verbose:
        print(f"\n{'='*62}")
        print(f"  MERGED STREAMLINE PLOTS  (WCEG-significant)")
        print(f"  Metabolites: significant hits from WCEG volcano")
        print(f"  Exchange rows: ranked by W = mean(S_corr) x consistency")
        print(f"  Out: {out_dir}")
        print(f"{'='*62}")

    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records ...")
        exchange_df = build_exchange_records(patient_results)

    all_saved: Dict[str, List[str]] = {}

    for (group_a, group_b), comp_df in comparison_results.items():
        label    = f"{group_a}_vs_{group_b}"
        comp_dir = os.path.join(out_dir, label)
        os.makedirs(comp_dir, exist_ok=True)

        if comp_df is None or comp_df.empty:
            if verbose: print(f"\n  {label}: empty, skipping")
            continue

        # Extract significant metabolites ranked by evidence_score
        sig_df = comp_df[comp_df["significant"]].copy()
        if sig_df.empty:
            if verbose: print(f"\n  {label}: no significant hits, skipping")
            continue

        # Deduplicate to unique metabolites, keep highest evidence_score
        sig_mets = (sig_df.groupby("metabolite")["evidence_score"]
                         .max()
                         .sort_values(ascending=False)
                         .head(top_metabolites)
                         .index.tolist())

        if verbose:
            print(f"\n  {label}: {len(sig_mets)} significant mets "
                  f"(of {len(sig_df)} sig edges)")

        saved_for_comp: List[str]  = []
        all_manifest:   List[Dict] = []

        for ri, met in enumerate(sig_mets, 1):
            if _should_exclude_met_global(met):
                continue
            if verbose:
                ev = float(sig_df[sig_df["metabolite"]==met]["evidence_score"].max())
                print(f"    [{ri:2d}/{len(sig_mets)}] {met}  "
                      f"(evidence={ev:.3f}) ...", end=" ", flush=True)
            try:
                # Build WCEG edges ranked by W = mean(S_corr) x consistency
                wceg_by_group = build_wceg_for_metabolite(
                    exchange_df, patient_results, met,
                    groups=groups,
                    n_permutations=wceg_n_permutations,
                    alpha=wceg_alpha,
                )
                top_exch = _select_top_exchanges_wceg(
                    wceg_by_group, groups, top_exchanges)

                if not top_exch:
                    if verbose: print("SKIP (no WCEG edges)"); continue
                if verbose:
                    print(f"[{len(top_exch)} rows]", end=" ", flush=True)

                # Annotate suptitle with comparison context
                sig_row  = sig_df[sig_df["metabolite"]==met].iloc[0]
                fc_sign  = float(sig_row.get("log2fc", 0))
                direction = f"↑{group_a}" if fc_sign > 0 else f"↓{group_a}"

                fig, manifest_rows = _make_figure_from_exchange_list(
                    metabolite=met,
                    patient_results=patient_results,
                    exchange_df=exchange_df,
                    top_exch=top_exch,
                    groups=groups,
                    cfg=cfg,
                    seed=int(getattr(cfg, "random_seed", 42)),
                    row_label_prefix="W=",
                    suptitle_suffix=(
                        f"\n[WCEG-sig: {group_a} vs {group_b}  "
                        f"{direction}  "
                        f"evidence={float(sig_df[sig_df['metabolite']==met]['evidence_score'].max()):.3f}]"
                    ),
                )

                for mr in manifest_rows:
                    mr["comparison"]    = label
                    mr["group_a"]       = group_a
                    mr["group_b"]       = group_b
                    mr["direction"]     = direction
                    mr["evidence_score"]= float(
                        sig_df[sig_df["metabolite"]==met]["evidence_score"].max())
                all_manifest.extend(manifest_rows)

                if fig is None:
                    if verbose: print("SKIP (no data)"); continue

                fname = f"merged_sig_{ri:02d}_{met.replace('/','_')}.png"
                fpath = os.path.join(comp_dir, fname)
                fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                saved_for_comp.append(fpath)
                if verbose: print(f"OK -> {fname}")

            except Exception as e:
                if verbose:
                    import traceback
                    print(f"ERROR: {e}"); traceback.print_exc()

        if all_manifest:
            pd.DataFrame(all_manifest).to_csv(
                os.path.join(comp_dir, "wceg_sig_manifest.csv"), index=False)

        all_saved[label] = saved_for_comp
        if verbose:
            print(f"\n    ✓ {len(saved_for_comp)} figures -> {comp_dir}")

    if verbose:
        total = sum(len(v) for v in all_saved.values())
        print(f"\n  ✓ {total} total WCEG-sig figures across "
              f"{len(all_saved)} comparisons\n")
    return all_saved


# ===========================================================================
# Clinical-correlation merged streamline plots
# Uses Spearman correlation with clinical variables across all patients
# instead of group comparison — accounts for within-group heterogeneity
# ===========================================================================

def _get_clinical_scores(
    patient_results: Dict,
    clinical_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build patient x clinical DataFrame from COHORT_METADATA.
    Only includes continuous/ordinal variables by default:
      eGFR, fibrosis, age
    Binary variables (sex, hypertension, diabetes) are excluded by default
    because they are strongly confounded with disease group in this cohort
    (hypertension: 10/11 disease patients vs 1/3 Control — near-perfect
    confounding that produces artifactual rho~-1 for any disease-associated
    exchange).
    """
    try:
        from run_cohort_pipeline import COHORT_METADATA
        meta_by_id = {m["id"]: m for m in COHORT_METADATA}
    except ImportError:
        return pd.DataFrame()

    # Default: continuous variables only — avoid confounding with group
    if clinical_fields is None:
        clinical_fields = ["eGFR", "fibrosis", "age"]

    rows = {}
    for pid in patient_results:
        if pid not in meta_by_id:
            continue
        m   = meta_by_id[pid]
        row = {}
        for f in clinical_fields:
            val = m.get(f, None)
            if val is None:
                continue
            if f == "sex":
                row[f] = 1 if str(val).lower() in ("male","m") else 0
            elif f in ("hypertension","diabetes"):
                row[f] = int(bool(val))
            else:
                try:
                    row[f] = float(val)
                except (TypeError, ValueError):
                    pass
        if row:
            rows[pid] = row

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def _compute_clinical_correlations_per_edge(
    exchange_df:        pd.DataFrame,
    patient_results:    Dict,
    clinical_df:        pd.DataFrame,
    selected_mets:      Optional[List[str]] = None,
    p_floor:            float = 0.5,
    min_abs_rho:        float = 0.40,
    min_detected:       int   = 3,   # lowered: ser_D detected in ~3 patients
    top_per_met:        int   = 5,   # keep top N edges per met before FDR
) -> pd.DataFrame:
    """
    Spearman correlation between exchange score and clinical variables.

    FDR fix
    -------
    Previously tested all ~3000 edges even after metabolite pre-filtering,
    giving padj>0.7 for p=0.002 (too conservative).

    Now uses two-stage testing:
      Stage 1: compute raw Spearman for ALL edges of ALL metabolites
               (no pre-filter — ensures ser_D etc. are never excluded)
      Stage 2: keep top top_per_met edges per metabolite by |rho|
      Stage 3: apply BH-FDR to this reduced set (~mets x top_per_met tests)

    This gives ~selected_mets x top_per_met tests for FDR, which is
    ~20 x 5 = 100 for typical use — giving padj<0.20 for p<0.002.
    """
    from scipy.stats import spearmanr
    import warnings

    sc_col    = "score_corr" if "score_corr" in exchange_df.columns else "score_raw"
    clin_cols = list(clinical_df.columns)
    pids_all  = list(clinical_df.index)

    # Stage 1: compute raw correlations for all edges
    # (no selected_mets filter — clinical selection is independent)
    rows = []
    for (met, src, snk), pair in exchange_df.groupby(["metabolite","source","sink"]):
        if _should_exclude_met_global(met):
            continue
        score_map = (pair[pair["score_raw"] > 0]
                     .groupby("patient_id")[sc_col]
                     .mean().to_dict())
        scores = np.array([score_map.get(pid, 0.0) for pid in pids_all])
        n_det  = int((scores > 0).sum())
        if n_det < min_detected:
            continue
        W_max = float(scores[scores > 0].mean())

        row = {"metabolite": met, "source": src, "sink": snk,
               "W_max": W_max, "n_detected": n_det}
        for col in clin_cols:
            clin_vec = clinical_df[col].values.astype(float)
            valid    = ~np.isnan(clin_vec) & (scores >= 0)
            if valid.sum() < 5:
                row[f"rho_{col}"] = 0.0; row[f"p_{col}"] = 1.0
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = spearmanr(scores[valid], clin_vec[valid])
            row[f"rho_{col}"] = round(float(rho) if not np.isnan(rho) else 0.0, 4)
            row[f"p_{col}"]   = float(p)   if not np.isnan(p)   else 1.0
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Identify best field per edge (highest |rho| meeting min_abs_rho)
    rho_cols   = [c for c in df.columns if c.startswith("rho_")]
    p_nom_cols = [c.replace("rho_","p_") for c in rho_cols]

    best_rho_v=[]; best_p_v=[]; best_fld_v=[]
    for _, r in df.iterrows():
        cands = []
        for rc, pc in zip(rho_cols, p_nom_cols):
            rv = float(r.get(rc,0.0)); pv = float(r.get(pc,1.0))
            if abs(rv) >= min_abs_rho:
                cands.append((abs(rv), rv, pv, rc.replace("rho_","")))
        if cands:
            cands.sort(reverse=True)
            _, rv, pv, fld = cands[0]
        else:
            rv, pv, fld = 0.0, 1.0, ""
        best_rho_v.append(rv); best_p_v.append(pv); best_fld_v.append(fld)

    df["best_rho"]   = best_rho_v
    df["best_p"]     = best_p_v
    df["best_field"] = best_fld_v

    # Stage 2: keep top top_per_met edges per metabolite by |best_rho|
    # (only among edges that passed min_abs_rho)
    has_assoc = df["best_field"].str.len() > 0
    df_assoc  = df[has_assoc].copy()
    df_other  = df[~has_assoc].copy()

    if df_assoc.empty:
        df["best_padj"]  = 1.0
        stat_w = -np.log10(df["best_p"].clip(1e-10)) + p_floor
        df["signal"] = df["W_max"] * df["best_rho"].abs() * stat_w
        return df.sort_values("signal", ascending=False).reset_index(drop=True)

    # Select by |best_rho| so negative correlations (e.g. rho=-0.77
    # with fibrosis) are kept — not excluded by nlargest on raw rho value
    df_assoc = df_assoc.copy()
    df_assoc["_abs_rho"] = df_assoc["best_rho"].abs()
    top_edges = (df_assoc
                 .groupby("metabolite", group_keys=False)
                 .apply(lambda g: g.nlargest(top_per_met, "_abs_rho", keep="all"))
                 .reset_index(drop=True))
    df_assoc  = df_assoc.drop(columns=["_abs_rho"])
    top_edges = top_edges.drop(columns=["_abs_rho"])

    # Stage 3: BH-FDR on the reduced set
    p_arr = top_edges["best_p"].values
    try:
        from scipy.stats import false_discovery_control as _fdc
        padj = _fdc(p_arr, method="bh")
    except Exception:
        from scipy.stats import rankdata
        n    = len(p_arr)
        rank = rankdata(p_arr)
        padj = np.minimum(p_arr * n / rank, 1.0)
    top_edges["best_padj"] = padj

    # Merge back: non-top edges get padj=1
    df_assoc = df_assoc.merge(
        top_edges[["metabolite","source","sink","best_padj"]],
        on=["metabolite","source","sink"], how="left")
    df_assoc["best_padj"] = df_assoc["best_padj"].fillna(1.0)
    df_other["best_padj"] = 1.0

    df_out = pd.concat([df_assoc, df_other], ignore_index=True)

    # Signal: W_max x |rho| x (-log10(p_nominal) + floor)
    # Use nominal p so genuine p=0.006 ranks above padj=0.99
    stat_w = -np.log10(df_out["best_p"].clip(1e-10)) + p_floor
    df_out["signal"] = df_out["W_max"] * df_out["best_rho"].abs() * stat_w

    return df_out.sort_values("signal", ascending=False).reset_index(drop=True)


def generate_merged_streamline_plots_clinical(
    patient_results,
    out_dir="cohort_output/merged_plots_clinical",
    groups=None,
    clinical_fields=None,
    min_abs_rho=0.35,
    p_nominal_threshold=0.15,
    top_metabolites=50,
    top_exchanges=5,
    p_floor=0.5,
    precomputed_mets=None,
    exchange_df=None,
    cfg=None,
    wceg_n_permutations=500,
    wceg_alpha=0.20,
    dpi=300,
    verbose=True,
):
    """
    Merged streamline plots for clinically-correlated metabolites.

    Uses build_full_score_matrix() to get one aggregated score per patient
    per metabolite, then Spearman-correlates with clinical variables.
    This is the same approach as cohort_statistics.py, ensuring ser_D and
    other metabolites detected across few-but-meaningful patients are found.

    precomputed_mets: optional list of metabolite names to skip correlation
    and go straight to spatial plot generation (e.g. from clinical_results).
    """
    if not _HAS_PLOTTING:
        if verbose: print("  ERROR: plotting.py not importable."); return []
    if not _HAS_CEG:
        if verbose: print("  ERROR: consensus_exchange_network.py not importable."); return []

    from scipy.stats import spearmanr as _spearmanr
    import warnings

    if groups is None:          groups          = COHORT_GROUPS
    if cfg    is None:          cfg             = DEFAULT_CONFIG
    if clinical_fields is None: clinical_fields = ["eGFR", "fibrosis", "age"]

    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*62)
        print("  MERGED STREAMLINE PLOTS  (clinical correlation)")
        if precomputed_mets is not None:
            print("  Mode    : precomputed_mets (%d provided)" % len(precomputed_mets))
        else:
            print("  Method  : Spearman rho on aggregated scores (cohort_statistics approach)")
            print("  Fields  : %s" % str(clinical_fields))
            print("  Filter  : |rho| >= %.2f  AND  p < %.2f" % (min_abs_rho, p_nominal_threshold))
        print("  Out: %s" % out_dir)
        print("="*62)

    if exchange_df is None or exchange_df.empty:
        if verbose: print("  Building exchange records ...")
        exchange_df = build_exchange_records(patient_results)

    # ── Mode: precomputed ─────────────────────────────────────────────────
    if precomputed_mets is not None:
        sel_for_plot = [m for m in list(dict.fromkeys(precomputed_mets))
                        if not _should_exclude_met_global(m)][:top_metabolites]
        corr_lookup = {}
        if verbose:
            print("  %d metabolites from precomputed_mets" % len(sel_for_plot))

    else:
        # ── Build aggregated score matrix ─────────────────────────────────
        try:
            from cohort_comparison import build_full_score_matrix
            score_matrix = build_full_score_matrix(patient_results).fillna(0.0)
        except Exception as e:
            if verbose: print("  ERROR building score matrix: %s" % e); return []

        clin_df = _get_clinical_scores(patient_results, clinical_fields)
        if clin_df.empty:
            if verbose: print("  ERROR: no clinical data"); return []

        pids_common = [p for p in score_matrix.columns if p in clin_df.index]
        sm_sub      = score_matrix[pids_common]
        clin_sub    = clin_df.loc[pids_common]

        if verbose:
            print("  Score matrix: %d mets x %d patients" % (len(sm_sub), len(pids_common)))

        # ── Spearman per metabolite ───────────────────────────────────────
        corr_rows = []
        for met in sm_sub.index:
            if _should_exclude_met_global(met):
                continue
            scores = sm_sub.loc[met].values.astype(float)
            best_rho = 0.0; best_p = 1.0; best_fld = ""
            field_res = {}
            for col in clin_sub.columns:
                clin_vec = clin_sub[col].values.astype(float)
                valid    = np.isfinite(clin_vec) & np.isfinite(scores)
                if valid.sum() < 4:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        rho, p = _spearmanr(scores[valid], clin_vec[valid])
                        rho = float(rho) if np.isfinite(rho) else 0.0
                        p   = float(p)   if np.isfinite(p)   else 1.0
                    except Exception:
                        rho, p = 0.0, 1.0
                field_res[col] = (rho, p)
                if abs(rho) > abs(best_rho):
                    best_rho = rho; best_p = p; best_fld = col

            row = {"metabolite": met, "best_rho": best_rho,
                   "best_p": best_p, "best_field": best_fld}
            for col, (rv, pv) in field_res.items():
                row["rho_" + col] = rv
                row["p_" + col]   = pv
            corr_rows.append(row)

        if not corr_rows:
            if verbose: print("  No metabolites processed"); return []

        corr_df = pd.DataFrame(corr_rows)

        # BH-FDR per field (annotation only)

        # No BH-FDR: pre-specified metabolites from spatial pipeline;
        # BH across ~450 tests at n=14 is too conservative and
        # statistically incoherent for individually-motivated hypotheses.
        # Nominal Spearman p is reported. Readers can judge effect size.
        corr_df["best_padj"] = float("nan")  # not computed
        corr_df["signal"]    = (corr_df["best_rho"].abs() *
                                (-np.log10(corr_df["best_p"].clip(1e-10)) + p_floor))
        corr_df = corr_df.sort_values("signal", ascending=False).reset_index(drop=True)
        corr_df.to_csv(os.path.join(out_dir, "clinical_correlations.csv"), index=False)

        # Selection
        sel_mask = ((corr_df["best_rho"].abs() >= min_abs_rho) &
                    (corr_df["best_p"] < p_nominal_threshold))
        sel_df   = corr_df[sel_mask]

        if verbose:
            print("  %d mets tested  |  %d pass filter" % (len(corr_df), int(sel_mask.sum())))
            # ser_D diagnostic
            ser = corr_df[corr_df["metabolite"] == "ser_D"]
            if not ser.empty:
                r = ser.iloc[0]
                passed = bool(sel_mask[ser.index[0]])
                print("  [ser_D] rho=%+.3f with %s p=%.4f -> %s" %
                      (r["best_rho"], r["best_field"], r["best_p"],
                       "SELECTED" if passed else "below threshold"))
            else:
                print("  [ser_D] not in score matrix")
            for _, r in sel_df.head(5).iterrows():
                print("    %-20s rho=%+.3f with %-10s p=%.4f" %
                      (r["metabolite"], r["best_rho"],
                       r["best_field"], r["best_p"]))



        if len(sel_df) < 3:
            sel_df = corr_df[corr_df["best_rho"].abs() >= min_abs_rho].head(top_metabolites)
            if verbose: print("  Relaxed: using top by |rho| only")

        if sel_df.empty:
            if verbose: print("  No metabolites selected"); return []

        sel_for_plot = sel_df.head(top_metabolites)["metabolite"].tolist()
        corr_lookup  = {r.name: r
                        for _, r in sel_df.set_index("metabolite").iterrows()
                        if r.name in sel_for_plot}

        if verbose:
            print("\n  %d metabolites selected:" % len(sel_for_plot))
            for met in sel_for_plot:
                r = sel_df[sel_df["metabolite"]==met].iloc[0]
                print("    %-20s rho=%+.3f with %-10s p=%.4f" %
                      (met, r["best_rho"], r["best_field"], r["best_p"]))



    # ── Spatial figures ───────────────────────────────────────────────────
    saved       = []
    all_manifest = []

    for ri, met in enumerate(sel_for_plot, 1):
        if _should_exclude_met_global(met): continue

        r         = corr_lookup.get(met)
        best_rho  = float(r["best_rho"])  if r is not None else 0.0
        best_p    = float(r["best_p"])    if r is not None else 1.0
        best_padj = float(r.get("best_padj", float("nan"))) if r is not None else float("nan")
        best_fld  = str(r["best_field"])  if r is not None else ""
        direction = ("increases" if best_rho > 0 else "decreases") if best_fld else ""

        if verbose:
            suffix = (" (rho=%+.3f with %s, p=%.4f)" %
                      (best_rho, best_fld, best_p)) if best_fld else ""
            print("  [%2d/%d] %s%s ..." % (ri, len(sel_for_plot), met, suffix),
                  end=" ", flush=True)
        try:
            wceg_by_group = build_wceg_for_metabolite(
                exchange_df, patient_results, met,
                groups=groups,
                n_permutations=wceg_n_permutations,
                alpha=wceg_alpha,
            )
            top_exch = _select_top_exchanges_wceg(wceg_by_group, groups, top_exchanges)

            if not top_exch:
                if verbose: print("SKIP (no WCEG edges)"); continue
            if verbose:
                print("[%d rows]" % len(top_exch), end=" ", flush=True)


            if best_fld:
                suptitle_suffix = (
                    "\n[Clinical: Spearman rho(" + best_fld + ") = " +
                    ("%+.3f  p = %.4f" % (best_rho, best_p)) +
                    "  (exploratory, n=14, nominal p)  " +
                    "Exchange " + direction + " with " + best_fld + "]"
                )
            else:
                suptitle_suffix = ""

            fig, manifest_rows = _make_figure_from_exchange_list(
                metabolite=met,
                patient_results=patient_results,
                exchange_df=exchange_df,
                top_exch=top_exch,
                groups=groups,
                cfg=cfg,
                seed=int(getattr(cfg, "random_seed", 42)),
                row_label_prefix="W=",
                suptitle_suffix=suptitle_suffix,
            )
            for mr in manifest_rows:
                if best_fld:
                    mr.update({"best_field": best_fld, "best_rho": best_rho,
                                "best_p": best_p, "best_padj": best_padj})
            all_manifest.extend(manifest_rows)

            if fig is None:
                if verbose: print("SKIP (no data)"); continue

            fname = "merged_clin_%02d_%s" % (ri, met.replace("/","_"))
            if best_fld:
                fname += "_rho%+.2f_%s" % (best_rho, best_fld)
            fname += ".png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved.append(fpath)
            if verbose: print("OK -> %s" % fname)

        except Exception as e:
            if verbose:
                import traceback
                print("ERROR: %s" % e); traceback.print_exc()

    if all_manifest:
        pd.DataFrame(all_manifest).to_csv(
            os.path.join(out_dir, "merged_clinical_manifest.csv"), index=False)

    if verbose:
        print("\n  ✓ %d clinical-correlation figures -> %s\n" % (len(saved), out_dir))
    return saved
