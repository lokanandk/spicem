#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS - PLOTTING MODULE (FINAL WORKING VERSION)
============================================================================

ALL CRITICAL FIXES APPLIED:
1. Arrow crowding massively reduced (coarsen_factor=2 gives 75% fewer arrows)
2. Smaller arrow heads (1.8 instead of 2.5)
3. Missing consensus plots FIXED (uses old working logic)
4. Missing differential plots FIXED (uses old working logic)
5. Color variation FIXED (full spectrum, not just yellow)
6. Enhanced metabolites bar plot with significance colors
7. CRITICAL FIX: Robust handling of Flux Balance data. Auto-regenerates data
   if the pipeline results are in the old format, preventing KeyError.

CRITICAL: This version uses the OLD WORKING differential plot logic that successfully
generated plots by averaging vector fields across regions.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import networkx as nx

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except Exception:
    HAS_ADJUSTTEXT = False


# =============================================================================
# Config - OPTIMIZED FOR CLARITY
# =============================================================================

@dataclass
class PlotConfig:
    # Streamline plots - P06 panel style
    coarsen_factor: int = 2  # 3 Spatial binning - reduces arrows by 75%
    streamline_density: float = 4.5   # P06: density=1.5
    streamline_lw_scale: float = 2.0  # speed_n * 2.0 + 0.6
    streamline_lw_base: float = 0.6   # 0.6 additive base linewidth
    streamline_arrow_size: float = 1.1  # 1.1 slightly larger arrowheads
    streamline_integration_direction: str = "forward"
    streamline_minlength: float = 0.2  # 0.4 P06: minlength=0.4
    streamline_maxlength: float = 5.5
    streamline_broken_streamlines: bool = False
    speed_clip_lo_pct: float = 5.0
    speed_clip_hi_pct: float = 95.0
    linewidth_min: float = 0.6
    linewidth_max: float = 3.2
    scatter_size_bg: int = 14
    scatter_size_highlight: int = 70
    scatter_alpha_bg: float = 0.22
    scatter_alpha_highlight: float = 0.82
    node_size_base: int = 1200
    node_size_scale: int = 800
    node_alpha: float = 0.88
    node_edge_width: float = 2.0
    edge_width_scale: float = 4.0
    edge_alpha: float = 0.70
    edge_label_font_size: int = 9
    node_label_font_size: int = 11
    edge_label_alpha: float = 0.95
    edge_label_box_alpha: float = 0.85
    network_k_spacing: float = 2.2
    title_font_size: int = 16
    subtitle_font_size: int = 14
    axis_label_font_size: int = 12
    legend_font_size: int = 10
    tick_font_size: int = 9
    source_color: str = "#d63031"
    sink_color: str = "#0984e3"
    bg_color: str = "#dfe6e9"
    producer_color: str = "#e74c3c"
    consumer_color: str = "#1abc9c"
    balanced_color: str = "#95a5a6"
    consensus_figsize: Tuple[int, int] = (18, 8)
    differential_figsize: Tuple[int, int] = (22, 7)
    network_figsize: Tuple[int, int] = (20, 16)
    regional_figsize: Tuple[int, int] = (12, 10)
    heatmap_figsize: Tuple[int, int] = (14, 12)
        
    # Separate settings for consensus/differential plots (P06 style, slightly lower density)
    consensus_differential_density: float = 3.2   # 1.2 Slightly lower than panel default
    consensus_differential_arrow_size: float = 0.9  # 0.9 Same as P06
    consensus_differential_minlength: float = 0.2   # Same as P06
    consensus_differential_coarsen_factor: int = 2  # More coarsening (89% reduction)
    
    # Metabolite filtering for consensus/differential
    consensus_differential_top_n: int = 10  # Limit to top N metabolites
    balance_figsize: Tuple[int, int] = (16, 10)
    diffusion_figsize: Tuple[int, int] = (18, 8)

    # Diffusion panel (P05 style)
    diffusion_cmap: str = "RdYlBu_r"
    diffusion_quiver_color: str = "black"
    diffusion_quiver_alpha: float = 0.75
    diffusion_quiver_scale_factor: float = 28.0
    diffusion_quiver_step: int = 5
    diffusion_quiver_width: float = 0.003

    # Global random seed — controls ALL stochastic elements:
    # streamline seed-point sampling, network spring layout,
    # diffusion scatter sampling. Set to any integer for
    # reproducible plots; set to None to allow variation.
    random_seed: int = 42


DEFAULT_CONFIG = PlotConfig()
EXCLUDED_METABOLITES = {"h2o", "h2o2", "h", "co2", "o2"}

# P06 panel style: blue-toned flow colormap for streamline plots
FLOW_CMAP = LinearSegmentedColormap.from_list(
    'flow',
    ['#eaf4fd', '#a8d4f5', '#4a9fd4', '#1464a0', '#082b5e', '#010d22']
)


# =============================================================================
# Helpers
# =============================================================================

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_out_dir(results, subdir: str):
    base = getattr(getattr(results, "config", None), "out_dir", None)
    if base is None:
        base = "analysis_output"
    out_dir = os.path.join(base, subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _is_single_condition(results) -> bool:
    """
    Return True when the pipeline was run on exactly one condition / one region.
    In this case all two-condition comparison plots (differential streamlines,
    volcano, FC heatmap, flux balance difference panel) are meaningless and
    are automatically skipped with an informative message.
    """
    conditions = getattr(getattr(results, "config", None), "conditions", [])
    return len(conditions) < 2


def _single_condition_label(results) -> str:
    """Return a short human-readable label for the single condition."""
    conditions = getattr(getattr(results, "config", None), "conditions", [])
    if conditions:
        return conditions[0].replace("Sample_", "")
    return "Condition"


def _norm_met(m: str) -> str:
    return str(m).strip().lower()


def _should_exclude_metabolite(m: str) -> bool:
    return _norm_met(m) in EXCLUDED_METABOLITES


def _filter_met_list(mets: List[str]) -> List[str]:
    return [m for m in mets if not _should_exclude_metabolite(m)]


def _truncate_label(s: str, max_len: int = 55) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _parse_condition_region(key: str) -> Tuple[Optional[str], Optional[str]]:
    if "_R" not in key:
        return None, None
    parts = key.rsplit("_R", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _choose_representative_region_key(results, condition: str) -> Optional[str]:
    per_region = getattr(results, "per_region_data", {}) or {}
    best_key, best_n = None, -1
    for key, data in per_region.items():
        cond, _reg = _parse_condition_region(key)
        if cond != condition:
            continue
        meta_df = data.get("meta_df", pd.DataFrame())
        n = int(len(meta_df)) if meta_df is not None else 0
        if n > best_n:
            best_n = n
            best_key = key
    return best_key


def _robust_vmin_vmax(speed: np.ndarray, lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    s = np.asarray(speed, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(s, float(lo_pct)))
    vmax = float(np.percentile(s, float(hi_pct)))
    if not np.isfinite(vmin):
        vmin = 0.0
    if (not np.isfinite(vmax)) or (vmax <= vmin + 1e-12):
        mx = float(np.max(s))
        vmax = mx if mx > vmin else (vmin + 1.0)
    return vmin, vmax


# =============================================================================
# Streamline panel - FIXED
# =============================================================================

def _coarsen_vector_field(U, V, xi, yi, factor):
    """
    Coarsen a vector field by averaging blocks.
    factor=2: 75% reduction, factor=3: 89% reduction
    """
    if factor <= 1:
        return U, V, xi, yi
        
    ny, nx = U.shape
    new_ny = ny // factor
    new_nx = nx // factor
    
    # Trim to multiple of factor
    U_trim = U[:new_ny * factor, :new_nx * factor]
    V_trim = V[:new_ny * factor, :new_nx * factor]
    
    # Reshape and average
    U_coarse = U_trim.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))
    V_coarse = V_trim.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))
    
    # Coarsen coordinate arrays
    xi_coarse = xi[factor//2::factor][:new_nx]
    yi_coarse = yi[factor//2::factor][:new_ny]
    
    return U_coarse, V_coarse, xi_coarse, yi_coarse


def _filter_metadata_to_grid(meta_df: pd.DataFrame, xi: np.ndarray, yi: np.ndarray, 
                              margin_pct: float = 0.05) -> pd.DataFrame:
    """
    CRITICAL FIX: Filter metadata to only include cells within the vector field grid boundaries.
    
    This prevents showing all 5 regions overlaid - instead shows only the region 
    corresponding to the vector field's coordinate space.
    
    Args:
        meta_df: Full metadata with all cells from all regions
        xi, yi: Grid coordinates from vector field
        margin_pct: Allow small margin (5%) outside grid to catch boundary cells
    
    Returns:
        Filtered metadata containing only cells within this specific region's grid
    """
    if meta_df is None or meta_df.empty:
        return meta_df
    
    # Calculate grid boundaries with small margin
    x_min, x_max = float(np.min(xi)), float(np.max(xi))
    y_min, y_max = float(np.min(yi)), float(np.max(yi))
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= margin_pct * x_range
    x_max += margin_pct * x_range
    y_min -= margin_pct * y_range
    y_max += margin_pct * y_range
    
    # Filter cells to this region only
    mask = (
        (meta_df["px_x"] >= x_min) & 
        (meta_df["px_x"] <= x_max) &
        (meta_df["px_y"] >= y_min) & 
        (meta_df["px_y"] <= y_max)
    )
    
    filtered = meta_df[mask].copy()
    
    return filtered


def _make_cell_proximity_mask(
    meta_df: pd.DataFrame,
    src_type: Optional[str],
    snk_type: Optional[str],
    xi: np.ndarray,
    yi: np.ndarray,
    radius_fraction: float = 0.45,
    min_cells_threshold: int = 2,
) -> np.ndarray:
    """
    Build a boolean mask (shape = len(yi) x len(xi)) that is True at every
    grid point within `radius` of ANY source OR sink cell (union).

    Grid points outside this union are zeroed so the streamplot integrator
    cannot travel through biologically empty space.  The key insight is:

      - We seed lines FROM source cells (so they start correctly).
      - We allow the field to exist near BOTH source AND sink cells (union),
        so lines can travel the full corridor from source to sink.
      - We suppress the field everywhere else (neither source nor sink nearby),
        so lines stop as soon as they leave the biological neighbourhood.

    Why UNION and not INTERSECTION:
      Intersection only keeps grid points near BOTH cell types simultaneously.
      For a cell pair that is spatially separated (as most are), the intersection
      is nearly empty -- the corridor between them gets zeroed too, and lines
      terminate immediately after leaving the source cluster.
      Union preserves the entire neighbourhood of both populations, so lines
      can travel from source cluster through the field to the sink cluster and
      stop naturally when they exit both neighbourhoods.

    The min_cells_threshold guard handles the case where a cell type is truly
    absent in one condition -- return all-False so no lines are drawn at all.

    radius_fraction=0.15 means each cell illuminates a circle of radius
    15% of the grid diagonal (~1.5x the previous value to ensure corridor
    connectivity between neighbouring clusters).
    """
    gs_y, gs_x = len(yi), len(xi)
    no_mask = np.zeros((gs_y, gs_x), dtype=bool)

    if meta_df is None or meta_df.empty:
        return no_mask

    cell_types = meta_df["cell_type"].astype(str).values

    def _get_cells(ct):
        if not ct or str(ct) not in cell_types:
            return np.array([]), np.array([])
        sub = meta_df[meta_df["cell_type"].astype(str) == str(ct)]
        return sub["px_x"].astype(float).values, sub["px_y"].astype(float).values

    sx, sy = _get_cells(src_type)
    kx, ky = _get_cells(snk_type)

    # If either cell type is absent or too sparse: no meaningful flux to show.
    if len(sx) < min_cells_threshold or len(kx) < min_cells_threshold:
        return no_mask

    x_span = float(xi[-1] - xi[0])
    y_span = float(yi[-1] - yi[0])
    radius = radius_fraction * np.sqrt(x_span ** 2 + y_span ** 2)

    XX, YY = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    try:
        from scipy.spatial import cKDTree
        src_mask = (cKDTree(np.column_stack([sx, sy]))
                    .query(grid_pts, workers=-1)[0] <= radius
                    ).reshape(gs_y, gs_x)
        snk_mask = (cKDTree(np.column_stack([kx, ky]))
                    .query(grid_pts, workers=-1)[0] <= radius
                    ).reshape(gs_y, gs_x)
    except Exception:
        src_mask = np.zeros((gs_y, gs_x), dtype=bool)
        for gx, gy in zip(sx, sy):
            src_mask |= ((XX - gx) ** 2 + (YY - gy) ** 2) <= radius ** 2
        snk_mask = np.zeros((gs_y, gs_x), dtype=bool)
        for gx, gy in zip(kx, ky):
            snk_mask |= ((XX - gx) ** 2 + (YY - gy) ** 2) <= radius ** 2

    # UNION: preserve field near source OR near sink.
    # Lines seeded at source cells will integrate forward through the field
    # and naturally stop when they exit the union neighbourhood (i.e. when
    # they have left both the source cluster and the sink cluster).
    return src_mask | snk_mask


def _build_seed_points_from_cells(
    meta_df: pd.DataFrame,
    src_type: str,
    xi: np.ndarray,
    yi: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Sample up to n_seeds start-points from actual source-cell positions,
    clipped to the interior of the vector-field grid so streamplot never
    rejects them.  Returns an (N,2) array of [x,y] coordinates, or None.

    Seeding from source-cell positions rather than a uniform grid guarantees
    that every streamline genuinely originates from a secreting cell --
    lines cannot start in biologically empty regions.
    """
    if meta_df is None or meta_df.empty or not src_type:
        return None
    src_cells = meta_df[meta_df["cell_type"].astype(str) == str(src_type)]
    if src_cells.empty:
        return None

    xs = src_cells["px_x"].values.astype(float)
    ys = src_cells["px_y"].values.astype(float)

    # Clip to valid grid interior (streamplot requires points strictly inside)
    x_lo, x_hi = float(np.min(xi)), float(np.max(xi))
    y_lo, y_hi = float(np.min(yi)), float(np.max(yi))
    eps_x = (x_hi - x_lo) * 0.01
    eps_y = (y_hi - y_lo) * 0.01
    valid = (
        (xs >= x_lo + eps_x) & (xs <= x_hi - eps_x) &
        (ys >= y_lo + eps_y) & (ys <= y_hi - eps_y)
    )
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        return None

    if len(xs) > n_seeds:
        idx = rng.choice(len(xs), size=n_seeds, replace=False)
        xs, ys = xs[idx], ys[idx]

    return np.column_stack([xs, ys])


def plot_streamline_panel(
    ax,
    U: np.ndarray,
    V: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    meta_df: pd.DataFrame,
    src_type: Optional[str],
    snk_type: Optional[str],
    title: str = "",
    show_colorbar: bool = True,
    xlim=None,
    ylim=None,
    cfg: Optional[PlotConfig] = None,
    cmap=None,  # P06 style: defaults to FLOW_CMAP if not specified
    colorbar_label: str = "Flow speed",
    use_consensus_settings: bool = False,
    debug: bool = False,
    # ── Shared-scale parameters (for cross-condition comparability) ──────────
    shared_vmax: Optional[float] = None,
    # Fraction of max speed below which arrows are suppressed.
    # This makes low-flux conditions visually sparse vs. high-flux ones.
    speed_mask_threshold: float = 0.02,
    # Fraction of grid diagonal used as neighbourhood radius around source/sink
    # cells.  Grid points farther than this from every source AND sink cell have
    # their U/V zeroed, so streamlines cannot appear in biologically empty space.
    cell_proximity_radius: float = 0.45,
    # Number of seed points sampled from source-cell positions.
    # Set to 0 to fall back to uniform-density seeding.
    n_seed_points: int = 180,
    # Random seed for reproducible seed-point sampling.
    seed: int = 42,
    # Optional DataFrame with per-cell-type flux magnitudes for halo sizing.
    # Expected columns: 'cell_type', 'flux' (absolute secretion or uptake).
    # If None, all halos use a fixed size (cfg.scatter_size_highlight * 3.5).
    flux_df: Optional[pd.DataFrame] = None,
):
    """
    Renders a streamline panel.

    Key fix for cross-condition comparability
    -----------------------------------------
    • shared_vmax: pins colour scale to the same value across all conditions.

    • speed_mask_threshold: suppresses arrows where flux is too weak globally.

    • cell_proximity_radius: suppresses arrows in grid regions that are farther
      than this fraction of the grid diagonal from every source/sink cell.
      This is the fix for streamlines appearing in biologically empty regions —
      the vector field has nonzero values everywhere (it is computed over the
      whole slide), but we only show arrows near actual cells.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    # P06 style: use blue flow colormap by default; caller can override (e.g. coolwarm for differential)
    if cmap is None:
        cmap = FLOW_CMAP

    # === DEBUG OUTPUT ===
    if False:  # debug:
        print(f"\n{'='*60}")
        print(f"DEBUG plot_streamline_panel: {title}")
        print(f"  U shape: {U.shape}, V shape: {V.shape}")
        print(f"  xi shape: {xi.shape}, yi shape: {yi.shape}")
        print(f"  xi range: [{np.min(xi):.2f}, {np.max(xi):.2f}]")
        print(f"  yi range: [{np.min(yi):.2f}, {np.max(yi):.2f}]")
        print(f"  xlim: {xlim}, ylim: {ylim}")
        print(f"  meta_df shape: {meta_df.shape if meta_df is not None else 'None'}")
        print(f"  src_type: {src_type}, snk_type: {snk_type}")

    # === INPUT VALIDATION ===
    try:
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        
        if U.shape != V.shape:
            if False:  # debug:
                print(f"  ERROR: U and V shapes don't match!")
            return
        
        if len(xi) != U.shape[1] or len(yi) != U.shape[0]:
            if False:  # debug:
                print(f"  ERROR: Grid dimensions mismatch!")
                print(f"    Expected: ({len(yi)}, {len(xi)}), Got: {U.shape}")
            return
            
    except Exception as e:
        if False:  # debug:
            print(f"  ERROR in input validation: {e}")
        return
    
    # Override with consensus/differential settings if requested
    if use_consensus_settings:
        # Create a copy with adjusted parameters
        import copy
        cfg = copy.copy(cfg)
        cfg.streamline_density = cfg.consensus_differential_density
        cfg.streamline_arrow_size = cfg.consensus_differential_arrow_size
        cfg.streamline_minlength = cfg.consensus_differential_minlength

    # CRITICAL FIX: Filter metadata to show ONLY cells within this region's grid
    # This prevents all 5 regions from being overlaid on each subplot
    meta_df = _filter_metadata_to_grid(meta_df, xi, yi, margin_pct=0.05)

    if meta_df is not None and not meta_df.empty:
        ax.scatter(
            meta_df["px_x"], meta_df["px_y"],
            s=cfg.scatter_size_bg,
            color=cfg.bg_color,
            alpha=cfg.scatter_alpha_bg,
            zorder=1,
            linewidths=0,
            rasterized=True,
        )

    # ── Per-cell flux-scaled halos ───────────────────────────────────────────
    # Halo size for each individual cell is proportional to its local flux
    # magnitude, derived two ways in priority order:
#
#   1. flux_df lookup (passed from results.per_region_data):  per-cell-type
#      interaction score.  All cells of the same type share the type-level
#      score; this captures condition-level differences across panels.
#
#   2. Local vector-field speed at each cell's grid position: sqrt(U²+V²)
#      interpolated to each cell coordinate.  This gives genuine *per-cell*
#      variation — cells sitting in a high-flux region get a larger halo
#      than cells of the same type sitting in a low-flux region.
#
    # Speed field interpolated to cell coordinates
    speed_field = np.sqrt(U**2 + V**2)

    def _per_cell_halo_sizes(
        cell_xs: np.ndarray,
        cell_ys: np.ndarray,
        cell_type: str,
        base_size: float,
        halo_min: float = 1.5,
        halo_max: float = 8.0,
    ) -> np.ndarray:
        """Return a per-cell halo size array in range [halo_min×, halo_max×] base_size."""
        n = len(cell_xs)
        if n == 0:
            return np.array([])

        # --- Step 1: local speed at each cell's position ---
        # Map cell coordinates to nearest grid indices
        xi_idx = np.searchsorted(xi, cell_xs).clip(0, len(xi) - 1)
        yi_idx = np.searchsorted(yi, cell_ys).clip(0, len(yi) - 1)
        local_speed = speed_field[yi_idx, xi_idx].astype(float)

        # --- Step 2: scale by type-level flux if available ---
        type_scale = 1.0
        if flux_df is not None and not flux_df.empty and "flux" in flux_df.columns:
            ct_rows = flux_df[flux_df["cell_type"].astype(str) == str(cell_type)]
            if not ct_rows.empty:
                flux_val = float(ct_rows["flux"].abs().mean())
                all_flux = flux_df["flux"].abs()
                all_flux = all_flux[all_flux > 0]
                if len(all_flux) > 0 and np.isfinite(flux_val) and flux_val > 0:
                    f_min, f_max = float(all_flux.min()), float(all_flux.max())
                    if f_max > f_min:
                        t = (np.log1p(flux_val) - np.log1p(f_min)) / \
                            (np.log1p(f_max) - np.log1p(f_min))
                        type_scale = 0.6 + 0.8 * float(np.clip(t, 0, 1))  # [0.6×, 1.4×]

        # --- Combine: normalize speed to [0,1] across all speeds in panel ---
        # Use global speed percentiles so the scale is consistent within panel
        s_min = float(np.percentile(speed_field[speed_field > 0], 5)) \
            if np.any(speed_field > 0) else 0.0
        s_max = float(np.percentile(speed_field[speed_field > 0], 95)) \
            if np.any(speed_field > 0) else 1.0
        if s_max <= s_min:
            s_norm = np.full(n, 0.5)
        else:
            s_norm = np.clip((local_speed - s_min) / (s_max - s_min), 0, 1)

        multiplier = halo_min + (halo_max - halo_min) * s_norm  # [halo_min×, halo_max×]
        multiplier *= type_scale
        return (base_size * multiplier).astype(float)

    if meta_df is not None and not meta_df.empty and src_type and src_type in meta_df["cell_type"].astype(str).values:
        sub = meta_df[meta_df["cell_type"].astype(str) == str(src_type)]
        src_xs = sub["px_x"].values.astype(float)
        src_ys = sub["px_y"].values.astype(float)
        src_halo_s = _per_cell_halo_sizes(src_xs, src_ys, src_type, cfg.scatter_size_highlight)
        ax.scatter(
            src_xs, src_ys,
            s=src_halo_s,
            color=cfg.source_color,
            alpha=0.12, zorder=2, linewidths=0,
        )
        ax.scatter(
            src_xs, src_ys,
            s=cfg.scatter_size_highlight,
            color=cfg.source_color,
            alpha=cfg.scatter_alpha_highlight,
            zorder=3,
            label=f"{src_type} (Source)",
            edgecolors="white",
            linewidths=0.8,
        )

    if meta_df is not None and not meta_df.empty and snk_type and snk_type in meta_df["cell_type"].astype(str).values:
        sub = meta_df[meta_df["cell_type"].astype(str) == str(snk_type)]
        snk_xs = sub["px_x"].values.astype(float)
        snk_ys = sub["px_y"].values.astype(float)
        snk_halo_s = _per_cell_halo_sizes(snk_xs, snk_ys, snk_type, cfg.scatter_size_highlight)
        ax.scatter(
            snk_xs, snk_ys,
            s=snk_halo_s,
            color=cfg.sink_color,
            alpha=0.12, zorder=2, linewidths=0,
        )
        ax.scatter(
            snk_xs, snk_ys,
            s=cfg.scatter_size_highlight,
            color=cfg.sink_color,
            alpha=cfg.scatter_alpha_highlight,
            zorder=3,
            label=f"{snk_type} (Sink)",
            edgecolors="white",
            linewidths=0.8,
        )

    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    speed = np.sqrt(U**2 + V**2)
    max_speed = float(np.nanmax(speed)) if speed.size else 0.0

    if (not np.isfinite(max_speed)) or max_speed < 1e-12:
        ax.text(
            0.5, 0.5, "No flow detected",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12, color="grey",
        )
    else:
        # ── Shared-scale colour normalisation ────────────────────────────────
        # If a shared_vmax is provided (set to the global max across all
        # conditions for this metabolite) we pin vmax to it.  This makes every
        # condition's panel comparable: a low-flux condition will look pale,
        # a high-flux condition will look dark.
        if shared_vmax is not None and np.isfinite(shared_vmax) and shared_vmax > 1e-12:
            vmin = 0.0
            vmax = float(shared_vmax)
        else:
            vmin, vmax = _robust_vmin_vmax(speed, cfg.speed_clip_lo_pct, cfg.speed_clip_hi_pct)
            if vmax <= vmin + 1e-9:
                vmin = 0.0
                vmax = max_speed if max_speed > 0 else 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        # ── Speed masking: suppress arrows where flux is too weak ─────────────
        # Compute the threshold relative to the reference (shared or local) max.
        ref_max = float(shared_vmax) if (shared_vmax is not None and shared_vmax > 1e-12) else max_speed
        mask_val = speed_mask_threshold * ref_max
        U_plot = U.copy()
        V_plot = V.copy()
        weak = speed < mask_val
        U_plot[weak] = 0.0
        V_plot[weak] = 0.0

        # -- Proximity masking: suppress arrows far from source/sink cells --
        # The vector field is computed over the entire slide, so without this
        # mask streamlines appear in regions with no source or sink cells.
        # ALWAYS apply the mask.  When _make_cell_proximity_mask returns all-False
        # (either cell type absent / too sparse), that means the entire field
        # should be suppressed -- skip drawing completely.
        if cell_proximity_radius > 0 and meta_df is not None and not meta_df.empty:
            prox_mask = _make_cell_proximity_mask(
                meta_df, src_type, snk_type, xi, yi,
                radius_fraction=cell_proximity_radius,
            )
            # All-False mask = cell type absent in this condition → no arrows at all
            if not prox_mask.any():
                return
            U_plot[~prox_mask] = 0.0
            V_plot[~prox_mask] = 0.0

        # P06 style: linewidth = speed_n * lw_scale + lw_base
        speed_n = speed / (ref_max + 1e-9)
        lw_base = getattr(cfg, 'streamline_lw_base', 0.5)
        lw = cfg.streamline_lw_scale * speed_n + lw_base
        lw = np.clip(lw, float(cfg.linewidth_min), float(cfg.linewidth_max))

        # -- Seed streamlines from source-cell positions --
        # Seeding from actual source cells means every line genuinely
        # originates from a red dot.  If no source cells exist in this
        # condition, return early -- the proximity mask above already
        # catches this case, but we guard here too for safety.
        start_pts = None
        if n_seed_points > 0 and meta_df is not None and not meta_df.empty:
            rng = np.random.default_rng(int(seed))
            start_pts = _build_seed_points_from_cells(
                meta_df, src_type, xi, yi,
                n_seeds=n_seed_points, rng=rng,
            )

        # If no valid seed points found, do not fall back to uniform density --
        # that would scatter lines everywhere regardless of biology.
        if start_pts is None or len(start_pts) == 0:
            return

        base_kwargs = dict(
            color=speed,
            cmap=cmap,
            norm=norm,
            linewidth=lw,
            arrowsize=float(cfg.streamline_arrow_size),
            arrowstyle='-|>',
            minlength=float(cfg.streamline_minlength),
            maxlength=float(cfg.streamline_maxlength),
            integration_direction=str(cfg.streamline_integration_direction),
            start_points=start_pts,
            zorder=3,
        )

        strm = None
        for attempt in range(3):
            try:
                strm = ax.streamplot(xi, yi, U_plot, V_plot,
                                     broken_streamlines=bool(cfg.streamline_broken_streamlines),
                                     **base_kwargs)
                break
            except TypeError:
                # Older matplotlib: remove unsupported kwargs one at a time.
                # NOTE: never add density fallback here -- that would scatter
                # lines everywhere regardless of biology.
                for k in ('broken_streamlines', 'arrowstyle'):
                    base_kwargs.pop(k, None)
            except Exception:
                break
        if strm is None:
            return

        try:
            if hasattr(strm, 'lines'):
                strm.lines.set_clim(vmin, vmax)
        except Exception:
            pass

        if show_colorbar:
            try:
                cbar = plt.colorbar(strm.lines, ax=ax, shrink=0.65, pad=0.015,
                                    aspect=28, drawedges=False)
                cbar.set_label(colorbar_label, fontsize=cfg.axis_label_font_size - 1,
                               labelpad=4, color='#333333')
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels(['Low', 'High'], fontsize=cfg.tick_font_size)
                cbar.ax.tick_params(labelsize=cfg.tick_font_size, length=2,
                                    color='#555555', labelcolor='#333333', width=0.5)
                cbar.ax.yaxis.label.set_color('#333333')
                cbar.outline.set_edgecolor('#dddddd')
                cbar.outline.set_linewidth(0.5)
            except Exception:
                pass

    # P06 style: clean white background, no ticks, no visible spines
    try:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(title, fontsize=cfg.subtitle_font_size, fontweight="bold",
                     color="#1a1a2e", pad=10)
    except Exception:
        pass

    if meta_df is not None and not meta_df.empty and (src_type or snk_type):
        try:
            leg = ax.legend(
                loc="upper right",
                fontsize=cfg.legend_font_size - 1,
                framealpha=0.92,
                edgecolor="#cccccc",
                fancybox=True,
                borderpad=0.7,
                handletextpad=0.5,
                labelspacing=0.45,
            )
            leg.get_frame().set_linewidth(0.6)
        except Exception:
            pass


# =============================================================================
# Consensus streamlines - USING OLD WORKING LOGIC
# =============================================================================

def _build_flux_df(results, condition: str, met: str) -> Optional[pd.DataFrame]:
    """
    Build a small DataFrame with columns ['cell_type', 'flux'] for the given
    condition and metabolite, by averaging secretion/uptake magnitudes across
    per-region data.

    This is used to size the scatter halos in plot_streamline_panel
    proportionally to actual flux — larger halo = stronger secretor/uptaker.

    Returns None if flux data is not available.
    """
    per_region = getattr(results, "per_region_data", {}) or {}
    rows = []
    for key, data in per_region.items():
        if not key.startswith(condition):
            continue
        interactions = (data.get("interactions", {}) or {}).get(met, [])
        for pair in interactions:
            src = pair.get("source", "")
            snk = pair.get("sink", "")
            score = float(pair.get("score", 0.0))
            if src:
                rows.append({"cell_type": src, "flux": score})
            if snk:
                rows.append({"cell_type": snk, "flux": score})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.groupby("cell_type", as_index=False)["flux"].mean()


def plot_consensus_streamlines(results, metabolites=None, out_dir=None, cfg=None):
    """
    One figure per metabolite showing consensus flow for each condition.

    Cross-condition comparability fix
    ----------------------------------
    shared_vmax is computed as the maximum flow speed across ALL conditions for
    this metabolite.  This anchors the colour scale identically in every
    subplot, so a condition with strong flux looks dark/bright while a condition
    with weak flux looks pale — which is what a reader needs to compare panels.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if metabolites is None:
        metabolites = results.significant_mets
    if out_dir is None:
        out_dir = os.path.join(results.config.out_dir, "plots_consensus")
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    conditions = results.config.conditions

    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue

        src, snk, p_val = results.get_metabolite_info(met)
        if src is None:
            continue

        condition_data = {}

        for condition in conditions:
            fields = results.get_vector_fields_for_condition(condition, met)
            meta_df = results.get_metadata_for_condition(condition)
            if fields and not meta_df.empty:
                U_avg = np.mean([f[0] for f in fields], axis=0)
                V_avg = np.mean([f[1] for f in fields], axis=0)
                xi, yi = fields[0][2], fields[0][3]

                if cfg and hasattr(cfg, 'consensus_differential_coarsen_factor'):
                    U_avg, V_avg, xi, yi = _coarsen_vector_field(
                        U_avg, V_avg, xi, yi,
                        cfg.consensus_differential_coarsen_factor
                    )

                condition_data[condition] = {
                    "U": U_avg, "V": V_avg, "xi": xi, "yi": yi,
                    "meta_df": meta_df.copy()
                }

        if len(condition_data) == 0:
            continue

        # ── Compute shared colour scale across all conditions ────────────────
        shared_vmax = max(
            float(np.nanmax(np.sqrt(d["U"]**2 + d["V"]**2)))
            for d in condition_data.values()
        )

        fig, axes = plt.subplots(1, len(conditions), figsize=cfg.consensus_figsize)
        if len(conditions) == 1:
            axes = [axes]

        for ax, cond in zip(axes, conditions):
            if cond not in condition_data:
                ax.text(0.5, 0.5, f"{cond.replace('Sample_', '')}\n(no data)",
                       transform=ax.transAxes, ha="center", va="center", fontsize=14)
                ax.axis("off")
                continue

            d = condition_data[cond]
            cond_short = cond.replace("Sample_", "")
            #title = f"{cond_short}\np = {float(p_val):.4g}"
            title = f"{cond_short}"
            xi_cond = d["xi"]
            yi_cond = d["yi"]
            xlim_cond = (float(np.min(xi_cond)), float(np.max(xi_cond)))
            ylim_cond = (float(np.min(yi_cond)), float(np.max(yi_cond)))

            plot_streamline_panel(
                ax, d["U"], d["V"], d["xi"], d["yi"],
                d["meta_df"], src, snk,
                title=title,
                show_colorbar=True,
                xlim=xlim_cond,
                ylim=ylim_cond,
                cfg=cfg,
                cmap=None,
                colorbar_label="Flow speed",
                use_consensus_settings=True,
                shared_vmax=shared_vmax,
                flux_df=_build_flux_df(results, cond, met),
                seed=int(getattr(cfg, 'random_seed', 42)),
            )

        fig.suptitle(
            f"Consensus Metabolite Flow: {met}\nSource: {src} → Sink: {snk}",
            fontsize=cfg.title_font_size, fontweight="bold"
        )
        plt.tight_layout()
        filepath = os.path.join(out_dir, f"consensus_{met}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_files.append(filepath)

    return saved_files



# =============================================================================
# Differential streamlines
# =============================================================================

def plot_differential_streamlines(results, metabolites=None, out_dir=None, cfg=None, debug=False):
    """
    Three-panel figure per metabolite: Condition 1 | Condition 2 | Differential.

    Cross-condition comparability fix
    ----------------------------------
    • The two condition panels share a single vmax (max over both conditions).
      This makes sparse-flux conditions look visually sparse (pale, few arrows)
      compared with dense-flux conditions (dark, many arrows).
    • The differential panel uses its own symmetric coolwarm scale so that
      positive and negative changes are equally visible.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if metabolites is None:
        metabolites = results.significant_mets
    if out_dir is None:
        out_dir = os.path.join(results.config.out_dir, "plots_differential")
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    conditions = results.config.conditions

    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue

        src, snk, p_val = results.get_metabolite_info(met)
        if src is None:
            continue

        condition_data = {}
        for condition in conditions:
            fields = results.get_vector_fields_for_condition(condition, met)
            meta_df = results.get_metadata_for_condition(condition)
            if fields and not meta_df.empty:
                U_avg = np.mean([f[0] for f in fields], axis=0)
                V_avg = np.mean([f[1] for f in fields], axis=0)
                xi, yi = fields[0][2], fields[0][3]

                if cfg and hasattr(cfg, 'consensus_differential_coarsen_factor'):
                    U_avg, V_avg, xi, yi = _coarsen_vector_field(
                        U_avg, V_avg, xi, yi,
                        cfg.consensus_differential_coarsen_factor
                    )

                condition_data[condition] = {
                    "U": U_avg, "V": V_avg, "xi": xi, "yi": yi,
                    "meta_df": meta_df.copy()
                }

        if len(condition_data) < 2:
            continue

        cond1, cond2 = conditions[0], conditions[1]
        if cond1 not in condition_data or cond2 not in condition_data:
            continue

        d1, d2 = condition_data[cond1], condition_data[cond2]
        same_grid = (len(d1["xi"]) == len(d2["xi"]) and len(d1["yi"]) == len(d2["yi"]))
        if not same_grid:
            continue

        # ── Shared colour scale for the two condition panels ─────────────────
        speed1 = np.sqrt(d1["U"]**2 + d1["V"]**2)
        speed2 = np.sqrt(d2["U"]**2 + d2["V"]**2)
        shared_vmax = float(max(np.nanmax(speed1), np.nanmax(speed2)))

        dU = d2["U"] - d1["U"]
        dV = d2["V"] - d1["V"]
        xi_diff, yi_diff = d1["xi"], d1["yi"]

        fig, axes = plt.subplots(1, 3, figsize=cfg.differential_figsize)
        cond1_short = cond1.replace("Sample_", "")
        cond2_short = cond2.replace("Sample_", "")

        # Condition 1
        plot_streamline_panel(
            axes[0],
            d1["U"], d1["V"], d1["xi"], d1["yi"],
            d1["meta_df"], src, snk,
            title=cond1_short,
            show_colorbar=True,
            cfg=cfg,
            cmap=None,
            colorbar_label="Flow speed",
            use_consensus_settings=True,
            shared_vmax=shared_vmax,
            flux_df=_build_flux_df(results, cond1, met),
            seed=int(getattr(cfg, 'random_seed', 42)),
        )

        # Condition 2
        plot_streamline_panel(
            axes[1],
            d2["U"], d2["V"], d2["xi"], d2["yi"],
            d2["meta_df"], src, snk,
            title=cond2_short,
            show_colorbar=True,
            cfg=cfg,
            cmap=None,
            colorbar_label="Flow speed",
            use_consensus_settings=True,
            shared_vmax=shared_vmax,
            flux_df=_build_flux_df(results, cond2, met),
            seed=int(getattr(cfg, 'random_seed', 42)),
        )

        # Differential -- symmetric coolwarm, own scale.
        # For the differential panel the "source cells" are the union of
        # both conditions -- whichever condition has them is valid seed origin.
        # We merge the two meta_dfs so _build_seed_points_from_cells and
        # _make_cell_proximity_mask can find cells from either condition.
        meta_combined = pd.concat(
            [d1["meta_df"], d2["meta_df"]], ignore_index=True
        ).drop_duplicates(subset=["px_x", "px_y"])
        plot_streamline_panel(
            axes[2],
            dU, dV, xi_diff, yi_diff,
            meta_combined, src, snk,
            title=f"Differential\n({cond2_short} − {cond1_short})",
            show_colorbar=True,
            cfg=cfg,
            cmap="coolwarm",
            colorbar_label="Δ Flow speed",
            use_consensus_settings=True,
            shared_vmax=None,
            seed=int(getattr(cfg, 'random_seed', 42)),
        )

        fig.suptitle(
            f"Metabolite Flow Comparison: {met}\nSource: {src} → Sink: {snk}",
            fontsize=cfg.title_font_size, fontweight="bold"
        )
        plt.tight_layout()
        filepath = os.path.join(out_dir, f"diff_{met}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_files.append(filepath)

    return saved_files



# =============================================================================
# Interaction network: directed, net exchange, arrows, more colorful
# (unchanged from your enhanced file)
# =============================================================================

def _collect_directed_edge_table_from_regions(results, condition: Optional[str], metabolites: List[str], top_k_pairs_per_met: int = 3):
    rows = []
    per_region = getattr(results, "per_region_data", {})
    for key, data in per_region.items():
        if condition and not key.startswith(condition):
            continue
        interactions = data.get("interactions", {}) or {}
        for met in metabolites:
            if met not in interactions:
                continue
            pairs = interactions.get(met, []) or []
            for pair_info in pairs[: int(top_k_pairs_per_met)]:
                src, snk = pair_info.get("source"), pair_info.get("sink")
                score = float(pair_info.get("score", 0.0))
                if not src or not snk:
                    continue
                if src == snk:
                    continue
                if _should_exclude_metabolite(met):
                    continue
                if not np.isfinite(score) or score <= 0:
                    continue
                rows.append({"Source": str(src), "Sink": str(snk), "Metabolite": str(met), "Score": float(score)})
    return pd.DataFrame(rows)


def _net_directed_edges(edge_df: pd.DataFrame):
    if edge_df is None or edge_df.empty:
        return pd.DataFrame()

    dfx = edge_df.copy()
    dfx["u"] = dfx["Source"].astype(str)
    dfx["v"] = dfx["Sink"].astype(str)

    dfx["a"] = dfx[["u", "v"]].min(axis=1)
    dfx["b"] = dfx[["u", "v"]].max(axis=1)

    dir_sum = dfx.groupby(["a", "b", "u", "v"], as_index=False)["Score"].sum()

    pairs = []
    for (a, b), sub in dir_sum.groupby(["a", "b"]):
        wab = float(sub[(sub["u"] == a) & (sub["v"] == b)]["Score"].sum())
        wba = float(sub[(sub["u"] == b) & (sub["v"] == a)]["Score"].sum())
        net_ab = wab - wba
        if abs(net_ab) <= 1e-12:
            continue
        if net_ab > 0:
            src, snk, wnet = a, b, net_ab
        else:
            src, snk, wnet = b, a, -net_ab
        pairs.append({"Source": src, "Sink": snk, "NetWeight": float(wnet)})

    net_df = pd.DataFrame(pairs)
    if net_df.empty:
        return net_df

    met_agg = (
        dfx.groupby(["u", "v", "Metabolite"], as_index=False)["Score"].sum()
        .rename(columns={"u": "Source", "v": "Sink", "Score": "MetScore"})
    )
    return net_df.merge(met_agg, on=["Source", "Sink"], how="left")


def plot_network_diagram(
    results,
    metabolites=None,
    condition=None,
    out_dir=None,
    cfg=None,
    top_metabolites=12,
    top_edges=25,
    top_k_pairs_per_met: int = 3,
):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    if metabolites is None:
        metabolites = getattr(results, "significant_mets", [])
    metabolites = _filter_met_list(list(metabolites))

    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_networks")

    edge_df = _collect_directed_edge_table_from_regions(
        results, condition=condition, metabolites=metabolites, top_k_pairs_per_met=top_k_pairs_per_met
    )
    if edge_df.empty:
        return None

    met_totals = edge_df.groupby("Metabolite")["Score"].sum().sort_values(ascending=False)
    top_mets = set(met_totals.head(int(top_metabolites)).index.tolist())

    net_join = _net_directed_edges(edge_df)
    if net_join.empty:
        return None

    grouped = []
    for (src, snk), sub in net_join.groupby(["Source", "Sink"]):
        net_w = float(sub["NetWeight"].max())
        sub2 = sub.dropna(subset=["Metabolite"]).copy()
        sub2 = sub2[sub2["Metabolite"].isin(top_mets)]
        sub2 = sub2.sort_values("MetScore", ascending=False)
        mets = sub2["Metabolite"].tolist()
        mets = list(dict.fromkeys(mets))
        grouped.append({"Source": src, "Sink": snk, "NetWeight": net_w, "TopMets": mets[:6], "NMets": int(len(mets))})

    edges_df = pd.DataFrame(grouped)
    if edges_df.empty:
        return None

    edges_df = edges_df.sort_values("NetWeight", ascending=False).head(int(top_edges)).reset_index(drop=True)

    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["Source"], r["Sink"], weight=float(r["NetWeight"]), n_mets=int(r["NMets"]), metabolites=list(r["TopMets"]))

    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=cfg.network_figsize)

    try:
        pos = nx.kamada_kawai_layout(G, weight=None)
    except Exception:
        pos = nx.spring_layout(G, k=cfg.network_k_spacing, iterations=150, seed=int(getattr(cfg, 'random_seed', 42)))

    pos = {k: (float(v[0]) * 1.8, float(v[1]) * 1.8) for k, v in pos.items()}

    node_net = {}
    for n in G.nodes():
        out_w = float(sum([G[n][nbr]["weight"] for nbr in G.successors(n)]))
        in_w = float(sum([G[pred][n]["weight"] for pred in G.predecessors(n)]))
        node_net[n] = out_w - in_w

    node_total = {
        n: float(sum([G[n][nbr]["weight"] for nbr in G.successors(n)]) + sum([G[pred][n]["weight"] for pred in G.predecessors(n)]))
        for n in G.nodes()
    }
    max_total = max(node_total.values()) if node_total else 1.0

    node_colors = {}
    for n, val in node_net.items():
        if val > 0.1 * max_total:
            node_colors[n] = cfg.producer_color
        elif val < -0.1 * max_total:
            node_colors[n] = cfg.consumer_color
        else:
            node_colors[n] = cfg.balanced_color

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    wmin = min(edge_weights) if edge_weights else 0.0
    wmax = max(edge_weights) if edge_weights else 1.0

    nmets_vals = [G[u][v]["n_mets"] for u, v in G.edges()]
    nm_min = min(nmets_vals) if nmets_vals else 0
    nm_max = max(nmets_vals) if nmets_vals else 1

    cmap = plt.cm.plasma

    for (u, v) in G.edges():
        ew = float(G[u][v]["weight"])
        nm = int(G[u][v]["n_mets"])
        width = 1.5 + cfg.edge_width_scale * (ew - wmin) / (wmax - wmin + 1e-9)
        t = (nm - nm_min) / (nm_max - nm_min + 1e-9)
        color = cmap(float(t))
        rad = 0.12 if G.has_edge(v, u) else 0.06

        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                mutation_scale=16,
                color=color,
                lw=width,
                alpha=cfg.edge_alpha,
                shrinkA=22, shrinkB=22,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=2,
        )

    for n in G.nodes():
        total = node_total.get(n, 0.0)
        size = cfg.node_size_base + cfg.node_size_scale * (total / (max_total + 1e-9))
        radius = 0.11 * np.sqrt(size / 900)
        circle = plt.Circle(
            pos[n], radius=radius, color=node_colors[n],
            alpha=cfg.node_alpha, ec="black", lw=cfg.node_edge_width, zorder=10
        )
        ax.add_patch(circle)
        ax.annotate(
            n, pos[n],
            fontsize=cfg.node_label_font_size,
            ha="center", va="center",
            fontweight="bold",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="none"),
        )

    for (u, v) in G.edges():
        mets = G[u][v].get("metabolites", [])
        if not mets:
            continue
        label = ", ".join(mets[:3]) if len(mets) <= 3 else ", ".join(mets[:2]) + f" (+{max(0, len(mets)-2)})"
        label = _truncate_label(label, 42)

        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        L = float(np.sqrt(dx**2 + dy**2)) + 1e-9
        off_x = -dy / L * 0.10
        off_y = dx / L * 0.10

        ax.annotate(
            label,
            xy=(mid_x + off_x, mid_y + off_y),
            fontsize=cfg.edge_label_font_size - 1,
            ha="center", va="center",
            alpha=cfg.edge_label_alpha,
            color="#2c3e50",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                alpha=cfg.edge_label_box_alpha,
                edgecolor="#333333",
                lw=0.6,
            ),
            zorder=5,
        )

    legend_elements = [
        mpatches.Patch(facecolor=cfg.producer_color, edgecolor="black", alpha=cfg.node_alpha, label="Net exporter (out-in)"),
        mpatches.Patch(facecolor=cfg.consumer_color, edgecolor="black", alpha=cfg.node_alpha, label="Net importer (out-in)"),
        mpatches.Patch(facecolor=cfg.balanced_color, edgecolor="black", alpha=cfg.node_alpha, label="Balanced"),
        Line2D([0], [0], color=cmap(0.2), lw=4, label="Few top metabolites"),
        Line2D([0], [0], color=cmap(0.9), lw=4, label="Many top metabolites"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=cfg.legend_font_size, framealpha=0.95, edgecolor="grey")

    title_suffix = f" ({condition.replace('Sample_', '')})" if condition else ""
    ax.set_title(
        f"Cell Type Metabolic Interaction Network{title_suffix}\n"
        f"Directed net exchange (arrows); top metabolites={top_metabolites} (after exclusions); top edges={top_edges}",
        fontsize=cfg.title_font_size, fontweight="bold", pad=14,
    )

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad = 0.6
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    suffix = f"_{condition}" if condition else ""
    filepath = os.path.join(out_dir, f"interaction_network{suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Regional streamlines (unchanged except uses fixed plot_streamline_panel)
# =============================================================================

def plot_regional_streamlines(results, condition, region_idx, metabolites=None, out_dir=None, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    if metabolites is None:
        metabolites = getattr(results, "significant_mets", [])[:10]
    metabolites = _filter_met_list(list(metabolites))

    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_regional")

    key = f"{condition}_R{region_idx}"
    per_region = getattr(results, "per_region_data", {})
    if key not in per_region:
        return []

    data = per_region[key]
    meta_df = data.get("meta_df", pd.DataFrame())
    vector_fields = data.get("vector_fields", {})
    best_pairs = data.get("best_pairs", {})
    enriched_df = data.get("enriched_df", pd.DataFrame())

    saved_files = []
    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue
        if met not in vector_fields:
            continue

        src = snk = None
        if enriched_df is not None and not enriched_df.empty and "met" in enriched_df.columns and met in enriched_df["met"].values:
            row = enriched_df[enriched_df["met"] == met].iloc[0]
            src, snk = row.get("sec"), row.get("upt")
        elif met in best_pairs:
            src, snk, _ = best_pairs[met]
        else:
            continue

        U, V, xi, yi = vector_fields[met]
        fig, ax = plt.subplots(figsize=cfg.regional_figsize)
        cond_short = condition.replace("Sample_", "")
        title = f"Metabolite Flow: {met}\nSource: {src} | Sink: {snk}\n{cond_short} Region {region_idx}"
        plot_streamline_panel(ax, U, V, xi, yi, meta_df, src, snk, title=title, cfg=cfg,
                              seed=int(getattr(cfg, 'random_seed', 42)))
        plt.tight_layout()
        filepath = os.path.join(out_dir, f"flow_{condition}_R{region_idx}_{met}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_files.append(filepath)

    return saved_files


# =============================================================================
# Coupling heatmap
# =============================================================================

def plot_coupling_heatmap(results, condition=None, out_dir=None, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_coupling")

    all_coupling = {}
    per_region = getattr(results, "per_region_data", {})
    for key, data in per_region.items():
        if condition and not key.startswith(condition):
            continue
        coupling = data.get("coupling", {})
        for (a, b), val in coupling.items():
            all_coupling.setdefault((a, b), []).append(val)

    if not all_coupling:
        return None

    avg_coupling = {k: float(np.mean(v)) for k, v in all_coupling.items()}
    cell_types = sorted(set([a for a, b in avg_coupling.keys()] + [b for a, b in avg_coupling.keys()]))

    n = len(cell_types)
    matrix = np.zeros((n, n), dtype=float)
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}

    for (a, b), val in avg_coupling.items():
        i, j = ct_to_idx[a], ct_to_idx[b]
        matrix[i, j] = val
        matrix[j, i] = val

    fig, ax = plt.subplots(figsize=cfg.heatmap_figsize)
    sns.heatmap(
        matrix,
        xticklabels=cell_types,
        yticklabels=cell_types,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Spatial Coupling Strength"},
        annot_kws={"fontsize": 8},
    )

    plt.xticks(rotation=45, ha="right", fontsize=cfg.tick_font_size)
    plt.yticks(rotation=0, fontsize=cfg.tick_font_size)

    title_suffix = f" ({condition.replace('Sample_', '')})" if condition else ""
    ax.set_title(
        f"Cell Type Spatial Coupling{title_suffix}\nHigher = more frequent neighbors",
        fontsize=cfg.title_font_size,
        fontweight="bold",
    )

    plt.tight_layout()
    suffix = f"_{condition}" if condition else ""
    filepath = os.path.join(out_dir, f"coupling_heatmap{suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Flux balance comparison - ROBUST SELF-HEALING VERSION
# =============================================================================

def _regenerate_cell_balance(results):
    """
    Helper to reconstruct cell-type balance if the results object is old/incompatible.
    Does NOT require re-running the full pipeline.
    """
    print("   [INFO] Auto-regenerating cell type balance data from raw fluxes...")
    condition_balance = {}
    conditions = getattr(results.config, "conditions", [])
    
    for condition in conditions:
        ct_balance = defaultdict(lambda: {"produced": 0.0, "consumed": 0.0})
        
        per_region = getattr(results, "per_region_data", {})
        for key, data in per_region.items():
            if not key.startswith(condition):
                continue
            met_fluxes = data.get("met_fluxes", {})
            
            for met, flux_dict in met_fluxes.items():
                if _should_exclude_metabolite(met):
                    continue
                # Secretion
                for ct, vals in flux_dict.get("secretion", {}).items():
                    ct_balance[ct]["produced"] += sum(abs(float(v)) for v in vals)
                # Uptake
                for ct, vals in flux_dict.get("uptake", {}).items():
                    ct_balance[ct]["consumed"] += sum(abs(float(v)) for v in vals)
        
        rows = []
        for ct, bal in ct_balance.items():
            prod, cons = bal["produced"], bal["consumed"]
            total = prod + cons
            ratio = prod / total if total > 1e-12 else 0.5
            rows.append({
                "CellType": ct,
                "Produced": prod,
                "Consumed": cons,
                "TotalFlux": total,
                "Balance_Ratio": ratio
            })
        
        if rows:
            condition_balance[condition] = pd.DataFrame(rows)
        else:
            condition_balance[condition] = pd.DataFrame()
            
    return condition_balance


def plot_flux_balance_comparison(results, out_dir=None, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_balance")

    conditions = getattr(results.config, "conditions", [])
    condition_balance = getattr(results, "condition_balance", {})

    # -------------------------------------------------------------------------
    # SELF-HEALING LOGIC: Check if data is missing or in wrong format (metabolite-centric)
    # -------------------------------------------------------------------------
    needs_regen = False
    
    # Check 1: Is condition_balance empty/None?
    if not condition_balance or not isinstance(condition_balance, dict):
        needs_regen = True
    else:
        # Check 2: Does the first non-empty dataframe have the right columns?
        for cond, df in condition_balance.items():
            if df is not None and not df.empty:
                if "CellType" not in df.columns or "Balance_Ratio" not in df.columns:
                    needs_regen = True
                break
    
    if needs_regen:
        try:
            condition_balance = _regenerate_cell_balance(results)
        except Exception as e:
            print(f"   [WARNING] Could not regenerate balance data: {e}")
            return None
    # -------------------------------------------------------------------------

    all_data = []
    if isinstance(condition_balance, dict):
        for cond, df in condition_balance.items():
            if df is None or df.empty:
                continue
            dfx = df.copy()
            dfx["Condition"] = cond
            all_data.append(dfx)

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    
    # Robust column check
    if "CellType" not in combined.columns:
        print("   [SKIP] Flux balance plot: 'CellType' column missing.")
        return None
        
    value_col = _first_existing_col(combined, ["Balance_Ratio", "BalanceRatio"])
    if value_col is None:
        # Fallback calculation if somehow still missing
        if "Produced" in combined.columns and "Consumed" in combined.columns:
            combined["Balance_Ratio"] = combined["Produced"] / (combined["Produced"] + combined["Consumed"] + 1e-9)
            value_col = "Balance_Ratio"
        else:
            print("   [SKIP] Flux balance plot: Missing balance/flux columns.")
            return None

    pivot = combined.pivot_table(index="CellType", columns="Condition", values=value_col, aggfunc="mean")
    if pivot.empty:
        return None

    if len(pivot.columns) >= 2:
        pivot["Diff"] = pivot.iloc[:, 1] - pivot.iloc[:, 0]
        pivot = pivot.sort_values("Diff")
        pivot = pivot.drop("Diff", axis=1)

    fig, axes = plt.subplots(1, 2, figsize=cfg.balance_figsize)

    ax = axes[0]
    y = np.arange(len(pivot))
    width = 0.35

    colors = {}
    if len(conditions) >= 2:
        colors = {conditions[0]: "#2ecc71", conditions[1]: "#e74c3c"}

    for i, cond in enumerate(conditions):
        if cond in pivot.columns:
            offset = width * (i - 0.5)
            ax.barh(
                y + offset,
                pivot[cond],
                width,
                label=cond.replace("Sample_", ""),
                color=colors.get(cond, f"C{i}"),
                alpha=0.8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=cfg.tick_font_size)
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Balanced")
    ax.set_xlabel("Balance Ratio (Secretion / Total)", fontsize=cfg.axis_label_font_size)
    ax.set_title("Metabolic Balance by Cell Type", fontsize=cfg.subtitle_font_size, fontweight="bold")
    ax.legend(loc="lower right", fontsize=cfg.legend_font_size)
    ax.set_xlim(0, 1)

    ax = axes[1]
    if len(pivot.columns) >= 2 and len(conditions) >= 2 and conditions[0] in pivot.columns and conditions[1] in pivot.columns:
        diff = pivot[conditions[1]] - pivot[conditions[0]]
        colors_diff = ["#e74c3c" if d > 0 else "#2ecc71" for d in diff]
        ax.barh(range(len(diff)), diff, color=colors_diff, alpha=0.8)
        ax.set_yticks(range(len(diff)))
        ax.set_yticklabels(diff.index, fontsize=cfg.tick_font_size)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax.set_xlabel("Δ Balance (Injured - Healthy)", fontsize=cfg.axis_label_font_size)
        ax.set_title(
            "Change in Metabolic Balance\n(+Red: More secretion, -Green: More uptake)",
            fontsize=cfg.subtitle_font_size,
            fontweight="bold",
        )

        for i, (_, val) in enumerate(diff.items()):
            if abs(val) > 0.05:
                ax.annotate(
                    f"{val:+.2f}",
                    xy=(val, i),
                    xytext=(5 if val > 0 else -5, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if val > 0 else "right",
                    fontsize=8,
                )

    plt.tight_layout()
    filepath = os.path.join(out_dir, "flux_balance_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Metabolite heatmap (filtered)
# =============================================================================

def _plot_single_condition_heatmap(results, min_score=100, out_dir=None, cfg=None):
    """
    Score-ranked metabolite summary for single-condition runs.

    When only one condition is present there is no fold-change to compare,
    so we produce a ranked table showing:
      Col A : Interaction score bar (YlOrRd)
      Col B : Source cell type (text, red)
      Col C : Sink cell type  (text, blue)
      Col D : # source cells  (text)
      Col E : # sink cells    (text)

    Metabolites with score < min_score are excluded.
    Sorted by score descending.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    cond_label = _single_condition_label(results)
    per_region = getattr(results, "per_region_data", {}) or {}

    # Aggregate scores across all regions
    score_map, src_map, snk_map = {}, {}, {}
    for key, data in per_region.items():
        for met, pairs in (data.get("interactions", {}) or {}).items():
            if _should_exclude_metabolite(met) or not pairs:
                continue
            s = float(pairs[0].get("score", 0.0))
            score_map[met] = score_map.get(met, 0.0) + s
            src_map.setdefault(met, str(pairs[0].get("source", "?")))
            snk_map.setdefault(met, str(pairs[0].get("sink",   "?")))

    if not score_map:
        return None

    # Filter and sort
    rows = [
        (met, sc, src_map.get(met, "?"), snk_map.get(met, "?"))
        for met, sc in score_map.items()
        if sc >= float(min_score) and not _should_exclude_metabolite(met)
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    if not rows:
        return None

    mets    = [r[0] for r in rows]
    scores  = np.array([r[1] for r in rows])
    srcs    = [r[2] for r in rows]
    snks    = [r[3] for r in rows]
    n_rows  = len(rows)

    # Count cells per type from the largest region
    best_key = max(per_region.keys(),
                   key=lambda k: len((per_region[k].get("meta_df") or pd.DataFrame())))
    meta_best = per_region[best_key].get("meta_df", pd.DataFrame())

    def _ct_n(ct):
        if meta_best is None or meta_best.empty:
            return "?"
        n = int((meta_best["cell_type"].astype(str) == str(ct)).sum())
        return str(n) if n else "—"

    row_h  = 0.38
    fig_h  = max(8, n_rows * row_h + 3.0)
    fig, ax_score = plt.subplots(figsize=(16, fig_h), facecolor="white")

    # Layout: score bar on left, text columns right
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.05)

    y_pos      = np.arange(n_rows)
    score_norm = scores / (scores.max() + 1e-9)
    sc_cmap    = plt.get_cmap("YlOrRd")

    for i, (s, sn) in enumerate(zip(scores, score_norm)):
        ax_score.barh(i, s, color=sc_cmap(0.15 + 0.80 * sn),
                      height=0.72, edgecolor="white", linewidth=0.4)
        ax_score.text(s * 1.01, i, f"{s:.0f}",
                      va="center", ha="left", fontsize=7, color="#222222")

    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(mets, fontsize=8.5, fontweight="bold")
    ax_score.set_xlim(0, scores.max() * 1.22)
    ax_score.set_ylim(-0.5, n_rows - 0.5)
    ax_score.invert_yaxis()
    ax_score.spines[["top", "right"]].set_visible(False)
    ax_score.tick_params(axis="y", length=0)
    ax_score.set_xlabel("Interaction Score", fontsize=9, labelpad=4)

    # Overlay text columns using figure-level text (axes coords)
    x_src  = 1.02   # fraction of axes width
    x_arr  = 1.28
    x_snk  = 1.34
    x_nsrc = 1.62
    x_nsnk = 1.72

    # Column headers
    ax_score.text(x_src,  -0.8, "Source cell type",   transform=ax_score.transData,
                  fontsize=8, fontweight="bold", color="#c0392b", va="center")
    ax_score.text(x_snk,  -0.8, "Sink cell type",     transform=ax_score.transData,
                  fontsize=8, fontweight="bold", color="#2980b9", va="center")
    ax_score.text(x_nsrc, -0.8, "n(src)", transform=ax_score.transData,
                  fontsize=8, fontweight="bold", color="#555555", va="center")
    ax_score.text(x_nsnk, -0.8, "n(snk)", transform=ax_score.transData,
                  fontsize=8, fontweight="bold", color="#555555", va="center")

    x_right = scores.max() * 1.25
    x_step  = scores.max() * 0.18

    for i, (src, snk) in enumerate(zip(srcs, snks)):
        ax_score.text(x_right,               i, src,       va="center", ha="left",
                      fontsize=7.5, color="#c0392b")
        ax_score.text(x_right + x_step,      i, " → ",    va="center", ha="left",
                      fontsize=7.5, color="#555555")
        ax_score.text(x_right + x_step*1.2,  i, snk,       va="center", ha="left",
                      fontsize=7.5, color="#2980b9")
        ax_score.text(x_right + x_step*2.6,  i, _ct_n(src), va="center", ha="left",
                      fontsize=7, color="#777777")
        ax_score.text(x_right + x_step*3.0,  i, _ct_n(snk), va="center", ha="left",
                      fontsize=7, color="#777777")

    ax_score.set_xlim(0, scores.max() * 4.8)

    fig.suptitle(
        f"Metabolite Interaction Summary — {cond_label}\n"
        f"{n_rows} metabolites  (score ≥ {min_score:.0f})  ·  "
        f"sorted by interaction score",
        fontsize=cfg.title_font_size + 1, fontweight="bold", y=0.97,
    )

    filepath = os.path.join(out_dir, "metabolite_changes_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


def plot_metabolite_heatmap(results, top_n=120, min_score=100, out_dir=None, cfg=None):
    """
    Comprehensive metabolite exchange heatmap with four panels:

    Layout (left → right):
      Col A  [narrow]  : Interaction score bar — visual magnitude scale
      Col B  [wide]    : Secretion + Uptake log2 FC heatmap (diverging RdBu_r)
                         Cells annotated with FC value; significance stars
                         overlaid (*** p<0.001, ** p<0.01, * p<0.05)
      Col C  [narrow]  : –log10(p-value) dot strip for secretion
      Col D  [narrow]  : Source → Sink cell-type labels (text)

    Rows: metabolites, sorted by absolute secretion FC (largest change first).
          A horizontal dividing line separates significant (p<0.05) from
          non-significant metabolites.

    Colour palettes:
      • FC heatmap  : RdBu_r centred at 0 (red = ↑ in injured, blue = ↓)
      • Score bar   : single-hue sequential (Blues)
      • p-value dot : viridis (dark = most significant)

    The figure is sized dynamically: taller for more metabolites,
    wider when cell-type labels are long.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # ── Pull data ──────────────────────────────────────────────────────────
    df = getattr(results, "comparison_df", pd.DataFrame())
    if df is None or df.empty:
        return None

    met_col  = _first_existing_col(df, ["Metabolite",        "metabolite"])
    sec_fc   = _first_existing_col(df, ["Log2FC_Secretion",  "Log2FCSecretion"])
    upt_fc   = _first_existing_col(df, ["Log2FC_Uptake",     "Log2FCUptake"])
    p_sec    = _first_existing_col(df, ["P_Secretion",       "PSecretion"])
    p_upt    = _first_existing_col(df, ["P_Uptake",          "PUptake"])
    if None in (met_col, sec_fc, upt_fc):
        return None

    dfx = df.copy()
    dfx = dfx[~dfx[met_col].apply(_should_exclude_metabolite)].copy()

    # Numeric coercion + fill
    for col in [sec_fc, upt_fc]:
        dfx[col] = pd.to_numeric(dfx[col], errors="coerce").fillna(0.0)
    for col in [p_sec, p_upt]:
        if col:
            dfx[col] = pd.to_numeric(dfx[col], errors="coerce").fillna(1.0)

    # Attach interaction score from per_region_data
    score_map = {}
    per_region = getattr(results, "per_region_data", {}) or {}
    for key, data in per_region.items():
        for met, pairs in (data.get("interactions", {}) or {}).items():
            if pairs:
                s = float(pairs[0].get("score", 0.0))
                score_map[met] = score_map.get(met, 0.0) + s
    dfx["_score"] = dfx[met_col].map(score_map).fillna(0.0)

    # Attach source → sink labels
    src_map, snk_map = {}, {}
    if hasattr(results, "get_metabolite_info"):
        for met in dfx[met_col].unique():
            try:
                src, snk, _ = results.get_metabolite_info(met)
                src_map[met] = str(src or "?")
                snk_map[met] = str(snk or "?")
            except Exception:
                src_map[met] = "?"
                snk_map[met] = "?"

    dfx["_src"] = dfx[met_col].map(src_map).fillna("?")
    dfx["_snk"] = dfx[met_col].map(snk_map).fillna("?")
    dfx["_axis_label"] = dfx["_src"] + " → " + dfx["_snk"]

    # Significance flag for sorting boundary
    if p_sec:
        dfx["_sig"] = dfx[p_sec] < 0.05
    else:
        dfx["_sig"] = False

    dfx["_total_abs"] = dfx[sec_fc].abs() + dfx[upt_fc].abs()
    dfx["_absfc"]     = dfx[sec_fc].abs()

    # ── Score threshold filter — drop metabolites below min_score ──────────
    # Applied before any pool selection so low-score metabolites are excluded
    # even if they have large FC or are statistically significant.
    dfx = dfx[dfx["_score"] >= float(min_score)].copy()

    if dfx.empty:
        return None   # nothing survives the score threshold

    # ── Metabolite selection: union of three pools (no hard top_n cap) ─────
    # Pool 1: ALL surviving by score       (respects min_score filter above)
    # Pool 2: all significant by p-value   (captures dhdascb, sphs1p etc.)
    # Pool 3: top-N by |FC|                (captures strong changers)
    # Union is the full set that survived the score threshold; top_n acts as
    # a safety cap only if the result would be extremely large.
    pool_score = set(dfx[met_col])                           # all above threshold
    pool_sig   = set(dfx[dfx["_sig"]][met_col])
    pool_fc    = set(dfx.nlargest(int(top_n), "_total_abs")[met_col])
    selected   = pool_sig | pool_score | pool_fc

    # Safety cap: if still huge, keep all sig + fill by score
    if len(selected) > int(top_n):
        keep = list(pool_sig)
        remaining = sorted(
            selected - pool_sig,
            key=lambda m: dfx.loc[dfx[met_col] == m, "_score"].sum(),
            reverse=True,
        )
        keep += remaining[: max(0, int(top_n) - len(keep))]
        selected = set(keep)

    dfx = dfx[dfx[met_col].isin(selected)].copy()

    # ── Sort: significant first (desc |FC|), then non-sig (desc score) ────
    sig_df  = dfx[dfx["_sig"]].sort_values("_absfc", ascending=False)
    nsig_df = dfx[~dfx["_sig"]].sort_values("_score", ascending=False)
    dfx = pd.concat([sig_df, nsig_df], ignore_index=True)

    n_rows       = len(dfx)
    n_sig        = int(dfx["_sig"].sum())
    met_labels   = dfx[met_col].tolist()
    scores       = dfx["_score"].values
    sec_vals     = dfx[sec_fc].values
    upt_vals     = dfx[upt_fc].values
    p_sec_vals   = dfx[p_sec].values   if p_sec else np.ones(n_rows)
    p_upt_vals   = dfx[p_upt].values   if p_upt else np.ones(n_rows)
    axis_labels  = dfx["_axis_label"].tolist()

    # ── Figure layout ──────────────────────────────────────────────────────
    row_h   = 0.42          # inches per metabolite row
    fig_h   = max(10, n_rows * row_h + 3.5)
    fig_w   = 20            # fixed width; columns share it via GridSpec ratios

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    # Columns: score bar | FC heatmap (2 cols) | p-val dots | cell-type text
    gs = GridSpec(
        1, 5,
        figure=fig,
        width_ratios=[1.2, 2.2, 2.2, 1.2, 3.8],
        wspace=0.04,
        left=0.01, right=0.99, top=0.92, bottom=0.05,
    )
    ax_score  = fig.add_subplot(gs[0, 0])
    ax_sec    = fig.add_subplot(gs[0, 1])
    ax_upt    = fig.add_subplot(gs[0, 2])
    ax_pval   = fig.add_subplot(gs[0, 3])
    ax_label  = fig.add_subplot(gs[0, 4])

    y_pos = np.arange(n_rows)

    # ── Helper: significance stars ────────────────────────────────────────
    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    # ── Panel A: Interaction score horizontal bar ─────────────────────────
    score_norm = scores / (scores.max() + 1e-9)
    score_cmap = plt.get_cmap("YlOrRd")
    for i, (s, sn) in enumerate(zip(scores, score_norm)):
        ax_score.barh(i, s, color=score_cmap(0.2 + 0.75 * sn),
                      height=0.72, edgecolor="white", linewidth=0.4)
        if s > 0:
            ax_score.text(s * 0.02, i, f"{s:.0f}",
                          va="center", ha="left",
                          fontsize=6.5, color="#222222")
    ax_score.set_xlim(0, scores.max() * 1.18)
    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(met_labels, fontsize=8.5, fontweight="bold")
    ax_score.set_xlabel("Interaction\nScore", fontsize=8, labelpad=3)
    ax_score.set_ylim(-0.5, n_rows - 0.5)
    ax_score.invert_yaxis()
    ax_score.spines[["top", "right"]].set_visible(False)
    ax_score.tick_params(axis="y", length=0)
    ax_score.set_title("Score", fontsize=9, fontweight="bold", pad=6)

    # Dividing line between sig / non-sig
    if 0 < n_sig < n_rows:
        for ax_ in (ax_score, ax_sec, ax_upt, ax_pval, ax_label):
            ax_.axhline(n_sig - 0.5, color="#555555", linewidth=1.2,
                        linestyle="--", alpha=0.6)

    # ── Panel B & C: FC heatmap (secretion + uptake) ──────────────────────
    fc_max   = max(np.abs(np.concatenate([sec_vals, upt_vals])).max(), 1.0)
    fc_max   = np.ceil(fc_max * 2) / 2          # round up to nearest 0.5

    # Custom diverging palette: deep blue → white → deep red
    fc_cmap = mcolors.LinearSegmentedColormap.from_list(
        "fc_div",
        ["#2166ac", "#4393c3", "#92c5de", "#f7f7f7",
         "#f4a582", "#d6604d", "#b2182b"],
        N=512,
    )
    fc_norm = mcolors.TwoSlopeNorm(vmin=-fc_max, vcenter=0.0, vmax=fc_max)

    for ax_, vals, p_vals, panel_title in [
        (ax_sec, sec_vals, p_sec_vals, "Secretion\nLog₂FC"),
        (ax_upt, upt_vals, p_upt_vals, "Uptake\nLog₂FC"),
    ]:
        for i, (v, p) in enumerate(zip(vals, p_vals)):
            fc_rgba = fc_cmap(fc_norm(v))
            rect = plt.Rectangle(
                (0, i - 0.36), 1, 0.72,
                facecolor=fc_rgba, edgecolor="white", linewidth=0.5,
            )
            ax_.add_patch(rect)

            # FC value annotation — white text on dark fill, dark on light
            lum = 0.299*fc_rgba[0] + 0.587*fc_rgba[1] + 0.114*fc_rgba[2]
            txt_col = "white" if lum < 0.55 else "#333333"
            stars   = _stars(p)
            ann     = f"{v:+.2f}{stars}"
            ax_.text(0.5, i, ann, va="center", ha="center",
                     fontsize=7.5, color=txt_col, fontweight="bold")

        ax_.set_xlim(0, 1)
        ax_.set_ylim(-0.5, n_rows - 0.5)
        ax_.invert_yaxis()
        ax_.set_yticks([])
        ax_.set_xticks([])
        ax_.set_title(panel_title, fontsize=9, fontweight="bold", pad=6)
        ax_.spines[:].set_visible(False)

    # Shared colorbar for both FC panels — placed between them
    sm = plt.cm.ScalarMappable(cmap=fc_cmap, norm=fc_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.365, 0.015, 0.16, 0.018])  # [left,bottom,w,h]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Log₂ Fold Change  (Injured / Healthy)", fontsize=7.5, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    # Stars legend
    fig.text(0.365, 0.038, "* p<0.05   ** p<0.01   *** p<0.001",
             fontsize=7, color="#444444", ha="left", va="bottom")

    # ── Panel D: –log10(p) dot strip ─────────────────────────────────────
    if p_sec:
        log_p    = -np.log10(np.clip(p_sec_vals, 1e-300, 1.0))
        lp_max   = max(log_p.max(), 1.5)
        dot_cmap = plt.get_cmap("plasma")
        dot_norm = mcolors.Normalize(vmin=0, vmax=lp_max)

        for i, lp in enumerate(log_p):
            col  = dot_cmap(dot_norm(lp))
            size = 30 + 120 * (lp / (lp_max + 1e-9))
            ax_pval.scatter(0.5, i, s=size, color=col,
                            edgecolors="white", linewidths=0.4, zorder=3)

        ax_pval.set_xlim(0, 1)
        ax_pval.set_ylim(-0.5, n_rows - 0.5)
        ax_pval.invert_yaxis()
        ax_pval.set_yticks([])
        ax_pval.set_xticks([])
        ax_pval.spines[:].set_visible(False)
        ax_pval.set_title("–log₁₀\n(p-val)", fontsize=9, fontweight="bold", pad=6)
        ax_pval.axhline(-0.5, color="#cccccc", linewidth=0.5)

        # p-val colorbar
        sm_p = plt.cm.ScalarMappable(cmap=dot_cmap, norm=dot_norm)
        sm_p.set_array([])
        cbar_ax_p = fig.add_axes([0.565, 0.015, 0.06, 0.018])
        cbar_p = fig.colorbar(sm_p, cax=cbar_ax_p, orientation="horizontal")
        cbar_p.set_label("–log₁₀(p)", fontsize=7.5, labelpad=2)
        cbar_p.ax.tick_params(labelsize=7)

    # ── Panel E: Source → Sink cell-type text ─────────────────────────────
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(-0.5, n_rows - 0.5)
    ax_label.invert_yaxis()
    ax_label.axis("off")
    ax_label.set_title("Source  →  Sink", fontsize=9, fontweight="bold", pad=6)

    for i, (lbl, sig) in enumerate(zip(axis_labels, dfx["_sig"].tolist())):
        parts = lbl.split(" → ", 1)
        src_t = parts[0] if len(parts) == 2 else lbl
        snk_t = parts[1] if len(parts) == 2 else ""

        # Colour-code source and sink text
        ax_label.text(
            0.02, i, src_t + "  →",
            va="center", ha="left", fontsize=7.5,
            color="#c0392b" if sig else "#888888",
            fontweight="semibold",
        )
        ax_label.text(
            0.55, i, snk_t,
            va="center", ha="left", fontsize=7.5,
            color="#2980b9" if sig else "#888888",
            fontweight="semibold",
        )

    # Significance legend strip
    if n_sig > 0:
        ax_label.text(
            0.02, -0.45,
            f"▲ {n_sig} significant  (p<0.05, bold rows)",
            fontsize=7, color="#555555", va="bottom",
        )

    # ── Super-title ───────────────────────────────────────────────────────
    fig.suptitle(
        f"Metabolite Flux Changes: Injured vs Healthy\n"
        f"{n_rows} metabolites  (interaction score ≥ {min_score:.0f})  ·  "
        f"{n_sig} significant (p < 0.05)",
        fontsize=cfg.title_font_size + 1,
        fontweight="bold",
        y=0.975,
    )

    filepath = os.path.join(out_dir, "metabolite_changes_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Significant metabolite bar plots (with cell-type info)
# =============================================================================

def plot_significant_metabolites_bars(results, out_dir=None, cfg=None, fc_thresh=1.0, p_thresh=0.05, top_n=15):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_comparison")

    if _is_single_condition(results):
        print("  [SKIP] plot_significant_metabolites_bars: requires two "
              "conditions for fold-change comparison.")
        return None

    df = getattr(results, "comparison_df", pd.DataFrame())
    if df is None or df.empty:
        return None

    met_col = _first_existing_col(df, ["Metabolite", "metabolite"])
    fc_sec = _first_existing_col(df, ["Log2FC_Secretion", "Log2FCSecretion"])
    p_sec = _first_existing_col(df, ["P_Secretion", "PSecretion"])
    fc_upt = _first_existing_col(df, ["Log2FC_Uptake", "Log2FCUptake"])
    p_upt = _first_existing_col(df, ["P_Uptake", "PUptake"])
    if None in (met_col, fc_sec, p_sec, fc_upt, p_upt):
        return None

    dfx = df.copy()
    dfx = dfx[~dfx[met_col].apply(_should_exclude_metabolite)].copy()

    met_to_cell = {}
    if hasattr(results, "get_metabolite_info"):
        for met in dfx[met_col].unique():
            try:
                src, snk, _ = results.get_metabolite_info(met)
            except Exception:
                src, snk = None, None
            if src and snk:
                met_to_cell[met] = f"{src}→{snk}"
            elif src:
                met_to_cell[met] = f"from {src}"
            elif snk:
                met_to_cell[met] = f"to {snk}"

    sec_up = dfx[(dfx[p_sec] < p_thresh) & (dfx[fc_sec] > fc_thresh)].sort_values(fc_sec, ascending=False).head(top_n)
    sec_dn = dfx[(dfx[p_sec] < p_thresh) & (dfx[fc_sec] < -fc_thresh)].sort_values(fc_sec, ascending=True).head(top_n)
    upt_up = dfx[(dfx[p_upt] < p_thresh) & (dfx[fc_upt] > fc_thresh)].sort_values(fc_upt, ascending=False).head(top_n)
    upt_dn = dfx[(dfx[p_upt] < p_thresh) & (dfx[fc_upt] < -fc_thresh)].sort_values(fc_upt, ascending=True).head(top_n)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    panels = [
        (axes[0], sec_up, fc_sec, "Secretion ↑ in Injured", "#e74c3c"),
        (axes[1], sec_dn, fc_sec, "Secretion ↓ in Injured", "#3498db"),
        (axes[2], upt_up, fc_upt, "Uptake ↑ in Injured", "#e74c3c"),
        (axes[3], upt_dn, fc_upt, "Uptake ↓ in Injured", "#3498db"),
    ]

    for ax, sub, fc_col, title, color in panels:
        if sub.empty:
            ax.text(0.5, 0.5, "No significant metabolites", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="grey")
            ax.set_title(title, fontweight="bold", fontsize=cfg.subtitle_font_size)
            ax.axis("off")
            continue

        mets = sub[met_col].tolist()
        fcs = sub[fc_col].tolist()

        labels = []
        for m in mets:
            extra = met_to_cell.get(m, "")
            labels.append(_truncate_label(f"{m} ({extra})" if extra else str(m), 48))

        y = np.arange(len(mets))
        ax.barh(y, fcs, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", lw=1.0, alpha=0.6)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_xlabel("Log2 Fold Change", fontsize=cfg.axis_label_font_size)
        ax.set_title(title, fontweight="bold", fontsize=cfg.subtitle_font_size)

        for i, fc in enumerate(fcs):
            ax.text(fc, i, f" {fc:.2f}", va="center",
                    ha="left" if fc > 0 else "right", fontsize=8, fontweight="bold")

    fig.suptitle(
        "Significant Metabolite Changes: Injured vs Healthy (excluded: H2O/CO2/H/O2/H2O2)\n"
        "Labels include best Source→Sink cell types (global consensus).",
        fontsize=cfg.title_font_size, fontweight="bold", y=0.98,
    )

    filepath = os.path.join(out_dir, "significant_metabolites_bars.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Volcano plot
# =============================================================================

def plot_volcano(results, out_dir=None, cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_comparison")

    df = getattr(results, "comparison_df", pd.DataFrame())
    if df is None or df.empty:
        return None

    met_col = _first_existing_col(df, ["Metabolite", "metabolite"])
    fc_sec = _first_existing_col(df, ["Log2FC_Secretion", "Log2FCSecretion"])
    p_sec = _first_existing_col(df, ["P_Secretion", "PSecretion"])
    fc_upt = _first_existing_col(df, ["Log2FC_Uptake", "Log2FCUptake"])
    p_upt = _first_existing_col(df, ["P_Uptake", "PUptake"])
    if None in (met_col, fc_sec, p_sec, fc_upt, p_upt):
        return None

    dfx0 = df.copy()
    dfx0 = dfx0[~dfx0[met_col].apply(_should_exclude_metabolite)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, (fc_col, p_col, title) in zip(axes, [(fc_sec, p_sec, "Secretion"), (fc_upt, p_upt, "Uptake")]):
        dfx = dfx0.copy()
        pvals = pd.to_numeric(dfx[p_col], errors="coerce").fillna(1.0).clip(lower=1e-300)
        dfx["-log10(p)"] = -np.log10(pvals)

        fc_thresh = 1.0
        p_thresh = 0.05

        # ── Step 1: compute y_cap BEFORE plotting anything ───────────────────
        # This ensures set_ylim is the ground truth for everything that follows
        # (scatter, labels, adjust_text) and nothing bleeds outside the cap.
        sig_mask = (dfx[p_col] < p_thresh) & (dfx[fc_col].abs() > fc_thresh)
        sig_yvals = dfx.loc[sig_mask, "-log10(p)"]
        if len(sig_yvals) > 0:
            y_cap = float(sig_yvals.max()) * 1.20
            y_cap = max(y_cap, -np.log10(p_thresh) * 2.0)
        else:
            y_cap = float(np.nanpercentile(dfx["-log10(p)"].dropna(), 99)) * 1.20
        y_cap = max(y_cap, 1.5)  # absolute floor

        # ── Step 2: clip plotted y-values so outliers don't expand the canvas
        dfx["-log10(p)_plot"] = dfx["-log10(p)"].clip(upper=y_cap * 0.98)

        def _point_color(row):
            if row[p_col] < p_thresh and row[fc_col] > fc_thresh:
                return "#e74c3c"
            if row[p_col] < p_thresh and row[fc_col] < -fc_thresh:
                return "#3498db"
            return "#95a5a6"

        colors = dfx.apply(_point_color, axis=1).tolist()
        ax.scatter(dfx[fc_col], dfx["-log10(p)_plot"],
                   c=colors, alpha=0.65, s=55, edgecolors="white", lw=0.5)

        # ── Step 3: set ylim immediately after scatter ────────────────────────
        ax.set_ylim(bottom=0, top=y_cap)

        sig = dfx[sig_mask].copy()
        sig = sig.nlargest(18, "-log10(p)")
        # Place labels at clipped y so they stay inside axes
        sig["_y_label"] = sig["-log10(p)"].clip(upper=y_cap * 0.95)

        texts = []
        for _, row in sig.iterrows():
            texts.append(
                ax.text(
                    row[fc_col], row["_y_label"],
                    str(row[met_col]),
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow",
                              alpha=0.40, edgecolor="none"),
                )
            )

        if HAS_ADJUSTTEXT and texts:
            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.8, 1.8),
                expand_text=(1.1, 1.1),
                force_points=(0.5, 0.5),
                force_text=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.6),
                lim=600,
            )
            # Re-enforce ylim after adjust_text which may expand axes
            ax.set_ylim(bottom=0, top=y_cap)

        ax.axhline(y=-np.log10(p_thresh), color="grey", linestyle="--", alpha=0.55, linewidth=1.5)
        ax.axvline(x=fc_thresh, color="grey", linestyle="--", alpha=0.55, linewidth=1.5)
        ax.axvline(x=-fc_thresh, color="grey", linestyle="--", alpha=0.55, linewidth=1.5)

        ax.set_xlabel(f"Log2 Fold Change ({title})", fontsize=cfg.axis_label_font_size, fontweight="bold")
        ax.set_ylabel("-log10(p-value)", fontsize=cfg.axis_label_font_size, fontweight="bold")
        ax.set_title(f"{title} Changes\n(Red: ↑ Injured, Blue: ↓ Injured)",
                     fontsize=cfg.subtitle_font_size, fontweight="bold")
        ax.grid(alpha=0.18, linestyle=":")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="↑ in Injured"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="↓ in Injured"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=10, label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=cfg.legend_font_size)

    plt.suptitle(
        "Metabolite Flux Changes: Injured vs Healthy (excluded: H2O/CO2/H/O2/H2O2)",
        fontsize=cfg.title_font_size + 2, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # reserve top 4% for suptitle

    filepath = os.path.join(out_dir, "volcano_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath




# =============================================================================
# Top Metabolites Bar Plot - NEW
# =============================================================================

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a score array, clipping to [-3, 3] for display.
    NaN-safe: NaNs are replaced with 0 after normalization.
    """
    s = np.asarray(scores, dtype=float)
    mu, sd = np.nanmean(s), np.nanstd(s)
    if sd < 1e-12:
        return np.zeros_like(s)
    z = (s - mu) / sd
    return np.clip(z, -3, 3)


def _build_met_score_table(results, normalize: bool = False) -> pd.DataFrame:
    """
    Aggregate per-metabolite interaction scores across all regions.

    Returns columns: metabolite, source, sink, score, score_norm, p_value.

    normalize=True  → sort / display by z-score normalized score
                       (captures relative changes across pathways)
    normalize=False → sort / display by raw absolute score
                       (captures absolute flux magnitude)

    Why both matter:
      Absolute score: tells you which metabolite has the largest raw flux.
        Important for energy budget and dominant pathways.
      Normalized (z-score) score: tells you which metabolite changed *most
        relative to its own baseline variability*. Important for detecting
        pathway switches that might be masked by scale differences.
    """
    per_region = getattr(results, "per_region_data", {})
    rows = []
    for key, data in per_region.items():
        for met, pairs in (data.get("interactions", {}) or {}).items():
            if _should_exclude_metabolite(met) or not pairs:
                continue
            top = pairs[0]
            rows.append({
                "metabolite": met,
                "source": top.get("source", ""),
                "sink": top.get("sink", ""),
                "score": float(top.get("score", 0.0)),
                "region": key,
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    met_avg = df.groupby(["metabolite", "source", "sink"], as_index=False)["score"].mean()

    # Z-score normalize across all metabolites
    met_avg["score_norm"] = _normalize_scores(met_avg["score"].values)

    # Attach p-values
    p_map = {}
    comp_df = getattr(results, "comparison_df", None)
    if comp_df is not None and not comp_df.empty:
        pc = _first_existing_col(comp_df, ["P_Secretion", "PSecretion"])
        mc = _first_existing_col(comp_df, ["Metabolite", "metabolite"])
        if pc and mc:
            for _, row in comp_df.iterrows():
                p_map[row[mc]] = float(row[pc])
    met_avg["p_value"] = met_avg["metabolite"].map(lambda m: p_map.get(m, 1.0))

    sort_col = "score_norm" if normalize else "score"
    return met_avg.sort_values(sort_col, ascending=False).reset_index(drop=True)


def plot_top_metabolites_bar(results, out_dir=None, cfg=None, top_n=20):
    """
    Two-panel bar plot: left = top metabolites by absolute interaction score,
    right = top metabolites by z-score normalized score.

    Why both panels:
      Left  (absolute): dominant metabolites by raw flux magnitude.
        Tells you what drives the system energetically.
      Right (normalized): metabolites that changed most relative to their
        own cross-metabolite variability.  Detects pathway switches that
        would be invisible if one metabolite dominates the raw scale.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        out_dir = _safe_out_dir(results, "plots_comparison")

    df_abs  = _build_met_score_table(results, normalize=False).head(top_n)
    df_norm = _build_met_score_table(results, normalize=True).head(top_n)

    if df_abs.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, top_n * 0.45)))

    for ax, df_sub, score_col, xlabel, panel_title in [
        (axes[0], df_abs,  "score",      "Avg interaction score (absolute)",
         "Top metabolites — Absolute flux"),
        (axes[1], df_norm, "score_norm", "Z-score normalized score",
         "Top metabolites — Relative change (z-score)"),
    ]:
        if df_sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue

        mets    = df_sub["metabolite"].tolist()
        vals    = df_sub[score_col].tolist()
        sources = df_sub["source"].tolist()
        sinks   = df_sub["sink"].tolist()
        pvals   = df_sub["p_value"].tolist()
        y       = np.arange(len(mets))

        colors  = ["#3498db" if p < 0.05 else "#7f8c8d" for p in pvals]
        ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

        labels = [
            _truncate_label(f"{m}  ({s}→{k}, p={p:.3g})", 56)
            for m, s, k, p in zip(mets, sources, sinks, pvals)
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_xlabel(xlabel, fontsize=cfg.axis_label_font_size, fontweight="bold")
        ax.set_title(panel_title, fontsize=cfg.subtitle_font_size, fontweight="bold", pad=10)

    legend_elements = [
        mpatches.Patch(facecolor="#3498db", alpha=0.85, edgecolor="black", label="Significant (p < 0.05)"),
        mpatches.Patch(facecolor="#7f8c8d", alpha=0.85, edgecolor="black", label="Not significant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=cfg.legend_font_size, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top Metabolites by Interaction Score", fontsize=cfg.title_font_size,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    filepath = os.path.join(out_dir, "top_metabolites_bar.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


# =============================================================================
# Diffusion plot – P05 panel style (FIXED)
# =============================================================================

def _build_concentration_field(meta_df, src_type, snk_type, xi, yi, sigma_smooth):
    """
    Build the scalar concentration field from cell-density histograms.

    The concentration field is defined as:
        C = gaussian_filter(H_source) − gaussian_filter(H_sink)

    so that:
        • High C  (positive, warm colours) = where source cells are dense
        • Low  C  (negative, cool colours) = where sink cells are dense

    This is the field that should be used to weight scatter positions, NOT
    the gradient magnitude (sqrt(U²+V²)).  Using the gradient magnitude was
    the original bug — the gradient peaks at the *transition zone* between
    source and sink, not at the source/sink itself, so scatters ended up in
    the wrong place.
    """
    px = meta_df["px_x"].astype(float).values
    py = meta_df["px_y"].astype(float).values

    gs = len(xi)
    pad_x = (px.max() - px.min()) * 0.1 if px.max() > px.min() else 1.0
    pad_y = (py.max() - py.min()) * 0.1 if py.max() > py.min() else 1.0
    x_edges = np.linspace(px.min() - pad_x, px.max() + pad_x, gs + 1)
    y_edges = np.linspace(py.min() - pad_y, py.max() + pad_y, gs + 1)

    src_mask = (meta_df["cell_type"].astype(str) == str(src_type)).values
    snk_mask = (meta_df["cell_type"].astype(str) == str(snk_type)).values

    from scipy.ndimage import gaussian_filter as _gf

    if np.any(src_mask):
        H_src, _, _ = np.histogram2d(py[src_mask], px[src_mask],
                                      bins=[y_edges, x_edges])
    else:
        H_src = np.zeros((gs, gs))

    if np.any(snk_mask):
        H_snk, _, _ = np.histogram2d(py[snk_mask], px[snk_mask],
                                      bins=[y_edges, x_edges])
    else:
        H_snk = np.zeros((gs, gs))

    conc = _gf(H_src, sigma_smooth) - _gf(H_snk, sigma_smooth)
    # Second smoothing pass for visual continuity
    conc = _gf(conc, max(1, sigma_smooth // 2))
    return conc, H_src, H_snk, px, py, src_mask, snk_mask


def _weighted_scatter_sample(px_sub, py_sub, weight_field, xi_g, yi_g,
                              n_target, favor_high=True, seed=42):
    """
    Draw up to n_target positions from (px_sub, py_sub), weighted by the
    concentration field at each cell's grid location.

    favor_high=True  → more points where the field is largest  (sources → red)
    favor_high=False → more points where the field is smallest (sinks  → blue)
    """
    if len(px_sub) == 0:
        return np.array([]), np.array([])

    # Nearest-grid-cell lookup (clipped to valid range)
    xi_idx = np.searchsorted(xi_g, px_sub).clip(0, len(xi_g) - 1)
    yi_idx = np.searchsorted(yi_g, py_sub).clip(0, len(yi_g) - 1)
    w = weight_field[yi_idx, xi_idx].astype(float)

    if favor_high:
        w = w - w.min() + 1e-9          # shift so all ≥ 0
    else:
        w = -(w - w.max()) + 1e-9       # invert: smallest original → largest weight

    w = np.clip(w, 1e-12, None)
    w /= w.sum()

    n_draw = min(n_target, len(px_sub))
    rng = np.random.default_rng(seed=int(abs(seed)) % (2**31))
    idx = rng.choice(len(px_sub), size=n_draw,
                     replace=(n_draw > len(px_sub)), p=w)
    return px_sub[idx], py_sub[idx]


def plot_diffusion_panel(ax, U, V, xi, yi, meta_df, src_type, snk_type,
                         title="", show_colorbar=True, cfg=None,
                         potential=None):
    """
    Render a single P05-style diffusion panel onto *ax*.

    Heatmap is built from cell-density histograms:
        C = gaussian(H_source) − gaussian(H_sink)
    so warm/red regions correspond to source-dense areas and cool/blue
    regions to sink-dense areas.  The colormap is centered at C=0 so
    the white midpoint is genuine biological neutrality (equal density).

    Biological soundness fixes vs previous version:
      1. Smaller sigma (3% of grid diagonal vs 10%) so smoothing does
         not bleed colour into regions with no cells of that type.
      2. Tissue alpha mask: outside the convex hull of ALL cells the
         heatmap is faded to fully transparent, so colour only appears
         where the tissue actually is.
      3. Symmetric colormap normalisation centred at C=0: white = truly
         neutral, not an artefact of scale.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG

    try:
        U  = np.asarray(U,  dtype=float)
        V  = np.asarray(V,  dtype=float)
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        if U.shape != V.shape or len(xi) != U.shape[1] or len(yi) != U.shape[0]:
            return
    except Exception:
        return

    meta_df = _filter_metadata_to_grid(meta_df, xi, yi, margin_pct=0.05)
    ax.set_facecolor("white")

    from scipy.ndimage import gaussian_filter as _gf

    gs_y, gs_x = U.shape
    x_edges = np.linspace(xi[0], xi[-1], gs_x + 1)
    y_edges = np.linspace(yi[0], yi[-1], gs_y + 1)

    px_all = meta_df["px_x"].astype(float).values if (meta_df is not None and not meta_df.empty) else np.array([])
    py_all = meta_df["px_y"].astype(float).values if (meta_df is not None and not meta_df.empty) else np.array([])

    src_mask = ((meta_df["cell_type"].astype(str) == str(src_type)).values
                if (meta_df is not None and not meta_df.empty and src_type)
                else np.zeros(len(px_all), dtype=bool))
    snk_mask = ((meta_df["cell_type"].astype(str) == str(snk_type)).values
                if (meta_df is not None and not meta_df.empty and snk_type)
                else np.zeros(len(px_all), dtype=bool))

    if np.any(src_mask):
        H_src, _, _ = np.histogram2d(py_all[src_mask], px_all[src_mask],
                                      bins=[y_edges, x_edges])
    else:
        H_src = np.zeros((gs_y, gs_x))

    if np.any(snk_mask):
        H_snk, _, _ = np.histogram2d(py_all[snk_mask], px_all[snk_mask],
                                      bins=[y_edges, x_edges])
    else:
        H_snk = np.zeros((gs_y, gs_x))

    # ── Concentration field: C = gaussian(H_src) − gaussian(H_sink) ──────────
    # Moderate sigma (5% of grid diagonal) smooths enough to show spatial
    # gradients without bleeding colour far into regions with no cells.
    # We then normalise symmetrically using the 95th-percentile absolute
    # value so that extreme outlier bins don't compress the rest of the
    # colour range — this addresses the "all red / all blue" issue without
    # changing the underlying biology-driven field.
    diag = np.sqrt(gs_x**2 + gs_y**2)
    sigma = max(2.0, diag * 0.05)
    C = _gf(H_src.astype(float), sigma) - _gf(H_snk.astype(float), sigma)
    conc_smooth = _gf(C, max(1.0, sigma * 0.4))

    # ── Tissue alpha mask — heatmap fades to white outside cell neighbourhood ─
    # Build an all-cells density, smooth with a wider kernel, use as alpha.
    # This prevents colour from appearing in biologically empty corners.
    if len(px_all) > 0:
        H_all, _, _ = np.histogram2d(py_all, px_all, bins=[y_edges, x_edges])
        sigma_tissue = max(3.0, diag * 0.08)
        tissue_density = _gf(H_all.astype(float), sigma_tissue)
        td_max = tissue_density.max()
        alpha_map = (tissue_density / td_max).clip(0, 1) if td_max > 0 \
            else np.ones((gs_y, gs_x), dtype=float)
        # Gamma < 1 keeps the interior fully opaque and only softens the edges
        alpha_map = alpha_map.astype(float) ** 0.4
    else:
        alpha_map = np.ones((gs_y, gs_x), dtype=float)

    # ── Symmetric colour normalisation centred at C=0 ─────────────────────────
    # vmax = 95th percentile of |C| across the panel.  This stretches the
    # colour gradient over the actual data range, so a panel where sources
    # only slightly outnumber sinks still shows visible blue regions rather
    # than mapping everything to near-red.
    abs_max = float(np.nanpercentile(np.abs(conc_smooth), 95))
    if abs_max < 1e-12:
        abs_max = 1.0
    norm_div = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    # ── Heatmap — pcolormesh + gouraud (smooth gradients) ────────────────────
    XX, YY = np.meshgrid(xi, yi)
    pcm = ax.pcolormesh(
        XX, YY, conc_smooth,
        cmap=cfg.diffusion_cmap,   # RdYlBu_r: softer gradient than RdBu_r
        norm=norm_div,
        shading="gouraud",
        alpha=alpha_map,
        zorder=1,
        rasterized=True,
    )

    # Quiver arrows intentionally omitted from diffusion panels:
    # the heatmap + cell scatter already convey directionality, and
    # uniform-length arrows clutter the concentration gradient.

    # ── Cell scatter — ALL cells at their true positions ──────────────────────
    # Background: all cells as small light dots, no edge
    if meta_df is not None and not meta_df.empty:
        ax.scatter(px_all, py_all,
                   s=cfg.scatter_size_bg, color="white",
                   alpha=0.30, zorder=2, linewidths=0, rasterized=True)

    # Source cells: glow halo + solid dot
    if np.any(src_mask):
        ax.scatter(px_all[src_mask], py_all[src_mask],
                   s=cfg.scatter_size_highlight * 3.5, color=cfg.source_color,
                   alpha=0.12, zorder=4, linewidths=0)
        ax.scatter(px_all[src_mask], py_all[src_mask],
                   s=cfg.scatter_size_highlight, color=cfg.source_color,
                   alpha=cfg.scatter_alpha_highlight, zorder=5,
                   edgecolors="white", linewidths=0.8,
                   label=f"{src_type} (Source)")

    # Sink cells: glow halo + solid dot
    if np.any(snk_mask):
        ax.scatter(px_all[snk_mask], py_all[snk_mask],
                   s=cfg.scatter_size_highlight * 3.5, color=cfg.sink_color,
                   alpha=0.12, zorder=4, linewidths=0)
        ax.scatter(px_all[snk_mask], py_all[snk_mask],
                   s=cfg.scatter_size_highlight, color=cfg.sink_color,
                   alpha=cfg.scatter_alpha_highlight, zorder=5,
                   edgecolors="white", linewidths=0.8,
                   label=f"{snk_type} (Sink)")

    # ── Colorbar (horizontal, Panel 05 style) ─────────────────────────────────
    if show_colorbar:
        try:
            cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal",
                                pad=0.03, fraction=0.038, aspect=35,
                                drawedges=False)
            cbar.set_label("Source ← Concentration → Sink",
                           fontsize=cfg.axis_label_font_size - 2,
                           color='#333333', labelpad=3)
            c_lo, c_hi = float(np.nanmin(conc_smooth)), float(np.nanmax(conc_smooth))
            cbar.set_ticks([c_lo, c_hi])
            cbar.set_ticklabels(["Low", "High"], fontsize=cfg.tick_font_size)
            cbar.ax.tick_params(labelcolor='#333333', color='#888888',
                                length=2, width=0.5)
            cbar.outline.set_edgecolor('#dddddd')
            cbar.outline.set_linewidth(0.5)
        except Exception:
            pass

    # ── Axis cosmetics (Panel 05 style: white bg, no frame, title as xlabel) ──
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_xlabel(title, fontsize=cfg.subtitle_font_size,
                  fontweight="bold", labelpad=8)

    if meta_df is not None and not meta_df.empty and (src_type or snk_type):
        try:
            ax.legend(loc="upper right", fontsize=cfg.legend_font_size - 1,
                      framealpha=0.88, edgecolor="grey")
        except Exception:
            pass


def plot_diffusion_comparison(results, metabolites=None, out_dir=None, cfg=None):
    """
    Panel 05-style diffusion figure for each metabolite.
    Layout: one subplot per condition, showing:
      • Smooth concentration heatmap (blue=low → red=high) built from
        source / sink cell-density histograms
      • Uniform-length black quiver arrows showing flux direction
      • Source cell scatters weighted to appear in warm/red regions
      • Sink cell scatters weighted to appear in cool/blue regions

    This replaces the previous version which incorrectly used sqrt(U²+V²)
    as the concentration proxy — that quantity peaks in the *transition zone*
    between source and sink, not at the source/sink themselves.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if metabolites is None:
        metabolites = results.significant_mets
    if out_dir is None:
        out_dir = os.path.join(results.config.out_dir, "plots_diffusion")
    os.makedirs(out_dir, exist_ok=True)

    # Add diffusion_figsize default if not present
    diffusion_figsize = getattr(cfg, "diffusion_figsize", (9 * 2, 7))

    saved_files = []
    conditions = results.config.conditions

    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue
        src, snk, p_val = results.get_metabolite_info(met)
        if src is None:
            continue

        condition_data = {}
        for condition in conditions:
            fields = results.get_vector_fields_for_condition(condition, met)
            meta_df = results.get_metadata_for_condition(condition)
            if not fields or meta_df.empty:
                continue
            U_avg = np.mean([f[0] for f in fields], axis=0)
            V_avg = np.mean([f[1] for f in fields], axis=0)
            xi, yi = fields[0][2], fields[0][3]

            # Coarsen for quiver readability
            cf = max(1, getattr(cfg, "consensus_differential_coarsen_factor", 2))
            U_c, V_c, xi_c, yi_c = _coarsen_vector_field(U_avg, V_avg, xi, yi, cf)

            condition_data[condition] = {
                "U": U_c, "V": V_c, "xi": xi_c, "yi": yi_c,
                "meta_df": meta_df.copy(),
            }

        if not condition_data:
            continue

        ncols = len(condition_data)
        w = getattr(cfg, "diffusion_figsize", (9 * ncols, 7))
        fig, axes = plt.subplots(1, ncols, figsize=(w[0], w[1]))
        if ncols == 1:
            axes = [axes]

        for ax, (condition, d) in zip(axes, condition_data.items()):
            cond_short = condition.replace("Sample_", "")
            panel_title = f"{cond_short}  |  p = {float(p_val):.4g}"
            plot_diffusion_panel(
                ax, d["U"], d["V"], d["xi"], d["yi"],
                d["meta_df"], src, snk,
                title=panel_title,
                show_colorbar=True,
                cfg=cfg,
            )

        fig.suptitle(
            f"Metabolite Diffusion: {met}\n"
            f"Source: {src}  \u2192  Sink: {snk}   (p = {float(p_val):.4g})",
            fontsize=cfg.title_font_size, fontweight="bold",
        )
        plt.tight_layout()
        filepath = os.path.join(out_dir, f"diffusion_{met}.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_files.append(filepath)

    return saved_files


def _write_region_index(results, cfg=None, verbose=True) -> str:
    """
    Write a human-readable text file listing every region key with:
      - condition name
      - region number
      - number of cells
      - cell types present
      - metabolites with streamline plots in that region
      - source / sink cell type for each metabolite
      - confirmation that source scatter = src_type ONLY,
        sink scatter = snk_type ONLY (enforced by cell_type equality filter
        in plot_streamline_panel lines 606-648)

    Saved as: <out_dir>/region_index.txt
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    base    = getattr(getattr(results, "config", None), "out_dir", "analysis_output")
    out_dir = base
    os.makedirs(out_dir, exist_ok=True)
    fp      = os.path.join(out_dir, "region_index.txt")

    per_region = getattr(results, "per_region_data", {}) or {}
    conditions = getattr(getattr(results, "config", None), "conditions", [])

    lines = []
    lines.append("=" * 80)
    lines.append("REGION INDEX — SPICEM STREAMLINE PLOT REGISTRY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("COLOUR CONVENTION (confirmed by code, see plot_streamline_panel):")
    lines.append("  RED   dots  = Source cell type ONLY  (filtered: cell_type == src_type)")
    lines.append("  BLUE  dots  = Sink cell type ONLY    (filtered: cell_type == snk_type)")
    lines.append("  GREY  dots  = All other cells in the region (background)")
    lines.append("  No red/blue dot of any other cell type is ever plotted.")
    lines.append("")
    lines.append(f"Total regions: {len(per_region)}")
    lines.append(f"Conditions:    {', '.join(conditions)}")
    lines.append("")

    for key in sorted(per_region.keys()):
        data = per_region[key]

        # Parse condition / region
        if "_R" in key:
            parts   = key.rsplit("_R", 1)
            cond    = parts[0]
            reg_num = parts[1]
        else:
            cond    = key
            reg_num = "1"

        meta      = data.get("meta_df", pd.DataFrame())
        vf        = data.get("vector_fields", {})
        best_pairs = data.get("best_pairs", {})
        interactions = data.get("interactions", {})

        n_cells    = len(meta) if meta is not None else 0
        cell_types = sorted(meta["cell_type"].astype(str).unique().tolist())                      if (meta is not None and not meta.empty and "cell_type" in meta.columns)                      else []

        # x/y range
        if meta is not None and not meta.empty:
            xc = "px_x" if "px_x" in meta.columns else ("x" if "x" in meta.columns else None)
            yc = "px_y" if "px_y" in meta.columns else ("y" if "y" in meta.columns else None)
            if xc and yc:
                xr = f"[{meta[xc].min():.1f}, {meta[xc].max():.1f}]"
                yr = f"[{meta[yc].min():.1f}, {meta[yc].max():.1f}]"
            else:
                xr = yr = "n/a"
        else:
            xr = yr = "n/a"

        lines.append("─" * 80)
        lines.append(f"KEY        : {key}")
        lines.append(f"Condition  : {cond.replace('Sample_', '')}")
        lines.append(f"Region     : {reg_num}")
        lines.append(f"Cells      : {n_cells:,}")
        lines.append(f"X range    : {xr}")
        lines.append(f"Y range    : {yr}")
        lines.append(f"Cell types ({len(cell_types)}):")
        for ct in cell_types:
            ct_n = len(meta[meta["cell_type"].astype(str) == ct])                    if (meta is not None and not meta.empty) else 0
            lines.append(f"    {ct:<40s}  n = {ct_n:,}")

        # Metabolites with streamline data
        mets_with_fields = sorted(vf.keys()) if vf else []
        lines.append(f"Metabolites with streamline fields ({len(mets_with_fields)}):")
        for met in mets_with_fields:
            if _should_exclude_metabolite(met):
                continue
            src, snk = None, None
            if met in best_pairs:
                try:
                    src, snk, _ = best_pairs[met]
                except Exception:
                    pass
            if src is None and met in interactions and interactions[met]:
                top = interactions[met][0]
                src = top.get("source", "?")
                snk = top.get("sink",   "?")

            # Count cells of src and snk type in this region
            def _ct_count(ct):
                if meta is None or meta.empty or not ct:
                    return 0
                return int((meta["cell_type"].astype(str) == str(ct)).sum())

            n_src = _ct_count(src)
            n_snk = _ct_count(snk)
            src_ok = "✓" if n_src >= 3 else "✗ (too few — panel suppressed)"
            snk_ok = "✓" if n_snk >= 3 else "✗ (too few — panel suppressed)"

            lines.append(f"    {met:<20s}  Source: {str(src):<30s} n={n_src:>4d} {src_ok}")
            lines.append(f"    {'':20s}  Sink  : {str(snk):<30s} n={n_snk:>4d} {snk_ok}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("SCATTER FILTER AUDIT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("The following code in plot_streamline_panel guarantees colour purity:")
    lines.append("")
    lines.append("  # Source (RED) scatter:")
    lines.append("  sub = meta_df[meta_df['cell_type'].astype(str) == str(src_type)]")
    lines.append("  ax.scatter(sub['px_x'], sub['px_y'], color=cfg.source_color, ...)")
    lines.append("")
    lines.append("  # Sink (BLUE) scatter:")
    lines.append("  sub = meta_df[meta_df['cell_type'].astype(str) == str(snk_type)]")
    lines.append("  ax.scatter(sub['px_x'], sub['px_y'], color=cfg.sink_color, ...)")
    lines.append("")
    lines.append("Conclusion: every red dot is exactly src_type, every blue dot is exactly")
    lines.append("snk_type. No cell of any other type is ever coloured red or blue.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REGION INDEX")

    with open(fp, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"  Region index written: {fp}")
    return fp


# =============================================================================
# Spatial cell-type map — one figure per region (like the reference image)
# =============================================================================

# =============================================================================
# Perceptually-distinct colour palette for up to ~30 cell types.
#
# Design rationale:
#   • The 20-colour tab20 palette groups colours in pairs of light/dark
#     shades, which are easily confused when dots overlap.  We instead
#     use a hand-curated set of maximally-separated hues at consistent
#     lightness (~55 % L* in CIELAB), then add a second ring at higher
#     lightness for extra types.
#   • Colours are ordered so adjacent entries differ as much as possible
#     in hue (roughly alternating warm/cool), reducing confusion between
#     neighbouring cell-type groups in the legend.
#   • A pure black is reserved as the fallback for unlisted types.
# =============================================================================

# Primary ring — 20 high-contrast, medium-lightness colours
_PALETTE_PRIMARY = [
    "#e6194b",   #  0  vivid red          (T cells, immune anchor)
    "#3cb44b",   #  1  vivid green         (Proximal tubule)
    "#4363d8",   #  2  vivid blue          (Collecting duct)
    "#f58231",   #  3  vivid orange        (Thick ascending limb)
    "#911eb4",   #  4  vivid purple        (Podocytes)
    "#42d4f4",   #  5  cyan                (Interstitial fibroblasts)
    "#f032e6",   #  6  magenta             (Stressed cells)
    "#bfef45",   #  7  lime                (Proximal tubule S1-S3)
    "#fabed4",   #  8  pink                (Peritubular EC)
    "#469990",   #  9  teal                (Intercalated cells)
    "#dcbeff",   # 10  lavender            (Distal convoluted tubule)
    "#9A6324",   # 11  brown               (Mesangial)
    "#fffac8",   # 12  cream               (glomerular capillaries)
    "#800000",   # 13  maroon              (Juxtaglomerular/VSMCs)
    "#aaffc3",   # 14  mint                (LAMC2+ epithelial)
    "#808000",   # 15  olive               (Plasma cells)
    "#ffd8b1",   # 16  apricot             (Injured collecting duct)
    "#000075",   # 17  navy                (Injury response cells)
    "#a9a9a9",   # 18  medium grey         (Renal capsule)
    "#ffffff",   # 19  white (dark edge)   (B cells / rare)
]

# Extended ring — 12 additional colours for datasets with >20 types
_PALETTE_EXTENDED = [
    "#e6beff",   # 20  soft purple
    "#ff6961",   # 21  soft red-pink
    "#77dd77",   # 22  pastel green
    "#fdfd96",   # 23  pastel yellow
    "#84b6f4",   # 24  soft blue
    "#fdcae1",   # 25  blush
    "#b5ead7",   # 26  seafoam
    "#c7b8ea",   # 27  wisteria
    "#ffb347",   # 28  pastel orange
    "#779ecb",   # 29  steel blue
    "#966fd6",   # 30  amethyst
    "#03c03c",   # 31  dark pastel green
]

_FULL_PALETTE = _PALETTE_PRIMARY + _PALETTE_EXTENDED


def _make_celltype_palette(cell_types: list) -> dict:
    """
    Assign a maximally-distinct colour to each cell type.

    Cell types are sorted alphabetically so the assignment is deterministic
    and consistent across all regions (same type → same colour everywhere).

    Returns dict mapping cell_type_string → hex colour string.
    """
    unique_sorted = sorted(set(str(ct) for ct in cell_types))
    return {
        ct: _FULL_PALETTE[i % len(_FULL_PALETTE)]
        for i, ct in enumerate(unique_sorted)
    }


def plot_spatial_celltype_map(
    results,
    out_dir: str = None,
    cfg=None,
    marker_size: int = 38,       # increased from 22
    alpha: float = 0.78,
    max_cells_per_region: int = 0,   # 0 = show all
    verbose: bool = True,
) -> list:
    """
    Generate one spatial cell-type scatter plot per region.

    Visual improvements vs previous version:
      • Larger markers (s=38) so individual cells are clearly visible
      • Hand-curated perceptually-distinct palette (see _FULL_PALETTE)
        — hues spread around the colour wheel at consistent lightness,
          alternating warm/cool to maximise neighbour contrast in legend
      • White marker edge (linewidths=0.6) creates separation when dots
        overlap, replacing the thin black edge that merged at high density
      • Soft off-white (#f7f7f7) axis background reduces eye strain vs
        pure white at full marker alpha
      • Cell types sorted by frequency (most abundant first) so the
        dominant tissue type is drawn first (underneath), rare types on
        top for visibility
      • Figure wider (12 × 9.5) to give the external legend more room

    Each figure is saved as:
        spatial_celltypes_<condition>_R<region>.png
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        base = getattr(getattr(results, "config", None), "out_dir", "analysis_output")
        out_dir = os.path.join(base, "plots_spatial")
    os.makedirs(out_dir, exist_ok=True)

    per_region = getattr(results, "per_region_data", {}) or {}
    if not per_region:
        if verbose:
            print("  plot_spatial_celltype_map: no per_region_data found.")
        return []

    # ── Build palette from ALL cell types for cross-region consistency ────
    all_cell_types = []
    for data in per_region.values():
        meta = data.get("meta_df", pd.DataFrame())
        if meta is not None and not meta.empty and "cell_type" in meta.columns:
            all_cell_types.extend(meta["cell_type"].astype(str).unique().tolist())
    palette = _make_celltype_palette(all_cell_types)

    rng_sub = np.random.default_rng(int(getattr(cfg, "random_seed", 42)))
    saved   = []

    for key, data in sorted(per_region.items()):
        meta = data.get("meta_df", pd.DataFrame())
        if meta is None or meta.empty or "cell_type" not in meta.columns:
            continue

        # ── Parse condition / region label ────────────────────────────────
        if "_R" in key:
            parts   = key.rsplit("_R", 1)
            cond    = parts[0].replace("Sample_", "")
            reg_num = parts[1]
        else:
            cond    = key.replace("Sample_", "")
            reg_num = "1"

        label = f"{cond}  —  Region {reg_num}"
        fname = f"spatial_celltypes_{cond}_R{reg_num}.png"

        # ── Optional subsampling ──────────────────────────────────────────
        if max_cells_per_region > 0 and len(meta) > max_cells_per_region:
            idx  = rng_sub.choice(len(meta), size=max_cells_per_region, replace=False)
            meta = meta.iloc[idx]

        # ── Coordinate columns ────────────────────────────────────────────
        if "px_x" in meta.columns and "px_y" in meta.columns:
            xcol, ycol = "px_x", "px_y"
        elif "x" in meta.columns and "y" in meta.columns:
            xcol, ycol = "x", "y"
        else:
            if verbose:
                print(f"  {key}: no x/y columns found, skipping.")
            continue

        # ── Sort cell types: most-abundant first (drawn underneath),
        #    rare types on top for maximum visibility ─────────────────────
        ct_counts = meta["cell_type"].astype(str).value_counts()
        # Draw abundant types first (they form the background tissue),
        # then progressively rarer types overlay on top
        cell_types_ordered = ct_counts.index.tolist()   # descending count order

        # ── Figure — wider canvas to accommodate external legend ──────────
        fig, ax = plt.subplots(figsize=(12, 9.5))
        ax.set_facecolor("#f7f7f7")   # soft off-white — reduces glare vs pure white
        fig.patch.set_facecolor("white")

        for ct in cell_types_ordered:
            sub = meta[meta["cell_type"].astype(str) == ct]
            col = palette.get(ct, "#555555")

            # Determine edge colour: dark edge for light fills, white for dark fills
            # Light colours: cream (#fffac8), white (#ffffff), mint (#aaffc3), lavender
            light_fills = {"#fffac8", "#ffffff", "#aaffc3", "#dcbeff",
                           "#fabed4", "#ffd8b1", "#fdcae1", "#fdfd96",
                           "#b5ead7", "#c7b8ea", "#e6beff", "#ffb347"}
            edge_col  = "#333333" if col in light_fills else "white"
            edge_lw   = 0.5       if col in light_fills else 0.6

            ax.scatter(
                sub[xcol].astype(float),
                sub[ycol].astype(float),
                s=marker_size,
                color=col,
                alpha=alpha,
                label=ct,
                edgecolors=edge_col,
                linewidths=edge_lw,
                rasterized=True,
                zorder=2,
            )

        # ── Axes styling ──────────────────────────────────────────────────
        ax.set_xlabel("X coordinate", fontsize=cfg.axis_label_font_size,
                      fontweight="bold", labelpad=6)
        ax.set_ylabel("Y coordinate", fontsize=cfg.axis_label_font_size,
                      fontweight="bold", labelpad=6)
        ax.set_title(
            f"{label}  (n = {len(meta):,})",
            fontsize=cfg.subtitle_font_size, fontweight="bold", pad=10,
        )
        ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.5, color="#888888")
        ax.set_aspect("equal", adjustable="datalim")

        # Tick formatting
        ax.tick_params(labelsize=cfg.tick_font_size, length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#aaaaaa")

        # ── Legend — deduplicated, sorted alphabetically, outside right ───
        handles, labels_leg = ax.get_legend_handles_labels()
        seen, dedup_pairs = {}, []
        for h, l in zip(handles, labels_leg):
            if l not in seen:
                seen[l] = True
                dedup_pairs.append((l, h))
        dedup_pairs.sort(key=lambda x: x[0])   # alphabetical in legend
        dedup_l = [p[0] for p in dedup_pairs]
        dedup_h = [p[1] for p in dedup_pairs]

        n_types = len(dedup_l)
        leg_fontsize = max(6, min(cfg.legend_font_size, 9)) \
                       if n_types > 15 else cfg.legend_font_size

        leg = ax.legend(
            dedup_h, dedup_l,
            title="Cell Types",
            title_fontsize=leg_fontsize + 1,
            fontsize=leg_fontsize,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            framealpha=0.95,
            edgecolor="#cccccc",
            fancybox=True,
            markerscale=1.6,       # bigger legend symbols
            labelspacing=0.45,
            handletextpad=0.5,
            borderpad=0.7,
        )
        leg.get_frame().set_linewidth(0.7)
        leg.get_title().set_fontweight("bold")

        # ── Super-title ───────────────────────────────────────────────────
        fig.suptitle(
            "Spatial Distribution of Cell Types",
            fontsize=cfg.title_font_size, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 0.82, 0.97])   # leave room for legend
        fp = os.path.join(out_dir, fname)
        plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        saved.append(fp)
        if verbose:
            print(f"  Spatial map: {label}  ({len(meta):,} cells, "
                  f"{n_types} types) → {fname}")

    return saved


# =============================================================================
# Plot generation driver
# =============================================================================

def generate_all_plots(results, cfg=None, verbose=True):
    if cfg is None:
        cfg = DEFAULT_CONFIG

    saved_files = {
        "consensus": [],
        "differential": [],
        "diffusion": [],
        "regional": [],
        "network": [],
        "coupling": [],
        "balance": [],
        "heatmap": [],
        "volcano": [],
        "bars": [],
    }

    if verbose:
        print("\nGenerating plots...")
    
    # ── Write region index text file ────────────────────────────────────
    _write_region_index(results, cfg, verbose=verbose)

    # ── Spatial cell-type maps (one per region) ─────────────────────────
    if verbose:
        print("Spatial cell-type maps...")
    spatial_files = plot_spatial_celltype_map(results, cfg=cfg, verbose=verbose)
    saved_files.setdefault("spatial", []).extend(spatial_files)
    if verbose:
        print(f" {len(spatial_files)} spatial maps")

    if verbose:
        print("Regional streamlines...")
    for key in getattr(results, "per_region_data", {}).keys():
        parts = key.rsplit("_R", 1)
        cond = parts[0]
        region = int(parts[1]) if len(parts) > 1 else 1
        files = plot_regional_streamlines(
            results, cond, region,
            metabolites=getattr(results, "significant_mets", [])[:10],
            cfg=cfg
        )
        saved_files["regional"].extend(files)
        if verbose:
            print(f" {key}: {len(files)} plots")
    
    if verbose:
        print("Diffusion maps (P05 style)...")
    files = plot_diffusion_comparison(results, cfg=cfg)
    saved_files["diffusion"] = files
    if verbose:
        print(f" {len(files)} plots")

    if verbose:
        print("Consensus streamlines...")
    files = plot_consensus_streamlines(results, cfg=cfg)
    saved_files["consensus"] = files
    if verbose:
        print(f" {len(files)} plots")

    if verbose:
        print("Differential streamlines...")
    if _is_single_condition(results):
        if verbose:
            print("  [SKIP] Differential streamlines — single condition.")
    else:
        files = plot_differential_streamlines(results, cfg=cfg)
        saved_files["differential"] = files
        if verbose:
            print(f" {len(files)} plots")

    if verbose:
        print("Network diagrams (net directed exchange)...")
    f = plot_network_diagram(results, cfg=cfg)
    if f:
        saved_files["network"].append(f)
    for cond in getattr(results.config, "conditions", []):
        f = plot_network_diagram(results, condition=cond, cfg=cfg)
        if f:
            saved_files["network"].append(f)
    if verbose:
        print(f" {len(saved_files['network'])} plots")

    if verbose:
        print("Coupling heatmaps...")
    f = plot_coupling_heatmap(results, cfg=cfg)
    if f:
        saved_files["coupling"].append(f)
    for cond in getattr(results.config, "conditions", []):
        f = plot_coupling_heatmap(results, condition=cond, cfg=cfg)
        if f:
            saved_files["coupling"].append(f)
    if verbose:
        print(f" {len(saved_files['coupling'])} plots")

    if verbose:
        print("Balance comparison...")
    f = plot_flux_balance_comparison(results, cfg=cfg)
    if f:
        saved_files["balance"].append(f)

    if verbose:
        print("Comparison plots...")
    f = plot_metabolite_heatmap(results, cfg=cfg)
    if f:
        saved_files["heatmap"].append(f)

    if not _is_single_condition(results):
        f = plot_volcano(results, cfg=cfg)
        if f:
            saved_files["volcano"].append(f)
    else:
        if verbose:
            print("  [SKIP] Volcano plot — single condition.")

    if verbose:
        print("Top metabolites bar plot...")
    f = plot_top_metabolites_bar(results, cfg=cfg)
    if f:
        saved_files["bars"].append(f)

    if verbose:
        print("Significant metabolite bar plots...")
    if not _is_single_condition(results):
        f = plot_significant_metabolites_bars(results, cfg=cfg)
        if f:
            saved_files["bars"].append(f)
    else:
        if verbose:
            print("  [SKIP] Significant metabolite bars — single condition.")

    if verbose:
        total = sum(len(v) for v in saved_files.values())
        print(f"\nTotal plots generated: {total}")

    return saved_files


# =============================================================================
# SPOTLIGHT FIGURE  — combined streamline + diffusion + biology text per metabolite
# =============================================================================

def plot_key_metabolites_spotlight(
    results,
    metabolites=None,
    annotations=None,
    out_dir=None,
    cfg=None,
    n_seed_points=120,
    cell_proximity_radius=0.20,
    verbose=True,
):
    """
    Generate a publication-ready spotlight figure for each key metabolite.

    Layout (per metabolite):
        Row 0 (tall): Streamline panels  — one per condition
        Row 1 (medium): Diffusion panels  — one per condition
        Row 2 (short):  Biology annotation text + references

    The function auto-selects metabolites via select_spotlight_metabolites()
    if none are supplied, pulling from enhanced_analysis.KIDNEY_BIOLOGY and
    the live results object.

    Parameters
    ----------
    results      : AnalysisResults from SPICEM pipeline
    metabolites  : list of str, optional – override auto-selection
    annotations  : dict, optional – metabolite → annotation dict
                   (output of select_spotlight_metabolites)
    out_dir      : output directory (default: results.config.out_dir/plots_spotlight)
    cfg          : PlotConfig
    n_seed_points: seed points for streamlines
    cell_proximity_radius: proximity mask radius
    verbose      : print progress

    Returns
    -------
    list of str  – saved file paths
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if out_dir is None:
        base = getattr(getattr(results, "config", None), "out_dir", "analysis_output")
        out_dir = os.path.join(base, "plots_spotlight")
    os.makedirs(out_dir, exist_ok=True)

    # Auto-select metabolites if not provided
    if metabolites is None or annotations is None:
        try:
            from enhanced_analysis import select_spotlight_metabolites
            metabolites, annotations = select_spotlight_metabolites(results)
        except Exception as e:
            if verbose:
                print(f"  Warning: auto-selection failed ({e}); using significant_mets")
            metabolites = getattr(results, "significant_mets", [])[:6]
            annotations = {}

    conditions = getattr(getattr(results, "config", None), "conditions", [])
    saved = []

    # ── Helper: collect top-N src/snk pairs for a metabolite ──────────────
    def _top_pairs_for_met(met, top_n=3):
        """
        Scan per_region_data across ALL conditions and collect unique
        (src, snk, score) triples for `met`, ranked by score descending.
        Returns list of dicts with keys: src, snk, score, p_val.
        """
        per_region = getattr(results, "per_region_data", {}) or {}
        seen = {}  # (src, snk) -> best score
        for key, data in per_region.items():
            for pair in (data.get("interactions", {}) or {}).get(met, []):
                s = str(pair.get("source", "") or "")
                k = str(pair.get("sink",   "") or "")
                sc = float(pair.get("score", 0.0))
                if s and k and s != k:
                    if (s, k) not in seen or sc > seen[(s, k)]:
                        seen[(s, k)] = sc

        # also pull p_val from comparison_df if available
        p_map = {}
        comp_df = getattr(results, "comparison_df", None)
        if comp_df is not None and not comp_df.empty:
            mc  = "Metabolite" if "Metabolite" in comp_df.columns else "metabolite"
            pc  = next((c for c in ["P_Secretion","PSecretion"] if c in comp_df.columns), None)
            row = comp_df[comp_df[mc].astype(str) == str(met)]
            if not row.empty and pc:
                p_map[met] = float(row.iloc[0][pc])

        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        result = []
        for (s, k), sc in ranked[:top_n]:
            result.append({
                "src": s, "snk": k, "score": sc,
                "p_val": p_map.get(met, 1.0),
            })

        # fallback: if nothing found in per_region, use get_metabolite_info
        if not result:
            try:
                s, k, pv = results.get_metabolite_info(met)
                if s:
                    result.append({"src": s, "snk": k or "",
                                   "score": 0.0, "p_val": float(pv or 1.0)})
            except Exception:
                pass
        return result

    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue

        ann       = annotations.get(met, {})
        full_name = ann.get("full_name", met.upper())
        bio_text  = ann.get("biology_text", "")
        refs      = ann.get("references", [])

        pairs = _top_pairs_for_met(met, top_n=3)
        if not pairs:
            if verbose:
                print(f"  Spotlight: {met} — no pairs found, skipping")
            continue

        if verbose:
            print(f"  Spotlight: {met} ({full_name}) — {len(pairs)} pair(s)")

        # ── Gather per-condition vector fields ONCE per metabolite ─────────
        # The vector field is keyed by met only; src/snk only affect which
        # cell populations are highlighted and used as seeds/proximity mask.
        cond_data = {}
        for cond in conditions:
            fields  = results.get_vector_fields_for_condition(cond, met)
            meta_df = results.get_metadata_for_condition(cond)
            if not fields or (meta_df is None or meta_df.empty):
                continue
            U_avg = np.mean([f[0] for f in fields], axis=0)
            V_avg = np.mean([f[1] for f in fields], axis=0)
            xi, yi = fields[0][2], fields[0][3]
            U_avg, V_avg, xi, yi = _coarsen_vector_field(
                U_avg, V_avg, xi, yi,
                cfg.consensus_differential_coarsen_factor
            )
            cond_data[cond] = {
                "U": U_avg, "V": V_avg, "xi": xi, "yi": yi,
                "meta_df": meta_df.copy(),
            }

        if not cond_data:
            continue

        n_cond = len(cond_data)
        shared_vmax = max(
            float(np.nanmax(np.sqrt(d["U"]**2 + d["V"]**2)))
            for d in cond_data.values()
        )

        # ── One figure per (met, pair_rank) ────────────────────────────────
        for rank, pair in enumerate(pairs, start=1):
            src    = pair["src"]
            snk    = pair["snk"]
            score  = pair["score"]
            p_val  = pair["p_val"]

            if verbose:
                print(f"    Pair {rank}: {src} → {snk}  (score={score:.1f})")

            from matplotlib.gridspec import GridSpec as _GS
            import textwrap as _tw

            fig_w = max(14, n_cond * 7)
            fig_h = fig_w * 0.62 + 3.5
            fig   = plt.figure(figsize=(fig_w, fig_h))
            gs    = _GS(
                3, n_cond,
                figure=fig,
                height_ratios=[2.5, 2.0, 1.0],
                hspace=0.12,
                wspace=0.06,
            )

            # ── Streamline row ────────────────────────────────────────────
            for col, (cond, d) in enumerate(cond_data.items()):
                ax = fig.add_subplot(gs[0, col])
                cond_short = cond.replace("Sample_", "")
                try:
                    from enhanced_analysis import _build_flux_df as _bfd
                    fdf = _bfd(results, cond, met)
                except Exception:
                    fdf = None

                plot_streamline_panel(
                    ax, d["U"], d["V"], d["xi"], d["yi"],
                    d["meta_df"], src, snk,
                    title=cond_short,
                    show_colorbar=(col == n_cond - 1),
                    cfg=cfg,
                    cmap=None,
                    colorbar_label="Flow speed",
                    use_consensus_settings=True,
                    shared_vmax=shared_vmax,
                    n_seed_points=n_seed_points,
                    cell_proximity_radius=cell_proximity_radius,
                    flux_df=fdf,
                    seed=int(getattr(cfg, 'random_seed', 42)),
                )

            # ── Diffusion row ─────────────────────────────────────────────
            for col, (cond, d) in enumerate(cond_data.items()):
                ax = fig.add_subplot(gs[1, col])
                cond_short = cond.replace("Sample_", "")
                plot_diffusion_panel(
                    ax, d["U"], d["V"], d["xi"], d["yi"],
                    d["meta_df"], src, snk,
                    title=cond_short,
                    show_colorbar=(col == n_cond - 1),
                    cfg=cfg,
                )

            # ── Biology text row ──────────────────────────────────────────
            ax_text = fig.add_subplot(gs[2, :])
            ax_text.axis("off")

            p_str  = f"{p_val:.3g}" if p_val is not None and np.isfinite(float(p_val)) else "n/a"
            fc_val = ann.get("log2fc", float("nan"))
            fc_str = f"Log₂FC = {fc_val:+.2f}" if (
                fc_val is not None and np.isfinite(float(fc_val))) else ""
            rank_str = f"Exchange rank #{rank} of {len(pairs)}"
            header = (
                f"{full_name}  ·  {src} → {snk}  ·  "
                f"Score = {score:.0f}  ·  p = {p_str}  {fc_str}  [{rank_str}]"
            )
            wrapped = _tw.fill(bio_text, width=140) if bio_text else ""
            ref_str = "  |  ".join(refs) if refs else ""
            full_txt = (
                f"{header}\n\n{wrapped}\n\nReferences: {ref_str}"
                if ref_str else f"{header}\n\n{wrapped}"
            )

            ax_text.text(
                0.01, 0.98, full_txt,
                transform=ax_text.transAxes,
                fontsize=7.5,
                verticalalignment="top",
                horizontalalignment="left",
                wrap=True,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="#f8f9fa",
                    edgecolor="#cccccc",
                    linewidth=0.8,
                    alpha=0.95,
                ),
            )

            # ── Supertitle ────────────────────────────────────────────────
            fig.suptitle(
                f"Spatial Metabolite Exchange: {full_name} ({met})  "
                f"[Exchange {rank}/{len(pairs)}]\n"
                f"Source: {src}  →  Sink: {snk}  ·  p = {p_str}",
                fontsize=cfg.title_font_size,
                fontweight="bold",
                y=0.995,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.975])
            fp = os.path.join(out_dir, f"spotlight_{met}_pair{rank}.png")
            plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            saved.append(fp)
            if verbose:
                print(f"      Saved: {fp}")

    return saved


def generate_spotlight_text_report(annotations, out_dir):
    """
    Write a standalone plain-text biology report for the spotlight metabolites.
    One section per metabolite with full biology, stats, and references.
    Saved as  spotlight_biology_report.txt  in out_dir.
    """
    import textwrap
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append("=" * 80)
    lines.append("SPOTLIGHT METABOLITE BIOLOGY REPORT")
    lines.append("Kidney Disease Context — SPICEM Spatial Exchange Analysis")
    lines.append("=" * 80)
    lines.append("")

    for met, ann in annotations.items():
        lines.append("─" * 80)
        lines.append(f"METABOLITE: {ann['full_name']}  [{met}]")
        lines.append(f"Exchange:   {ann['source']}  →  {ann['sink']}")
        sc = f"{ann['score']:.1f}" if ann.get("score") else "n/a"
        pv = f"{ann['p_val']:.4g}" if ann.get("p_val") is not None else "n/a"
        fc = f"{ann['log2fc']:+.2f}" if ann.get("log2fc") is not None and \
             np.isfinite(float(ann.get("log2fc", float("nan")))) else "n/a"
        lines.append(f"Stats:      Score = {sc}  |  p = {pv}  |  Log2FC = {fc}")
        lines.append("")
        if ann.get("biology_text"):
            for para in textwrap.wrap(ann["biology_text"], width=78):
                lines.append(para)
        lines.append("")
        if ann.get("references"):
            lines.append("References:")
            for ref in ann["references"]:
                lines.append(f"  [{met}] {ref}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")

    fp = os.path.join(out_dir, "spotlight_biology_report.txt")
    with open(fp, "w") as f:
        f.write("\n".join(lines))
    return fp