#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS - CORE MODULE (FINAL FIXED VERSION)
============================================

COMPLETE FIXES:
1. Magnitude-preserving vector fields (not normalized to unit vectors)
2. Support for proper vector field averaging across regions
3. Coarsening/binning support for reducing arrow crowding
4. Helper methods for consensus and differential plots

Contains:
1) Data loading (robust)
2) Flux aggregation & enrichment
3) Interactions & coupling
4) Potential-flow vector field generation (magnitude-preserving with coarsening)
5) Permutation statistics
6) Condition comparison utilities
7) Vector field averaging and coarsening utilities
"""

import os
import re
import json
import warnings
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu, chi2
from scipy.ndimage import gaussian_filter

from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION (legacy; pipeline.py defines its own AnalysisConfig)
# =============================================================================

@dataclass
class SpatialAnalysisConfig:
    base_dir: str = ""
    out_dir: str = "analysis_output"
    n_neighbors: int = 30
    spatial_radius: float = 100.0
    significance_threshold: float = 0.05
    fold_change_threshold: float = 0.5
    grid_size: int = 100
    potential_sigma: float = 4.0
    conditions: List[str] = field(default_factory=list)
    n_workers: int = 1


@dataclass
class SpatialAnalysisResult:
    config: SpatialAnalysisConfig
    significant_mets: List[str] = field(default_factory=list)
    per_region_data: Dict[str, Any] = field(default_factory=dict)
    comparison_df: pd.DataFrame = None
    condition_balance: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Minimal helpers (kept for compatibility with some older plotting paths)
    def get_metabolite_info(self, met: str) -> Tuple[Optional[str], Optional[str], float]:
        """Get source, sink, and p-value for a metabolite from comparison_df."""
        if self.comparison_df is None or self.comparison_df.empty:
            return None, None, 1.0
        row = self.comparison_df[self.comparison_df["Metabolite"] == met]
        if row.empty:
            return None, None, 1.0
        
        # Try to get source/sink from best_pairs if available
        src, snk = None, None
        for key, data in self.per_region_data.items():
            best_pairs = data.get("best_pairs", {})
            if met in best_pairs:
                src, snk, _ = best_pairs[met]
                break
        
        p_val = float(row.iloc[0].get("PSecretion", 1.0))
        return src, snk, p_val

    def get_metadata_for_condition(self, condition: str) -> pd.DataFrame:
        """Get combined metadata for all regions of a condition."""
        dfs = []
        for key, data in self.per_region_data.items():
            if key.startswith(condition):
                dfs.append(data.get("meta_df", pd.DataFrame()))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def get_vector_fields_for_condition(self, condition: str, metabolite: str) -> List[Tuple]:
        """Get all vector fields for a metabolite across all regions of a condition."""
        fields = []
        for key, data in self.per_region_data.items():
            if key.startswith(condition):
                vector_fields = data.get("vector_fields", {})
                if metabolite in vector_fields:
                    fields.append(vector_fields[metabolite])
        return fields


# =============================================================================
# DATA LOADING
# =============================================================================

def load_flux_txt(path: str) -> pd.DataFrame:
    """
    Loads exchange flux file. Works with both TSV and CSV outputs.
    Keeps only rows with abs(Flux) > 1e-6.
    """
    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=",")
        if "Flux" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["Flux"] = pd.to_numeric(df["Flux"], errors="coerce")
        df = df.dropna(subset=["Flux"])
        return df[df["Flux"].abs() > 1e-6]
    except Exception:
        return pd.DataFrame()


def get_model_to_cell_map(grid: dict) -> dict:
    """
    Map model IDs (model1, model2, ...) to barcodes for a given community grid.
    model1 is center_cell, model2.. are neighbors in the JSON ordering.
    """
    model_map = {"model1": grid["center_cell"]}
    for i, nb in enumerate(grid["neighbor_cells"]):
        model_map[f"model{i+2}"] = nb
    return model_map


def extract_metabolite(rxn_side: str) -> Optional[str]:
    """
    Parse metabolite ID from reaction side strings like 'model3_lac_L[e]'.
    """
    match = re.search(r"model\d+_(\w+)\[(e|u)\]", str(rxn_side))
    return match.group(1) if match else None


def load_region_data(
    base_dir: str,
    condition: str,
    region_idx: int,
    grid_pattern: str = "communities_hexagonal_Region_{}.json",
    flux_folder: str = "exchange_fluxes",
):
    """
    Loads:
    - community grids JSON
    - metadata.csv (expects x/y + Graph.based columns; renames to px_x/px_y/cell_type)
    - flux_dir (folder of comm_{center_cell}.txt)
    """
    region_dir = os.path.join(base_dir, condition, f"Region_{region_idx}")

    # Grid JSON
    grid_json = os.path.join(region_dir, grid_pattern.format(region_idx))
    if not os.path.exists(grid_json):
        grid_json = os.path.join(region_dir, f"communities_Region_{region_idx}.json")
    if not os.path.exists(grid_json):
        raise FileNotFoundError(f"Grid JSON missing: {region_dir}")

    with open(grid_json, "r") as f:
        grids = json.load(f)

    # Metadata
    meta_csv = os.path.join(region_dir, "metadata.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata CSV missing: {meta_csv}")

    meta_df = pd.read_csv(meta_csv)

    # ── Standardize column names ──────────────────────────────────────────
    # Handles multiple naming conventions in priority order:
    #   Visium/Seurat exports  : pxl_col / pxl_row / Idents / barcode_clean
    #   Older pipeline exports : x / y / Graph.based / barcode
    rename_map = {}

    # x coordinate
    for col in ["pxl_col", "x", "coord_x", "pos_x"]:
        if col in meta_df.columns:
            rename_map[col] = "px_x"
            break

    # y coordinate
    for col in ["pxl_row", "y", "coord_y", "pos_y"]:
        if col in meta_df.columns:
            rename_map[col] = "px_y"
            break

    # cell type label
    for col in ["Idents", "Graph.based", "celltype", "CellType",
                "cell_type_annotation", "annotation"]:
        if col in meta_df.columns:
            rename_map[col] = "cell_type"
            break

    meta_df = meta_df.rename(columns=rename_map)

    # Basic validation
    for required in ["px_x", "px_y", "cell_type"]:
        if required not in meta_df.columns:
            raise KeyError(
                f"metadata.csv missing required column: {required}\n"
                f"Available columns: {list(meta_df.columns)}\n"
                f"Expected one of: x/pxl_col (→px_x), y/pxl_row (→px_y), "
                f"Graph.based/Idents (→cell_type)"
            )

    meta_df = meta_df.dropna(subset=["px_x", "px_y", "cell_type"]).copy()
    meta_df["cell_type"] = meta_df["cell_type"].astype(str)
    meta_df["px_x"] = pd.to_numeric(meta_df["px_x"], errors="coerce")
    meta_df["px_y"] = pd.to_numeric(meta_df["px_y"], errors="coerce")
    meta_df = meta_df.dropna(subset=["px_x", "px_y"]).copy()

    # ── Set barcode as index ──────────────────────────────────────────────
    # Priority: barcode_clean (no sample suffix, e.g. AAACAAGTATCTCCCA-1)
    #        >  barcode (may have sample suffix, e.g. AAACAAGTATCTCCCA-1_9)
    #        >  first unnamed column (R/Seurat CSV row names)
    # barcode_clean is preferred because it matches the community JSON
    # center_cell / neighbor_cells values and comm_<barcode>.txt filenames.
    if "barcode_clean" in meta_df.columns:
        meta_df["barcode_clean"] = meta_df["barcode_clean"].astype(str).str.strip()
        meta_df = meta_df.set_index("barcode_clean", drop=True)
    elif "barcode" in meta_df.columns:
        meta_df["barcode"] = meta_df["barcode"].astype(str).str.strip()
        meta_df = meta_df.set_index("barcode", drop=True)
    elif meta_df.columns[0] not in ["px_x", "px_y", "cell_type"]:
        # First column is likely the unnamed row-index from R's write.csv()
        meta_df = meta_df.set_index(meta_df.columns[0], drop=True)

    meta_df.index = meta_df.index.astype(str).str.strip()

    # Flux folder
    flux_dir = os.path.join(region_dir, flux_folder)
    if not os.path.isdir(flux_dir):
        raise FileNotFoundError(f"Flux dir missing: {flux_dir}")

    return grids, meta_df, flux_dir


# =============================================================================
# FLUX AGGREGATION
# =============================================================================

def aggregate_fluxes(grids, meta_df: pd.DataFrame, flux_dir: str):
    """
    Parses comm_{center_cell}.txt for each grid. Builds:
    - met_fluxes[met]["secretion"][cell_type] = list of fluxes
    - met_fluxes[met]["uptake"][cell_type] = list of fluxes
    - cell_met_pool[cell_id][met] = signed net (sec positive, upt negative)
    - neighbor_map[cell_id] = set(neighbor cell_ids)
    """
    met_fluxes = defaultdict(lambda: {"secretion": defaultdict(list), "uptake": defaultdict(list)})
    cell_met_pool = defaultdict(dict)
    neighbor_map = defaultdict(set)

    files_found, exchanges_found = 0, 0
    rxn_cols = ["Reaction_Equation", "Reaction", "reaction_equation", "reaction"]

    for grid in grids:
        center_cell = grid["center_cell"]

        # Build undirected neighbor graph at cell/barcode level
        for nb in grid.get("neighbor_cells", []):
            neighbor_map[center_cell].add(nb)
            neighbor_map[nb].add(center_cell)

        flux_path = os.path.join(flux_dir, f"comm_{center_cell}.txt")
        if not os.path.exists(flux_path):
            continue

        files_found += 1
        flux_df = load_flux_txt(flux_path)
        if flux_df.empty:
            continue

        m_map = get_model_to_cell_map(grid)

        for _, row in flux_df.iterrows():
            rxn = None
            for col in rxn_cols:
                if col in row.index and pd.notna(row.get(col)):
                    rxn = str(row[col])
                    break
            if not rxn or "->" not in rxn:
                continue

            flux = float(row["Flux"])
            lhs, rhs = [s.strip() for s in rxn.split("->")]

            met, mid, mode = None, None, None
            # secretion: [e] on lhs and [u] on rhs
            if ("[e]" in lhs) and ("[u]" in rhs):
                met = extract_metabolite(lhs)
                mid = lhs.split("_")[0]
                mode = "secretion"
            # uptake: [u] on lhs and [e] on rhs
            elif ("[u]" in lhs) and ("[e]" in rhs):
                met = extract_metabolite(rhs)
                mid = rhs.split("_")[0]
                mode = "uptake"
            else:
                continue

            if not met or (mid not in m_map):
                continue

            cid = m_map[mid]
            if cid not in meta_df.index:
                continue

            exchanges_found += 1
            cell_type = meta_df.loc[cid, "cell_type"]

            met_fluxes[met][mode][cell_type].append(flux)

            cur = float(cell_met_pool[cid].get(met, 0.0))
            cell_met_pool[cid][met] = (cur + flux) if (mode == "secretion") else (cur - flux)

    # Convert nested defaultdicts to plain dicts for serialization
    met_fluxes_clean = {}
    for met, d in met_fluxes.items():
        met_fluxes_clean[met] = {"secretion": dict(d["secretion"]), "uptake": dict(d["uptake"])}

    return met_fluxes_clean, dict(cell_met_pool), dict(neighbor_map), files_found, exchanges_found


# =============================================================================
# ENRICHMENT & COUPLING
# =============================================================================

def find_enriched_metabolites(met_fluxes: Dict, p_threshold: float = 0.05):
    """
    Identifies metabolites where secretion or uptake differs across cell types via Kruskal-Wallis,
    then selects a candidate metabolite if:
      - best secretion cell type exists
      - best uptake cell type exists
      - secretion_best != uptake_best
    Candidate score = |mean(secretion_best)| * |mean(uptake_best)|

    Returns:
      enriched_df with columns: met, sec, upt, val, p_sec, p_upt (sorted by val desc)
      enrichment_info dict per metabolite (best types + p-values)
    """
    candidates = []
    enrichment_info = {}

    for met, d in met_fluxes.items():
        info = {"secretion_best": None, "uptake_best": None, "p_sec": 1.0, "p_upt": 1.0}

        for mode, key_p, key_best in [
            ("secretion", "p_sec", "secretion_best"),
            ("uptake", "p_upt", "uptake_best"),
        ]:
            if len(d.get(mode, {})) >= 2:
                try:
                    groups = [f for f in d[mode].values() if len(f) > 0]
                    if len(groups) >= 2:
                        _, p = kruskal(*groups)
                        info[key_p] = float(p)
                        if p < p_threshold:
                            info[key_best] = max(
                                d[mode].items(),
                                key=lambda x: abs(np.mean(x[1])) if len(x[1]) else 0.0,
                            )[0]
                except Exception:
                    pass

        enrichment_info[met] = info

        if info["secretion_best"] and info["uptake_best"] and (info["secretion_best"] != info["uptake_best"]):
            sec_val = abs(np.mean(d["secretion"][info["secretion_best"]]))
            upt_val = abs(np.mean(d["uptake"][info["uptake_best"]]))
            candidates.append(
                {
                    "met": met,
                    "sec": info["secretion_best"],
                    "upt": info["uptake_best"],
                    "val": float(sec_val * upt_val),
                    "p_sec": float(info["p_sec"]),
                    "p_upt": float(info["p_upt"]),
                }
            )

    df = pd.DataFrame(candidates)
    if not df.empty:
        df = df.sort_values("val", ascending=False).reset_index(drop=True)

    return df, enrichment_info


def compute_spatial_coupling_matrix(meta_df: pd.DataFrame, neighbor_map: Dict[str, set]):
    """
    Computes coupling between cell types based on neighbor edges.

    For each unordered pair (A,B):
      coupling(A,B) = edge_count(A,B) / sqrt(count(A)*count(B))
    """
    edge_counts: Dict[Tuple[str, str], int] = {}
    type_counts = meta_df["cell_type"].value_counts().to_dict()

    for cell_a, neighbors in neighbor_map.items():
        if cell_a not in meta_df.index:
            continue
        ta = meta_df.loc[cell_a, "cell_type"]
        for cell_b in neighbors:
            if cell_b not in meta_df.index:
                continue
            tb = meta_df.loc[cell_b, "cell_type"]
            key = tuple(sorted((ta, tb)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    coupling = {}
    for (a, b), count in edge_counts.items():
        norm = np.sqrt(type_counts.get(a, 1) * type_counts.get(b, 1))
        coupling[(a, b)] = float(count / norm) if norm > 0 else 0.0

    return coupling


def analyze_all_interactions(met_fluxes: Dict, coupling_matrix: Dict, min_coupling: float = 0.001):
    """
    For each metabolite:
      score(src->snk) = |mean_sec(src)| * |mean_upt(snk)| * coupling(src,snk)
    Returns:
      interactions[met] = list of dicts sorted by score desc
      best_pairs[met] = (best_src, best_snk, best_score)
    """
    interactions: Dict[str, List[dict]] = {}
    best_pairs: Dict[str, Tuple[str, str, float]] = {}

    for met, d in met_fluxes.items():
        pairs = []
        for src in d.get("secretion", {}):
            for snk in d.get("uptake", {}):
                if src == snk:
                    continue
                coup = float(coupling_matrix.get(tuple(sorted((src, snk))), 0.0))
                if coup < min_coupling:
                    continue
                msec = float(np.mean(d["secretion"][src])) if len(d["secretion"][src]) else 0.0
                mupt = float(np.mean(d["uptake"][snk])) if len(d["uptake"][snk]) else 0.0
                score = float(abs(msec) * abs(mupt) * coup)
                if score > 1e-9:
                    pairs.append({"source": src, "sink": snk, "score": score, "coupling": coup})

        pairs.sort(key=lambda x: x["score"], reverse=True)
        interactions[met] = pairs
        if pairs:
            best_pairs[met] = (pairs[0]["source"], pairs[0]["sink"], float(pairs[0]["score"]))

    return interactions, best_pairs


# =============================================================================
# VECTOR FIELD (POTENTIAL FLOW) - MAGNITUDE PRESERVING WITH COARSENING
# =============================================================================

def compute_potential_flow_field(
    meta_df: pd.DataFrame,
    src_type: str,
    snk_type: str,
    grid_size: int = 100,
    sigma: float = 4.0,
    coarsen_factor: int = 1,
):
    """
    Builds a smooth potential field from source and sink cell densities.
    
    **CRITICAL FIX**: This version PRESERVES MAGNITUDE information by NOT 
    normalizing to unit vectors. The speed (magnitude) now represents actual
    flux potential strength, which is essential for proper colormap scaling.
    
    **NEW**: coarsen_factor parameter allows reducing arrow density by binning.
    coarsen_factor=2 means we average every 2x2 block, resulting in 50x50 grid
    from original 100x100, reducing arrows by 75%.

    IMPORTANT:
    - np.histogram2d expects bin EDGES.
    - We return xi/yi as bin CENTERS (len grid_size) for matplotlib.streamplot.
    - U,V returned are shape (len(yi), len(xi)) == (grid_size, grid_size).
    - U,V preserve magnitude information (not normalized to unit vectors)
    - With coarsen_factor > 1, output grid is reduced appropriately

    Returns: U, V, xi, yi
    """
    if meta_df is None or meta_df.empty:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.array([]), np.array([])

    x = meta_df["px_x"].astype(float).values
    y = meta_df["px_y"].astype(float).values
    if len(x) == 0:
        return np.zeros((1, 1)), np.zeros((1, 1)), np.array([]), np.array([])

    pad_x = (x.max() - x.min()) * 0.1 if x.max() > x.min() else 1.0
    pad_y = (y.max() - y.min()) * 0.1 if y.max() > y.min() else 1.0

    # EDGES (len = grid_size + 1)
    x_edges = np.linspace(x.min() - pad_x, x.max() + pad_x, int(grid_size) + 1)
    y_edges = np.linspace(y.min() - pad_y, y.max() + pad_y, int(grid_size) + 1)

    # CENTERS (len = grid_size)
    xi = 0.5 * (x_edges[:-1] + x_edges[1:])
    yi = 0.5 * (y_edges[:-1] + y_edges[1:])

    src_mask = (meta_df["cell_type"] == src_type).values
    snk_mask = (meta_df["cell_type"] == snk_type).values

    if (not np.any(src_mask)) or (not np.any(snk_mask)):
        U0 = np.zeros((len(yi), len(xi)), dtype=float)
        V0 = np.zeros((len(yi), len(xi)), dtype=float)
        return U0, V0, xi, yi

    # histogram2d -> shape (grid_size, grid_size)
    H_src, _, _ = np.histogram2d(y[src_mask], x[src_mask], bins=[y_edges, x_edges])
    H_snk, _, _ = np.histogram2d(y[snk_mask], x[snk_mask], bins=[y_edges, x_edges])

    # Create potential field with Gaussian smoothing
    potential = gaussian_filter(H_src, float(sigma)) - gaussian_filter(H_snk, float(sigma))

    # Compute gradient (negative gradient gives flow direction)
    dV, dU = np.gradient(potential)  # (d/dy, d/dx)
    U, V = -dU, -dV

    # **CRITICAL FIX**: DO NOT normalize to unit vectors!
    # Keep the actual magnitude which represents flux potential strength
    
    # Optional: Apply a gentle cap to prevent extreme outliers from dominating
    # but preserve relative magnitudes
    speed = np.sqrt(U**2 + V**2)
    speed_99 = np.percentile(speed[speed > 0], 99) if np.any(speed > 0) else 1.0
    
    # Cap at 99th percentile but preserve relative strengths below that
    cap_factor = np.minimum(speed_99 / (speed + 1e-12), 1.0)
    U = U * cap_factor
    V = V * cap_factor
    
    # COARSENING: bin/average to reduce arrow density if requested
    if coarsen_factor > 1:
        cf = int(coarsen_factor)
        U, V, xi, yi = _coarsen_vector_field(U, V, xi, yi, cf)

    return U, V, xi, yi


def _coarsen_vector_field(U, V, xi, yi, factor):
    """
    Coarsen a vector field by averaging blocks of size factor x factor.
    This reduces the number of arrows while preserving overall flow patterns.
    
    Example: factor=2 converts 100x100 grid to 50x50 grid (75% fewer arrows)
             factor=3 converts 100x100 grid to 33x33 grid (89% fewer arrows)
    """
    ny, nx = U.shape
    
    # Calculate new dimensions
    new_ny = ny // factor
    new_nx = nx // factor
    
    # Trim to multiple of factor
    U_trim = U[:new_ny * factor, :new_nx * factor]
    V_trim = V[:new_ny * factor, :new_nx * factor]
    
    # Reshape and average
    U_coarse = U_trim.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))
    V_coarse = V_trim.reshape(new_ny, factor, new_nx, factor).mean(axis=(1, 3))
    
    # Coarsen coordinate arrays (take center of each block)
    xi_coarse = xi[factor//2::factor][:new_nx]
    yi_coarse = yi[factor//2::factor][:new_ny]
    
    return U_coarse, V_coarse, xi_coarse, yi_coarse


# =============================================================================
# STATISTICS (PERMUTATION + COMBINATION)
# =============================================================================

def fishers_method(p_values_list: List[float]) -> float:
    parr = np.clip(np.array(p_values_list, dtype=float), 1e-300, 1.0)
    X2 = -2.0 * float(np.sum(np.log(parr)))
    return float(1.0 - chi2.cdf(X2, df=2 * len(parr)))


def combine_permutation_results(per_region_perm_p: Dict[str, Dict[str, Dict[Tuple[str, str], float]]]):
    """
    Combine per-region permutation p-values for each metabolite + (src,snk) pair using Fisher's method.
    Input structure:
      per_region_perm_p[region_key][met][(src,snk)] = p
    Output:
      globalp[met][(src,snk)] = combined p
    """
    metpairpvals = defaultdict(dict)

    for _region_key, permdict in per_region_perm_p.items():
        for met, pairs in (permdict or {}).items():
            for pair, p in (pairs or {}).items():
                metpairpvals[met].setdefault(pair, []).append(float(p))

    globalp = {}
    for met, pairs in metpairpvals.items():
        globalp[met] = {
            pair: fishers_method(plist) if len(plist) > 1 else float(plist[0])
            for pair, plist in pairs.items()
        }

    return globalp


def permutation_test_vectorized(
    met_fluxes: Dict,
    meta_df: pd.DataFrame,
    neighbor_map: Dict[str, set],
    best_pairs: Dict[str, Tuple[str, str, float]],
    n_iter: int = 500,
    seed: int = 42,
):
    """
    Permutation test for spatial coupling of best (src,snk) pairs.
    We permute cell_type labels over cells, recompute coupling for each iteration,
    and compute p = P(coupling_null >= coupling_real).
    """
    if not best_pairs:
        return {}

    rng = np.random.default_rng(int(seed))

    coupling_real = compute_spatial_coupling_matrix(meta_df, neighbor_map)

    labels = meta_df["cell_type"].values
    unique_types = sorted(set(labels))
    type_to_code = {t: i for i, t in enumerate(unique_types)}
    label_codes = np.array([type_to_code[t] for t in labels], dtype=np.int32)

    barcodes = meta_df.index.tolist()
    b_to_i = {b: i for i, b in enumerate(barcodes)}

    edge_a, edge_b = [], []
    for ca, nbs in neighbor_map.items():
        if ca not in b_to_i:
            continue
        ia = b_to_i[ca]
        for cb in nbs:
            if cb in b_to_i:
                edge_a.append(ia)
                edge_b.append(b_to_i[cb])
    edge_a = np.array(edge_a, dtype=np.int32)
    edge_b = np.array(edge_b, dtype=np.int32)

    targets = []
    for met, (src, snk, score) in best_pairs.items():
        if (src in type_to_code) and (snk in type_to_code):
            real_c = float(coupling_real.get(tuple(sorted((src, snk))), 0.0))
            targets.append((met, src, snk, float(score), real_c))

    if not targets:
        return {}

    null_dist = defaultdict(list)

    for _ in range(int(n_iter)):
        shuffled = label_codes.copy()
        rng.shuffle(shuffled)

        counts = defaultdict(int)
        et_a, et_b = shuffled[edge_a], shuffled[edge_b]
        for i in range(len(et_a)):
            key = tuple(sorted((int(et_a[i]), int(et_b[i]))))
            counts[key] += 1

        type_counts = np.bincount(shuffled, minlength=len(unique_types))

        for met, src, snk, _score, _real_c in targets:
            sc, kc = type_to_code[src], type_to_code[snk]
            norm = np.sqrt(float(type_counts[sc]) * float(type_counts[kc]))
            c_null = float(counts.get(tuple(sorted((sc, kc))), 0) / norm) if norm > 0 else 0.0
            null_dist[(met, src, snk)].append(c_null)

    pvals: Dict[str, Dict[Tuple[str, str], float]] = {}
    for met, src, snk, _score, real_c in targets:
        nulls = np.array(null_dist[(met, src, snk)], dtype=float)
        p = float((np.sum(nulls >= float(real_c)) + 1) / (len(nulls) + 1))
        pvals.setdefault(met, {})[(src, snk)] = p

    return pvals


# =============================================================================
# CONDITION COMPARISON
# =============================================================================

def compare_conditions(per_region_met_fluxes: Dict[str, dict], conditions: List[str]) -> pd.DataFrame:
    """
    Compare Injured vs Healthy (or any two conditions) for each metabolite using Mann-Whitney U
    on pooled flux samples across all regions for each condition.

    Output columns:
      Metabolite, Log2FCSecretion, PSecretion, Log2FCUptake, PUptake
    """
    if conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    cond_fluxes = {cond0: defaultdict(lambda: {"sec": [], "upt": []}),
                   cond1: defaultdict(lambda: {"sec": [], "upt": []})}

    for key, met_fluxes in (per_region_met_fluxes or {}).items():
        cond = key.rsplit("_R", 1)[0]
        if cond not in cond_fluxes:
            continue
        for met, d in (met_fluxes or {}).items():
            for _ct, fs in d.get("secretion", {}).items():
                cond_fluxes[cond][met]["sec"].extend(list(fs))
            for _ct, fs in d.get("uptake", {}).items():
                cond_fluxes[cond][met]["upt"].extend(list(fs))

    rows = []
    all_mets = set(cond_fluxes[cond0].keys()) | set(cond_fluxes[cond1].keys())

    for met in sorted(all_mets):
        d0 = cond_fluxes[cond0][met]
        d1 = cond_fluxes[cond1][met]

        # Secretion
        try:
            _, psec = mannwhitneyu(d0["sec"], d1["sec"])
            psec = float(psec)
        except Exception:
            psec = 1.0

        m0s = float(np.mean(d0["sec"])) if len(d0["sec"]) else 0.0
        m1s = float(np.mean(d1["sec"])) if len(d1["sec"]) else 0.0
        l2fc_sec = float(np.log2((abs(m1s) + 1e-9) / (abs(m0s) + 1e-9)))

        # Uptake
        try:
            _, pupt = mannwhitneyu(d0["upt"], d1["upt"])
            pupt = float(pupt)
        except Exception:
            pupt = 1.0

        m0u = float(np.mean(d0["upt"])) if len(d0["upt"]) else 0.0
        m1u = float(np.mean(d1["upt"])) if len(d1["upt"]) else 0.0
        l2fc_upt = float(np.log2((abs(m1u) + 1e-9) / (abs(m0u) + 1e-9)))

        rows.append(
            {
                "Metabolite": met,
                "Log2FCSecretion": l2fc_sec,
                "PSecretion": psec,
                "Log2FCUptake": l2fc_upt,
                "PUptake": pupt,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# (Optional legacy helper) Single region wrapper
# =============================================================================

def process_region(region_id: str, config: SpatialAnalysisConfig):
    """
    Legacy single-region wrapper (kept because some notebooks used it).
    """
    try:
        cond, idx = region_id.rsplit("_R", 1)
        idx = int(idx)
        grids, meta_df, flux_dir = load_region_data(config.base_dir, cond, idx)
    except Exception as e:
        print(f"Error loading {region_id}: {e}")
        return None

    met_fluxes, _cell_pool, neighbor_map, _files_found, _exchanges_found = aggregate_fluxes(grids, meta_df, flux_dir)
    enriched, _ = find_enriched_metabolites(met_fluxes)
    coupling = compute_spatial_coupling_matrix(meta_df, neighbor_map)
    interactions, best_pairs = analyze_all_interactions(met_fluxes, coupling)

    vector_fields = {}
    top_mets = enriched["met"].head(20).tolist() if (enriched is not None and not enriched.empty) else []
    for met in top_mets:
        if met not in best_pairs and (enriched is None or enriched.empty):
            continue
        if met in best_pairs:
            src, snk, _ = best_pairs[met]
        else:
            row = enriched[enriched["met"] == met].iloc[0]
            src, snk = row["sec"], row["upt"]
        if src and snk:
            U, V, xi, yi = compute_potential_flow_field(
                meta_df, src, snk, 
                config.grid_size, 
                config.potential_sigma,
                coarsen_factor=1  # No coarsening in initial generation
            )
            vector_fields[met] = (U, V, xi, yi)

    return {
        "id": region_id,
        "meta_df": meta_df,
        "met_fluxes": met_fluxes,
        "enriched_df": enriched,
        "interactions": interactions,
        "best_pairs": best_pairs,
        "coupling": coupling,
        "vector_fields": vector_fields,
    }