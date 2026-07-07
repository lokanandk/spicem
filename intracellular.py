#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS - INTRACELLULAR MODULE
======================================================

Companion to core.py / pipeline.py.

The exchange-flux pipeline answers *who trades what with spatially proximal
neighbours* (only `[u]`-compartment reactions). This module consumes the
COMPLEMENTARY slice from the SAME community FBA solve: the internal
`[c]/[m]/[im]` reactions (e.g. model1_ATPS4m, model5_RNAsyn), which describe
*what each cell type is doing internally* — its metabolic phenotype.

Every intracellular reaction ID carries a `model{N}_` prefix, so it is already
attributable to a specific member of the community. Using the community grid
JSON + metadata.csv (via load_region_data / get_model_to_cell_map) we resolve
    model{N}  ->  barcode  ->  cell_type  ->  (px_x, px_y)
exactly as the exchange pipeline does.

Provides:
1) Robust loading of intracellular flux CSV/TSV files
2) Aggregation into per-cell-type reaction-flux distributions
   (normalised by Community_Biomass) + a metabolite production/consumption index
3) Cell-type enrichment of reactions (Kruskal-Wallis)
4) Between-condition comparison of reactions (Mann-Whitney + log2FC)
5) A cell_type x reaction flux matrix (for heatmaps)
6) Cross-link: mechanistically explain each exchange interaction
   (src secretes met -> snk uptakes met) with the internal producing/consuming
   reactions in src / snk.
7) Pathway/subsystem-level analysis using the model's REAL subsystem
   annotations (see model_building/extract_reaction_subsystems.m, which reads
   the `subSystems` field off the REFERENCE Recon3D model - the same
   `modelO2Fixed` every cell-type/community model is GIMME-pruned from - and
   writes a Reaction -> Subsystem lookup CSV). Using the reference model
   rather than scanning the pruned per-cell models is deliberate: GIMME
   pruning keeps only each cell type's active subset, so any single
   cell-type model (or their union) is a less complete, less generalizable
   annotation source than the full reconstruction it was carved from.
   Subsystem activity is aggregated
   PER COMMUNITY MEMBER (one sample = one cell's total |flux| through a
   subsystem in one community solve), which is what makes
   find_enriched_subsystems() a fair Kruskal-Wallis test and is far more
   robust than any single reaction (fewer alternate-optima artifacts, since
   summing many reactions cancels solver-specific routing choices).

IMPORTANT CAVEATS (see module docstring in the notebook too):
- FBA internal fluxes are NOT unique (alternate optima). Interpret at the
  pathway / aggregate / ratio level, not per single reaction value.
- Fluxes scale with biomass; normalise before comparing cells/patients.
- COST_SINK / exchange_cost bookkeeping rows are filtered out.
"""

import os
import re
import glob
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu

# Resilient import: works whether this file is imported as part of the package
# (spicem_kidney_4.intracellular) or as a top-level module in the notebook.
try:
    from .core import load_region_data, get_model_to_cell_map
except ImportError:  # pragma: no cover - notebook / standalone usage
    from core import load_region_data, get_model_to_cell_map

warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

# Bookkeeping / pseudo reactions that are not real intracellular metabolism.
_SKIP_RXN_IDS = {"COST_SINK", "SINK_c_pseudo"}
_SKIP_RXN_PREFIXES = ("EX_", "DM_", "SINK_", "sink_")

_MODEL_PREFIX_RE = re.compile(r"^model(\d+)_")
# Base metabolite name from an internal token like 'model5_atp[c]' or
# 'model1_h[im][c]'  ->  'atp' / 'h'
_MET_RE = re.compile(r"model\d+_([A-Za-z0-9\-]+?)\[")

_RXN_EQ_COLS = ["Reaction_Equation", "Reaction", "reaction_equation", "reaction"]
_RXN_ID_COLS = ["Reaction_ID", "Reaction_Id", "reaction_id", "ID", "id"]


# =============================================================================
# CONFIG / RESULTS CONTAINERS
# =============================================================================

@dataclass
class IntracellularConfig:
    base_dir: str = ""
    conditions: List[str] = field(default_factory=lambda: ["Sample_Healthy"])
    regions: List[int] = field(default_factory=lambda: [1])

    grid_pattern: str = "communities_hexagonal_Region_{}.json"
    intracellular_folder: str = "intracellular_fluxes"

    normalize_by_biomass: bool = True
    p_threshold: float = 0.05


@dataclass
class IntracellularResults:
    config: IntracellularConfig
    per_region_rxn_fluxes: Dict[str, dict] = field(default_factory=dict)
    per_region_met_index: Dict[str, dict] = field(default_factory=dict)
    rxn_equations: Dict[str, str] = field(default_factory=dict)

    pooled_rxn_fluxes: Dict[str, dict] = field(default_factory=dict)
    enriched_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    celltype_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_intracellular_txt(path: str) -> pd.DataFrame:
    """
    Load an intracellular flux file. Works with both TSV and CSV.
    Keeps only rows with abs(Flux) > 1e-6. Mirrors core.load_flux_txt but
    tolerates the extra Exchange_Cost_Used / Community_Biomass columns.
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


def _find_comm_file(flux_dir: str, center_cell: str) -> Optional[str]:
    """
    Locate the per-community intracellular file for a given center cell.
    Tries the common naming conventions, then falls back to a glob so the
    loader is robust to however the extractor named its outputs.
    """
    candidates = [
        f"comm_{center_cell}.txt",
        f"comm_{center_cell}.csv",
        f"comm_{center_cell}_intracellular.txt",
        f"comm_{center_cell}_intracellular_fluxes.txt",
        f"comm_{center_cell}_intracellular_fluxes.csv",
        f"commModel_{center_cell}_intracellular_fluxes.csv",
        f"commModel_{center_cell}.csv",
    ]
    for name in candidates:
        p = os.path.join(flux_dir, name)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(flux_dir, f"*{center_cell}*"))
    return hits[0] if hits else None


# =============================================================================
# PARSING HELPERS
# =============================================================================

def _reaction_model(rxn_id: str) -> Optional[str]:
    m = _MODEL_PREFIX_RE.match(str(rxn_id))
    return f"model{m.group(1)}" if m else None


def _reaction_base(rxn_id: str) -> str:
    """Strip the model prefix so the same reaction is comparable across cells."""
    m = _MODEL_PREFIX_RE.match(str(rxn_id))
    return str(rxn_id)[m.end():] if m else str(rxn_id)


def _should_skip_reaction(rxn_id: str, equation: Optional[str]) -> bool:
    rid = str(rxn_id)
    if rid in _SKIP_RXN_IDS:
        return True
    if rid.startswith(_SKIP_RXN_PREFIXES):
        return True
    eq = str(equation) if equation is not None else ""
    # [u]-compartment exchanges are handled by the exchange pipeline; skip here.
    if "[u]" in eq:
        return True
    if "exchange_cost" in eq:
        return True
    return False


def _row_get(row: pd.Series, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in row.index and pd.notna(row.get(c)):
            return str(row[c])
    return None


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_intracellular_fluxes(
    grids,
    meta_df: pd.DataFrame,
    flux_dir: str,
    normalize_by_biomass: bool = True,
):
    """
    Parse each community's intracellular file and build:

    - rxn_fluxes[reaction_base][cell_type] = list of (normalised) fluxes
    - rxn_equations[reaction_base]         = one example reaction equation
    - met_index[met][cell_type]["produced"|"consumed"] = list of (reaction_base, flux)
        (used by link_exchange_to_intracellular to explain exchange interactions)

    Fluxes are divided by the per-community Community_Biomass (if present and
    normalize_by_biomass=True) so cells/communities are comparable.
    """
    rxn_fluxes: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rxn_equations: Dict[str, str] = {}
    met_index: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"produced": [], "consumed": []})
    )

    files_found, rxns_found = 0, 0

    for grid in grids:
        center_cell = grid["center_cell"]
        path = _find_comm_file(flux_dir, center_cell)
        if not path:
            continue

        df = load_intracellular_txt(path)
        if df.empty:
            continue

        files_found += 1
        m_map = get_model_to_cell_map(grid)

        # Per-community biomass normaliser (same value repeated across rows).
        biomass = 1.0
        if normalize_by_biomass and "Community_Biomass" in df.columns:
            b = pd.to_numeric(df["Community_Biomass"], errors="coerce").dropna()
            if len(b) and float(b.iloc[0]) > 0:
                biomass = float(b.iloc[0])

        for _, row in df.iterrows():
            rid = _row_get(row, _RXN_ID_COLS)
            eqn = _row_get(row, _RXN_EQ_COLS)
            if rid is None:
                continue
            if _should_skip_reaction(rid, eqn):
                continue

            mid = _reaction_model(rid)
            if mid is None or mid not in m_map:
                continue

            cid = m_map[mid]
            if cid not in meta_df.index:
                continue

            cell_type = meta_df.loc[cid, "cell_type"]
            flux = float(row["Flux"]) / biomass
            base = _reaction_base(rid)

            rxn_fluxes[base][cell_type].append(flux)
            rxn_equations.setdefault(base, eqn or "")
            rxns_found += 1

            # Metabolite production / consumption index.
            # Fluxes come from an irreversible model, so flux >= 0 means the
            # reaction runs left -> right (reactants consumed, products made).
            if eqn and "->" in eqn:
                lhs, rhs = [s.strip() for s in eqn.split("->", 1)]
                if flux >= 0:
                    cons_side, prod_side = lhs, rhs
                else:
                    cons_side, prod_side = rhs, lhs
                mag = abs(flux)
                for met in set(_MET_RE.findall(prod_side)):
                    met_index[met][cell_type]["produced"].append((base, mag))
                for met in set(_MET_RE.findall(cons_side)):
                    met_index[met][cell_type]["consumed"].append((base, mag))

    rxn_fluxes_clean = {b: dict(d) for b, d in rxn_fluxes.items()}
    met_index_clean = {
        m: {ct: dict(v) for ct, v in d.items()} for m, d in met_index.items()
    }
    return rxn_fluxes_clean, rxn_equations, met_index_clean, files_found, rxns_found


def _pool_rxn_fluxes(list_of_rxn_fluxes) -> Dict[str, Dict[str, list]]:
    """Merge several per-region rxn_fluxes dicts into one pooled dict."""
    pooled: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rxn_fluxes in list_of_rxn_fluxes:
        for base, ct_dict in rxn_fluxes.items():
            for ct, fluxes in ct_dict.items():
                pooled[base][ct].extend(fluxes)
    return {b: dict(d) for b, d in pooled.items()}


# =============================================================================
# ENRICHMENT (cell-type-specific reactions)
# =============================================================================

def find_enriched_reactions(
    rxn_fluxes: Dict[str, Dict[str, list]],
    p_threshold: float = 0.05,
    rxn_equations: Optional[Dict[str, str]] = None,
    min_cell_types: int = 2,
) -> pd.DataFrame:
    """
    For each reaction, test whether its flux differs across cell types
    (Kruskal-Wallis). Mirrors core.find_enriched_metabolites.

    Returns a DataFrame (one row per testable reaction) sorted by p-value with:
      Reaction, P, Significant, Best_CellType, Best_MeanFlux, N_CellTypes, Equation
    """
    rxn_equations = rxn_equations or {}
    rows = []

    for base, d in rxn_fluxes.items():
        groups = [np.asarray(f, dtype=float) for f in d.values() if len(f) > 0]
        if len(groups) < min_cell_types:
            continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            continue

        means = {ct: float(np.mean(f)) for ct, f in d.items() if len(f) > 0}
        best_ct = max(means, key=lambda k: abs(means[k]))

        rows.append({
            "Reaction": base,
            "P": float(p),
            "Significant": bool(p < p_threshold),
            "Best_CellType": best_ct,
            "Best_MeanFlux": float(means[best_ct]),
            "N_CellTypes": len(groups),
            "Equation": rxn_equations.get(base, ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


# =============================================================================
# BETWEEN-CONDITION COMPARISON
# =============================================================================

def compare_reactions_between_conditions(
    per_region_rxn_fluxes: Dict[str, Dict[str, Dict[str, list]]],
    conditions: List[str],
) -> pd.DataFrame:
    """
    Compare two conditions per reaction (Mann-Whitney U on pooled fluxes across
    all cell types and regions of each condition). Mirrors core.compare_conditions.

    Region keys are expected to look like '<Condition>_R<idx>'.

    Output columns: Reaction, MeanFlux_<cond0>, MeanFlux_<cond1>, Log2FC, P
    """
    if conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    pooled = {cond0: defaultdict(list), cond1: defaultdict(list)}

    for key, rxn_fluxes in (per_region_rxn_fluxes or {}).items():
        cond = key.rsplit("_R", 1)[0]
        if cond not in pooled:
            continue
        for base, ct_dict in (rxn_fluxes or {}).items():
            for _ct, fluxes in ct_dict.items():
                pooled[cond][base].extend(fluxes)

    rows = []
    all_rxns = set(pooled[cond0].keys()) | set(pooled[cond1].keys())
    for base in sorted(all_rxns):
        f0 = pooled[cond0][base]
        f1 = pooled[cond1][base]
        try:
            _, p = mannwhitneyu(f0, f1)
            p = float(p)
        except Exception:
            p = 1.0
        m0 = float(np.mean(f0)) if len(f0) else 0.0
        m1 = float(np.mean(f1)) if len(f1) else 0.0
        l2fc = float(np.log2((abs(m1) + 1e-9) / (abs(m0) + 1e-9)))
        rows.append({
            "Reaction": base,
            f"MeanFlux_{cond0}": m0,
            f"MeanFlux_{cond1}": m1,
            "Log2FC": l2fc,
            "P": p,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


# =============================================================================
# CELL-TYPE x REACTION MATRIX
# =============================================================================

def build_celltype_reaction_matrix(
    rxn_fluxes: Dict[str, Dict[str, list]],
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Build a (reactions x cell types) matrix of aggregated flux, for heatmaps /
    clustering of metabolic phenotype. agg is 'mean' or 'median'.
    """
    agg_fn = np.median if agg == "median" else np.mean
    records: Dict[str, Dict[str, float]] = {}
    for base, ct_dict in rxn_fluxes.items():
        records[base] = {
            ct: float(agg_fn(f)) for ct, f in ct_dict.items() if len(f) > 0
        }
    df = pd.DataFrame(records).T  # rows = reactions, cols = cell types
    return df.sort_index()


# =============================================================================
# CROSS-LINK: explain exchange interactions with intracellular reactions
# =============================================================================

def link_exchange_to_intracellular(
    met_index: Dict[str, Dict[str, dict]],
    exchange_best_pairs: Dict[str, Any],
    top_k: int = 3,
) -> pd.DataFrame:
    """
    For each exchange interaction (metabolite: source secretes -> sink uptakes),
    surface the internal reactions that PRODUCE the metabolite in the source
    cell type and CONSUME it in the sink cell type — the mechanistic 'why'
    behind the coupling.

    exchange_best_pairs: {met: (src, snk)} or {met: (src, snk, score)}
        e.g. AnalysisResults.global_best_pairs.

    Returns a tidy DataFrame:
        Metabolite, Source, Sink, Role, CellType, Reaction, TotalFlux
    """
    rows = []
    for met, pair in (exchange_best_pairs or {}).items():
        if not pair:
            continue
        src, snk = pair[0], pair[1]
        entry = met_index.get(met, {})

        def _top(role_cell, role_key, role_label):
            contribs = entry.get(role_cell, {}).get(role_key, [])
            agg: Dict[str, float] = defaultdict(float)
            for rxn_base, flux in contribs:
                agg[rxn_base] += float(flux)
            for rxn_base, total in sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                rows.append({
                    "Metabolite": met,
                    "Source": src,
                    "Sink": snk,
                    "Role": role_label,
                    "CellType": role_cell,
                    "Reaction": rxn_base,
                    "TotalFlux": float(total),
                })

        _top(src, "produced", "produces_in_source")
        _top(snk, "consumed", "consumes_in_sink")

    return pd.DataFrame(rows)


# =============================================================================
# SPATIAL FRAME (per-cell intracellular flux at its own coordinates)
# =============================================================================

def build_spatial_flux_frame(
    config: "IntracellularConfig",
    reactions: List[str],
    normalize_by_biomass: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a per-cell spatial table of intracellular flux for the requested
    reactions, so internal metabolism can be mapped in (px_x, px_y) space —
    the intracellular analogue of the exchange streamline plots.

    Attribution: **center-cell only**. For each community file the center cell
    is model1 (get_model_to_cell_map). We take model1's reactions and place
    them at the center cell's coordinates. This yields exactly one flux value
    per cell per reaction — the cell's metabolism in its own native
    neighbourhood context — with no double counting.

    Parameters
    ----------
    reactions : list of reaction *base* names (model prefix stripped),
        e.g. ["ATPS4m", "PGK", "HEX1"]. Keep this targeted; passing every
        reaction re-reads and melts all community files.

    Returns a long DataFrame:
        condition, region, barcode, cell_type, px_x, px_y, reaction, flux
    """
    target = set(reactions or [])
    if not target:
        return pd.DataFrame(
            columns=["condition", "region", "barcode", "cell_type",
                     "px_x", "px_y", "reaction", "flux"]
        )

    rows = []
    for cond in config.conditions:
        for r in config.regions:
            try:
                grids, meta_df, flux_dir = load_region_data(
                    config.base_dir, cond, r,
                    config.grid_pattern, config.intracellular_folder,
                )
            except Exception as e:
                if verbose:
                    print(f"{cond} R{r}: SKIP - {e}")
                continue

            for grid in grids:
                center_cell = grid["center_cell"]
                if center_cell not in meta_df.index:
                    continue
                path = _find_comm_file(flux_dir, center_cell)
                if not path:
                    continue
                df = load_intracellular_txt(path)
                if df.empty:
                    continue

                biomass = 1.0
                if normalize_by_biomass and "Community_Biomass" in df.columns:
                    b = pd.to_numeric(df["Community_Biomass"], errors="coerce").dropna()
                    if len(b) and float(b.iloc[0]) > 0:
                        biomass = float(b.iloc[0])

                cx = float(meta_df.loc[center_cell, "px_x"])
                cy = float(meta_df.loc[center_cell, "px_y"])
                ctype = meta_df.loc[center_cell, "cell_type"]

                for _, row in df.iterrows():
                    rid = _row_get(row, _RXN_ID_COLS)
                    if rid is None or _reaction_model(rid) != "model1":
                        continue  # center cell only
                    base = _reaction_base(rid)
                    if base not in target:
                        continue
                    eqn = _row_get(row, _RXN_EQ_COLS)
                    if _should_skip_reaction(rid, eqn):
                        continue
                    rows.append({
                        "condition": cond,
                        "region": r,
                        "barcode": center_cell,
                        "cell_type": ctype,
                        "px_x": cx,
                        "px_y": cy,
                        "reaction": base,
                        "flux": float(row["Flux"]) / biomass,
                    })

    return pd.DataFrame(rows)


# =============================================================================
# SUBSYSTEM / PATHWAY ANALYSIS (uses the model's real subsystem annotations)
# =============================================================================

def load_reaction_subsystems(path: str) -> Dict[str, str]:
    """
    Load the Reaction -> Subsystem lookup produced by
    model_building/extract_reaction_subsystems.m.

    Accepts a CSV with columns named (case-insensitively) 'Reaction' and
    'Subsystem'; falls back to the first two columns if names differ.
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    rcol = cols_lower.get("reaction", df.columns[0])
    scol = cols_lower.get("subsystem", df.columns[1])
    out = {}
    for rid, sub in zip(df[rcol].astype(str), df[scol].astype(str)):
        sub = sub.strip()
        out[rid.strip()] = sub if sub else "Unassigned"
    return out


def build_subsystem_activity_table(
    config: "IntracellularConfig",
    subsystem_map: Dict[str, str],
    normalize_by_biomass: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    One pass over every community file that yields a table with ONE ROW PER
    (community, member cell, subsystem): the member's total |flux| routed
    through that subsystem in that community's FBA solution.

    This is the unit both statistics and spatial maps are built from:
    - enrichment / comparison: each row is an independent sample of
      "how active is subsystem S in a cell of type T", exactly analogous to
      how find_enriched_reactions samples one reaction at a time.
    - spatial maps: px_x/px_y come straight from meta_df for every member
      (center AND neighbours), since metadata.csv covers the whole region -
      no restriction to center-cell-only is needed here.

    Columns: condition, region, center_cell, barcode, cell_type, role
             ("center"/"neighbor"), px_x, px_y, subsystem, activity,
             n_reactions (how many reactions of that subsystem fired).
    """
    rows = []
    for cond in config.conditions:
        for r in config.regions:
            try:
                grids, meta_df, flux_dir = load_region_data(
                    config.base_dir, cond, r,
                    config.grid_pattern, config.intracellular_folder,
                )
            except Exception as e:
                if verbose:
                    print(f"{cond} R{r}: SKIP - {e}")
                continue

            for grid in grids:
                center_cell = grid["center_cell"]
                path = _find_comm_file(flux_dir, center_cell)
                if not path:
                    continue
                df = load_intracellular_txt(path)
                if df.empty:
                    continue

                biomass = 1.0
                if normalize_by_biomass and "Community_Biomass" in df.columns:
                    b = pd.to_numeric(df["Community_Biomass"], errors="coerce").dropna()
                    if len(b) and float(b.iloc[0]) > 0:
                        biomass = float(b.iloc[0])

                m_map = get_model_to_cell_map(grid)
                member_activity: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
                member_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

                for _, row in df.iterrows():
                    rid = _row_get(row, _RXN_ID_COLS)
                    eqn = _row_get(row, _RXN_EQ_COLS)
                    if rid is None or _should_skip_reaction(rid, eqn):
                        continue
                    mid = _reaction_model(rid)
                    if mid is None or mid not in m_map:
                        continue
                    cid = m_map[mid]
                    if cid not in meta_df.index:
                        continue

                    base = _reaction_base(rid)
                    sub = subsystem_map.get(base, "Unassigned")
                    flux = float(row["Flux"]) / biomass

                    member_activity[cid][sub] += abs(flux)
                    member_count[cid][sub] += 1

                for cid, subd in member_activity.items():
                    ctype = meta_df.loc[cid, "cell_type"]
                    cx = float(meta_df.loc[cid, "px_x"])
                    cy = float(meta_df.loc[cid, "px_y"])
                    role = "center" if cid == center_cell else "neighbor"
                    for sub, val in subd.items():
                        rows.append({
                            "condition": cond,
                            "region": r,
                            "center_cell": center_cell,
                            "barcode": cid,
                            "cell_type": ctype,
                            "role": role,
                            "px_x": cx,
                            "px_y": cy,
                            "subsystem": sub,
                            "activity": float(val),
                            "n_reactions": int(member_count[cid][sub]),
                        })

    return pd.DataFrame(rows)


def find_enriched_subsystems(
    table: pd.DataFrame,
    p_threshold: float = 0.05,
    min_cell_types: int = 2,
) -> pd.DataFrame:
    """
    Kruskal-Wallis test per subsystem across cell types, using every
    (community, member) activity value as one sample. Mirrors
    find_enriched_reactions but at pathway resolution.

    Returns: Subsystem, P, Significant, Best_CellType, Best_MeanActivity,
             N_CellTypes, N_Samples, Mean_N_Reactions
    """
    if table is None or table.empty:
        return pd.DataFrame()

    rows = []
    for sub, d in table.groupby("subsystem"):
        by_ct = {ct: g["activity"].values for ct, g in d.groupby("cell_type")}
        groups = [v for v in by_ct.values() if len(v) > 0]
        if len(groups) < min_cell_types:
            continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            continue

        means = {ct: float(np.mean(v)) for ct, v in by_ct.items()}
        best_ct = max(means, key=means.get)

        rows.append({
            "Subsystem": sub,
            "P": float(p),
            "Significant": bool(p < p_threshold),
            "Best_CellType": best_ct,
            "Best_MeanActivity": float(means[best_ct]),
            "N_CellTypes": len(groups),
            "N_Samples": int(len(d)),
            "Mean_N_Reactions": float(d["n_reactions"].mean()),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("P").reset_index(drop=True)
    return out


def build_subsystem_celltype_matrix(
    table: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot the activity table into a (subsystem x cell_type) matrix."""
    if table is None or table.empty:
        return pd.DataFrame()
    return table.pivot_table(index="subsystem", columns="cell_type",
                              values="activity", aggfunc=agg)


def build_subsystem_prevalence_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of (community, member) samples of each cell type in which the
    subsystem carries nonzero flux at all. Used as dot-size in the
    enrichment dot plot (a pathway can have a strong mean but be rare, or a
    weak mean but be near-universal - prevalence disambiguates the two).
    """
    if table is None or table.empty:
        return pd.DataFrame()
    totals = table.groupby("cell_type")["barcode"].count()
    # a subsystem row only exists when activity > 0 upstream, so counting
    # rows per (subsystem, cell_type) directly gives the "active" count.
    active = table.groupby(["subsystem", "cell_type"]).size().unstack(fill_value=0)
    return active.div(totals, axis=1).fillna(0.0)


def build_spatial_subsystem_frame(
    table: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Collapse the activity table to one value per (barcode, subsystem) by
    averaging across every community that cell participated in (as center
    or neighbour) - a per-cell subsystem-activity map ready for plotting.

    Returns: barcode, cell_type, px_x, px_y, subsystem, value
    """
    if table is None or table.empty:
        return pd.DataFrame()
    g = table.groupby(["barcode", "cell_type", "px_x", "px_y", "subsystem"],
                       as_index=False)["activity"].agg(agg)
    return g.rename(columns={"activity": "value"})


def compare_subsystems_between_conditions(
    table: pd.DataFrame,
    conditions: List[str],
) -> pd.DataFrame:
    """
    Mann-Whitney + log2FC per subsystem between two conditions, pooling
    activity across cell types and communities (mirrors
    compare_reactions_between_conditions).
    """
    if table is None or table.empty or conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    rows = []
    for sub, d in table.groupby("subsystem"):
        v0 = d.loc[d["condition"] == cond0, "activity"].values
        v1 = d.loc[d["condition"] == cond1, "activity"].values
        if len(v0) == 0 and len(v1) == 0:
            continue
        try:
            _, p = mannwhitneyu(v0, v1)
            p = float(p)
        except Exception:
            p = 1.0
        m0 = float(np.mean(v0)) if len(v0) else 0.0
        m1 = float(np.mean(v1)) if len(v1) else 0.0
        l2fc = float(np.log2((abs(m1) + 1e-9) / (abs(m0) + 1e-9)))
        rows.append({
            "Subsystem": sub,
            f"MeanActivity_{cond0}": m0,
            f"MeanActivity_{cond1}": m1,
            "Log2FC": l2fc,
            "P": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("P").reset_index(drop=True)
    return out


def compare_celltype_subsystems_between_conditions(
    table: pd.DataFrame,
    conditions: List[str],
    subsystems: Optional[List[str]] = None,
    min_n: int = 2,
) -> pd.DataFrame:
    """
    Per (subsystem, cell_type) Mann-Whitney + log2FC between two conditions -
    the cell-type-STRATIFIED analogue of compare_subsystems_between_conditions
    (which pools across cell types, discarding which cell type drives a
    pathway's shift). Mirrors exchange_celltype_analysis.
    compare_celltype_metabolite_between_conditions exactly, one level up at
    pathway resolution.

    Input `table` is the same per-(community, member, subsystem) activity
    table build_subsystem_activity_table produces (or a cohort-wide
    concatenation of several patients' tables, tagged with each patient's
    own `condition`).

    Returns long-form: Subsystem, CellType, Mean_<c0>, Mean_<c1>, Log2FC, P,
    N_<c0>, N_<c1>. Log2FC > 0 means higher in the second condition.
    """
    if table is None or table.empty or conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    subs = subsystems or sorted(table["subsystem"].unique())
    sub = table[table["subsystem"].isin(subs)]

    rows = []
    for (s, ct), d in sub.groupby(["subsystem", "cell_type"]):
        v0 = d.loc[d["condition"] == cond0, "activity"].values
        v1 = d.loc[d["condition"] == cond1, "activity"].values
        if len(v0) < min_n or len(v1) < min_n:
            continue
        try:
            _, p = mannwhitneyu(v0, v1)
            p = float(p)
        except Exception:
            p = 1.0
        m0, m1 = float(np.mean(v0)), float(np.mean(v1))
        l2fc = float(np.log2((m1 + 1e-9) / (m0 + 1e-9)))
        rows.append({
            "Subsystem": s, "CellType": ct,
            f"Mean_{cond0}": m0, f"Mean_{cond1}": m1,
            "Log2FC": l2fc, "P": p,
            f"N_{cond0}": int(len(v0)), f"N_{cond1}": int(len(v1)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


def build_subsystem_differential_matrices(
    diff_df: pd.DataFrame,
    subsystems: Optional[List[str]] = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Pivot compare_celltype_subsystems_between_conditions's long-form output
    into two (subsystem x cell_type) matrices: Log2FC and P.
    """
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    subs = subsystems or list(pd.unique(diff_df["Subsystem"]))
    l2fc = diff_df.pivot_table(index="Subsystem", columns="CellType", values="Log2FC")
    pval = diff_df.pivot_table(index="Subsystem", columns="CellType", values="P")
    return l2fc.reindex(subs), pval.reindex(subs)


# =============================================================================
# METABOLITE-FLUX-SUM (mirrors pathway-flux-sum, but grouped by METABOLITE
# NODE instead of pathway ANNOTATION - needs no subsystem_map/Recon3D lookup
# at all, since metabolite names are parsed directly from each row's own
# Reaction_Equation string)
# =============================================================================
#
# Pathway-flux-sum groups reactions by a CURATED CATEGORY (the reference
# model's subSystems annotation: "TCA cycle", "Glycolysis", ...).
# Metabolite-flux-sum instead groups reactions by a shared NETWORK NODE: sum
# |flux| over every reaction that has a given metabolite as EITHER a
# substrate or a product, regardless of which pathway(s) those reactions
# belong to. A metabolite like ATP or NADH is used by reactions scattered
# across dozens of different pathways, so this is a fundamentally different
# (and complementary) grouping axis, not a finer/coarser version of the same
# thing. NOT prefixed by intracellular.build_pathway_metabolite_allocation_
# table below, which is the piece that connects the two axes.

def build_metabolite_activity_table(
    config: "IntracellularConfig",
    normalize_by_biomass: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    One pass over every community file, one row per (community, member cell,
    metabolite): the member's total |flux| through every reaction that
    touches that metabolite (as substrate OR product) in that community's
    FBA solution. Exactly mirrors build_subsystem_activity_table's sample
    unit and columns, just keyed by metabolite instead of subsystem.

    Requires no subsystem_map - metabolite names come straight off each
    row's Reaction_Equation (via the same regex used by
    link_exchange_to_intracellular's met_index), so this works even before
    model_building/extract_reaction_subsystems.m has been run.

    Columns: condition, region, center_cell, barcode, cell_type, role
             ("center"/"neighbor"), px_x, px_y, metabolite, activity,
             n_reactions.
    """
    rows = []
    for cond in config.conditions:
        for r in config.regions:
            try:
                grids, meta_df, flux_dir = load_region_data(
                    config.base_dir, cond, r,
                    config.grid_pattern, config.intracellular_folder,
                )
            except Exception as e:
                if verbose:
                    print(f"{cond} R{r}: SKIP - {e}")
                continue

            for grid in grids:
                center_cell = grid["center_cell"]
                path = _find_comm_file(flux_dir, center_cell)
                if not path:
                    continue
                df = load_intracellular_txt(path)
                if df.empty:
                    continue

                biomass = 1.0
                if normalize_by_biomass and "Community_Biomass" in df.columns:
                    b = pd.to_numeric(df["Community_Biomass"], errors="coerce").dropna()
                    if len(b) and float(b.iloc[0]) > 0:
                        biomass = float(b.iloc[0])

                m_map = get_model_to_cell_map(grid)
                member_activity: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
                member_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

                for _, row in df.iterrows():
                    rid = _row_get(row, _RXN_ID_COLS)
                    eqn = _row_get(row, _RXN_EQ_COLS)
                    if rid is None or eqn is None or _should_skip_reaction(rid, eqn):
                        continue
                    mid = _reaction_model(rid)
                    if mid is None or mid not in m_map:
                        continue
                    cid = m_map[mid]
                    if cid not in meta_df.index:
                        continue
                    if "->" not in eqn:
                        continue

                    lhs, rhs = [s.strip() for s in eqn.split("->", 1)]
                    mets = set(_MET_RE.findall(lhs)) | set(_MET_RE.findall(rhs))
                    if not mets:
                        continue

                    flux = abs(float(row["Flux"]) / biomass)
                    for met in mets:
                        member_activity[cid][met] += flux
                        member_count[cid][met] += 1

                for cid, metd in member_activity.items():
                    ctype = meta_df.loc[cid, "cell_type"]
                    cx = float(meta_df.loc[cid, "px_x"])
                    cy = float(meta_df.loc[cid, "px_y"])
                    role = "center" if cid == center_cell else "neighbor"
                    for met, val in metd.items():
                        rows.append({
                            "condition": cond, "region": r,
                            "center_cell": center_cell, "barcode": cid,
                            "cell_type": ctype, "role": role,
                            "px_x": cx, "px_y": cy,
                            "metabolite": met, "activity": float(val),
                            "n_reactions": int(member_count[cid][met]),
                        })

    return pd.DataFrame(rows)


def find_enriched_metabolite_activity(
    table: pd.DataFrame,
    p_threshold: float = 0.05,
    min_cell_types: int = 2,
) -> pd.DataFrame:
    """Kruskal-Wallis per metabolite across cell types. Mirrors
    find_enriched_subsystems exactly, one axis over."""
    if table is None or table.empty:
        return pd.DataFrame()

    rows = []
    for met, d in table.groupby("metabolite"):
        by_ct = {ct: g["activity"].values for ct, g in d.groupby("cell_type")}
        groups = [v for v in by_ct.values() if len(v) > 0]
        if len(groups) < min_cell_types:
            continue
        try:
            _, p = kruskal(*groups)
        except Exception:
            continue

        means = {ct: float(np.mean(v)) for ct, v in by_ct.items()}
        best_ct = max(means, key=means.get)

        rows.append({
            "Metabolite": met,
            "P": float(p),
            "Significant": bool(p < p_threshold),
            "Best_CellType": best_ct,
            "Best_MeanActivity": float(means[best_ct]),
            "N_CellTypes": len(groups),
            "N_Samples": int(len(d)),
            "Mean_N_Reactions": float(d["n_reactions"].mean()),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("P").reset_index(drop=True)
    return out


def build_metabolite_activity_celltype_matrix(
    table: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot the metabolite activity table into a (metabolite x cell_type)
    matrix. Mirrors build_subsystem_celltype_matrix."""
    if table is None or table.empty:
        return pd.DataFrame()
    return table.pivot_table(index="metabolite", columns="cell_type",
                              values="activity", aggfunc=agg)


def build_spatial_metabolite_activity_frame(
    table: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """Collapse to one value per (barcode, metabolite), averaging across
    every community the cell participated in. Mirrors
    build_spatial_subsystem_frame. Returns: barcode, cell_type, px_x, px_y,
    metabolite, value."""
    if table is None or table.empty:
        return pd.DataFrame()
    g = table.groupby(["barcode", "cell_type", "px_x", "px_y", "metabolite"],
                       as_index=False)["activity"].agg(agg)
    return g.rename(columns={"activity": "value"})


def compare_metabolite_activity_between_conditions(
    table: pd.DataFrame,
    conditions: List[str],
) -> pd.DataFrame:
    """Mann-Whitney + log2FC per metabolite between two conditions, pooling
    across cell types. Mirrors compare_subsystems_between_conditions."""
    if table is None or table.empty or conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    rows = []
    for met, d in table.groupby("metabolite"):
        v0 = d.loc[d["condition"] == cond0, "activity"].values
        v1 = d.loc[d["condition"] == cond1, "activity"].values
        if len(v0) == 0 and len(v1) == 0:
            continue
        try:
            _, p = mannwhitneyu(v0, v1)
            p = float(p)
        except Exception:
            p = 1.0
        m0 = float(np.mean(v0)) if len(v0) else 0.0
        m1 = float(np.mean(v1)) if len(v1) else 0.0
        l2fc = float(np.log2((abs(m1) + 1e-9) / (abs(m0) + 1e-9)))
        rows.append({
            "Metabolite": met,
            f"MeanActivity_{cond0}": m0,
            f"MeanActivity_{cond1}": m1,
            "Log2FC": l2fc,
            "P": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("P").reset_index(drop=True)
    return out


def compare_celltype_metabolite_activity_between_conditions(
    table: pd.DataFrame,
    conditions: List[str],
    metabolites: Optional[List[str]] = None,
    min_n: int = 2,
) -> pd.DataFrame:
    """
    Per (metabolite, cell_type) Mann-Whitney + log2FC between two conditions.
    Mirrors compare_celltype_subsystems_between_conditions.

    NOTE: this is a DIFFERENT quantity from exchange_celltype_analysis.
    compare_celltype_metabolite_between_conditions, despite the similar name.
    That one compares EXCHANGE flux (secretion/uptake crossing the [e]/[u]
    cell boundary). This one compares a metabolite's total INTRACELLULAR
    turnover (every internal reaction that touches it, regardless of
    pathway) - a metabolite can show a strong intracellular-activity shift
    with no exchange shift at all, or vice versa.
    """
    if table is None or table.empty or conditions is None or len(conditions) < 2:
        return pd.DataFrame()

    cond0, cond1 = conditions[0], conditions[1]
    mets = metabolites or sorted(table["metabolite"].unique())
    sub = table[table["metabolite"].isin(mets)]

    rows = []
    for (met, ct), d in sub.groupby(["metabolite", "cell_type"]):
        v0 = d.loc[d["condition"] == cond0, "activity"].values
        v1 = d.loc[d["condition"] == cond1, "activity"].values
        if len(v0) < min_n or len(v1) < min_n:
            continue
        try:
            _, p = mannwhitneyu(v0, v1)
            p = float(p)
        except Exception:
            p = 1.0
        m0, m1 = float(np.mean(v0)), float(np.mean(v1))
        l2fc = float(np.log2((m1 + 1e-9) / (m0 + 1e-9)))
        rows.append({
            "Metabolite": met, "CellType": ct,
            f"Mean_{cond0}": m0, f"Mean_{cond1}": m1,
            "Log2FC": l2fc, "P": p,
            f"N_{cond0}": int(len(v0)), f"N_{cond1}": int(len(v1)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


def build_metabolite_activity_differential_matrices(
    diff_df: pd.DataFrame,
    metabolites: Optional[List[str]] = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Pivot compare_celltype_metabolite_activity_between_conditions's
    long-form output into (metabolite x cell_type) Log2FC/P matrices."""
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    mets = metabolites or list(pd.unique(diff_df["Metabolite"]))
    l2fc = diff_df.pivot_table(index="Metabolite", columns="CellType", values="Log2FC")
    pval = diff_df.pivot_table(index="Metabolite", columns="CellType", values="P")
    return l2fc.reindex(mets), pval.reindex(mets)


# =============================================================================
# COMBINED: pathway-flux-sum x metabolite-flux-sum -> flux ALLOCATION
# =============================================================================
#
# Neither axis alone can answer "is this metabolite's flux being ROUTED
# differently" - pathway-flux-sum pools every metabolite together within a
# pathway; metabolite-flux-sum pools every pathway together for one
# metabolite. Crossing them gives, for each metabolite, how its total
# turnover is SPLIT across the pathways that touch it - a compositional
# quantity neither margin can see. A metabolite's total activity can stay
# perfectly flat between two conditions while its pathway allocation
# reorganises completely (e.g. ATP turnover unchanged in total, but the
# fraction produced by OXPHOS vs glycolysis flips) - that reorganisation IS
# the emergent, new insight this section is built to surface.

def build_metabolite_pathway_allocation_table(
    config: "IntracellularConfig",
    subsystem_map: Dict[str, str],
    normalize_by_biomass: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    One pass, one row per (community, member cell, metabolite, subsystem):
    the member's total |flux| through reactions that BOTH touch that
    metabolite AND belong to that subsystem, in that community's solution.
    This is the joint table pathway-flux-sum and metabolite-flux-sum are
    each a margin of (sum over subsystem -> metabolite-flux-sum; sum over
    metabolite -> pathway-flux-sum).

    Columns: condition, region, center_cell, barcode, cell_type, role,
             px_x, px_y, metabolite, subsystem, activity, n_reactions.
    """
    rows = []
    for cond in config.conditions:
        for r in config.regions:
            try:
                grids, meta_df, flux_dir = load_region_data(
                    config.base_dir, cond, r,
                    config.grid_pattern, config.intracellular_folder,
                )
            except Exception as e:
                if verbose:
                    print(f"{cond} R{r}: SKIP - {e}")
                continue

            for grid in grids:
                center_cell = grid["center_cell"]
                path = _find_comm_file(flux_dir, center_cell)
                if not path:
                    continue
                df = load_intracellular_txt(path)
                if df.empty:
                    continue

                biomass = 1.0
                if normalize_by_biomass and "Community_Biomass" in df.columns:
                    b = pd.to_numeric(df["Community_Biomass"], errors="coerce").dropna()
                    if len(b) and float(b.iloc[0]) > 0:
                        biomass = float(b.iloc[0])

                m_map = get_model_to_cell_map(grid)
                member_joint: Dict[str, Dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
                member_count: Dict[str, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

                for _, row in df.iterrows():
                    rid = _row_get(row, _RXN_ID_COLS)
                    eqn = _row_get(row, _RXN_EQ_COLS)
                    if rid is None or eqn is None or _should_skip_reaction(rid, eqn):
                        continue
                    mid = _reaction_model(rid)
                    if mid is None or mid not in m_map:
                        continue
                    cid = m_map[mid]
                    if cid not in meta_df.index:
                        continue
                    if "->" not in eqn:
                        continue

                    lhs, rhs = [s.strip() for s in eqn.split("->", 1)]
                    mets = set(_MET_RE.findall(lhs)) | set(_MET_RE.findall(rhs))
                    if not mets:
                        continue

                    base = _reaction_base(rid)
                    sub = subsystem_map.get(base, "Unassigned")
                    flux = abs(float(row["Flux"]) / biomass)

                    for met in mets:
                        key = (met, sub)
                        member_joint[cid][key] += flux
                        member_count[cid][key] += 1

                for cid, jointd in member_joint.items():
                    ctype = meta_df.loc[cid, "cell_type"]
                    cx = float(meta_df.loc[cid, "px_x"])
                    cy = float(meta_df.loc[cid, "px_y"])
                    role = "center" if cid == center_cell else "neighbor"
                    for (met, sub), val in jointd.items():
                        rows.append({
                            "condition": cond, "region": r,
                            "center_cell": center_cell, "barcode": cid,
                            "cell_type": ctype, "role": role,
                            "px_x": cx, "px_y": cy,
                            "metabolite": met, "subsystem": sub,
                            "activity": float(val),
                            "n_reactions": int(member_count[cid][(met, sub)]),
                        })

    return pd.DataFrame(rows)


def compute_pathway_allocation(
    table: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Collapse the joint (metabolite, subsystem) activity table to fractional
    pathway allocation: for each (group_cols..., metabolite), what fraction
    of its total activity comes from each subsystem.

    group_cols: e.g. ["condition"], ["cell_type"], ["condition","cell_type"],
    or None to pool everything (one allocation per metabolite overall).

    Returns long-form: group_cols..., metabolite, subsystem, mean_activity,
    fraction (0-1, sums to 1 within each group_cols x metabolite).
    """
    if table is None or table.empty:
        return pd.DataFrame()
    gcols = list(group_cols) if group_cols else []
    key = gcols + ["metabolite", "subsystem"]
    g = table.groupby(key)["activity"].agg(agg).reset_index().rename(columns={"activity": "mean_activity"})

    totals = g.groupby(gcols + ["metabolite"])["mean_activity"].transform("sum")
    g["fraction"] = g["mean_activity"] / totals.replace(0, np.nan)
    return g


def rank_metabolites_by_allocation_shift(
    table: pd.DataFrame,
    conditions: List[str],
    min_pathways: int = 2,
    min_n: int = 2,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    THE EMERGENT STATISTIC: for every metabolite, compute its fractional
    pathway-allocation vector separately in each of two conditions, then the
    total variation distance (half the L1 distance) between them - 0 means
    identical pathway usage in both conditions, 1 means completely disjoint
    (all flux moved to different pathways entirely). This can flag a
    metabolite even when compare_metabolite_activity_between_conditions sees
    no overall magnitude change, because it measures ROUTING, not magnitude.

    Returns: Metabolite, N_Pathways, Allocation_Shift (0-1),
             Top_Pathway_<cond0>, Frac_Top_Pathway_<cond0>,
             Top_Pathway_<cond1>, Frac_Top_Pathway_<cond1>,
             N_<cond0>, N_<cond1>. Sorted by Allocation_Shift descending.
    """
    if table is None or table.empty or conditions is None or len(conditions) < 2:
        return pd.DataFrame()
    cond0, cond1 = conditions[0], conditions[1]

    counts = table.groupby(["metabolite", "condition"]).size().unstack(fill_value=0)
    valid_mets = counts.index[(counts.get(cond0, 0) >= min_n) & (counts.get(cond1, 0) >= min_n)]

    rows = []
    for met in valid_mets:
        d = table[table["metabolite"] == met]
        subs = sorted(d["subsystem"].unique())
        if len(subs) < min_pathways:
            continue

        alloc0 = d[d["condition"] == cond0].groupby("subsystem")["activity"].mean()
        alloc1 = d[d["condition"] == cond1].groupby("subsystem")["activity"].mean()
        alloc0 = (alloc0 / alloc0.sum()).reindex(subs, fill_value=0.0) if alloc0.sum() > 0 else pd.Series(0.0, index=subs)
        alloc1 = (alloc1 / alloc1.sum()).reindex(subs, fill_value=0.0) if alloc1.sum() > 0 else pd.Series(0.0, index=subs)

        tvd = float(np.abs(alloc0 - alloc1).sum() / 2.0)
        top0 = alloc0.idxmax() if alloc0.sum() > 0 else None
        top1 = alloc1.idxmax() if alloc1.sum() > 0 else None

        rows.append({
            "Metabolite": met, "N_Pathways": len(subs), "Allocation_Shift": tvd,
            f"Top_Pathway_{cond0}": top0, f"Frac_Top_Pathway_{cond0}": float(alloc0.get(top0, 0.0)) if top0 else 0.0,
            f"Top_Pathway_{cond1}": top1, f"Frac_Top_Pathway_{cond1}": float(alloc1.get(top1, 0.0)) if top1 else 0.0,
            f"N_{cond0}": int(counts.loc[met, cond0]), f"N_{cond1}": int(counts.loc[met, cond1]),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Allocation_Shift", ascending=False).reset_index(drop=True)
        if top_n:
            out = out.head(top_n)
    return out


# =============================================================================
# TOP-LEVEL DRIVER
# =============================================================================

def run_intracellular_analysis(
    config: IntracellularConfig,
    verbose: bool = True,
) -> IntracellularResults:
    """
    Load + aggregate intracellular fluxes for every (condition, region), then
    run cell-type enrichment, build the cell_type x reaction matrix, and (if
    >= 2 conditions) the between-condition comparison.
    """
    results = IntracellularResults(config=config)

    if verbose:
        print("=" * 70)
        print("INTRACELLULAR METABOLIC FLUX ANALYSIS")
        print(f"Folder: {config.intracellular_folder}  |  "
              f"normalize_by_biomass={config.normalize_by_biomass}")
        print("=" * 70)

    for cond in config.conditions:
        for r in config.regions:
            try:
                grids, meta_df, flux_dir = load_region_data(
                    config.base_dir, cond, r,
                    config.grid_pattern, config.intracellular_folder,
                )
            except Exception as e:
                if verbose:
                    print(f"{cond} R{r}: SKIP - {e}")
                continue

            rxn_fluxes, rxn_eqs, met_index, nfiles, nrxns = aggregate_intracellular_fluxes(
                grids, meta_df, flux_dir, config.normalize_by_biomass,
            )
            if not rxn_fluxes:
                if verbose:
                    print(f"{cond} R{r}: no intracellular fluxes found")
                continue

            key = f"{cond}_R{r}"
            results.per_region_rxn_fluxes[key] = rxn_fluxes
            results.per_region_met_index[key] = met_index
            results.rxn_equations.update(rxn_eqs)

            if verbose:
                print(f"{cond} R{r}: OK "
                      f"({len(rxn_fluxes)} reactions, {nrxns} rows, {nfiles} communities)")

    if not results.per_region_rxn_fluxes:
        if verbose:
            print("ERROR: No intracellular data processed!")
        return results

    results.pooled_rxn_fluxes = _pool_rxn_fluxes(results.per_region_rxn_fluxes.values())
    results.enriched_df = find_enriched_reactions(
        results.pooled_rxn_fluxes, config.p_threshold, results.rxn_equations,
    )
    results.celltype_matrix = build_celltype_reaction_matrix(results.pooled_rxn_fluxes)

    if len(config.conditions) >= 2:
        results.comparison_df = compare_reactions_between_conditions(
            results.per_region_rxn_fluxes, config.conditions,
        )

    if verbose:
        n_sig = int(results.enriched_df["Significant"].sum()) if not results.enriched_df.empty else 0
        print(f"Pooled reactions: {len(results.pooled_rxn_fluxes)}")
        print(f"Cell-type-enriched reactions (p<{config.p_threshold}): {n_sig}")
        print(f"Cell-type x reaction matrix: {results.celltype_matrix.shape}")
        if not results.comparison_df.empty:
            print(f"Condition-compared reactions: {len(results.comparison_df)}")

    return results


if __name__ == "__main__":
    print("Intracellular flux module loaded.")
    print(f"Skipping reaction IDs: {sorted(_SKIP_RXN_IDS)} and prefixes {_SKIP_RXN_PREFIXES}")
