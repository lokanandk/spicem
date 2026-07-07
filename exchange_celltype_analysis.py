#!/usr/bin/env python3
"""
EXCHANGE FLUX - PER-CELL-TYPE SECRETION/UPTAKE ANALYSIS
=========================================================

Companion to core.py / pipeline.py / cohort_comparison.py.

Existing exchange-side analyses either collapse across metabolites (
pipeline.compute_flux_balance_by_condition: total Produced/Consumed per cell
type, metabolite-agnostic) or collapse across cell types (
plotting.plot_top_metabolites_bar, cohort_comparison.compare_two_groups: one
metabolite tied to its single majority-vote source/sink pair). Neither keeps
BOTH dimensions - (cell type, metabolite) - at once, which is what's needed
to ask "what are cell type X's top secreted/taken-up metabolites, and how
does that change DKD vs Control?" or "which individual metabolites dominate
secretion/uptake, broken out per cell type?".

This module builds that missing long-form table directly from the same
`met_fluxes[met]["secretion"/"uptake"][cell_type] = [flux, ...]` structure
core.aggregate_fluxes() already produces (stored per-region in
AnalysisResults.per_region_data), tags every sample with patient/clinical
group, and provides ranking/pivoting helpers for two figure families:

1) Per-cell-type: top-N secreted / top-N taken-up metabolites, compared
   across clinical groups (DKD vs Control vs HKD, etc.)
2) Cohort-wide: top-N individual metabolites (not aggregated into one score)
   by secretion / uptake magnitude, broken out across ALL cell types.

Uses the same clinical-group plumbing as consensus_exchange_network.py
(COHORT_METADATA -> patient_id -> group) and the same group colour palette,
so figures are visually consistent with the existing WCEG plots.
"""

import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

try:
    from plotting import _should_exclude_metabolite
except ImportError:
    def _should_exclude_metabolite(m):
        return False

try:
    from run_cohort_pipeline import COHORT_METADATA
    _PATIENT_GROUP = {m["id"]: m["group"] for m in COHORT_METADATA}
except ImportError:
    _PATIENT_GROUP = {}

# Same palette as consensus_exchange_network.GROUP_COLORS, kept in sync so
# every cohort-level figure (WCEG, secretion/uptake) uses the same group hue.
GROUP_COLORS = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}


def _patient_group(pid: str) -> str:
    return _PATIENT_GROUP.get(pid, "Unknown")


def _get_condition(results) -> Optional[str]:
    conds = getattr(getattr(results, "config", None), "conditions", [])
    return conds[0] if conds else None


# =============================================================================
# CORE TABLE
# =============================================================================

def build_secretion_uptake_table(patient_results: Dict, min_flux: float = 1e-6) -> pd.DataFrame:
    """
    Long-form table: one row per individually-detected exchange flux value.

    Columns: patient_id, group, region, cell_type, metabolite, role
             ("secretion"/"uptake"), flux (signed, as stored), abs_flux.

    `flux` keeps its sign (solver-dependent routing convention); ranking and
    colour-mapping downstream use `abs_flux` as the magnitude, consistent
    with how intracellular.py treats |flux| as pathway "activity".
    """
    rows = []
    for pid, results in patient_results.items():
        group = _patient_group(pid)
        cond = _get_condition(results)
        per_region = getattr(results, "per_region_data", {}) or {}

        for key, data in per_region.items():
            if cond and not key.startswith(cond):
                continue
            met_fluxes = data.get("met_fluxes", {}) or {}
            for met, d in met_fluxes.items():
                if _should_exclude_metabolite(met):
                    continue
                for role in ("secretion", "uptake"):
                    for ct, fluxes in (d.get(role, {}) or {}).items():
                        for f in fluxes:
                            f = float(f)
                            if abs(f) < min_flux:
                                continue
                            rows.append({
                                "patient_id": pid,
                                "group": group,
                                "region": key,
                                "cell_type": ct,
                                "metabolite": met,
                                "role": role,
                                "flux": f,
                                "abs_flux": abs(f),
                            })

    return pd.DataFrame(rows)


# =============================================================================
# PER-CELL-TYPE TOP-N (across clinical groups)
# =============================================================================

def top_metabolites_for_celltype(
    table: pd.DataFrame,
    cell_type: str,
    role: str,
    top_n: int = 10,
    agg: str = "mean",
) -> List[str]:
    """Rank a cell type's metabolites by aggregate |flux| for one role,
    pooling across every group/patient (used to pick which metabolites a
    cross-condition comparison bar chart should show)."""
    sub = table[(table["cell_type"] == cell_type) & (table["role"] == role)]
    if sub.empty:
        return []
    ranked = sub.groupby("metabolite")["abs_flux"].agg(agg).sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


def celltype_metabolite_by_group(
    table: pd.DataFrame,
    cell_type: str,
    role: str,
    metabolites: Optional[List[str]] = None,
    top_n: int = 10,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Pivot one cell type's metabolite x group flux for a given role.

    Returns: metabolite x group DataFrame of aggregate |flux| (NaN where a
    metabolite was never detected for that group), plus a parallel
    `n_detections` DataFrame accessible via .attrs['n_detections'].
    """
    sub = table[(table["cell_type"] == cell_type) & (table["role"] == role)]
    if sub.empty:
        return pd.DataFrame()

    mets = metabolites or top_metabolites_for_celltype(table, cell_type, role, top_n, agg)
    sub = sub[sub["metabolite"].isin(mets)]
    if sub.empty:
        return pd.DataFrame()

    mat = sub.pivot_table(index="metabolite", columns="group", values="abs_flux", aggfunc=agg)
    mat = mat.reindex(mets)
    counts = sub.pivot_table(index="metabolite", columns="group", values="abs_flux", aggfunc="count")
    counts = counts.reindex(mets)
    mat.attrs["n_detections"] = counts
    return mat


def rank_celltypes_by_activity(table: pd.DataFrame, top_n: Optional[int] = None) -> List[str]:
    """Cell types ranked by total |flux| (secretion + uptake, all metabolites,
    all groups) - used to pick which cell types to feature by default when
    there are too many to show all of them."""
    if table.empty:
        return []
    ranked = table.groupby("cell_type")["abs_flux"].sum().sort_values(ascending=False)
    return ranked.head(top_n).index.tolist() if top_n else ranked.index.tolist()


# =============================================================================
# COHORT-WIDE TOP-N INDIVIDUAL METABOLITES ACROSS CELL TYPES
# =============================================================================

def top_metabolites_overall(
    table: pd.DataFrame,
    role: str,
    top_n: int = 20,
    agg: str = "mean",
    group: Optional[str] = None,
) -> List[str]:
    """Rank individual metabolites by aggregate |flux| for one role, pooled
    across cell types. Optionally restrict to one clinical group."""
    sub = table[table["role"] == role]
    if group is not None:
        sub = sub[sub["group"] == group]
    if sub.empty:
        return []
    ranked = sub.groupby("metabolite")["abs_flux"].agg(agg).sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()


def build_celltype_metabolite_matrix(
    table: pd.DataFrame,
    role: str,
    metabolites: Optional[List[str]] = None,
    top_n: int = 20,
    agg: str = "mean",
    group: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pivot to a (metabolite x cell_type) matrix of aggregate |flux| for one
    role, optionally restricted to one clinical group. This is the exchange
    analogue of intracellular.build_celltype_reaction_matrix.
    """
    sub = table[table["role"] == role]
    if group is not None:
        sub = sub[sub["group"] == group]
    if sub.empty:
        return pd.DataFrame()

    mets = metabolites or top_metabolites_overall(table, role, top_n, agg, group)
    sub = sub[sub["metabolite"].isin(mets)]
    if sub.empty:
        return pd.DataFrame()

    mat = sub.pivot_table(index="metabolite", columns="cell_type", values="abs_flux", aggfunc=agg)
    return mat.reindex(mets)


# =============================================================================
# DIFFERENTIAL: (metabolite, cell_type) flux between two clinical groups
# =============================================================================

def compare_celltype_metabolite_between_conditions(
    table: pd.DataFrame,
    role: str,
    group0: str,
    group1: str,
    metabolites: Optional[List[str]] = None,
    top_n: int = 20,
    agg: str = "mean",
    min_n: int = 2,
) -> pd.DataFrame:
    """
    For every (metabolite, cell_type) pair, compare |flux| between two
    clinical groups: Mann-Whitney U + log2 fold-change. This is the
    per-cell-type analogue of core.compare_conditions (which pools across
    cell types) - it keeps both dimensions so you can see WHICH cell type
    drives a metabolite's shift between conditions, not just that the
    metabolite shifted somewhere.

    Returns long-form: Metabolite, CellType, Mean_<group0>, Mean_<group1>,
    Log2FC, P, N_<group0>, N_<group1>. Log2FC > 0 means higher in group1.
    """
    if table is None or table.empty:
        return pd.DataFrame()

    sub = table[table["role"] == role]
    mets = metabolites or top_metabolites_overall(sub, role, top_n, agg)
    sub = sub[sub["metabolite"].isin(mets)]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for (met, ct), d in sub.groupby(["metabolite", "cell_type"]):
        v0 = d.loc[d["group"] == group0, "abs_flux"].values
        v1 = d.loc[d["group"] == group1, "abs_flux"].values
        if len(v0) < min_n or len(v1) < min_n:
            continue
        try:
            _, p = mannwhitneyu(v0, v1)
            p = float(p)
        except Exception:
            p = 1.0
        m0 = float(np.mean(v0))
        m1 = float(np.mean(v1))
        l2fc = float(np.log2((m1 + 1e-9) / (m0 + 1e-9)))
        rows.append({
            "Metabolite": met, "CellType": ct,
            f"Mean_{group0}": m0, f"Mean_{group1}": m1,
            "Log2FC": l2fc, "P": p,
            f"N_{group0}": int(len(v0)), f"N_{group1}": int(len(v1)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


def build_differential_matrices(
    diff_df: pd.DataFrame,
    metabolites: Optional[List[str]] = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Pivot compare_celltype_metabolite_between_conditions's long-form output
    into two (metabolite x cell_type) matrices: Log2FC and P. Row order
    follows `metabolites` if given, else appearance order in diff_df.
    """
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    mets = metabolites or list(pd.unique(diff_df["Metabolite"]))
    l2fc = diff_df.pivot_table(index="Metabolite", columns="CellType", values="Log2FC")
    pval = diff_df.pivot_table(index="Metabolite", columns="CellType", values="P")
    return l2fc.reindex(mets), pval.reindex(mets)
