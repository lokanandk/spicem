#!/usr/bin/env python3
"""
INTRACELLULAR FLUX - CLINICAL COVARIATE CORRELATION
=====================================================

Companion to intracellular.py / cohort_statistics.py.

cohort_statistics.py already correlates EXCHANGE metabolite scores against
eGFR/Fibrosis (analyze_egfr_correlation / analyze_fibrosis_association),
using Spearman rho with a nominal p<0.05 threshold (no multiple-testing
correction there - see module docstring note below) on a metabolite x
patient score matrix pooled across cell types and regions. This module
mirrors that exact statistical convention for INTRACELLULAR reactions and
subsystems, and adds the one thing the exchange-side analysis doesn't have:
a genuine PER-CELL-TYPE-STRATIFIED correlation, not just cell-type-as-
annotation.

Two views, both against the same clinical covariates:
1) Aggregate  - one value per (patient, entity), pooling across cell types
   and communities -> entity x patient matrix -> Spearman rho vs eGFR/
   fibrosis. Mirrors the exchange-side convention exactly.
2) Per-cell-type - one value per (patient, entity, cell_type) -> a separate
   entity x patient matrix PER CELL TYPE -> correlated independently, so you
   can see e.g. "OXPHOS activity in PT correlates with eGFR" even when the
   whole-tissue-pooled OXPHOS signal does not.

`entity` is generic: pass reaction-level or subsystem-level activity tables
(anything with an `entity_col` + `value_col` + optionally `cell_type`) and
the same functions work for either granularity.

Statistics: Spearman rho (scipy.stats.spearmanr), matching cohort_statistics.
py's convention. `significant` uses the SAME nominal p<0.05 threshold as
egfr_df for direct comparability. An additional `padj` (Benjamini-Hochberg,
scipy.stats.false_discovery_control) is also reported per correlation call
as a stricter, multiple-testing-aware alternative - `padj_significant` uses
the same 0.20 threshold the notebook's PADJ_THRESHOLD already uses elsewhere
in this pipeline, since a stricter 0.05 FDR is typically unusable at the
n~10-15 patient cohort sizes here.
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from scipy.stats import false_discovery_control as _bh_fdr
except ImportError:  # older scipy
    def _bh_fdr(pvals):
        p = np.asarray(pvals, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(ranked, 0, 1)
        return out

try:
    from run_cohort_pipeline import COHORT_METADATA
except ImportError:
    COHORT_METADATA = []

try:
    from exchange_celltype_analysis import GROUP_COLORS
except ImportError:
    GROUP_COLORS = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}

try:
    from intracellular_plots import _save, _footer
except ImportError:
    def _save(fig, out_path):
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    def _footer(fig, text):
        fig.text(0.01, -0.02, text, fontsize=7.5, color="#666666", ha="left", va="top")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# CLINICAL METADATA
# =============================================================================

def _clinical_df() -> pd.DataFrame:
    """Patient-indexed clinical covariates, mirroring cohort_statistics._clinical_df
    field names (eGFR, fibrosis, group) so the two modules stay interchangeable."""
    rows = []
    for m in COHORT_METADATA:
        rows.append({
            "patient_id": m["id"],
            "eGFR": float(m["eGFR"]),
            "fibrosis": float(m["fibrosis"]),
            "group": m["group"],
        })
    df = pd.DataFrame(rows)
    return df.set_index("patient_id") if not df.empty else df


# =============================================================================
# PATIENT x ENTITY MATRICES (aggregate and per-cell-type)
# =============================================================================

def build_patient_entity_matrix(
    per_patient_tables: Dict[str, pd.DataFrame],
    entity_col: str = "subsystem",
    value_col: str = "activity",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate (cell-type-pooled) view: entity x patient matrix, one value per
    (patient, entity) pooling across cell types and communities.

    per_patient_tables: {patient_id: activity_table}, where activity_table is
    one patient's output of intracellular.build_subsystem_activity_table (or
    any table with the same entity_col/value_col/cell_type shape).
    """
    cols = {}
    for pid, table in per_patient_tables.items():
        if table is None or table.empty or entity_col not in table.columns:
            continue
        cols[pid] = table.groupby(entity_col)[value_col].agg(agg)
    return pd.DataFrame(cols)


def build_patient_celltype_entity_matrices(
    per_patient_tables: Dict[str, pd.DataFrame],
    entity_col: str = "subsystem",
    value_col: str = "activity",
    agg: str = "mean",
) -> Dict[str, pd.DataFrame]:
    """
    Per-cell-type view: {cell_type: entity x patient matrix}, one value per
    (patient, entity, cell_type). Lets a pathway's clinical correlation be
    tested independently within each cell type.
    """
    per_ct: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pid, table in per_patient_tables.items():
        if table is None or table.empty or "cell_type" not in table.columns:
            continue
        g = table.groupby([entity_col, "cell_type"])[value_col].agg(agg)
        for (ent, ct), val in g.items():
            per_ct.setdefault(ct, {}).setdefault(ent, {})[pid] = val
    return {ct: pd.DataFrame(ent_d).T for ct, ent_d in per_ct.items()}


# =============================================================================
# CORRELATION
# =============================================================================

def correlate_with_clinical(
    entity_patient_matrix: pd.DataFrame,
    clinical_field: str,
    p_thresh: float = 0.05,
    min_patients: int = 5,
    cell_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Spearman rho between each entity's per-patient value and a clinical
    covariate (eGFR/fibrosis), one row per entity.

    Returns: Entity, CellType (None for aggregate), rho, p_value,
             significant (p<p_thresh, nominal), padj (BH-FDR across this
             call's entities), padj_significant (padj<0.20), n_patients.
    Sorted by |rho| descending.
    """
    clin = _clinical_df()
    if clin.empty or entity_patient_matrix is None or entity_patient_matrix.empty:
        return pd.DataFrame()

    common = [p for p in entity_patient_matrix.columns
              if p in clin.index and pd.notna(clin.loc[p, clinical_field])]
    if len(common) < min_patients:
        return pd.DataFrame()
    cov = clin.loc[common, clinical_field].astype(float).values

    rows = []
    for entity, row in entity_patient_matrix[common].iterrows():
        vals = row.values.astype(float)
        mask = np.isfinite(vals)
        if mask.sum() < min_patients:
            continue
        try:
            rho, p = spearmanr(vals[mask], cov[mask])
        except Exception:
            rho, p = 0.0, 1.0
        rows.append({
            "Entity": entity, "CellType": cell_type,
            "rho": float(rho), "p_value": float(p),
            "n_patients": int(mask.sum()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["padj"] = _bh_fdr(df["p_value"].values)
    df["significant"] = df["p_value"] < p_thresh
    df["padj_significant"] = df["padj"] < 0.20
    df = df.reindex(df["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


def correlate_celltype_with_clinical(
    celltype_matrices: Dict[str, pd.DataFrame],
    clinical_field: str,
    p_thresh: float = 0.05,
    min_patients: int = 5,
) -> pd.DataFrame:
    """Run correlate_with_clinical independently per cell type, then combine.
    BH-FDR (padj) is computed ACROSS the combined (entity, cell_type) result,
    not per cell type, since that's the full multiple-testing family."""
    frames = []
    for ct, mat in celltype_matrices.items():
        d = correlate_with_clinical(mat, clinical_field, p_thresh, min_patients, cell_type=ct)
        if not d.empty:
            frames.append(d.drop(columns=["padj", "padj_significant"]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["padj"] = _bh_fdr(out["p_value"].values)
    out["padj_significant"] = out["padj"] < 0.20
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================

def plot_clinical_correlation_bar(
    corr_df: pd.DataFrame,
    clinical_field: str,
    top_n: int = 15,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Ranked horizontal bar of the strongest positive AND negative correlations
    (top_n each), coloured by direction, mirroring cohort_statistics's
    egfr/fibrosis bar chart convention. '*' = nominal p<0.05.
    """
    if corr_df is None or corr_df.empty:
        return None

    top = corr_df.reindex(corr_df["rho"].sort_values(ascending=False).index).head(top_n)
    bot = corr_df.reindex(corr_df["rho"].sort_values(ascending=True).index).head(top_n)
    key_cols = ["Entity", "CellType"] if "CellType" in corr_df.columns else ["Entity"]
    bar_df = pd.concat([top, bot]).drop_duplicates(subset=key_cols).sort_values("rho")
    if bar_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(bar_df))))
    colors = ["#2980b9" if r > 0 else "#c0392b" for r in bar_df["rho"]]
    y = np.arange(len(bar_df))
    ax.barh(y, bar_df["rho"], color=colors, edgecolor="white", linewidth=0.6)

    labels = []
    for _, r in bar_df.iterrows():
        lbl = str(r["Entity"])
        ct = r.get("CellType")
        if ct:
            lbl += f"  ({ct})"
        if r["significant"]:
            lbl += "  *"
        labels.append(lbl)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel(f"Spearman rho vs {clinical_field}")
    ax.set_title(title or f"Pathways correlated with {clinical_field}", fontsize=12, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee", linewidth=0.8)
    ax.set_axisbelow(True)

    n_sig = int(bar_df["significant"].sum())
    n_padj = int(bar_df["padj_significant"].sum()) if "padj_significant" in bar_df.columns else 0
    _footer(fig, f"* p<0.05 (Spearman, nominal - not multiple-testing corrected). "
                 f"{n_sig}/{len(bar_df)} shown are nominally significant; "
                 f"{n_padj} survive BH-FDR<0.20. n up to {int(bar_df['n_patients'].max())} patients.")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_clinical_correlation_scatter(
    entity_patient_matrix: pd.DataFrame,
    clinical_field: str,
    entity: str,
    cell_type: Optional[str] = None,
    out_path: Optional[str] = None,
):
    """
    Per-patient scatter of one entity's activity vs a clinical covariate,
    dots coloured by clinical group (same palette as the exchange-side WCEG/
    secretion-uptake plots), patient IDs annotated, dashed linear trend line,
    Spearman rho/p in the title.
    """
    clin = _clinical_df()
    if clin.empty or entity_patient_matrix is None or entity not in entity_patient_matrix.index:
        return None

    common = [p for p in entity_patient_matrix.columns if p in clin.index]
    xvals = clin.loc[common, clinical_field].astype(float)
    yvals = entity_patient_matrix.loc[entity, common].astype(float)
    mask = xvals.notna() & yvals.notna()
    xvals, yvals = xvals[mask], yvals[mask]
    if len(xvals) < 3:
        return None
    common = list(xvals.index)

    try:
        rho, p = spearmanr(yvals, xvals)
    except Exception:
        rho, p = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = clin.loc[common, "group"] if "group" in clin.columns else pd.Series("", index=common)
    for g in sorted(groups.unique()):
        m = groups == g
        ax.scatter(xvals[m], yvals[m], s=60, color=GROUP_COLORS.get(g, "#7f8c8d"),
                  edgecolor="white", linewidth=0.5, label=g, zorder=3)

    coeffs = np.polyfit(xvals, yvals, 1)
    xs = np.linspace(float(xvals.min()), float(xvals.max()), 50)
    ax.plot(xs, np.polyval(coeffs, xs), linestyle="--", color="#555555", linewidth=1.2, zorder=2)

    for pid in common:
        ax.annotate(pid, (xvals[pid], yvals[pid]), fontsize=6, color="#555555",
                   xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(clinical_field)
    ax.set_ylabel(f"{entity} activity" + (f" — {cell_type}" if cell_type else " (aggregate, pooled cell types)"))
    title = entity + (f"  ({cell_type})" if cell_type else "")
    ax.set_title(f"{title}\nSpearman rho={rho:.2f}, p={p:.3g}, n={len(xvals)}",
                 fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=8, title="Group", title_fontsize=8, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#f0f0f0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_path)


if __name__ == "__main__":
    print("Intracellular clinical-correlation helpers loaded.")
