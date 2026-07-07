#!/usr/bin/env python3
"""
consensus_exchange_network.py
==============================
Weighted Consensus Exchange Graph (WCEG) for cohort-level metabolite exchange.

Mathematical framework
----------------------
For metabolite M and disease group G:

  Nodes V  = all cell types observed in group G
  Edges E  = all (src, snk) pairs detected for M in any patient of G

Edge weight  W(src, snk, M, G):
  W = mean_S_corr × consistency
  where:
    mean_S_corr = mean composition-corrected score across detected patients
    consistency = n_patients_with_edge / n_patients_in_group

Composition-corrected score per patient:
  S_corr = log2(S_obs / max(pi_src * pi_snk * N_eff, eps) + 1)
  pi_c   = N_c / N_total   (cell-type proportion in this biopsy section)
  N_eff  = sqrt(N_src * N_snk)

Permutation test per edge:
  Shuffle patient group labels 1000×, recompute W, compare to observed.
  p = (n_exceed + 1) / (n_perms + 1)
  Edges with p < alpha: "consensus edges" (statistically dominant).

Patient selection for streamline plots:
  For a given (metabolite, src, snk, group), the best patient to show is the
  one with the highest abundance of the source cell type AND a detected exchange.
  This maximises the signal in the streamline diffusion field.
"""

import os
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import mannwhitneyu, fisher_exact

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from plotting import _should_exclude_metabolite
except ImportError:
    def _should_exclude_metabolite(m): return False

try:
    from run_cohort_pipeline import COHORT_METADATA
    _PATIENT_GROUP = {m["id"]: m["group"] for m in COHORT_METADATA}
    _PATIENT_META  = {m["id"]: m            for m in COHORT_METADATA}
except ImportError:
    _PATIENT_GROUP = {}
    _PATIENT_META  = {}

COHORT_GROUPS = ["Control", "HKD", "DKD"]
GROUP_COLORS  = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}


# ===========================================================================
# Helpers
# ===========================================================================

def _patient_group(pid):
    return _PATIENT_GROUP.get(pid, "Unknown")

def _get_condition(results):
    conds = getattr(getattr(results, "config", None), "conditions", [])
    return conds[0] if conds else None

def _get_ct_counts(results):
    """Extract {cell_type: n_spots} from patient metadata."""
    cond = _get_condition(results)
    per_region = getattr(results, "per_region_data", {}) or {}
    counts = Counter()
    for key, data in per_region.items():
        if cond and not key.startswith(cond):
            continue
        mdf = data.get("meta_df", None)
        if mdf is None or mdf.empty:
            continue
        for col in ["cell_type","Graph.based","Idents","celltype",
                    "CellType","annotation"]:
            if col in mdf.columns:
                counts.update(
                    mdf[col].dropna().astype(str).value_counts().to_dict())
                break
    return dict(counts)


# ===========================================================================
# Build per-patient exchange records with composition correction
# ===========================================================================

def build_exchange_records(patient_results: Dict, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Long-form DataFrame of ALL (patient, metabolite, src, snk) exchanges
    with composition-corrected scores.

    Key design decision
    -------------------
    We keep EVERY (src, snk) pair returned by the model for each patient,
    not just the dominant one.  The model's `interactions[metabolite]`
    contains a list of candidate exchange pairs with scores.  Previously
    we took `max(pairs)` and discarded the rest, which meant exchange_df
    had at most one row per (patient, metabolite) — making it impossible
    to distinguish different (src,snk) rows in the bar chart (all rows
    would share the same patient coverage and therefore the same bar heights).

    By keeping all pairs, exchange_df now has one row per
    (patient, metabolite, src, snk).  This enables:
      • Bar charts that are genuinely specific to the (src,snk) row shown
      • Correct detection counts per routing
      • WCEG edge weights that reflect actual per-pair prevalence

    Across regions (if a patient has multiple regions), we keep the maximum
    score per (met, src, snk) triplet — this deduplicates multi-region
    patients correctly.

    Columns: patient_id, group, metabolite, source, sink,
             score_raw, N_src, N_snk, N_total, pi_src, pi_snk,
             N_eff, ES, score_corr
    """
    rows = []
    for pid, results in patient_results.items():
        group = _patient_group(pid)
        cond  = _get_condition(results)
        ct    = _get_ct_counts(results)
        N     = max(sum(ct.values()), 1)
        per_region = getattr(results, "per_region_data", {}) or {}

        # seen[(met, src, snk)] = best score across regions
        seen: Dict[tuple, float] = {}

        for key, data in per_region.items():
            if cond and not key.startswith(cond):
                continue
            for met, pairs in (data.get("interactions", {}) or {}).items():
                if _should_exclude_metabolite(met) or not pairs:
                    continue
                # ── Keep ALL pairs, not just the best ─────────────────────
                for pair in pairs:
                    src = str(pair.get("source", "?"))
                    snk = str(pair.get("sink",   "?"))
                    scr = float(pair.get("score", 0.0))
                    if scr <= 0:          # skip zero/negative scores
                        continue
                    k = (met, src, snk)
                    if k not in seen or scr > seen[k]:
                        seen[k] = scr

        for (met, src, snk), S_obs in seen.items():
            N_src = ct.get(src, 0); N_snk = ct.get(snk, 0)
            pi_s  = N_src / N;      pi_k  = N_snk / N
            N_eff = np.sqrt(max(N_src, 1) * max(N_snk, 1))
            ES    = S_obs / max(pi_s * pi_k * N_eff, epsilon)
            rows.append({
                "patient_id":  pid, "group": group,
                "metabolite":  met, "source": src, "sink": snk,
                "score_raw":   round(S_obs,  4),
                "N_src": N_src, "N_snk": N_snk, "N_total": N,
                "pi_src": round(pi_s, 5), "pi_snk": round(pi_k, 5),
                "N_eff":  round(N_eff, 2),
                "ES":     round(ES,    4),
                "score_corr":  round(float(np.log2(ES + 1.0)), 4),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===========================================================================
# WCEG edge computation
# ===========================================================================

def compute_wceg_edges(
    exchange_df: pd.DataFrame,
    group: str,
    metabolite: str,
    all_pids_in_group: List[str],
    n_permutations: int = 1000,
    alpha: float = 0.20,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute WCEG edges for one (group, metabolite).

    Edge weight W = mean(S_corr across detected patients) × consistency
    Permutation test: shuffle patient assignments 1000×.

    Returns DataFrame: source, sink, n_patients, n_group, consistency,
                       mean_score_corr, edge_weight, perm_p, is_consensus
    """
    rng     = np.random.RandomState(rng_seed)
    n_group = len(all_pids_in_group)
    sub = exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"] == group)
    ]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for (src, snk), grp in sub.groupby(["source", "sink"]):
        n_det       = grp["patient_id"].nunique()
        consistency = n_det / max(n_group, 1)
        mean_corr   = float(grp["score_corr"].mean())
        W_obs       = mean_corr * consistency

        all_sub = exchange_df[
            (exchange_df["metabolite"] == metabolite) &
            (exchange_df["source"] == src) &
            (exchange_df["sink"]   == snk)
        ]
        all_scores   = all_sub["score_corr"].values.astype(float)
        all_pids_met = all_sub["patient_id"].values

        n_perm_exceed = 0
        if len(all_scores) >= 2 and n_group >= 2:
            for _ in range(n_permutations):
                idx = rng.choice(len(all_pids_met),
                                 size=min(n_group, len(all_pids_met)),
                                 replace=False)
                perm_scores = all_scores[idx]
                perm_W      = float(perm_scores.mean()) * (len(perm_scores)/n_group)
                if perm_W >= W_obs:
                    n_perm_exceed += 1
            perm_p = (n_perm_exceed + 1) / (n_permutations + 1)
        else:
            perm_p = 1.0 / max(n_det, 1)

        rows.append({
            "source":          src,
            "sink":            snk,
            "n_patients":      n_det,
            "n_group":         n_group,
            "consistency":     round(consistency, 3),
            "mean_score_corr": round(mean_corr, 4),
            "edge_weight":     round(W_obs, 4),
            "perm_p":          round(perm_p, 4),
            "is_consensus":    perm_p < alpha,
        })

    return (pd.DataFrame(rows)
            .sort_values("edge_weight", ascending=False)
            .reset_index(drop=True))


def build_wceg_for_metabolite(
    exchange_df: pd.DataFrame,
    patient_results: Dict,
    metabolite: str,
    groups: Optional[List[str]] = None,
    n_permutations: int = 1000,
    alpha: float = 0.20,
    rng_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Build WCEG edge tables for all groups for one metabolite."""
    if groups is None:
        groups = COHORT_GROUPS
    return {
        g: compute_wceg_edges(
            exchange_df, g, metabolite,
            [p for p in patient_results if _patient_group(p) == g],
            n_permutations=n_permutations,
            alpha=alpha, rng_seed=rng_seed,
        )
        for g in groups
    }


# ===========================================================================
# Patient selection for streamline plots
# ===========================================================================

def select_best_patient_for_streamline(
    exchange_df: pd.DataFrame,
    patient_results: Dict,
    group: str,
    metabolite: str,
    source_cell: str,
    sink_cell: str,
) -> Optional[str]:
    """
    Select the best patient in the group to show the streamline diffusion plot
    for a given (metabolite, source, sink) exchange.

    Strategy:
      1. Restrict to patients in the group that DETECTED this exchange.
      2. Among those, pick the patient with highest N_src (absolute count of
         source cell type) — this maximises the signal in the diffusion field
         because the vector field magnitude scales with source abundance.
      3. If no patient detected the exchange, fall back to highest N_src
         regardless of detection.

    Returns patient_id or None.
    """
    pids_in_group = [p for p in patient_results if _patient_group(p) == group]
    if not pids_in_group:
        return None

    # Patients that detected this specific edge
    sub = exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"]      == group) &
        (exchange_df["source"]     == source_cell) &
        (exchange_df["sink"]       == sink_cell)
    ]
    detected_pids = set(sub["patient_id"].tolist())

    # Get N_src for all patients in group
    src_counts = {}
    for pid in pids_in_group:
        ct = _get_ct_counts(patient_results[pid])
        src_counts[pid] = ct.get(source_cell, 0)

    # Priority 1: detected patients, ranked by N_src
    det_with_src = [(pid, src_counts.get(pid, 0))
                    for pid in pids_in_group if pid in detected_pids]
    if det_with_src:
        return max(det_with_src, key=lambda x: x[1])[0]

    # Priority 2: any patient in group with highest N_src
    all_with_src = [(pid, src_counts.get(pid, 0)) for pid in pids_in_group]
    if all_with_src:
        return max(all_with_src, key=lambda x: x[1])[0]

    return None


def get_dominant_exchange_for_group(
    exchange_df: pd.DataFrame,
    patient_results: Dict,
    group: str,
    metabolite: str,
    wceg_edges: Optional[pd.DataFrame] = None,
    n_permutations: int = 1000,
    alpha: float = 0.20,
) -> Optional[Tuple[str, str]]:
    """
    Return the (source, sink) of the dominant exchange for a metabolite
    in a group, defined as the WCEG edge with highest edge_weight.

    Falls back to majority-vote if WCEG gives no result.
    """
    if wceg_edges is not None and not wceg_edges.empty:
        top = wceg_edges.iloc[0]
        return str(top["source"]), str(top["sink"])

    # Compute WCEG on the fly
    pids = [p for p in patient_results if _patient_group(p) == group]
    if not pids:
        return None
    edges = compute_wceg_edges(
        exchange_df, group, metabolite, pids,
        n_permutations=n_permutations, alpha=alpha,
    )
    if not edges.empty:
        top = edges.iloc[0]
        return str(top["source"]), str(top["sink"])

    # Majority vote fallback
    sub = exchange_df[
        (exchange_df["metabolite"] == metabolite) &
        (exchange_df["group"]      == group)
    ]
    if sub.empty:
        return None
    c = Counter(zip(sub["source"], sub["sink"]))
    if c:
        return c.most_common(1)[0][0]
    return None


# ===========================================================================
# Cell-type abundance helper
# ===========================================================================

def get_group_ct_abundance(patient_results: Dict, group: str) -> Dict[str, float]:
    """Mean cell-type proportions across all patients in group."""
    ct_comp = {}
    for pid, results in patient_results.items():
        if _patient_group(pid) != group:
            continue
        ct = _get_ct_counts(results)
        N  = max(sum(ct.values()), 1)
        for c, n in ct.items():
            ct_comp.setdefault(c, []).append(n / N)
    return {ct: float(np.mean(v)) for ct, v in ct_comp.items()}


# ===========================================================================
# WCEG-based group comparison
# ===========================================================================

def compare_groups_by_wceg(
    exchange_df: pd.DataFrame,
    patient_results: Dict,
    group_a: str,
    group_b: str,
    metabolites: Optional[List[str]] = None,
    n_permutations: int = 2000,
    alpha: float = 0.20,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Compare WCEG edge weights between two groups for all metabolites.
    Tests each (metabolite, src, snk) edge separately — handles alt. optima.
    """
    rng    = np.random.RandomState(rng_seed)
    pids_a = ([p for p in patient_results
                if _patient_group(p) in ("DKD","HKD")]
              if group_a == "Diseased"
              else [p for p in patient_results if _patient_group(p) == group_a])
    pids_b = ([p for p in patient_results
                if _patient_group(p) in ("DKD","HKD")]
              if group_b == "Diseased"
              else [p for p in patient_results if _patient_group(p) == group_b])
    n_a, n_b = len(pids_a), len(pids_b)
    if n_a == 0 or n_b == 0:
        return pd.DataFrame()

    if metabolites is None:
        metabolites = exchange_df["metabolite"].unique().tolist()

    rows = []
    for met in metabolites:
        if _should_exclude_metabolite(met):
            continue
        sub = exchange_df[exchange_df["metabolite"] == met]
        for (src, snk), grp_df in sub.groupby(["source","sink"]):
            s_a = grp_df[grp_df["patient_id"].isin(pids_a)]["score_corr"].values.astype(float)
            s_b = grp_df[grp_df["patient_id"].isin(pids_b)]["score_corr"].values.astype(float)
            det_a, det_b = len(s_a), len(s_b)
            contingency = np.array([[det_a, n_a-det_a], [det_b, n_b-det_b]])
            try:
                _, fisher_p = fisher_exact(contingency, alternative="two-sided")
            except Exception:
                fisher_p = 1.0
            if det_a >= 2 and det_b >= 2:
                try:
                    obs_diff = s_a.mean() - s_b.mean()
                    combined = np.concatenate([s_a, s_b])
                    n_exc = 0
                    for _ in range(n_permutations):
                        rng.shuffle(combined)
                        d = combined[:det_a].mean() - combined[det_a:].mean()
                        if abs(d) >= abs(obs_diff): n_exc += 1
                    perm_p = (n_exc+1)/(n_permutations+1)
                    sp = np.sqrt(((det_a-1)*np.var(s_a,ddof=1)+
                                  (det_b-1)*np.var(s_b,ddof=1))/(det_a+det_b-2))
                    cohens_d = obs_diff / max(sp, 1e-9)
                except Exception:
                    perm_p = 1.0; cohens_d = 0.0
            else:
                perm_p = 1.0; cohens_d = 0.0
            mean_a = float(s_a.mean()) if det_a > 0 else 0.0
            mean_b = float(s_b.mean()) if det_b > 0 else 0.0
            log2fc = float(np.log2((mean_a+1e-6)/(mean_b+1e-6)))
            r_a = det_a/n_a; r_b = det_b/n_b
            p_min = min(fisher_p, perm_p)
            evidence = abs(cohens_d)*abs(r_a-r_b)*(-np.log10(max(p_min,1e-10)))
            rows.append({
                "metabolite": met, "source": src, "sink": snk,
                "exchange_axis": f"{src}→{snk}",
                "n_a": n_a, "n_b": n_b, "det_a": det_a, "det_b": det_b,
                f"rate_{group_a}": round(r_a,3),
                f"rate_{group_b}": round(r_b,3),
                f"mean_w_{group_a}": round(mean_a,4),
                f"mean_w_{group_b}": round(mean_b,4),
                "log2fc": round(log2fc,3), "cohens_d": round(cohens_d,3),
                "fisher_p": round(fisher_p,4), "perm_p": round(perm_p,4),
                "evidence_score": round(evidence,4),
                "score_type": "wceg_corrected",
            })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("evidence_score", ascending=False)
    result["significant"] = ((result["perm_p"]<0.05)|(result["fisher_p"]<0.05))
    return result.reset_index(drop=True)
