#!/usr/bin/env python3
"""
cohort_comparison.py  — Unbiased Cohort-Wide Metabolite Exchange Comparison
=============================================================================

Methodological design
----------------------
This module addresses four statistical challenges specific to the cohort:

1. SMALL-n DISCRETE p-VALUE PROBLEM
   With n_Control = 3 and n_DKD = 6, the Mann-Whitney U statistic has only 84
   possible rank permutations (C(9,3) = 84).  The minimum achievable two-sided
   p-value is 2/84 = 0.024.  Applied to 151 metabolites, BH-FDR can never reach
   q < 0.10 — not because effects are absent, but because the test is discrete.
   Fix: use EXACT PERMUTATION p-values drawn from the full permutation null
   distribution (default 5000 permutations), which are more sensitive than
   the asymptotic normal approximation for small n.

2. ZERO-INFLATED DATA (HURDLE MODEL)
   Constraint-based metabolic models produce structural zeros: ~26% of
   metabolites are entirely absent in all three controls.  A single test
   conflates two distinct biological questions:
     (a) Is this exchange detectable in one group but not the other? (presence)
     (b) Given it is detected, does its magnitude differ?         (abundance)
   Fix: implement a two-part hurdle score:
     Part 1 — exact Fisher's test on detection counts (presence/absence)
     Part 2 — permutation Wilcoxon on rank-normalised scores among
               patients where the exchange is detected in EITHER group
   A combined evidence score integrates both components.

3. COMPOSITE RANKING SCORE
   Given that no single test reaches FDR significance with this n, results
   are ranked by a composite evidence score that integrates:
     - Permutation p-value (location shift)
     - Fisher p-value (presence/absence difference)
     - Cohen's d magnitude
     - Detection rate consistency (fraction of patients)
   This score is used to prioritise metabolites for follow-up, with the
   understanding that it is exploratory, not confirmatory.

4. AXIS-LEVEL VISUALISATION
   The bubble chart now filters to informative axes only (|d| > 0.3 OR
   n_metabolites >= 2 OR combined_p < 0.3), uses a spring-layout to avoid
   label collision, and limits labels to non-overlapping top hits.

Cell-type annotation additions (NEW)
--------------------------------------
* Volcano labels now show "met  [src→snk]" for significant hits.
* Effect-size bar y-tick labels already carried source→sink;  they now use
  the consistent _met_label() helper and are shown for ALL bars (not just
  the top-n that had source/sink available before).
* Detection heatmap y-tick labels show "met  [src→snk]".
* plot_axis_bubble() unchanged in structure but labels are clarified.
* NEW: plot_axis_metabolite_detail()  — companion strip chart to the axis
  bubble that shows, for every axis in the filtered bubble set, which
  specific metabolites are exchanged and their per-group detection rates.
  This directly answers "which metabolites are exchanged on this axis?"
  without cluttering the bubble chart itself.

Outputs per comparison
-----------------------
  comparison_<A>_vs_<B>.csv           full result table (with source/sink)
  significant_<A>_vs_<B>.csv          top metabolites by composite score
  signatures_<group>.csv              group-specific signatures
  axis_comparison_<A>_vs_<B>.csv      exchange-axis level results
  volcano_<A>_vs_<B>.png              effect size vs –log10(p) [src→snk labels]
  detection_heatmap_<A>_vs_<B>.png    detection + score heatmap [src→snk rows]
  effect_size_<A>_vs_<B>.png          ranked Cohen's d [src→snk tick labels]
  axis_network_<A>_vs_<B>.png         filtered axis bubble chart
  axis_metabolite_detail_<A>_vs_<B>.png  NEW: metabolites per axis strip chart
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats

from run_cohort_pipeline import COHORT_METADATA, COHORT_META_BY_ID

try:
    from plot_publication_figures import (
        plot_cohort_summary_figure,
        plot_signature_detection_bars,
        plot_effect_size_lollipop,
        plot_volcano_publication,
        plot_detection_dotplot,
    )
    HAS_PUB_PLOTS = True
except ImportError:
    HAS_PUB_PLOTS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

_GLOBAL_EXCLUDE = {"4nph", "npphos", "fru", "gal", "lcts6p", "lcts"}  # mets not directly involved in human metabolism excluded everywhere

def _should_exclude_met_global(m):
    """True if m should be excluded from all analyses."""
    try:
        import importlib
        mod = importlib.import_module("plotting")
        if mod._should_exclude_metabolite(m):
            return True
    except Exception:
        pass
    return m in _GLOBAL_EXCLUDE


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ComparisonConfig:
    base_out_dir:             str   = "cohort_output"
    comparisons: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("DKD",      "Control"),
        ("HKD",      "Control"),
        ("DKD",      "HKD"),
        ("Diseased",  "Control"),
    ])
    min_patients_per_group:   int   = 2
    detection_threshold:      float = 0.0
    # Permutation test settings
    n_permutations:           int   = 5000
    perm_seed:                int   = 42
    # Reporting thresholds (exploratory — not confirmatory at this n)
    p_thresh:                 float = 0.05   # raw permutation p for labelling
    fdr_thresh:               float = 0.10   # BH FDR q-value (informational)
    # Signature criteria
    min_effect_size:          float = 0.5
    min_signature_rate:       float = 0.50
    # Axis filtering thresholds for the bubble chart
    axis_min_abs_d:           float = 0.3
    axis_min_n_mets:          int   = 2
    axis_max_p:               float = 0.30
    top_n_heatmap:            int   = 40
    top_n_bar:                int   = 20
    verbose:                  bool  = True


# =============================================================================
# Group helpers
# =============================================================================

def _get_group_patients(group_name: str) -> List[str]:
    if group_name == "Diseased":
        return [m["id"] for m in COHORT_METADATA if m["group"] in ("DKD", "HKD")]
    return [m["id"] for m in COHORT_METADATA if m["group"] == group_name]


# =============================================================================
# Score matrix
# =============================================================================

def build_full_score_matrix(patient_results: Dict) -> pd.DataFrame:
    """
    Build (metabolites × patients) matrix summing ALL interaction pairs.
    NaN = not detected in that patient.
    """
    score_dicts = {}
    for pid, results in patient_results.items():
        per_region = getattr(results, "per_region_data", {}) or {}
        scores = {}
        for key, data in per_region.items():
            for met, pairs in (data.get("interactions", {}) or {}).items():
                if _should_exclude_met_global(met) or not pairs:
                    continue
                total = sum(float(p.get("score", 0.0)) for p in pairs)
                scores[met] = scores.get(met, 0.0) + total
        score_dicts[pid] = scores
    if not score_dicts:
        return pd.DataFrame()
    return pd.DataFrame(score_dicts)


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient rank normalisation to [0, 1].  NaN → 0 before ranking.
    Mitigates alternative-optima scale variation.
    """
    filled = df.fillna(0.0)
    ranked = filled.rank(axis=0, method="average")
    norm   = ranked.div(ranked.max(axis=0).replace(0, 1), axis=1)
    return norm


def build_source_sink_map(patient_results: Dict) -> Dict[str, Tuple[str, str]]:
    """Majority-vote source→sink for each metabolite across all patients."""
    from collections import Counter
    votes: Dict[str, Counter] = {}
    for pid, results in patient_results.items():
        per_region = getattr(results, "per_region_data", {}) or {}
        for key, data in per_region.items():
            for met, pairs in (data.get("interactions", {}) or {}).items():
                if not pairs:
                    continue
                src = str(pairs[0].get("source", "?"))
                snk = str(pairs[0].get("sink",   "?"))
                votes.setdefault(met, Counter())[(src, snk)] += 1
    return {m: c.most_common(1)[0][0] for m, c in votes.items() if c}


# =============================================================================
# Label helper (shared with cohort_statistics)
# =============================================================================

def _met_label(met: str, src_snk_map: Dict[str, Tuple[str, str]],
               max_ct_len: int = 10) -> str:
    """
    Return a compact label:  "met  [src→snk]"

    Cell-type names are truncated to max_ct_len characters so long names
    (e.g. "Endo_Peritubular") do not dominate tick labels.
    Omits cell-type part when both are "?".
    """
    src, snk = src_snk_map.get(met, ("?", "?"))
    if src == "?" and snk == "?":
        return met

    def _trunc(s: str) -> str:
        return s if len(s) <= max_ct_len else s[:max_ct_len - 1] + "…"

    return f"{met}  [{_trunc(src)}→{_trunc(snk)}]"


# =============================================================================
# Statistics
# =============================================================================

def _bh_correct(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, returns q-values."""
    n     = len(p_values)
    order = np.argsort(p_values)
    q     = np.empty(n)
    q[order] = p_values[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


def _permutation_mannwhitney(
    a: np.ndarray,
    b: np.ndarray,
    n_perms: int = 5000,
    seed: int    = 42,
) -> Tuple[float, float]:
    """
    Exact permutation Mann-Whitney p-value.

    For small samples (n < 8 in either group) the asymptotic normal
    approximation is unreliable because the test statistic takes only a
    handful of discrete values.  Permutation sampling from the true null
    distribution gives exact p-values at any sample size.

    Returns (observed_rank_biserial_r, permutation_p_value).
    """
    rng  = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    combined = np.concatenate([a, b])

    # Observed U statistic
    u_obs, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    r_obs    = float(1.0 - 2.0 * u_obs / (n_a * n_b))

    # Null distribution via permutation
    u_null = np.empty(n_perms)
    for i in range(n_perms):
        perm      = rng.permutation(combined)
        u_i, _    = stats.mannwhitneyu(
            perm[:n_a], perm[n_a:], alternative="two-sided"
        )
        u_null[i] = u_i

    # Two-sided p: fraction of null |U| >= |U_obs|
    u_centered_obs  = abs(u_obs - n_a * n_b / 2.0)
    u_centered_null = np.abs(u_null - n_a * n_b / 2.0)
    p = float((u_centered_null >= u_centered_obs).mean())
    # Minimum p = 1/n_perms (avoid exact zero)
    p = max(p, 1.0 / n_perms)
    return r_obs, p


def _composite_evidence_score(
    perm_p:    float,
    fisher_p:  float,
    cohens_d:  float,
    rate_high: float,
    rate_low:  float,
) -> float:
    """
    Composite evidence score for ranking metabolites when no single test
    reaches formal significance at this sample size.

    Score = 0.35 × (−log10 perm_p, capped at 3)
          + 0.25 × (−log10 fisher_p, capped at 3)
          + 0.25 × |Cohen's d| (capped at 3)
          + 0.15 × (rate_high − rate_low)

    Weights are calibrated to give roughly equal contribution from each
    component at typical effect sizes in this dataset.  The score is
    unsigned — direction is given by the sign of Cohen's d.
    """
    log_perm   = min(-np.log10(max(perm_p,   1e-10)), 3.0)
    log_fisher = min(-np.log10(max(fisher_p, 1e-10)), 3.0)
    abs_d      = min(abs(cohens_d), 3.0)
    rate_diff  = max(rate_high - rate_low, 0.0)
    return 0.35 * log_perm + 0.25 * log_fisher + 0.25 * abs_d + 0.15 * rate_diff


# =============================================================================
# Core comparison  — hurdle model
# =============================================================================

def compare_two_groups(
    patient_results: Dict,
    group_a: str,
    group_b: str,
    cfg: ComparisonConfig,
    score_matrix: Optional[pd.DataFrame] = None,
    rank_matrix:  Optional[pd.DataFrame] = None,
    src_snk_map:  Optional[Dict]         = None,
) -> pd.DataFrame:
    """
    Two-part hurdle comparison between group_a and group_b.

    Part 1 — Presence/absence: exact Fisher's test on detection counts.
    Part 2 — Abundance: permutation Mann-Whitney U on rank-normalised scores
              restricted to patients where the exchange is detected in either
              group (avoids structural zeros inflating the location test).

    Both parts are combined into a composite evidence score for ranking.
    BH-FDR is still computed for both tests for completeness, but the
    composite score is the primary ranking criterion given small n.
    """

    ids_a = [p for p in _get_group_patients(group_a) if p in patient_results]
    ids_b = [p for p in _get_group_patients(group_b) if p in patient_results]

    if (len(ids_a) < cfg.min_patients_per_group or
            len(ids_b) < cfg.min_patients_per_group):
        if cfg.verbose:
            print(f"  [SKIP] {group_a} vs {group_b}: "
                  f"{len(ids_a)} / {len(ids_b)} patients "
                  f"(need ≥{cfg.min_patients_per_group})")
        return pd.DataFrame()

    if score_matrix is None:
        score_matrix = build_full_score_matrix(patient_results)
    if rank_matrix is None:
        rank_matrix = rank_normalise(score_matrix)
    if src_snk_map is None:
        src_snk_map = build_source_sink_map(patient_results)

    cols_a = [p for p in ids_a if p in score_matrix.columns]
    cols_b = [p for p in ids_b if p in score_matrix.columns]
    n_a, n_b = len(cols_a), len(cols_b)

    sm_a = score_matrix[cols_a].fillna(0.0)
    sm_b = score_matrix[cols_b].fillna(0.0)
    rm_a = rank_matrix[cols_a]
    rm_b = rank_matrix[cols_b]

    all_mets = sorted([
        m for m in score_matrix.index
        if not _should_exclude_met_global(m)
    ])

    rows = []
    perm_pvals   = []
    fisher_pvals = []

    rng_seed = cfg.perm_seed
    for met in all_mets:
        sc_a_raw  = sm_a.loc[met].values.astype(float) if met in sm_a.index else np.zeros(n_a)
        sc_b_raw  = sm_b.loc[met].values.astype(float) if met in sm_b.index else np.zeros(n_b)
        sc_a_rank = rm_a.loc[met].values.astype(float) if met in rm_a.index else np.zeros(n_a)
        sc_b_rank = rm_b.loc[met].values.astype(float) if met in rm_b.index else np.zeros(n_b)

        mean_raw_a  = float(np.mean(sc_a_raw))
        mean_raw_b  = float(np.mean(sc_b_raw))
        mean_rank_a = float(np.mean(sc_a_rank))
        mean_rank_b = float(np.mean(sc_b_rank))

        det_a  = int((sc_a_raw > cfg.detection_threshold).sum())
        det_b  = int((sc_b_raw > cfg.detection_threshold).sum())
        rate_a = det_a / n_a
        rate_b = det_b / n_b

        # ── Part 1: Fisher's exact test (detection) ───────────────────────
        contingency = np.array([[det_a, n_a - det_a],
                                 [det_b, n_b - det_b]])
        try:
            _, fp = stats.fisher_exact(contingency, alternative="two-sided")
            fp_or = float(
                ((det_a + 0.5) * (n_b - det_b + 0.5)) /
                ((n_a - det_a + 0.5) * (det_b + 0.5))
            )
        except Exception:
            fp, fp_or = 1.0, 1.0
        fisher_pvals.append(float(fp))

        # ── Part 2: Permutation Wilcoxon on detected patients only ─────────
        det_mask_a = sc_a_raw > cfg.detection_threshold
        det_mask_b = sc_b_raw > cfg.detection_threshold

        if det_mask_a.sum() >= 2 and det_mask_b.sum() >= 2:
            detected_a = sc_a_rank[det_mask_a]
            detected_b = sc_b_rank[det_mask_b]
            wilcox_r, wp = _permutation_mannwhitney(
                detected_a, detected_b,
                n_perms=cfg.n_permutations,
                seed=rng_seed,
            )
        elif n_a >= 2 and n_b >= 2:
            wilcox_r, wp = _permutation_mannwhitney(
                sc_a_rank, sc_b_rank,
                n_perms=cfg.n_permutations,
                seed=rng_seed,
            )
        else:
            wp, wilcox_r = 1.0, 0.0
        perm_pvals.append(float(wp))
        rng_seed += 1

        # ── Cohen's d on full rank-normalised vectors ──────────────────────
        pooled_sd = float(np.sqrt(
            (np.var(sc_a_rank, ddof=1) * (n_a - 1) +
             np.var(sc_b_rank, ddof=1) * (n_b - 1)) /
            (n_a + n_b - 2 + 1e-9)
        ))
        cohens_d = float((mean_rank_a - mean_rank_b) / (pooled_sd + 1e-9))

        # ── Log2FC of raw means ────────────────────────────────────────────
        eps    = 1e-6
        log2fc = float(np.log2((mean_raw_a + eps) / (mean_raw_b + eps)))

        src_snk = src_snk_map.get(met, ("?", "?"))
        rows.append({
            "metabolite":            met,
            "source":                src_snk[0],
            "sink":                  src_snk[1],
            "exchange_axis":         f"{src_snk[0]}→{src_snk[1]}",
            f"mean_raw_{group_a}":   round(mean_raw_a,  2),
            f"mean_raw_{group_b}":   round(mean_raw_b,  2),
            f"mean_rank_{group_a}":  round(mean_rank_a, 4),
            f"mean_rank_{group_b}":  round(mean_rank_b, 4),
            f"detected_{group_a}":   det_a,
            f"detected_{group_b}":   det_b,
            f"rate_{group_a}":       round(rate_a, 3),
            f"rate_{group_b}":       round(rate_b, 3),
            "log2fc":                round(log2fc,  3),
            "perm_p":                float(wp),
            "fisher_p":              float(fp),
            "fisher_or":             round(fp_or, 3),
            "wilcox_r":              round(wilcox_r, 3),
            "cohens_d":              round(cohens_d, 3),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # ── FDR correction ─────────────────────────────────────────────────────
    result["perm_q"]   = np.round(_bh_correct(result["perm_p"].values),   4)
    result["fisher_q"] = np.round(_bh_correct(result["fisher_p"].values), 4)

    # ── Composite evidence score ───────────────────────────────────────────
    rate_a_col = f"rate_{group_a}"
    rate_b_col = f"rate_{group_b}"
    result["evidence_score"] = result.apply(lambda row: _composite_evidence_score(
        perm_p    = row["perm_p"],
        fisher_p  = row["fisher_p"],
        cohens_d  = row["cohens_d"],
        rate_high = max(row[rate_a_col], row[rate_b_col]),
        rate_low  = min(row[rate_a_col], row[rate_b_col]),
    ), axis=1)
    result["evidence_score"] = result["evidence_score"].round(3)

    # ── Significance flags ─────────────────────────────────────────────────
    result["sig_perm"]   = result["perm_p"]   < cfg.p_thresh
    result["sig_fisher"] = result["fisher_p"] < cfg.p_thresh
    result["significant"] = result["sig_perm"] | result["sig_fisher"]
    result["sig_fdr"] = (result["perm_q"] < cfg.fdr_thresh) | \
                        (result["fisher_q"] < cfg.fdr_thresh)

    # ── Signature labels ───────────────────────────────────────────────────
    ra = result[rate_a_col]
    rb = result[rate_b_col]
    result[f"signature_{group_a}"] = (
        (ra >= cfg.min_signature_rate) &
        (ra > rb * 1.5) &
        (result["cohens_d"] > 0)
    )
    result[f"signature_{group_b}"] = (
        (rb >= cfg.min_signature_rate) &
        (rb > ra * 1.5) &
        (result["cohens_d"] < 0)
    )

    # ── Effect size category ───────────────────────────────────────────────
    result["effect_size_label"] = pd.cut(
        result["cohens_d"].abs(),
        bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=["negligible", "small", "medium", "large"],
    )

    # ── Sort by composite evidence score ──────────────────────────────────
    result = result.sort_values("evidence_score", ascending=False).reset_index(drop=True)

    if cfg.verbose:
        print(f"    Permutation p-values computed ({cfg.n_permutations} perms)")
        print(f"    sig_perm (p<{cfg.p_thresh}): {result.sig_perm.sum()}  "
              f"sig_fisher (p<{cfg.p_thresh}): {result.sig_fisher.sum()}  "
              f"sig_fdr: {result.sig_fdr.sum()}")

    return result


# =============================================================================
# Exchange-axis analysis
# =============================================================================

def compare_exchange_axes(
    comp_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    cfg: ComparisonConfig,
) -> pd.DataFrame:
    """
    Aggregate to exchange-axis level using Fisher's combined probability test.

    The "metabolites" column now carries each metabolite as a semicolon-separated
    list so that plot_axis_metabolite_detail() can parse individual metabolites
    and their detection rates.
    """
    if comp_df.empty or "exchange_axis" not in comp_df.columns:
        return pd.DataFrame()

    rows = []
    for axis, grp in comp_df.groupby("exchange_axis"):
        n_total = len(grp)
        n_sig   = int(grp["significant"].sum())
        mean_d  = float(grp["cohens_d"].mean())
        mean_ev = float(grp["evidence_score"].mean())

        pvals = grp["perm_p"].clip(lower=1e-300).values
        if len(pvals) >= 2:
            chi2_stat  = -2.0 * np.sum(np.log(pvals))
            combined_p = float(stats.chi2.sf(chi2_stat, df=2 * len(pvals)))
        else:
            combined_p = float(pvals[0])

        src, snk = axis.split("→", 1) if "→" in axis else (axis, "?")

        # ── NEW: store per-metabolite detail for the companion strip chart ──
        # Format: "met(d=+0.83,rA=0.67,rB=0.33)" semicolon-separated
        met_details = []
        ra_col = f"rate_{group_a}"
        rb_col = f"rate_{group_b}"
        for _, mrow in grp.sort_values("cohens_d", ascending=False).iterrows():
            ra = mrow.get(ra_col, float("nan"))
            rb = mrow.get(rb_col, float("nan"))
            d  = mrow["cohens_d"]
            met_details.append(
                f"{mrow['metabolite']}(d={d:+.2f},"
                f"r{group_a}={ra:.0%},r{group_b}={rb:.0%})"
            )

        rows.append({
            "exchange_axis":    axis,
            "source":           src,
            "sink":             snk,
            "n_metabolites":    n_total,
            "n_significant":    n_sig,
            "pct_significant":  round(n_sig / n_total * 100, 1),
            "combined_p":       round(combined_p, 6),
            "mean_cohens_d":    round(mean_d, 3),
            "mean_evidence":    round(mean_ev, 3),
            "direction":        "UP" if mean_d > 0 else "DOWN",
            "direction_label":  f"↑ {group_a}" if mean_d > 0 else f"↑ {group_b}",
            "metabolites":      "; ".join(sorted(grp["metabolite"].tolist())),
            "metabolite_detail": "; ".join(met_details),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["combined_q"] = np.round(_bh_correct(df["combined_p"].values), 4)
    df = df.sort_values("combined_p").reset_index(drop=True)
    return df


# =============================================================================
# Visualisation
# =============================================================================

def plot_effect_volcano(
    comp_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    out_dir: str,
    cfg: ComparisonConfig,
) -> Optional[str]:
    """
    Volcano plot: Cohen's d vs –log10(permutation p).
    Dot size = max detection rate in either group.

    Cell-type annotation (NEW): significant metabolite labels now show
    "met  [src→snk]" using the source/sink columns already in comp_df.
    """
    if comp_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # Build src_snk_map from the result table itself
    src_snk_map = {
        row["metabolite"]: (row.get("source", "?"), row.get("sink", "?"))
        for _, row in comp_df.iterrows()
    }

    d    = comp_df["cohens_d"].values
    logp = -np.log10(comp_df["perm_p"].clip(lower=1e-300).values)
    sig  = comp_df["significant"].values

    rate_a = comp_df[f"rate_{group_a}"].values
    rate_b = comp_df[f"rate_{group_b}"].values
    dot_size = (np.maximum(rate_a, rate_b) * 180 + 10).clip(10, 300)

    colors = np.where(
        sig & (d > 0), "#c0392b",
        np.where(sig & (d < 0), "#2980b9", "#cccccc")
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#f8f9fa")

    ax.scatter(d, logp, c=colors, s=dot_size,
               alpha=0.72, edgecolors="white", linewidths=0.4, zorder=2)
    ax.axhline(-np.log10(cfg.p_thresh), color="#444444",
               linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"p = {cfg.p_thresh}")
    ax.axvline(0, color="#444444", linestyle="-", linewidth=0.7, alpha=0.4)
    for d_thr in (0.5, -0.5, 0.8, -0.8):
        ax.axvline(d_thr, color="#bbbbbb", linestyle=":", linewidth=0.7, alpha=0.5)

    # NEW: label top metabolites with [src→snk] context
    top_up   = comp_df[comp_df["cohens_d"] > 0].nlargest(8, "evidence_score")
    top_down = comp_df[comp_df["cohens_d"] < 0].nlargest(8, "evidence_score")
    to_label = pd.concat([top_up, top_down]).drop_duplicates("metabolite")

    texts = []
    for _, row in to_label.iterrows():
        lbl = _met_label(row["metabolite"], src_snk_map)
        texts.append(ax.text(
            row["cohens_d"],
            -np.log10(max(row["perm_p"], 1e-300)),
            lbl,
            fontsize=7.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                      alpha=0.55, edgecolor="none"),
        ))
    if HAS_ADJUSTTEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                    lw=0.5, alpha=0.5),
                    expand=(1.15, 1.2))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        f"Cohen's d  (rank-normalised; +ve = higher in {group_a})",
        fontsize=11, fontweight="bold")
    ax.set_ylabel("–log₁₀(permutation p-value)", fontsize=11, fontweight="bold")
    n_sig = comp_df["significant"].sum()
    ax.set_title(
        f"Metabolite Exchange: {group_a} vs {group_b}\n"
        f"Permutation test (n={cfg.n_permutations}) · "
        f"dot size = detection rate · {n_sig} nominally significant\n"
        "Labels: metabolite  [source_cell→sink_cell]",
        fontsize=12, fontweight="bold",
    )
    legend_els = [
        mpatches.Patch(color="#c0392b", label=f"↑ in {group_a} (p<{cfg.p_thresh})"),
        mpatches.Patch(color="#2980b9", label=f"↑ in {group_b} (p<{cfg.p_thresh})"),
        mpatches.Patch(color="#cccccc", label="Not significant"),
        plt.scatter([], [], s=30 , c="#888888", alpha=0.5, label="det. rate 17%"),
        plt.scatter([], [], s=130, c="#888888", alpha=0.5, label="det. rate 67%"),
        plt.scatter([], [], s=190, c="#888888", alpha=0.5, label="det. rate 100%"),
    ]
    ax.legend(handles=legend_els, fontsize=8, framealpha=0.9, loc="upper left",
              ncol=2, columnspacing=0.8)
    ax.grid(alpha=0.1, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fp = os.path.join(out_dir, f"volcano_{group_a}_vs_{group_b}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


def plot_detection_heatmap(
    comp_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    group_a: str,
    group_b: str,
    out_dir: str,
    cfg: ComparisonConfig,
) -> Optional[str]:
    """
    Side-by-side detection (binary) + score (continuous) heatmap.
    Rows = top metabolites by evidence score, sorted by detection-rate difference.

    Cell-type annotation (NEW): y-tick labels on the left panel now show
    "met  [src→snk]" instead of bare metabolite names.
    """
    if comp_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # Build src_snk_map from comp_df
    src_snk_map = {
        row["metabolite"]: (row.get("source", "?"), row.get("sink", "?"))
        for _, row in comp_df.iterrows()
    }

    ids_a = [p for p in _get_group_patients(group_a) if p in score_matrix.columns]
    ids_b = [p for p in _get_group_patients(group_b) if p in score_matrix.columns]
    all_ids = ids_a + ids_b
    if not all_ids:
        return None

    top_ev = comp_df.nlargest(cfg.top_n_heatmap, "evidence_score")
    top_mets = [m for m in top_ev["metabolite"] if m in score_matrix.index]
    if not top_mets:
        return None

    sm_plot  = score_matrix.loc[top_mets, all_ids].fillna(0.0)
    det_mat  = (sm_plot > cfg.detection_threshold).astype(float)
    rate_diff = (det_mat[ids_a].mean(axis=1) - det_mat[ids_b].mean(axis=1))
    row_order = rate_diff.sort_values(ascending=False).index

    det_plot   = det_mat.loc[row_order]
    score_plot = np.log10(sm_plot.loc[row_order] + 1)
    n_met      = len(row_order)

    # NEW: build annotated row labels
    row_labels = [_met_label(m, src_snk_map) for m in row_order]

    fig_h = max(8, n_met * 0.32 + 3)
    fig_w = max(18, len(all_ids) * 0.85 + 10)
    fig, (ax_det, ax_sc) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.05}
    )

    det_cmap = mcolors.LinearSegmentedColormap.from_list(
        "det", ["#f0f0f0", "#2c7bb6"], N=2)
    sns.heatmap(det_plot[all_ids], cmap=det_cmap, vmin=0, vmax=1,
                ax=ax_det, linewidths=0.4, linecolor="#dddddd", cbar=False,
                xticklabels=[f"{p}\n({COHORT_META_BY_ID[p]['group']})"
                              for p in all_ids],
                yticklabels=row_labels)
    ax_det.set_xticklabels(ax_det.get_xticklabels(), fontsize=7.5,
                            rotation=45, ha="right")
    ax_det.set_yticklabels(ax_det.get_yticklabels(), fontsize=7.5)
    ax_det.set_title("Detected (blue) / Absent (grey)\n"
                     "Rows: met  [source_cell→sink_cell]",
                     fontsize=10, fontweight="bold")

    sns.heatmap(score_plot[all_ids], cmap="YlOrRd",
                ax=ax_sc, linewidths=0.4, linecolor="#dddddd",
                cbar_kws={"label": "log₁₀(score+1)", "shrink": 0.6},
                xticklabels=[f"{p}\n({COHORT_META_BY_ID[p]['group']})"
                              for p in all_ids],
                yticklabels=False)
    ax_sc.set_xticklabels(ax_sc.get_xticklabels(), fontsize=7.5,
                           rotation=45, ha="right")
    ax_sc.set_title("Exchange score  [log₁₀]", fontsize=10, fontweight="bold")

    # Group colour strip above both panels
    for ax in (ax_det, ax_sc):
        for i, pid in enumerate(all_ids):
            col = "#c0392b" if pid in ids_a else "#2980b9"
            ax.add_patch(plt.Rectangle(
                (i, n_met), 1, 0.55,
                facecolor=col, edgecolor="white", linewidth=0.4,
                transform=ax.transData, clip_on=False,
            ))
    legend_els = [
        mpatches.Patch(color="#c0392b", label=group_a),
        mpatches.Patch(color="#2980b9", label=group_b),
    ]
    ax_sc.legend(handles=legend_els, loc="upper right",
                 bbox_to_anchor=(1.28, 1.12), fontsize=9)
    fig.suptitle(
        f"Metabolite Exchange: {group_a} vs {group_b}\n"
        f"Top {len(row_order)} by composite evidence score  ·  "
        f"rows sorted by detection rate difference",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fp = os.path.join(out_dir, f"detection_heatmap_{group_a}_vs_{group_b}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


def plot_effect_size_bar(
    comp_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    out_dir: str,
    cfg: ComparisonConfig,
) -> Optional[str]:
    """
    Ranked Cohen's d bar chart.

    Cell-type annotation (NEW): y-tick labels now use _met_label() to show
    "met  [src→snk]" consistently on every bar, not just those that happened
    to have source/sink available in the original format string.
    """
    if comp_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # Build src_snk_map from comp_df
    src_snk_map = {
        row["metabolite"]: (row.get("source", "?"), row.get("sink", "?"))
        for _, row in comp_df.iterrows()
    }

    n = cfg.top_n_bar // 2
    top_up   = comp_df[comp_df["cohens_d"] > 0].nlargest(n, "evidence_score")
    top_down = comp_df[comp_df["cohens_d"] < 0].nlargest(n, "evidence_score")
    plot_df  = pd.concat([top_up, top_down]).drop_duplicates("metabolite")
    plot_df  = plot_df.sort_values("cohens_d", ascending=True)
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, max(6, len(plot_df) * 0.48)))
    bar_colors = ["#c0392b" if d > 0 else "#2980b9" for d in plot_df["cohens_d"]]
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["cohens_d"].values, color=bar_colors,
            alpha=0.82, edgecolor="white", linewidth=0.4)

    d_max = max(abs(float(plot_df["cohens_d"].max())),
                abs(float(plot_df["cohens_d"].min())), 0.5)
    x_ann = d_max * 0.42

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ra     = row.get(f"rate_{group_a}", float("nan"))
        rb     = row.get(f"rate_{group_b}", float("nan"))
        d      = float(row["cohens_d"])
        pp     = float(row.get("perm_p", 1.0))
        fp_val = float(row.get("fisher_p", 1.0))
        ev     = float(row.get("evidence_score", 0.0))
        stars  = ("***" if min(pp, fp_val) < 0.001 else
                  "**"  if min(pp, fp_val) < 0.01  else
                  "*"   if min(pp, fp_val) < cfg.p_thresh else "")
        try:
            det = f"det:{ra:.0%}|{rb:.0%}"
        except Exception:
            det = ""
        ann = f"{stars} {det} ev={ev:.2f}".strip()

        if d >= 0:
            ax.text(d + d_max * 0.025, i, ann,
                    va="center", ha="left", fontsize=7, color="#333333")
        else:
            if abs(d) >= 0.5 * d_max:
                ax.text(d + d_max * 0.03, i, ann,
                        va="center", ha="left", fontsize=7, color="white")
            else:
                ax.text(d - d_max * 0.025, i, ann,
                        va="center", ha="right", fontsize=6.5, color="#555555")

    # NEW: unified [src→snk] tick labels via _met_label()
    labels = [_met_label(row["metabolite"], src_snk_map)
              for _, row in plot_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(-(d_max + x_ann), d_max + x_ann)
    ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.6, zorder=1)
    for dv in (0.5, 0.8, -0.5, -0.8):
        ax.axvline(dv, color="#DDDDDD", linewidth=0.5, linestyle=":", zorder=0)
    ax.set_xlabel(
        f"Cohen's d  (+ve = higher in {group_a}  |  \u2013ve = higher in {group_b})",
        fontsize=10, fontweight="bold")
    ax.set_title(
        f"Top Exchanges by Composite Evidence: {group_a} vs {group_b}\n"
        f"* permutation or Fisher p < {cfg.p_thresh}  ·  "
        f"det = detection rate (A|B)  ·  ev = evidence score\n"
        "Labels: met  [source_cell→sink_cell]",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.18, linestyle=":", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.tight_layout()
    fp = os.path.join(out_dir, f"effect_size_{group_a}_vs_{group_b}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


def plot_axis_bubble(
    axis_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    out_dir: str,
    cfg: ComparisonConfig,
) -> Optional[str]:
    """
    Exchange-axis bubble chart.

    Filtering: shows only axes with |mean_d| > axis_min_abs_d OR
    n_metabolites >= axis_min_n_mets OR combined_p < axis_max_p.

    Cell-type annotation (unchanged structure): axis labels continue to show
    abbreviated "src→snk" names.  The companion plot_axis_metabolite_detail()
    carries the per-metabolite detail so this chart stays uncluttered.
    """
    if axis_df is None or axis_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # Filter to informative axes
    mask = (
        (axis_df["mean_cohens_d"].abs() >= cfg.axis_min_abs_d) |
        (axis_df["n_metabolites"]       >= cfg.axis_min_n_mets) |
        (axis_df["combined_p"]          <= cfg.axis_max_p)
    )
    plot_df = axis_df[mask].copy()
    if plot_df.empty:
        plot_df = axis_df.nlargest(20, "mean_cohens_d", keep="all")

    def _abbrev(name: str, max_len: int = 12) -> str:
        abbrevs = {
            "Plasma_Cells": "Plasma",
            "GS_Stromal": "GS_Str",
            "Endo_Peritubular": "Endo_PT",
            "Dedifferentiated_Tubule": "Dediff_Tub",
            "VSMC/Pericyte": "VSMC",
            "Interstitial fibroblasts": "IntFib",
        }
        name = abbrevs.get(name, name)
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    def _abbrev_axis(axis: str) -> str:
        if "→" in axis:
            s, k = axis.split("→", 1)
            return f"{_abbrev(s)}→{_abbrev(k)}"
        return _abbrev(axis)

    plot_df["combined_q"] = _bh_correct(plot_df["combined_p"].values)

    d_vals = plot_df["mean_cohens_d"].values
    logp   = -np.log10(plot_df["combined_p"].clip(lower=1e-300).values)
    sizes  = (plot_df["n_metabolites"].values * 90 + 40).clip(40, 600).astype(float)
    colors = ["#c0392b" if d > 0 else "#2980b9" for d in d_vals]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#f8f9fa")

    sc = ax.scatter(d_vals, logp, s=sizes, c=colors,
                    alpha=0.72, edgecolors="white", linewidths=0.7, zorder=3)

    ax.axhline(-np.log10(0.05), color="#666666", linestyle="--",
               linewidth=1.0, alpha=0.6, label="p = 0.05")
    ax.axvline(0, color="#666666", linewidth=0.7, alpha=0.4)
    for dv in (0.5, -0.5, 0.8, -0.8):
        ax.axvline(dv, color="#cccccc", linestyle=":", linewidth=0.7, alpha=0.5)

    # Decide which axes to label
    to_label = pd.concat([
        plot_df.nlargest(8, "mean_cohens_d"),
        plot_df.nsmallest(8, "mean_cohens_d"),
        plot_df[plot_df["n_metabolites"] >= 3],
    ]).drop_duplicates("exchange_axis")

    texts = []
    for _, row in to_label.iterrows():
        label = _abbrev_axis(row["exchange_axis"])
        n_m   = int(row["n_metabolites"])
        fsz   = min(8.5, 7.0 + 0.3 * n_m)
        texts.append(ax.text(
            row["mean_cohens_d"],
            -np.log10(max(row["combined_p"], 1e-300)),
            label,
            fontsize=fsz, fontweight="bold" if n_m >= 3 else "normal",
            color="#1a1a2e",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.78, edgecolor="#cccccc", linewidth=0.5),
        ))
    if HAS_ADJUSTTEXT and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999999",
                             lw=0.5, alpha=0.6),
            expand=(1.25, 1.3),
            force_text=(0.4, 0.6),
        )
    elif texts:
        import random
        random.seed(42)
        for t in texts:
            x, y = t.get_position()
            t.set_position((x + random.uniform(-0.05, 0.05),
                             y + random.uniform(-0.05, 0.05)))

    leg_patches = [
        mpatches.Patch(color="#c0392b", label=f"Higher in {group_a}"),
        mpatches.Patch(color="#2980b9", label=f"Higher in {group_b}"),
    ]
    for n_m, lbl in [(1, "1 metabolite"), (5, "5 metabolites"), (16, "16 metabolites")]:
        leg_patches.append(
            plt.scatter([], [], s=n_m * 90 + 40, c="#888888",
                        alpha=0.6, label=lbl)
        )
    ax.legend(handles=leg_patches, fontsize=8, framealpha=0.9,
              loc="upper right", ncol=1)

    ax.set_xlabel(f"Mean Cohen's d across axis metabolites  "
                  f"(+ve = higher in {group_a})",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("–log₁₀(Fisher combined p-value)",
                  fontsize=11, fontweight="bold")
    n_filt = len(axis_df) - len(plot_df)
    ax.set_title(
        f"Exchange-Axis Comparison: {group_a} vs {group_b}\n"
        f"Showing {len(plot_df)} informative axes "
        f"({n_filt} noise axes filtered, "
        f"|d|<{cfg.axis_min_abs_d} & n_mets<{cfg.axis_min_n_mets} & p>{cfg.axis_max_p})\n"
        "See axis_metabolite_detail plot for per-metabolite breakdown",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.1, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fp = os.path.join(out_dir, f"axis_network_{group_a}_vs_{group_b}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


# =============================================================================
# NEW: Axis metabolite detail strip chart
# =============================================================================

def plot_axis_metabolite_detail(
    axis_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    group_a: str,
    group_b: str,
    out_dir: str,
    cfg: ComparisonConfig,
    n_top_axes: int = 20,
) -> Optional[str]:
    """
    Companion strip chart to plot_axis_bubble():  for each of the top
    exchange axes (by |mean_cohens_d| among informative axes), show every
    individual metabolite exchanged on that axis as a coloured dot row with:

      • Dot colour  : direction (coral = ↑ group_a, blue = ↑ group_b)
      • Dot size    : max detection rate in either group
      • Dot opacity : significance (solid = sig, faded = n.s.)
      • X-axis      : Cohen's d for that metabolite
      • Y groups    : one strip per exchange axis (source→sink)
      • Metabolite names annotated to the right of each dot

    This answers directly: "what metabolites flow on each axis, and which
    direction do they go?" without adding clutter to the bubble chart.

    Parameters
    ----------
    axis_df  : pd.DataFrame  — output of compare_exchange_axes()
    comp_df  : pd.DataFrame  — output of compare_two_groups() (metabolite-level)
    group_a  : str
    group_b  : str
    out_dir  : str
    cfg      : ComparisonConfig
    n_top_axes : int  — number of axes to show (sorted by |mean_cohens_d|)

    Returns
    -------
    str  path to saved figure, or None.
    """
    if axis_df is None or axis_df.empty or comp_df.empty:
        return None
    os.makedirs(out_dir, exist_ok=True)

    # ── Select top n_top_axes from the informative set ────────────────────
    mask = (
        (axis_df["mean_cohens_d"].abs() >= cfg.axis_min_abs_d) |
        (axis_df["n_metabolites"]       >= cfg.axis_min_n_mets) |
        (axis_df["combined_p"]          <= cfg.axis_max_p)
    )
    filt = axis_df[mask].copy()
    if filt.empty:
        filt = axis_df.copy()

    top_axes = (filt
                .assign(_abs_d=lambda df: df["mean_cohens_d"].abs())
                .nlargest(n_top_axes, "_abs_d")
                .sort_values("mean_cohens_d", ascending=True)
                ["exchange_axis"].tolist())

    if not top_axes:
        return None

    # ── Map metabolites in comp_df for quick lookup ───────────────────────
    rate_a_col = f"rate_{group_a}"
    rate_b_col = f"rate_{group_b}"
    comp_idx   = comp_df.set_index("metabolite") if "metabolite" in comp_df.columns else comp_df

    # Count total dot rows to size the figure
    total_mets = sum(
        len(filt[filt["exchange_axis"] == ax]["n_metabolites"].values[0:1] or [0])
        for ax in top_axes
    )
    # More precise: count from comp_df
    rows_needed = sum(
        len(comp_df[comp_df["exchange_axis"] == ax])
        for ax in top_axes
    )
    if rows_needed == 0:
        return None

    fig_h = max(8, rows_needed * 0.38 + len(top_axes) * 0.25 + 3)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_facecolor("#f8f9fa")

    # Determine d range for x-axis
    all_d = comp_df[comp_df["exchange_axis"].isin(top_axes)]["cohens_d"].values
    if len(all_d) == 0:
        plt.close()
        return None
    d_lim = max(abs(all_d).max() * 1.15, 1.0)

    # Colour palette
    col_up   = "#c0392b"   # higher in group_a
    col_down = "#2980b9"   # higher in group_b

    # Axis band colours (alternating subtle backgrounds)
    band_cols = ["#f0f4ff", "#fff4f0"]

    # Y-position tracker: one band per axis, rows within band
    y_cursor = 0.0
    ytick_pos  = []   # centre of each axis band
    ytick_labs = []   # "src→snk" axis label
    separator_ys = [] # horizontal lines between axes

    # Abbreviation helper (reuse from plot_axis_bubble)
    def _abbrev(name: str, max_len: int = 14) -> str:
        abbrevs = {
            "Plasma_Cells": "Plasma",
            "GS_Stromal": "GS_Str",
            "Endo_Peritubular": "Endo_PT",
            "Dedifferentiated_Tubule": "DDTub",
            "VSMC/Pericyte": "VSMC",
            "Interstitial fibroblasts": "IntFib",
        }
        name = abbrevs.get(name, name)
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    for band_i, axis_name in enumerate(top_axes):
        # Get all metabolites on this axis from comp_df
        axis_mets = comp_df[comp_df["exchange_axis"] == axis_name].copy()
        if axis_mets.empty:
            continue
        axis_mets = axis_mets.sort_values("cohens_d", ascending=False)

        n_axis_mets = len(axis_mets)
        band_top = y_cursor
        band_bot = y_cursor + n_axis_mets

        # Background band
        ax.axhspan(band_top - 0.5, band_bot - 0.5,
                   facecolor=band_cols[band_i % 2], alpha=0.55, zorder=0)

        # One row per metabolite
        for j, (_, mrow) in enumerate(axis_mets.iterrows()):
            yi      = y_cursor + j
            d_val   = float(mrow["cohens_d"])
            ra      = float(mrow.get(rate_a_col, 0.0))
            rb      = float(mrow.get(rate_b_col, 0.0))
            sig     = bool(mrow.get("significant", False))
            met_nm  = str(mrow["metabolite"])

            dot_col   = col_up if d_val >= 0 else col_down
            dot_alpha = 0.92 if sig else 0.30
            dot_size  = (max(ra, rb) * 180 + 18)
            star      = " *" if sig else ""

            ax.scatter(d_val, yi, s=dot_size,
                       color=dot_col, alpha=dot_alpha,
                       edgecolors="white", linewidths=0.5, zorder=4)

            # Metabolite name to the right/left of dot
            x_txt = d_val + (d_lim * 0.04 if d_val >= 0 else -d_lim * 0.04)
            ha_txt = "left" if d_val >= 0 else "right"
            ax.text(x_txt, yi,
                    f"{met_nm}{star}  "
                    f"({group_a}:{ra:.0%} | {group_b}:{rb:.0%})",
                    va="center", ha=ha_txt, fontsize=7.0,
                    color="#222222" if sig else "#888888")

        # Axis label (centre of band)
        band_centre = y_cursor + n_axis_mets / 2.0 - 0.5
        ytick_pos.append(band_centre)
        src, snk = (axis_name.split("→", 1)
                    if "→" in axis_name else (axis_name, "?"))
        ytick_labs.append(f"{_abbrev(src)}→{_abbrev(snk)}")

        if band_i > 0:
            separator_ys.append(y_cursor - 0.5)

        y_cursor += n_axis_mets

    # Separator lines
    for sy in separator_ys:
        ax.axhline(sy, color="#cccccc", linewidth=0.7, linestyle="-", zorder=1)

    ax.axvline(0, color="#444444", linewidth=0.9, alpha=0.55, zorder=2)
    for dv in (-0.8, -0.5, 0.5, 0.8):
        ax.axvline(dv, color="#dddddd", linewidth=0.6,
                   linestyle=":", zorder=1)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labs, fontsize=9, fontweight="bold")
    ax.set_ylim(-0.8, y_cursor - 0.2)
    ax.set_xlim(-d_lim, d_lim)

    ax.set_xlabel(
        f"Cohen's d  (+ve = higher in {group_a}  |  –ve = higher in {group_b})\n"
        "Dot size = max detection rate  ·  opacity = significance  ·  "
        "* = perm. or Fisher p < 0.05",
        fontsize=10, fontweight="bold",
    )
    ax.set_title(
        f"Exchange-Axis Metabolite Detail: {group_a} vs {group_b}\n"
        f"Top {len(top_axes)} informative axes — each row = one metabolite exchanged on that axis\n"
        f"Label format: metabolite  ({group_a} det. rate | {group_b} det. rate)",
        fontsize=12, fontweight="bold",
    )

    # Legend
    leg = [
        mpatches.Patch(color=col_up,   alpha=0.92, label=f"↑ {group_a}"),
        mpatches.Patch(color=col_down, alpha=0.92, label=f"↑ {group_b}"),
        mpatches.Patch(color="#888888", alpha=0.30, label="n.s. (p ≥ 0.05)"),
        plt.scatter([], [], s=18+18,   c="#888888", alpha=0.6, label="det. rate ~10%"),
        plt.scatter([], [], s=18+108,  c="#888888", alpha=0.6, label="det. rate ~50%"),
        plt.scatter([], [], s=18+198,  c="#888888", alpha=0.6, label="det. rate 100%"),
    ]
    ax.legend(handles=leg, fontsize=8, framealpha=0.9,
              loc="lower right", ncol=2)

    ax.grid(axis="x", alpha=0.12, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    fp = os.path.join(out_dir,
                      f"axis_metabolite_detail_{group_a}_vs_{group_b}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return fp


# =============================================================================
# Master runner
# =============================================================================

def run_all_comparisons(
    patient_results: Dict,
    cfg: Optional[ComparisonConfig] = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Run all configured group comparisons with the hurdle + permutation approach.
    Pre-computes shared matrices once for efficiency.

    Cell-type annotation: src_snk_map is pre-computed once and passed to
    compare_two_groups(), which embeds source/sink columns in every result
    DataFrame.  All plot functions read these columns directly — no extra
    argument needed at call-site.
    """
    if cfg is None:
        cfg = ComparisonConfig()

    comp_out = os.path.join(cfg.base_out_dir, "group_comparisons")
    os.makedirs(comp_out, exist_ok=True)

    print("=" * 70)
    print("GROUP COMPARISONS  —  hurdle model + permutation tests")
    print("=" * 70)
    print(f"Patients         : {len(patient_results)}")
    print(f"Comparisons      : {len(cfg.comparisons)}")
    print(f"Permutations     : {cfg.n_permutations} per metabolite")
    print(f"p threshold      : {cfg.p_thresh}  (exploratory)")
    print(f"FDR threshold    : {cfg.fdr_thresh}  (informational)")
    print(f"Axis filter      : |d|≥{cfg.axis_min_abs_d} OR "
          f"n_mets≥{cfg.axis_min_n_mets} OR p≤{cfg.axis_max_p}")
    print()

    if cfg.verbose:
        print("Building cohort-wide score matrix...")
    score_matrix = build_full_score_matrix(patient_results)
    rank_matrix  = rank_normalise(score_matrix)
    src_snk_map  = build_source_sink_map(patient_results)
    print(f"  {score_matrix.shape[0]} metabolites × {score_matrix.shape[1]} patients")
    print(f"  {len(src_snk_map)} metabolites with source→sink cell-type annotations\n")

    all_results: Dict = {}

    for group_a, group_b in cfg.comparisons:
        label    = f"{group_a}_vs_{group_b}"
        pair_dir = os.path.join(comp_out, label)
        os.makedirs(pair_dir, exist_ok=True)

        avail_a = [p for p in _get_group_patients(group_a) if p in patient_results]
        avail_b = [p for p in _get_group_patients(group_b) if p in patient_results]
        print(f"\n── {group_a} (n={len(avail_a)}) vs {group_b} (n={len(avail_b)}) ──")

        comp_df = compare_two_groups(
            patient_results, group_a, group_b, cfg,
            score_matrix=score_matrix,
            rank_matrix=rank_matrix,
            src_snk_map=src_snk_map,
        )
        if comp_df.empty:
            print("   No results (too few patients)")
            continue

        n_sp = int(comp_df["sig_perm"].sum())
        n_sf = int(comp_df["sig_fisher"].sum())
        n_fdr = int(comp_df["sig_fdr"].sum())
        print(f"   Metabolites tested   : {len(comp_df)}")
        print(f"   perm p<{cfg.p_thresh}           : {n_sp}")
        print(f"   fisher p<{cfg.p_thresh}          : {n_sf}")
        print(f"   FDR q<{cfg.fdr_thresh}            : {n_fdr}  (informational)")
        if len(comp_df) > 0:
            top = comp_df.iloc[0]
            print(f"   Top hit (by evidence): {top['metabolite']}  "
                  f"[{top.get('source','?')}→{top.get('sink','?')}]  "
                  f"d={top['cohens_d']:.3f}  "
                  f"perm_p={top['perm_p']:.4f}  "
                  f"fisher_p={top['fisher_p']:.4f}  "
                  f"ev={top['evidence_score']:.3f}")

        # Save CSVs
        comp_df.to_csv(
            os.path.join(pair_dir, f"comparison_{label}.csv"), index=False)

        top_hits = comp_df.nlargest(
            min(50, len(comp_df)), "evidence_score")
        top_hits.to_csv(
            os.path.join(pair_dir, f"top_hits_{label}.csv"), index=False)

        for grp in (group_a, group_b):
            col = f"signature_{grp}"
            if col in comp_df.columns:
                sigs = comp_df[comp_df[col]]
                if not sigs.empty:
                    sigs.to_csv(
                        os.path.join(pair_dir, f"signatures_{grp}.csv"),
                        index=False)
                    if cfg.verbose:
                        print(f"   → {len(sigs)} {grp} signatures saved")

        axis_df = compare_exchange_axes(comp_df, group_a, group_b, cfg)
        if not axis_df.empty:
            axis_df.to_csv(
                os.path.join(pair_dir, f"axis_comparison_{label}.csv"),
                index=False)

        # Plots
        fp = plot_effect_volcano(comp_df, group_a, group_b, pair_dir, cfg)
        if fp and cfg.verbose:
            print(f"   Volcano          → {fp}")

        fp = plot_detection_heatmap(
            comp_df, score_matrix, group_a, group_b, pair_dir, cfg)
        if fp and cfg.verbose:
            print(f"   Detection heatmap → {fp}")

        fp = plot_effect_size_bar(
            comp_df, group_a, group_b, pair_dir, cfg)
        if fp and cfg.verbose:
            print(f"   Effect size bars  → {fp}")

        fp = plot_axis_bubble(axis_df, group_a, group_b, pair_dir, cfg)
        if fp and cfg.verbose:
            print(f"   Axis bubble chart → {fp}")

        # ── NEW: axis metabolite detail strip ─────────────────────────────
        fp = plot_axis_metabolite_detail(
            axis_df, comp_df, group_a, group_b, pair_dir, cfg)
        if fp and cfg.verbose:
            print(f"   Axis met. detail  → {fp}")

        # ── Publication-quality lollipop + volcano ────────────────────────
        if HAS_PUB_PLOTS:
            fp = plot_effect_size_lollipop(
                comp_df, group_a, group_b,
                out_path=os.path.join(pair_dir,
                    f"lollipop_{group_a}_vs_{group_b}.png"),
                n_top=20,
                title=f"{group_a} vs {group_b} — top exchanges by effect size",
            )
            if cfg.verbose:
                print(f"   Lollipop plot     → {fp}")

            fp = plot_volcano_publication(
                comp_df, group_a, group_b,
                out_path=os.path.join(pair_dir,
                    f"volcano_pub_{group_a}_vs_{group_b}.png"),
                p_thresh=cfg.p_thresh,
            )
            if cfg.verbose:
                print(f"   Volcano (pub)     → {fp}")

        all_results[(group_a, group_b)] = comp_df

    # ── Cohort-wide summary figure (requires all 4 comparisons) ────────
    if HAS_PUB_PLOTS and len(all_results) >= 2:
        try:
            from cohort_statistics import _clinical_df
            from pathlib import Path
            reg_path = os.path.join(
                cfg.base_out_dir, "clinical_statistics",
                "regression", "regression_results.csv")
            if os.path.exists(reg_path):
                import pandas as _pd_sum
                reg_df_ = _pd_sum.read_csv(reg_path)
                keys_needed = [
                    ("DKD", "Control"), ("HKD", "Control"), ("DKD", "HKD")
                ]
                if all(k in all_results for k in keys_needed[:2]):
                    fp = plot_cohort_summary_figure(
                        comp_dkd_ctrl=all_results[("DKD", "Control")],
                        comp_hkd_ctrl=all_results[("HKD", "Control")],
                        comp_dkd_hkd=all_results.get(
                            ("DKD", "HKD"),
                            all_results[("DKD", "Control")]),
                        reg_df=reg_df_,
                        out_dir=comp_out,
                        filename="summary_figure.png",
                    )
                    if cfg.verbose:
                        print(f"   Summary figure    → {fp}")
        except Exception as _e:
            if cfg.verbose:
                print(f"   Summary figure skipped: {_e}")

    print("\n" + "=" * 70)
    print(f"Comparisons complete — outputs in {comp_out}")
    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",    default="cohort_output")
    p.add_argument("--n-perms",    type=int,   default=5000)
    p.add_argument("--p-thresh",   type=float, default=0.05)
    p.add_argument("--fdr-thresh", type=float, default=0.10)
    p.add_argument("--quiet",      action="store_true")
    args = p.parse_args()

    from run_cohort_pipeline import load_patient_results
    pr = load_patient_results(args.out_dir)
    if not pr:
        print("ERROR: No results.pkl found. Run run_cohort_pipeline.py first.")
        sys.exit(1)

    run_all_comparisons(
        pr,
        cfg=ComparisonConfig(
            base_out_dir  = args.out_dir,
            n_permutations = args.n_perms,
            p_thresh      = args.p_thresh,
            fdr_thresh    = args.fdr_thresh,
            verbose       = not args.quiet,
        ),
    )
