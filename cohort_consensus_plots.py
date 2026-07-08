#!/usr/bin/env python3
"""
cohort_consensus_plots.py  (v7)
================================
Cohort-wide consensus metabolite-exchange summary plots.

Layout matches the preferred "simpler_version.png":

  Row 0 (tall):  violin | dot×3 | topology
  Row 1 (med):   donut×3 | detection strip
  Row 2 (slim):  clinical correlation strip


Output folders
--------------
  plots_consensus_cohort/      ← this module (simple summary figures)
  merged_plots/                ← merged_streamline_plots.py (streamline 3-panel)
  cohort_comparison_wceg/      ← cohort_comparison.py (WCEG statistical comparison)
"""

import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from plotting import _should_exclude_metabolite
except ImportError:
    def _should_exclude_metabolite(m): return False

try:
    from consensus_exchange_network import build_exchange_records
    _HAS_CEG = True
except ImportError:
    _HAS_CEG = False

COHORT_GROUPS = ["Control", "HKD", "DKD"]
GROUP_COLORS  = {"Control": "#2ecc71", "HKD": "#e67e22", "DKD": "#e74c3c"}
GROUP_LABELS  = {"Control": "Control",  "HKD": "HKD",     "DKD": "DKD"}

try:
    from run_cohort_pipeline import COHORT_METADATA
    _PATIENT_GROUP = {m["id"]: m["group"] for m in COHORT_METADATA}
    _PATIENT_META  = {m["id"]: m            for m in COHORT_METADATA}
except ImportError:
    _PATIENT_GROUP = {}
    _PATIENT_META  = {}


# ===========================================================================
# Helpers
# ===========================================================================

def _patient_group(pid):
    return _PATIENT_GROUP.get(pid, "Unknown")

def _get_condition_for_patient(results):
    conds = getattr(getattr(results, "config", None), "conditions", [])
    return conds[0] if conds else None

def _clinical_series(patient_results, field):
    return pd.Series({
        pid: float(_PATIENT_META[pid][field])
        for pid in patient_results
        if field in _PATIENT_META.get(pid, {})
    })

def _ct_color(ct):
    palette = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
               "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
               "#aecbf0","#ffbe7d","#a9c574","#87c9a1","#d4b5d4"]
    return palette[abs(hash(ct)) % len(palette)]


# ===========================================================================
# Score tables (shared with other modules)
# ===========================================================================

def build_cohort_metabolite_table(patient_results: Dict) -> pd.DataFrame:
    """Long-form: (patient_id, group, metabolite, source, sink, score)."""
    rows = []
    for pid, results in patient_results.items():
        group = _patient_group(pid)
        cond  = _get_condition_for_patient(results)
        per_region = getattr(results, "per_region_data", {}) or {}
        seen: Dict = {}
        for key, data in per_region.items():
            if cond and not key.startswith(cond):
                continue
            for met, pairs in (data.get("interactions", {}) or {}).items():
                if _should_exclude_metabolite(met) or not pairs:
                    continue
                best = max(pairs, key=lambda p: float(p.get("score", 0)))
                src  = str(best.get("source", "?"))
                snk  = str(best.get("sink",   "?"))
                scr  = float(best.get("score", 0.0))
                if met not in seen or scr > seen[met][2]:
                    seen[met] = (src, snk, scr)
        for met, (src, snk, scr) in seen.items():
            rows.append({"patient_id": pid, "group": group,
                         "metabolite": met, "source": src,
                         "sink": snk, "score": scr})
    if not rows:
        return pd.DataFrame(columns=["patient_id","group","metabolite",
                                     "source","sink","score"])
    return pd.DataFrame(rows)


def compute_composition_corrected_scores(
    cohort_df: pd.DataFrame,
    patient_results: Dict,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """S_corr = log2(S_obs/(pi_src*pi_snk*N_eff)+1). Reuses CEG module if available."""
    if _HAS_CEG:
        return build_exchange_records(patient_results, epsilon=epsilon)
    # Inline fallback
    from collections import Counter as _C
    rows = []
    for pid, results in patient_results.items():
        cond = _get_condition_for_patient(results)
        per_region = getattr(results, "per_region_data", {}) or {}
        ct = _C()
        for key, data in per_region.items():
            if cond and not key.startswith(cond): continue
            mdf = data.get("meta_df", None)
            if mdf is None or mdf.empty: continue
            for col in ["cell_type","Graph.based","Idents"]:
                if col in mdf.columns:
                    ct.update(mdf[col].dropna().astype(str).value_counts().to_dict()); break
        N = max(sum(ct.values()), 1)
        sub = cohort_df[cohort_df["patient_id"] == pid]
        for _, row in sub.iterrows():
            src, snk = str(row["source"]), str(row["sink"])
            S_obs = float(row["score"])
            N_src = ct.get(src, 0); N_snk = ct.get(snk, 0)
            pi_s = N_src/N; pi_k = N_snk/N
            N_eff = np.sqrt(max(N_src,1)*max(N_snk,1))
            ES = S_obs / max(pi_s*pi_k*N_eff, epsilon)
            rows.append({
                "patient_id": pid, "group": str(row["group"]),
                "metabolite": str(row["metabolite"]),
                "source": src, "sink": snk,
                "score_raw": round(S_obs,4),
                "N_src": N_src, "N_snk": N_snk, "N_total": N,
                "pi_src": round(pi_s,5), "pi_snk": round(pi_k,5),
                "N_eff": round(N_eff,2), "ES": round(ES,4),
                "score_corr": round(float(np.log2(ES+1.0)),4),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_score_matrix_corrected(corr_scores_df: pd.DataFrame,
                                  use_corrected: bool = True) -> pd.DataFrame:
    if corr_scores_df is None or corr_scores_df.empty:
        return pd.DataFrame()
    col = "score_corr" if use_corrected else "score_raw"
    return (corr_scores_df
            .groupby(["metabolite","patient_id"])[col]
            .sum().unstack(fill_value=0.0))


def build_score_matrix_for_ranking(patient_results: Dict,
                                    cohort_df: pd.DataFrame) -> pd.DataFrame:
    if cohort_df.empty:
        return pd.DataFrame()
    return (cohort_df.groupby(["metabolite","patient_id"])["score"]
            .sum().unstack(fill_value=0.0))


# ===========================================================================
# Clinical correlations
# ===========================================================================

def compute_clinical_correlations(
    score_matrix: pd.DataFrame,
    patient_results: Dict,
    clinical_fields: Optional[List[str]] = None,
    min_patients: int = 5,
) -> pd.DataFrame:
    if clinical_fields is None:
        clinical_fields = ["eGFR","fibrosis"]
    if score_matrix.empty:
        return pd.DataFrame(columns=["metabolite"])
    rows = []
    for met in score_matrix.index:
        if _should_exclude_metabolite(met): continue
        row = {"metabolite": met}
        s = score_matrix.loc[met]
        for field in clinical_fields:
            clin   = _clinical_series(patient_results, field)
            common = s.index.intersection(clin.index)
            if len(common) < min_patients:
                row[f"rho_{field}"] = 0.0; row[f"p_{field}"] = 1.0; continue
            try:
                rho, p = scipy_stats.spearmanr(
                    s[common].values.astype(float),
                    clin[common].values.astype(float))
                row[f"rho_{field}"] = float(rho) if np.isfinite(rho) else 0.0
                row[f"p_{field}"]   = float(p)   if np.isfinite(p)   else 1.0
            except Exception:
                row[f"rho_{field}"] = 0.0; row[f"p_{field}"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["metabolite"])


def _clinical_boost(corr_df, met, weights=None, p_s=0.10, p_w=0.25):
    if weights is None: weights = {"eGFR":1.2,"fibrosis":1.0}
    if corr_df.empty or met not in corr_df["metabolite"].values: return 0.0
    cr = corr_df[corr_df["metabolite"]==met].iloc[0]
    tot = 0.0
    for f, w in weights.items():
        rho  = abs(float(cr.get(f"rho_{f}",0.0)))
        p    =     float(cr.get(f"p_{f}",  1.0))
        conf = 1.0 if p<p_s else (0.6 if p<p_w else 0.2)
        tot += w * rho * conf
    return tot


# ===========================================================================
# Data-driven metabolite selection
# ===========================================================================

def select_target_metabolites(
    cohort_df: pd.DataFrame,
    top_n: int = 20,
    patient_results: Optional[Dict] = None,
    corr_scores_df: Optional[pd.DataFrame] = None,
    clinical_fields: Optional[List[str]] = None,
    clinical_weight: float = 0.40,
    ckd_weight: float = 0.60,
    top_clinical_k: int = 5,
    return_metadata: bool = False,
):
    """
    Select metabolites: top_n by combined score UNION top_clinical_k per field.
    Returns (list, rank_df, clin_ext_dict) if return_metadata=True.
    """
    if clinical_fields is None: clinical_fields = ["eGFR","fibrosis"]
    if cohort_df.empty:
        return ([], pd.DataFrame(), {}) if return_metadata else []

    corr_df = pd.DataFrame()
    if patient_results and clinical_fields:
        sm = (build_score_matrix_corrected(corr_scores_df)
              if corr_scores_df is not None and not corr_scores_df.empty
              else build_score_matrix_for_ranking(patient_results, cohort_df))
        if not sm.empty:
            corr_df = compute_clinical_correlations(
                sm, patient_results, clinical_fields=clinical_fields)

    grp_counts   = cohort_df.groupby(["metabolite","group"])["patient_id"].nunique()
    total_by_grp = cohort_df.groupby("group")["patient_id"].nunique()
    mean_scores  = cohort_df.groupby("metabolite")["score"].mean()

    rows = []
    for met in cohort_df["metabolite"].unique():
        if _should_exclude_metabolite(met): continue
        dc = int(grp_counts.get((met,"Control"),0))
        dh = int(grp_counts.get((met,"HKD"),    0))
        dd = int(grp_counts.get((met,"DKD"),    0))
        nc = int(total_by_grp.get("Control",1))
        nh = int(total_by_grp.get("HKD",    1))
        nd = int(total_by_grp.get("DKD",    1))
        rc = dc/max(nc,1)
        ckd_ev = (((dh+dd)/max(nh+nd,1))*(1.0-rc+0.1)*
                  np.log1p(float(mean_scores.get(met,0.0)))*
                  np.log1p(dc+dh+dd))
        clin_b   = _clinical_boost(corr_df, met)
        combined = ckd_weight*ckd_ev + clinical_weight*clin_b
        reason   = ("both" if clin_b>0.3 and ckd_ev>0.1
                    else ("clinical" if clin_b>0.3 else "ckd_specificity"))
        rho_row = {}
        if not corr_df.empty and met in corr_df["metabolite"].values:
            cr = corr_df[corr_df["metabolite"]==met].iloc[0]
            for f in clinical_fields:
                rho_row[f"rho_{f}"] = round(float(cr.get(f"rho_{f}",0.0)),3)
                rho_row[f"p_{f}"]   = round(float(cr.get(f"p_{f}",  1.0)),4)
        rows.append({
            "metabolite": met,
            "ckd_evidence": round(ckd_ev,4),
            "clinical_boost": round(clin_b,4),
            "combined_score": round(combined,4),
            "det_ctrl":dc,"det_hkd":dh,"det_dkd":dd,
            "r_ctrl":round(rc,3),
            "r_hkd":round(dh/max(nh,1),3),
            "r_dkd":round(dd/max(nd,1),3),
            "selection_reason": reason,
            **rho_row,
        })

    rank_df  = pd.DataFrame(rows).sort_values("combined_score", ascending=False)
    top_mets = rank_df["metabolite"].head(top_n).tolist()

    clin_ext: Dict[str, List[str]] = {}
    if top_clinical_k > 0 and not corr_df.empty:
        for field in clinical_fields:
            rc = f"rho_{field}"
            if rc not in corr_df.columns: continue
            top_c = (corr_df.assign(_ar=lambda d: d[rc].abs())
                     .nlargest(top_clinical_k,"_ar")["metabolite"].tolist())
            clin_ext[field] = top_c

    combined_set  = set(top_mets)
    combined_list = list(top_mets)
    for field, mets in clin_ext.items():
        for m in mets:
            if m not in combined_set and not _should_exclude_metabolite(m):
                combined_set.add(m)
                combined_list.append(m)
                mask = rank_df["metabolite"] == m
                rank_df.loc[mask,"selection_reason"] = f"clinical_{field}"

    return (combined_list, rank_df, clin_ext) if return_metadata else combined_list


# ===========================================================================
# Per-group data helpers
# ===========================================================================

def _get_group_data(cohort_df, corr_scores_df, group, metabolite, patient_results):
    pids_all  = [p for p in patient_results if _patient_group(p)==group]
    sub  = cohort_df[(cohort_df["metabolite"]==metabolite) &
                     (cohort_df["group"]==group)]
    csub = (corr_scores_df[
                (corr_scores_df["metabolite"]==metabolite) &
                (corr_scores_df["group"]==group)]
            if corr_scores_df is not None and not corr_scores_df.empty
            else pd.DataFrame())
    pids_with = sub["patient_id"].unique().tolist()
    ss_counts = Counter(zip(sub["source"],sub["sink"]))
    dom = ss_counts.most_common(1)
    dom_src, dom_snk = dom[0][0] if dom else ("?","?")
    from consensus_exchange_network import _get_ct_counts
    ct_comp = {}
    for pid in pids_all:
        ctc = _get_ct_counts(patient_results[pid])
        N   = max(sum(ctc.values()),1)
        for ct,n in ctc.items():
            ct_comp.setdefault(ct,[]).append(n/N)
    ct_mean = {ct:float(np.mean(v)) for ct,v in ct_comp.items()}
    return {
        "pids_in_group": pids_all,
        "patients_with": pids_with,
        "ss_counts":     ss_counts,
        "scores_raw":    sub["score"].tolist(),
        "scores_corr":   csub["score_corr"].tolist() if not csub.empty else [],
        "ct_mean":       ct_mean,
        "dominant_src":  dom_src,
        "dominant_snk":  dom_snk,
    }


def _get_topology_counts(cohort_df, group, metabolite, top_k=5):
    sub = (cohort_df[(cohort_df["group"]==group) &
                     (cohort_df["metabolite"]==metabolite)]
           if group else cohort_df[cohort_df["metabolite"]==metabolite])
    if sub.empty: return []
    c = (sub.groupby(["source","sink"])["patient_id"]
         .nunique().sort_values(ascending=False).head(top_k))
    return [(s,k,int(n)) for (s,k),n in c.items()]


# ===========================================================================
# Panel drawing — simple clean style
# ===========================================================================

def _draw_score_violin(ax, group_data_per_group, groups, metabolite):
    """Panel A: violin of S_corr per detected group only."""
    ax.set_facecolor("#f8f9fa")
    ax.spines[["top","right"]].set_visible(False)
    plot_groups = [g for g in groups
                   if (group_data_per_group.get(g,{}).get("scores_corr") or
                       group_data_per_group.get(g,{}).get("scores_raw"))]
    if not plot_groups:
        ax.text(0.5,0.5,"No data",transform=ax.transAxes,
                ha="center",va="center",fontsize=9,color="#aaa"); return
    data_list,positions,colors,xticks = [],[],[],[]
    for i,g in enumerate(plot_groups):
        gd = group_data_per_group.get(g,{})
        sc = gd.get("scores_corr") or gd.get("scores_raw") or []
        data_list.append(sc); positions.append(i); colors.append(GROUP_COLORS[g])
        xticks.append(f"{GROUP_LABELS[g]}\n(n={len(gd.get('patients_with',[]))})")
    try:
        parts = ax.violinplot(data_list,positions=positions,
                              widths=0.62,showmedians=True,showextrema=True)
        for pc,col in zip(parts["bodies"],colors):
            pc.set_facecolor(col); pc.set_alpha(0.50); pc.set_edgecolor("none")
        for k in ("cmedians","cmins","cmaxes","cbars"):
            if k in parts:
                parts[k].set_color("#333"); parts[k].set_linewidth(1.2)
    except Exception:
        pass
    rng = np.random.RandomState(42)
    for pos,sc,col in zip(positions,data_list,colors):
        jit = rng.uniform(-0.12,0.12,len(sc))
        ax.scatter(np.full(len(sc),pos)+jit,sc,color=col,s=28,alpha=0.82,
                   edgecolors="white",linewidths=0.5,zorder=4)
    ax.axhline(0,color="#bbb",lw=0.8,ls=":",zorder=1)
    ax.set_xticks(positions); ax.set_xticklabels(xticks,fontsize=7.5)
    ax.set_ylabel("S_corr = log₂(ES+1)",fontsize=7.5)
    ax.set_title("Corrected\nexchange score",fontsize=8,fontweight="bold",color="#333")
    ax.tick_params(axis="y",labelsize=7)


def _draw_patient_dotplot(ax, group_data, group, cohort_df,
                           metabolite, corr_scores_df, max_pts=7):
    """Panel B: patient dot-plot matrix. rows=topology, cols=patients."""
    ax.set_facecolor("#f8f9fa")
    ax.spines[["top","right"]].set_visible(False)
    pids = sorted(group_data.get("pids_in_group",[]))[:max_pts]
    topo = _get_topology_counts(cohort_df,group,metabolite,top_k=6)
    if not topo:
        ax.text(0.5,0.5,"No exchange\ndetected",
                transform=ax.transAxes,ha="center",va="center",
                fontsize=9,color="#888")
        ax.axis("off")
        ax.set_title(f"{GROUP_LABELS[group]}",fontsize=9,fontweight="bold",
                     color=GROUP_COLORS[group]); return
    csub = (corr_scores_df[
                (corr_scores_df["metabolite"]==metabolite)&
                (corr_scores_df["group"]==group)]
            if corr_scores_df is not None and not corr_scores_df.empty
            else pd.DataFrame())
    sc_lu = {}
    if not csub.empty:
        for _,r in csub.iterrows():
            sc_lu[(r["patient_id"],r["source"],r["sink"])] = float(r["score_corr"])
    row_labels = [f"{s[:9]}→{k[:9]}" for s,k,_ in topo]
    nr,nc = len(row_labels),len(pids)
    s_max = max(sc_lu.values()) if sc_lu else 1.0
    for ri,(src,snk,_) in enumerate(topo):
        for ci,pid in enumerate(pids):
            sc  = sc_lu.get((pid,src,snk),0.0); det = sc>0
            r_d = 0.40*(sc/max(s_max,1e-9)) if det else 0.0
            ax.add_patch(plt.Circle(
                (ci,ri),max(r_d,0.09 if det else 0.06),
                color=GROUP_COLORS[group] if det else "none",
                ec=GROUP_COLORS[group],lw=1.2 if not det else 0.4,
                alpha=0.85 if det else 0.28,zorder=3))
    ax.set_xlim(-0.7,nc-0.3); ax.set_ylim(-0.7,nr-0.3)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([p[-5:] for p in pids],fontsize=6,rotation=45,ha="right")
    ax.set_yticks(range(nr)); ax.set_yticklabels(row_labels,fontsize=6.5)
    ax.invert_yaxis()
    for ri in range(nr): ax.axhline(ri-0.5,color="#e0e0e0",lw=0.5,zorder=1)
    for ci in range(nc): ax.axvline(ci-0.5,color="#e0e0e0",lw=0.5,zorder=1)
    ax.set_aspect("equal"); ax.tick_params(length=2,pad=2)
    n_det = len(group_data.get("patients_with",[]))
    ax.set_title(f"{GROUP_LABELS[group]}  ({n_det}/{len(pids)} patients)",
                 fontsize=9,fontweight="bold",color=GROUP_COLORS[group],pad=5)


def _draw_composition_donut(ax, group_data, group, metabolite, max_slices=10):
    """Panel C: mean cell-type composition donut."""
    ct_mean  = group_data.get("ct_mean",{})
    dom_src  = group_data.get("dominant_src","?")
    dom_snk  = group_data.get("dominant_snk","?")
    has_exch = bool(group_data.get("patients_with",[]))
    if not ct_mean:
        ax.axis("off"); return
    sorted_ct = sorted(ct_mean.items(),key=lambda x:-x[1])
    top_ct    = sorted_ct[:max_slices]
    other_sum = sum(v for _,v in sorted_ct[max_slices:])
    if other_sum>0.005: top_ct.append(("other",other_sum))
    total = sum(v for _,v in top_ct)
    if total<1e-9: ax.axis("off"); return
    labels=[c for c,_ in top_ct]; sizes=[v/total for _,v in top_ct]
    clrs=[]; wprops=[]
    for lbl in labels:
        if not has_exch:
            clrs.append(_ct_color(lbl) if lbl!="other" else "#dddddd")
            wprops.append({"lw":0.4,"ec":"white"})
        elif lbl==dom_src:
            clrs.append(GROUP_COLORS[group]); wprops.append({"lw":2.5,"ec":"#111"})
        elif lbl==dom_snk:
            clrs.append("#ffffff"); wprops.append({"lw":2.5,"ec":GROUP_COLORS[group],"ls":"--"})
        elif lbl=="other":
            clrs.append("#dddddd"); wprops.append({"lw":0.4,"ec":"white"})
        else:
            clrs.append(_ct_color(lbl)); wprops.append({"lw":0.4,"ec":"white"})
    wedges,_=ax.pie(sizes,colors=clrs,wedgeprops={"width":0.52},
                    startangle=90,counterclock=False)
    for w,wp in zip(wedges,wprops):
        w.set_linewidth(wp["lw"]); w.set_edgecolor(wp["ec"])
        if wp.get("ls"): w.set_linestyle(wp["ls"])
    if has_exch:
        for w,lbl in zip(wedges,labels):
            if lbl in (dom_src,dom_snk):
                ang=(w.theta1+w.theta2)/2; rad=np.deg2rad(ang)
                ax.annotate(lbl[:10],xy=(0.82*np.cos(rad),0.82*np.sin(rad)),
                            fontsize=5.8,ha="center",va="center",
                            color="#111",fontweight="bold")
        ax.text(0,0.12,dom_src[:9],ha="center",va="center",
                fontsize=6.0,fontweight="bold",color=GROUP_COLORS[group])
        ax.text(0,-0.14,f"↓\n{dom_snk[:9]}",ha="center",va="center",
                fontsize=5.8,color="#555")
    else:
        ax.text(0,0,"no\nexchange",ha="center",va="center",
                fontsize=6.5,color="#888",fontstyle="italic")
    ttl = "Composition" if has_exch else "Composition\n(no exchange)"
    ax.set_title(ttl,fontsize=7.5,fontweight="bold",pad=2,color="#444")
    if has_exch:
        ax.legend(handles=[
            mpatches.Patch(fc=GROUP_COLORS[group],ec="#111",lw=1.5,
                           label=f"Src: {dom_src[:12]}"),
            mpatches.Patch(fc="white",ec=GROUP_COLORS[group],lw=1.5,
                           ls="--",label=f"Snk: {dom_snk[:12]}")],
            loc="lower center",bbox_to_anchor=(0.5,-0.26),
            fontsize=5.5,framealpha=0.85,ncol=1)


def _draw_topology_summary(ax, topo_by_group, groups):
    """Right panel: exchange topology summary with per-group count bars."""
    ax.set_facecolor("#f8f9fa")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    y=0.97
    ax.text(0.5,y,"Exchange topology",transform=ax.transAxes,
            ha="center",va="top",fontsize=8,fontweight="bold",color="#333")
    y-=0.07
    for g in groups:
        topo=topo_by_group.get(g,[])
        if not topo: continue
        ax.text(0.02,y,GROUP_LABELS[g],transform=ax.transAxes,
                ha="left",va="top",fontsize=7.5,fontweight="bold",
                color=GROUP_COLORS[g])
        y-=0.055
        n_grp=max(sum(1 for p in _PATIENT_GROUP if _PATIENT_GROUP[p]==g),1)
        for src,snk,cnt in topo:
            bw=min(cnt/n_grp,0.88)
            ax.barh(y,bw,left=0.04,height=0.045,
                    color=GROUP_COLORS[g],alpha=0.60,
                    transform=ax.transAxes,zorder=2)
            ax.text(0.06+bw,y+0.022,
                    f"{src[:8]}→{snk[:8]}  {cnt}/{n_grp}",
                    transform=ax.transAxes,ha="left",va="center",
                    fontsize=5.8,color="#222")
            y-=0.065
            if y<0.06: break
        y-=0.018
        if y<0.06: break
    ax.text(0.5,0.01,"Multiple rows = alt. optima",
            transform=ax.transAxes,ha="center",va="bottom",
            fontsize=5.5,color="#888",fontstyle="italic")


def _draw_detection_strip(ax, metabolite, cohort_df, patient_results, groups=None):
    """Row 1 right: detection circle strip."""
    if groups is None: groups=COHORT_GROUPS
    totals={g:sum(1 for p in patient_results if _patient_group(p)==g)
            for g in groups}
    sub=cohort_df[cohort_df["metabolite"]==metabolite]
    dets={g:int(sub[sub["group"]==g]["patient_id"].nunique()) for g in groups}
    ax.set_xlim(-0.5,len(groups)-0.5); ax.set_ylim(-0.55,0.55); ax.axis("off")
    for i,g in enumerate(groups):
        nd=dets.get(g,0); nt=totals.get(g,1); f=nd/max(nt,1)
        ax.add_patch(plt.Circle((i,0),0.38,color=GROUP_COLORS[g],
                                fill=False,lw=1.2,alpha=0.35,zorder=2))
        ax.add_patch(plt.Circle((i,0),0.32*f+0.04,color=GROUP_COLORS[g],
                                alpha=0.75+0.25*f,zorder=3))
        ax.text(i,0,f"{nd}/{nt}",ha="center",va="center",
                fontsize=8.5,fontweight="bold",
                color="white" if f>0.5 else GROUP_COLORS[g],zorder=4)
        ax.text(i,-0.50,GROUP_LABELS[g],ha="center",va="bottom",
                fontsize=8,color=GROUP_COLORS[g],fontweight="bold")
    ax.text(-0.48,0,"Detected:",ha="left",va="center",
            fontsize=7.5,color="#555",fontstyle="italic")


def _draw_clinical_strip(ax, metabolite, corr_df,
                          clinical_fields=None, rank_df=None):
    """Row 2: clinical correlation bars with selection reason badge."""
    if clinical_fields is None: clinical_fields=["eGFR","fibrosis"]
    field_labels={"eGFR":"eGFR","fibrosis":"Fibrosis %","age":"Age"}
    sel_reason=""
    if rank_df is not None and not rank_df.empty:
        sub=rank_df[rank_df["metabolite"]==metabolite]
        if not sub.empty:
            sel_reason=str(sub.iloc[0].get("selection_reason",""))
    n_fields=len(clinical_fields)
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-0.6,n_fields-0.4); ax.axis("off")
    if corr_df.empty or metabolite not in corr_df["metabolite"].values:
        ax.text(0,(n_fields-1)/2,"Clinical correlations: no data",
                ha="center",va="center",fontsize=8,color="#aaa"); return
    cr=corr_df[corr_df["metabolite"]==metabolite].iloc[0]
    for i,field in enumerate(reversed(clinical_fields)):
        rho=float(cr.get(f"rho_{field}",0.0))
        p  =float(cr.get(f"p_{field}",  1.0))
        col="#27ae60" if rho>=0 else "#c0392b"
        alp=0.85 if p<0.20 else 0.35
        for xr in(-0.5,0.5):
            ax.axvline(xr,color="#e0e0e0",lw=0.6,ls=":",zorder=0)
        ax.axvline(0,color="#aaa",lw=0.9,zorder=1)
        ax.barh(i,rho,color=col,alpha=alp,height=0.55,edgecolor="none",zorder=2)
        star=" *" if p<0.20 else ""
        ax.text(rho+(0.04 if rho>=0 else -0.04),i,
                f"ρ={rho:+.2f}{star}",va="center",
                ha="left" if rho>=0 else "right",
                fontsize=7.5,color=col,fontweight="bold",zorder=3)
        ax.text(-1.12,i,field_labels.get(field,field),
                va="center",ha="left",fontsize=7.5,color="#444")
    reason_colors={"clinical_eGFR":"#2980b9","clinical_fibrosis":"#8e44ad",
                   "clinical":"#2980b9","both":"#16a085","ckd_specificity":"#c0392b"}
    if sel_reason:
        rc=reason_colors.get(sel_reason,"#888")
        ax.text(1.12,n_fields/2-0.5,f"Selected:\n{sel_reason}",
                va="center",ha="right",fontsize=6.5,color=rc,fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=rc+"22",edgecolor=rc,lw=0.8))
    ax.text(-1.12,n_fields-0.05,"Clinical correlations (* p<0.20):",
            va="bottom",ha="left",fontsize=7,color="#555",fontstyle="italic")


# ===========================================================================
# Main figure builder — simple 5-column layout
# ===========================================================================

def _make_consensus_figure(
    metabolite, patient_results, cohort_df,
    corr_scores_df, corr_df, rank_df, groups,
    clinical_fields=None,
    fig_width=20.0, fig_height=12.0,
):
    """
    Simple 3-row layout (matches preferred simpler_version.png):

    Row 0: [violin | dot×3 | topology]
    Row 1: [donut×3 | detection strip]
    Row 2: [clinical strip]
    """
    if clinical_fields is None: clinical_fields=["eGFR","fibrosis"]

    group_data={g: _get_group_data(cohort_df,corr_scores_df,g,
                                    metabolite,patient_results)
                for g in groups}
    if not any(bool(group_data[g]["patients_with"]) for g in groups):
        return None

    topo_by_group={g:_get_topology_counts(cohort_df,g,metabolite,5)
                   for g in groups}
    group_totals={g:sum(1 for p in patient_results if _patient_group(p)==g)
                  for g in groups}

    sel_tag=""
    if not rank_df.empty and metabolite in rank_df["metabolite"].values:
        r=rank_df[rank_df["metabolite"]==metabolite].iloc[0].get("selection_reason","")
        sel_tag={"clinical_eGFR":"  [↑ eGFR correlation]",
                 "clinical_fibrosis":"  [↑ Fibrosis correlation]",
                 "clinical":"  [↑ clinical correlation]",
                 "both":"  [CKD + clinical]",
                 "ckd_specificity":""}.get(r,"")

    det_str="  |  ".join([
        f"{GROUP_LABELS[g]}: "
        f"{int(cohort_df[(cohort_df['metabolite']==metabolite)&(cohort_df['group']==g)]['patient_id'].nunique())}"
        f"/{group_totals.get(g,1)}"
        for g in groups])

    fig=plt.figure(figsize=(fig_width,fig_height))
    fig.patch.set_facecolor("white")

    # Row 0: violin(1) | dot×3(2-4) | topology(5)
    # Row 1: donut×3(1-3) | detection(4)
    # Row 2: clinical strip
    outer=gridspec.GridSpec(3,1,figure=fig,height_ratios=[6.0,2.0,1.0],
                            hspace=0.14,top=0.91,bottom=0.03,
                            left=0.04,right=0.98)
    top_gs=gridspec.GridSpecFromSubplotSpec(
        1,5,subplot_spec=outer[0],wspace=0.12,
        width_ratios=[1.4,2.0,2.0,2.0,1.2])
    mid_gs=gridspec.GridSpecFromSubplotSpec(
        1,4,subplot_spec=outer[1],wspace=0.10,
        width_ratios=[1,1,1,2.2])

    fig.suptitle(
        f"Metabolite Exchange: {metabolite}{sel_tag}\n"
        f"[Detection — {det_str}]",
        fontsize=13.5,fontweight="bold",y=0.98,color="#1a1a2e")

    # Panel A: violin
    _draw_score_violin(fig.add_subplot(top_gs[0]),group_data,groups,metabolite)

    # Panels B: dot-plots
    for ci,g in enumerate(groups,1):
        _draw_patient_dotplot(
            fig.add_subplot(top_gs[ci]),
            group_data.get(g,{}),g,cohort_df,metabolite,corr_scores_df)

    # Topology
    _draw_topology_summary(fig.add_subplot(top_gs[4]),topo_by_group,groups)

    # Middle row: donuts + detection
    for ci,g in enumerate(groups):
        _draw_composition_donut(
            fig.add_subplot(mid_gs[ci]),group_data.get(g,{}),g,metabolite)
    _draw_detection_strip(fig.add_subplot(mid_gs[3]),
                          metabolite,cohort_df,patient_results,groups)

    # Clinical strip
    _draw_clinical_strip(fig.add_subplot(outer[2]),
                         metabolite,corr_df,
                         clinical_fields=clinical_fields,rank_df=rank_df)
    return fig


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_cohort_consensus_plots(
    patient_results: Dict,
    out_dir: str = "cohort_output/plots_consensus_cohort",
    top_n: int = 20,
    groups: Optional[List[str]] = None,
    clinical_fields: Optional[List[str]] = None,
    clinical_weight: float = 0.40,
    ckd_weight: float = 0.60,
    top_clinical_k: int = 5,
    dpi: int = 300,
    verbose: bool = True,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Generate simple consensus summary figures → plots_consensus_cohort/

    Returns (saved_paths, selected_metabolites, rank_df)
    so the caller can pass the same metabolite list to merged_streamline_plots.
    """
    os.makedirs(out_dir,exist_ok=True)
    if groups is None:          groups=COHORT_GROUPS
    if clinical_fields is None: clinical_fields=["eGFR","fibrosis"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CONSENSUS SUMMARY PLOTS  (v7 — simple layout)")
        print(f"  Out: {out_dir}  |  Top N: {top_n}  |  clin_k: {top_clinical_k}")
        print(f"{'='*60}")

    cohort_df=build_cohort_metabolite_table(patient_results)
    if cohort_df.empty:
        if verbose: print("  WARNING: empty cohort table"); return [],[], pd.DataFrame()
    if verbose:
        print(f"  {cohort_df['metabolite'].nunique()} metabolites × "
              f"{cohort_df['patient_id'].nunique()} patients")

    corr_scores_df=compute_composition_corrected_scores(cohort_df,patient_results)
    sm=build_score_matrix_corrected(corr_scores_df)
    corr_df=pd.DataFrame()
    if not sm.empty:
        corr_df=compute_clinical_correlations(
            sm,patient_results,clinical_fields=clinical_fields)

    selected_mets,rank_df,clin_ext=select_target_metabolites(
        cohort_df,top_n=top_n,patient_results=patient_results,
        corr_scores_df=corr_scores_df,clinical_fields=clinical_fields,
        clinical_weight=clinical_weight,ckd_weight=ckd_weight,
        top_clinical_k=top_clinical_k,return_metadata=True)

    if not corr_df.empty and not rank_df.empty:
        corr_df=corr_df.merge(
            rank_df[["metabolite","selection_reason"]],
            on="metabolite",how="left")

    if verbose:
        print(f"  {len(selected_mets)} metabolites selected")
        for r,n in rank_df[rank_df["metabolite"].isin(selected_mets)
                           ]["selection_reason"].value_counts().items():
            print(f"    {r}: {n}")

    saved=[]
    for ri,met in enumerate(selected_mets,1):
        sel=rank_df[rank_df["metabolite"]==met]["selection_reason"].values
        if verbose:
            print(f"  [{ri:2d}/{len(selected_mets)}] {met} "
                  f"[{sel[0] if len(sel) else '?'}] …",end=" ",flush=True)
        try:
            fig=_make_consensus_figure(
                met,patient_results,cohort_df,
                corr_scores_df,corr_df,rank_df,groups,
                clinical_fields=clinical_fields)
            if fig is None:
                if verbose: print("SKIP (no data)"); continue
            fname=f"cohort_consensus_{ri:02d}_{met.replace('/','_')}.png"
            fpath=os.path.join(out_dir,fname)
            fig.savefig(fpath,dpi=dpi,bbox_inches="tight",facecolor="white")
            plt.close(fig)
            saved.append(fpath)
            if verbose: print(f"OK → {fname}")
        except Exception as e:
            if verbose:
                import traceback; print(f"ERROR: {e}"); traceback.print_exc()

    # Save metadata CSVs
    rank_df.to_csv(os.path.join(out_dir,"cohort_ranking.csv"),index=False)
    if not corr_df.empty:
        corr_df.to_csv(os.path.join(out_dir,"clinical_correlations_all.csv"),index=False)
    if corr_scores_df is not None and not corr_scores_df.empty:
        corr_scores_df.to_csv(
            os.path.join(out_dir,"composition_corrected_scores.csv"),index=False)
    if clin_ext:
        pd.DataFrame([{"field":f,"metabolite":m}
                      for f,mets in clin_ext.items()
                      for m in mets]).to_csv(
            os.path.join(out_dir,"clinical_extension_list.csv"),index=False)

    if verbose: print(f"\n  ✓ {len(saved)} figures → {out_dir}\n")
    return saved, selected_mets, rank_df
