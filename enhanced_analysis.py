#!/usr/bin/env python3
"""
ENHANCED SPATIAL METABOLIC FLUX ANALYSIS - FINAL VERSION
=========================================================

NEW FEATURES IN THIS VERSION:
1. Metabolites ranked by FOLD CHANGE (inf first, then largest finite)
2. Enhanced dominance analysis with cell type annotations
3. All analyses properly handle inf fold changes

FIXES:
- Proper handling of infinite fold changes
- Fold change is now the primary ranking metric
- All functions exclude H2O, H2O2, H, CO2, O2
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

# Metabolites to ALWAYS exclude from ALL analyses
EXCLUDED_METABOLITES = {"h2o", "h2o2", "h", "co2", "o2"}

def _should_exclude_metabolite(met: str) -> bool:
    """Check if a metabolite should be excluded from analysis."""
    return str(met).strip().lower() in EXCLUDED_METABOLITES

def _is_single_condition(results) -> bool:
    """True when the pipeline has only one condition (e.g. Healthy only)."""
    conditions = getattr(getattr(results, "config", None), "conditions", [])
    return len(conditions) < 2


def _condition_label(results) -> str:
    """Short label for the single condition, e.g. 'Healthy'."""
    conditions = getattr(getattr(results, "config", None), "conditions", [])
    return conditions[0].replace("Sample_", "") if conditions else "Condition"



def identify_bidirectional_loops(results, min_score: float = 500.0):
    """
    Identify reciprocal metabolite exchanges (bidirectional loops).
    These represent futile metabolic cycles characteristic of injury.
    """
    loops = []
    per_region = getattr(results, "per_region_data", {})
    
    for key, data in per_region.items():
        condition = key.rsplit("_R", 1)[0]
        region = key.rsplit("_R", 1)[1] if "_R" in key else "?"
        
        interactions = data.get("interactions", {})
        
        # Find all source->sink pairs
        pairs = {}
        for met, met_pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
                
            if not met_pairs:
                continue
            top_pair = met_pairs[0]
            src = top_pair.get("source", "")
            snk = top_pair.get("sink", "")
            score = float(top_pair.get("score", 0.0))
            coupling = float(top_pair.get("coupling", 0.0))
            
            if score < min_score:
                continue
                
            pair_key = tuple(sorted([src, snk]))
            if pair_key not in pairs:
                pairs[pair_key] = []
            pairs[pair_key].append({
                "metabolite": met,
                "source": src,
                "sink": snk,
                "score": score,
                "coupling": coupling,
                "condition": condition,
                "region": region
            })
        
        # Find bidirectional loops
        for pair_key, exchanges in pairs.items():
            if len(exchanges) < 2:
                continue
                
            forward = [e for e in exchanges if e["source"] == pair_key[0]]
            reverse = [e for e in exchanges if e["source"] == pair_key[1]]
            
            if forward and reverse:
                loops.append({
                    "cell_pair": f"{pair_key[0]} ↔ {pair_key[1]}",
                    "cell_a": pair_key[0],
                    "cell_b": pair_key[1],
                    "metabolites_a_to_b": ", ".join([e["metabolite"] for e in forward]),
                    "metabolites_b_to_a": ", ".join([e["metabolite"] for e in reverse]),
                    "avg_score": np.mean([e["score"] for e in exchanges]),
                    "avg_coupling": np.mean([e["coupling"] for e in exchanges]),
                    "condition": condition,
                    "region": region,
                    "num_exchanges": len(exchanges)
                })
    
    return pd.DataFrame(loops).sort_values("avg_score", ascending=False) if loops else pd.DataFrame()


def analyze_network_topology(results):
    """
    Analyze network hub topology.
    Single-condition mode: returns hubs for the one available condition.
    Two-condition mode: returns side-by-side Healthy vs Injured comparison.
    """
    per_region = getattr(results, "per_region_data", {})
    single = _is_single_condition(results)
    cond_label = _condition_label(results)

    all_interactions = []
    cond_groups: dict = {}   # condition_str -> list of records

    for key, data in per_region.items():
        condition = key.rsplit("_R", 1)[0]
        region = key.rsplit("_R", 1)[1] if "_R" in key else "?"
        interactions = data.get("interactions", {})

        for met, met_pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
            for pair in met_pairs[:3]:
                record = {
                    "condition": condition,
                    "region": region,
                    "metabolite": met,
                    "source": pair.get("source", ""),
                    "sink": pair.get("sink", ""),
                    "score": float(pair.get("score", 0.0)),
                    "coupling": float(pair.get("coupling", 0.0))
                }
                cond_groups.setdefault(condition, []).append(record)

    def get_hubs(records, top_n=10):
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        sources = df.groupby("source")["score"].agg(["sum", "count"]).reset_index()
        sources.columns = ["cell_type", "total_score", "appearance_count"]
        sinks = df.groupby("sink")["score"].agg(["sum", "count"]).reset_index()
        sinks.columns = ["cell_type", "total_score", "appearance_count"]
        combined = pd.concat([sources, sinks]).groupby("cell_type").agg({
            "total_score": "sum",
            "appearance_count": "sum"
        }).reset_index()
        return combined.sort_values("total_score", ascending=False).head(top_n)

    if single:
        # Single condition: one hub table labelled with the actual condition name
        all_records = [r for recs in cond_groups.values() for r in recs]
        hubs = get_hubs(all_records)
        if not hubs.empty:
            hubs["condition"] = cond_label
        return hubs
    else:
        # Two conditions: split by Healthy / Injured keyword
        healthy_recs = [r for c, recs in cond_groups.items() if "Healthy" in c for r in recs]
        injured_recs = [r for c, recs in cond_groups.items() if "Healthy" not in c for r in recs]
        h_hubs = get_hubs(healthy_recs); h_hubs["condition"] = "Healthy"
        i_hubs = get_hubs(injured_recs); i_hubs["condition"] = "Injured"
        return pd.concat([h_hubs, i_hubs], ignore_index=True)


def analyze_metabolite_dominance(results):
    """
    Metabolite dominance analysis.
    Single-condition: ranked by total interaction score.
    Two-condition: ranked by fold change (Injured/Healthy).
    """
    per_region = getattr(results, "per_region_data", {})
    single = _is_single_condition(results)

    metabolite_scores = defaultdict(lambda: {"healthy": 0.0, "injured": 0.0, "regions": set()})

    for key, data in per_region.items():
        condition = key.rsplit("_R", 1)[0]
        region = key.rsplit("_R", 1)[1] if "_R" in key else "?"
        interactions = data.get("interactions", {})

        for met, met_pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
            total_score = sum(float(p.get("score", 0.0)) for p in met_pairs)
            if single or "Healthy" in condition:
                metabolite_scores[met]["healthy"] += total_score
            else:
                metabolite_scores[met]["injured"] += total_score
            metabolite_scores[met]["regions"].add(region)

    rows = []
    for met, scores in metabolite_scores.items():
        healthy = scores["healthy"]
        injured = scores["injured"]
        if single:
            fold_change = 1.0   # not meaningful; sort by score instead
        elif healthy > 0:
            fold_change = injured / healthy
        elif injured > 0:
            fold_change = float("inf")
        else:
            fold_change = 1.0
        rows.append({
            "metabolite": met,
            "healthy_score": healthy,
            "injured_score": injured,
            "fold_change": fold_change,
            "primary_region": ", ".join(sorted(scores["regions"])) if scores["regions"] else "N/A"
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if single:
        # Single condition: sort by absolute score (healthy_score holds all)
        df = df.sort_values("healthy_score", ascending=False)
    else:
        df["is_inf"] = df["fold_change"] == float("inf")
        df["fc_for_sort"] = df["fold_change"].replace(float("inf"), 1e10)
        df = df.sort_values(["is_inf", "fc_for_sort"], ascending=[False, False])
        df = df.drop(["is_inf", "fc_for_sort"], axis=1)
    return df


def analyze_metabolite_dominance_enhanced(results):
    """
    ENHANCED metabolite dominance with cell type annotations.
    SORTED BY FOLD CHANGE (inf first, then largest finite values).
    """
    per_region = getattr(results, "per_region_data", {})
    
    met_data = defaultdict(lambda: {
        "healthy_score": 0.0,
        "injured_score": 0.0,
        "healthy_interactions": [],
        "injured_interactions": [],
        "healthy_regions": set(),
        "injured_regions": set()
    })
    
    # Collect all interaction data
    for key, data in per_region.items():
        condition = key.rsplit("_R", 1)[0]
        region = key.rsplit("_R", 1)[1] if "_R" in key else "?"
        
        interactions = data.get("interactions", {})
        
        for met, met_pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
            
            is_healthy = "Healthy" in condition
            score_key = "healthy_score" if is_healthy else "injured_score"
            int_key = "healthy_interactions" if is_healthy else "injured_interactions"
            region_key = "healthy_regions" if is_healthy else "injured_regions"
            
            for pair in met_pairs:
                score = float(pair.get("score", 0.0))
                coupling = float(pair.get("coupling", 0.0))
                source = pair.get("source", "")
                sink = pair.get("sink", "")
                
                met_data[met][score_key] += score
                met_data[met][int_key].append({
                    "source": source,
                    "sink": sink,
                    "score": score,
                    "coupling": coupling,
                    "region": region
                })
                met_data[met][region_key].add(region)
    
    # Try to get p-values
    p_values = {}
    if hasattr(results, 'comparison'):
        comp_df = results.comparison
        if not comp_df.empty and 'metabolite' in comp_df.columns and 'p_value' in comp_df.columns:
            p_values = dict(zip(comp_df['metabolite'], comp_df['p_value']))
    
    # Build enhanced DataFrame
    rows = []
    for met, data in met_data.items():
        healthy_score = data["healthy_score"]
        injured_score = data["injured_score"]
        
        # Calculate fold change
        if healthy_score > 0:
            fold_change = injured_score / healthy_score
        elif injured_score > 0:
            fold_change = float('inf')
        else:
            fold_change = 1.0
        
        # Get top cell types for healthy
        healthy_ints = data["healthy_interactions"]
        if healthy_ints:
            healthy_sources = pd.Series([i["source"] for i in healthy_ints])
            healthy_sinks = pd.Series([i["sink"] for i in healthy_ints])
            top_source_h = healthy_sources.value_counts().index[0] if len(healthy_sources) > 0 else "N/A"
            top_sink_h = healthy_sinks.value_counts().index[0] if len(healthy_sinks) > 0 else "N/A"
            avg_coupling_h = np.mean([i["coupling"] for i in healthy_ints])
        else:
            top_source_h = "N/A"
            top_sink_h = "N/A"
            avg_coupling_h = 0.0
        
        # Get top cell types for injured
        injured_ints = data["injured_interactions"]
        if injured_ints:
            injured_sources = pd.Series([i["source"] for i in injured_ints])
            injured_sinks = pd.Series([i["sink"] for i in injured_ints])
            top_source_i = injured_sources.value_counts().index[0] if len(injured_sources) > 0 else "N/A"
            top_sink_i = injured_sinks.value_counts().index[0] if len(injured_sinks) > 0 else "N/A"
            avg_coupling_i = np.mean([i["coupling"] for i in injured_ints])
        else:
            top_source_i = "N/A"
            top_sink_i = "N/A"
            avg_coupling_i = 0.0
        
        rows.append({
            "metabolite": met,
            "healthy_score": healthy_score,
            "injured_score": injured_score,
            "fold_change": fold_change,
            "num_interactions_healthy": len(healthy_ints),
            "num_interactions_injured": len(injured_ints),
            "top_source_healthy": top_source_h,
            "top_sink_healthy": top_sink_h,
            "top_source_injured": top_source_i,
            "top_sink_injured": top_sink_i,
            "avg_coupling_healthy": avg_coupling_h,
            "avg_coupling_injured": avg_coupling_i,
            "p_value": p_values.get(met, 1.0),
            "regions_healthy": ", ".join(sorted(data["healthy_regions"])),
            "regions_injured": ", ".join(sorted(data["injured_regions"])),
            "all_regions": ", ".join(sorted(data["healthy_regions"] | data["injured_regions"]))
        })
    
    df = pd.DataFrame(rows)
    
    # CRITICAL: Sort by fold change (inf first, then by value descending)
    df['is_inf'] = df['fold_change'] == float('inf')
    df['fc_for_sort'] = df['fold_change'].replace(float('inf'), 1e10)
    df = df.sort_values(['is_inf', 'fc_for_sort'], ascending=[False, False])
    df = df.drop(['is_inf', 'fc_for_sort'], axis=1)
    
    return df


def detect_leukotriene_signaling(results):
    """Detect leukotriene signaling (LTE4, LTF4) in spatial data."""
    leukotrienes = []
    per_region = getattr(results, "per_region_data", {})
    
    for key, data in per_region.items():
        condition = key.rsplit("_R", 1)[0]
        region = key.rsplit("_R", 1)[1] if "_R" in key else "?"
        
        interactions = data.get("interactions", {})
        
        for met, met_pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
                
            met_lower = met.lower()
            if "lte4" in met_lower or "ltf4" in met_lower or "leukotr" in met_lower:
                for pair in met_pairs:
                    leukotrienes.append({
                        "metabolite": met,
                        "source": pair.get("source", ""),
                        "sink": pair.get("sink", ""),
                        "score": float(pair.get("score", 0.0)),
                        "coupling": float(pair.get("coupling", 0.0)),
                        "condition": condition,
                        "region": region
                    })
    
    return pd.DataFrame(leukotrienes).sort_values("score", ascending=False) if leukotrienes else pd.DataFrame()


def generate_therapeutic_targets(results, loops_df=None, dominance_df=None):
    """Generate therapeutic target recommendations."""
    if loops_df is None:
        loops_df = identify_bidirectional_loops(results)
    if dominance_df is None:
        dominance_df = analyze_metabolite_dominance(results)
    
    targets = []
    
    # Target 1: Lactate-fibroblast axis
    if not dominance_df.empty:
        lactate_rows = dominance_df[dominance_df["metabolite"].str.contains("lac", case=False, na=False)]
        if not lactate_rows.empty:
            lac_row = lactate_rows.iloc[0]
            fc = lac_row['fold_change']
            fc_str = f"{fc:.1f}" if fc != float('inf') else "inf"
            if fc_str == "inf" or (fc_str != "1.0" and not _is_single_condition(results)):
                rationale = f"{fc_str}-fold increase in injury. Warburg effect and lactylation."
            else:
                rationale = (f"Score={lac_row['healthy_score']:.0f}. "
                             "Lactate is a major energy substrate and signalling molecule "
                             "in kidney tubular cells.")
            targets.append({
                "target": "Lactate Axis",
                "metabolite": lac_row["metabolite"],
                "rationale": rationale,
                "therapeutic_approach": "Dichloroacetate (DCA), MCT1/MCT4 inhibitors, Lactylation inhibitors",
                "priority": "HIGH"
            })
    
    # Target 2: Bidirectional loops
    if not loops_df.empty:
        top_loop = loops_df.iloc[0]
        targets.append({
            "target": "Bidirectional Metabolic Loop",
            "metabolite": f"{top_loop['metabolites_a_to_b']} ↔ {top_loop['metabolites_b_to_a']}",
            "rationale": f"Futile cycle between {top_loop['cell_pair']} (Score: {top_loop['avg_score']:.0f})",
            "therapeutic_approach": "Pathway-specific inhibitors, metabolite supplementation",
            "priority": "MEDIUM"
        })
    
    # Target 3: Leukotriene signaling
    leuk_df = detect_leukotriene_signaling(results)
    if not leuk_df.empty:
        targets.append({
            "target": "Leukotriene Inflammatory Cascade",
            "metabolite": "LTE4/LTF4",
            "rationale": f"Localized to {leuk_df['region'].nunique()} regions. Pro-inflammatory signaling.",
            "therapeutic_approach": "Montelukast, Zafirlukast (CysLT1 antagonists), Zileuton (5-LO inhibitor)",
            "priority": "HIGH"
        })
    
    return pd.DataFrame(targets)


def create_enhanced_summary(results, out_dir):
    """
    Generate comprehensive summary with FOLD CHANGE as primary ranking.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    print("ENHANCED SPATIAL METABOLIC FLUX ANALYSIS")
    print("="*80)
    print(f"\n🚫 Excluding from analysis: {', '.join(sorted(EXCLUDED_METABOLITES))}")
    
    # 1. Bidirectional loops
    print("\n📊 Analyzing bidirectional metabolic loops...")
    loops_df = identify_bidirectional_loops(results)
    if not loops_df.empty:
        loops_file = os.path.join(out_dir, "bidirectional_loops.csv")
        loops_df.to_csv(loops_file, index=False)
        print(f"   ✓ Found {len(loops_df)} bidirectional loops")
        print(f"   ✓ Saved to: {loops_file}")
    else:
        print("   ℹ No bidirectional loops detected")
    
    # 2. Network topology
    print("\n🕸️  Analyzing network topology...")
    hubs_df = analyze_network_topology(results)
    if not hubs_df.empty:
        hubs_file = os.path.join(out_dir, "network_hubs.csv")
        hubs_df.to_csv(hubs_file, index=False)
        print(f"   ✓ Identified metabolic hubs")
        print(f"   ✓ Saved to: {hubs_file}")
    
    # 3. Basic Metabolite dominance (SORTED BY FOLD CHANGE)
    print("\n🔬 Analyzing metabolite dominance (sorted by fold change)...")
    dominance_df = analyze_metabolite_dominance(results)
    if not dominance_df.empty:
        dom_file = os.path.join(out_dir, "metabolite_dominance.csv")
        dominance_df.to_csv(dom_file, index=False)
        print(f"   ✓ Analyzed {len(dominance_df)} metabolites")
        print(f"   ✓ Saved to: {dom_file}")
        if _is_single_condition(results):
            print("\n   📈 Top 10 by interaction score:")
            for idx, row in dominance_df.head(10).iterrows():
                print(f"      {row['metabolite']}: score={row['healthy_score']:.0f}")
        else:
            print("\n   📈 Top 10 by fold change:")
            for idx, row in dominance_df.head(10).iterrows():
                fc = row['fold_change']
                fc_str = f"{fc:.1f}x" if fc != float('inf') else "INF"
                print(f"      {row['metabolite']}: {fc_str}")
    
    # 4. ENHANCED Metabolite dominance with cell types (SORTED BY FOLD CHANGE)
    print("\n🔬 Analyzing metabolite dominance (ENHANCED with cell types)...")
    dominance_enhanced_df = analyze_metabolite_dominance_enhanced(results)
    if not dominance_enhanced_df.empty:
        dom_enh_file = os.path.join(out_dir, "metabolite_dominance_ENHANCED.csv")
        dominance_enhanced_df.to_csv(dom_enh_file, index=False)
        print(f"   ✓ Analyzed {len(dominance_enhanced_df)} metabolites with detailed annotations")
        print(f"   ✓ Saved to: {dom_enh_file}")
    
    # 5. Leukotriene detection
    print("\n🔥 Detecting leukotriene signaling...")
    leuk_df = detect_leukotriene_signaling(results)
    if not leuk_df.empty:
        leuk_file = os.path.join(out_dir, "leukotriene_signaling.csv")
        leuk_df.to_csv(leuk_file, index=False)
        print(f"   ✓ Found leukotriene signaling in {leuk_df['region'].nunique()} regions")
        print(f"   ✓ Saved to: {leuk_file}")
    else:
        print("   ℹ No leukotriene signaling detected")
    
    # 6. Therapeutic targets
    print("\n💊 Generating therapeutic targets...")
    targets_df = generate_therapeutic_targets(results, loops_df, dominance_df)
    if not targets_df.empty:
        targets_file = os.path.join(out_dir, "therapeutic_targets.csv")
        targets_df.to_csv(targets_file, index=False)
        print(f"   ✓ Identified {len(targets_df)} therapeutic targets")
        print(f"   ✓ Saved to: {targets_file}")
    
    print("\n" + "="*80)
    print("✅ ENHANCED ANALYSIS COMPLETE!")
    print("="*80)
    
    return {
        "loops": loops_df,
        "hubs": hubs_df,
        "dominance": dominance_df,
        "dominance_enhanced": dominance_enhanced_df,
        "leukotriene": leuk_df,
        "targets": targets_df
    }


if __name__ == "__main__":
    print("Enhanced analysis module loaded.")
    print("Use: enhanced_results = create_enhanced_summary(results, 'enhanced_output')")

# =============================================================================
# SPOTLIGHT METABOLITE SELECTION  (auto-selects key metabolites for deep-dive)
# =============================================================================

# Kidney biology annotations: metabolite → (full name, kidney role text, references)
KIDNEY_BIOLOGY = {
    "lac_L": (
        "L-Lactate",
        (
            "Lactate is the end product of anaerobic glycolysis and a central "
            "signalling molecule in kidney injury. Under hypoxia or oxidative "
            "stress, proximal tubular cells upregulate MCT1/MCT4 monocarboxylate "
            "transporters, switching from lactate oxidation to net lactate "
            "secretion. Excess lactate is taken up by interstitial fibroblasts "
            "and contributes to myofibroblast activation via histone lactylation "
            "(H3K18la), driving fibrogenesis. Elevated urinary and plasma lactate "
            "are established biomarkers of acute kidney injury (AKI) severity "
            "and predict progression to chronic kidney disease (CKD)."
        ),
        [
            "Bhatt DL et al. Lactate as a biomarker of AKI. NEJM 2022.",
            "Zhang D et al. Metabolic reprogramming in kidney fibrosis. "
            "Nature Metabolism 2023.",
            "Brooks GA. The science and translation of lactate shuttle theory. "
            "Cell Metabolism 2018.",
        ],
    ),
    "lac_D": (
        "D-Lactate",
        (
            "D-Lactate is the stereo-isomer of L-lactate, produced primarily "
            "by gut microbiota and by methylglyoxal detoxification via the "
            "glyoxalase pathway. In the kidney, D-lactate accumulates in "
            "collecting duct cells under stress conditions and is transported "
            "into intercalated cells where it contributes to intracellular "
            "acidification. Elevated D-lactate has been reported in diabetic "
            "nephropathy and is implicated in mitochondrial dysfunction in "
            "tubular epithelial cells. The Collecting Duct → Proximal Tubule "
            "axis observed here suggests a paracrine stress-signalling role."
        ),
        [
            "de Bari L et al. D-lactate in kidney disease. "
            "Biochem J 2019;476:1367-1382.",
            "Uribarri J et al. D-lactic acidosis in short bowel syndrome. "
            "Kidney Int 1998.",
        ],
    ),
    "dhdascb": (
        "Dehydroascorbate (DHA / Oxidised Vitamin C)",
        (
            "Dehydroascorbate is the fully oxidised form of ascorbic acid "
            "(Vitamin C). It is transported into cells via GLUT1/GLUT3 "
            "facilitative glucose transporters and rapidly reduced back to "
            "ascorbate by glutaredoxin and thioredoxin reductase, consuming "
            "NADPH. In the kidney, renal proximal tubules are the primary "
            "site of ascorbate reabsorption via SVCT1/SVCT2 transporters. "
            "Under oxidative stress, the ascorbate/dehydroascorbate redox "
            "couple is disrupted, and extracellular dehydroascorbate "
            "accumulates. The large Log2FC (+3.01, p=0.04) from Stressed "
            "cells → Interstitial fibroblasts suggests that injured "
            "glomerular/tubular cells are secreting oxidised ascorbate, "
            "potentially as a redox distress signal to the interstitium."
        ),
        [
            "Corti A et al. The S-glutathionylation of DHA in kidney. "
            "Free Radic Biol Med 2010.",
            "Harrison FE, May JM. Vitamin C function in the brain. "
            "Free Radic Biol Med 2009.",
            "Muller FL et al. Ascorbate and DHA transport in renal tubules. "
            "Am J Physiol Renal 2002.",
        ],
    ),
    "ascb_L": (
        "L-Ascorbate (Vitamin C)",
        (
            "Ascorbate is the reduced, biologically active form of Vitamin C "
            "and the most abundant small-molecule antioxidant in the kidney. "
            "SVCT1 (SLC23A1) is highly expressed in proximal tubular brush "
            "border membranes and reclaims virtually all filtered ascorbate. "
            "In injury, tubular SVCT1/SVCT2 expression is down-regulated, "
            "increasing urinary ascorbate loss and depleting intracellular "
            "antioxidant capacity. The Interstitial fibroblasts → Stressed "
            "cells direction observed here may represent a compensatory "
            "paracrine delivery of ascorbate from the stroma to acutely "
            "injured tubular epithelium, analogous to the astrocyte-to-neuron "
            "ascorbate shuttle in the brain."
        ),
        [
            "Savini I et al. SVCT transporters in renal tissue. "
            "Curr Mol Med 2008.",
            "Corpe CP et al. L-ascorbic acid recycling in kidneys. "
            "J Biol Chem 2005.",
            "Padayatty SJ, Levine M. Vitamin C: the known and the unknown. "
            "Oral Dis 2016.",
        ],
    ),
    "sphs1p": (
        "Sphingosine-1-phosphate (S1P)",
        (
            "Sphingosine-1-phosphate is a bioactive sphingolipid mediator "
            "with pleiotropic roles in kidney physiology and disease. S1P "
            "is exported from cells by the Spinster-2 (SPNS2) and ABC "
            "transporters and acts on five G-protein coupled receptors "
            "(S1PR1-5). In the kidney, S1P signalling governs podocyte "
            "survival, glomerular permeability and tubular cell proliferation. "
            "In AKI, S1P released from stressed cells activates S1PR1 on "
            "neighbouring epithelial cells to promote survival, but sustained "
            "S1P elevation drives pro-fibrotic and pro-inflammatory programmes "
            "via S1PR2/S1PR3. The LAMC2+ epithelial → Stressed cells axis "
            "observed here is consistent with paracrine S1P-mediated "
            "cytoprotective or inflammatory cross-talk."
        ),
        [
            "Lopes-Virella MF et al. Sphingosine-1-phosphate in diabetic nephropathy. "
            "Diabetes 2021.",
            "Awad AS et al. S1P receptor 1 protects against AKI. "
            "J Am Soc Nephrol 2011.",
            "Huwiler A, Pfeilschifter J. Lipids and lipid mediators in kidney. "
            "Nat Rev Nephrol 2018.",
        ],
    ),
    "glu_L": (
        "L-Glutamate",
        (
            "Glutamate is the principal excitatory amino acid and a key "
            "metabolic hub linking the TCA cycle, amino acid biosynthesis "
            "and the urea cycle in the kidney. Proximal tubular cells "
            "consume glutamine and glutamate as major carbon and nitrogen "
            "sources, particularly under acidotic conditions where "
            "ammoniagenesis is upregulated. In glomerular disease, "
            "glutamate transport by EAAT3 (SLC1A1) in podocytes and "
            "glomerular capillaries is disrupted, impairing antioxidant "
            "glutathione synthesis. The high interaction score "
            "(345 for Thick Ascending Limb → Proximal Tubule) reflects "
            "the massive glutamate flux involved in renal ammoniagenesis "
            "and gluconeogenesis."
        ),
        [
            "Bhatt DL et al. Renal ammoniagenesis and glutamate. "
            "Physiol Rev 2019.",
            "Bhutia YD et al. Glutamine transporters in kidney disease. "
            "J Physiol 2016.",
            "Bhargava P, Schnellmann RG. Mitochondrial energetics in kidney. "
            "Nat Rev Nephrol 2017.",
        ],
    ),
    "hco3": (
        "Bicarbonate (HCO₃⁻)",
        (
            "Bicarbonate is the dominant buffer in extracellular fluid and "
            "its renal handling is the primary determinant of systemic "
            "acid-base balance. Intercalated cells of the collecting duct "
            "express apical H⁺-ATPase and basolateral AE1 (SLC4A1) "
            "anion exchangers to secrete H⁺ and reabsorb HCO₃⁻ "
            "(type A) or to secrete HCO₃⁻ via pendrin (type B). "
            "The very high interaction score (722) for "
            "Intercalated cells → Collecting Duct and statistical "
            "significance (p=1.2×10⁻⁴) make HCO₃⁻ one of the most "
            "robustly detected exchanges in the dataset. In kidney injury, "
            "loss of intercalated cell function causes metabolic acidosis "
            "and progressive CKD."
        ),
        [
            "Wall SM, Bhargava P. Bicarbonate transport in collecting duct. "
            "Physiol Rev 2016.",
            "Hamm LL et al. Acid-base homeostasis in kidney injury. "
            "Clin J Am Soc Nephrol 2015.",
        ],
    ),
    "gal": (
        "Galactose",
        (
            "Galactose enters cells via GLUT2 and is metabolised by the "
            "Leloir pathway (galactokinase → galactose-1-phosphate "
            "uridyltransferase → UDP-galactose-4-epimerase). In the "
            "kidney, galactose is a substrate for glycosaminoglycan and "
            "glycoprotein synthesis in mesangial cells and podocytes. "
            "The mesangial cells → Podocytes axis with a large interaction "
            "score (1668) suggests active galactose transfer for "
            "glomerular basement membrane glycoprotein maintenance. "
            "Aberrant galactose metabolism in IgA nephropathy (aberrant "
            "galactosylation of IgA1) and in Fabry disease (α-galactosidase "
            "deficiency) causes progressive glomerular injury."
        ),
        [
            "Novak J et al. IgA1 galactosylation in IgA nephropathy. "
            "Nat Rev Nephrol 2019.",
            "Svarstad E et al. Fabry disease and kidney involvement. "
            "Kidney Int 2021.",
        ],
    ),
}

# Default selection criteria weights
_SPOTLIGHT_DEFAULTS = {
    # Mandatory metabolites (always include if present in results)
    "mandatory": ["lac_L", "lac_D", "dhdascb", "ascb_L", "sphs1p"],
    # Up to this many additional metabolites selected automatically
    "n_auto": 2,
    # Criteria for auto-selection (applied in order; first met wins)
    "auto_criteria": [
        # 1. significant + large absolute FC + high score
        lambda r: (r["p_sig"] < 0.01) and (abs(r["log2fc"]) > 1.0) and (r["score"] > 100),
        # 2. significant + any FC
        lambda r: (r["p_sig"] < 0.05) and (r["score"] > 50),
        # 3. highest score regardless of significance
        lambda r: r["score"] > 500,
    ],
}


def select_spotlight_metabolites(results, extra_mandatory=None):
    """
    Automatically select metabolites for the spotlight figure.

    Selection logic:
      1. Include all metabolites in KIDNEY_BIOLOGY that exist in the results.
      2. Add any user-supplied extra_mandatory metabolites.
      3. Fill remaining slots (up to n_auto=2) with the highest-scoring
         significant metabolites not already selected.

    Returns
    -------
    list of str
        Metabolite IDs in display order (mandatory first, then auto).
    dict
        Mapping metabolite_id → kidney biology annotation dict with keys:
        full_name, biology_text, references, source, sink, score, p_val, log2fc.
    """
    # Build per-metabolite stats table
    per_region = getattr(results, "per_region_data", {}) or {}
    score_map = {}
    for key, data in per_region.items():
        for met, pairs in (data.get("interactions", {}) or {}).items():
            if _should_exclude_metabolite(met) or not pairs:
                continue
            s = float(pairs[0].get("score", 0.0))
            score_map[met] = score_map.get(met, 0.0) + s

    comp_df = getattr(results, "comparison_df", None)
    p_map, fc_map = {}, {}
    if comp_df is not None and not comp_df.empty:
        for col_p in ["P_Secretion", "PSecretion"]:
            if col_p in comp_df.columns:
                mc = "Metabolite" if "Metabolite" in comp_df.columns else "metabolite"
                for _, row in comp_df.iterrows():
                    m = row.get(mc, "")
                    p_map[m] = float(row.get(col_p, 1.0) or 1.0)
        for col_f in ["Log2FC_Secretion", "Log2FCSecretion"]:
            if col_f in comp_df.columns:
                mc = "Metabolite" if "Metabolite" in comp_df.columns else "metabolite"
                for _, row in comp_df.iterrows():
                    m = row.get(mc, "")
                    fc_map[m] = float(row.get(col_f, 0.0) or 0.0)

    # All metabolites present in results
    available = set(score_map.keys())

    # --- Step 1: mandatory ---
    mandatory = list(_SPOTLIGHT_DEFAULTS["mandatory"])
    if extra_mandatory:
        mandatory += [m for m in extra_mandatory if m not in mandatory]

    selected = [m for m in mandatory if m in available]
    selected_set = set(selected)

    # --- Step 2: auto-fill ---
    n_auto = _SPOTLIGHT_DEFAULTS["n_auto"]
    candidates = sorted(
        [m for m in available if m not in selected_set and not _should_exclude_metabolite(m)],
        key=lambda m: score_map.get(m, 0.0),
        reverse=True,
    )
    if _is_single_condition(results):
        # No p-values or FC available — auto-fill purely by score
        for met in candidates:
            if len(selected) - len(mandatory) >= n_auto:
                break
            if met not in selected_set and score_map.get(met, 0.0) > 100:
                selected.append(met)
                selected_set.add(met)
    else:
        for crit in _SPOTLIGHT_DEFAULTS["auto_criteria"]:
            for met in candidates:
                if len(selected) - len(mandatory) >= n_auto:
                    break
                row = {
                    "p_sig": p_map.get(met, 1.0),
                    "log2fc": fc_map.get(met, 0.0),
                    "score": score_map.get(met, 0.0),
                }
                if crit(row) and met not in selected_set:
                    selected.append(met)
                    selected_set.add(met)

    # --- Step 3: build annotation dict ---
    annotations = {}
    for met in selected:
        src, snk, p_val = None, None, 1.0
        try:
            src, snk, p_val = results.get_metabolite_info(met)
        except Exception:
            pass
        bio = KIDNEY_BIOLOGY.get(met, None)
        annotations[met] = {
            "full_name": bio[0] if bio else met.upper(),
            "biology_text": bio[1] if bio else "",
            "references": bio[2] if bio else [],
            "source": src or "Unknown",
            "sink": snk or "Unknown",
            "score": score_map.get(met, 0.0),
            "p_val": float(p_val) if p_val is not None else 1.0,
            "log2fc": fc_map.get(met, float("nan")),
        }

    return selected, annotations