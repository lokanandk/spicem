#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS - PIPELINE MODULE (HEALTHY-ONLY SAFE)
=====================================================================

Fixes:
1. compute_flux_balance_by_condition aggregates by CELL TYPE to match plotting.
2. H2O, H2O2, H, CO2, O2 are excluded from all analyses.
3. Single-condition runs (e.g. Sample_Healthy only) are fully supported.
4. Condition comparison is skipped gracefully when fewer than 2 conditions exist.
"""

import os
import time

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .core import (
    load_region_data,
    aggregate_fluxes,
    find_enriched_metabolites,
    compute_spatial_coupling_matrix,
    analyze_all_interactions,
    compute_potential_flow_field,
    permutation_test_vectorized,
    combine_permutation_results,
    compare_conditions,
)

RUN_COASSOCIATION = True
if RUN_COASSOCIATION:
    try:
        from .coassociation import run_coassociation
    except Exception:
        RUN_COASSOCIATION = False
        run_coassociation = None

EXCLUDED_METABOLITES = {"h2o", "h2o2", "h", "co2", "o2"}


def _should_exclude_metabolite(met: str) -> bool:
    return str(met).strip().lower() in EXCLUDED_METABOLITES


@dataclass
class AnalysisConfig:
    base_dir: str = ""
    out_dir: str = "analysis_output"

    # Healthy-only default
    conditions: List[str] = field(default_factory=lambda: ["Sample_Healthy"])
    regions: List[int] = field(default_factory=lambda: [1, 3, 4, 5])

    diffusion_coefficient: float = 0.1
    k_neighbors: int = 4

    grid_resolution: int = 80
    gaussian_sigma: float = 4.0

    permutation_iters: int = 500
    min_coupling_strength: float = 0.001

    top_n_plots: int = 20
    num_workers: int = 10

    grid_pattern: str = "communities_hexagonal_Region_{}.json"
    flux_folder: str = "exchange_fluxes"


@dataclass
class RegionResult:
    condition: str
    region_idx: int
    met_fluxes: Dict
    cell_met_pool: Dict
    neighbor_map: Dict
    interactions: Dict
    best_pairs: Dict
    enriched_df: pd.DataFrame
    perm_p: Dict
    coupling: Dict
    vector_fields: Dict
    meta_df: pd.DataFrame
    mmeta: pd.DataFrame
    stats: Dict


class AnalysisResults:
    def __init__(self, config: AnalysisConfig):
        self.config = config

        self.per_region_data: Dict[str, dict] = {}

        self.global_perm_p: Dict[str, Dict[Tuple[str, str], float]] = {}
        self.global_best_pairs: Dict[str, Tuple[str, str]] = {}

        self.met_scores: List[Tuple[str, float]] = []
        self.significant_mets: List[str] = []

        self.comparison_df: pd.DataFrame = pd.DataFrame()

        self.centrality_df: pd.DataFrame = pd.DataFrame()
        self.balance_df: pd.DataFrame = pd.DataFrame()
        self.condition_balance: Dict[str, pd.DataFrame] = {}

        self.network_graph = None
        self.coassoc: Dict[str, pd.DataFrame] = {}

    def get_vector_fields_for_condition(self, condition: str, metabolite: str):
        fields = []
        for key, data in self.per_region_data.items():
            if key.startswith(condition) and metabolite in data.get("vector_fields", {}):
                fields.append(data["vector_fields"][metabolite])
        return fields

    def get_metadata_for_condition(self, condition: str) -> pd.DataFrame:
        dfs = []
        for key, data in self.per_region_data.items():
            if key.startswith(condition):
                dfs.append(data.get("meta_df", pd.DataFrame()))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_metabolite_info(self, met: str) -> Tuple[Optional[str], Optional[str], float]:
        src, snk = self.global_best_pairs.get(met, (None, None))
        p = self.global_perm_p.get(met, {}).get((src, snk), 1.0) if src and snk else 1.0
        return src, snk, float(p)


def process_one_region_worker(task):
    config, condition, region_idx = task
    try:
        grids, meta_df, flux_dir = load_region_data(
            config.base_dir, condition, region_idx, config.grid_pattern, config.flux_folder
        )

        if not grids or meta_df.empty or not os.path.isdir(flux_dir):
            return condition, region_idx, None, "missing data"

        met_fluxes, cell_pool, neighbor_map, _nfiles, _nexchanges = aggregate_fluxes(
            grids, meta_df, flux_dir
        )
        if not met_fluxes:
            return condition, region_idx, None, "no fluxes"

        enriched_df, _enrichment_info = find_enriched_metabolites(met_fluxes)
        coupling = compute_spatial_coupling_matrix(meta_df, neighbor_map)
        interactions, best_pairs = analyze_all_interactions(
            met_fluxes, coupling, config.min_coupling_strength
        )
        perm_p = permutation_test_vectorized(
            met_fluxes, meta_df, neighbor_map, best_pairs, n_iter=config.permutation_iters
        )

        vector_fields = {}
        for met, _fluxes in met_fluxes.items():
            if _should_exclude_metabolite(met):
                continue
            try:
                src, snk, _score = best_pairs.get(met, (None, None, 0.0))
                if src and snk:
                    U, V, xi, yi = compute_potential_flow_field(
                        meta_df, src, snk,
                        grid_size=config.grid_resolution,
                        sigma=config.gaussian_sigma
                    )
                    vector_fields[met] = (U, V, xi, yi)
            except Exception:
                pass

        result = {
            "condition": condition,
            "region": region_idx,
            "met_fluxes": met_fluxes,
            "cell_met_pool": cell_pool,
            "neighbor_map": neighbor_map,
            "interactions": interactions,
            "best_pairs": best_pairs,
            "enriched_df": enriched_df,
            "perm_p": perm_p,
            "coupling": coupling,
            "vector_fields": vector_fields,
            "meta_df": meta_df,
            "mmeta": enriched_df,
            "stats": {
                "metabolites": len(met_fluxes),
                "best_pairs": len(best_pairs),
                "vector_fields": len(vector_fields),
            },
        }
        return condition, region_idx, result, None

    except Exception as e:
        return condition, region_idx, None, str(e)


def compute_flux_balance(per_region_met_fluxes: Dict[str, Dict]) -> pd.DataFrame:
    met_balance = {}
    for _key, met_fluxes in per_region_met_fluxes.items():
        for met, flux_dict in met_fluxes.items():
            if _should_exclude_metabolite(met):
                continue

            if met not in met_balance:
                met_balance[met] = {"produced": 0.0, "consumed": 0.0}

            for _cell_type, flux_list in flux_dict.get("secretion", {}).items():
                for val in flux_list:
                    met_balance[met]["produced"] += abs(float(val))

            for _cell_type, flux_list in flux_dict.get("uptake", {}).items():
                for val in flux_list:
                    met_balance[met]["consumed"] += abs(float(val))

    rows = []
    for met, bal in met_balance.items():
        net = bal["produced"] - bal["consumed"]
        rows.append({
            "Metabolite": met,
            "Produced": float(bal["produced"]),
            "Consumed": float(bal["consumed"]),
            "NetFlux": float(net),
        })

    return pd.DataFrame(rows).sort_values("NetFlux", ascending=False) if rows else pd.DataFrame()


def compute_flux_balance_by_condition(
    per_region_met_fluxes: Dict[str, Dict],
    conditions: List[str]
) -> Dict[str, pd.DataFrame]:
    condition_balance = {}

    for condition in conditions:
        ct_balance = defaultdict(lambda: {"produced": 0.0, "consumed": 0.0})

        for key, met_fluxes in per_region_met_fluxes.items():
            if not key.startswith(condition):
                continue

            for met, flux_dict in met_fluxes.items():
                if _should_exclude_metabolite(met):
                    continue

                for cell_type, flux_list in flux_dict.get("secretion", {}).items():
                    total_flux = sum(abs(float(val)) for val in flux_list)
                    ct_balance[cell_type]["produced"] += total_flux

                for cell_type, flux_list in flux_dict.get("uptake", {}).items():
                    total_flux = sum(abs(float(val)) for val in flux_list)
                    ct_balance[cell_type]["consumed"] += total_flux

        rows = []
        for ct, bal in ct_balance.items():
            prod = bal["produced"]
            cons = bal["consumed"]
            total = prod + cons
            ratio = prod / total if total > 1e-12 else 0.5
            rows.append({
                "CellType": ct,
                "Produced": prod,
                "Consumed": cons,
                "TotalFlux": total,
                "Balance_Ratio": ratio,
            })

        condition_balance[condition] = (
            pd.DataFrame(rows).sort_values("Balance_Ratio", ascending=False)
            if rows else pd.DataFrame()
        )

    return condition_balance


def compute_network_centrality(
    per_region_interactions: Dict[str, Dict]
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    G = nx.DiGraph()
    edge_metabolites = {}

    for _key, interactions in per_region_interactions.items():
        for met, pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
            for pair in pairs:
                src = pair.get("source", "")
                snk = pair.get("sink", "")
                score = float(pair.get("score", 0.0))
                if src and snk and src != snk:
                    if G.has_edge(src, snk):
                        G[src][snk]["weight"] += score
                    else:
                        G.add_edge(src, snk, weight=score)
                    edge_metabolites.setdefault((src, snk), set()).add(met)

    for (src, snk) in edge_metabolites:
        G[src][snk].update(
            metabolites=list(edge_metabolites[(src, snk)]),
            n_metabolites=len(edge_metabolites[(src, snk)]),
        )

    if len(G.nodes()) == 0:
        return pd.DataFrame(), G

    degree_cent = nx.degree_centrality(G)
    indeg = dict(G.in_degree(weight="weight"))
    outdeg = dict(G.out_degree(weight="weight"))
    try:
        between_cent = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        between_cent = {n: 0.0 for n in G.nodes()}

    rows = []
    for node in G.nodes():
        rows.append({
            "CellType": node,
            "DegreeCentrality": float(degree_cent.get(node, 0.0)),
            "WeightedInDegree": float(indeg.get(node, 0.0)),
            "WeightedOutDegree": float(outdeg.get(node, 0.0)),
            "BetweennessCentrality": float(between_cent.get(node, 0.0)),
            "TotalPartners": int(G.degree(node)),
        })

    return pd.DataFrame(rows).sort_values("BetweennessCentrality", ascending=False), G


def run_analysis_pipeline(config: AnalysisConfig, verbose: bool = True) -> AnalysisResults:
    results = AnalysisResults(config)

    if verbose:
        print("=" * 70)
        print("SPATIAL METABOLIC FLUX ANALYSIS")
        print(f"Using {config.num_workers} parallel workers")
        print(f"Excluding: {', '.join(sorted(EXCLUDED_METABOLITES))}")
        print("=" * 70)
        print("Phase A: Processing regions...")

    t0 = time.time()
    tasks = [(config, cond, r) for cond in config.conditions for r in config.regions]

    per_region_met_fluxes: Dict[str, dict] = {}
    per_region_interactions: Dict[str, dict] = {}
    per_region_best_pairs: Dict[str, dict] = {}
    per_region_perm_p: Dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=int(config.num_workers)) as executor:
        futures = [executor.submit(process_one_region_worker, task) for task in tasks]

        for future in as_completed(futures):
            condition, region_idx, result, error = future.result()

            if error:
                if verbose:
                    print(f"{condition} R{region_idx}: SKIP - {error}")
                continue

            if result is None:
                continue

            stats = result.get("stats", {})
            if verbose:
                print(
                    f"{condition} R{region_idx}: OK "
                    f"({stats.get('metabolites', 0)} mets, "
                    f"{stats.get('best_pairs', 0)} pairs, "
                    f"{stats.get('vector_fields', 0)} fields)"
                )

            key = f"{condition}_R{region_idx}"
            results.per_region_data[key] = result
            per_region_met_fluxes[key] = result["met_fluxes"]
            per_region_interactions[key] = result["interactions"]
            per_region_best_pairs[key] = result["best_pairs"]
            per_region_perm_p[key] = result["perm_p"]

    if verbose:
        print(f"Phase A complete in {time.time() - t0:.1f}s")
        print(f"Regions processed: {len(results.per_region_data)}")

    if not results.per_region_data:
        if verbose:
            print("ERROR: No regions processed!")
        return results

    if verbose:
        print("Phase B: Statistical analysis...")
    t1 = time.time()

    results.global_perm_p = combine_permutation_results(per_region_perm_p)
    results.centrality_df, results.network_graph = compute_network_centrality(per_region_interactions)
    results.balance_df = compute_flux_balance(per_region_met_fluxes)
    results.condition_balance = compute_flux_balance_by_condition(per_region_met_fluxes, config.conditions)

    if len(config.conditions) >= 2:
        results.comparison_df = compare_conditions(per_region_met_fluxes, config.conditions)
    else:
        results.comparison_df = pd.DataFrame()

    if verbose:
        print(f"Combined p-values for {len(results.global_perm_p)} metabolites")
        if not results.comparison_df.empty:
            print(f"Compared {len(results.comparison_df)} metabolites between conditions")
        else:
            print("Condition comparison skipped (fewer than 2 conditions)")
        print(f"Phase B complete in {time.time() - t1:.1f}s")

    if verbose:
        print("Phase C: Selecting metabolites...")

    all_best_pairs: Dict[str, List[Tuple[str, str, float]]] = {}
    for _key, bp in per_region_best_pairs.items():
        for met, (src, snk, score) in (bp or {}).items():
            if _should_exclude_metabolite(met):
                continue
            all_best_pairs.setdefault(met, [])
            all_best_pairs[met].append((src, snk, float(score)))

    for met, pairs in all_best_pairs.items():
        pair_counts = Counter([(s, k) for s, k, _ in pairs])
        best_pair = pair_counts.most_common(1)[0][0]
        results.global_best_pairs[met] = best_pair

    results.met_scores = []
    for met, pairs in all_best_pairs.items():
        if _should_exclude_metabolite(met):
            continue
        avg_score = float(np.mean([s for _src, _snk, s in pairs])) if pairs else 0.0
        results.met_scores.append((met, avg_score))

    results.met_scores.sort(key=lambda x: x[1], reverse=True)
    results.significant_mets = [m for m, _ in results.met_scores[: int(config.top_n_plots)]]

    if verbose:
        print(f"Total metabolites: {len(all_best_pairs)} (after exclusion)")
        print(f"Selected for plotting: {len(results.significant_mets)}")
        if results.significant_mets:
            print(f"Top 5: {results.significant_mets[:5]}")

    if RUN_COASSOCIATION:
        if verbose:
            print("Phase D: Co-association analysis + outputs...")
        co_dir = os.path.join(config.out_dir, "coassociation")
        try:
            results.coassoc = run_coassociation(results, out_dir=co_dir, conditions=config.conditions)
            if verbose:
                print(f"Co-association outputs saved to: {co_dir}")
        except Exception as e:
            if verbose:
                print(f"Co-association failed: {e}")

    if verbose:
        print(f"Analysis complete in {time.time() - t0:.1f}s")

    return results


def save_results_to_csv(results: AnalysisResults, out_dir: str, verbose: bool = False):
    if verbose:
        print("Saving CSV files...")

    rows = []
    for met, score in results.met_scores:
        src, snk, p_val = results.get_metabolite_info(met)
        rows.append({
            "Metabolite": met,
            "Score": float(score),
            "Source": src,
            "Sink": snk,
            "PValue": float(p_val),
        })
    df = pd.DataFrame(rows)
    outfile = os.path.join(out_dir, "metabolite_summary.csv")
    df.to_csv(outfile, index=False)
    if verbose:
        print(f"Saved metabolite_summary.csv ({len(df)} rows)")

    rows = []
    for key, data in results.per_region_data.items():
        interactions = data.get("interactions", {})
        for met, pairs in interactions.items():
            if _should_exclude_metabolite(met):
                continue
            for pair in pairs:
                rows.append({
                    "Region": key,
                    "Metabolite": met,
                    "Source": pair.get("source", ""),
                    "Sink": pair.get("sink", ""),
                    "Score": float(pair.get("score", 0.0)),
                    "Coupling": float(pair.get("coupling", 0.0)),
                })
    df = pd.DataFrame(rows)
    outfile = os.path.join(out_dir, "interaction_network.csv")
    df.to_csv(outfile, index=False)
    if verbose:
        print(f"Saved interaction_network.csv ({len(df)} rows)")

    if not results.centrality_df.empty:
        outfile = os.path.join(out_dir, "cell_type_hubs.csv")
        results.centrality_df.to_csv(outfile, index=False)
        if verbose:
            print("Saved cell_type_hubs.csv")

    if not results.balance_df.empty:
        outfile = os.path.join(out_dir, "flux_balance.csv")
        results.balance_df.to_csv(outfile, index=False)
        if verbose:
            print("Saved flux_balance.csv")

    if not results.comparison_df.empty:
        outfile = os.path.join(out_dir, "condition_comparison.csv")
        results.comparison_df.to_csv(outfile, index=False)
        if verbose:
            print("Saved condition_comparison.csv")


if __name__ == "__main__":
    print("Pipeline module loaded.")
    print(f"Will exclude these metabolites from analysis: {', '.join(sorted(EXCLUDED_METABOLITES))}")
