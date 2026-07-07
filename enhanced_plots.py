#!/usr/bin/env python3
"""
ENHANCED PLOTTING FOR KEY INSIGHTS
===================================

Creates publication-ready visualizations that make insights obvious:
1. Bidirectional loop network diagrams
2. Healthy vs Injured network comparison
3. Metabolite dominance heatmaps
4. Leukotriene spatial localization maps
5. Therapeutic target summary figure

FIXED: generate_all_enhanced_plots signature to accept results object directly
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_bidirectional_loops(loops_df, out_dir, cfg=None):
    """
    Visualize bidirectional metabolic loops as a network diagram.
    """
    if loops_df.empty:
        return None
    
    # Take top 6 loops
    top_loops = loops_df.head(6)
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle("Bidirectional Metabolic Loops (Futile Cycles in Injury)",
                 fontsize=20, fontweight="bold", y=0.98)
    
    for idx, (i, loop) in enumerate(top_loops.iterrows()):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Draw the loop
        cell_a = loop["cell_a"]
        cell_b = loop["cell_b"]
        mets_ab = loop["metabolites_a_to_b"]
        mets_ba = loop["metabolites_b_to_a"]
        score = loop["avg_score"]
        coupling = loop["avg_coupling"]
        
        # Circular layout
        theta_a = np.pi / 4
        theta_b = 3 * np.pi / 4
        
        x_a, y_a = np.cos(theta_a), np.sin(theta_a)
        x_b, y_b = np.cos(theta_b), np.sin(theta_b)
        
        # Draw cells as circles
        circle_a = plt.Circle((x_a, y_a), 0.15, color='#e74c3c', alpha=0.7, zorder=3)
        circle_b = plt.Circle((x_b, y_b), 0.15, color='#3498db', alpha=0.7, zorder=3)
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        
        # Labels
        ax.text(x_a, y_a, _truncate(cell_a, 12), ha='center', va='center',
               fontsize=9, fontweight='bold', color='white', zorder=4)
        ax.text(x_b, y_b, _truncate(cell_b, 12), ha='center', va='center',
               fontsize=9, fontweight='bold', color='white', zorder=4)
        
        # Curved arrows for bidirectional flow
        # A -> B (top arc)
        arrow_ab = mpatches.FancyArrowPatch(
            (x_a + 0.12, y_a + 0.08), (x_b - 0.12, y_b + 0.08),
            arrowstyle='-|>', mutation_scale=25, lw=3,
            color='#e74c3c', alpha=0.8, zorder=2,
            connectionstyle="arc3,rad=0.3"
        )
        ax.add_patch(arrow_ab)
        
        # B -> A (bottom arc)
        arrow_ba = mpatches.FancyArrowPatch(
            (x_b - 0.12, y_b - 0.08), (x_a + 0.12, y_a - 0.08),
            arrowstyle='-|>', mutation_scale=25, lw=3,
            color='#3498db', alpha=0.8, zorder=2,
            connectionstyle="arc3,rad=0.3"
        )
        ax.add_patch(arrow_ba)
        
        # Metabolite labels on arrows
        mid_x_top = (x_a + x_b) / 2
        mid_y_top = (y_a + y_b) / 2 + 0.2
        ax.text(mid_x_top, mid_y_top, _truncate(mets_ab, 15),
               ha='center', va='bottom', fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.3))
        
        mid_y_bot = (y_a + y_b) / 2 - 0.2
        ax.text(mid_x_top, mid_y_bot, _truncate(mets_ba, 15),
               ha='center', va='top', fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.3))
        
        # Score and coupling
        ax.text(0, -0.8, f"Score: {score:.0f}",
               ha='center', fontsize=10, fontweight='bold')
        ax.text(0, -0.95, f"Coupling: {coupling:.2f}",
               ha='center', fontsize=9)
        ax.text(0, -1.1, f"Region {loop['region']}",
               ha='center', fontsize=9, color='gray')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, "bidirectional_loops_network.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


def plot_network_comparison(hubs_df, out_dir):
    """
    Network hub bar chart.
    Two conditions: side-by-side Healthy vs Injured.
    Single condition: one panel using whatever condition label is present.
    """
    if hubs_df.empty:
        return None

    conditions_present = hubs_df["condition"].unique().tolist()
    n_panels = len(conditions_present)
    palette = ["#27ae60", "#e74c3c", "#3498db", "#9b59b6"]

    fig, axes = plt.subplots(1, max(1, n_panels),
                              figsize=(10 * max(1, n_panels), 8))
    if n_panels == 1:
        axes = [axes]

    title = ("Network Hub Analysis: " + " vs ".join(conditions_present))
    fig.suptitle(title, fontsize=18, fontweight='bold')

    for ax, condition, color in zip(axes, conditions_present, palette):
        data = hubs_df[hubs_df["condition"] == condition].head(10)
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14)
            ax.set_title(condition, fontsize=16, fontweight='bold')
            ax.axis('off')
            continue
        y_pos = np.arange(len(data))
        scores = data["total_score"].values
        ax.barh(y_pos, scores, color=color, alpha=0.7,
                edgecolor='black', linewidth=1.5)
        for i, score in enumerate(scores):
            ax.text(score + max(scores) * 0.02, i, f"{score:.0f}",
                    va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_truncate(ct, 25) for ct in data["cell_type"]], fontsize=11)
        ax.set_xlabel("Total Interaction Score", fontsize=12, fontweight='bold')
        ax.set_title(condition, fontsize=16, fontweight='bold', color=color)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(out_dir, "network_hubs_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


def plot_metabolite_dominance_heatmap(dominance_df, out_dir):
    """
    Metabolite dominance visualisation.
    Two conditions: Healthy vs Injured heatmap with fold-change column.
    Single condition: horizontal bar chart ranked by score.
    """
    if dominance_df.empty:
        return None

    top_mets = dominance_df.head(20).copy()
    is_single = (top_mets["injured_score"] == 0).all()

    if is_single:
        # ── Single-condition: score bar chart ────────────────────────
        top_mets = top_mets.sort_values("healthy_score", ascending=True)
        fig, ax = plt.subplots(figsize=(12, max(6, len(top_mets) * 0.42)))
        sc_cmap = plt.get_cmap("YlOrRd")
        scores = top_mets["healthy_score"].values
        norm_s = scores / (scores.max() + 1e-9)
        y_pos  = np.arange(len(top_mets))
        for i, (s, sn) in enumerate(zip(scores, norm_s)):
            ax.barh(i, s, color=sc_cmap(0.2 + 0.75 * sn),
                    height=0.72, edgecolor="white", linewidth=0.4)
            ax.text(s * 1.01, i, f"{s:.0f}", va="center", ha="left",
                    fontsize=8, color="#222222")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_mets["metabolite"].values, fontsize=10)
        ax.set_xlabel("Interaction Score", fontsize=12, fontweight="bold")
        ax.set_title("Metabolite Dominance by Interaction Score",
                     fontsize=16, fontweight="bold", pad=15)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        filepath = os.path.join(out_dir, "metabolite_dominance_heatmap.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return filepath

    # ── Two-condition: Healthy vs Injured heatmap ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    metabolites = top_mets["metabolite"].values
    matrix_data = top_mets[["healthy_score", "injured_score"]].values
    matrix_log  = np.log10(matrix_data + 1)
    im = ax.imshow(matrix_log, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy", "Injured"], fontsize=14, fontweight="bold")
    ax.set_yticks(np.arange(len(metabolites)))
    ax.set_yticklabels(metabolites, fontsize=10)
    for i in range(len(metabolites)):
        for j in range(2):
            val = matrix_data[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color="white" if matrix_log[i, j] > matrix_log.max() * 0.6 else "black",
                    fontsize=9, fontweight="bold")
    for i, fc in enumerate(top_mets["fold_change"]):
        ax.text(2.3, i, f"{fc:.1f}x", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="orange" if fc > 5 else "yellow", alpha=0.6))
    ax.text(2.3, -1, "Fold\nChange", ha="center", va="center",
            fontsize=10, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, pad=0.15)
    cbar.set_label("log10(Score + 1)", fontsize=12, fontweight="bold")
    ax.set_title("Metabolite Dominance: Healthy vs Injured",
                 fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    filepath = os.path.join(out_dir, "metabolite_dominance_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return filepath


def plot_leukotriene_localization(leuk_df, out_dir):
    """
    Bar plot showing leukotriene signaling by region and source.
    """
    if leuk_df.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Leukotriene Signaling: Spatial Localization",
                 fontsize=16, fontweight='bold')
    
    # Plot 1: By region
    ax = axes[0]
    region_scores = leuk_df.groupby("region")["score"].sum().sort_values(ascending=False)
    
    bars = ax.bar(range(len(region_scores)), region_scores.values,
                  color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(region_scores)))
    ax.set_xticklabels([f"R{r}" for r in region_scores.index], fontsize=12)
    ax.set_ylabel("Total LTE4/LTF4 Score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Region", fontsize=12, fontweight='bold')
    ax.set_title("Regional Distribution", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Annotate bars
    for i, val in enumerate(region_scores.values):
        ax.text(i, val + max(region_scores) * 0.02, f"{val:.0f}",
               ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: By source cell type
    ax = axes[1]
    source_scores = leuk_df.groupby("source")["score"].sum().sort_values(ascending=False).head(8)
    
    y_pos = np.arange(len(source_scores))
    bars = ax.barh(y_pos, source_scores.values,
                   color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_truncate(s, 25) for s in source_scores.index], fontsize=10)
    ax.set_xlabel("Total LTE4/LTF4 Score", fontsize=12, fontweight='bold')
    ax.set_title("Source Cell Types", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Annotate bars
    for i, val in enumerate(source_scores.values):
        ax.text(val + max(source_scores) * 0.02, i, f"{val:.0f}",
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, "leukotriene_spatial_localization.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


def plot_therapeutic_targets_summary(targets_df, out_dir):
    """
    Summary figure for therapeutic targets with priority coding.
    """
    if targets_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color by priority
    colors = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#95a5a6'}
    target_colors = [colors.get(p, '#95a5a6') for p in targets_df["priority"]]
    
    y_pos = np.arange(len(targets_df))
    
    # Create horizontal bars (just for visual structure)
    for i, (_, target) in enumerate(targets_df.iterrows()):
        # Background bar
        ax.barh(i, 1, height=0.8, color=target_colors[i], alpha=0.2)
        
        # Target name
        ax.text(0.02, i, target["target"],
               va='center', ha='left', fontsize=12, fontweight='bold')
        
        # Metabolite
        ax.text(0.35, i + 0.25, f"Metabolite: {target['metabolite']}",
               va='center', ha='left', fontsize=9, style='italic')
        
        # Rationale
        ax.text(0.35, i, _truncate(target["rationale"], 60),
               va='center', ha='left', fontsize=9)
        
        # Approaches
        ax.text(0.35, i - 0.25, f"Rx: {_truncate(target['therapeutic_approach'], 50)}",
               va='center', ha='left', fontsize=8, color='#2c3e50')
        
        # Priority badge
        ax.text(0.98, i, target["priority"],
               va='center', ha='right', fontsize=11, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=target_colors[i], alpha=0.9))
    
    ax.set_ylim(-0.5, len(targets_df) - 0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title("Therapeutic Target Recommendations",
                fontsize=18, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=colors[p], label=p, alpha=0.8)
                      for p in ['HIGH', 'MEDIUM', 'LOW'] if p in targets_df["priority"].values]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, title="Priority", title_fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(out_dir, "therapeutic_targets_summary.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath


def _truncate(text, max_len):
    """Helper to truncate long text."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len-3] + "..."


def generate_all_enhanced_plots(results, cfg=None, verbose=False):
    """
    FIXED: Accept results object directly, compute enhanced analyses internally.
    Generate all enhanced visualization plots.
    
    Args:
        results: AnalysisResults object from run_analysis_pipeline
        cfg: PlotConfig (optional, not used currently)
        verbose: Print progress messages
    """
    # Import here to avoid circular imports
    from enhanced_analysis import (
        identify_bidirectional_loops,
        analyze_network_topology,
        analyze_metabolite_dominance,
        detect_leukotriene_signaling,
        generate_therapeutic_targets
    )
    
    out_dir = results.config.out_dir
    
    if verbose:
        print("\n" + "="*80)
        print("GENERATING ENHANCED VISUALIZATIONS")
        print("="*80)
    
    # Compute all enhanced analyses
    if verbose:
        print("\n🔬 Computing enhanced analyses...")
    
    # Bidirectional loops need two conditions; use lower min_score for single
    from enhanced_analysis import _is_single_condition as _isc
    loop_min_score = 100.0 if _isc(results) else 500.0
    loops_df = identify_bidirectional_loops(results, min_score=loop_min_score)
    hubs_df = analyze_network_topology(results)
    dominance_df = analyze_metabolite_dominance(results)
    leuk_df = detect_leukotriene_signaling(results)
    targets_df = generate_therapeutic_targets(results, loops_df, dominance_df)
    
    # Package results
    enhanced_results = {
        "loops": loops_df,
        "hubs": hubs_df,
        "dominance": dominance_df,
        "leukotrienes": leuk_df,  # Note: key name is "leukotrienes" for plot function
        "targets": targets_df
    }
    
    plots = {}
    
    # 1. Bidirectional loops
    if not enhanced_results["loops"].empty:
        if verbose:
            print("\n📊 Creating bidirectional loops network...")
        plots["loops"] = plot_bidirectional_loops(enhanced_results["loops"], out_dir)
        if plots["loops"] and verbose:
            print(f"   ✓ Saved: {plots['loops']}")
    
    # 2. Network hubs comparison
    if not enhanced_results["hubs"].empty:
        if verbose:
            print("\n🕸️  Creating network hubs comparison...")
        plots["hubs"] = plot_network_comparison(enhanced_results["hubs"], out_dir)
        if plots["hubs"] and verbose:
            print(f"   ✓ Saved: {plots['hubs']}")
    
    # 3. Metabolite dominance heatmap
    if not enhanced_results["dominance"].empty:
        if verbose:
            print("\n🔬 Creating metabolite dominance heatmap...")
        plots["dominance"] = plot_metabolite_dominance_heatmap(enhanced_results["dominance"], out_dir)
        if plots["dominance"] and verbose:
            print(f"   ✓ Saved: {plots['dominance']}")
    
    # 4. Leukotriene localization
    if not enhanced_results["leukotrienes"].empty:
        if verbose:
            print("\n🔥 Creating leukotriene localization plot...")
        plots["leukotrienes"] = plot_leukotriene_localization(enhanced_results["leukotrienes"], out_dir)
        if plots["leukotrienes"] and verbose:
            print(f"   ✓ Saved: {plots['leukotrienes']}")
    
    # 5. Therapeutic targets
    if not enhanced_results["targets"].empty:
        if verbose:
            print("\n💊 Creating therapeutic targets summary...")
        plots["targets"] = plot_therapeutic_targets_summary(enhanced_results["targets"], out_dir)
        if plots["targets"] and verbose:
            print(f"   ✓ Saved: {plots['targets']}")
    
    if verbose:
        print("\n" + "="*80)
        print("✅ ALL ENHANCED PLOTS GENERATED!")
        print("="*80)
    
    return plots


if __name__ == "__main__":
    print("Enhanced plotting module loaded.")
    print("Use: plots = generate_all_enhanced_plots(results, verbose=True)")


# =============================================================================
# SPOTLIGHT REPORT  — top-level entry point for key metabolite deep-dives
# =============================================================================

def generate_spotlight_report(results, cfg=None, extra_metabolites=None, verbose=True):
    """
    End-to-end spotlight pipeline:

      1. Auto-select key metabolites via select_spotlight_metabolites()
         (mandatory: lac_L, lac_D, dhdascb, ascb_L, sphs1p + up to 2 auto)
      2. Add any extra_metabolites the caller specifies (e.g. ['hco3', 'gal'])
      3. Generate one combined streamline+diffusion+text figure per metabolite
      4. Write the standalone biology text report

    Parameters
    ----------
    results           : AnalysisResults from SPICEM pipeline
    cfg               : PlotConfig (optional)
    extra_metabolites : list of str, additional metabolites to include
    verbose           : print progress

    Returns
    -------
    dict with keys:
        'figures'  → list of PNG file paths
        'report'   → path to spotlight_biology_report.txt
        'metabolites' → list of selected metabolite IDs
        'annotations' → annotation dict (full_name, biology, refs, stats)
    """
    from enhanced_analysis import select_spotlight_metabolites
    from plotting import (
        plot_key_metabolites_spotlight,
        generate_spotlight_text_report,
        DEFAULT_CONFIG,
    )

    if cfg is None:
        cfg = DEFAULT_CONFIG

    base = getattr(getattr(results, "config", None), "out_dir", "analysis_output")
    out_dir = os.path.join(base, "plots_spotlight")
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 80)
        print("SPOTLIGHT METABOLITE REPORT")
        print("=" * 80)

    # ── 1. Select metabolites ──────────────────────────────────────────────
    metabolites, annotations = select_spotlight_metabolites(
        results, extra_mandatory=extra_metabolites
    )
    if verbose:
        print(f"\nSelected {len(metabolites)} metabolites for spotlight:")
        for m in metabolites:
            ann = annotations.get(m, {})
            print(f"  • {m:15s}  {ann.get('full_name', '')} "
                  f"({ann.get('source','')} → {ann.get('sink','')})")

    # ── 2. Generate figures ────────────────────────────────────────────────
    if verbose:
        print("\nGenerating spotlight figures...")
    figures = plot_key_metabolites_spotlight(
        results,
        metabolites=metabolites,
        annotations=annotations,
        out_dir=out_dir,
        cfg=cfg,
        verbose=verbose,
    )

    # ── 3. Write text report ───────────────────────────────────────────────
    if verbose:
        print("\nWriting biology text report...")
    report_path = generate_spotlight_text_report(annotations, out_dir)
    if verbose:
        print(f"  Saved: {report_path}")

    if verbose:
        print("\n" + "=" * 80)
        print(f"SPOTLIGHT COMPLETE  —  {len(figures)} figures + 1 report")
        print(f"Output directory: {out_dir}")
        print("=" * 80)

    return {
        "figures": figures,
        "report": report_path,
        "metabolites": metabolites,
        "annotations": annotations,
    }