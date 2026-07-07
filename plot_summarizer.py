#!/usr/bin/env python3
"""
PLOT RESULTS SUMMARIZER FOR JUPYTER NOTEBOOK
============================================

This script analyzes your generated plots and creates comprehensive summaries:
- CSV files with quantitative metrics for each metabolite/condition
- Text reports with interpretable findings
- Statistical summaries across all plots

Run this AFTER you've generated all your plots with generate_all_plots()
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

class SummaryConfig:
    """Configuration for summary generation."""
    
    def __init__(self, results, output_dir: str = None):
        """
        Parameters
        ----------
        results : AnalysisResults
            Your analysis results object from the pipeline
        output_dir : str, optional
            Directory for summary outputs. If None, uses results.config.out_dir/summaries
        """
        self.results = results
        
        if output_dir is None:
            base_dir = getattr(results.config, 'out_dir', 'analysis_output')
            self.output_dir = os.path.join(base_dir, 'summaries')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories for different plot types
        self.plots_dir = getattr(results.config, 'out_dir', 'analysis_output')
        self.regional_dir = os.path.join(self.plots_dir, 'plots_regional')
        self.consensus_dir = os.path.join(self.plots_dir, 'plots_consensus')
        self.differential_dir = os.path.join(self.plots_dir, 'plots_differential')


# =============================================================================
# EXTRACT METABOLITE INFO FROM RESULTS
# =============================================================================

def extract_metabolite_summary(results) -> pd.DataFrame:
    """
    Extract comprehensive metabolite information from results.
    
    Returns
    -------
    pd.DataFrame
        Columns: Metabolite, Source, Sink, Score, PValue, Log2FC_Secretion, 
                 P_Secretion, Log2FC_Uptake, P_Uptake, Significant
    """
    rows = []
    
    # Get metabolite scores
    met_scores = getattr(results, 'met_scores', [])
    
    for met, score in met_scores:
        # Get source/sink info
        src, snk, p_val = results.get_metabolite_info(met)
        
        # Get comparison stats if available
        comp_df = getattr(results, 'comparison_df', pd.DataFrame())
        if not comp_df.empty and met in comp_df['Metabolite'].values:
            comp_row = comp_df[comp_df['Metabolite'] == met].iloc[0]
            log2fc_sec = comp_row.get('Log2FCSecretion', np.nan)
            p_sec = comp_row.get('PSecretion', np.nan)
            log2fc_upt = comp_row.get('Log2FCUptake', np.nan)
            p_upt = comp_row.get('PUptake', np.nan)
        else:
            log2fc_sec = np.nan
            p_sec = np.nan
            log2fc_upt = np.nan
            p_upt = np.nan
        
        # Determine significance
        significant = False
        if not np.isnan(p_sec) and p_sec < 0.05:
            significant = True
        elif not np.isnan(p_val) and p_val < 0.05:
            significant = True
        
        rows.append({
            'Metabolite': met,
            'Source': src or 'Unknown',
            'Sink': snk or 'Unknown',
            'InteractionScore': float(score),
            'PermutationPValue': float(p_val) if p_val is not None else np.nan,
            'Log2FC_Secretion': float(log2fc_sec),
            'PValue_Secretion': float(p_sec),
            'Log2FC_Uptake': float(log2fc_upt),
            'PValue_Uptake': float(p_upt),
            'Significant': significant,
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# SCAN PLOT DIRECTORIES
# =============================================================================

def scan_plot_directory(directory: str, pattern: str = None) -> Dict[str, List[str]]:
    """
    Scan a directory for plot files and categorize them.
    
    Parameters
    ----------
    directory : str
        Directory to scan
    pattern : str, optional
        Regex pattern to match filenames
        
    Returns
    -------
    dict
        Mapping of metabolite names to list of plot files
    """
    if not os.path.exists(directory):
        return {}
    
    plot_files = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not filename.endswith('.png'):
            continue
            
        # Extract metabolite name from filename
        # Patterns: consensus_METABOLITE.png, diff_METABOLITE.png, regional_CONDITION_REGION_METABOLITE.png
        if filename.startswith('consensus_'):
            met = filename.replace('consensus_', '').replace('.png', '')
            plot_files[met].append(os.path.join(directory, filename))
        elif filename.startswith('diff_'):
            met = filename.replace('diff_', '').replace('.png', '')
            plot_files[met].append(os.path.join(directory, filename))
        elif 'regional_' in filename:
            # Pattern: regional_CONDITION_RX_METABOLITE.png
            parts = filename.replace('regional_', '').replace('.png', '').split('_')
            if len(parts) >= 3:
                met = '_'.join(parts[2:])  # Everything after CONDITION_RX
                plot_files[met].append(os.path.join(directory, filename))
    
    return dict(plot_files)


def create_plot_inventory(config: SummaryConfig) -> pd.DataFrame:
    """
    Create inventory of all generated plots.
    
    Returns
    -------
    pd.DataFrame
        Columns: Metabolite, PlotType, Condition, Region, FilePath, Exists
    """
    rows = []
    
    # Scan each directory
    regional_plots = scan_plot_directory(config.regional_dir)
    consensus_plots = scan_plot_directory(config.consensus_dir)
    differential_plots = scan_plot_directory(config.differential_dir)
    
    # Regional plots
    for met, files in regional_plots.items():
        for filepath in files:
            filename = os.path.basename(filepath)
            # Extract condition and region from filename
            match = re.match(r'regional_(\w+)_R(\d+)_', filename)
            if match:
                condition = match.group(1)
                region = int(match.group(2))
            else:
                condition = 'Unknown'
                region = -1
                
            rows.append({
                'Metabolite': met,
                'PlotType': 'Regional',
                'Condition': condition,
                'Region': region,
                'FilePath': filepath,
                'Exists': os.path.exists(filepath)
            })
    
    # Consensus plots
    for met, files in consensus_plots.items():
        for filepath in files:
            rows.append({
                'Metabolite': met,
                'PlotType': 'Consensus',
                'Condition': 'All',
                'Region': -1,
                'FilePath': filepath,
                'Exists': os.path.exists(filepath)
            })
    
    # Differential plots
    for met, files in differential_plots.items():
        for filepath in files:
            rows.append({
                'Metabolite': met,
                'PlotType': 'Differential',
                'Condition': 'Comparison',
                'Region': -1,
                'FilePath': filepath,
                'Exists': os.path.exists(filepath)
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# GENERATE TEXT SUMMARIES
# =============================================================================

def generate_metabolite_report(met_df: pd.DataFrame, plot_inv: pd.DataFrame) -> str:
    """
    Generate a human-readable text report for all metabolites.
    
    Parameters
    ----------
    met_df : pd.DataFrame
        Metabolite summary dataframe
    plot_inv : pd.DataFrame
        Plot inventory dataframe
        
    Returns
    -------
    str
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("METABOLITE FLOW ANALYSIS SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall statistics
    n_total = len(met_df)
    n_significant = met_df['Significant'].sum()
    n_plots_total = len(plot_inv)
    n_plots_regional = len(plot_inv[plot_inv['PlotType'] == 'Regional'])
    n_plots_consensus = len(plot_inv[plot_inv['PlotType'] == 'Consensus'])
    n_plots_differential = len(plot_inv[plot_inv['PlotType'] == 'Differential'])
    
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Total metabolites analyzed: {n_total}")
    lines.append(f"Significant metabolites (p < 0.05): {n_significant} ({n_significant/n_total*100:.1f}%)")
    lines.append(f"Total plots generated: {n_plots_total}")
    lines.append(f"  - Regional plots: {n_plots_regional}")
    lines.append(f"  - Consensus plots: {n_plots_consensus}")
    lines.append(f"  - Differential plots: {n_plots_differential}")
    lines.append("")
    
    # Top metabolites by score
    lines.append("TOP 10 METABOLITES BY INTERACTION SCORE")
    lines.append("-" * 80)
    top_10 = met_df.nlargest(10, 'InteractionScore')
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        lines.append(f"{i}. {row['Metabolite']}")
        lines.append(f"   Source: {row['Source']} → Sink: {row['Sink']}")
        lines.append(f"   Interaction Score: {row['InteractionScore']:.4f}")
        
        if not np.isnan(row['PermutationPValue']):
            lines.append(f"   Permutation p-value: {row['PermutationPValue']:.4g}")
        
        if not np.isnan(row['PValue_Secretion']):
            fc_direction = "↑" if row['Log2FC_Secretion'] > 0 else "↓"
            lines.append(f"   Secretion: Log2FC = {row['Log2FC_Secretion']:.2f} {fc_direction}, p = {row['PValue_Secretion']:.4g}")
        
        # Count plots for this metabolite
        met_plots = plot_inv[plot_inv['Metabolite'] == row['Metabolite']]
        has_consensus = any(met_plots['PlotType'] == 'Consensus')
        has_differential = any(met_plots['PlotType'] == 'Differential')
        n_regional = len(met_plots[met_plots['PlotType'] == 'Regional'])
        
        plot_info = []
        if has_consensus:
            plot_info.append("Consensus")
        if has_differential:
            plot_info.append("Differential")
        if n_regional > 0:
            plot_info.append(f"{n_regional} Regional")
        
        lines.append(f"   Plots: {', '.join(plot_info)}")
        lines.append("")
    
    # Significantly changed metabolites
    significant_mets = met_df[met_df['Significant']].copy()
    if len(significant_mets) > 0:
        lines.append("")
        lines.append("SIGNIFICANTLY CHANGED METABOLITES (p < 0.05)")
        lines.append("-" * 80)
        
        # Sort by p-value
        significant_mets = significant_mets.sort_values('PValue_Secretion')
        
        for i, (_, row) in enumerate(significant_mets.iterrows(), 1):
            lines.append(f"{i}. {row['Metabolite']}")
            lines.append(f"   {row['Source']} → {row['Sink']}")
            
            if not np.isnan(row['Log2FC_Secretion']):
                change = "Increased" if row['Log2FC_Secretion'] > 0 else "Decreased"
                lines.append(f"   Secretion: {change} (Log2FC = {row['Log2FC_Secretion']:.2f}, p = {row['PValue_Secretion']:.4g})")
            
            if not np.isnan(row['Log2FC_Uptake']):
                change = "Increased" if row['Log2FC_Uptake'] > 0 else "Decreased"
                lines.append(f"   Uptake: {change} (Log2FC = {row['Log2FC_Uptake']:.2f}, p = {row['PValue_Uptake']:.4g})")
            
            lines.append("")
    
    # Cell type hub analysis
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_plot_coverage_report(plot_inv: pd.DataFrame, met_df: pd.DataFrame) -> str:
    """
    Generate report on plot coverage - which metabolites have which plots.
    
    Parameters
    ----------
    plot_inv : pd.DataFrame
        Plot inventory
    met_df : pd.DataFrame
        Metabolite summary
        
    Returns
    -------
    str
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PLOT COVERAGE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Metabolites with all plot types
    all_mets = set(met_df['Metabolite'])
    
    has_all = []
    has_consensus_diff = []
    has_partial = []
    has_none = []
    
    for met in all_mets:
        met_plots = plot_inv[plot_inv['Metabolite'] == met]
        plot_types = set(met_plots['PlotType'])
        
        if 'Consensus' in plot_types and 'Differential' in plot_types and 'Regional' in plot_types:
            has_all.append(met)
        elif 'Consensus' in plot_types and 'Differential' in plot_types:
            has_consensus_diff.append(met)
        elif len(plot_types) > 0:
            has_partial.append(met)
        else:
            has_none.append(met)
    
    lines.append(f"COMPLETE COVERAGE (Regional + Consensus + Differential): {len(has_all)} metabolites")
    if has_all:
        for met in sorted(has_all)[:10]:  # Show first 10
            lines.append(f"  - {met}")
        if len(has_all) > 10:
            lines.append(f"  ... and {len(has_all) - 10} more")
    lines.append("")
    
    lines.append(f"CONSENSUS + DIFFERENTIAL (no regional): {len(has_consensus_diff)} metabolites")
    if has_consensus_diff:
        for met in sorted(has_consensus_diff):
            lines.append(f"  - {met}")
    lines.append("")
    
    lines.append(f"PARTIAL COVERAGE: {len(has_partial)} metabolites")
    if has_partial:
        for met in sorted(has_partial):
            met_plots = plot_inv[plot_inv['Metabolite'] == met]
            plot_types = ', '.join(sorted(set(met_plots['PlotType'])))
            lines.append(f"  - {met}: {plot_types}")
    lines.append("")
    
    if has_none:
        lines.append(f"NO PLOTS GENERATED: {len(has_none)} metabolites")
        for met in sorted(has_none):
            lines.append(f"  - {met}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_regional_breakdown(plot_inv: pd.DataFrame, results) -> str:
    """
    Generate report showing regional plot distribution.
    
    Parameters
    ----------
    plot_inv : pd.DataFrame
        Plot inventory
    results : AnalysisResults
        Analysis results object
        
    Returns
    -------
    str
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("REGIONAL ANALYSIS BREAKDOWN")
    lines.append("=" * 80)
    lines.append("")
    
    regional_plots = plot_inv[plot_inv['PlotType'] == 'Regional'].copy()
    
    if len(regional_plots) == 0:
        lines.append("No regional plots found.")
        return "\n".join(lines)
    
    # Group by condition and region
    conditions = results.config.conditions
    regions = results.config.regions
    
    for condition in conditions:
        lines.append(f"CONDITION: {condition}")
        lines.append("-" * 80)
        
        cond_plots = regional_plots[regional_plots['Condition'] == condition]
        
        for region in regions:
            reg_plots = cond_plots[cond_plots['Region'] == region]
            n_plots = len(reg_plots)
            
            if n_plots > 0:
                lines.append(f"  Region {region}: {n_plots} plots")
                
                # List metabolites
                metabolites = sorted(set(reg_plots['Metabolite']))
                if len(metabolites) <= 5:
                    for met in metabolites:
                        lines.append(f"    - {met}")
                else:
                    for met in metabolites[:5]:
                        lines.append(f"    - {met}")
                    lines.append(f"    ... and {len(metabolites) - 5} more")
        
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# MAIN SUMMARY FUNCTION
# =============================================================================

def create_comprehensive_summaries(results, output_dir: str = None, verbose: bool = True):
    """
    Create comprehensive summaries of all analysis results and plots.
    
    This function generates:
    1. metabolite_summary.csv - Detailed metabolite statistics
    2. plot_inventory.csv - Complete list of all generated plots
    3. metabolite_report.txt - Human-readable summary report
    4. plot_coverage_report.txt - Which metabolites have which plots
    5. regional_breakdown.txt - Regional analysis details
    
    Parameters
    ----------
    results : AnalysisResults
        Your analysis results object from run_analysis_pipeline()
    output_dir : str, optional
        Directory for output files. If None, uses results.config.out_dir/summaries
    verbose : bool, default True
        Print progress messages
        
    Returns
    -------
    dict
        Paths to all generated summary files
    """
    config = SummaryConfig(results, output_dir)
    
    if verbose:
        print("=" * 80)
        print("CREATING COMPREHENSIVE SUMMARIES")
        print("=" * 80)
        print(f"Output directory: {config.output_dir}")
        print()
    
    output_files = {}
    
    # 1. Extract metabolite summary
    if verbose:
        print("1. Extracting metabolite information...")
    
    met_df = extract_metabolite_summary(results)
    out_path = os.path.join(config.output_dir, 'metabolite_summary.csv')
    met_df.to_csv(out_path, index=False)
    output_files['metabolite_summary'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
        print(f"   ({len(met_df)} metabolites)")
    
    # 2. Create plot inventory
    if verbose:
        print("\n2. Scanning plot directories...")
    
    plot_inv = create_plot_inventory(config)
    out_path = os.path.join(config.output_dir, 'plot_inventory.csv')
    plot_inv.to_csv(out_path, index=False)
    output_files['plot_inventory'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
        print(f"   ({len(plot_inv)} plots found)")
    
    # 3. Generate metabolite report
    if verbose:
        print("\n3. Generating metabolite report...")
    
    report_text = generate_metabolite_report(met_df, plot_inv)
    out_path = os.path.join(config.output_dir, 'metabolite_report.txt')
    with open(out_path, 'w') as f:
        f.write(report_text)
    output_files['metabolite_report'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
    
    # 4. Generate plot coverage report
    if verbose:
        print("\n4. Generating plot coverage report...")
    
    coverage_text = generate_plot_coverage_report(plot_inv, met_df)
    out_path = os.path.join(config.output_dir, 'plot_coverage_report.txt')
    with open(out_path, 'w') as f:
        f.write(coverage_text)
    output_files['plot_coverage'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
    
    # 5. Generate regional breakdown
    if verbose:
        print("\n5. Generating regional breakdown...")
    
    regional_text = generate_regional_breakdown(plot_inv, results)
    out_path = os.path.join(config.output_dir, 'regional_breakdown.txt')
    with open(out_path, 'w') as f:
        f.write(regional_text)
    output_files['regional_breakdown'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
    
    # 6. Summary statistics
    if verbose:
        print("\n6. Creating summary statistics...")
    
    stats = {
        'Total_Metabolites': len(met_df),
        'Significant_Metabolites': int(met_df['Significant'].sum()),
        'Total_Plots': len(plot_inv),
        'Regional_Plots': len(plot_inv[plot_inv['PlotType'] == 'Regional']),
        'Consensus_Plots': len(plot_inv[plot_inv['PlotType'] == 'Consensus']),
        'Differential_Plots': len(plot_inv[plot_inv['PlotType'] == 'Differential']),
        'Unique_Metabolites_Plotted': len(set(plot_inv['Metabolite'])),
    }
    
    stats_df = pd.DataFrame([stats])
    out_path = os.path.join(config.output_dir, 'summary_statistics.csv')
    stats_df.to_csv(out_path, index=False)
    output_files['summary_statistics'] = out_path
    
    if verbose:
        print(f"   Saved: {out_path}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY COMPLETE")
        print("=" * 80)
        print("\nGenerated files:")
        for key, path in output_files.items():
            print(f"  - {key}: {path}")
        print()
        print("Quick Stats:")
        print(f"  Total metabolites: {stats['Total_Metabolites']}")
        print(f"  Significant (p<0.05): {stats['Significant_Metabolites']}")
        print(f"  Total plots: {stats['Total_Plots']}")
        print(f"    Regional: {stats['Regional_Plots']}")
        print(f"    Consensus: {stats['Consensus_Plots']}")
        print(f"    Differential: {stats['Differential_Plots']}")
        print()
    
    return output_files


# =============================================================================
# JUPYTER NOTEBOOK CONVENIENCE FUNCTIONS
# =============================================================================

def display_summary_preview(results, n_metabolites: int = 5):
    """
    Display a quick preview of results in Jupyter notebook.
    
    Parameters
    ----------
    results : AnalysisResults
        Analysis results object
    n_metabolites : int
        Number of top metabolites to show
    """
    met_df = extract_metabolite_summary(results)
    
    print("=" * 80)
    print(f"TOP {n_metabolites} METABOLITES BY INTERACTION SCORE")
    print("=" * 80)
    
    top_n = met_df.nlargest(n_metabolites, 'InteractionScore')
    
    for i, (_, row) in enumerate(top_n.iterrows(), 1):
        print(f"\n{i}. {row['Metabolite']}")
        print(f"   Source: {row['Source']} → Sink: {row['Sink']}")
        print(f"   Score: {row['InteractionScore']:.4f}")
        
        if row['Significant']:
            print(f"   ⭐ SIGNIFICANT (p < 0.05)")
        
        if not np.isnan(row['Log2FC_Secretion']):
            direction = "↑ Increased" if row['Log2FC_Secretion'] > 0 else "↓ Decreased"
            print(f"   Secretion: {direction} (Log2FC={row['Log2FC_Secretion']:.2f})")


def quick_summary(results):
    """
    Ultra-quick summary - just the key numbers.
    
    Parameters
    ----------
    results : AnalysisResults
        Analysis results object
    """
    met_df = extract_metabolite_summary(results)
    config = SummaryConfig(results)
    plot_inv = create_plot_inventory(config)
    
    print("QUICK SUMMARY")
    print("=" * 40)
    print(f"Metabolites analyzed: {len(met_df)}")
    print(f"Significant (p<0.05): {met_df['Significant'].sum()}")
    print(f"Total plots: {len(plot_inv)}")
    print(f"  Regional: {len(plot_inv[plot_inv['PlotType'] == 'Regional'])}")
    print(f"  Consensus: {len(plot_inv[plot_inv['PlotType'] == 'Consensus'])}")
    print(f"  Differential: {len(plot_inv[plot_inv['PlotType'] == 'Differential'])}")


# =============================================================================
# EXAMPLE USAGE FOR JUPYTER NOTEBOOK
# =============================================================================

if __name__ == "__main__":
    print("""
    USAGE IN JUPYTER NOTEBOOK:
    ==========================
    
    # After running your analysis pipeline:
    from spatialmetabolicanalysis import run_analysis_pipeline, AnalysisConfig
    
    config = AnalysisConfig(...)
    results = run_analysis_pipeline(config)
    
    # Import this script
    %run plot_summarizer.py
    # OR
    # from plot_summarizer import create_comprehensive_summaries, quick_summary
    
    # Quick preview
    quick_summary(results)
    
    # Full summaries
    output_files = create_comprehensive_summaries(results, verbose=True)
    
    # View specific summaries
    import pandas as pd
    met_summary = pd.read_csv(output_files['metabolite_summary'])
    display(met_summary.head(10))
    
    # Or display preview
    display_summary_preview(results, n_metabolites=10)
    """)