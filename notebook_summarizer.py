# PLOT RESULTS SUMMARIZER - JUPYTER NOTEBOOK
# ============================================
# 
# This notebook creates comprehensive summaries of your analysis results:
# - CSV files with metabolite statistics
# - Text reports with key findings
# - Plot inventories.
# 
# Run this AFTER you've completed your analysis and generated plots

# %% [markdown]
# ## Setup and Imports

# %%
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import the summarizer script
# Adjust this path to where you saved plot_summarizer.py
sys.path.insert(0, '/path/to/your/scripts')
from plot_summarizer import (
    create_comprehensive_summaries,
    quick_summary,
    display_summary_preview,
    extract_metabolite_summary,
    create_plot_inventory,
    SummaryConfig
)

# %% [markdown]
# ## Load Your Analysis Results
# 
# Make sure you've already run your analysis pipeline.
# If not, run it first:

# %%
# OPTION 1: If results are already in memory (you just ran the analysis)
# results should already be defined

# OPTION 2: If you need to reload results from a previous run
# You'll need to re-run the analysis pipeline or load saved results
# from spatialmetabolicanalysis import run_analysis_pipeline, AnalysisConfig
# 
# config = AnalysisConfig(
#     base_dir="/path/to/data",
#     out_dir="/path/to/output",
#     conditions=["Sample_Healthy", "Sample_Injured"],
#     regions=[1, 2, 3, 4, 5],
#     # ... other config params
# )
# results = run_analysis_pipeline(config)

# %% [markdown]
# ## Quick Summary
# 
# Get a fast overview of your results:

# %%
quick_summary(results)

# %% [markdown]
# ## Preview Top Metabolites
# 
# See the top metabolites ranked by interaction score:

# %%
display_summary_preview(results, n_metabolites=10)

# %% [markdown]
# ## Generate Comprehensive Summaries
# 
# This will create:
# - metabolite_summary.csv
# - plot_inventory.csv
# - metabolite_report.txt
# - plot_coverage_report.txt
# - regional_breakdown.txt
# - summary_statistics.csv

# %%
# Create all summaries
output_files = create_comprehensive_summaries(
    results, 
    output_dir=None,  # Will use results.config.out_dir/summaries
    verbose=True
)

# Print paths to generated files
print("\nGenerated files:")
for name, path in output_files.items():
    print(f"  {name}: {path}")

# %% [markdown]
# ## Explore Metabolite Summary
# 
# Load and examine the metabolite summary CSV:

# %%
# Load the metabolite summary
met_summary = pd.read_csv(output_files['metabolite_summary'])

# Display first 10 rows
print("Metabolite Summary (first 10):")
display(met_summary.head(10))

# %%
# Filter for significant metabolites only
significant = met_summary[met_summary['Significant'] == True].copy()
significant = significant.sort_values('PValue_Secretion')

print(f"\nSignificant Metabolites (p < 0.05): {len(significant)}")
display(significant)

# %%
# Top metabolites by interaction score
top_by_score = met_summary.nlargest(20, 'InteractionScore')

print("\nTop 20 Metabolites by Interaction Score:")
display(top_by_score[['Metabolite', 'Source', 'Sink', 'InteractionScore', 
                       'Log2FC_Secretion', 'PValue_Secretion', 'Significant']])

# %% [markdown]
# ## Explore Plot Inventory
# 
# See which plots were generated for each metabolite:

# %%
# Load plot inventory
plot_inv = pd.read_csv(output_files['plot_inventory'])

print(f"Total plots generated: {len(plot_inv)}")
print("\nPlots by type:")
print(plot_inv['PlotType'].value_counts())

# %%
# See which metabolites have complete coverage (all plot types)
plot_coverage = plot_inv.groupby('Metabolite')['PlotType'].apply(set).reset_index()
plot_coverage['HasRegional'] = plot_coverage['PlotType'].apply(lambda x: 'Regional' in x)
plot_coverage['HasConsensus'] = plot_coverage['PlotType'].apply(lambda x: 'Consensus' in x)
plot_coverage['HasDifferential'] = plot_coverage['PlotType'].apply(lambda x: 'Differential' in x)
plot_coverage['Complete'] = (plot_coverage['HasRegional'] & 
                              plot_coverage['HasConsensus'] & 
                              plot_coverage['HasDifferential'])

print("\nPlot Coverage Summary:")
print(f"  Metabolites with Regional plots: {plot_coverage['HasRegional'].sum()}")
print(f"  Metabolites with Consensus plots: {plot_coverage['HasConsensus'].sum()}")
print(f"  Metabolites with Differential plots: {plot_coverage['HasDifferential'].sum()}")
print(f"  Metabolites with ALL plot types: {plot_coverage['Complete'].sum()}")

# %%
# Show metabolites with complete coverage
complete = plot_coverage[plot_coverage['Complete']]['Metabolite'].tolist()
print(f"\nMetabolites with complete plot coverage ({len(complete)}):")
for met in sorted(complete)[:20]:  # Show first 20
    print(f"  - {met}")
if len(complete) > 20:
    print(f"  ... and {len(complete) - 20} more")

# %% [markdown]
# ## View Text Reports
# 
# Read and display the generated text reports:

# %%
# Display metabolite report
print("=" * 80)
print("METABOLITE REPORT")
print("=" * 80)
with open(output_files['metabolite_report'], 'r') as f:
    print(f.read())

# %%
# Display plot coverage report
print("\n" + "=" * 80)
print("PLOT COVERAGE REPORT")
print("=" * 80)
with open(output_files['plot_coverage'], 'r') as f:
    print(f.read())

# %%
# Display regional breakdown
print("\n" + "=" * 80)
print("REGIONAL BREAKDOWN")
print("=" * 80)
with open(output_files['regional_breakdown'], 'r') as f:
    print(f.read())

# %% [markdown]
# ## Custom Analysis Examples
# 
# Here are some examples of custom analyses you can do with the summary data:

# %%
# Example 1: Metabolites with largest fold changes
print("Top 10 Metabolites with Largest Secretion Changes:")
print("-" * 60)

large_fc = met_summary.dropna(subset=['Log2FC_Secretion']).copy()
large_fc['AbsFC'] = large_fc['Log2FC_Secretion'].abs()
large_fc = large_fc.nlargest(10, 'AbsFC')

for _, row in large_fc.iterrows():
    direction = "Increased" if row['Log2FC_Secretion'] > 0 else "Decreased"
    sig_marker = "***" if row['Significant'] else ""
    print(f"{row['Metabolite']:20s} {direction:10s} "
          f"Log2FC={row['Log2FC_Secretion']:+.2f} "
          f"p={row['PValue_Secretion']:.4g} {sig_marker}")

# %%
# Example 2: Identify cell type pairs with most metabolite exchanges
print("\nTop Cell Type Pairs by Number of Metabolites:")
print("-" * 60)

pairs = met_summary.groupby(['Source', 'Sink']).size().reset_index(name='Count')
pairs = pairs.sort_values('Count', ascending=False).head(10)

for _, row in pairs.iterrows():
    print(f"{row['Source']:25s} → {row['Sink']:25s} : {row['Count']:3d} metabolites")

# %%
# Example 3: Distribution of interaction scores
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of interaction scores
axes[0].hist(met_summary['InteractionScore'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Interaction Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Interaction Scores')
axes[0].grid(alpha=0.3)

# Scatter: Interaction score vs p-value
valid = met_summary.dropna(subset=['PermutationPValue'])
axes[1].scatter(valid['InteractionScore'], -np.log10(valid['PermutationPValue']), 
                alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
axes[1].set_xlabel('Interaction Score')
axes[1].set_ylabel('-log10(p-value)')
axes[1].set_title('Interaction Score vs Significance')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results.config.out_dir, 'summaries', 'score_distributions.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# %%
# Example 4: Create a summary table for publication
publication_table = met_summary[met_summary['Significant']].copy()
publication_table = publication_table.sort_values('PValue_Secretion')

# Select and rename columns for publication
pub_cols = {
    'Metabolite': 'Metabolite',
    'Source': 'Source Cell Type',
    'Sink': 'Sink Cell Type',
    'InteractionScore': 'Interaction Score',
    'Log2FC_Secretion': 'Log₂ FC (Secretion)',
    'PValue_Secretion': 'P-value (Secretion)',
}

pub_table = publication_table[list(pub_cols.keys())].copy()
pub_table.columns = list(pub_cols.values())

# Round numeric columns
pub_table['Interaction Score'] = pub_table['Interaction Score'].round(4)
pub_table['Log₂ FC (Secretion)'] = pub_table['Log₂ FC (Secretion)'].round(2)
pub_table['P-value (Secretion)'] = pub_table['P-value (Secretion)'].apply(lambda x: f"{x:.2e}")

print("\nPublication-Ready Table (Significant Metabolites):")
display(pub_table)

# Save to CSV
pub_table.to_csv(os.path.join(results.config.out_dir, 'summaries', 'publication_table.csv'), 
                 index=False)
print(f"\nSaved to: {os.path.join(results.config.out_dir, 'summaries', 'publication_table.csv')}")

# %% [markdown]
# ## Export for Further Analysis
# 
# Save combined data for further analysis in other tools:

# %%
# Merge metabolite summary with plot inventory
merged = met_summary.merge(
    plot_inv.groupby('Metabolite').agg({
        'PlotType': lambda x: ', '.join(sorted(set(x))),
        'FilePath': 'count'
    }).rename(columns={'FilePath': 'NumPlots'}).reset_index(),
    on='Metabolite',
    how='left'
)

# Fill NaN values
merged['PlotType'] = merged['PlotType'].fillna('None')
merged['NumPlots'] = merged['NumPlots'].fillna(0).astype(int)

# Save
output_path = os.path.join(results.config.out_dir, 'summaries', 'complete_metabolite_data.csv')
merged.to_csv(output_path, index=False)

print(f"Complete metabolite data saved to: {output_path}")
print(f"Columns: {', '.join(merged.columns)}")

# %% [markdown]
# ## Summary
# 
# All summary files have been generated and can be found in the summaries directory.
# 
# Key outputs:
# 1. **CSV files** - Machine-readable data for further analysis
# 2. **Text reports** - Human-readable summaries
# 3. **Plots** - Visual distributions and relationships
# 
# You can now use these summaries for:
# - Manuscript preparation
# - Presentations
# - Further statistical analysis
# - Sharing with collaborators

# %%
print("\n" + "=" * 80)
print("SUMMARY GENERATION COMPLETE!")
print("=" * 80)
print(f"\nAll files saved to: {results.config.out_dir}/summaries/")
print("\nGenerated files:")
for name, path in output_files.items():
    print(f"  - {name}")
print("\nAdditional files:")
print(f"  - score_distributions.png")
print(f"  - publication_table.csv")
print(f"  - complete_metabolite_data.csv")
