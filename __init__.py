#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS PACKAGE
"""

# Import from core
from .core import (
    load_flux_txt,
    load_region_data,
    aggregate_fluxes,
    find_enriched_metabolites,
    compute_spatial_coupling_matrix,
    analyze_all_interactions,
    compute_potential_flow_field,
    process_region,
    permutation_test_vectorized,
    fishers_method,
    combine_permutation_results,
    compare_conditions,
    SpatialAnalysisConfig,
    SpatialAnalysisResult
)

# Import from pipeline - THIS IS THE CORRECT AnalysisConfig TO USE!
from .pipeline import (
    AnalysisConfig,              # Import the REAL AnalysisConfig from pipeline
    AnalysisResults,             # Import the REAL AnalysisResults from pipeline
    run_analysis_pipeline,
    save_results_to_csv,
    run_coassociation
)

# Import from plotting
from .plotting import (
    PlotConfig,
    generate_all_plots,
    plot_consensus_streamlines,
    plot_differential_streamlines,
    plot_network_diagram,
    plot_coupling_heatmap,
    plot_flux_balance_comparison,
    plot_metabolite_heatmap,
    plot_volcano,
    plot_significant_metabolites_bars,
    plot_top_metabolites_bar,
)

# Import intracellular flux analysis (complements the exchange-flux pipeline)
from .intracellular import (
    IntracellularConfig,
    IntracellularResults,
    load_intracellular_txt,
    aggregate_intracellular_fluxes,
    find_enriched_reactions,
    compare_reactions_between_conditions,
    build_celltype_reaction_matrix,
    link_exchange_to_intracellular,
    build_spatial_flux_frame,
    load_reaction_subsystems,
    build_subsystem_activity_table,
    find_enriched_subsystems,
    build_subsystem_celltype_matrix,
    build_subsystem_prevalence_matrix,
    build_spatial_subsystem_frame,
    compare_subsystems_between_conditions,
    compare_celltype_subsystems_between_conditions,
    build_subsystem_differential_matrices,
    build_metabolite_activity_table,
    find_enriched_metabolite_activity,
    build_metabolite_activity_celltype_matrix,
    build_spatial_metabolite_activity_frame,
    compare_metabolite_activity_between_conditions,
    compare_celltype_metabolite_activity_between_conditions,
    build_metabolite_activity_differential_matrices,
    build_metabolite_pathway_allocation_table,
    compute_pathway_allocation,
    rank_metabolites_by_allocation_shift,
    run_intracellular_analysis,
)

# Import intracellular plotting helpers
from .intracellular_plots import (
    plot_celltype_reaction_heatmap,
    plot_enrichment_bar,
    plot_mechanistic_links,
    plot_condition_volcano,
    plot_condition_top_reactions,
    plot_spatial_reaction,
    plot_spatial_pathway,
    plot_subsystem_enrichment_dot,
    plot_subsystem_heatmap,
    plot_subsystem_signature_bars,
    plot_spatial_subsystem,
    plot_subsystem_differential_heatmap,
    plot_metabolite_pathway_allocation,
    plot_allocation_shift_ranking,
)

# Import exchange-flux per-cell-type secretion/uptake analysis
from .exchange_celltype_analysis import (
    GROUP_COLORS,
    build_secretion_uptake_table,
    top_metabolites_for_celltype,
    celltype_metabolite_by_group,
    rank_celltypes_by_activity,
    top_metabolites_overall,
    build_celltype_metabolite_matrix,
    compare_celltype_metabolite_between_conditions,
    build_differential_matrices,
)

# Import exchange-flux per-cell-type plotting helpers
from .exchange_celltype_plots import (
    plot_celltype_secretion_uptake,
    plot_top_metabolites_celltype_heatmap,
    plot_differential_celltype_heatmap,
)

# Import intracellular flux clinical-covariate (eGFR/Fibrosis) correlation
from .intracellular_clinical import (
    build_patient_entity_matrix,
    build_patient_celltype_entity_matrices,
    correlate_with_clinical,
    correlate_celltype_with_clinical,
    plot_clinical_correlation_bar,
    plot_clinical_correlation_scatter,
)

# Import enhanced analysis functions
from .enhanced_analysis import (
    identify_bidirectional_loops,
    analyze_network_topology,
    analyze_metabolite_dominance,
    detect_leukotriene_signaling,
    generate_therapeutic_targets,
    create_enhanced_summary,
)

# Import enhanced plotting functions
from .enhanced_plots import (
    plot_bidirectional_loops,
    plot_network_comparison,
    plot_metabolite_dominance_heatmap,
    plot_leukotriene_localization,
    plot_therapeutic_targets_summary,
    generate_all_enhanced_plots,
)

# No aliases needed - AnalysisConfig and AnalysisResults are already imported from pipeline!
