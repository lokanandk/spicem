#!/usr/bin/env python3
"""
SPATIAL METABOLIC FLUX ANALYSIS - STANDALONE RUNNER
"""

import os
import sys
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from spatialmetabolicanalysis import (
        AnalysisConfig,
        run_analysis_pipeline,
        save_results_to_csv,
        PlotConfig,
        generate_all_plots,
    )

    BASEDIR = "/path/to/data"
    OUTDIR = "/path/to/output"

    CONDITIONS = ["Sample_Healthy", "Sample_Injured"]
    REGIONS = [1, 2, 3, 4, 5]

    config = AnalysisConfig(
        base_dir=BASEDIR,
        out_dir=OUTDIR,
        conditions=CONDITIONS,
        regions=REGIONS,
        diffusion_coefficient=0.1,
        k_neighbors=4,
        grid_resolution=80,
        gaussian_sigma=1.2,
        permutation_iters=500,
        min_coupling_strength=0.001,
        top_n_plots=20,
        num_workers=10,
        grid_pattern="communities_hexagonal_Region_{}.json",
        flux_folder="exchange_fluxes",
    )

    plotcfg = PlotConfig(
        # You can tune these:
        network_top_edge_ratio=0.1,
        network_label_top_edges=10,
        retain_top_per_metabolite=2,
        retain_score_quantile=0.90,
        retain_region_local_topk=2,
        arrow_mutation_scale=18.0,
        remove_bidirectional_loops=True,
    )

    os.makedirs(OUTDIR, exist_ok=True)

    results = run_analysis_pipeline(config, verbose=True)
    save_results_to_csv(results, OUTDIR, verbose=True)
    generate_all_plots(results, cfg=plotcfg, verbose=True)

    print("\nDONE")
