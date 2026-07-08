<p align="center">
  <img src="assets/spicem_logo.svg" alt="SPICEM logo" width="640">
</p>

<p align="center"><strong>SP</strong>atial <strong>I</strong>nference of <strong>C</strong>ellular <strong>E</strong>nvironments via <strong>M</strong>etabolic Modeling</p>

<p align="center"><em>Work in progress — manuscript in preparation.</em></p>

## About

SPICEM is a spatial metabolic modeling pipeline for tissue transcriptomics. Given
spatially resolved single-cell/spot data, it builds local multi-cell community
metabolic models and analyzes the resulting flux solutions along two
complementary axes:

- **Exchange fluxes** — metabolite secretion/uptake between spatially proximal
  cells, used to map cell-cell metabolic interactions and how they shift across
  clinical groups.
- **Intracellular fluxes** — each cell's own internal metabolic activity
  (individual reactions and, via reference-model subsystem annotations,
  whole pathways), used to characterize cell-type-specific metabolic phenotypes
  and their spatial organization.

This repository accompanies ongoing work applying the pipeline to human kidney
tissue, relating spatial metabolic dysfunction to disease-relevant cell types.
The associated manuscript is currently in preparation and unpublished; this
repo is shared to document active development.

*Spatial metabolic modelling maps GWAS risk to cell-type-specific dysfunction
in diabetic kidney disease*

## Usage

The pipeline runs as a set of Python modules driven from a Jupyter notebook
(`spicem_human_cohort.ipynb`), with MATLAB/COBRA scripts (`model_building/`)
handling community model construction and flux extraction upstream.

> **Note:** `model_building/` contains placeholder files in this shared copy
> of the repository — filenames and each script's documented purpose are
> preserved so the pipeline's structure is clear, but the MATLAB
> implementations are withheld while the associated manuscript is in
> preparation. Contact the author for access.

```python
from pipeline import AnalysisConfig, run_analysis_pipeline

cfg = AnalysisConfig(base_dir="<path-to-region-data>", conditions=["Sample_Condition"])
results = run_analysis_pipeline(cfg)
```

See the notebook for the full cohort workflow (per-patient runs, cohort
aggregation, statistical comparisons, and figure generation).

## Status & license

Pre-publication, active development. All rights reserved — no license is
granted for reuse or redistribution at this time. This will be revisited upon
publication.

## Contact

Lokanand Koduru, Senior Scientist at the Genome Institute of Singapore (GIS),
Agency for Science, Technology and Research (A*STAR).
