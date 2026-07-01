# Condition-Aware V-SoC Calibration for CALCE LiCoO2/Graphite Cells

This repository contains the reproducibility code for the manuscript:

**Condition-Aware V-SoC Calibration for Li-ion Batteries: Analysis of a Controlled 4x2x3 CALCE Factorial Dataset**

The code supports dataset consistency checks, representative-cell selection, condition-family voltage--SoC visualization, statistical analysis, sparse gated correction, and condition-indexed isotonic LUT construction for condition-aware voltage-reference calibration.

## Scope

This repository contains code and lightweight reproducibility outputs. The large processed Excel input files are not redistributed in this repository due to file-size and data-hosting constraints.

The original CALCE battery dataset and the associated Mendeley Data record are cited in the manuscript. The processed intermediate files generated in this study are available from the corresponding author upon reasonable request.

## Required input files

The workflow expects the following local input files:

- `Processed_Outputv3_1.xlsx`
- `Processed_Outputv3_2.xlsx`
- `Processed_Outputv3_3.xlsx`

These files should be placed in the repository root when running the notebooks locally or in the configured project root when running in Google Colab.

## Key reproducibility checks

Using the processed input files, the consistency-check pipeline reports:

- 2,358,728 processed records
- 181 unique sample numbers
- sample-number range: 1--187
- cycle-index range: 1--611

After block-aware representative-cell selection:

- 309,055 retained records before discharge-only filtering
- 25 globally unique representative sample identifiers
- 24 / 24 nominal operating conditions covered

After restricting the workflow to discharge-only records:

- 309,020 discharge-only analysis records

## Main outputs reproduced by the code

### Section 6: Sparse gated correction experiment

The sparse gated correction experiment applies conservative thermal and rate-history corrections only in measured, non-negligible operating contexts.

Overall sparse gated correction results:

- Baseline MAE/RMSE = 0.130499 / 0.164297 V
- Thermal MAE/RMSE = 0.129149 / 0.163376 V
- RateHist MAE/RMSE = 0.129797 / 0.163788 V
- Both MAE/RMSE = 0.128447 / 0.162864 V

Ablation versus baseline:

- Thermal Delta MAE = 1.350 mV
- Thermal Delta RMSE = 0.921 mV
- RateHist Delta MAE = 0.702 mV
- RateHist Delta RMSE = 0.510 mV
- Both Delta MAE = 2.052 mV
- Both Delta RMSE = 1.434 mV
- Both correction coverage = 510 corrected evaluation points

### Section 7: Measured-grid isotonic LUT evaluation

The condition-indexed isotonic LUT atlas is evaluated as a non-held-out measured-grid reconstruction. These results quantify in-grid reconstruction fidelity, not predictive generalization to unseen cells, cycles, temperatures, or operating tuples.

Overall measured-grid LUT evaluation:

- Baseline MAE/RMSE = 0.125735 / 0.161573 V
- Thermal MAE/RMSE = 0.089544 / 0.125924 V
- T+Rate MAE/RMSE = 0.010742 / 0.035762 V

Ablation versus baseline:

- Thermal Delta MAE = 36.19 mV
- Thermal Delta RMSE = 35.65 mV
- Thermal MAE/RMSE reduction = 28.78% / 22.06%
- T+Rate Delta MAE = 114.99 mV
- T+Rate Delta RMSE = 125.81 mV
- T+Rate MAE/RMSE reduction = 91.46% / 77.87%

Bootstrap diagnostics:

- Thermal Delta MAE 95% CI = [34.39, 38.18] mV
- T+Rate Delta MAE 95% CI = [112.93, 117.25] mV
- Paired Wilcoxon signed-rank tests give p < 1e-6 for both calibrated variants.

## Repository structure

```text
calce-vsoc-condition-aware-calibration/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_dataset_consistency_check.ipynb
│   ├── 02_representative_cell_selection.ipynb
│   ├── 03_condition_family_visualization.ipynb
│   ├── 04_section5_statistical_analysis.ipynb
│   ├── 05_section6_calibration_evaluation.ipynb
│   └── 06_section7_lut_profile_evaluation.ipynb
├── outputs_minimal/
│   ├── representative_condition_coverage.csv
│   ├── overall_metrics.csv
│   ├── bandwise_metrics.csv
│   ├── ablation_vs_baseline.csv
│   ├── bandwise_delta_mae_vs_baseline_mV.csv
│   ├── lut_specification.csv
│   ├── overall_metrics_from_pred.csv
│   ├── sec7_ablation_vs_baseline.csv
│   ├── sec7_bandwise_delta_mae_mV.csv
│   └── sec7_bootstrap_wilcoxon.csv
└── data_manifest/
    ├── input_files_required.txt
    └── processed_output_sha256_manifest.csv
