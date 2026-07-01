# calce-vsoc-condition-aware-calibration
Reproducibility code for condition-aware V-SoC calibration and isotonic LUT construction using CALCE LiCoO2/graphite battery data.
# Condition-Aware V-SoC Calibration for CALCE LiCoO2/Graphite Cells

This repository contains the reproducibility code for the manuscript:

**Condition-Aware V-SoC Calibration for Li-ion Batteries: Analysis of a Controlled 4x2x3 CALCE Factorial Dataset**

The code supports preprocessing checks, representative-cell selection, condition-level voltage--SoC analysis, sparse gated correction, and isotonic LUT construction for condition-aware voltage-reference calibration.

## Scope

This repository contains code and lightweight reproducibility outputs. The large processed Excel input files are not redistributed in this repository due to file-size and data-hosting constraints.

## Required input files

The workflow expects the following local input files:

- `Processed_Outputv3_1.xlsx`
- `Processed_Outputv3_2.xlsx`
- `Processed_Outputv3_3.xlsx`

These are processed working files derived from the publicly available CALCE accelerated cycle-life dataset. The original CALCE data source and the associated Mendeley Data record are cited in the manuscript.

## Key reproducibility checks

Using the processed input files, the consistency-check pipeline reports:

- 2,358,728 processed records
- 181 unique sample numbers
- sample-number range: 1--187
- cycle-index range: 1--611

After representative-cell selection:

- 309,020 retained records
- 25 globally unique representative sample identifiers
- 24 / 24 nominal operating conditions covered

## Main outputs reproduced by the code

Sparse gated correction experiment:

- Both correction: Delta MAE = 2.052 mV
- Both correction: Delta RMSE = 1.434 mV

Measured-grid LUT evaluation:

- T+Rate MAE/RMSE = 0.028/0.118 V
- MAE/RMSE reduction = 81.6%/47.8%

## Repository structure

```text
notebooks/          Colab notebooks used in the manuscript workflow
outputs_minimal/    Lightweight CSV outputs for reproducibility checking
data_manifest/      Input-file requirements and checksum manifest
