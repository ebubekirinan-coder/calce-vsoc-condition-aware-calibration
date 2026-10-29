# Frozen validation ledger (October 2026)

The following values come from the approved Colab execution logs and the independently audited 2026-10-18 submission archive. They are test fixtures for the reproduced CALCE experiment, not general numerical truths about arbitrary batteries.

## R2.2 and SoC-band inference
- Nominal-concordant cycles: **83,045**; physical cells: **180**; conditions: **24**.
- Primary >=80% coverage band counts: [0,20) 79,218 cycles; [20,60) 79,218; [60,90) 79,223; 173 cells per band.
- 81 contrasts planned, 75 inference eligible, 6 descriptive.
- Of 48 planned 2C-related contrasts, 42 eligible contrasts retained a consistent direction and bootstrap %95 confidence intervals excluding zero across all four models at cycle 108.

## Unified estimator (mean physical-cell error, mV)

| Source history | LOCO | Held-out discharge rate | Held-out temperature |
|---|---:|---:|---:|
| Full | 5.240437 | 19.531124 | 32.141306 |
| Cycle_Index 52–132 | 4.353667 | 18.041400 | 33.189926 |

Window evaluated node coverage: LOCO 15,639/15,639; rate 5,096/5,642; temperature 8,112/8,736. This denominator is the number of finite validation targets, not 24×91 grid positions.

## Representation comparison, 52–132 common-five-node support

| Protocol | Global-isotonic MAE (mV) | Condition-isotonic MAE (mV) | Relative reduction | Bootstrap absolute gain and %95 CI (mV) |
|---|---:|---:|---:|---|
| LOCO | 103.688425 | 4.353667 | 95.8012% | 99.334759 [98.288425,100.305394] |
| Rate hold-out | 99.451909 | 18.041400 | 81.8592% | 81.410509 [79.252341,83.302963] |
| Temperature hold-out | 89.328318 | 33.189926 | 62.8450% | 56.138392 [54.858995,57.583692] |

Float32 LUT: 24 rows × 91 SoC nodes, 2,184 finite values. Final Unified vs five-representation LUT node difference: 0.0 mV in the approved reference rerun.

Scientific test output identity: `data_manifest/audited_repo_payload_sha256.json`.

## Additional independently checked R1–R3 supplement (25 October 2026)

- R1 at cycle 108: four models each with 75 eligible contrasts; CR2/BH supported counts are **66** (reference), **68** (condition slopes), **67** (cell balanced), and **73** (52–132 window). Across each model, all 42 eligible 2C discharge-rate contrasts remain supported; 2C thermal comparisons are a distinct family.
- R2 nodewise seven-coefficient ridge: the condition-specific isotonic LUT has lower matched-node mean cell MAE in all six full-history and 52–132 LOCO/rate/temperature validations. The full-history temperature interpolation-axis control reduced MAE **32.14 → 28.70 mV** using reciprocal absolute temperature.
- R3 exploratory 52–132 LOCO common-node results: original Float32 LUT **4.260269 mV / 8.531250 KiB**; 2-point Float32 PCHIP **4.3065 mV / 4.3125 KiB**; 1-point Float16 **4.301049 mV / 4.265625 KiB**. Payload excludes support masks/metadata/runtime. Full model/validation comparisons must not be inferred from this one row.
- Sources: `supplementary/R123/R1_summary_recomputed.csv`, `R2_compact_comparison.csv`, `R2_axis_controls_summary.csv`, `R3_LOCO_compact_comparison.csv`, `R3_storage_budget.csv`, and `REPRO_MANIFEST.json`.
- Validation scope: stored output consistency and original code hash integrity. The source ZIPs are absent; scientific rerun not yet independently completed.
