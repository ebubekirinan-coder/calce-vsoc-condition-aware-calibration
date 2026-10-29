# Methods-to-code crosswalk

| Manuscript component | Executable source | Safeguards |
|---|---|---|
| Three processed CALCE source workbooks and sample metadata QA | `notebooks/00_R2_to_R2_2_upstream.ipynb`; `scripts/upstream/R2_cycle_reconstruction_clean.py` | Negative Step 5 plateau, current tolerance max(0.05 A,3%), no cross-gap interpolation over 10 SoC points |
| Global nominal-concordant, cross-file physical-cycle reconstruction | `scripts/upstream/R2_2_global_reconstruction_clean.py` (same-session after R2) | Duplicate audit and exactly 83,045 accepted nominal cycles, 180 cells |
| Broader 52–132 cycle-window sensitivity | `scripts/upstream/R4B_broad_52_132_clean.py` | 13,181 nominal cycles in 52–132 window; cell median within cell |
| Primary 80% SoC band and inferential contrasts | `scripts/analyses/01_band_coverage.py` | 81 planned, 75 inference-eligible, 6 descriptive |
| 4 age/cell sensitivity models | `scripts/analyses/02_age_sensitivity.py` | CR1 physical-cell clustering and 2,000 condition-wise cell bootstrap |
| Unified LUT and leakage-safe LOCO/rate/temp validation | `scripts/analyses/03_unified_validation.py` | Unified weighted-isotonic estimator; five paired nodes minimum; target availability explicitly tracked |
| Fixed predictions, node-support threshold sensitivity | `scripts/analyses/04_support_audit.py` | Does not refit models; supports 1/2/3 physical cell node rules |
| Five-model ablation and paired bootstrap | `scripts/analyses/05_five_representations.py` | The same finite physical-cell/SoC nodes for each model; 2000 resamples; LUT comparison |
| Manuscript figure, supplementary, LUT exports | `scripts/publication/build_final_submission.py` | Verifies audited input payloads and 42 robust 2C contrasts |
| Submission ZIP independent integrity | `scripts/publication/audit_final_submission.py` | Hash-checked payloads and manifest consistency |

The historical R9/R10/R11 quality-assurance utilities remain under `scripts/upstream/` for audit reference. They are not substitutes for the canonical Unified output.

## Post hoc R1–R3 manuscript extensions

| Manuscript method / result | Preserved executable script | Verified supporting results | Limitation |
|---|---|---|---|
| CR2/Satterthwaite, BH-81, C5 generalised-inverse sensitivity | `scripts/r123/r1_cr2.py` | `supplementary/R123/R1_CR2_all_models_972.csv`; `R1_summary_recomputed.csv`; `R1_run_config.json` | 42/42 refers specifically to 2C–1C and 0.7C–2C discharge-rate contrasts |
| Physical-cell Welch and leave-one-cell-out sensitivity | `scripts/r123/r1_welch.py` and `r1_cr2.py` | `R1_cell_level_Welch_secondary.csv`; `R1_small_cell_leave_one_out.csv` | No independent-cell replication created |
| Nodewise ridge with three-fold inner cell CV and external hold-outs | `scripts/r123/r2_ridge.py` | `R2_compact_comparison.csv`; `R2_B3_support_summary.csv`; `R2_B3_run_manifest.json` | Comparison limited to specified seven-coefficient ridge |
| Fixed OOF paired-cell bootstrap and coefficient packaging | `scripts/r123/r2_postaudit.py` | `R2_B3_conditional_paired_bootstrap.csv` | Fixed-prediction conditional bootstrap |
| Log-rate and reciprocal-temperature LUT interpolation | `scripts/r123/r2_axis.py` | `R2_axis_controls_summary.csv` | No inferred electrochemical mechanism |
| 13 held-out LUT storage/interpolation variants | `scripts/r123/r3_heldout.py` | `R3_OOF_common_support.csv`; `R3_OOF_own_support.csv`; `R3_LOCO_compact_comparison.csv`; `R3_storage_budget.csv` | Voltage-array bytes only, preserve support mask |

Full file lineage, original source hashes, and missing input dependencies are documented in `supplementary/R123/REPRO_MANIFEST.json` and `docs/R123_REPRODUCIBILITY.md`. The stage-01–05 provenance manifest is unchanged.
