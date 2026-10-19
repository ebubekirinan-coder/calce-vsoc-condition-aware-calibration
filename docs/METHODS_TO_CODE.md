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
