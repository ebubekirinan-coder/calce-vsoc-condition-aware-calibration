# Condition-dependent dynamic V–SoC profiles: CALCE reproducibility code

Code and selected verification outputs for the manuscript **“Condition-Dependent Dynamic Voltage–State-of-Charge Profiles in Lithium-Ion Batteries: Cell-Level Inference and Compact LUT Calibration.”**

## Publication scope

The investigated CALCE LiCoO₂/graphite data cover a 4 × 2 × 3 factorial design: temperature 10/25/45/60 °C, charge cut-off C/5 or C/40, and discharge 0.7C/1C/2C. The source Excel field `Rate` refers to charge cut-off, while `Charge_Rate` refers to discharge rate. R2.2 contains **83,045 nominal-concordant physical cycles from 180 cells** in 24 conditions. The canonical dynamic terminal-voltage LUT covers **SoC 0–90% inclusive**, with **24 × 91 Float32** values.

The primary analysis plans 81 SoC-band contrasts (75 inferentially eligible, 6 descriptive); 42 eligible 2C contrasts retain their sign and bootstrap confidence intervals excluding zero across four predefined model analyses at Cycle_Index 108. Condition-specific isotonic MAEs for the 52–132-cycle subset are **4.353667, 18.041400, 33.189926 mV** under LOCO / rate hold-out / temperature hold-out, respectively. These correspond to **95.80%, 81.86%, 62.85%** lower MAE than globally pooled isotonic representation on a common finite-node support. These are validation estimates, not real-time embedded-device guarantees.

### Repository status

- The source-aligned analyses and chosen outputs correspond to the previously validated October 2026 research results.
- The small files in `supplementary/`, `lut/`, `figures/` were copied byte-for-byte from the **independently audited 2026-10-18 submission ZIP** and verified against its manifest. The 2026-10-19 repackaging passed its own Colab QA; its new archive bytes are not silently substituted in this repository.
- A completely fresh Colab **end-to-end run of this assembled repository** is still a release gate. Source code is made available with its original per-cell provenance. Full scientific stage archives, raw workbooks, and 39 MB cycle store are not tracked in Git.
- Older representative-cell, sparse gated correction and 0–100 LUT code/output artifacts are preserved in `legacy/2026-07-representative-gated/` and are **not sources for current manuscript results**.

## Dependencies and data access

Python 3 with NumPy, pandas, scikit-learn, SciPy, statsmodels, matplotlib, openpyxl, threadpoolctl. Install via `python -m pip install -r requirements.txt` in a virtual environment or Colab runtime. See `data_manifest/input_files_required.txt`. Obtain `Processed_Outputv3_1.xlsx`, `_2.xlsx`, `_3.xlsx` from the processed CALCE dataset provenance described in the manuscript. These large workbooks are **not included**. Do not interpret missing workbooks as reusable or imputed data.

## Run order (Google Colab recommended)

1. Open `notebooks/00_R2_to_R2_2_upstream.ipynb` and run its cells top-to-bottom in **one Colab session**, uploading the three source Excel files. R2.2 depends on the Python functions and calibration variables created by R2. Do not execute the R2.2 `.py` alone in a fresh process.
2. Run `scripts/upstream/R4B_broad_52_132_clean.py` to produce the 52–132 window profiles. The script's default local output directory is `outputs/R4B_broad_52_132`; place the CSV at `/content/outputs/R4B_broad_52_132/broad_52_132_cell_profiles.csv` for the standard five-stage Colab notebook, or configure `VSOC_OUTPUT_DIR` accordingly.
3. Run `notebooks/01_to_05_scientific_rebuild.ipynb` **sequentially**. All five stages should run in the same runtime; the Stage 02 runner requires 01's ZIP and stages 04/05 require 03's ZIP. The code fails without overwriting a nonempty prior stage directory / ZIP.
4. After checking five scientific stage outputs, run `notebooks/final_package.ipynb` or `scripts/publication/build_final_submission.py`. This v3 packager verifies known scientific payload SHA256 values, key analyses, and Lut equivalence before producing a submission ZIP. A new ZIP may have a different container SHA even when audited scientific members match exactly.
5. Run `python -m unittest discover -s tests -v` to verify included publication results and original code fingerprints. Tests of packaged outputs are lightweight and do not replace a full CALCE data re-run.

**Stage dependency table**

| Stage | Purpose | Input | Standard archive |
|---|---|---|---|
| 01 | SoC-band 80% coverage and primary contrasts | R2.2 global PKL | `SoC_band_coverage_ConditionvCodes2.zip` |
| 02 | Age/sensitivity (4 models) | 01 ZIP | `SoC_age_sensitivity_ConditionvCodes2.zip` |
| 03 | Unified LUT, LOCO, rate, temperature hold-out | R2.2 full cells + R4B cells CSV | `SoC_unified_ConditionvCodes2.zip` |
| 04 | Fixed-prediction support audit | 03 ZIP | `SoC_coverage_audit_ConditionvCodes2.zip` |
| 05 | Five-representation common-node comparison | 03 ZIP | `SoC_step5_ablation_ConditionvCodes2.zip` |

## Scientific outputs in this repository

- `supplementary/contrast_results.csv`: four-model / age-time inferential contrast table (486 rows, includes 81 reference contrasts).
- `supplementary/validation_by_condition.csv`: validation by condition, methods, support, and scope.
- `supplementary/representation_gains.csv`: paired physical-cell bootstrap common-five-model MAE gains.
- `supplementary/Table_S1_81_planned_contrasts.csv` and `Table_S2_6_descriptive_contrasts.csv`: reference cycle 108; inferential fields are explicitly not applicable to the six descriptive contrasts.
- `lut/final_LUT_0_90_float32.npy` plus readable CSV matrix. No extrapolation to SoC 90–100.
- `figures/`: exactly the eight figures cited in the current manuscript (the 31-figure submission archive remains separately preserved).

## Traceability and cautions

- `data_manifest/archived_submission_manifest_2026-10-18.json`: approved scientific input fingerprints and SHA256 for source publication members.
- `data_manifest/audited_repo_payload_sha256.json`: strict SHA256 for selected publication members committed in this repository.
- `docs/CODE_PROVENANCE.md` and `docs/CODE_PROVENANCE.json`: original Colab source snapshot and stage-cell fingerprints.
- `docs/METHODS_TO_CODE.md`: manuscript methodology / executable block crosswalk.
- `docs/VALIDATION_LEDGER.md`: six validation MAEs, 42 contrasts, LUT and coverage milestones.
- Source Excel workbooks, intermediate PKL/ZIP archives, and other third-party data may require separate distribution permissions. The repository carries no raw cell measurement rows. Only load trusted R2.2 pickles.
- This repository's existing **MIT** license applies to its code; it does not override rights in source CALCE datasets.

## Citation

See `CITATION.cff`. Repository contributors and affiliations were retained from the prior revision; manuscript bibliographic details and a formal software release identifier should be updated when assigned, rather than invented in advance.
