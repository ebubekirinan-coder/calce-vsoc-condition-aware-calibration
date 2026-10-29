# SETA research supplementary: verified R1–R3 calculations (25 Oct 2026)

## Sources and versions
- Original journal three CSV tables (19 Oct 2026) are preserved in the archive root.
- R1/CR2/Satterthwaite results, R2/ridge and interpolation-axis results, and R3/compressed held-out LUT comparisons were extracted without numerical modification from the user-executed Colab ZIP `SETA_R1_R2_R3_RECOMPUTED_20261025T184824Z.zip`.
- `REPRO_MANIFEST.json` records source-input archive hashes, code versions, and environment. Scripts are provided under `scripts/`.
- R1 inputs: `SoC_band_coverage_ConditionvCodes2.zip` (includes `primary80_cycle_band_data.csv.gz`).
- R2/R3 inputs: `SoC_unified_ConditionvCodes2.zip` (includes cell-median profiles and validation source).
- This supplementary subset is not a self-contained raw-data archive. Re-execution needs the two above input ZIPs, which are not redistributed here.

## What to read
- `R1_summary_recomputed.csv`: CR1/CR2 BH counts at cycle 108; 81 planned tests and 75 eligible.
- `R1_CR2_all_models_972.csv`: model/contrast results including C5 exclusion and Satterthwaite df; generalized inverse needed for a perfect-leverage condition.
- `R2_compact_comparison.csv`: common-node comparison for six original LUT/ridge tests; `R2_axis_controls_summary.csv` contains factor-axis controls.
- `R2_B3_support_summary.csv`: model-specific support, to avoid comparing errors on different test domains.
- `R3_OOF_common_support.csv`: per representation, test and SoC band on common finite held-out nodes.
- `R3_OOF_own_support.csv`: model-specific finite-node support, not equivalent to the common-domain table.
- `R3_storage_budget.csv`: voltage-array payload only; excludes masks/metadata/runtime.

## Reproducibility caveats
- The original SoC coordinate depends on the endpoint discharge capacity of the same cycle and is retrospective.
- CR2 assumes identity working covariance on weighted observations within cells and uses generalized inverses for perfect leverage. The reference CR1 and physical-cell bootstrap remain primary.
- The two condition levels for charge cut-off do not permit an interior cut-off hold-out.
- Comparisons involving compression require common-node scoring because support can differ.
- 42/42 refers strictly to discharge-rate contrasts 2C–1C and 0.7C–2C (21 each), not to every temperature contrast at 2C.
- The analysis packages recompute from audited intermediate data, not from original CALCE records.
