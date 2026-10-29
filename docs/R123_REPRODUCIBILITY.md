# R1–R3 supplemental calculations — source and reproducibility guide

**Record of the 25 October 2026 computed supplement; this guide does not claim a fresh independent rerun.**

## Source integrity and contents

- Original source-code bytes: `scripts/r123/r1_cr2.py`, `r1_welch.py`, `r2_ridge.py`, `r2_postaudit.py`, `r2_axis.py`, `r3_heldout.py`. They are unchanged from the uploaded supplementary archive.
- Fourteen CSV and two JSON result/config files, together with the original supplement README, are in `supplementary/R123/`. The SHA-256 source-code fingerprint records and source-archive fingerprints are in `supplementary/R123/REPRO_MANIFEST.json`.
- Original journal/core CSV outputs `contrast_results.csv`, `validation_by_condition.csv`, `representation_gains.csv` remain only in the top-level `supplementary/` directory. Their bytes were checked against the supplement's archived root-level copies; no overwrite was required.
- Source environment: Python 3.13.16, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, statsmodels 0.15.0, threadpoolctl 3.7.0. The current repo's Python 3.11 GitHub Actions workflow checks artifacts and source syntax only.

## Required unpublished stage archives

| Research stage | Required audited ZIP | Recorded SHA-256 |
|---|---|---|
| R1 | `SoC_band_coverage_ConditionvCodes2.zip` | `b5d8099e62a3a1178d366db44c31930b2ce5cec5b8957a2845f4357e91ef9370` |
| R2/R3 | `SoC_unified_ConditionvCodes2.zip` | `ca2e0ebddbf41f4ec67052487efaa580ba5fd313a9b8fa68232333b138594985` |

These two archives **are not included** in the repository. Before running, compare each acquired ZIP's SHA-256 against the manifest; stop on mismatch. No preprocessing is reconstructed from the raw CALCE source by the R1–R3 scripts themselves.

## Execution in the historical Colab environment

The original scripts record absolute paths under `/content/SETA_R123_REPRO_20261025T184824Z/`, and some refer to an R2 source file inside `source_code/`. To rerun **unaltered source**, restore that directory layout, with `inputs/`, `source_code/`, and `results/` subdirectories, and place source files from `scripts/r123/` into `source_code/`. Do not silently edit source path literals or source fingerprint manifests.

Run in sequence:

1. `r1_cr2.py` against the band-coverage ZIP; writes R1 CR2 and deletion outputs.
2. `r1_welch.py` against the same ZIP and R1 outputs.
3. `r2_ridge.py` against the unified ZIP (supports `R2_SOURCE` and `R2_OUTPUT` overrides), then `r2_postaudit.py` against its fixed out-of-fold outputs.
4. `r2_axis.py` against the unified ZIP for log-rate / reciprocal-temperature controls.
5. `r3_heldout.py` against the unified ZIP to regenerate 13 fold-specific representations and compare common-node errors.

Save fresh logs, output file hashes, environment details, elapsed times, and any numerical deviations in a separate rerun report. The archived manifest refers to logs but does not bundle those log files; it records historical exits of zero. Do not rewrite original manifests to make a later run match.

## Verification available without stage archives

Run `python -m unittest discover -s tests -v`. In particular, `test_r123_artifacts.py` checks all six source-code fingerprints, 16 preserved output/config files against committed SHA-256 digests, key R1/R2/R3 results, and selected cross-checks against the primary validation CSVs. The separate `data_manifest/r123_payload_sha256.json` is a file-integrity ledger, not an independent result reproduction certificate.

## Interpretation

- Repeated observations within one physical cell are not independent experimental units. CR2 uses a working covariance and C5 leverage handling; CR1 and physical-cell bootstrap are primary.
- The 42 supported tests are **2C–1C and 0.7C–2C discharge-rate** contrasts; this does not include the temperature contrasts at 2C.
- Ridge and LUT comparisons use matched SoC nodes. Ridge may predict at additional nodes, which are excluded from common-support MAEs.
- Compression results are conditional on common scored nodes, and storage is **voltage-array payload only**, excluding support masks, metadata, runtime and working memory.
- Capacity-normalized V–SoC is a **retrospective offline reference**, not a validated real-time BMS SoC observer.
