# Reproducibility protocol

1. Record upstream workbook SHA256 before running the pipeline. The historical original workbook hashes are in `data_manifest/processed_output_sha256_manifest.csv`; do not rewrite them to force a match.
2. Use one Colab kernel for R2 → R2.2 (`notebooks/00_R2_to_R2_2_upstream.ipynb`). Save outputs to `/content/revision_R2_2_global/`.
3. Run R4B using the nominal-concordant cycle PKL and save `/content/outputs/R4B_broad_52_132/broad_52_132_cell_profiles.csv`.
4. Run notebook stages 01–05 in order. It deliberately refuses to overwrite an existing stage output or ZIP.
5. Compare output counts with `docs/VALIDATION_LEDGER.md`. Any mismatch requires inspection rather than parameter tuning.
6. Preserve stage ZIPs and calculate SHA256. ZIP container hashes may change after a rerun even when known scientific members remain bytewise identical. The provenance-safe v3 final packager verifies pinned scientific member fingerprints; metadata fields such as timestamp are checked as metadata. This proves equivalence of selected scientific members, **not all ZIP members**.
7. Run lightweight repository tests via `python -m unittest discover -s tests -v`; they check included published artifacts and code provenance but cannot stand in for a clean Colab re-execution.
8. For a publication release, provide code, documented input acquisition, and verified supplementary files. Preserve original raw CALCE source under its own attribution and license conditions.

## R1–R3 supplementary reproducibility

The separate research supplement dated 25 October 2026 is mapped in [`R123_REPRODUCIBILITY.md`](R123_REPRODUCIBILITY.md). R1/R2/R3 original source files are preserved unedited under `scripts/r123/`. Their outputs and original run manifests are under `supplementary/R123/`. The upstream stage archives and matching recorded source hashes are necessary for a fresh execution; the automated repository checks validate committed artifacts and do **not** rerun scientific fits.
