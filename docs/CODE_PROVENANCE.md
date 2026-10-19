# Source code provenance

The uploaded `revisedcalce.py` file is an **export of consecutively run Colab cells**. Its original bytes are preserved as `docs/source_snapshot/revisedcalce_colab_export.txt` (SHA256: `e1e19f12f7327ac5c558d41e8075767f8207c3699d59435855805f67a92c1426`). This export contains multiple entry points and mid-file `from __future__` imports; it is not a standalone executable Python program.

Five publication analysis scripts under `scripts/analyses/` consist of the exact cell 2 shared stage scaffold followed by the exact scientific cell for that stage from `notebooks/01_to_05_scientific_rebuild.ipynb`. Source cell SHA256 fingerprints are recorded in `docs/CODE_PROVENANCE.json` and checked in `tests/test_repository_artifacts.py`. No scientific functions were modified in this export step.

The upstream R2 and R2.2 scientific script bodies are byte-for-byte copies of the verified source scripts. Their intended same-session execution is provided in `notebooks/00_R2_to_R2_2_upstream.ipynb`.

**Verification levels**: Scientific calculations were executed and reviewed in prior Colab runs, and the 2026-10-18 publication ZIP was independently audited. The assembled GitHub repository scripts have passed local syntax/structure and released-artifact identity checks, but a clean, end-to-end Colab run of this assembled repository is a remaining release gate. The separately reported 2026-10-19 ZIP passed its Colab final QA; its actual bytes are not bundled here.
