"""R2.1 — Audit nominal sample-to-condition assignments in the R2 outputs.

This descriptive quality-assurance stage reads only the R2 CSV artifacts.
It does not modify source data, select reconstruction cycles, or change
any manuscript analysis. The five original CSV outputs are preserved.

Usage (Colab): run this cell/script after R2 output files have been saved.
Usage (CLI): SOC_R2_DIR=/path/to/revision_R2_final_profiles python script.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Match the CALCE sample assignment recorded in the frozen manuscript.
# Conditions 1–22 have eight documented IDs each; conditions 23–24
# have three each. Samples 180–184 and 188–192 have no documented map.
NOMINAL_SAMPLE_RANGES = tuple(
    [(8 * (cid - 1) + 1, 8 * cid, cid) for cid in range(1, 23)]
    + [(177, 179, 23), (185, 187, 24)]
)

CELL_COLUMNS = (
    "sample_int", "Condition_ID", "Temperature", "Rate", "Charge_Rate"
)
QC_COLUMNS = ("sample_int", "Condition_ID", "Cycle_Index", "usable")
OUTPUT_NAMES = (
    "conditions_per_sample.csv",
    "observed_sample_condition_pairs.csv",
    "nominal_condition_mismatches.csv",
    "cells_per_condition.csv",
    "sample_condition_cycle_support.csv",
)


def nominal_condition(sample_id: int) -> float | int:
    """Return the documented condition, or NaN for an unmapped sample ID."""
    sid = int(sample_id)
    for lower, upper, cid in NOMINAL_SAMPLE_RANGES:
        if lower <= sid <= upper:
            return cid
    return np.nan


def _join_sorted_ids(values: pd.Series) -> str:
    return ",".join(str(int(x)) for x in sorted(pd.unique(values)))


def _require_columns(path: Path, columns: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing R2 input artifact: {path}")
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(columns) - set(header.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")


def _normalize_boolean(values: pd.Series) -> pd.Series:
    """Read the documented R2 usable flag safely across CSV dtype inference."""
    normalized = values.astype("string").str.strip().str.lower()
    accepted = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(accepted)
    if bool(invalid.any()):
        bad = values[invalid].head(5).tolist()
        raise ValueError(f"Unknown values in R2 cycle QC 'usable' column: {bad}")
    return normalized.map(accepted).astype(bool)


def run_audit(r2_directory: str | Path) -> dict[str, pd.DataFrame]:
    """Recreate the original five R2.1 audit CSVs without altering inputs."""
    r2_dir = Path(r2_directory).expanduser()
    cell_path = r2_dir / "r2_cell_profiles.csv"
    qc_path = r2_dir / "r2_cycle_qc.csv"
    out = r2_dir / "R2_1_design_audit"

    _require_columns(cell_path, CELL_COLUMNS)
    _require_columns(qc_path, QC_COLUMNS)

    cell = pd.read_csv(cell_path, usecols=list(CELL_COLUMNS))
    qc = pd.read_csv(qc_path, usecols=list(QC_COLUMNS))
    qc["usable"] = _normalize_boolean(qc["usable"])

    for label, frame in (("cell profiles", cell), ("cycle QC", qc)):
        for key in ("sample_int", "Condition_ID"):
            key_numeric = pd.to_numeric(frame[key], errors="coerce")
            if key_numeric.isna().any() or not (key_numeric % 1 == 0).all():
                raise ValueError(f"Invalid {key} in {label}")
            frame[key] = key_numeric.astype("int64")

    pairs = (
        cell[list(CELL_COLUMNS)]
        .drop_duplicates()
        .sort_values(["sample_int", "Condition_ID"])
        .reset_index(drop=True)
    )
    # A sample-condition pair must have a unique factor assignment in R2.
    uniqueness = pairs.groupby(["sample_int", "Condition_ID"], observed=True).size()
    if (uniqueness > 1).any():
        raise ValueError("Conflicting factor metadata for a sample-condition pair")

    sample_condition_count = (
        pairs.groupby("sample_int", as_index=False, observed=True)
        .agg(
            n_conditions=("Condition_ID", "nunique"),
            observed_conditions=("Condition_ID", _join_sorted_ids),
        )
    )
    multi = sample_condition_count.loc[
        sample_condition_count["n_conditions"] > 1
    ].copy()

    pairs["Nominal_Condition_ID"] = pairs["sample_int"].map(nominal_condition)
    pairs["matches_nominal"] = (
        pairs["Condition_ID"] == pairs["Nominal_Condition_ID"]
    )
    known_nominal = pairs.loc[pairs["Nominal_Condition_ID"].notna()]
    unknown_nominal = pairs.loc[pairs["Nominal_Condition_ID"].isna()]
    matches = known_nominal.loc[known_nominal["matches_nominal"]]
    mismatches = known_nominal.loc[~known_nominal["matches_nominal"]].copy()

    condition_cells = (
        pairs.groupby(
            ["Condition_ID", "Temperature", "Rate", "Charge_Rate"],
            as_index=False,
            observed=True,
        )
        .agg(
            n_observed_cells=("sample_int", "nunique"),
            observed_samples=("sample_int", _join_sorted_ids),
        )
        .sort_values("Condition_ID")
    )

    usable = qc.loc[qc["usable"]].copy()
    pair_cycle_support = (
        usable.groupby(["sample_int", "Condition_ID"], as_index=False, observed=True)
        .agg(
            n_usable_cycles=("Cycle_Index", "nunique"),
            cycle_min=("Cycle_Index", "min"),
            cycle_median=("Cycle_Index", "median"),
            cycle_max=("Cycle_Index", "max"),
        )
    )
    # Every accepted physical cycle must belong to a reconstructed cell profile.
    paired_keys = pairs[["sample_int", "Condition_ID"]]
    missing_pairs = pair_cycle_support.merge(
        paired_keys, on=["sample_int", "Condition_ID"], how="left", indicator=True
    )
    if (missing_pairs["_merge"] == "left_only").any():
        raise ValueError("Accepted QC cycles lack a corresponding R2 cell profile")

    pair_audit = pairs.merge(
        pair_cycle_support,
        on=["sample_int", "Condition_ID"],
        how="left",
    )

    out.mkdir(parents=True, exist_ok=True)
    tables = {
        OUTPUT_NAMES[0]: sample_condition_count,
        OUTPUT_NAMES[1]: pairs,
        OUTPUT_NAMES[2]: mismatches,
        OUTPUT_NAMES[3]: condition_cells,
        OUTPUT_NAMES[4]: pair_audit,
    }
    for filename, frame in tables.items():
        frame.to_csv(out / filename, index=False)

    print("=" * 70)
    print("R2.1 DESIGN CONSISTENCY AUDIT COMPLETE")
    print("=" * 70)
    print(f"Observed sample-condition pairs: {len(pairs):,}")
    print(f"Unique physical sample IDs: {pairs['sample_int'].nunique():,}")
    print(f"Observed conditions: {pairs['Condition_ID'].nunique():,}")
    print(f"Samples assigned to multiple conditions: {len(multi):,}")
    print(f"Nominally concordant pairs: {len(matches):,}")
    print(f"Mismatched documented pairs: {len(mismatches):,}")
    print(f"Unmapped sample-condition pairs: {len(unknown_nominal):,}")
    fraction = float(known_nominal["matches_nominal"].mean()) if len(known_nominal) else np.nan
    print(f"Nominal match fraction among mapped pairs: {fraction:.6f}")
    print(f"R2 usable cycle records audited: {len(usable):,}")
    print("\nSamples assigned to multiple conditions:")
    print(multi.head(40).to_string(index=False) if not multi.empty else "None")
    print("\nFirst 50 nominal assignment mismatches:")
    columns = (
        "sample_int", "Nominal_Condition_ID", "Condition_ID",
        "Temperature", "Rate", "Charge_Rate",
    )
    print(mismatches[list(columns)].head(50).to_string(index=False) if not mismatches.empty else "None")
    print("\nObserved cells per condition:")
    print(condition_cells.to_string(index=False))
    print(f"\nOutput directory: {out}")
    print("Source R2 artifacts were not modified.")
    return tables


if __name__ == "__main__":
    run_audit(
        os.environ.get(
            "SOC_R2_DIR",
            "/content/revision_R2_final_profiles"
            if Path("/content").is_dir() else "revision_R2_final_profiles",
        )
    )
