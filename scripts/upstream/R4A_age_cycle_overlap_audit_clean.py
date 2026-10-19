"""R4A: Audit cycle-index imbalance and overlap in nominal-concordant CALCE cycles.

This metadata-only analysis does not filter/reconstruct cycles, fit voltage models,
or select an inferential analysis window. It reads the R2.2 global cycle store
and writes five descriptive CSV tables with the original R4A schemas.

Environment overrides:
    SOC_R22_DIR:  directory containing global_cycle_profiles.pkl
    SOC_OUTPUT_DIR: fallback to the same directory (STEP 3 convention)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "sample_int", "Condition_ID", "Cycle_Index", "Temperature",
    "Rate", "Charge_Rate", "matches_nominal",
]
N_CONDITIONS = 24


def contiguous_windows(values):
    """Return inclusive (start, end, number_of_indices) consecutive runs."""
    indices = sorted({int(value) for value in values})
    if not indices:
        return []
    result = []
    start = previous = indices[0]
    for current in indices[1:]:
        if current != previous + 1:
            result.append((start, previous, previous - start + 1))
            start = current
        previous = current
    result.append((start, previous, previous - start + 1))
    return result


def window_table(coverage):
    """Preserve the original three common-index criteria and CSV column names."""
    criteria = {
        "all_24_conditions": "all_24_conditions",
        "all24_min_2_cells": "all24_with_min2cells",
        "all24_min_3_cells": "all24_with_min3cells",
    }
    records = []
    for label, column in criteria.items():
        for start, end, width in contiguous_windows(
            coverage.loc[coverage[column], "Cycle_Index"]
        ):
            records.append({
                "criterion": label,
                "start_cycle": start,
                "end_cycle": end,
                "width_cycles": width,
            })
    columns = ["criterion", "start_cycle", "end_cycle", "width_cycles"]
    result = pd.DataFrame.from_records(records, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["criterion", "width_cycles"], ascending=[True, False]
        ).reset_index(drop=True)
    return result


def run_age_cycle_audit(cycle_file: Path, output_dir: Path):
    """Return the five QA tables and save them without changing R2.2 inputs."""
    cycle_file = Path(cycle_file)
    output_dir = Path(output_dir)
    if not cycle_file.is_file():
        raise FileNotFoundError(f"R2.2 global cycle store not found: {cycle_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pickle loads the whole table; immediately discard unused voltage columns.
    source = pd.read_pickle(cycle_file)
    missing = [name for name in REQUIRED_COLUMNS if name not in source.columns]
    if missing:
        raise KeyError(f"R2.2 global store is missing columns: {', '.join(missing)}")
    d = source.loc[source["matches_nominal"].eq(True), REQUIRED_COLUMNS].copy()
    del source
    if d.empty:
        raise ValueError("R2.2 contains no nominal-concordant cycles")

    for name in ("sample_int", "Condition_ID", "Cycle_Index"):
        numeric = pd.to_numeric(d[name], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"Invalid or nonintegral nominal metadata: {name}")
        d[name] = numeric.astype("int32")

    if d["Condition_ID"].nunique() != N_CONDITIONS:
        raise ValueError("Expected the full 24-condition nominal-concordant design")
    if not d["Condition_ID"].between(1, N_CONDITIONS).all():
        raise ValueError("Condition IDs must be 1 through 24")

    cycle_key = ["sample_int", "Condition_ID", "Cycle_Index"]
    if d.duplicated(subset=cycle_key).any():
        raise ValueError("Duplicate physical-cell/condition/cycle keys in R2.2 store")

    print(f"Nominal-concordant cycles: {len(d):,}")
    print(f"Cells: {d['sample_int'].nunique()}")
    print(f"Conditions: {d['Condition_ID'].nunique()}")

    # Quantiles are calculated across cycle observations, as in original R4A.
    condition_age = (
        d.groupby(
            ["Condition_ID", "Temperature", "Rate", "Charge_Rate"],
            as_index=False, observed=True,
        )
        .agg(
            n_cycles=("Cycle_Index", "size"),
            n_cells=("sample_int", "nunique"),
            cycle_min=("Cycle_Index", "min"),
            cycle_q05=("Cycle_Index", lambda x: float(np.quantile(x, 0.05))),
            cycle_q25=("Cycle_Index", lambda x: float(np.quantile(x, 0.25))),
            cycle_median=("Cycle_Index", "median"),
            cycle_q75=("Cycle_Index", lambda x: float(np.quantile(x, 0.75))),
            cycle_q95=("Cycle_Index", lambda x: float(np.quantile(x, 0.95))),
            cycle_max=("Cycle_Index", "max"),
        )
        .sort_values("Condition_ID")
        .reset_index(drop=True)
    )

    cell_age = d.groupby(
        ["Condition_ID", "sample_int"], as_index=False, observed=True
    ).agg(
        n_cycles=("Cycle_Index", "nunique"),
        cycle_min=("Cycle_Index", "min"),
        cycle_q25=("Cycle_Index", lambda x: float(np.quantile(x, 0.25))),
        cycle_median=("Cycle_Index", "median"),
        cycle_q75=("Cycle_Index", lambda x: float(np.quantile(x, 0.75))),
        cycle_max=("Cycle_Index", "max"),
    )

    lower = int(condition_age["cycle_min"].max())
    upper = int(condition_age["cycle_max"].min())

    # Equivalent to the original cycle-by-cycle filter, without 611 rescans.
    count_by_cycle_condition = (
        d.groupby(["Cycle_Index", "Condition_ID"], observed=True)["sample_int"]
        .nunique()
        .rename("n_cells")
        .reset_index()
    )
    group_coverage = count_by_cycle_condition.groupby(
        "Cycle_Index", as_index=False, observed=True
    ).agg(
        n_conditions=("Condition_ID", "nunique"),
        min_cells_per_present_condition=("n_cells", "min"),
        median_cells_per_present_condition=("n_cells", "median"),
    )
    # Retain the exact original columns and semantics.
    coverage = group_coverage[["Cycle_Index", "n_conditions"]].copy()
    coverage["all_24_conditions"] = coverage["n_conditions"].eq(N_CONDITIONS)
    coverage["min_cells_per_present_condition"] = group_coverage[
        "min_cells_per_present_condition"
    ]
    coverage["median_cells_per_present_condition"] = group_coverage[
        "median_cells_per_present_condition"
    ]
    coverage["all24_with_min2cells"] = (
        coverage["all_24_conditions"]
        & coverage["min_cells_per_present_condition"].ge(2)
    )
    coverage["all24_with_min3cells"] = (
        coverage["all_24_conditions"]
        & coverage["min_cells_per_present_condition"].ge(3)
    )
    windows = window_table(coverage)

    # Preserve the original unordered-pair definition (24 choose 2 = 276).
    pairs = []
    for i in range(len(condition_age)):
        a = condition_age.iloc[i]
        for j in range(i + 1, len(condition_age)):
            b = condition_age.iloc[j]
            pairs.append({
                "Condition_A": int(a["Condition_ID"]),
                "Condition_B": int(b["Condition_ID"]),
                "median_cycle_A": float(a["cycle_median"]),
                "median_cycle_B": float(b["cycle_median"]),
                "abs_median_cycle_difference": abs(
                    float(a["cycle_median"]) - float(b["cycle_median"])
                ),
            })
    pair_age = pd.DataFrame(pairs).sort_values(
        "abs_median_cycle_difference", ascending=False
    ).reset_index(drop=True)

    output_tables = {
        "condition_cycle_age_distribution.csv": condition_age,
        "cell_cycle_age_distribution.csv": cell_age,
        "cycle_index_condition_coverage.csv": coverage,
        "common_cycle_windows.csv": windows,
        "pairwise_condition_age_difference.csv": pair_age,
    }
    for filename, table in output_tables.items():
        table.to_csv(output_dir / filename, index=False)

    print("\n=== CONDITION-LEVEL CYCLE DISTRIBUTIONS ===")
    print(condition_age.to_string(index=False))
    print("\n=== SIMPLE AGE-RANGE INTERSECTION ===")
    print(f"max(condition minimum) = {lower}")
    print(f"min(condition maximum) = {upper}")
    if lower <= upper:
        print(f"Common min/max range: {lower}–{upper}")
        print(f"Width: {upper - lower + 1} cycle indices")
    else:
        print("No common min/max range across the 24 conditions")

    print("\n=== LONGEST COMMON-CYCLE WINDOWS ===")
    for criterion in [
        "all_24_conditions", "all24_min_2_cells", "all24_min_3_cells"
    ]:
        part = windows.loc[windows["criterion"].eq(criterion)]
        print(f"\n{criterion}")
        print(part.head(10).to_string(index=False) if not part.empty else "  NONE")

    low = float(condition_age["cycle_median"].min())
    high = float(condition_age["cycle_median"].max())
    print("\n=== CONDITION MEDIAN-AGE SPREAD ===")
    print(f"Lowest condition median cycle: {low}")
    print(f"Highest condition median cycle: {high}")
    print(f"Range: {high - low}")
    print("\n=== TOP 20 CONDITION AGE SEPARATIONS ===")
    print(pair_age.head(20).to_string(index=False))
    print("\n" + "=" * 72)
    print("R4A AGE/CYCLE OVERLAP AUDIT COMPLETE")
    print("=" * 72)
    print(f"Output directory:\n{output_dir}")
    print("Note: exact-index cell counts are descriptive; they do not replace")
    print("cell-level bracketing or SoC-band coverage in later inference.")
    return output_tables


def main():
    default_dir = (
        Path("/content/revision_R2_2_global")
        if Path("/content").is_dir()
        else Path("outputs/revision_R2_2_global")
    )
    r22_dir = Path(os.getenv("SOC_R22_DIR", os.getenv("SOC_OUTPUT_DIR", str(default_dir))))
    run_age_cycle_audit(
        cycle_file=r22_dir / "global_cycle_profiles.pkl",
        output_dir=r22_dir / "R4_age_cycle",
    )


if __name__ == "__main__":
    main()
