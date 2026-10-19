# -*- coding: utf-8 -*-
"""CALCE V-SoC | R4B (clean): 52-132 cycle-window cell and condition profiles.

Input: global_cycle_profiles.pkl (created by R2.2).
GitHub: scripts/build_window_cell_profiles.py

Run (from repository root):
    python scripts/build_window_cell_profiles.py

Optional paths (also work in Google Colab):
    VSOC_GLOBAL_CYCLES=/path/to/global_cycle_profiles.pkl
    VSOC_OUTPUT_DIR=/path/to/output

Scientific protocol (unchanged from manuscript-aligned R4B):
  nominal-concordant cycles -> [52, 132] inclusive -> pointwise median
  within each cell -> pointwise median across physical cells.
  Retains all 101 SoC output nodes; unsupported nodes remain NaN.
  Band-level 80% eligibility and inferential modelling happen downstream.
  QA counts are explicit and intentionally tied to manuscript inputs.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd


# ======================= CONFIGURATION =======================
WINDOW_START = 52                    # inclusive
WINDOW_END = 132                     # inclusive
ANALYSIS_NAME = "broad_52_132"
SOC_GRID = np.arange(0, 101, dtype=float)
V_COLS = [f"V_{i:03d}" for i in range(101)]

# Verified counts from the current paper's R2.2/R4B run.
# Stop rather than publish silently altered source data or output.
EXPECTED = {
    "nominal_cycles": 83_045,
    "nominal_cells": 180,
    "nominal_conditions": 24,
    "window_cycles": 13_181,
    "window_cells": 179,
    "window_conditions": 24,
}


def resolve_input():
    explicit = os.environ.get("VSOC_GLOBAL_CYCLES")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"VSOC_GLOBAL_CYCLES input not found: {path}")
        return path

    candidates = [
        Path("data/interim/global_cycle_profiles.pkl"),
        Path("global_cycle_profiles.pkl"),
        Path("/content/revision_R2_2_global/global_cycle_profiles.pkl"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "R2.2 global_cycle_profiles.pkl not found. "
        "Place it under data/interim/ or set VSOC_GLOBAL_CYCLES."
    )


def require_equal(label, actual, expected):
    if actual != expected:
        raise RuntimeError(
            f"QA FAILED: {label}: observed={actual:,}, expected={expected:,}. "
            "Check the R2.2 input before publication."
        )


def build_profiles(cycle):
    required = [
        "matches_nominal", "sample_int", "Condition_ID", "Cycle_Index",
        "Temperature", "Rate", "Charge_Rate", *V_COLS,
    ]
    missing = [c for c in required if c not in cycle.columns]
    if missing:
        raise KeyError(f"Missing columns in R2.2 global cycle store: {missing}")

    nominal = cycle.loc[cycle["matches_nominal"] == True].copy()
    require_equal("Nominal cycles", len(nominal), EXPECTED["nominal_cycles"])
    require_equal("Nominal cells", nominal["sample_int"].nunique(), EXPECTED["nominal_cells"])
    require_equal("Nominal conditions", nominal["Condition_ID"].nunique(), EXPECTED["nominal_conditions"])

    keys = ["sample_int", "Condition_ID", "Cycle_Index"]
    if nominal[keys].isna().any().any() or nominal.duplicated(keys).any():
        raise ValueError("Missing or duplicate physical cycle keys in nominal branch.")

    metadata_cols = ["Condition_ID", "Temperature", "Rate", "Charge_Rate"]
    condition_meta = nominal[metadata_cols].drop_duplicates()
    if condition_meta["Condition_ID"].duplicated().any():
        raise ValueError("Conflicting condition metadata for the same Condition_ID.")

    # R4B original inclusive window: Cycle_Index >= 52 AND <= 132.
    window = nominal.loc[
        nominal["Cycle_Index"].between(WINDOW_START, WINDOW_END, inclusive="both")
    ].copy()
    require_equal("Window cycles", len(window), EXPECTED["window_cycles"])
    require_equal("Window cells", window["sample_int"].nunique(), EXPECTED["window_cells"])
    require_equal("Window conditions", window["Condition_ID"].nunique(), EXPECTED["window_conditions"])

    # Original R4B: distinct cycle count per physical cell and condition.
    cell_support = window.groupby(
        ["Condition_ID", "sample_int"], as_index=False, observed=True
    ).agg(
        n_cycles=("Cycle_Index", "nunique"),
        cycle_min=("Cycle_Index", "min"),
        cycle_median=("Cycle_Index", "median"),
        cycle_max=("Cycle_Index", "max"),
    )
    support = cell_support.groupby(
        "Condition_ID", as_index=False, observed=True
    ).agg(
        n_cells=("sample_int", "nunique"),
        total_cycles=("n_cycles", "sum"),
        min_cycles_per_cell=("n_cycles", "min"),
        median_cycles_per_cell=("n_cycles", "median"),
        max_cycles_per_cell=("n_cycles", "max"),
    )
    support["analysis"] = ANALYSIS_NAME

    # Original R4B calculation: for each cell, median over its cycles
    # independently at each SoC node, with node-specific cycle support.
    cell_rows = []
    for (cid, sid), g in window.groupby(
        ["Condition_ID", "sample_int"], observed=True, sort=False
    ):
        A = g[V_COLS].to_numpy(dtype=float)
        if np.isinf(A).any():
            raise ValueError(
                f"Nonfinite infinite voltage in condition={cid}, cell={sid}."
            )

        # Preserve unsupported SoC nodes as NaN. Compute each median only
        # on columns with finite cycle observations; this avoids spurious
        # NumPy All-NaN warnings without changing finite-node estimates.
        n = np.isfinite(A).sum(axis=0)
        supported = n > 0
        med = np.full(len(V_COLS), np.nan, dtype=float)
        if supported.any():
            med[supported] = np.nanmedian(A[:, supported], axis=0)
        for j, soc in enumerate(SOC_GRID):
            cell_rows.append({
                "analysis": ANALYSIS_NAME,
                "Condition_ID": int(cid),
                "sample_int": int(sid),
                "SoC": float(soc),
                "Voltage": float(med[j]) if np.isfinite(med[j]) else np.nan,
                "n_cycles_at_soc": int(n[j]),
            })
    cell_profiles = pd.DataFrame(cell_rows)

    # Original R4B calculation: equal weight per physical cell.
    condition_profiles = (
        cell_profiles.groupby(
            ["Condition_ID", "SoC"], as_index=False, observed=True
        ).agg(
            Voltage=("Voltage", "median"),
            n_cells_at_soc=("Voltage", lambda z: int(np.isfinite(z).sum())),
        ).merge(condition_meta, on="Condition_ID", how="left", validate="many_to_one")
    )
    condition_profiles["analysis"] = ANALYSIS_NAME

    require_equal("Cell-profile rows", len(cell_profiles), EXPECTED["window_cells"] * len(SOC_GRID))
    require_equal("Condition-profile rows", len(condition_profiles), EXPECTED["window_conditions"] * len(SOC_GRID))
    require_equal("Support-cycle total", int(support["total_cycles"].sum()), len(window))
    if (cell_profiles.loc[cell_profiles["n_cycles_at_soc"] == 0, "Voltage"].notna()).any():
        raise RuntimeError("Voltage found at a zero-cycle-support SoC node.")
    return cell_profiles, condition_profiles, support


def main():
    input_file = resolve_input()
    output_dir = Path(os.environ.get("VSOC_OUTPUT_DIR", "outputs/R4B_broad_52_132")).expanduser()
    print(f"Input: {input_file}")
    cycle = pd.read_pickle(input_file)  # trusted, locally generated R2.2 pickle only
    cell_profiles, condition_profiles, support = build_profiles(cycle)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "broad_52_132_cell_profiles.csv": cell_profiles,
        "broad_52_132_condition_profiles.csv": condition_profiles,
        "broad_52_132_condition_support.csv": support,
    }
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(frame):,} rows)")
    print("\nR4B 52–132 QA PASSED")
    print(f"Window: {WINDOW_START}–{WINDOW_END} (inclusive)")
    print(f"Cycles: {int(support['total_cycles'].sum()):,}")
    print(f"Cells: {int(support['n_cells'].sum()):,}")
    print(f"Conditions: {support['Condition_ID'].nunique():,}")
    print(f"Cell-profile rows: {len(cell_profiles):,}")
    print(f"Condition-profile rows: {len(condition_profiles):,}")
    print("\nCONDITION SUPPORT")
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
