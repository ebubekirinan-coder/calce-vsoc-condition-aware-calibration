"""R11 — Physical-cell support and common-domain audit for the CALCE V–SoC LUT.

Inputs (from previously completed R2.2 and R9):
    cell_profiles_nominal_concordant.csv
    R9_final_LUT/R9_final_LUT_long.csv

This script audits support only. It does not refit the voltage model, modify
measurements, impose a new support threshold, or change the canonical 0–90% LUT.
The common-domain QA here uses R2.2's all-cycle cell profiles; validation
eligibility and held-out evaluation have separate, downstream support rules.

GitHub: scripts/audit_soc_support.py
Colab: set VSOC_R22_DIR to the R2.2 output directory if it differs from the
       default /content/revision_R2_2_global.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SOC_MIN, SOC_MAX = 0, 90
CONDITIONS = tuple(range(1, 25))
CONDITION_OF_INTEREST = 5

CRITERIA = {
    "all24_ge1": ">=1 physical cell",
    "all24_ge2": ">=2 physical cells",
    "all24_ge3": ">=3 physical cells",
    "all24_ge50pct": ">=50% nominal cells",
    "all24_ge75pct": ">=75% nominal cells",
    "all24_full": "100% nominal cells",
}

OUTPUT_NAMES = {
    "support": "R11_condition_soc_support.csv",
    "soc_audit": "R11_soc_bottleneck_audit.csv",
    "windows": "R11_all_common_support_windows.csv",
    "longest": "R11_longest_common_support_windows.csv",
    "condition_support": "R11_condition_support_summary.csv",
    "c5_support": "R11_condition5_soc_support.csv",
    "c5_cell_coverage": "R11_condition5_cell_coverage.csv",
    "correction_by_support": "R11_correction_by_support.csv",
    "domain_summary": "R11_domain_summary.csv",
}


def resolve_root() -> Path:
    """Locate an existing R2.2 directory without rerunning earlier stages."""
    explicit = os.environ.get("VSOC_R22_DIR")
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"R2.2 directory not found: {root}")
        return root
    for root in (
        Path("/content/revision_R2_2_global"),
        Path("data/interim"),
        Path("."),
    ):
        if (root / "cell_profiles_nominal_concordant.csv").is_file() and (
            root / "R9_final_LUT" / "R9_final_LUT_long.csv"
        ).is_file():
            return root
    raise FileNotFoundError(
        "R2.2 cell profiles and R9 LUT were not found together. "
        "Set VSOC_R22_DIR to their parent R2.2 directory."
    )


def read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_path = root / "cell_profiles_nominal_concordant.csv"
    r9_path = root / "R9_final_LUT" / "R9_final_LUT_long.csv"
    for path in (cell_path, r9_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input missing: {path}")

    cell = pd.read_csv(cell_path)
    required_cell = {"sample_int", "Condition_ID", "SoC", "V_cell_median"}
    if required_cell - set(cell):
        raise KeyError(f"R2.2 missing columns: {sorted(required_cell - set(cell))}")
    for col in ("sample_int", "Condition_ID", "SoC", "V_cell_median"):
        cell[col] = pd.to_numeric(cell[col], errors="coerce")
    cell = cell.dropna(subset=["sample_int", "Condition_ID", "SoC"]).copy()
    cell[["sample_int", "Condition_ID"]] = cell[[
        "sample_int", "Condition_ID"
    ]].astype(int)
    cell = cell.loc[cell["SoC"].between(SOC_MIN, SOC_MAX)].copy()

    # Guard against double-counting a physical cell at the same SoC node.
    if cell.duplicated(["Condition_ID", "sample_int", "SoC"]).any():
        raise ValueError("Duplicate R2.2 physical-cell/condition/SoC keys.")
    if set(cell["Condition_ID"].unique()) != set(CONDITIONS):
        raise ValueError("R2.2 nominal profiles must cover conditions 1 through 24.")
    if cell[["Condition_ID", "sample_int"]].drop_duplicates().shape[0] != 180:
        raise ValueError("Expected 180 nominal physical-cell/condition pairs.")

    r9 = pd.read_csv(r9_path)
    required_r9 = {"Condition_ID", "SoC", "abs_correction_mV", "n_cells_at_soc"}
    if required_r9 - set(r9):
        raise KeyError(f"R9 missing columns: {sorted(required_r9 - set(r9))}")
    for col in ("Condition_ID", "SoC", "abs_correction_mV", "n_cells_at_soc"):
        r9[col] = pd.to_numeric(r9[col], errors="coerce")
    r9 = r9.loc[r9["SoC"].between(SOC_MIN, SOC_MAX)].copy()
    if r9.duplicated(["Condition_ID", "SoC"]).any():
        raise ValueError("Duplicate R9 condition/SoC keys.")
    return cell, r9


def contiguous_windows(values) -> list[tuple[int, int, int]]:
    """Return sorted inclusive (start, end, width) runs of integer nodes."""
    vals = sorted({int(x) for x in values})
    if not vals:
        return []
    windows = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
        else:
            windows.append((start, prev, prev - start + 1))
            start = prev = value
    windows.append((start, prev, prev - start + 1))
    return windows


def build_audit(cell: pd.DataFrame, r9: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nominal_n = (
        cell[["Condition_ID", "sample_int"]]
        .drop_duplicates()
        .groupby("Condition_ID")["sample_int"]
        .nunique()
    )

    grid = pd.MultiIndex.from_product(
        [CONDITIONS, range(SOC_MIN, SOC_MAX + 1)],
        names=["Condition_ID", "SoC"],
    ).to_frame(index=False)
    grouped = (
        cell.groupby(["Condition_ID", "SoC"], as_index=False, observed=True)
        .agg(n_cells_support=("V_cell_median", lambda x: int(np.isfinite(x).sum())))
    )
    support = grid.merge(grouped, on=["Condition_ID", "SoC"], how="left")
    support["n_cells_support"] = support["n_cells_support"].fillna(0).astype(int)
    support["nominal_n_cells"] = support["Condition_ID"].map(nominal_n).astype(int)
    support["support_fraction"] = support["n_cells_support"] / support["nominal_n_cells"]
    n = support["n_cells_support"]
    nominal = support["nominal_n_cells"]
    support["support_ge_1"] = n >= 1
    support["support_ge_2"] = n >= 2
    support["support_ge_3"] = n >= 3
    support["support_ge_half"] = n >= np.ceil(nominal * 0.50)
    support["support_ge_75pct"] = n >= np.ceil(nominal * 0.75)
    support["support_full"] = n == nominal

    soc_rows = []
    for soc, g in support.groupby("SoC", observed=True):
        min_cells = int(g["n_cells_support"].min())
        min_fraction = float(g["support_fraction"].min())
        soc_rows.append({
            "SoC": int(soc),
            "minimum_cells_over_24_conditions": min_cells,
            "minimum_support_fraction_over_24": min_fraction,
            "bottleneck_conditions_by_cell_count": ",".join(map(str, g.loc[
                g["n_cells_support"] == min_cells, "Condition_ID"
            ].astype(int).tolist())),
            "bottleneck_conditions_by_fraction": ",".join(map(str, g.loc[
                np.isclose(g["support_fraction"], min_fraction), "Condition_ID"
            ].astype(int).tolist())),
            "all24_ge1": bool(g["support_ge_1"].all()),
            "all24_ge2": bool(g["support_ge_2"].all()),
            "all24_ge3": bool(g["support_ge_3"].all()),
            "all24_ge50pct": bool(g["support_ge_half"].all()),
            "all24_ge75pct": bool(g["support_ge_75pct"].all()),
            "all24_full": bool(g["support_full"].all()),
        })
    soc_audit = pd.DataFrame(soc_rows)

    window_rows = [
        {"criterion": label, "start_soc": start, "end_soc": end, "width_nodes": width}
        for col, label in CRITERIA.items()
        for start, end, width in contiguous_windows(
            soc_audit.loc[soc_audit[col], "SoC"].tolist()
        )
    ]
    # Retain the CSV schema even if no criterion has a qualifying interval.
    windows = pd.DataFrame(window_rows, columns=[
        "criterion", "start_soc", "end_soc", "width_nodes"
    ])
    if not windows.empty:
        windows = windows.sort_values(
            ["criterion", "width_nodes"], ascending=[True, False]
        )

    longest_rows = []
    for label in CRITERIA.values():
        q = windows.loc[windows["criterion"] == label]
        if q.empty:
            longest_rows.append({
                "criterion": label, "start_soc": np.nan,
                "end_soc": np.nan, "width_nodes": 0,
            })
        else:
            row = q.sort_values("width_nodes", ascending=False).iloc[0]
            longest_rows.append({
                "criterion": label, "start_soc": int(row["start_soc"]),
                "end_soc": int(row["end_soc"]), "width_nodes": int(row["width_nodes"]),
            })
    longest = pd.DataFrame(longest_rows)

    condition_support = (
        support.groupby("Condition_ID", as_index=False, observed=True)
        .agg(
            nominal_n_cells=("nominal_n_cells", "first"),
            min_cell_support=("n_cells_support", "min"),
            median_cell_support=("n_cells_support", "median"),
            mean_cell_support=("n_cells_support", "mean"),
            min_support_fraction=("support_fraction", "min"),
            median_support_fraction=("support_fraction", "median"),
            n_nodes_zero_support=("support_ge_1", lambda x: int((~x).sum())),
            n_nodes_support_1=("n_cells_support", lambda x: int((x == 1).sum())),
            n_nodes_support_lt2=("n_cells_support", lambda x: int((x < 2).sum())),
            n_nodes_support_lt3=("n_cells_support", lambda x: int((x < 3).sum())),
            n_nodes_below_half=("support_ge_half", lambda x: int((~x).sum())),
            n_nodes_below_75pct=("support_ge_75pct", lambda x: int((~x).sum())),
        )
    )

    c5_support = support.loc[support["Condition_ID"] == CONDITION_OF_INTEREST].copy()
    c5 = cell.loc[cell["Condition_ID"] == CONDITION_OF_INTEREST]
    c5_rows = []
    for sid, g in c5.groupby("sample_int", observed=True):
        finite_soc = sorted(g.loc[
            np.isfinite(g["V_cell_median"]), "SoC"
        ].astype(int).unique().tolist())
        if finite_soc:
            longest_cell = max(contiguous_windows(finite_soc), key=lambda x: x[2])
            text = f"{longest_cell[0]}-{longest_cell[1]}"
            min_soc, max_soc = min(finite_soc), max(finite_soc)
        else:
            text, min_soc, max_soc = "", np.nan, np.nan
        c5_rows.append({
            "sample_int": int(sid),
            "n_finite_soc_nodes": len(finite_soc),
            "min_supported_soc": min_soc,
            "max_supported_soc": max_soc,
            "longest_contiguous_window": text,
            "all_supported_soc_nodes": ",".join(map(str, finite_soc)),
        })
    c5_cell_coverage = pd.DataFrame(c5_rows)

    # R9's support must match the underlying R2.2 cell profile exactly.
    r9_match = r9[["Condition_ID", "SoC", "n_cells_at_soc"]].merge(
        support[["Condition_ID", "SoC", "n_cells_support"]],
        on=["Condition_ID", "SoC"], how="outer", validate="one_to_one",
        indicator=True,
    )
    if (len(r9_match) != 24 * 91 or (r9_match["_merge"] != "both").any()
            or not np.array_equal(r9_match["n_cells_at_soc"].to_numpy(),
                                  r9_match["n_cells_support"].to_numpy())):
        raise RuntimeError("R9 SoC-wise cell support does not match R2.2 nominal profiles.")

    r9_valid = r9.loc[r9["abs_correction_mV"].notna()].copy()
    r9_valid["support_category"] = pd.cut(
        r9_valid["n_cells_at_soc"],
        bins=[-0.1, 1, 2, 3, 4, np.inf],
        labels=["1 cell", "2 cells", "3 cells", "4 cells", ">=5 cells"],
    )
    correction_by_support = (
        r9_valid.groupby("support_category", observed=True, as_index=False)
        .agg(
            n_nodes=("abs_correction_mV", "size"),
            mean_abs_correction_mV=("abs_correction_mV", "mean"),
            median_abs_correction_mV=("abs_correction_mV", "median"),
            p95_abs_correction_mV=(
                "abs_correction_mV", lambda x: float(np.quantile(x, 0.95))
            ),
            max_abs_correction_mV=("abs_correction_mV", "max"),
        )
    )

    domain_summary = pd.DataFrame([{
        "domain": "0-90",
        "n_soc_nodes": SOC_MAX - SOC_MIN + 1,
        "all24_have_at_least_1_cell_nodes": int(soc_audit["all24_ge1"].sum()),
        "all24_have_at_least_2_cells_nodes": int(soc_audit["all24_ge2"].sum()),
        "all24_have_at_least_3_cells_nodes": int(soc_audit["all24_ge3"].sum()),
        "all24_have_at_least_50pct_nodes": int(soc_audit["all24_ge50pct"].sum()),
        "all24_have_at_least_75pct_nodes": int(soc_audit["all24_ge75pct"].sum()),
        "all24_have_full_nominal_support_nodes": int(soc_audit["all24_full"].sum()),
    }])
    return dict(
        support=support, soc_audit=soc_audit, windows=windows, longest=longest,
        condition_support=condition_support, c5_support=c5_support,
        c5_cell_coverage=c5_cell_coverage,
        correction_by_support=correction_by_support, domain_summary=domain_summary,
    )


def main() -> None:
    root = resolve_root()
    cell, r9 = read_inputs(root)
    results = build_audit(cell, r9)
    output = Path(os.environ.get("VSOC_R11_OUTPUT_DIR", str(root / "R11_support_domain_audit")))
    output.mkdir(parents=True, exist_ok=True)
    for key, name in OUTPUT_NAMES.items():
        results[key].to_csv(output / name, index=False)

    print("Conditions:", cell["Condition_ID"].nunique())
    print("Physical cell-condition pairs:", cell[["Condition_ID", "sample_int"]].drop_duplicates().shape[0])
    print("\n" + "=" * 82 + "\nR11 FINAL SUPPORT-DOMAIN AUDIT COMPLETE\n" + "=" * 82)
    print("\n=== 0–90 DOMAIN SUPPORT SUMMARY ===")
    print(results["domain_summary"].to_string(index=False))
    print("\n=== LONGEST COMMON SUPPORT WINDOWS ===")
    print(results["longest"].to_string(index=False))
    print("\n=== CONDITION SUPPORT SUMMARY ===")
    print(results["condition_support"].to_string(index=False))
    print("\n=== CONDITION 5 — CELL COVERAGE ===")
    print(results["c5_cell_coverage"].to_string(index=False))
    print("\n=== CONDITION 5 — SoC SUPPORT COUNTS ===")
    print(results["c5_support"][[
        "SoC", "n_cells_support", "nominal_n_cells", "support_fraction",
        "support_ge_half", "support_ge_75pct", "support_full"
    ]].to_string(index=False))
    print("\n=== ISOTONIC CORRECTION BY CELL SUPPORT ===")
    print(results["correction_by_support"].to_string(index=False))
    print("\n=== 20 WORST GLOBAL SoC BOTTLENECKS ===")
    print(results["soc_audit"].sort_values(
        ["minimum_support_fraction_over_24", "minimum_cells_over_24_conditions", "SoC"],
        ascending=[True, True, True],
    ).head(20).to_string(index=False))
    print("\nOutput directory:\n", output)
    print("R9/R2.2 support consistency: PASS")
    print("NOTE: R11 audits all-cycle LUT support; manuscript validation uses its own coverage masks.")


if __name__ == "__main__":
    main()
