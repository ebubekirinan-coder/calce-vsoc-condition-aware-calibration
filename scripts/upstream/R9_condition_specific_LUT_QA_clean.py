#!/usr/bin/env python3
"""R9: condition-specific voltage--SoC LUT and technical QA.

Scientific protocol preserved from the historical R9 script:
- R2.2 nominal-concordant cell profiles, all cells and equal cell weights
- pointwise median across physical cells, weighted isotonic regression
  using the number of supporting cells at each SoC node
- 0..100 integer SoC grid, no anchor, no guard, no extrapolation
- internal unsupported nodes CAN receive interpolated isotonic estimates;
  this is why R9 must not substitute for the final unified validation LUT.

R9 is retained as a diagnostic/technical historical stage. It is not the
canonical source for final manuscript LOCO, rate or temperature validation.

Inputs:  cell_profiles_nominal_concordant.csv (R2.2)
Outputs: original nine R9 files, with original names and column schemas.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SOC_GRID = np.arange(0, 101, dtype=float)
MONO_TOL_V = 1e-9
CORRECTION_THRESHOLDS_MV = (1, 2, 5, 10)
EXPECTED_CONDITIONS = 24
REGIONS = (("0-90", 0, 90), ("90-100", 90, 100.0001),
           ("95-100", 95, 100.0001))


def resolve_input():
    explicit = os.environ.get("VSOC_R9_CELL_FILE")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"R9 input file not found: {p}")
        return p
    for p in (
        Path("data/interim/cell_profiles_nominal_concordant.csv"),
        Path("cell_profiles_nominal_concordant.csv"),
        Path("/content/revision_R2_2_global/cell_profiles_nominal_concordant.csv"),
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Missing R2.2 cell_profiles_nominal_concordant.csv. "
        "Set VSOC_R9_CELL_FILE to its path."
    )


def resolve_output(input_file):
    explicit = os.environ.get("VSOC_R9_OUTPUT_DIR")
    if explicit:
        return Path(explicit).expanduser()
    if str(input_file).startswith("/content/"):
        return Path("/content/revision_R2_2_global/R9_final_LUT")
    return Path("outputs/R9_final_LUT")


def _finite_quantile(values, quantile):
    a = np.asarray(values, dtype=float)
    return float(np.nanquantile(a, quantile)) if np.isfinite(a).any() else np.nan


def _finite_count(values):
    return int(np.isfinite(np.asarray(values, dtype=float)).sum())


def load_cell_profiles(path):
    d = pd.read_csv(path)
    required = ["sample_int", "Condition_ID", "Temperature", "Rate",
                "Charge_Rate", "SoC", "V_cell_median"]
    missing = [c for c in required if c not in d]
    if missing:
        raise KeyError(f"R9 input missing columns: {missing}")

    for c in ("sample_int", "Condition_ID", "Temperature", "SoC", "V_cell_median"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Preserve the original R9 treatment of missing identifiers.
    d = d.dropna(subset=["sample_int", "Condition_ID", "SoC"]).copy()
    d["sample_int"] = d["sample_int"].astype(int)
    d["Condition_ID"] = d["Condition_ID"].astype(int)
    return d


def prepare_raw(d):
    meta = (d[["Condition_ID", "Temperature", "Rate", "Charge_Rate"]]
            .drop_duplicates(subset=["Condition_ID"])
            .sort_values("Condition_ID"))
    nominal_n = (d[["Condition_ID", "sample_int"]].drop_duplicates()
                 .groupby("Condition_ID")["sample_int"].nunique().to_dict())

    raw = (d.groupby(["Condition_ID", "SoC"], as_index=False, observed=True)
           .agg(V_raw_median=("V_cell_median", "median"),
                V_q25=("V_cell_median", lambda x: _finite_quantile(x, 0.25)),
                V_q75=("V_cell_median", lambda x: _finite_quantile(x, 0.75)),
                n_cells_at_soc=("V_cell_median", _finite_count)))
    raw = raw.merge(meta, on="Condition_ID", how="left")
    raw["nominal_n_cells"] = raw["Condition_ID"].map(nominal_n)
    raw["support_fraction"] = raw["n_cells_at_soc"] / raw["nominal_n_cells"]
    raw["minimum_strong_support"] = np.ceil(raw["nominal_n_cells"] / 2.0).astype(int)
    raw["strong_support"] = raw["n_cells_at_soc"] >= raw["minimum_strong_support"]
    return raw, meta, nominal_n


def build_condition_luts(raw, meta, nominal_n):
    lut_rows, correction_rows, qa_rows = [], [], []
    for cid in range(1, EXPECTED_CONDITIONS + 1):
        q = raw.loc[raw["Condition_ID"] == cid].sort_values("SoC").copy()
        fitdata = q.loc[q["V_raw_median"].notna() & (q["n_cells_at_soc"] > 0)].copy()
        if len(fitdata) < 2:
            raise RuntimeError(f"Condition {cid}: insufficient observed SoC support.")
        x = fitdata["SoC"].to_numpy(float)
        y = fitdata["V_raw_median"].to_numpy(float)
        weights = fitdata["n_cells_at_soc"].to_numpy(float)
        dy_raw = np.diff(y)
        n_raw_violations = int(np.sum(dy_raw < -MONO_TOL_V))
        worst_raw_drop_V = float(np.min(dy_raw)) if len(dy_raw) else np.nan

        iso = IsotonicRegression(increasing=True, out_of_bounds="nan")
        iso.fit(x, y, sample_weight=weights)
        pred_grid = iso.predict(SOC_GRID)
        observed_min_soc, observed_max_soc = float(np.min(x)), float(np.max(x))
        pred_grid[(SOC_GRID < observed_min_soc) | (SOC_GRID > observed_max_soc)] = np.nan

        qgrid = pd.DataFrame({"SoC": SOC_GRID}).merge(
            q[["SoC", "V_raw_median", "V_q25", "V_q75",
               "n_cells_at_soc", "nominal_n_cells", "support_fraction", "strong_support"]],
            on="SoC", how="left"
        )
        qgrid["V_LUT"] = pred_grid
        qgrid["is_observed_node"] = qgrid["V_raw_median"].notna()
        qgrid["is_interpolated_node"] = (
            qgrid["V_raw_median"].isna() & qgrid["V_LUT"].notna()
        )
        qgrid["correction_V"] = qgrid["V_LUT"] - qgrid["V_raw_median"]
        qgrid["abs_correction_mV"] = np.abs(qgrid["correction_V"]) * 1000.0
        final_valid = qgrid["V_LUT"].dropna().to_numpy(float)
        final_violations = int(np.sum(np.diff(final_valid) < -MONO_TOL_V))
        observed_corrections = qgrid.loc[
            qgrid["V_raw_median"].notna() & qgrid["V_LUT"].notna(),
            "abs_correction_mV"
        ]
        n_missing_final = int(qgrid["V_LUT"].isna().sum())
        n_weak_support = int((
            qgrid["is_observed_node"] & (qgrid["strong_support"] != True)
        ).sum())
        meta_row = meta.loc[meta["Condition_ID"] == cid].iloc[0]
        qa = {
            "Condition_ID": cid,
            "Temperature": int(meta_row["Temperature"]),
            "Rate": str(meta_row["Rate"]),
            "Charge_Rate": str(meta_row["Charge_Rate"]),
            "nominal_n_cells": int(nominal_n[cid]),
            "observed_soc_min": observed_min_soc,
            "observed_soc_max": observed_max_soc,
            "n_observed_soc_nodes": int(qgrid["is_observed_node"].sum()),
            "n_interpolated_internal_nodes": int(qgrid["is_interpolated_node"].sum()),
            "n_missing_final_nodes": n_missing_final,
            "n_weak_support_nodes": n_weak_support,
            "raw_monotonicity_violations": n_raw_violations,
            "worst_raw_single_step_drop_mV": worst_raw_drop_V * 1000.0,
            "final_monotonicity_violations": final_violations,
            "mean_abs_isotonic_correction_mV": (
                float(observed_corrections.mean()) if len(observed_corrections) else np.nan
            ),
            "median_abs_isotonic_correction_mV": (
                float(observed_corrections.median()) if len(observed_corrections) else np.nan
            ),
            "max_abs_isotonic_correction_mV": (
                float(observed_corrections.max()) if len(observed_corrections) else np.nan
            ),
        }
        for threshold in CORRECTION_THRESHOLDS_MV:
            qa[f"n_corrections_gt_{threshold}mV"] = int((observed_corrections > threshold).sum())
        qa_rows.append(qa)

        for _, r in qgrid.iterrows():
            lut_rows.append({
                "Condition_ID": cid,
                "Temperature": int(meta_row["Temperature"]),
                "Rate": str(meta_row["Rate"]),
                "Charge_Rate": str(meta_row["Charge_Rate"]),
                "SoC": float(r["SoC"]),
                "V_raw_median": r["V_raw_median"],
                "V_LUT": r["V_LUT"],
                "n_cells_at_soc": r["n_cells_at_soc"],
                "nominal_n_cells": r["nominal_n_cells"],
                "support_fraction": r["support_fraction"],
                "strong_support": r["strong_support"],
                "is_observed_node": bool(r["is_observed_node"]),
                "is_interpolated_node": bool(r["is_interpolated_node"]),
                "correction_V": r["correction_V"],
                "abs_correction_mV": r["abs_correction_mV"],
            })
            if pd.notna(r["V_raw_median"]):
                correction_rows.append({
                    "Condition_ID": cid, "SoC": float(r["SoC"]),
                    "V_raw_median": float(r["V_raw_median"]),
                    "V_LUT": float(r["V_LUT"]),
                    "correction_mV": float(r["correction_V"] * 1000.0),
                    "abs_correction_mV": float(r["abs_correction_mV"]),
                    "n_cells_at_soc": int(r["n_cells_at_soc"]),
                })
    return pd.DataFrame(lut_rows), pd.DataFrame(correction_rows), pd.DataFrame(qa_rows)


def build_support_qa(lut):
    records = []
    # Preserve original R9 bin semantics: 0-90 => [0,90), 90-100 => [90,100].
    for cid, g in lut.groupby("Condition_ID", observed=True):
        for region, lo, hi in REGIONS:
            q = g.loc[(g["SoC"] >= lo) & (g["SoC"] < hi)]
            observed = q.loc[q["is_observed_node"] == True]
            if len(observed):
                min_support = int(observed["n_cells_at_soc"].min())
                median_support = float(observed["n_cells_at_soc"].median())
                min_fraction = float(observed["support_fraction"].min())
            else:
                min_support, median_support, min_fraction = 0, np.nan, np.nan
            records.append({
                "Condition_ID": int(cid), "SoC_region": region,
                "n_grid_nodes": int(len(q)),
                "n_observed_nodes": int(q["is_observed_node"].sum()),
                "n_LUT_nodes": int(q["V_LUT"].notna().sum()),
                "minimum_cells_support": min_support,
                "median_cells_support": median_support,
                "minimum_support_fraction": min_fraction,
                "n_weak_support_nodes": int((
                    q["is_observed_node"] & (q["strong_support"] != True)
                ).sum()),
            })
    return pd.DataFrame(records)


def export_results(lut, qa, corrections, support_qa, out):
    matrix = (lut.pivot(index="Condition_ID", columns="SoC", values="V_LUT")
              .reindex(index=range(1, 25), columns=SOC_GRID))
    matrix.columns = [f"SoC_{int(x):03d}" for x in matrix.columns]
    expected_values = 24 * 101
    actual_rows = len(lut)
    n_finite_lut = int(np.isfinite(lut["V_LUT"]).sum())
    all_monotone = bool((qa["final_monotonicity_violations"] == 0).all())
    complete_grid = actual_rows == expected_values
    complete_values = n_finite_lut == expected_values

    arr32 = matrix.to_numpy(dtype=np.float32)
    arr16 = matrix.to_numpy(dtype=np.float16)
    finite = np.isfinite(arr32) & np.isfinite(arr16)
    qerr = np.abs(arr32[finite] - arr16[finite].astype(np.float32))
    quantization = pd.DataFrame([{
        "n_values": int(finite.sum()),
        "float32_bytes": int(arr32.nbytes),
        "float16_bytes": int(arr16.nbytes),
        "float32_KiB": float(arr32.nbytes / 1024),
        "float16_KiB": float(arr16.nbytes / 1024),
        "float16_mean_abs_quantization_error_mV": (
            float(np.mean(qerr) * 1000) if len(qerr) else np.nan
        ),
        "float16_p95_abs_quantization_error_mV": (
            float(np.quantile(qerr, 0.95) * 1000) if len(qerr) else np.nan
        ),
        "float16_max_abs_quantization_error_mV": (
            float(np.max(qerr) * 1000) if len(qerr) else np.nan
        ),
    }])

    global_qa = pd.DataFrame([{
        "n_conditions": int(lut["Condition_ID"].nunique()),
        "expected_nodes": expected_values,
        "actual_rows": actual_rows,
        "finite_LUT_values": n_finite_lut,
        "complete_24x101_grid": complete_grid,
        "all_2424_values_finite": complete_values,
        "all_final_curves_monotone": all_monotone,
        "total_raw_monotonicity_violations": int(qa["raw_monotonicity_violations"].sum()),
        "total_final_monotonicity_violations": int(qa["final_monotonicity_violations"].sum()),
        "global_mean_abs_correction_mV": float(corrections["abs_correction_mV"].mean()),
        "global_median_abs_correction_mV": float(corrections["abs_correction_mV"].median()),
        "global_p95_abs_correction_mV": float(corrections["abs_correction_mV"].quantile(0.95)),
        "global_max_abs_correction_mV": float(corrections["abs_correction_mV"].max()),
    }])

    # As in the historical R9, enforce monotonicity and coordinate completeness.
    if not all_monotone:
        raise RuntimeError("FINAL QA FAILED: at least one LUT is non-monotone.")
    if not complete_grid:
        raise RuntimeError("FINAL QA FAILED: 24x101 row grid is incomplete.")

    out.mkdir(parents=True, exist_ok=True)
    lut.to_csv(out / "R9_final_LUT_long.csv", index=False)
    matrix.to_csv(out / "R9_final_LUT_matrix.csv")
    qa.to_csv(out / "R9_condition_QA.csv", index=False)
    support_qa.to_csv(out / "R9_SoC_support_QA.csv", index=False)
    corrections.to_csv(out / "R9_monotonic_corrections.csv", index=False)
    quantization.to_csv(out / "R9_quantization_QA.csv", index=False)
    global_qa.to_csv(out / "R9_global_QA.csv", index=False)
    np.save(out / "R9_final_LUT_float32.npy", arr32)
    np.save(out / "R9_final_LUT_float16.npy", arr16)

    print("\nExpected LUT nodes:", expected_values)
    print("Actual LUT rows:", actual_rows)
    print("Finite LUT values:", n_finite_lut)
    print("All final curves monotone:", all_monotone)
    print("\n" + "=" * 80)
    print("R9 FINAL LUT + QA COMPLETE")
    print("=" * 80)
    print("\n=== GLOBAL LUT QA ===\n", global_qa.to_string(index=False))
    print("\n=== CONDITION QA ===\n", qa.to_string(index=False))
    print("\n=== 90–100% SUPPORT QA ===\n",
          support_qa[support_qa["SoC_region"] == "90-100"].to_string(index=False))
    print("\n=== FLOAT16 / FLOAT32 QUANTIZATION QA ===\n",
          quantization.to_string(index=False))
    print("\n=== LARGEST 25 ISOTONIC CORRECTIONS ===\n",
          corrections.sort_values("abs_correction_mV", ascending=False).head(25).to_string(index=False))
    if not complete_values:
        print("\nWARNING: The 24×101 coordinate grid includes unsupported NaN "
              "endpoints. DO NOT extrapolate them silently.")
    print("\nOutput directory:\n", out)


def main():
    input_path = resolve_input()
    out = resolve_output(input_path)
    d = load_cell_profiles(input_path)
    print("Input:", input_path)
    print("Physical cells:", d["sample_int"].nunique())
    print("Conditions:", d["Condition_ID"].nunique())
    print("Cell-condition pairs:",
          d[["sample_int", "Condition_ID"]].drop_duplicates().shape[0])
    raw, meta, nominal_n = prepare_raw(d)
    lut, corrections, qa = build_condition_luts(raw, meta, nominal_n)
    support_qa = build_support_qa(lut)
    export_results(lut, qa, corrections, support_qa, out)


if __name__ == "__main__":
    main()
