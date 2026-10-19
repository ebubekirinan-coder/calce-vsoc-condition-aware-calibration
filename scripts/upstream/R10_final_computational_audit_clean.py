#!/usr/bin/env python3
"""R10 | Package the historical R9 LUT on the common 0..90% SoC domain.

No model is fitted here. Scientific operations from the original R10 are
preserved: exact R9 0..90 inclusive extraction, 24x91 matrix, float32/float16
quantization, scalar/vectorized linear lookup, condition support diagnostics,
optional R5/R6/R7/R8 audit entries, and the ten original output artifacts.

R9/R10 are technical packaging/QA stages. Final manuscript LOCO, holdout, and
other inferential results must be reproduced using the later unified estimator.

Configuration:
    VSOC_R10_ROOT       Root directory holding R9_final_LUT/, R5_... etc.
    VSOC_R10_OUTPUT_DIR Optional export directory.
    VSOC_R10_SCALAR_BENCH, VSOC_R10_VECTOR_BENCH  Optional benchmark sizes.

Run: python R10_final_computational_audit_clean.py
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

SOC_MIN, SOC_MAX = 0, 90
N_CONDITIONS, N_SOC = 24, 91
EXPECTED_VALUES = N_CONDITIONS * N_SOC
RNG_SEED = 20260924
MONO_TOL_V = 1e-9
DEFAULT_SCALAR_BENCH = 250_000
DEFAULT_VECTOR_BENCH = 2_000_000


def resolve_root() -> Path:
    explicit = os.environ.get("VSOC_R10_ROOT")
    if explicit:
        root = Path(explicit).expanduser()
        if not (root / "R9_final_LUT" / "R9_final_LUT_long.csv").is_file():
            raise FileNotFoundError(f"Missing R9 LUT under VSOC_R10_ROOT: {root}")
        return root
    candidates = (
        Path("."), Path("outputs"),
        Path("/content/revision_R2_2_global"),
    )
    for root in candidates:
        if (root / "R9_final_LUT" / "R9_final_LUT_long.csv").is_file():
            return root
    raise FileNotFoundError("R9_final_LUT_long.csv not found. Set VSOC_R10_ROOT.")


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str], name: str) -> None:
    missing = [col for col in columns if col not in frame]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def load_canonical(path: Path) -> pd.DataFrame:
    lut = pd.read_csv(path)
    needed = ("Condition_ID", "SoC", "V_LUT", "n_cells_at_soc", "support_fraction")
    require_columns(lut, needed, "R9 final LUT")
    for col in needed:
        lut[col] = pd.to_numeric(lut[col], errors="coerce")
    canonical = (lut.loc[lut["SoC"].between(SOC_MIN, SOC_MAX, inclusive="both")]
                 .sort_values(["Condition_ID", "SoC"]).reset_index(drop=True).copy())
    if canonical[["Condition_ID", "SoC"]].isna().any().any():
        raise ValueError("Canonical LUT contains missing condition/SoC keys")
    if canonical.duplicated(["Condition_ID", "SoC"]).any():
        raise ValueError("Canonical LUT contains duplicate condition/SoC keys")
    if len(canonical) != EXPECTED_VALUES:
        raise RuntimeError(f"Canonical R10 has {len(canonical)} rows; expected {EXPECTED_VALUES}")
    return canonical


def prepare_lut(canonical: pd.DataFrame):
    mono_rows = []
    for cid, g in canonical.groupby("Condition_ID", observed=True):
        v = g.sort_values("SoC")["V_LUT"].to_numpy(dtype=float)
        dv = np.diff(v)
        mono_rows.append({
            "Condition_ID": int(cid),
            "n_nodes": int(len(g)),
            "n_finite": int(np.isfinite(v).sum()),
            "monotonicity_violations": int((dv < -MONO_TOL_V).sum()),
            "minimum_dV_per_1pct_mV": float(np.min(dv) * 1000) if len(dv) else np.nan,
            "maximum_dV_per_1pct_mV": float(np.max(dv) * 1000) if len(dv) else np.nan,
        })
    mono = pd.DataFrame(mono_rows)
    grid_counts = canonical.groupby("Condition_ID")["SoC"].nunique()
    all_conditions_91 = bool(len(grid_counts) == N_CONDITIONS and (grid_counts == N_SOC).all())
    n_finite = int(np.isfinite(canonical["V_LUT"]).sum())
    all_finite = bool(n_finite == EXPECTED_VALUES)
    all_monotone = bool((mono["monotonicity_violations"] == 0).all())

    matrix = (canonical.pivot(index="Condition_ID", columns="SoC", values="V_LUT")
              .reindex(index=range(1, N_CONDITIONS + 1),
                       columns=np.arange(SOC_MIN, SOC_MAX + 1, dtype=float)))
    matrix.columns = [f"SoC_{i:03d}" for i in range(N_SOC)]
    arr64 = matrix.to_numpy(dtype=np.float64)
    arr32 = matrix.to_numpy(dtype=np.float32)
    arr16 = matrix.to_numpy(dtype=np.float16)
    if not all_conditions_91 or arr32.shape != (N_CONDITIONS, N_SOC):
        raise RuntimeError("R10 requires exactly 24 conditions with each 0..90% node")
    if not all_finite or not np.isfinite(arr32).all():
        raise RuntimeError("The canonical 0..90 LUT has unsupported or invalid nodes")
    if not all_monotone:
        raise RuntimeError("The canonical 0..90 LUT violates monotonicity")
    return matrix, arr64, arr32, arr16, mono, n_finite, all_monotone


def quantization_qa(arr64: np.ndarray, arr32: np.ndarray, arr16: np.ndarray):
    qerr = np.abs(arr32 - arr16.astype(np.float32))
    memory_quant = pd.DataFrame([{
        "n_conditions": N_CONDITIONS,
        "n_soc_nodes": N_SOC,
        "n_values": int(arr32.size),
        "float64_bytes": int(arr64.nbytes),
        "float32_bytes": int(arr32.nbytes),
        "float16_bytes": int(arr16.nbytes),
        "float64_KiB": float(arr64.nbytes / 1024),
        "float32_KiB": float(arr32.nbytes / 1024),
        "float16_KiB": float(arr16.nbytes / 1024),
        "float16_mean_abs_error_mV": float(np.mean(qerr) * 1000),
        "float16_p95_abs_error_mV": float(np.quantile(qerr, 0.95) * 1000),
        "float16_max_abs_error_mV": float(np.max(qerr) * 1000),
    }])
    return qerr, memory_quant


def lookup_scalar(table: np.ndarray, condition_id: int, soc: float) -> float:
    """Original two-point 1%-node interpolation; NaN outside the SoC domain."""
    cid = int(condition_id) - 1
    if not 0 <= cid < N_CONDITIONS:
        raise ValueError("condition_id must be 1...24")
    s = float(soc)
    if not np.isfinite(s) or s < SOC_MIN or s > SOC_MAX:
        return np.nan
    lo = int(np.floor(s))
    if lo >= SOC_MAX:
        return float(table[cid, SOC_MAX])
    hi = lo + 1
    w = s - lo
    return float((1.0 - w) * table[cid, lo] + w * table[cid, hi])


def lookup_vectorized(table: np.ndarray, condition_ids, soc) -> np.ndarray:
    """Original vectorized 1%-node linear interpolation, float32 output."""
    condition_ids = np.asarray(condition_ids, dtype=np.int32)
    soc = np.asarray(soc, dtype=np.float32)
    if condition_ids.ndim != 1 or soc.ndim != 1 or len(condition_ids) != len(soc):
        raise ValueError("Condition and SoC inputs must be equal-length 1D vectors")
    out = np.full(len(soc), np.nan, dtype=np.float32)
    valid = ((condition_ids >= 1) & (condition_ids <= N_CONDITIONS)
             & (soc >= SOC_MIN) & (soc <= SOC_MAX))
    if not valid.any():
        return out
    cid = condition_ids[valid] - 1
    sv = soc[valid]
    lo = np.floor(sv).astype(np.int32)
    hi = np.minimum(lo + 1, SOC_MAX)
    w = (sv - lo).astype(np.float32)
    out[valid] = ((1.0 - w) * table[cid, lo] + w * table[cid, hi])
    return out


def check_lookup_consistency(arr32: np.ndarray, rng: np.random.Generator) -> float:
    ids = rng.integers(1, 25, size=10_000, dtype=np.int32)
    socs = rng.uniform(0, 90, size=10_000).astype(np.float32)
    vec = lookup_vectorized(arr32, ids, socs)
    sca = np.array([lookup_scalar(arr32, c, s) for c, s in zip(ids, socs)],
                   dtype=np.float32)
    return float(np.max(np.abs(vec - sca)) * 1e6)


def run_benchmark(arr32: np.ndarray, rng: np.random.Generator):
    scalar_n = int(os.environ.get("VSOC_R10_SCALAR_BENCH", DEFAULT_SCALAR_BENCH))
    vector_n = int(os.environ.get("VSOC_R10_VECTOR_BENCH", DEFAULT_VECTOR_BENCH))
    if scalar_n < 1_000 or vector_n < 10_000:
        raise ValueError("R10 benchmark needs >=1,000 scalar and >=10,000 vector lookups")
    bench_rows = []
    scalar_ids = rng.integers(1, 25, size=scalar_n)
    scalar_soc = rng.uniform(0, 90, size=scalar_n)
    for i in range(1_000):
        lookup_scalar(arr32, scalar_ids[i], scalar_soc[i])
    t0 = time.perf_counter()
    checksum_scalar = 0.0
    for cid, soc in zip(scalar_ids, scalar_soc):
        checksum_scalar += lookup_scalar(arr32, cid, soc)
    scalar_seconds = time.perf_counter() - t0
    bench_rows.append({
        "benchmark": "scalar_python_float32", "n_lookups": scalar_n,
        "elapsed_seconds": scalar_seconds,
        "lookups_per_second": scalar_n / scalar_seconds,
        "mean_time_per_lookup_us": scalar_seconds / scalar_n * 1e6,
        "checksum": checksum_scalar,
    })
    vector_ids = rng.integers(1, 25, size=vector_n, dtype=np.int32)
    vector_soc = rng.uniform(0, 90, size=vector_n).astype(np.float32)
    _ = lookup_vectorized(arr32, vector_ids[:10_000], vector_soc[:10_000])
    t0 = time.perf_counter()
    vec = lookup_vectorized(arr32, vector_ids, vector_soc)
    vector_seconds = time.perf_counter() - t0
    bench_rows.append({
        "benchmark": "vectorized_numpy_float32", "n_lookups": vector_n,
        "elapsed_seconds": vector_seconds,
        "lookups_per_second": vector_n / vector_seconds,
        "mean_time_per_lookup_us": vector_seconds / vector_n * 1e6,
        "checksum": float(np.nansum(vec)),
    })
    benchmark = pd.DataFrame(bench_rows)
    benchmark["python_version"] = platform.python_version()
    benchmark["platform"] = platform.platform()
    return benchmark


def condition_support(canonical: pd.DataFrame):
    return (canonical.groupby("Condition_ID", as_index=False, observed=True)
            .agg(n_nodes=("SoC", "size"),
                 minimum_cell_support=("n_cells_at_soc", "min"),
                 median_cell_support=("n_cells_at_soc", "median"),
                 minimum_support_fraction=("support_fraction", "min"),
                 n_nodes_support_1=("n_cells_at_soc", lambda x: int(np.sum(np.asarray(x) == 1))),
                 n_nodes_support_lt_half=("support_fraction", lambda x: int(np.sum(np.asarray(x) < 0.5)))))


def build_audit(root: Path, arr32: np.ndarray, mono: pd.DataFrame,
                consistency_uV: float, n_finite: int):
    audit_rows = []
    def record(item, status, value, criterion, note=""):
        audit_rows.append({"item": item, "status": status, "value": value,
                           "criterion": criterion, "note": note})

    record("canonical_LUT_shape", "PASS" if arr32.shape == (24, 91) else "FAIL",
           str(arr32.shape), "(24, 91)")
    record("canonical_LUT_finite", "PASS" if n_finite == EXPECTED_VALUES else "FAIL",
           f"{n_finite}/{EXPECTED_VALUES}", "2184/2184 finite")
    record("canonical_LUT_monotone",
           "PASS" if (mono["monotonicity_violations"] == 0).all() else "FAIL",
           int(mono["monotonicity_violations"].sum()), "0 violations")
    record("scalar_vector_lookup_consistency",
           "PASS" if consistency_uV < 1.0 else "REVIEW", consistency_uV,
           "<1 microvolt numerical difference")

    r9qa_path = root / "R9_final_LUT" / "R9_condition_QA.csv"
    r9qa = pd.read_csv(r9qa_path)
    require_columns(r9qa, ["n_missing_final_nodes"], "R9_condition_QA")
    r9_missing = int(r9qa["n_missing_final_nodes"].sum())
    record("common_0_100_LUT_support", "REVIEW", r9_missing,
           "0 missing would be required for full 0–100 common LUT",
           "Full 0–100 common grid is not supported; final common domain restricted to 0–90.")

    # Optional historical comparisons only. They do not validate/re-fit the
    # canonical LUT, and absent source files produce no fabricated PASS entries.
    r5_path = root / "R5_out_of_sample_validation" / "R5_validation_summary.csv"
    if r5_path.exists():
        r5 = pd.read_csv(r5_path)
        require_columns(r5, ["analysis", "SoC_band", "validation", "mean_MAE_mV"], "R5 summary")
        main = r5.loc[(r5["analysis"] == "broad_52_132") & (r5["SoC_band"] == "0-90")]
        for validation in ("leave_one_cell_out", "discharge_rate_level_holdout",
                           "temperature_level_holdout"):
            q = main.loc[main["validation"] == validation]
            if len(q):
                record(f"R5_{validation}_MAE_mV", "PASS", float(q.iloc[0]["mean_MAE_mV"]),
                       "reported descriptive OOS result", "No pass/fail accuracy threshold imposed.")

    r61_path = (root / "R6_high_soc_guard" / "R6_1_unguarded_comparison"
                / "unguarded_vs_guarded_summary.csv")
    if r61_path.exists():
        r61 = pd.read_csv(r61_path)
        require_columns(r61, ["analysis", "SoC_band", "method", "mean_MAE_mV"], "R6.1 summary")
        high = r61.loc[(r61["analysis"] == "broad_52_132") & (r61["SoC_band"] == "90-100")]
        ng, g95 = high.loc[high["method"] == "no_guard"], high.loc[high["method"] == "guard_95"]
        if len(ng) and len(g95):
            a, b = float(ng.iloc[0]["mean_MAE_mV"]), float(g95.iloc[0]["mean_MAE_mV"])
            record("R6_no_guard_vs_guard95", "PASS" if a < b else "FAIL",
                   f"no_guard={a:.3f} mV; guard95={b:.3f} mV",
                   "no_guard MAE < guard95 MAE", "Supports removal of anchor/guard mechanism.")

    r7_path = root / "R7_final_inference" / "R7_model_summary.csv"
    if r7_path.exists():
        r7m = pd.read_csv(r7_path)
        require_columns(r7m, ["R2_age_adjusted", "R2_unadjusted"], "R7 model summary")
        diffs = (r7m["R2_age_adjusted"] - r7m["R2_unadjusted"])
        record("R7_age_adjustment_R2", "PASS" if bool((diffs > 0).all()) else "REVIEW",
               diffs.round(6).tolist(), "Adjusted R2 greater than unadjusted in all 3 bands")

    r8_path = root / "R8_cell_bootstrap" / "R8_cell_bootstrap_contrast_summary.csv"
    if r8_path.exists():
        r8 = pd.read_csv(r8_path)
        require_columns(r8, ["contrast_family", "sign_stability", "bootstrap_CI_excludes_zero"],
                        "R8 bootstrap summary")
        aggressive = r8.loc[r8["contrast_family"].isin(
            ["rate_0.7C_minus_2C", "rate_2C_minus_1C"])]
        if aggressive.empty:
            # Avoid falsely passing an empty group (all([]) == True).
            record("R8_aggressive_rate_sign_stability", "REVIEW", np.nan, ">=0.999",
                   "No eligible aggressive-rate contrasts in R8 source.")
            record("R8_aggressive_rate_bootstrap_CI", "REVIEW", "0/0", "eligible CIs exclude zero",
                   "No eligible aggressive-rate contrasts in R8 source.")
        else:
            stable = pd.to_numeric(aggressive["sign_stability"], errors="coerce")
            ci = aggressive["bootstrap_CI_excludes_zero"]
            # Same result for normal true/false CSV input, explicit for strings.
            ci_bool = ci.map(lambda x: x if isinstance(x, (bool, np.bool_)) else
                             str(x).strip().lower() == "true")
            record("R8_aggressive_rate_sign_stability",
                   "PASS" if bool((stable >= 0.999).all()) else "REVIEW",
                   float(stable.min()), ">=0.999")
            record("R8_aggressive_rate_bootstrap_CI",
                   "PASS" if bool(ci_bool.all()) else "REVIEW", int(ci_bool.sum()),
                   f"{len(aggressive)}/{len(aggressive)} CI exclude zero")
    return pd.DataFrame(audit_rows)


def make_summary(arr16: np.ndarray, arr32: np.ndarray, qerr16: np.ndarray):
    return pd.DataFrame([{
        "final_common_soc_domain": "0-90% inclusive",
        "n_conditions": N_CONDITIONS,
        "soc_nodes_per_condition": N_SOC,
        "total_LUT_values": int(arr32.size),
        "float32_bytes": int(arr32.nbytes),
        "float32_KiB": float(arr32.nbytes / 1024),
        "float16_bytes": int(arr16.nbytes),
        "float16_KiB": float(arr16.nbytes / 1024),
        "float16_max_quantization_error_mV": float(np.max(qerr16) * 1000),
        "lookup_complexity": "O(1)",
        "lookup_operation": "two adjacent LUT reads + linear interpolation",
        "high_soc_guard": "removed",
        "common_0_100_LUT": "not claimed",
        "boundary_90_100": "condition-specific measured-support sensitivity only",
    }])


def run(root: Path, out: Path):
    canonical = load_canonical(root / "R9_final_LUT" / "R9_final_LUT_long.csv")
    matrix, arr64, arr32, arr16, mono, n_finite, all_monotone = prepare_lut(canonical)
    qerr16, memory_quant = quantization_qa(arr64, arr32, arr16)
    rng = np.random.default_rng(RNG_SEED)
    consistency_uV = check_lookup_consistency(arr32, rng)
    benchmark = run_benchmark(arr32, rng)
    support = condition_support(canonical)
    audit = build_audit(root, arr32, mono, consistency_uV, n_finite)
    summary = make_summary(arr16, arr32, qerr16)
    out.mkdir(parents=True, exist_ok=True)
    export_csv = {
        "R10_final_LUT_0_90_long.csv": canonical,
        "R10_final_LUT_0_90_matrix.csv": matrix,
        "R10_memory_quantization.csv": memory_quant,
        "R10_lookup_benchmark.csv": benchmark,
        "R10_monotonicity_QA.csv": mono,
        "R10_condition_support_0_90.csv": support,
        "R10_final_audit.csv": audit,
        "R10_final_summary.csv": summary,
    }
    for filename, frame in export_csv.items():
        frame.to_csv(out / filename, index=(filename == "R10_final_LUT_0_90_matrix.csv"))
    np.save(out / "R10_final_LUT_0_90_float32.npy", arr32)
    np.save(out / "R10_final_LUT_0_90_float16.npy", arr16)

    print("\n" + "=" * 82)
    print("R10 FINAL COMPUTATIONAL AUDIT COMPLETE")
    print("=" * 82)
    for name, table in (
        ("FINAL LUT STRUCTURE", summary),
        ("MEMORY / QUANTIZATION", memory_quant),
        ("LOOKUP BENCHMARK", benchmark[["benchmark", "n_lookups", "elapsed_seconds",
                                         "lookups_per_second", "mean_time_per_lookup_us"]]),
        ("0–90% CONDITION SUPPORT", support),
        ("FINAL CONSISTENCY AUDIT", audit),
    ):
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
    print("\nIMPORTANT: Python/NumPy Colab-reference benchmark; NOT an embedded-MCU guarantee.")
    print("\nStatus counts:")
    print(audit["status"].value_counts())
    print("\nOutput directory:", out)
    return export_csv, arr32, arr16


def main():
    root = resolve_root()
    explicit = os.environ.get("VSOC_R10_OUTPUT_DIR")
    out = Path(explicit).expanduser() if explicit else root / "R10_final_package"
    print("Input:", root / "R9_final_LUT" / "R9_final_LUT_long.csv")
    run(root, out)


if __name__ == "__main__":
    main()
