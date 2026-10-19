# -*- coding: utf-8 -*-
"""CALCE V--SoC | verified scientific payloads with dynamic audit provenance, v3.

One standalone Colab cell / one Python GitHub script. Run AFTER the audited
R2.2, coverage_80, 4-model age, unified and 5-model representation stages.

NO legacy R5/R7/R8 tables; NO unpaired high-SoC guard; NO broad wildcard
CSV/NPY inclusion. Fail closed on missing/ambiguous/outdated inputs.

Local configuration (optional):
  SOC_ROOT=/content/revision_R2_2_global
  SOC_PACKAGE_OUT=/content/SoC_Final_Submission
  SOC_INPUT_<KEY>=/path/to/file-or-archive.zip
Examples: SOC_INPUT_AGE=.../all_model_contrasts.csv
          SOC_INPUT_REP_GAINS=.../SoC_step5_ablation_...zip
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import math
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SOC_ROOT", "/content/revision_R2_2_global")).expanduser().resolve()
OUTPUT = Path(os.environ.get("SOC_PACKAGE_OUT", str(ROOT.parent / "SoC_Final_Submission"))).expanduser().resolve()
PACKAGE = "SoC_Final_Submission"

# Audited in Colab on 2026-10-18; pin the current stage identity and fail
# closed if any intermediate was replaced silently.
STAGE_ZIPS = {
    "coverage": ("SoC_band_coverage_ConditionvCodes2.zip", "f408ad28d0d8563f7768bc9d90d51aa130c77f8c089364b463839c2a5efabe7d"),
    "age": ("SoC_age_sensitivity_ConditionvCodes2.zip", "4f595cbae0945c3095e2362bcba4b20bfe00656b956bf1f09e4960937ec59014"),
    "unified": ("SoC_unified_ConditionvCodes2.zip", "2ca4399fed6adaeedabc2d499cb4916dac2a69a9a4499c8fb87f1eb311ff2b4d"),
    "support_audit": ("SoC_coverage_audit_ConditionvCodes2.zip", "88e9982e43292c9adfab8aec8d428b56468234583e9e6e62a219b8ef86502440"),
    "representations": ("SoC_step5_ablation_ConditionvCodes2.zip", "d91cd2a8ba1af34713f4af42c826165a0cdd0ed25e9482a03f61d7f18452aba6"),
}

# Baseline: independently audited final submission, 2026-10-18.
# These hashes describe the uncompressed, exact scientific payload bytes.
# A ZIP container may be regenerated with different metadata/compression.
BASELINE_SCIENCE_SHA256 = {'cells': '5235543f70469e799149ee81501479f381f22fda1631147ee71028cfd2bdcc0a', 'cycles': '4fcf66cd4e93a2fa8c08cef70ac6a87b86c3e1cc157ca301178a008d80501475', 'coverage': '25e192affa42a12de06bc48e7183217f17f104e15cd80efc6807b4a054b79ce7', 'age': '5d7a0b1ee88d964a525c985fd4b0f35b290e81c51d41002c547a1b06e11860e4', 'validation': '4f330bb333e272217fd00062524e3ccc214bf8f0800519a727806140f2c71fbd', 'validation_cond': '2608442372ef9e752475d66f9065e2e099783be210369bd400a6ea33c3ec1cb9', 'repr_main': '8132822d7e376a612b60150bce960a89d7092109f5c9baf73c91f64faeacaa9d', 'repr_gains': 'd520d8b589202f3987be8036fecdb7a2f5d47e5bb588ea37ed1092fe3d7daad2', 'lut_long': 'a16861d1537efe6c1e0a6c2db11f84260fa0b1459146b5930a89f450ba2d2aa7', 'lut32': '2dc424f9e539cea6d37ea57ca64eb7a35d689467d6495dbd622c3093e3e6a2ba', 'repr_lut32': '2dc424f9e539cea6d37ea57ca64eb7a35d689467d6495dbd622c3093e3e6a2ba'}
BASELINE_SUPPORT_SHA256 = {'Condition5_node_support_audit.csv': '6d45d6bcace6acd70cc5a5bd744d443b0a3c93deeaee982c7b96341900f69ac8', 'audit_config.json': '850f9cbe5dae16007eef3f43532979a11d8f6904d239bb6a88a4113102720ba2', 'cell_exclusion_reasons.csv': '1443fb4bbe535bef4ebfd200a32e35f06b5179f362d11a5e87230eddeb7a66bf', 'coverage_by_condition_and_threshold.csv': '5a3941f2fd3a2d8b44be71679872c763d3e7579efbeb9ec76eb614acb237f00e', 'primary_coverage_overall.csv': 'c43cbf9992bca02f2ef1055da2eea68c5aaf939555179d8d11f04e59c10a6554', 'support_threshold_and_C5_sensitivity.csv': 'ca888a40fd2554e057d9a840f9bbb652cfa43dc47a5b326ec3248deae70753bb'}
BASELINE_STAGE_MEMBERS = {
    "coverage": {"contrasts_all_rules.csv": BASELINE_SCIENCE_SHA256["coverage"]},
    "age": {"all_model_contrasts.csv": BASELINE_SCIENCE_SHA256["age"]},
    "unified": {
        "validation_summary.csv": BASELINE_SCIENCE_SHA256["validation"],
        "validation_summary_by_condition.csv": BASELINE_SCIENCE_SHA256["validation_cond"],
        "LUT_all_0_90_inclusive.csv": BASELINE_SCIENCE_SHA256["lut_long"],
        "final_LUT_float32.npy": BASELINE_SCIENCE_SHA256["lut32"],
    },
    "support_audit": {k: v for k, v in BASELINE_SUPPORT_SHA256.items()
                      if k != "audit_config.json"},
    "representations": {
        "MAIN_common_five.csv": BASELINE_SCIENCE_SHA256["repr_main"],
        "MAIN_gains_vs_condition_iso.csv": BASELINE_SCIENCE_SHA256["repr_gains"],
        "final_condition_LUT_float32.npy": BASELINE_SCIENCE_SHA256["repr_lut32"],
    },
}

INPUT_STAGE = {
    "coverage": "coverage", "age": "age",
    "validation": "unified", "validation_cond": "unified", "lut_long": "unified", "lut32": "unified",
    "repr_main": "representations", "repr_gains": "representations", "repr_lut32": "representations",
}
EXPECTED_VAL = {
    ("full_history", "LOCO"): (180, 15856, 15806, 5.240437, 6.114910),
    ("full_history", "rate_holdout"): (62, 5642, 5642, 19.531124, 20.877565),
    ("full_history", "temperature_holdout"): (96, 8736, 8736, 32.141306, 33.121660),
    ("broad_52_132", "LOCO"): (179, 15639, 15639, 4.353667, 4.755326),
    ("broad_52_132", "rate_holdout"): (62, 5642, 5096, 18.041400, 19.618651),
    ("broad_52_132", "temperature_holdout"): (96, 8736, 8112, 33.189926, 34.688317),
}
EXPECTED_PERCENT_GAINS = {"LOCO": 95.80, "rate_holdout": 81.86, "temperature_holdout": 62.85}
EXPECTED_GAIN_CI = {"LOCO": (99.334759, 98.288425, 100.305394),
                    "rate_holdout": (81.410509, 79.252341, 83.302963),
                    "temperature_holdout": (56.138392, 54.858995, 57.583692)}

FIGURE_DPI = int(os.environ.get("SOC_FIGURE_DPI", "300"))
require_dpi = 50 <= FIGURE_DPI <= 600
if not require_dpi: raise ValueError("SOC_FIGURE_DPI must be between 50 and 600")
SOURCE_NAMES = {
    "cells": "cell_profiles_nominal_concordant.csv",
    "cycles": "global_cycle_profiles.pkl",
    "coverage": "contrasts_all_rules.csv",
    "age": "all_model_contrasts.csv",
    "validation": "validation_summary.csv",
    "validation_cond": "validation_summary_by_condition.csv",
    "repr_main": "MAIN_common_five.csv",
    "repr_gains": "MAIN_gains_vs_condition_iso.csv",
    "lut_long": "LUT_all_0_90_inclusive.csv",
    "lut32": "final_LUT_float32.npy",
    "repr_lut32": "final_condition_LUT_float32.npy",
}
ARCHIVE_PREFIXES = ("SoC_unified_", "SoC_band_coverage_", "SoC_age_sensitivity_", "SoC_step5_ablation_")
CONDITION_META = pd.DataFrame([
    (1 + 6*t + 2*r + k, T, cutoff, discharge)
    for t, T in enumerate((10, 25, 45, 60))
    for r, discharge in enumerate(("0.7C", "1C", "2C"))
    for k, cutoff in enumerate(("C/5", "C/40"))
], columns=["Condition_ID", "Temperature", "Rate", "Charge_Rate"])
FAMILIES = {
    "cutoff_C40_minus_C5": "C/40 - C/5",
    "rate_0.7C_minus_1C": "0.7C - 1C",
    "rate_2C_minus_1C": "2C - 1C",
    "rate_0.7C_minus_2C": "0.7C - 2C",
    "temperature_60_minus_25": "60 C - 25 C",
}
FAMILY_FILES = ([f"family_{i:02d}_T_sweep.png" for i in range(1, 7)] +
                [f"family_{i:02d}_Rate_sweep.png" for i in range(7, 19)] +
                [f"family_{i:02d}_DR_sweep.png" for i in range(19, 27)])
EXTRA_FIGURES = ["fig_cycle_index_by_condition.png", "fig_lut_cell_support.png",
                 "fig_age_adjusted_contrasts.png", "fig_oos_validation.png",
                 "fig_thermal_model_sensitivity.png"]
MANUSCRIPT_FIGURES = ["family_02_T_sweep.png", "family_05_T_sweep.png",
                      "family_08_Rate_sweep.png", "family_18_Rate_sweep.png",
                      "family_19_DR_sweep.png", "family_26_DR_sweep.png",
                      "fig_thermal_model_sensitivity.png", "fig_lut_cell_support.png"]


def require_cols(d: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(d.columns))
    if missing:
        raise ValueError(f"{label}: eksik sutunlar: {missing}")


def assert_unique(d: pd.DataFrame, cols: list[str], label: str) -> None:
    if d.duplicated(cols).any():
        raise ValueError(f"{label}: tekil olmasi gereken anahtar tekrarli: {cols}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def candidates_for(name: str) -> list[tuple[str, Path, str | None]]:
    """File candidates from exact basename; archives limited to verified stage prefixes."""
    candidates: list[tuple[str, Path, str | None]] = []
    seen: set[str] = set()
    roots = [ROOT]
    if ROOT.parent.exists():
        roots.extend(p for p in ROOT.parent.iterdir() if p.is_dir() and
                     (p.name.startswith(ARCHIVE_PREFIXES) or p == ROOT.parent))
    for root in roots:
        if not root.exists():
            continue
        # The unified output ZIP's 'inputs/' may contain cell profiles.
        for p in root.rglob(name) if root != ROOT.parent else root.glob(name):
            if p.is_file() and str(p) not in seen and OUTPUT not in p.parents:
                seen.add(str(p)); candidates.append(("file", p, None))
    if ROOT.parent.exists():
        for p in ROOT.parent.iterdir():
            if p.is_file() and p.suffix.lower() == ".zip" and p.name.startswith(ARCHIVE_PREFIXES):
                with zipfile.ZipFile(p) as zf:
                    matches = [n for n in zf.namelist()
                               if Path(n).name == name and not n.endswith("/")]
                for member in matches:
                    candidates.append(("zip", p, member))
    return candidates


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_dynamic_support_config(raw: bytes, unified_zip_sha: str) -> None:
    """Compare stable scientific policy, validate per-run provenance independently.

    Exact CSV hashes are checked separately. Config timestamp/source archive
    SHA256 are volatile and must not be compared to an older JSON byte stream.
    """
    try:
        config = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as ex:
        raise ValueError("support_audit audit_config.json is unreadable JSON") from ex
    require(isinstance(config, dict), "support_audit configuration must be an object")
    keys = {
        "source_zip_sha256", "timestamp_utc", "python", "pandas", "numpy",
        "support_thresholds", "minimum_paired_nodes", "refitted",
        "reference_summary_check",
    }
    require(set(config) == keys,
            f"support_audit: config schema changed; missing={keys-set(config)}, extra={set(config)-keys}")
    # Stable scientific method fields from independently approved 18-Oct release
    expected = {
        "support_thresholds": [1, 2, 3],
        "minimum_paired_nodes": 5,
        "refitted": False,
        "reference_summary_check": "passed",
    }
    for key, target in expected.items():
        require(type(config[key]) is type(target) and config[key] == target,
                f"support_audit scientific protocol changed: {key}: {config[key]!r} vs {target!r}")
    src_sha = config["source_zip_sha256"]
    require(isinstance(src_sha, str) and re.fullmatch(r"[0-9a-f]{64}", src_sha) is not None,
            "support_audit: malformed unified source SHA256")
    require(src_sha == unified_zip_sha,
            "support_audit: audit is based on a different Unified ZIP; re-run audit on the selected unified archive")
    stamp = config["timestamp_utc"]
    try:
        parsed_stamp = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
    except ValueError:
        parsed_stamp = None
    require(parsed_stamp is not None and parsed_stamp.tzinfo is not None,
            "support_audit: UTC timestamp missing or invalid")
    require(parsed_stamp.utcoffset().total_seconds() == 0,
            "support_audit: timestamp not UTC")
    for version_field in ("python", "pandas", "numpy"):
        value = config[version_field]
        require(isinstance(value, str) and bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.+_-]*)", value)),
                f"support_audit: malformed environment version: {version_field}={value!r}")
    print("[PASS] support_audit audit_config.json: stable scientific settings identical")
    print("[PASS] support_audit source ZIP fingerprint points to current verified Unified ZIP")
    print("[INFO] audit timestamp/software versions are provenance metadata, not frozen science")


def verified_archives() -> dict[str, Path]:
    """Validate ZIP structure, science hashes and audited methodological metadata.

    Scientific reference: SHA256 of exact input members in the 18-Oct approved
    final package. No numeric rounding or silent science version substitution.
    Other ancillary ZIP members are outside the fingerprinted release contract.
    """
    found: dict[str, Path] = {}
    for stage, (filename, original_zip_sha) in STAGE_ZIPS.items():
        override = os.environ.get(f"SOC_STAGE_{stage.upper()}", "").strip()
        path = Path(override).expanduser().resolve() if override else ROOT.parent / filename
        require(path.is_file(), f"{stage}: input stage ZIP missing: {path}")
        archive_sha = sha256_file(path)
        expected_members = BASELINE_STAGE_MEMBERS[stage]
        with zipfile.ZipFile(path) as zf:
            require(zf.testzip() is None, f"{stage}: ZIP CRC error")
            info = [i for i in zf.infolist() if not i.is_dir()]
            for basename, expected_sha in expected_members.items():
                members = [i for i in info if Path(i.filename).name == basename]
                require(len(members) == 1,
                        f"{stage}: exactly one {basename} required, found {len(members)}")
                actual_sha = hashlib.sha256(zf.read(members[0])).hexdigest()
                require(actual_sha == expected_sha,
                        f"{stage}: SCIENTIFIC PAYLOAD DIFFERENT: {basename}\n"
                        f"Reference SHA256: {expected_sha}\nCurrent SHA256:   {actual_sha}\n"
                        "STOP: do not change expected fingerprints; inspect the rerun.")
            if stage == "support_audit":
                conf_members = [i for i in info if Path(i.filename).name == "audit_config.json"]
                require(len(conf_members) == 1,
                        f"support_audit: exactly one audit_config.json required, found {len(conf_members)}")
                require("unified" in found, "Verify unified archive before support audit")
                verify_dynamic_support_config(zf.read(conf_members[0]), sha256_file(found["unified"]))
        if archive_sha == original_zip_sha:
            print(f"[ARCHIVE MATCH] {stage}: ZIP SHA256 unchanged; {len(expected_members)} payloads match")
        else:
            print(f"[CONTENT-EQUIVALENT ZIP] {stage}: ZIP container SHA changed, "
                  f"but {len(expected_members)} audited payload(s) match exact reference bytes")
            print(f"  prior ZIP:   {original_zip_sha}")
            print(f"  current ZIP: {archive_sha}")
            print("  Note: other ZIP members/metadata have NOT been proved identical")
        found[stage] = path
    return found


def audited_source(key: str, arches: dict[str, Path]) -> tuple[bytes, str]:
    filename = SOURCE_NAMES[key]
    if key in ("cells", "cycles"):
        path = ROOT / filename
        require(path.is_file(), f"R2.2 upstream input missing: {path}")
        return path.read_bytes(), str(path)
    stage = INPUT_STAGE[key]
    path = arches[stage]
    with zipfile.ZipFile(path) as zf:
        matches = [n for n in zf.namelist() if Path(n).name == filename and not n.endswith("/")]
        require(len(matches) == 1,
                f"{stage} archive: {filename} must occur exactly once, got {matches}")
        return zf.read(matches[0]), f"{path}!{matches[0]}"


def load_source(key: str) -> tuple[bytes, str]:
    name = SOURCE_NAMES[key]
    override = os.environ.get(f"SOC_INPUT_{key.upper()}", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        require(path.exists(), f"SOC_INPUT_{key.upper()}: yol bulunamadi: {path}")
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                matches = [n for n in zf.namelist() if Path(n).name == name]
                require(len(matches) == 1, f"{path}: {name} ZIP icinde tek olmali: {matches}")
                return zf.read(matches[0]), f"{path}!{matches[0]}"
        require(path.is_file() and path.name == name,
                f"{key} icin {name} adli CSV/NPY/PKL veya uygun ZIP secilmeli")
        return path.read_bytes(), str(path)
    choices = candidates_for(name)
    # Identical byte hashes across a loose file and its ZIP can be deduplicated.
    unique: dict[str, tuple[bytes, str]] = {}
    for mode, path, member in choices:
        blob = path.read_bytes() if mode == "file" else zipfile.ZipFile(path).read(member)
        checksum = hashlib.sha256(blob).hexdigest()
        unique.setdefault(checksum, (blob, str(path) if member is None else f"{path}!{member}"))
    if len(unique) != 1:
        example = f"SOC_INPUT_{key.upper()}=/content/.../{name}"
        raise FileNotFoundError(
            f"{name}: {'bulunamadi' if not unique else 'farkli surumler bulundu'}; "
            f"kaynak sayisi={len(unique)}. Kesin yolu belirtin: {example}. "
            f"Adaylar: {[x[1] for x in unique.values()]}")
    return next(iter(unique.values()))


def read_csv(raw: bytes, label: str) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(raw), low_memory=False)
    d.columns = d.columns.str.strip()
    require(len(d) > 0, f"{label}: bos dosya")
    return d


def verify_contrasts(age: pd.DataFrame, cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_cols = ["model", "reference_cycle", "SoC_band", "contrast_id", "contrast_family",
                "lhs_condition", "rhs_condition", "inference_eligible", "status",
                "delta_adjusted_mV", "bootstrap_CI_low_mV", "bootstrap_CI_high_mV"]
    cov_cols = ["rule", "SoC_band", "contrast_id", "inference_eligible",
                "delta_age_adjusted_mV", "n_cells_lhs", "n_cells_rhs"]
    require_cols(age, age_cols, "age"); require_cols(cov, cov_cols, "coverage")
    require(set(age["model"]) == {"reference", "condition_slopes", "cell_balanced", "window_52_132"},
            "Age model isimleri beklenen dort modelden farkli")
    ref = age[(age.model == "reference") & (age.reference_cycle == 108)].copy()
    cover = cov[cov.rule == "coverage_80"].copy()
    require(len(ref) == len(cover) == 81, "%80 kapsam/4-model referansta 81 kontrast olmali")
    assert_unique(ref, ["SoC_band", "contrast_id"], "ref")
    assert_unique(cover, ["SoC_band", "contrast_id"], "coverage80")
    pairs = ref.merge(cover[["SoC_band", "contrast_id", "delta_age_adjusted_mV",
                             "inference_eligible", "n_cells_lhs", "n_cells_rhs"]],
                      on=["SoC_band", "contrast_id"], validate="one_to_one", suffixes=("", "_coverage"))
    require(len(pairs) == 81, "%80 kapsam ve model referans anahtarlari uyusmuyor")
    require((pairs.inference_eligible.astype(str).str.lower() ==
             pairs.inference_eligible_coverage.astype(str).str.lower()).all(),
            "Iki analizin cikarimsal uygunluk bayraklari farkli")
    a = pd.to_numeric(pairs.delta_adjusted_mV, errors="coerce").to_numpy(float)
    b = pd.to_numeric(pairs.delta_age_adjusted_mV, errors="coerce").to_numpy(float)
    require(np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-5),
            "4-model ve coverage80 sayisal referans sonuclari uyusmuyor (>1e-5mV)")
    eligible = ref.inference_eligible.astype(str).str.lower().eq("true")
    aggressive = ref.contrast_family.isin(("rate_2C_minus_1C", "rate_0.7C_minus_2C"))
    require(int(eligible.sum()) == 75 and int((~eligible).sum()) == 6,
            "Beklenen 75 cikarimsal + 6 betimsel ayrimi bulunamadi")
    require(int((eligible & aggressive).sum()) == 42 and int((~eligible & aggressive).sum()) == 6,
            "Beklenen 42 cikarimsal + 6 betimsel 2C sonucu bulunamadi")
    for col in ["p_value", "cluster_CI_low_mV", "cluster_CI_high_mV", "q_BH81",
                "bootstrap_CI_low_mV", "bootstrap_CI_high_mV"]:
        require(col in ref, f"Guncel yas cikarisinda {col} sutunu bulunamadi")
        require(ref.loc[~eligible, col].isna().all(),
                f"Betimsel 6 satirda {col} bos olmali")
    require(len(age) == 486, "4-model analizde 486 satir (81x6) bekleniyor")
    return ref, age


def verify_validation(validation: pd.DataFrame, bycond: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = ["analysis", "validation", "method", "comparison", "band", "MAE_mV",
           "n_planned_cells", "n_evaluated_cells"]
    require_cols(validation, req, "unified validation")
    require_cols(bycond, req + ["Condition_ID"], "unified by condition")
    p = validation[(validation.method == "unified") & (validation.comparison == "own_support") &
                   (validation.band == "all_0_90_inclusive")].copy()
    c = bycond[(bycond.method == "unified") & (bycond.comparison == "own_support") &
               (bycond.band == "all_0_90_inclusive")].copy()
    assert_unique(p, ["analysis", "validation"], "validation")
    require(len(p) == 6 and set(p.analysis) == {"full_history", "broad_52_132"},
            "Unified final alanda her kapsam icin 3 dogrulama gerekir")
    require(set(p.validation) == {"LOCO", "rate_holdout", "temperature_holdout"},
            "Unified dogrulama protokolleri degismis")
    require(len(c) > 0 and c.Condition_ID.between(1, 24).all(),
            "validation_by_condition beklenmedik")
    require_cols(p, ["n_target", "n_used", "mean_cell_RMSE_mV"], "unified validation QA")
    for row in p.itertuples(index=False):
        n_cells, n_target, n_used, mae, rmse = EXPECTED_VAL[(row.analysis, row.validation)]
        require((int(row.n_evaluated_cells), int(row.n_target), int(row.n_used)) ==
                (n_cells, n_target, n_used),
                f"Coverage denominator mismatch: {row.analysis}/{row.validation}")
        require(abs(float(row.MAE_mV)-mae) <= .005 and
                abs(float(row.mean_cell_RMSE_mV)-rmse) <= .005,
                f"Manuscript/real-CALCE validation mismatch: {row.analysis}/{row.validation}")
    return p, c


def verify_repr(main: pd.DataFrame, gains: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_cols(main, ["analysis", "validation", "model", "comparison", "band", "MAE_mV"], "five-way main")
    require_cols(gains, ["analysis", "validation", "comparison", "left_model", "right_model"], "five-way gains")
    main = main[(main.comparison == "common_five") & (main.band == "all_0_90_inclusive")].copy()
    gains = gains[(gains.comparison == "common_five") &
                  gains.right_model.eq("condition_iso")].copy()
    require(len(main) == 30, "Beş model x iki analiz x uc protokol = 30 satir gerekir")
    require(set(main.model) == {"global_iso", "temperature_iso", "rate_iso",
                                "condition_raw", "condition_iso"}, "Besli model seti uyusmuyor")
    require(len(gains) == 24, "Dort kazanc x iki analiz x uc protokol = 24 satir gerekir")
    require(set(gains.left_model) == {"global_iso", "temperature_iso", "rate_iso", "condition_raw"},
            "Izotonik kazanc modelleri uyusmuyor")
    assert_unique(main, ["analysis", "validation", "model"], "representation")
    return main, gains


def verify_42_robust(age: pd.DataFrame) -> None:
    models = {"reference", "condition_slopes", "cell_balanced", "window_52_132"}
    m = age[age.reference_cycle.eq(108) & age.model.isin(models) &
            age.contrast_family.isin(("rate_2C_minus_1C", "rate_0.7C_minus_2C"))].copy()
    require_cols(m, ["point_sign", "bootstrap_CI_low_V", "bootstrap_CI_high_V"], "42 check")
    m["eligible"] = m.inference_eligible.astype(str).str.lower().eq("true")
    m["ci_excludes_zero"] = (m.bootstrap_CI_low_V.gt(0) | m.bootstrap_CI_high_V.lt(0))
    z = m.groupby(["SoC_band", "contrast_id"]).agg(
        models=("model", "nunique"), eligible=("eligible", "all"),
        n_signs=("point_sign", "nunique"), robust=("ci_excludes_zero", "all"))
    require(len(z) == 48 and z.models.eq(4).all(), "2C: expected 48 planned x 4 models")
    passed = z[z.eligible & z.n_signs.eq(1) & z.robust]
    require(len(passed) == 42, f"2C: 42 robust contrasts expected, got {len(passed)}")
    print("[PASS] 42/42 eligible 2C contrasts: four-model sign + bootstrap CI")


def verify_five_manuscript(main: pd.DataFrame, gains: pd.DataFrame,
                           val: pd.DataFrame) -> None:
    for test, expected in EXPECTED_PERCENT_GAINS.items():
        m = main[(main.analysis == "broad_52_132") & (main.validation == test)]
        g = float(m.loc[m.model.eq("global_iso"), "MAE_mV"].iloc[0])
        c = float(m.loc[m.model.eq("condition_iso"), "MAE_mV"].iloc[0])
        gain = 100 * (g-c)/g
        require(round(gain,2) == expected,
                f"{test}: manuscript percentage gain mismatch: {gain:.7f}%")
        x = gains[(gains.analysis == "broad_52_132") &
                  (gains.validation == test) & gains.left_model.eq("global_iso")]
        require(len(x) == 1, f"{test}: missing global-v-condition bootstrap gain")
        expected_row = EXPECTED_GAIN_CI[test]
        for key, target in zip(("gain_right_mV", "CI95_low_mV", "CI95_high_mV"), expected_row):
            require(abs(float(x.iloc[0][key])-target) < .005,
                    f"{test}: manuscript bootstrap {key} mismatch")
        actual = val[(val.analysis.eq("broad_52_132")) & (val.validation.eq(test))]
        require(len(actual) == 1 and abs(c-float(actual.iloc[0].MAE_mV)) < 1e-6,
                f"{test}: five-model condition ISO differs from unified result")
        print(f"[PASS] 52-132 {test}: gain {gain:.4f}%, {expected_row[0]:.3f}mV")
    comparison = main[main.model.isin(["condition_raw", "condition_iso"])].pivot(
        index=["analysis", "validation"], columns="model", values="MAE_mV")
    require((comparison.condition_iso-comparison.condition_raw).abs().max() < .07,
            "Manuscript isotonic effect >=0.07mV")


def write_audit_support_zip(archive: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wanted = {"primary_coverage_overall.csv", "support_threshold_and_C5_sensitivity.csv",
              "coverage_by_condition_and_threshold.csv", "Condition5_node_support_audit.csv",
              "cell_exclusion_reasons.csv", "audit_config.json"}
    with zipfile.ZipFile(archive) as zf:
        names = {Path(n).name: n for n in zf.namelist() if Path(n).name in wanted}
        require(wanted.issubset(names),
                f"Unified support audit: missing mandatory sources: {sorted(wanted-set(names))}")
        for name in sorted(wanted):
            (out/name).write_bytes(zf.read(names[name]))
    print(f"[PASS] Support audit: {len(wanted)} supporting records retained")


def clean_cell_data(raw: bytes) -> pd.DataFrame:
    d = read_csv(raw, "cell profiles")
    if "Voltage" in d and "V_cell_median" not in d:
        d = d.rename(columns={"Voltage": "V_cell_median"})
    req = ["sample_int", "Condition_ID", "SoC", "V_cell_median"]
    require_cols(d, req, "cell profiles")
    for c in ("sample_int", "Condition_ID", "SoC", "V_cell_median"):
        d[c] = pd.to_numeric(d[c], errors="raise")
    require(d.Condition_ID.between(1, 24).all(), "Hucre profili Condition_ID disinda")
    assert_unique(d, ["Condition_ID", "sample_int", "SoC"], "cell profiles")
    require(d.Condition_ID.nunique() == 24 and d.sample_int.nunique() == 180,
            "Tam gecmiste 24 kosul ve 180 fiziksel hucre bekleniyor")
    if "Temperature" in d: d = d.drop(columns=["Temperature"])
    if "Rate" in d: d = d.drop(columns=["Rate"])
    if "Charge_Rate" in d: d = d.drop(columns=["Charge_Rate"])
    return d.merge(CONDITION_META, on="Condition_ID", validate="many_to_one")


def build_support_and_curves(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = d[d.SoC.between(0, 90)].copy()
    summary = q.groupby(["Condition_ID", "SoC"], observed=True).agg(
        V_median=("V_cell_median", "median"),
        V_q25=("V_cell_median", lambda x: float(np.nanquantile(x, .25)) if x.notna().any() else np.nan),
        V_q75=("V_cell_median", lambda x: float(np.nanquantile(x, .75)) if x.notna().any() else np.nan),
        n_cells=("V_cell_median", lambda x: int(np.isfinite(x).sum()))).reset_index()
    counts = q.groupby("Condition_ID").sample_int.nunique().rename("retained_cells")
    complete = pd.MultiIndex.from_product([range(1, 25), range(91)],
                                          names=["Condition_ID", "SoC"]).to_frame(index=False)
    summary = complete.merge(summary, on=["Condition_ID", "SoC"], validate="one_to_one", how="left")
    summary["n_cells"] = summary["n_cells"].fillna(0).astype(int)
    summary = summary.merge(counts, on="Condition_ID", validate="many_to_one")
    summary["support_fraction"] = summary.n_cells/summary.retained_cells
    require(len(summary) == 2184 and summary.n_cells.ge(1).all(),
            "24x91 gridde tam en az-bir-hucre desteği yok")
    summary = summary.merge(CONDITION_META, on="Condition_ID", validate="many_to_one")
    support = summary[["Condition_ID", "SoC", "n_cells", "retained_cells", "support_fraction"]].copy()
    support["support_ge_2"] = support.n_cells.ge(2)
    support["support_ge_3"] = support.n_cells.ge(3)
    support["support_ge_50pct"] = support.support_fraction.ge(.5)
    support["support_ge_75pct"] = support.support_fraction.ge(.75)
    support["support_full"] = support.n_cells.eq(support.retained_cells)
    common = support.groupby("SoC").agg(
        at_least_one=("n_cells", lambda v: bool((v >= 1).all())),
        at_least_two=("n_cells", lambda v: bool((v >= 2).all())),
        at_least_three=("n_cells", lambda v: bool((v >= 3).all())),
        half_support=("support_ge_50pct", "all"),
        three_quarter_support=("support_ge_75pct", "all"),
        full_support=("support_full", "all"),
    ).reset_index()
    # R11 full-history expected domain counts.
    expected = [91, 41, 18, 13, 10, 10]
    counts_actual = [int(common[c].sum()) for c in common.columns[1:]]
    require(counts_actual == expected,
            f"R11 sayilari degismis: {counts_actual}; beklenen {expected}")
    return summary, support


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.12g")


def plot_families(summary: pd.DataFrame, out: Path) -> None:
    v = summary.V_median.to_numpy(float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    margin = max(.03, (hi - lo)*.035)
    ylim = (lo-margin, hi+margin)
    def draw(ids: list[int], labels: list[str], title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        nlines = 0
        for cid, lab in zip(ids, labels):
            q = summary[summary.Condition_ID.eq(cid)].sort_values("SoC")
            require(len(q) == 91, f"family plot {filename}: Condition {cid} eksik")
            x = q.SoC.to_numpy(float)
            y = q.V_median.to_numpy(float)
            line, = ax.plot(x, y, linewidth=1.7, label=lab)
            valid = q.n_cells.to_numpy(int) >= 2
            ax.fill_between(x, q.V_q25.to_numpy(float), q.V_q75.to_numpy(float),
                            where=valid, alpha=.13, color=line.get_color(), interpolate=False)
            weak = q.support_fraction.to_numpy(float) < .5
            if weak.any():
                ax.scatter(x[weak], y[weak], s=17, marker="o", facecolors="none",
                           edgecolors=line.get_color(), linewidths=0.8)
            nlines += 1
        require(nlines == len(ids), f"Family eksik: {filename}")
        ax.set(xlim=(0, 90), ylim=ylim, xlabel="State of charge (%)",
               ylabel="Terminal voltage (V)", title=title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out/filename, dpi=FIGURE_DPI, bbox_inches="tight"); plt.close(fig)
    ix = CONDITION_META.set_index(["Temperature", "Rate", "Charge_Rate"]).Condition_ID.to_dict()
    n = 1
    for cutoff in ("C/5", "C/40"):
        for dr in ("0.7C", "1C", "2C"):
            draw([ix[T,cutoff,dr] for T in (10,25,45,60)],
                 [f"{T} °C" for T in (10,25,45,60)],
                 f"Temperature sweep — {cutoff}, {dr}", f"family_{n:02d}_T_sweep.png")
            n += 1
    n = 7
    for T in (10,25,45,60):
        for dr in ("0.7C", "1C", "2C"):
            draw([ix[T,c,dr] for c in ("C/5", "C/40")], ["C/5", "C/40"],
                 f"Charge cut-off sweep — {T} °C, {dr}", f"family_{n:02d}_Rate_sweep.png")
            n += 1
    n = 19
    for T in (10,25,45,60):
        for cutoff in ("C/5", "C/40"):
            draw([ix[T,cutoff,dr] for dr in ("0.7C", "1C", "2C")],
                 ["0.7C", "1C", "2C"], f"Discharge-rate sweep — {T} °C, {cutoff}",
                 f"family_{n:02d}_DR_sweep.png")
            n += 1


def plot_cycle(raw: bytes, out: Path) -> None:
    # Trusted own R2.2 input only: pickle is unsafe for untrusted remote files.
    d = pd.read_pickle(io.BytesIO(raw))
    require_cols(d, ["matches_nominal", "Cycle_Index", "Condition_ID"], "cycle R2.2")
    d = d[d.matches_nominal.eq(True)].copy()
    require(d.Condition_ID.nunique() == 24 and len(d) == 83045,
            "Nominal cevrim sayisi/deney kosulu R2.2 ile uyusmuyor")
    vals = [d.loc[d.Condition_ID.eq(cid), "Cycle_Index"].to_numpy(float) for cid in range(1,25)]
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    ax.axhspan(52,132,alpha=.12, label="Broad Cycle_Index window (52–132)")
    ax.axhline(108,linestyle="--", linewidth=1.2, label="Reference cycle 108")
    ax.boxplot(vals, positions=range(1,25), widths=.6, showfliers=False)
    ax.set(xlim=(.3,24.7), xlabel="Condition ID", ylabel="Cycle_Index",
           title="Cycle-history distributions across operating conditions")
    ax.set_xticks(range(1,25)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/"fig_cycle_index_by_condition.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_support(support: pd.DataFrame, out: Path) -> None:
    matrix = support.pivot(index="Condition_ID", columns="SoC", values="n_cells").reindex(
        index=range(1,25), columns=range(91))
    require(matrix.notna().all().all(), "Support heatmap grid incomplete")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(matrix.to_numpy(float), origin="upper", interpolation="nearest", aspect="auto")
    bar = fig.colorbar(im, ax=ax); bar.set_label("Supporting physical cells")
    ax.set(xlabel="State of charge (%)", ylabel="Condition ID",
           title="Physical-cell support across the 0–90% grid")
    ax.set_xticks(range(0,91,10)); ax.set_xticklabels(range(0,91,10))
    ax.set_yticks(range(24)); ax.set_yticklabels(range(1,25))
    ax.axhline(3.5, lw=1.2); ax.axhline(4.5, lw=1.2)
    fig.tight_layout(); fig.savefig(out/"fig_lut_cell_support.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_inference(ref: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10.3, 17.5), sharex=True)
    rank = {"0-20": 0, "20-60": 1, "60-90": 2}
    markers = {"0-20": "o", "20-60": "s", "60-90": "^"}
    for ax, (family, label) in zip(axes, FAMILIES.items()):
        q = ref[ref.contrast_family.eq(family)].copy()
        q["_band_order"] = q.SoC_band.map(rank)
        q = q.sort_values(["contrast_id", "_band_order"]).reset_index(drop=True)
        y = np.arange(len(q))
        for band, marker in markers.items():
            mask = q.SoC_band.eq(band) & q.inference_eligible.astype(str).str.lower().eq("true")
            rows = q.loc[mask]
            if len(rows):
                x = rows.delta_adjusted_mV.to_numpy(float)
                low = rows.cluster_CI_low_mV.to_numpy(float)
                high = rows.cluster_CI_high_mV.to_numpy(float)
                require(np.isfinite(np.r_[x,low,high]).all(), f"Cluster CI eksik: {family}/{band}")
                ax.errorbar(x, y[mask.to_numpy()], xerr=np.stack([x-low, high-x]),
                            fmt=marker, ls="none", capsize=2, markersize=3, label=band)
            descriptive = q.SoC_band.eq(band) & ~mask & ~q.inference_eligible.astype(str).str.lower().eq("true")
            dq = q.loc[descriptive]
            if len(dq):
                x = dq.delta_adjusted_mV.to_numpy(float)
                finite = np.isfinite(x)
                ax.scatter(x[finite], y[descriptive.to_numpy()][finite],
                           marker="x", s=20, color="gray")
        ax.axvline(0, lw=.8, color="gray")
        ax.set_yticks(y); ax.set_yticklabels(q.contrast_id.tolist(), fontsize=6.7)
        ax.set_title(label)
    axes[-1].set_xlabel("Cycle-history-adjusted ΔV (mV)")
    axes[0].legend(title="SoC band", frameon=False, ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out/"fig_age_adjusted_contrasts.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_validation(validation: pd.DataFrame, out: Path) -> None:
    labels = {"LOCO": "Unseen cell", "rate_holdout": "Held-out 1C",
              "temperature_holdout": "Held-out temperature"}
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    order = list(labels)
    for scope, lab, dx in (("full_history", "Full history", -.12),
                           ("broad_52_132", "Cycle_Index 52–132", .12)):
        q = validation.set_index(["analysis", "validation"])
        nums = [float(q.loc[(scope, protocol), "MAE_mV"]) for protocol in order]
        ax.plot(np.arange(3)+dx, nums, marker="o", linewidth=1.5, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels([labels[x] for x in order])
    ax.set(ylabel="Mean cell MAE (mV)", title="Unified out-of-sample validation, SoC 0–90 inclusive")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/"fig_oos_validation.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_thermal(age: pd.DataFrame, out: Path) -> None:
    q = age[(age.reference_cycle == 108) &
            age.contrast_family.eq("temperature_60_minus_25")].copy()
    models = ["reference", "condition_slopes", "cell_balanced", "window_52_132"]
    model_labels = ["Reference", "Condition slopes", "Cell-balanced", "Window 52–132"]
    require(len(q) == 36, "Thermal sensitivity: 9 kontrast × 4 model gerekir")
    assert_unique(q, ["model", "SoC_band", "contrast_id"], "thermal")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), sharey=True)
    for ax, (rate, rate_code) in zip(axes, [("0.7C", "0.7C"), ("1C", "1C"), ("2C", "2C")]):
        qq = q[q.contrast_id.eq(f"temp60minus25_C40_DR{rate_code}")]
        for i, model in enumerate(models):
            z = qq[qq.model.eq(model)].set_index("SoC_band")
            require(len(z) == 3 and z.inference_eligible.astype(str).str.lower().eq("true").all(),
                    f"Thermal model uygunluk eksik: {model} / {rate}")
            bands = ["0-20", "20-60", "60-90"]
            v = z.loc[bands, "delta_adjusted_mV"].to_numpy(float)
            low = z.loc[bands, "bootstrap_CI_low_mV"].to_numpy(float)
            high = z.loc[bands, "bootstrap_CI_high_mV"].to_numpy(float)
            require(np.isfinite(np.r_[v,low,high]).all(),
                    f"Thermal bootstrap CI eksik: {model} / {rate}")
            y = np.arange(3)+(i-1.5)*.17
            ax.errorbar(v, y, xerr=np.vstack([v-low, high-v]), fmt="o", markersize=3,
                        linewidth=.9, capsize=2, label=model_labels[i])
        ax.axvline(0, color="gray", lw=.8)
        ax.set(title=f"{rate} discharge", xlabel="Adjusted 60–25 °C contrast (mV)")
        ax.set_yticks(range(3)); ax.set_yticklabels(["0–20", "20–60", "60–90"])
        ax.invert_yaxis()
    axes[0].set_ylabel("SoC band (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,1.05), ncol=4, frameon=False)
    fig.tight_layout(); fig.savefig(out/"fig_thermal_model_sensitivity.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def make_supplementary(ref: pd.DataFrame, stage: Path) -> None:
    label = CONDITION_META.set_index("Condition_ID")
    names = {int(i): f"{int(r.Temperature)}°C, {r.Rate}, {r.Charge_Rate}"
             for i, r in label.iterrows()}
    refs = ref.copy()
    refs["LHS condition label"] = refs.lhs_condition.map(names)
    refs["RHS condition label"] = refs.rhs_condition.map(names)
    refs["Contrast family label"] = refs.contrast_family.map(FAMILIES)
    require(refs[["LHS condition label", "RHS condition label", "Contrast family label"]]
            .notna().all().all(), "Supplementary etiket eşleştirme hatasi")
    cols = ["SoC_band", "contrast_family", "contrast_id", "LHS condition label",
            "RHS condition label", "n_cells_lhs", "n_cells_rhs", "inference_eligible",
            "delta_unadjusted_mV", "delta_adjusted_mV", "point_sign", "point_bucket",
            "p_value", "q_BH81", "cluster_CI_low_mV", "cluster_CI_high_mV",
            "bootstrap_CI_low_mV", "bootstrap_CI_high_mV", "bootstrap_sign_stability"]
    require_cols(refs, cols, "S1")
    refs = refs[cols].sort_values(["SoC_band", "contrast_family", "contrast_id"])
    require(len(refs) == 81, "S1 81 satir olmali")
    description = refs[~refs.inference_eligible.astype(str).str.lower().eq("true")].copy()
    require(len(description) == 6, "S2 altı betimsel satir olmali")
    out = stage/"supplementary"
    write_csv(refs, out/"Table_S1_81_planned_contrasts.csv")
    write_csv(description, out/"Table_S2_6_descriptive_contrasts.csv")
    workbook = out/"Supplementary_Tables_S1_S2.xlsx"
    # Excel is a convenient derived view. CSVs preserve unrounded numbers.
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        refs.to_excel(writer, sheet_name="Table S1", index=False)
        description.to_excel(writer, sheet_name="Table S2", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 32
            for c in sheet[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="17324D")
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for i, column in enumerate(sheet.columns, start=1):
                head = sheet.cell(1, i).value
                width = 29 if head in ("LHS condition label", "RHS condition label") else 20
                sheet.column_dimensions[get_column_letter(i)].width = width
            for row in sheet.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float): c.number_format = "0.000000"
                    c.alignment = Alignment(vertical="top")


def verify_lut(data: bytes, repr_data: bytes, lut_csv: pd.DataFrame, support: pd.DataFrame) -> np.ndarray:
    a = np.load(io.BytesIO(data), allow_pickle=False)
    b = np.load(io.BytesIO(repr_data), allow_pickle=False)
    require(a.shape == b.shape == (24,91), "Nihai LUT 24 × 91 olmali")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nihai LUT sonlu olmali")
    require(a.dtype == np.float32 and b.dtype == np.float32, "Float32 LUT bekleniyor")
    require(np.max(abs(a-b)) <= 1e-6, "Unified ve beşli karşılaştırma LUT uyumsuz")
    g = lut_csv[lut_csv.analysis.eq("full_history") & lut_csv.SoC.between(0,90)]
    require_cols(g, ["analysis", "Condition_ID", "SoC", "V_lut", "n_cells_at_soc"], "unified LUT long")
    require(len(g) == 2184, "Tam gecmis LUT 2184 dugum olmali")
    lut_m = g.pivot(index="Condition_ID",columns="SoC",values="V_lut").reindex(
        index=range(1,25),columns=range(91)).to_numpy(float)
    require(np.isfinite(lut_m).all() and np.max(abs(lut_m-a.astype(float))) <= 1e-6,
            "Uzun LUT/float32 LUT farkı tolerans disinda")
    support_m = g.pivot(index="Condition_ID",columns="SoC",values="n_cells_at_soc").reindex(
        index=range(1,25),columns=range(91)).to_numpy(float)
    expected_m = support.pivot(index="Condition_ID",columns="SoC",values="n_cells").reindex(
        index=range(1,25),columns=range(91)).to_numpy(float)
    require(np.array_equal(support_m, expected_m), "LUT eğitim desteği ve R2.2 hücre profili desteği farklı")
    require((np.diff(a.astype(float), axis=1) >= -1e-6).all(), "LUT monotone degil")
    return a


def make_manifest(stage: Path, src_meta: dict[str, dict[str, str]]) -> None:
    file_paths = sorted(x for x in stage.rglob("*") if x.is_file())
    rows = []
    for path in file_paths:
        if path.name in ("checksums.sha256", "manifest.json"):
            continue
        blob = path.read_bytes()
        rows.append((hashlib.sha256(blob).hexdigest(), path.relative_to(stage).as_posix(), len(blob)))
    (stage/"checksums.sha256").write_text("".join(f"{sha}  {name}\n" for sha,name,_ in rows), encoding="utf-8")
    manifest = {
        "package": PACKAGE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "coverage_80 + four-age-model + unified + common_five",
        "provenance_policy": "exact SHA256 of audited scientific ZIP members, independent of ZIP container metadata; baseline 2026-10-18",
        "scope_note": "Ancillary archive members not fingerprint-verified; scientific inputs and six support audit files are byte identical to approved release",
        "inputs": src_meta,
        "planned_contrasts": 81, "eligible_contrasts": 75, "descriptive_contrasts": 6,
        "LUT": {"shape": [24, 91], "dtype": "float32", "SoC": "0..90 inclusive"},
        "figures": {"total": len(FAMILY_FILES)+len(EXTRA_FIGURES),
                    "manuscript_figures": MANUSCRIPT_FIGURES},
        "files": [{"path": n, "size": size, "sha256": sha} for sha,n,size in rows],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }
    (stage/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> Path:
    require(ROOT.exists(), f"SOC_ROOT bulunamadi: {ROOT}")
    require(not (OUTPUT == ROOT or ROOT in OUTPUT.parents),
            "Cikti klasoru veri dizininin icine konulamaz")
    print("\nSoC FINAL PACKAGE: audited current stages only\n")
    archives = verified_archives()
    content: dict[str, bytes] = {}
    provenance: dict[str, dict[str,str]] = {}
    for key in SOURCE_NAMES:
        blob, location = audited_source(key, archives)
        require(hashlib.sha256(blob).hexdigest() == BASELINE_SCIENCE_SHA256[key],
                f"{key}: science input changed against approved 2026-10-18 manifest. STOP.")
        content[key] = blob
        digest = hashlib.sha256(blob).hexdigest()
        if key in ("cells", "cycles"):
            expected = BASELINE_SCIENCE_SHA256[key]
            require(digest == expected,
                    f"R2.2 {key} input is different from approved version: {digest}; expected {expected}")
            print(f"[PASS] R2.2 {key}: approved source SHA256")
        provenance[key] = {"location": location, "sha256": digest}
        print(f"[INPUT] {key}: {location}")
    cov = read_csv(content["coverage"], "coverage")
    age = read_csv(content["age"], "age")
    ref, age = verify_contrasts(age, cov)
    validation, bycond = verify_validation(read_csv(content["validation"], "validation"),
                                          read_csv(content["validation_cond"], "validation_cond"))
    repr_main, repr_gains = verify_repr(read_csv(content["repr_main"], "repr main"),
                                       read_csv(content["repr_gains"], "repr gains"))
    verify_42_robust(age)
    verify_five_manuscript(repr_main, repr_gains, validation)
    cells = clean_cell_data(content["cells"])
    curves, support = build_support_and_curves(cells)
    provenance["stage_archives"] = {name: {"location": str(path), "sha256": sha256_file(path)}
                                     for name, path in archives.items()}
    canonical_lut = verify_lut(content["lut32"], content["repr_lut32"],
                               read_csv(content["lut_long"], "LUT long"), support)
    # Do not replace a prior successful package unless this new one has passed QA.
    staging = Path(tempfile.mkdtemp(prefix="soc_final_stage_", dir=str(OUTPUT.parent)))
    try:
        figures = staging/"figures"; figures.mkdir(parents=True)
        plot_families(curves, figures)
        plot_cycle(content["cycles"], figures)
        plot_support(support, figures)
        plot_inference(ref, figures)
        plot_validation(validation, figures)
        plot_thermal(age, figures)
        expected_figs = set(FAMILY_FILES+EXTRA_FIGURES)
        actual_figs = {p.name for p in figures.glob("*.png")}
        require(actual_figs == expected_figs, f"Sekiller eksik/fazla: {actual_figs ^ expected_figs}")
        for name in MANUSCRIPT_FIGURES:
            require((figures/name).stat().st_size > 5000, f"Makalede gecen PNG bos: {name}")
        supp = staging/"supplementary"; supp.mkdir()
        # Reference + three sensitivity models (486 rows) kept at full precision.
        write_csv(age, supp/"contrast_results.csv")
        write_csv(ref, supp/"contrast_reference_81.csv")
        write_csv(bycond, supp/"validation_by_condition.csv")
        write_csv(validation, supp/"validation_summary.csv")
        write_csv(repr_gains, supp/"representation_gains.csv")
        write_csv(repr_main, supp/"representation_common_five.csv")
        make_supplementary(ref, staging)
        diagnostics = staging/"diagnostics"; diagnostics.mkdir()
        write_audit_support_zip(archives["support_audit"], diagnostics/"04_support_audit")
        write_csv(support, diagnostics/"condition_soc_support.csv")
        write_csv(curves, diagnostics/"condition_curves_q25_q75.csv")
        write_csv(cov[cov.rule.eq("coverage_80")], diagnostics/"coverage80_contrasts.csv")
        lutdir = staging/"lut"; lutdir.mkdir()
        np.save(lutdir/"final_LUT_0_90_float32.npy", canonical_lut, allow_pickle=False)
        write_csv(pd.DataFrame(canonical_lut, index=range(1,25),
                               columns=[f"SoC_{j:03d}" for j in range(91)]).rename_axis("Condition_ID").reset_index(),
                  lutdir/"final_LUT_0_90_matrix.csv")
        (staging/"README.md").write_text(
            "# CALCE condition-aware V--SoC: submission outputs\n\n"
            "Generated from the audited %80 band coverage, four-model age, unified estimator "
            "and five-model common-node comparison stages.\n\n"
            "`supplementary/contrast_results.csv`: all 486 four-model contrasts "
            "(81 reference; 75 eligible; 6 descriptive). Model-specific age reference cycles "
            "must be considered when interpreting rows.\n\n"
            "`supplementary/validation_by_condition.csv`: unified method, own-support, "
            "SoC 0..90 inclusive, both history scopes. A held-out cell contributes if >=5 paired nodes.\n\n"
            "`supplementary/representation_gains.csv`: common five-model support; "
            "paired cell bootstrap gains. See companion `representation_common_five.csv`.\n\n"
            "`Table_S1_81_planned_contrasts.csv`: reference cycle 108; inferential "
            "statistics blank for six descriptive contrasts. `Table_S2_6_descriptive_contrasts.csv`: "
            "six single-cell-arm reference contrasts, no p/q/CI claims.\n\n"
            "26 condition-family figures are descriptive physical-cell medians. "
            "The age, validation, and thermal figures use only audited current results. "
            "High-SoC guard sensitivity is withheld pending common-node matching.\n"
            "Diagnostics include the audited threshold/C5 evaluation-support sensitivity.\n\n"
            "LUT: float32, 24 x 91, SoC 0..90 inclusive; no extrapolation to 90..100. "
            "Condition 5 has sparse support on many nodes. Grid support is not equivalent "
            "to replicated cell support.\n\n"
            "This ZIP is a manuscript-output release. Full GitHub source scripts are "
            "distributed separately and must be uploaded before claiming a complete "
            "source-code repository.\n\n"
            "Original CALCE data and raw profiles are not bundled; reproduce using the "
            "public upstream source and the project's scripts. Source SHA256 fingerprints "
            "are recorded in manifest.json. Local Python/NumPy benchmark measurements "
            "are not MCU real-time guarantees.\n",
            encoding="utf-8")
        make_manifest(staging, provenance)
        # Versioned delivery: never remove or replace a previously delivered package.
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        delivered_name = f"{PACKAGE}_{tag}"
        destination = OUTPUT.parent / delivered_name
        zip_path = OUTPUT.parent / f"{delivered_name}.zip"
        zip_temp = OUTPUT.parent / f".{delivered_name}.tmp.zip"
        require(not destination.exists() and not zip_path.exists(),
                f"Package version already exists: {destination}")
        with zipfile.ZipFile(zip_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=f"{delivered_name}/{path.relative_to(staging).as_posix()}")
        with zipfile.ZipFile(zip_temp) as zf:
            require(zf.testzip() is None, "ZIP CRC hatasi")
            names = zf.namelist()
            require(len([n for n in names if n.endswith(".png")]) == 31,
                    "ZIP icinde 31 PNG olmali")
            for x in ("contrast_results.csv", "validation_by_condition.csv", "representation_gains.csv"):
                require(any(n.endswith("/supplementary/"+x) for n in names),
                        f"Zorunlu CSV ZIP icinde yok: {x}")
            require(all(any(n.endswith("/figures/"+f) for n in names)
                        for f in MANUSCRIPT_FIGURES), "Makalede referans verilen sekiz PNG eksik")
        shutil.move(str(staging), str(destination))
        os.replace(zip_temp, zip_path)
        print("\nFINAL QA PASSED")
        print("26 family figures + 5 current figures = 31 PNG")
        print("S1: 81 planned (75 inferential + 6 descriptive); S2: 6 descriptive")
        print("Unified validation + five-model gains + 24x91 float32 LUT")
        print("42 four-model robust 2C contrasts: PASS")
        print("Five audited source archives + detailed unified support audit: PASS")
        print(f"Package: {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MiB)")
        # Enable automatic Colab download unless explicitly disabled.
        if os.environ.get("SOC_AUTO_DOWNLOAD", "1") == "1":
            try:
                from google.colab import files
                files.download(str(zip_path))
            except ImportError:
                pass
        return zip_path
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
