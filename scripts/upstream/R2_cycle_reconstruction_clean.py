# ============================================================
# REVISION R2
# FINAL CYCLE-AWARE, CURRENT-PLATEAU, GAP-SAFE RECONSTRUCTION
#
# Self-contained / memory-safe.
#
# Outputs:
#   1) r2_cell_profiles.csv
#   2) r2_cycle_qc.csv
#   3) r2_condition_current_targets.csv
#   4) r2_condition_cycle_coverage.csv
#
# DOES NOT:
#   - pool cycles before interpolation
#   - extrapolate outside observed SoC support
#   - bridge >10 %-point SoC gaps
#   - use voltage magnitude as an automatic deletion rule
#   - use Hampel as a primary cleaner
# ============================================================

import os
import glob
import gc
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

FILES = [
    "Processed_Outputv3_1.xlsx",
    "Processed_Outputv3_2.xlsx",
    "Processed_Outputv3_3.xlsx"
]

ROOTS = [
    "/content",
    "/content/drive/MyDrive",
    "/content/drive/MyDrive/Colab Notebooks",
    "/mnt/data",
    "."
]

OUT = (
    "/content/revision_R2_final_profiles"
    if os.path.isdir("/content")
    else "revision_R2_final_profiles"
)

os.makedirs(OUT, exist_ok=True)

REQ = [
    "sample_number",
    "Rate",
    "Temperature",
    "Charge_Rate",
    "Step_Index",
    "Cycle_Index",
    "VoltageV",
    "CurrentA",
    "SoC"
]

SOC_GRID = np.arange(
    0.0,
    101.0,
    1.0
)

# Constant-current plateau tolerance
CURRENT_REL_TOL = 0.03
CURRENT_ABS_TOL = 0.05

# Empirically justified from R1.2:
# >10 %-point spacing is rare (~2.4%);
# 5–10% spacing is still common.
MAX_INTERP_GAP_SOC = 10.0

# Minimum usable points in a physical cycle
MIN_POINTS_CYCLE = 5

# Require at least this much SoC support
MIN_SOC_SPAN = 5.0


# ============================================================
# CONDITION MAP
# ============================================================

COND_MAP = {
     1:(10,"C/5","0.7C"),
     2:(10,"C/40","0.7C"),
     3:(10,"C/5","1C"),
     4:(10,"C/40","1C"),
     5:(10,"C/5","2C"),
     6:(10,"C/40","2C"),

     7:(25,"C/5","0.7C"),
     8:(25,"C/40","0.7C"),
     9:(25,"C/5","1C"),
    10:(25,"C/40","1C"),
    11:(25,"C/5","2C"),
    12:(25,"C/40","2C"),

    13:(45,"C/5","0.7C"),
    14:(45,"C/40","0.7C"),
    15:(45,"C/5","1C"),
    16:(45,"C/40","1C"),
    17:(45,"C/5","2C"),
    18:(45,"C/40","2C"),

    19:(60,"C/5","0.7C"),
    20:(60,"C/40","0.7C"),
    21:(60,"C/5","1C"),
    22:(60,"C/40","1C"),
    23:(60,"C/5","2C"),
    24:(60,"C/40","2C")
}

COND_INV = {
    (T, R, DR): cid
    for cid, (T, R, DR)
    in COND_MAP.items()
}


# ============================================================
# HELPERS
# ============================================================

def find_file(name):

    for root in ROOTS:

        p = os.path.join(
            root,
            name
        )

        if os.path.exists(p):
            return os.path.abspath(p)

        hits = glob.glob(
            os.path.join(
                root,
                "**",
                name
            ),
            recursive=True
        )

        if hits:

            return os.path.abspath(
                sorted(
                    hits,
                    key=lambda x: (len(x), x)
                )[0]
            )

    return None


def sample_int(x):

    try:

        s = (
            str(x)
            .strip()
            .split(".")[0]
            .lstrip("0")
        )

        return (
            int(s)
            if s
            else 0
        )

    except Exception:
        return np.nan


def norm_R(x):

    if pd.isna(x):
        return np.nan

    s = (
        str(x)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("\\", "/")
    )

    aliases = {
        "C5": "C/5",
        "0.2C": "C/5",
        "0.20C": "C/5",

        "C40": "C/40",
        "0.025C": "C/40"
    }

    return aliases.get(
        s,
        s
    )


def norm_DR(x):

    if pd.isna(x):
        return np.nan

    s = (
        str(x)
        .strip()
        .upper()
        .replace(" ", "")
        .replace(",", ".")
    )

    aliases = {
        "0.7": "0.7C",
        ".7C": "0.7C",

        "1": "1C",
        "1.0": "1C",
        "1.0C": "1C",

        "2": "2C",
        "2.0": "2C",
        "2.0C": "2C"
    }

    return aliases.get(
        s,
        s
    )


def gap_safe_interp(
    soc,
    voltage,
    grid,
    max_gap=10.0
):
    """
    Interpolate only between observed neighbouring SoC values
    whose separation <= max_gap.

    No extrapolation.
    No interpolation over large missing regions.
    """

    soc = np.asarray(
        soc,
        dtype=float
    )

    voltage = np.asarray(
        voltage,
        dtype=float
    )

    valid = (
        np.isfinite(soc)
        &
        np.isfinite(voltage)
    )

    soc = soc[valid]
    voltage = voltage[valid]

    if len(soc) < 2:

        return np.full(
            len(grid),
            np.nan
        )


    # Duplicate SoC within ONE cycle:
    # robust median voltage.
    tmp = pd.DataFrame({
        "SoC": soc,
        "V": voltage
    })

    tmp = (
        tmp
        .groupby(
            "SoC",
            as_index=False,
            observed=True
        )["V"]
        .median()
        .sort_values("SoC")
    )


    x = tmp[
        "SoC"
    ].to_numpy(float)

    y = tmp[
        "V"
    ].to_numpy(float)


    out = np.full(
        len(grid),
        np.nan,
        dtype=float
    )


    if len(x) < 2:
        return out


    # Exact observed grid hits
    for j, q in enumerate(grid):

        idx_exact = np.where(
            np.isclose(
                x,
                q,
                atol=1e-9
            )
        )[0]

        if len(idx_exact):

            out[j] = y[
                idx_exact[0]
            ]

            continue


        right = np.searchsorted(
            x,
            q,
            side="right"
        )

        left = right - 1


        if (
            left < 0
            or
            right >= len(x)
        ):
            continue


        gap = (
            x[right]
            -
            x[left]
        )


        if (
            gap <= 0
            or
            gap > max_gap
        ):
            continue


        frac = (
            (q - x[left])
            /
            gap
        )

        out[j] = (
            y[left]
            +
            frac
            *
            (
                y[right]
                -
                y[left]
            )
        )


    return out


# ============================================================
# SAFE SUPPORTED-NODE CELL AGGREGATION
# ============================================================

def summarize_supported_cycle_profiles(M):
    """Return median, 25th/75th percentiles, and cycle support.

    Each column is a SoC node and each row is an eligible cycle.
    Nodes with zero finite contributing cycles retain NaN statistics.
    Only finite-voltage support is used, consistent with the original
    source-record gap masking and cell-level percentile definition.
    """
    A = np.asarray(M, dtype=float)
    if A.ndim != 2:
        raise ValueError("Cycle profiles must be a 2D array")

    n_support = np.isfinite(A).sum(axis=0)
    supported = n_support > 0
    n_nodes = A.shape[1]
    med = np.full(n_nodes, np.nan, dtype=float)
    q25 = np.full(n_nodes, np.nan, dtype=float)
    q75 = np.full(n_nodes, np.nan, dtype=float)

    if supported.any():
        values = A[:, supported]
        med[supported] = np.nanmedian(values, axis=0)
        q25[supported] = np.nanpercentile(values, 25, axis=0)
        q75[supported] = np.nanpercentile(values, 75, axis=0)

    return med, q25, q75, n_support


# ============================================================
# INPUT PATHS
# ============================================================

paths = [
    find_file(f)
    for f in FILES
]

if any(
    p is None
    for p in paths
):

    raise FileNotFoundError(
        "All three Processed_Outputv3_*.xlsx files are required."
    )


# ============================================================
# PASS 1
# Estimate constant-current target for every Condition_ID
#
# We use the median magnitude of NEGATIVE current under Step 5.
# Near-zero startup points have negligible effect on the median.
# ============================================================

condition_current_values = {
    cid: []
    for cid in range(1, 25)
}


print(
    "PASS 1 — estimating nominal discharge-current plateaus"
)


for k, path in enumerate(
    paths,
    start=1
):

    print(
        f"[{k}/3]",
        os.path.basename(path)
    )


    try:

        sheet = "Step5_Data"

        head = pd.read_excel(
            path,
            sheet_name=sheet,
            nrows=0,
            engine="openpyxl"
        )

    except Exception:

        sheet = 0


    d = pd.read_excel(
        path,
        sheet_name=sheet,
        usecols=REQ,
        engine="openpyxl"
    )


    for c in [
        "Temperature",
        "Step_Index",
        "CurrentA"
    ]:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )


    d["Rate"] = (
        d["Rate"]
        .map(norm_R)
    )

    d["Charge_Rate"] = (
        d["Charge_Rate"]
        .map(norm_DR)
    )


    cond = []

    for t, r, dr in zip(
        d["Temperature"],
        d["Rate"],
        d["Charge_Rate"]
    ):

        if pd.isna(t):

            cond.append(
                np.nan
            )

        else:

            cond.append(
                COND_INV.get(
                    (
                        int(round(t)),
                        r,
                        dr
                    ),
                    np.nan
                )
            )


    d["Condition_ID"] = cond


    d = d[
        (d["Step_Index"] == 5)
        &
        (d["CurrentA"] < 0)
        &
        d["Condition_ID"].notna()
    ]


    for cid, g in d.groupby(
        "Condition_ID",
        observed=True
    ):

        vals = np.abs(
            g["CurrentA"]
            .to_numpy(float)
        )

        # Exclude numerically near-zero negative transition points
        vals = vals[
            vals > 0.1
        ]

        if len(vals):

            # Store per-file median rather than all raw currents.
            condition_current_values[
                int(cid)
            ].append(
                float(
                    np.median(vals)
                )
            )


    del d
    gc.collect()


CURRENT_TARGET = {}

current_rows = []


for cid in range(1, 25):

    vals = (
        condition_current_values[
            cid
        ]
    )

    if not vals:

        CURRENT_TARGET[
            cid
        ] = np.nan

    else:

        CURRENT_TARGET[
            cid
        ] = float(
            np.median(vals)
        )


    current_rows.append({

        "Condition_ID":
            cid,

        "Temperature":
            COND_MAP[cid][0],

        "Rate":
            COND_MAP[cid][1],

        "Charge_Rate":
            COND_MAP[cid][2],

        "target_abs_current_A":
            CURRENT_TARGET[cid],

        "allowed_abs_deviation_A":
            (
                max(
                    CURRENT_ABS_TOL,
                    CURRENT_REL_TOL
                    *
                    CURRENT_TARGET[cid]
                )
                if np.isfinite(
                    CURRENT_TARGET[cid]
                )
                else np.nan
            )
    })


current_table = pd.DataFrame(
    current_rows
)

current_table.to_csv(
    os.path.join(
        OUT,
        "r2_condition_current_targets.csv"
    ),
    index=False
)


print(
    "\nEstimated current plateaus:"
)

print(
    current_table[
        [
            "Condition_ID",
            "Charge_Rate",
            "target_abs_current_A",
            "allowed_abs_deviation_A"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PASS 2
# Cycle-aware reconstruction
# ============================================================

print(
    "\nPASS 2 — cycle-aware reconstruction"
)


all_cell_profiles = []
all_cycle_qc = []


for k, path in enumerate(
    paths,
    start=1
):

    print(
        f"\n[{k}/3]",
        os.path.basename(path)
    )


    try:

        sheet = "Step5_Data"

        head = pd.read_excel(
            path,
            sheet_name=sheet,
            nrows=0,
            engine="openpyxl"
        )

    except Exception:

        sheet = 0


    d = pd.read_excel(
        path,
        sheet_name=sheet,
        usecols=REQ,
        engine="openpyxl"
    )


    d["__source_row__"] = np.arange(
        len(d),
        dtype=np.int32
    )


    for c in [
        "Temperature",
        "Step_Index",
        "Cycle_Index",
        "VoltageV",
        "CurrentA",
        "SoC"
    ]:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )


    d["sample_int"] = (
        d["sample_number"]
        .map(sample_int)
    )

    d["Rate"] = (
        d["Rate"]
        .map(norm_R)
    )

    d["Charge_Rate"] = (
        d["Charge_Rate"]
        .map(norm_DR)
    )


    cond = []

    for t, r, dr in zip(
        d["Temperature"],
        d["Rate"],
        d["Charge_Rate"]
    ):

        if pd.isna(t):

            cond.append(
                np.nan
            )

        else:

            cond.append(
                COND_INV.get(
                    (
                        int(round(t)),
                        r,
                        dr
                    ),
                    np.nan
                )
            )


    d["Condition_ID"] = cond


    # --------------------------------------------------------
    # Basic discharge filter
    # --------------------------------------------------------

    d = d[
        (d["Step_Index"] == 5)
        &
        (d["CurrentA"] < 0)
        &
        d["Condition_ID"].notna()
        &
        d["Cycle_Index"].notna()
        &
        d["VoltageV"].notna()
        &
        d["SoC"].notna()
        &
        d["sample_int"].notna()
    ].copy()


    d["sample_int"] = (
        d["sample_int"]
        .astype(np.int32)
    )

    d["Condition_ID"] = (
        d["Condition_ID"]
        .astype(np.int16)
    )

    d["Cycle_Index"] = (
        d["Cycle_Index"]
        .astype(np.int32)
    )


    # --------------------------------------------------------
    # Constant-current plateau filter
    # --------------------------------------------------------

    target = (
        d["Condition_ID"]
        .map(CURRENT_TARGET)
        .astype(float)
    )


    tolerance = np.maximum(
        CURRENT_ABS_TOL,
        CURRENT_REL_TOL
        *
        target
    )


    current_error = np.abs(
        np.abs(
            d["CurrentA"]
        )
        -
        target
    )


    d["on_current_plateau"] = (
        current_error
        <=
        tolerance
    )


    # Keep this diagnostic count
    n_before = len(d)

    d = d[
        d["on_current_plateau"]
    ].copy()

    n_after = len(d)


    print(
        f"  negative-current rows: {n_before:,}"
    )

    print(
        f"  plateau rows kept    : {n_after:,} "
        f"({100*n_after/n_before:.2f}%)"
    )


    d = d.sort_values(
        [
            "sample_int",
            "Condition_ID",
            "Cycle_Index",
            "__source_row__"
        ]
    )


    # ========================================================
    # Work cell-by-cell.
    # We NEVER retain all cycle-grid profiles globally.
    # ========================================================

    for (
        sid,
        cid
    ), cell_df in d.groupby(
        [
            "sample_int",
            "Condition_ID"
        ],
        sort=False,
        observed=True
    ):

        cycle_profiles = []

        condition_meta = (
            cell_df.iloc[0]
        )


        for cyc, g in cell_df.groupby(
            "Cycle_Index",
            sort=False,
            observed=True
        ):

            g = (
                g
                .sort_values(
                    "__source_row__"
                )
                .copy()
            )


            source_rows = (
                g["__source_row__"]
                .to_numpy(
                    dtype=np.int32
                )
            )

            soc = (
                g["SoC"]
                .to_numpy(float)
            )

            voltage = (
                g["VoltageV"]
                .to_numpy(float)
            )

            current = (
                g["CurrentA"]
                .to_numpy(float)
            )


            n_plateau_points = len(g)


            if n_plateau_points < MIN_POINTS_CYCLE:

                all_cycle_qc.append({

                    "sample_int":
                        int(sid),

                    "Condition_ID":
                        int(cid),

                    "Cycle_Index":
                        int(cyc),

                    "usable":
                        False,

                    "reason":
                        "too_few_plateau_points",

                    "n_plateau_points":
                        n_plateau_points
                })

                continue


            # ------------------------------------------------
            # Because non-plateau points were removed,
            # source-row gaps indicate interrupted plateau.
            #
            # Split and retain longest continuous plateau
            # segment.
            # ------------------------------------------------

            breaks = (
                np.diff(
                    source_rows
                )
                > 1
            )


            seg_id = np.concatenate(
                [
                    np.array([0]),
                    np.cumsum(
                        breaks
                    )
                ]
            )


            unique_seg, counts = np.unique(
                seg_id,
                return_counts=True
            )


            best_seg = (
                unique_seg[
                    np.argmax(
                        counts
                    )
                ]
            )


            keep = (
                seg_id
                ==
                best_seg
            )


            soc = soc[keep]
            voltage = voltage[keep]
            current = current[keep]


            if len(soc) < MIN_POINTS_CYCLE:

                all_cycle_qc.append({

                    "sample_int":
                        int(sid),

                    "Condition_ID":
                        int(cid),

                    "Cycle_Index":
                        int(cyc),

                    "usable":
                        False,

                    "reason":
                        "short_after_plateau_segmentation",

                    "n_plateau_points":
                        n_plateau_points,

                    "n_selected_points":
                        int(len(soc))
                })

                continue


            soc_span = float(
                np.nanmax(soc)
                -
                np.nanmin(soc)
            )


            if soc_span < MIN_SOC_SPAN:

                all_cycle_qc.append({

                    "sample_int":
                        int(sid),

                    "Condition_ID":
                        int(cid),

                    "Cycle_Index":
                        int(cyc),

                    "usable":
                        False,

                    "reason":
                        "insufficient_soc_span",

                    "n_plateau_points":
                        n_plateau_points,

                    "n_selected_points":
                        int(len(soc)),

                    "soc_span":
                        soc_span
                })

                continue


            # ------------------------------------------------
            # Cycle diagnostics BEFORE interpolation
            # ------------------------------------------------

            dsoc = np.diff(soc)
            dv = np.diff(voltage)


            valid_down = (
                dsoc < 0
            )


            reverse50 = int(
                np.sum(
                    valid_down
                    &
                    (dv >= 0.050)
                )
            )


            very_large_gap = int(
                np.sum(
                    np.abs(dsoc)
                    >
                    MAX_INTERP_GAP_SOC
                )
            )


            # ------------------------------------------------
            # GAP-SAFE interpolation
            # ------------------------------------------------

            profile = gap_safe_interp(
                soc=soc,
                voltage=voltage,
                grid=SOC_GRID,
                max_gap=MAX_INTERP_GAP_SOC
            )


            n_grid = int(
                np.isfinite(
                    profile
                ).sum()
            )


            if n_grid < 5:

                all_cycle_qc.append({

                    "sample_int":
                        int(sid),

                    "Condition_ID":
                        int(cid),

                    "Cycle_Index":
                        int(cyc),

                    "usable":
                        False,

                    "reason":
                        "insufficient_interpolated_support",

                    "n_selected_points":
                        int(len(soc)),

                    "n_grid_points":
                        n_grid
                })

                continue


            cycle_profiles.append(
                profile
            )


            all_cycle_qc.append({

                "sample_int":
                    int(sid),

                "Condition_ID":
                    int(cid),

                "Cycle_Index":
                    int(cyc),

                "usable":
                    True,

                "reason":
                    "ok",

                "n_plateau_points":
                    n_plateau_points,

                "n_selected_points":
                    int(len(soc)),

                "soc_min":
                    float(
                        np.min(soc)
                    ),

                "soc_max":
                    float(
                        np.max(soc)
                    ),

                "soc_span":
                    soc_span,

                "n_reverse_ge50mV":
                    reverse50,

                "n_gaps_gt10pct":
                    very_large_gap,

                "n_grid_points":
                    n_grid
            })


        # ====================================================
        # Robust aggregation:
        # cycles -> ONE CELL profile
        # ====================================================

        if not cycle_profiles:
            continue


        M = np.asarray(
            cycle_profiles,
            dtype=float
        )


        # Compute statistics only at supported SoC nodes.
        # Unsupported nodes remain NaN, as in the original method.
        # This avoids repetitive all-NaN slice warnings without
        # changing which nodes contribute to the cell profile.
        (
            cell_median,
            cell_q25,
            cell_q75,
            n_cycles_soc,
        ) = summarize_supported_cycle_profiles(M)


        for j, soc_q in enumerate(
            SOC_GRID
        ):

            all_cell_profiles.append({

                "sample_int":
                    int(sid),

                "Condition_ID":
                    int(cid),

                "Temperature":
                    int(
                        condition_meta[
                            "Temperature"
                        ]
                    ),

                "Rate":
                    str(
                        condition_meta[
                            "Rate"
                        ]
                    ),

                "Charge_Rate":
                    str(
                        condition_meta[
                            "Charge_Rate"
                        ]
                    ),

                "SoC":
                    float(soc_q),

                "V_cell_median":
                    (
                        float(
                            cell_median[j]
                        )
                        if np.isfinite(
                            cell_median[j]
                        )
                        else np.nan
                    ),

                "V_cell_q25":
                    (
                        float(
                            cell_q25[j]
                        )
                        if np.isfinite(
                            cell_q25[j]
                        )
                        else np.nan
                    ),

                "V_cell_q75":
                    (
                        float(
                            cell_q75[j]
                        )
                        if np.isfinite(
                            cell_q75[j]
                        )
                        else np.nan
                    ),

                "n_cycles_at_soc":
                    int(
                        n_cycles_soc[j]
                    )
            })


        del M
        del cycle_profiles


    del d
    gc.collect()


# ============================================================
# SAVE FINAL TABLES
# ============================================================

cell_profiles = pd.DataFrame(
    all_cell_profiles
)

cycle_qc = pd.DataFrame(
    all_cycle_qc
)


cell_profiles.to_csv(
    os.path.join(
        OUT,
        "r2_cell_profiles.csv"
    ),
    index=False
)


cycle_qc.to_csv(
    os.path.join(
        OUT,
        "r2_cycle_qc.csv"
    ),
    index=False
)


# ============================================================
# CONDITION/CYCLE COVERAGE
# ============================================================

usable_qc = cycle_qc[
    cycle_qc["usable"] == True
].copy()


coverage = (

    usable_qc

    .groupby(
        [
            "Condition_ID",
            "sample_int"
        ],
        as_index=False,
        observed=True
    )

    .agg(

        n_usable_cycles=(
            "Cycle_Index",
            "nunique"
        ),

        cycle_min=(
            "Cycle_Index",
            "min"
        ),

        cycle_median=(
            "Cycle_Index",
            "median"
        ),

        cycle_max=(
            "Cycle_Index",
            "max"
        ),

        cycles_with_reverse50=(
            "n_reverse_ge50mV",
            lambda x:
                int(
                    np.sum(
                        np.asarray(x)
                        > 0
                    )
                )
        ),

        total_large_gaps=(
            "n_gaps_gt10pct",
            "sum"
        )
    )

)


coverage.to_csv(
    os.path.join(
        OUT,
        "r2_condition_cycle_coverage.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "R2 FILE-LOCAL RECONSTRUCTION COMPLETE — R2.2 REQUIRED"
)

print(
    "=" * 70
)


print(
    "\nCell profiles:",
    f"{len(cell_profiles):,}"
)

print(
    "Unique cells:",
    cell_profiles[
        "sample_int"
    ].nunique()
)

print(
    "Conditions:",
    cell_profiles[
        "Condition_ID"
    ].nunique()
)


print(
    "\nCycle QC:"
)

print(
    cycle_qc[
        "usable"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nReasons:"
)

print(
    cycle_qc[
        "reason"
    ].value_counts()
)


print(
    "\nCycles with >=1 reverse >=50 mV event:"
)

print(
    int(
        (
            usable_qc[
                "n_reverse_ge50mV"
            ]
            > 0
        ).sum()
    ),
    "/",
    len(
        usable_qc
    )
)


print(
    "\nCycles containing >10% SoC gaps:"
)

print(
    int(
        (
            usable_qc[
                "n_gaps_gt10pct"
            ]
            > 0
        ).sum()
    ),
    "/",
    len(
        usable_qc
    )
)


print(
    "\nOutputs:"
)

for f in [
    "r2_condition_current_targets.csv",
    "r2_cell_profiles.csv",
    "r2_cycle_qc.csv",
    "r2_condition_cycle_coverage.csv"
]:

    print(
        " -",
        os.path.join(
            OUT,
            f
        )
    )