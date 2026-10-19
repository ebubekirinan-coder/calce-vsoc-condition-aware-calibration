# ============================================================
# REVISION R2.2 — CLEAN REPRODUCIBILITY VERSION
# GLOBAL CYCLE STORE, CROSS-FILE DEDUPLICATION,
# OBSERVED-METADATA AND NOMINAL-CONCORDANT BRANCHES
#
# Execute after the validated R2 reconstruction in the same Colab
# session. This script preserves the original selection, fitting,
# float32 storage, and output conventions.
#
# Manuscript is frozen: the QA checks verify outcomes but never
# change scientific selection rules to force a particular count.
#
# Main goals:
# 1) One row = one physical cell × condition × cycle
# 2) Remove accidental cross-file duplicate cycles
# 3) Build exact global cell profiles across ALL files
# 4) Produce:
#       A) observed-metadata branch
#       B) nominal-concordant branch
#
# Memory-safe wide format:
# ~89k rows × 101 voltage columns
# ============================================================

import os
import gc
import numpy as np
import pandas as pd

OUT22 = os.environ.get(
    "SOC_R22_OUTPUT_DIR",
    "/content/revision_R2_2_global" if os.path.isdir("/content")
    else "outputs/revision_R2_2_global",
)
os.makedirs(OUT22, exist_ok=True)

# ------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------

required_globals = [
    "paths",
    "REQ",
    "CURRENT_TARGET",
    "COND_INV",
    "sample_int",
    "norm_R",
    "norm_DR",
    "gap_safe_interp"
]

missing = [
    x for x in required_globals
    if x not in globals()
]

if missing:
    raise RuntimeError(
        "R2 must be run first. Missing: "
        + ", ".join(missing)
    )


SOC_GRID22 = np.arange(
    0.0,
    101.0,
    1.0
)

V_COLS = [
    f"V_{i:03d}"
    for i in range(101)
]

CURRENT_REL_TOL22 = 0.03
CURRENT_ABS_TOL22 = 0.05
MAX_GAP22 = 10.0
MIN_POINTS22 = 5
MIN_SOC_SPAN22 = 5.0


# ============================================================
# NOMINAL MAP
# ============================================================

def nominal_condition(sample_id):

    s = int(sample_id)

    ranges = [
        (1,   8,   1),
        (9,   16,  2),
        (17,  24,  3),
        (25,  32,  4),
        (33,  40,  5),
        (41,  48,  6),
        (49,  56,  7),
        (57,  64,  8),
        (65,  72,  9),
        (73,  80, 10),
        (81,  88, 11),
        (89,  96, 12),
        (97, 104, 13),
        (105,112, 14),
        (113,120, 15),
        (121,128, 16),
        (129,136, 17),
        (137,144, 18),
        (145,152, 19),
        (153,160, 20),
        (161,168, 21),
        (169,176, 22),
        (177,179, 23),
        (185,187, 24)
    ]

    for lo, hi, cid in ranges:

        if lo <= s <= hi:
            return cid

    return np.nan




# ============================================================
# SUPPORTED-NODE AGGREGATION (WARNING-FREE)
# ============================================================

def summarize_supported_profile_nodes(profiles):
    """Compute original nodewise median, Q25, Q75, and support.

    A SoC node with no finite cycle voltages retains NaN in every
    summary. This avoids all-NaN percentile warnings without adding
    support, extrapolating, or altering the underlying aggregation.
    """
    array = np.asarray(profiles, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional cycle-profile array")

    n_support = np.isfinite(array).sum(axis=0)
    supported = n_support > 0
    n_nodes = array.shape[1]
    median = np.full(n_nodes, np.nan, dtype=float)
    q25 = np.full(n_nodes, np.nan, dtype=float)
    q75 = np.full(n_nodes, np.nan, dtype=float)
    if supported.any():
        values = array[:, supported]
        median[supported] = np.nanmedian(values, axis=0)
        q25[supported] = np.nanpercentile(values, 25, axis=0)
        q75[supported] = np.nanpercentile(values, 75, axis=0)

    return median, q25, q75, n_support

# ============================================================
# PROCESS EACH WORKBOOK
# ============================================================

part_files = []

for file_no, path in enumerate(
    paths,
    start=1
):

    source_name = os.path.basename(
        path
    )

    print(
        f"\n[{file_no}/3] Processing:",
        source_name
    )

    try:

        sheet = "Step5_Data"

        pd.read_excel(
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


    # --------------------------------------------------------
    # Observed metadata -> Condition_ID
    # --------------------------------------------------------

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
    # Discharge rows
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
    # Nominal current plateau
    # --------------------------------------------------------

    target = (
        d["Condition_ID"]
        .map(CURRENT_TARGET)
        .astype(float)
    )

    tol = np.maximum(
        CURRENT_ABS_TOL22,
        CURRENT_REL_TOL22
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

    d = d[
        current_error <= tol
    ].copy()


    d = d.sort_values(
        [
            "sample_int",
            "Condition_ID",
            "Cycle_Index",
            "__source_row__"
        ]
    )


    meta_rows = []
    profile_rows = []


    # ========================================================
    # ONE GROUP = ONE OBSERVED PHYSICAL CYCLE
    # ========================================================

    for (
        sid,
        cid,
        cyc
    ), g in d.groupby(
        [
            "sample_int",
            "Condition_ID",
            "Cycle_Index"
        ],
        sort=False,
        observed=True
    ):

        g = g.sort_values(
            "__source_row__"
        )

        rows = (
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


        if len(g) < MIN_POINTS22:
            continue


        # ----------------------------------------------------
        # Longest contiguous plateau segment
        # ----------------------------------------------------

        breaks = (
            np.diff(rows) > 1
        )

        seg = np.concatenate(
            [
                [0],
                np.cumsum(
                    breaks
                )
            ]
        )

        ids, counts = np.unique(
            seg,
            return_counts=True
        )

        best = ids[
            np.argmax(counts)
        ]

        keep = (
            seg == best
        )

        soc = soc[keep]
        voltage = voltage[keep]


        if len(soc) < MIN_POINTS22:
            continue


        soc_span = (
            np.nanmax(soc)
            -
            np.nanmin(soc)
        )

        if soc_span < MIN_SOC_SPAN22:
            continue


        profile = gap_safe_interp(
            soc=soc,
            voltage=voltage,
            grid=SOC_GRID22,
            max_gap=MAX_GAP22
        )


        n_grid = int(
            np.isfinite(
                profile
            ).sum()
        )

        if n_grid < 5:
            continue


        row0 = g.iloc[0]

        meta_rows.append({
            "__source_file__":
                source_name,

            "sample_int":
                int(sid),

            "Condition_ID":
                int(cid),

            "Cycle_Index":
                int(cyc),

            "Temperature":
                int(
                    row0[
                        "Temperature"
                    ]
                ),

            "Rate":
                str(
                    row0[
                        "Rate"
                    ]
                ),

            "Charge_Rate":
                str(
                    row0[
                        "Charge_Rate"
                    ]
                ),

            "n_grid_points":
                n_grid
        })

        profile_rows.append(
            profile.astype(
                np.float32
            )
        )


    # --------------------------------------------------------
    # Build wide DataFrame for this workbook
    # --------------------------------------------------------

    meta_df = pd.DataFrame(
        meta_rows
    )

    if not profile_rows:
        raise ValueError(
            f"No usable discharge cycle profiles in {source_name}. "
            "Inspect source metadata and the current-plateau filter."
        )

    V = np.vstack(
        profile_rows
    )

    V_df = pd.DataFrame(
        V,
        columns=V_COLS
    )

    part = pd.concat(
        [
            meta_df.reset_index(
                drop=True
            ),
            V_df
        ],
        axis=1
    )


    part_path = os.path.join(
        OUT22,
        f"cycle_profiles_part_{file_no}.pkl"
    )

    part.to_pickle(
        part_path
    )

    part_files.append(
        part_path
    )


    print(
        "  usable cycle profiles:",
        f"{len(part):,}"
    )


    del d
    del meta_df
    del V
    del V_df
    del part
    del meta_rows
    del profile_rows

    gc.collect()


# ============================================================
# COMBINE ALL THREE WORKBOOKS
# ============================================================

print(
    "\nCombining workbook-level cycle tables..."
)

cycle = pd.concat(
    [
        pd.read_pickle(p)
        for p in part_files
    ],
    ignore_index=True
)

print(
    "Total cycle rows before cross-file dedup:",
    f"{len(cycle):,}"
)


# ============================================================
# CROSS-FILE DUPLICATE CYCLE AUDIT
# ============================================================

KEY = [
    "sample_int",
    "Condition_ID",
    "Cycle_Index"
]

dup_count = (
    cycle
    .groupby(
        KEY,
        observed=True
    )
    .size()
    .rename(
        "n_source_occurrences"
    )
    .reset_index()
)

dup_keys = dup_count[
    dup_count[
        "n_source_occurrences"
    ] > 1
].copy()

print(
    "Duplicate physical cycle keys:",
    len(dup_keys)
)


duplicate_audit = []


for _, keyrow in dup_keys.iterrows():

    mask = (
        (cycle["sample_int"] == keyrow["sample_int"])
        &
        (cycle["Condition_ID"] == keyrow["Condition_ID"])
        &
        (cycle["Cycle_Index"] == keyrow["Cycle_Index"])
    )

    g = cycle[
        mask
    ]

    A = g[
        V_COLS
    ].to_numpy(
        dtype=float
    )

    reference = A[0]

    maes = []
    maxes = []

    for j in range(
        1,
        len(A)
    ):

        common = (
            np.isfinite(reference)
            &
            np.isfinite(A[j])
        )

        if common.sum() == 0:

            maes.append(
                np.nan
            )

            maxes.append(
                np.nan
            )

        else:

            diff = np.abs(
                reference[common]
                -
                A[j][common]
            )

            maes.append(
                float(
                    np.mean(diff)
                )
            )

            maxes.append(
                float(
                    np.max(diff)
                )
            )


    duplicate_audit.append({

        "sample_int":
            int(
                keyrow["sample_int"]
            ),

        "Condition_ID":
            int(
                keyrow["Condition_ID"]
            ),

        "Cycle_Index":
            int(
                keyrow["Cycle_Index"]
            ),

        "n_occurrences":
            int(len(g)),

        "source_files":
            "|".join(
                g[
                    "__source_file__"
                ]
                .astype(str)
                .tolist()
            ),

        "max_pairwise_MAE_V":
            (
                np.nanmax(maes)
                if len(maes)
                else 0.0
            ),

        "max_pairwise_abs_V":
            (
                np.nanmax(maxes)
                if len(maxes)
                else 0.0
            )
    })


duplicate_audit = pd.DataFrame(
    duplicate_audit
)


duplicate_audit.to_csv(
    os.path.join(
        OUT22,
        "duplicate_cycle_audit.csv"
    ),
    index=False
)


if len(
    duplicate_audit
):

    print(
        "\n=== DUPLICATE CYCLE AUDIT ==="
    )

    print(
        duplicate_audit[
            [
                "n_occurrences",
                "max_pairwise_MAE_V",
                "max_pairwise_abs_V"
            ]
        ]
        .describe()
        .to_string()
    )


# ============================================================
# DEDUP RULE
#
# Only auto-collapse cycles that are practically identical:
# MAE <=1 mV AND max difference <=5 mV.
#
# Materially different duplicate cycles are NOT silently merged.
# ============================================================

if len(
    duplicate_audit
):

    material = duplicate_audit[
        (
            duplicate_audit[
                "max_pairwise_MAE_V"
            ] > 0.001
        )
        |
        (
            duplicate_audit[
                "max_pairwise_abs_V"
            ] > 0.005
        )
    ].copy()

else:

    material = pd.DataFrame()


print(
    "\nMaterially different duplicate cycle keys:",
    len(material)
)


if len(material) > 0:

    material.to_csv(
        os.path.join(
            OUT22,
            "MATERIAL_duplicate_cycles.csv"
        ),
        index=False
    )

    print(
        "\nSTOP: Some identical cell-condition-cycle keys "
        "contain materially different profiles."
    )

    print(
        "Inspect MATERIAL_duplicate_cycles.csv before proceeding."
    )

    GLOBAL_OK = False

else:

    # Drop practically identical duplicates
    cycle = (
        cycle
        .sort_values(
            [
                "sample_int",
                "Condition_ID",
                "Cycle_Index",
                "n_grid_points"
            ],
            ascending=[
                True,
                True,
                True,
                False
            ]
        )
        .drop_duplicates(
            subset=KEY,
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    GLOBAL_OK = True


# ============================================================
# CONTINUE ONLY IF CROSS-FILE CYCLE IDENTITY IS CLEAN
# ============================================================

if GLOBAL_OK:

    print(
        "\nCycle rows after dedup:",
        f"{len(cycle):,}"
    )


    # --------------------------------------------------------
    # Nominal-concordance markers
    # --------------------------------------------------------

    cycle[
        "Nominal_Condition_ID"
    ] = (
        cycle[
            "sample_int"
        ]
        .map(
            nominal_condition
        )
    )


    cycle[
        "matches_nominal"
    ] = (
        cycle[
            "Nominal_Condition_ID"
        ].notna()
        &
        (
            cycle[
                "Condition_ID"
            ]
            ==
            cycle[
                "Nominal_Condition_ID"
            ]
        )
    )


    # Save global cycle store
    cycle.to_pickle(
        os.path.join(
            OUT22,
            "global_cycle_profiles.pkl"
        )
    )


    # ========================================================
    # CELL-PROFILE BUILDER
    # ========================================================

    def build_global_cell_profiles(
        cyc,
        branch_name
    ):

        records = []

        for (
            sid,
            cid
        ), g in cyc.groupby(
            [
                "sample_int",
                "Condition_ID"
            ],
            observed=True,
            sort=True
        ):

            A = g[
                V_COLS
            ].to_numpy(
                dtype=float
            )


            med, q25, q75, n_support = (
                summarize_supported_profile_nodes(A)
            )


            meta = g.iloc[0]

            for j, soc in enumerate(
                SOC_GRID22
            ):

                records.append({

                    "branch":
                        branch_name,

                    "sample_int":
                        int(sid),

                    "Condition_ID":
                        int(cid),

                    "Temperature":
                        int(
                            meta[
                                "Temperature"
                            ]
                        ),

                    "Rate":
                        str(
                            meta[
                                "Rate"
                            ]
                        ),

                    "Charge_Rate":
                        str(
                            meta[
                                "Charge_Rate"
                            ]
                        ),

                    "SoC":
                        float(soc),

                    "V_cell_median":
                        (
                            float(med[j])
                            if np.isfinite(
                                med[j]
                            )
                            else np.nan
                        ),

                    "V_cell_q25":
                        (
                            float(q25[j])
                            if np.isfinite(
                                q25[j]
                            )
                            else np.nan
                        ),

                    "V_cell_q75":
                        (
                            float(q75[j])
                            if np.isfinite(
                                q75[j]
                            )
                            else np.nan
                        ),

                    "n_cycles_at_soc":
                        int(
                            n_support[j]
                        )
                })


        return pd.DataFrame(
            records
        )


    # --------------------------------------------------------
    # A) Observed metadata branch
    # --------------------------------------------------------

    observed_cycles = (
        cycle.copy()
    )

    observed_profiles = (
        build_global_cell_profiles(
            observed_cycles,
            "observed_metadata"
        )
    )


    # --------------------------------------------------------
    # B) Nominal-concordant branch
    # --------------------------------------------------------

    nominal_cycles = cycle[
        cycle[
            "matches_nominal"
        ]
    ].copy()

    nominal_profiles = (
        build_global_cell_profiles(
            nominal_cycles,
            "nominal_concordant"
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    observed_profiles.to_csv(
        os.path.join(
            OUT22,
            "cell_profiles_observed_metadata.csv"
        ),
        index=False
    )

    nominal_profiles.to_csv(
        os.path.join(
            OUT22,
            "cell_profiles_nominal_concordant.csv"
        ),
        index=False
    )


    # ========================================================
    # BRANCH SUMMARY
    # ========================================================

    summary = pd.DataFrame([

        {
            "branch":
                "observed_metadata",

            "n_cycles":
                len(
                    observed_cycles
                ),

            "n_cells":
                observed_cycles[
                    "sample_int"
                ].nunique(),

            "n_sample_condition_pairs":
                observed_cycles[
                    [
                        "sample_int",
                        "Condition_ID"
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "n_conditions":
                observed_cycles[
                    "Condition_ID"
                ].nunique()
        },

        {
            "branch":
                "nominal_concordant",

            "n_cycles":
                len(
                    nominal_cycles
                ),

            "n_cells":
                nominal_cycles[
                    "sample_int"
                ].nunique(),

            "n_sample_condition_pairs":
                nominal_cycles[
                    [
                        "sample_int",
                        "Condition_ID"
                    ]
                ]
                .drop_duplicates()
                .shape[0],

            "n_conditions":
                nominal_cycles[
                    "Condition_ID"
                ].nunique()
        }
    ])


    summary.to_csv(
        os.path.join(
            OUT22,
            "branch_summary.csv"
        ),
        index=False
    )


    # Per-condition cells
    cond_summary = (

        nominal_cycles[
            [
                "sample_int",
                "Condition_ID"
            ]
        ]

        .drop_duplicates()

        .groupby(
            "Condition_ID",
            as_index=False
        )

        .agg(
            n_nominal_cells=(
                "sample_int",
                "nunique"
            )
        )
    )


    cond_summary.to_csv(
        os.path.join(
            OUT22,
            "nominal_cells_per_condition.csv"
        ),
        index=False
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "R2.2 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\n=== BRANCH SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== NOMINAL CELLS PER CONDITION ==="
    )

    print(
        cond_summary.to_string(
            index=False
        )
    )

    print(
        "\nObserved profile rows:",
        f"{len(observed_profiles):,}"
    )

    print(
        "Nominal-concordant profile rows:",
        f"{len(nominal_profiles):,}"
    )

    print(
        "\nOutput directory:",
        OUT22
    )


# ============================================================
# FINAL SCIENTIFIC REPRODUCIBILITY CHECKS
# ============================================================
# These are post hoc checks on the already reconstructed outputs.
# They do not determine which records are retained.

if GLOBAL_OK:
    expected_branch_counts = {
        "observed_metadata": (84195, 181, 203, 24),
        "nominal_concordant": (83045, 180, 180, 24),
    }
    observed_branch_counts = {
        str(record.branch): (
            int(record.n_cycles), int(record.n_cells),
            int(record.n_sample_condition_pairs), int(record.n_conditions)
        )
        for record in summary.itertuples(index=False)
    }
    expected_cells_per_condition = {
        condition_id: (7 if condition_id in (3, 4)
                       else 3 if condition_id in (23, 24)
                       else 8)
        for condition_id in range(1, 25)
    }
    observed_cells_per_condition = dict(zip(
        cond_summary["Condition_ID"].astype(int),
        cond_summary["n_nominal_cells"].astype(int),
    ))

    print("\n" + "=" * 70)
    print("R2.2 — MANUSCRIPT REPRODUCIBILITY AUDIT")
    print("=" * 70)
    print(f"Unique duplicate cycle keys: {len(dup_keys):,}")
    print(f"Nominally excluded cycles: {len(observed_cycles)-len(nominal_cycles):,}")
    print(f"Output profile rows: observed={len(observed_profiles):,}, "
          f"nominal={len(nominal_profiles):,}")

    valid = (
        observed_branch_counts == expected_branch_counts
        and observed_cells_per_condition == expected_cells_per_condition
        and len(dup_keys) == 0
        and len(observed_profiles) == 20503
        and len(nominal_profiles) == 18180
    )
    print("Manuscript-reference invariants:", "PASS" if valid else "MISMATCH")
    if not valid:
        print("Observed branches:", observed_branch_counts)
        print("Expected branches:", expected_branch_counts)
        print("Observed condition cell counts:", observed_cells_per_condition)
        print("Expected condition cell counts:", expected_cells_per_condition)
        # Explicit failure is useful for the fixed CALCE input series.
        # Disable only for a deliberately different input dataset.
        if os.environ.get("SOC_R22_ASSERT_MANUSCRIPT", "1") == "1":
            raise AssertionError(
                "R2.2 outputs differ from the frozen manuscript. "
                "Investigate inputs and processing; do not alter selection rules."
            )
    print("=" * 70)
