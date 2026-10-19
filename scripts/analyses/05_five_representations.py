# -*- coding: utf-8 -*-
# CALCE V-SoC Colab scientific stage script.
# Run on Google Colab with upstream files present in /content.
# Outputs are non-destructive: existing stage directory and ZIP are not replaced.

from pathlib import Path
import hashlib, os, shutil, json

SOURCE_DIR = Path('/content/revision_R2_2_global')
BROAD_DIR = Path('/content/outputs/R4B_broad_52_132')
REBUILD_DIR = Path('/content/SoC_Rebuild_ConditionvCodes2')
REBUILD_DIR.mkdir(exist_ok=True)

INPUT_CYCLES = SOURCE_DIR / 'global_cycle_profiles.pkl'
INPUT_CELLS = SOURCE_DIR / 'cell_profiles_nominal_concordant.csv'
INPUT_BROAD = BROAD_DIR / 'broad_52_132_cell_profiles.csv'


def require_path(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required input missing: {path}')
    return path


def stage_dir(stage):
    path = REBUILD_DIR / stage
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Nonempty stage directory (refuse to overwrite): {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_zip(stage, prefix):
    folder = REBUILD_DIR / stage
    path = Path('/content') / f'{prefix}ConditionvCodes2.zip'
    if path.exists():
        raise FileExistsError(f'Refuse to replace existing scientific archive: {path}')
    archive = Path(shutil.make_archive(str(path.with_suffix('')), 'zip', root_dir=str(folder)))
    with __import__('zipfile').ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise RuntimeError(f'Archive CRC failed: {archive}')
    print('STAGE FILES:', folder)
    print('ARCHIVE:', archive)
    print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive

for name, file in [('Nominal cycles',INPUT_CYCLES), ('Full cell profiles',INPUT_CELLS), ('Window cell profiles',INPUT_BROAD)]:
    print(name, 'FOUND' if file.is_file() else 'MISSING', str(file))

# SOURCE-ALIGNED SCIENTIFIC BLOCK: 05 — Five representations, paired bootstrap
# Scientific block SHA256: e175b8cf0803cd06643d20603cb6108207f2fe2282bc68f1466dbdc8f03acee6
# Original notebook cell 12 copied byte-for-byte below.

# Commented out IPython magic to ensure Python compatibility.
# GOOGLE COLAB — TAMAMINI TEK HÜCREYE YAPIŞTIRIP ÇALIŞTIRIN.
#
# Zorunlu manuel girdi:
#   SoC_unified_xaevhp2i.zip
#
# İsteğe bağlı olarak aynı anda şu CSV'leri de seçebilirsiniz:
#   cell_profiles_nominal_concordant.csv
#   broad_52_132_cell_profiles.csv
#
# CSV seçilmezse ZIP içindeki inputs/ klasöründen okunur.
# Colab'ın dosya adına eklediği " (1)", " (2)" ekleri kabul edilir.
#
# Karşılaştırmalar:
#   1. global_iso       : tek ortak izotonik eğri
#   2. temperature_iso  : yalnız sıcaklığa bağlı izotonik eğriler
#   3. rate_iso         : yalnız deşarj hızına bağlı izotonik eğriler
#   4. condition_raw    : koşula özgü hücre medyanı, izotonik uygulanmaz
#   5. condition_iso    : mevcut birleşik izotonik LUT
#
# Doğrulama:
#   LOCO, hız tutma, sıcaklık tutma
#   Tam geçmiş ve 52–132 ortak pencere
#
# Ana karşılaştırma:
#   Beş yöntemin ortak hücre–SoC düğümleri
#   Hücre başına en az 5 değerlendirilebilir düğüm
#   Hücre MAE'lerinin aritmetik ortalaması
#
# Bootstrap:
#   Sabit test tahminlerindeki eşleştirilmiş hata farklarının
#   koşul-tabakalı fiziksel hücre bootstrap'ı; 2000 tekrar.
#   Modeller bootstrap içinde yeniden eğitilmez.

import io
import re
import json
import hashlib
import warnings
import shutil
import tempfile
import platform
import zipfile

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression


# ============================================================
# AYARLAR
# ============================================================

BOOTSTRAP_REPS = 2000
SEED = 20260928
MIN_EVAL_POINTS = 5
NUMERIC_TIE_MV = 1e-9

GRID = np.arange(101)

MODELS = (
    "global_iso",
    "temperature_iso",
    "rate_iso",
    "condition_raw",
    "condition_iso",
)

FILES = {
    "full_history": "cell_profiles_nominal_concordant.csv",
    "broad_52_132": "broad_52_132_cell_profiles.csv",
}

CELL_KEY = [
    "analysis",
    "validation",
    "Condition_ID",
    "sample_int",
]

NODE_KEY = CELL_KEY + ["SoC"]

# Bunlar LUT doğrulamasının yardımcı hata bantlarıdır.
# Son bant 90'ı içerir; kontrast analizindeki [60,90) ile karıştırmayın.
BANDS = [
    ("all_0_90_inclusive", 0, 91),
    ("[0,20)", 0, 20),
    ("[20,60)", 20, 60),
    ("[60,90]", 60, 91),
]

META = pd.DataFrame(
    [
        (1 + 6*t + 2*r + k, T, cutoff, rate)
        for t, T in enumerate((10, 25, 45, 60))
        for r, rate in enumerate((0.7, 1.0, 2.0))
        for k, cutoff in enumerate(("C/5", "C/40"))
    ],
    columns=[
        "Condition_ID",
        "Temperature",
        "Rate",
        "Charge_Rate",
    ],
)

MAP = META.set_index("Condition_ID")

LOOKUP = {
    (r.Temperature, r.Rate, r.Charge_Rate): r.Condition_ID
    for r in META.itertuples()
}


# ============================================================
# YARDIMCILAR VE GİRDİ KONTROLLERİ
# ============================================================

def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def csv_save(df, out, name):
    df.to_csv(Path(out) / name, index=False)


def canonical(name):
    p = Path(name)
    stem = re.sub(r"\s*\(\d+\)$", "", p.stem).lower()
    return stem + p.suffix.lower()


def member(z, basename):
    found = [
        n for n in z.namelist()
        if not n.endswith("/") and Path(n).name == basename
    ]

    require(
        len(found) == 1,
        f"ZIP: {basename} tekil bulunamadı: {found}",
    )

    info = z.getinfo(found[0])

    require(
        info.file_size < 300_000_000,
        f"ZIP üyesi beklenenden büyük: {basename}",
    )

    return z.read(found[0])


def load_profiles(blob, name):
    d = pd.read_csv(io.BytesIO(blob))
    d.columns = d.columns.str.strip()

    if "V_cell_median" in d:
        require(
            "Voltage" not in d,
            f"{name}: iki voltaj sütunu var; belirsiz girdi.",
        )
        d = d.rename(columns={"V_cell_median": "Voltage"})

    required = ["Condition_ID", "sample_int", "SoC", "Voltage"]

    require(
        set(required).issubset(d.columns),
        f"{name}: gerekli sütunlar: {required}",
    )

    for c in required:
        d[c] = pd.to_numeric(d[c], errors="raise")

    for c in required[:3]:
        a = d[c].to_numpy(float)

        require(
            np.isfinite(a).all()
            and np.allclose(a, np.round(a), rtol=0, atol=1e-8),
            f"{name}: {c} tam sayı ve sonlu olmalı.",
        )

        d[c] = np.round(a).astype(int)

    require(
        not np.isinf(d.Voltage).any(),
        f"{name}: sonsuz voltaj değeri var.",
    )

    require(
        d.Condition_ID.between(1, 24).all()
        and d.SoC.between(0, 100).all(),
        f"{name}: koşul veya SoC aralığı hatalı.",
    )

    require(
        not d.duplicated(required[:3]).any(),
        f"{name}: yinelenen hücre–koşul–SoC kaydı var.",
    )

    require(
        not (d.groupby("sample_int").Condition_ID.nunique() > 1).any(),
        f"{name}: bir fiziksel hücre birden çok koşulda.",
    )

    for (cid, sid), g in d.groupby(["Condition_ID", "sample_int"]):
        if cid <= 22:
            allowed = range(8*(cid-1)+1, 8*cid+1)
        elif cid == 23:
            allowed = range(177, 180)
        else:
            allowed = range(185, 188)

        require(
            sid in allowed,
            f"{name}: nominal harita uyuşmazlığı: koşul={cid}, hücre={sid}",
        )

        require(
            set(g.SoC) == set(GRID),
            f"{name}: hücre {sid} için 0..100 satırları eksik. "
            "Kırpılmamış profil dosyasını kullanın; voltajlar NaN olabilir.",
        )

    if name == "full_history":
        require(
            set(d.Condition_ID) == set(range(1, 25)),
            "Tam geçmiş dosyasında 24 koşulun tamamı bulunmalı.",
        )

    # Kaynak değişken eşlemesi:
    # Rate        = şarj sonlandırma akımı
    # Charge_Rate = deşarj C-oranı
    for c in ("Temperature", "Rate", "Charge_Rate"):
        if c not in d:
            continue

        expected = d.Condition_ID.map(MAP[c])

        if c == "Rate":
            actual = (
                d[c].astype(str)
                .str.replace(" ", "", regex=False)
                .str.upper()
            )
            ok = actual.eq(expected)
        else:
            actual = d[c].astype(str)
            actual = actual.str.replace("°", "", regex=False)
            actual = actual.str.replace("º", "", regex=False)
            actual = actual.str.replace("C", "", regex=False)
            actual = pd.to_numeric(actual.str.strip(), errors="coerce")

            ok = np.isclose(
                actual,
                expected,
                rtol=0,
                atol=1e-8,
            )

        require(
            np.asarray(ok).all(),
            f"{name}: {c} metadata uyuşmazlığı.",
        )

    return d[required].merge(
        META,
        on="Condition_ID",
        validate="many_to_one",
    )


def load_inputs(uploaded):
    zipnames = [
        n for n in uploaded
        if n.lower().endswith(".zip")
    ]

    require(
        len(zipnames) == 1,
        "Yalnız bir önceki SoC_unified_....zip dosyasını seçin.",
    )

    z = zipfile.ZipFile(io.BytesIO(uploaded[zipnames[0]]))

    old_config = json.loads(member(z, "run_config.json"))

    datasets = {}
    hashes = {}
    provenance = {}

    for analysis, filename in FILES.items():
        matched = [
            n for n in uploaded
            if canonical(n) == filename.lower()
        ]

        require(
            len(matched) <= 1,
            f"Aynı CSV birden fazla seçilmiş: {matched}",
        )

        if matched:
            blob = uploaded[matched[0]]
            provenance[analysis] = matched[0]
        else:
            blob = member(z, filename)
            provenance[analysis] = "reference_zip/inputs/" + filename

        hashes[filename] = hashlib.sha256(blob).hexdigest()
        expected = old_config.get("input_sha256", {}).get(filename)

        require(
            expected is not None,
            f"Referans ZIP girdi hash kaydı eksik: {filename}",
        )

        require(
            hashes[filename] == expected,
            f"{filename}: önceki çalışmanın girdisiyle SHA256 uyuşmuyor. "
            "Aynı dosyayı kullanın.",
        )

        datasets[analysis] = load_profiles(blob, analysis)

    full = datasets["full_history"][
        ["Condition_ID", "sample_int"]
    ].drop_duplicates()

    broad = datasets["broad_52_132"][
        ["Condition_ID", "sample_int"]
    ].drop_duplicates()

    require(
        set(map(tuple, broad.values)) <= set(map(tuple, full.values)),
        "Ortak pencere hücreleri tam geçmişin alt kümesi olmalı.",
    )

    reference = pd.read_csv(
        io.BytesIO(member(z, "validation_node_predictions.csv"))
    )
    reference = reference[reference.method.eq("unified")].copy()

    summary = pd.read_csv(
        io.BytesIO(member(z, "validation_summary.csv"))
    )
    summary = summary[
        summary.method.eq("unified")
        & summary.comparison.eq("own_support")
    ].copy()

    lut = np.load(
        io.BytesIO(member(z, "final_LUT_float64.npy")),
        allow_pickle=False,
    )

    support = np.load(
        io.BytesIO(member(z, "final_LUT_support.npy")),
        allow_pickle=False,
    )

    refhash = hashlib.sha256(
        uploaded[zipnames[0]]
    ).hexdigest()

    z.close()

    return (
        datasets,
        reference,
        summary,
        lut,
        support,
        hashes,
        provenance,
        refhash,
    )


# ============================================================
# ORTAK EĞRİ KURMA FONKSİYONU
# ============================================================

def build_curve(A, raw=False):
    """
    A: benzersiz eğitim hücreleri x 101 SoC düğümü.

    Bir satır bir fiziksel hücrenin medyan profilidir.
    Fit alanı: 0..100.
    Minimum destek: düğüm başına 1 eğitim hücresi.
    Minimum destekli fit düğümü: 2.
    Sıfır destekli düğüm: NaN, doldurma yok.

    raw=False:
        Hücre sayısıyla ağırlıklı izotonik regresyon.

    raw=True:
        Aynı eğitim hücrelerinin düğüm medyanı.
        Destek ve minimum fit düğümü şartları aynıdır.
    """
    A = np.asarray(A, dtype=float).reshape(-1, 101)

    n = np.isfinite(A).sum(axis=0)
    med = np.full(101, np.nan)

    if len(A):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(A, axis=0)

    ok = n >= 1
    p = np.full(101, np.nan)
    status = "ok"

    if not len(A):
        status = "insufficient_training_cells"
    elif ok.sum() < 2:
        status = "insufficient_fit_nodes"
    elif raw:
        p[ok] = med[ok]
    else:
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="nan",
        )

        iso.fit(
            GRID[ok],
            med[ok],
            sample_weight=n[ok],
        )

        p[ok] = iso.predict(GRID[ok])

        require(
            np.all(np.diff(p[ok]) >= -1e-12),
            "Monotonluk kontrolü başarısız.",
        )

    require(
        not np.isfinite(p[n == 0]).any(),
        "Sıfır destekli düğüm doldurulmuş.",
    )

    return {
        "pred": p,
        "support": n,
        "raw": med,
        "status": status,
        "n_train_cells": len(A),
        "n_fit_nodes": int(ok.sum()),
    }


class Fold:
    """
    Tutulan hücre veya faktör düzeyi baştan çıkarılır.
    Beş temsil de bu eğitim kümesinden kurulur.
    """

    def __init__(self, frame, analysis, fold_id, audit):
        self.frame = frame
        self.analysis = analysis
        self.fold_id = fold_id
        self.audit = audit
        self.cache = {}

    def curve(self, model, cid, T, rate):
        if model == "global_iso":
            key = (model, "all")
            mask = np.ones(len(self.frame), bool)

        elif model == "temperature_iso":
            key = (model, float(T))
            mask = self.frame.Temperature.eq(T)

        elif model == "rate_iso":
            key = (model, float(rate))
            mask = self.frame.Charge_Rate.eq(rate)

        else:
            key = (model, int(cid))
            mask = self.frame.Condition_ID.eq(cid)

        if key not in self.cache:
            g = self.frame.loc[mask]
            A = g[list(GRID)].to_numpy(float)

            fit = build_curve(
                A,
                raw=(model == "condition_raw"),
            )

            fit["fit_id"] = len(self.audit)

            self.audit.append({
                "fit_id": fit["fit_id"],
                "analysis": self.analysis,
                "fold": self.fold_id,
                "model": model,
                "group": str(key[1]),
                "n_train_cells": fit["n_train_cells"],
                "n_fit_nodes": fit["n_fit_nodes"],
                "status": fit["status"],
                "training_cells": ";".join(
                    map(str, g.index.tolist())
                ),
                "training_conditions": ";".join(
                    map(str, sorted(g.Condition_ID.unique()))
                ),
            })

            self.cache[key] = fit

        return self.cache[key]

    def predict(self, model, cid, validation):
        row = MAP.loc[cid]

        T = row.Temperature
        rate = row.Charge_Rate
        cutoff = row.Rate

        uses_temperature = model in (
            "temperature_iso",
            "condition_raw",
            "condition_iso",
        )

        uses_rate = model in (
            "rate_iso",
            "condition_raw",
            "condition_iso",
        )

        if validation == "temperature_holdout" and uses_temperature:
            lo, hi = (10, 45) if T == 25 else (25, 60)
            w = (T - lo) / (hi - lo)

            a = self.curve(
                model,
                LOOKUP[lo, cutoff, rate],
                lo,
                rate,
            )

            b = self.curve(
                model,
                LOOKUP[hi, cutoff, rate],
                hi,
                rate,
            )

        elif validation == "rate_holdout" and uses_rate:
            lo, hi = 0.7, 2.0
            w = (1.0 - lo) / (hi - lo)

            a = self.curve(
                model,
                LOOKUP[T, cutoff, lo],
                T,
                lo,
            )

            b = self.curve(
                model,
                LOOKUP[T, cutoff, hi],
                T,
                hi,
            )

        else:
            w = np.nan
            a = self.curve(model, cid, T, rate)
            b = None

        if b is None:
            p = a["pred"].copy()
        else:
            # Bir komşu NaN ise sonuç NaN.
            # Tek taraftan taşıma yapılmaz.
            p = (1 - w)*a["pred"] + w*b["pred"]

        return p, a, b, w


# ============================================================
# TAHMİNLER
# ============================================================

def predict_dataset(d, analysis):
    frame = d.pivot(
        index="sample_int",
        columns="SoC",
        values="Voltage",
    ).reindex(columns=GRID)

    metadata = (
        d.drop_duplicates("sample_int")
        .set_index("sample_int")[
            ["Condition_ID", "Temperature", "Rate", "Charge_Rate"]
        ]
    )

    frame = frame.join(metadata).sort_index()

    rows = []
    audit = []
    factor_folds = {}

    specifications = [
        ("rate_holdout", "Charge_Rate", [1.0]),
        ("temperature_holdout", "Temperature", [25, 45]),
    ]

    for validation, column, levels in specifications:
        for level in levels:
            train = frame.loc[frame[column].ne(level)].copy()

            require(
                not train[column].eq(level).any(),
                "Tutulan faktör eğitimde kaldı.",
            )

            factor_folds[validation, level] = Fold(
                train,
                analysis,
                f"{validation}_{level}",
                audit,
            )

    for k, (sid, target) in enumerate(frame.iterrows(), 1):
        cid = int(target.Condition_ID)

        loco = Fold(
            frame.drop(index=sid),
            analysis,
            f"LOCO_{sid}",
            audit,
        )

        require(
            sid not in loco.frame.index,
            "LOCO test hücresi eğitimde kaldı.",
        )

        tasks = [("LOCO", loco)]

        if target.Charge_Rate == 1.0:
            tasks.append((
                "rate_holdout",
                factor_folds["rate_holdout", 1.0],
            ))

        if target.Temperature in (25, 45):
            tasks.append((
                "temperature_holdout",
                factor_folds[
                    "temperature_holdout",
                    target.Temperature,
                ],
            ))

        for validation, fold in tasks:
            q = pd.DataFrame({
                "analysis": analysis,
                "validation": validation,
                "Condition_ID": cid,
                "sample_int": int(sid),
                "SoC": np.arange(91),
                "Temperature": target.Temperature,
                "Rate": target.Rate,
                "Charge_Rate": target.Charge_Rate,
                "target_V": target[
                    list(range(91))
                ].to_numpy(float),
            })

            for model in MODELS:
                p, a, b, w = fold.predict(
                    model,
                    cid,
                    validation,
                )

                q[model] = p[:91]
                q[model + "_support_a"] = a["support"][:91]
                q[model + "_support_b"] = (
                    b["support"][:91]
                    if b is not None else np.nan
                )
                q[model + "_fit_a"] = a["fit_id"]
                q[model + "_fit_b"] = (
                    b["fit_id"]
                    if b is not None else np.nan
                )
                q[model + "_weight_b"] = w

                reason = np.full(
                    91,
                    "evaluable",
                    dtype=object,
                )

                reason[~np.isfinite(p[:91])] = (
                    "insufficient_training_curve"
                )

                zero = a["support"][:91] == 0

                if b is not None:
                    zero |= b["support"][:91] == 0

                reason[zero] = "zero_training_support"

                reason[
                    ~np.isfinite(q.target_V.to_numpy())
                ] = "missing_test_target"

                q[model + "_reason"] = reason

            q["common_five"] = (
                np.isfinite(q.target_V)
                & np.isfinite(q[list(MODELS)]).all(axis=1)
            )

            require(
                np.array_equal(
                    np.isfinite(q.condition_raw),
                    np.isfinite(q.condition_iso),
                ),
                "Ham/izotonik destek maskeleri farklı.",
            )

            rows.append(q)

        if k % 20 == 0 or k == len(frame):
            print(
                f"{analysis}: {k}/{len(frame)} hücre tamamlandı.",
                flush=True,
            )

    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(audit),
    )


# ============================================================
# ESKİ SONUÇLARI YENİDEN ÜRETME KONTROLLERİ
# ============================================================

def check_reference(nodes, reference, out):
    require(
        not reference.duplicated(NODE_KEY).any(),
        "Referans düğüm anahtarı yineleniyor.",
    )

    require(
        not nodes.duplicated(NODE_KEY).any(),
        "Yeni düğüm anahtarı yineleniyor.",
    )

    old = reference[
        NODE_KEY + ["target_V", "prediction_V"]
    ]

    new = nodes[
        NODE_KEY + ["target_V", "condition_iso"]
    ]

    q = new.merge(
        old,
        on=NODE_KEY,
        how="outer",
        suffixes=("_new", "_ref"),
        indicator=True,
        validate="one_to_one",
    )

    require(
        q["_merge"].eq("both").all(),
        "Referansla test hücresi/düğümü kümesi farklı.",
    )

    checks = []

    quantities = [
        ("target", "target_V_new", "target_V_ref"),
        ("prediction", "condition_iso", "prediction_V"),
    ]

    for kind, x, y in quantities:
        for key, g in q.groupby(["analysis", "validation"]):
            v = g[x].to_numpy(float)
            u = g[y].to_numpy(float)

            same = np.array_equal(
                np.isfinite(v),
                np.isfinite(u),
            )

            mask = np.isfinite(v) & np.isfinite(u)

            error = (
                float(np.max(np.abs(v[mask] - u[mask]))*1000)
                if mask.any() else 0.0
            )

            checks.append({
                "analysis": key[0],
                "validation": key[1],
                "quantity": kind,
                "same_finite_mask": same,
                "max_abs_difference_mV": error,
                "n_nodes": len(g),
            })

    checks = pd.DataFrame(checks)
    csv_save(checks, out, "reference_node_check.csv")

    require(
        checks.same_finite_mask.all()
        and checks.max_abs_difference_mV.le(1e-6).all(),
        "Referans tahminleri/gerçekleri yeniden üretilemedi.",
    )

    print(
        "Referans düğüm kontrolü geçti; en büyük fark (mV):",
        checks.max_abs_difference_mV.max(),
    )


# ============================================================
# HÜCRE HATALARI VE KAPSAM
# ============================================================

def metric_row(q, prediction, mask):
    n = int(mask.sum())
    eligible = n >= MIN_EVAL_POINTS

    errors = (
        prediction[mask]
        - q.target_V.to_numpy(float)[mask]
    )*1000

    n_target = int(np.isfinite(q.target_V).sum())

    return {
        "n_grid": len(q),
        "n_target": n_target,
        "n_paired": n,
        "n_used": n if eligible else 0,
        "eligible": eligible,
        "n_missing_target": len(q) - n_target,
        "n_target_not_paired": n_target - n,
        "n_pairs_discarded_min_nodes": n if not eligible else 0,
        "MAE_mV": (
            float(np.mean(abs(errors)))
            if eligible else np.nan
        ),
        "RMSE_mV": (
            float(np.sqrt(np.mean(errors**2)))
            if eligible else np.nan
        ),
        "bias_mV": (
            float(np.mean(errors))
            if eligible else np.nan
        ),
        "max_abs_mV": (
            float(np.max(abs(errors)))
            if eligible else np.nan
        ),
    }


def compute_metrics(nodes):
    rows = []
    pairs = []

    for key, g in nodes.groupby(CELL_KEY, sort=True):
        base = dict(zip(CELL_KEY, key))

        base.update(
            Temperature=g.Temperature.iloc[0],
            Rate=g.Rate.iloc[0],
            Charge_Rate=g.Charge_Rate.iloc[0],
        )

        for band, lo, hi in BANDS:
            q = g[g.SoC.between(lo, hi-1)]

            target = np.isfinite(
                q.target_V.to_numpy(float)
            )
            common = q.common_five.to_numpy(bool)

            for model in MODELS:
                p = q[model].to_numpy(float)
                own = target & np.isfinite(p)

                for scope, mask in [
                    ("own_support", own),
                    ("common_five", common),
                ]:
                    rows.append({
                        **base,
                        "band": band,
                        "model": model,
                        "comparison": scope,
                        **metric_row(q, p, mask),
                    })

            # İkili karşılaştırmalar tam 0..90 alanında yapılır.
            if band != "all_0_90_inclusive":
                continue

            for left, right in combinations(MODELS, 2):
                p = q[left].to_numpy(float)
                r = q[right].to_numpy(float)

                pairwise = (
                    target
                    & np.isfinite(p)
                    & np.isfinite(r)
                )

                for scope, mask in [
                    ("common_five", common),
                    ("pairwise_common", pairwise),
                ]:
                    L = metric_row(q, p, mask)
                    R = metric_row(q, r, mask)

                    pairs.append({
                        **base,
                        "band": band,
                        "comparison": scope,
                        "left_model": left,
                        "right_model": right,
                        "n_grid": len(q),
                        "n_target": int(target.sum()),
                        "n_paired": L["n_paired"],
                        "n_used": L["n_used"],
                        "eligible": L["eligible"],
                        "left_MAE_mV": L["MAE_mV"],
                        "right_MAE_mV": R["MAE_mV"],
                        # Pozitif: sağdaki yöntem daha iyi.
                        "gain_right_mV": (
                            L["MAE_mV"] - R["MAE_mV"]
                        ),
                    })

    return pd.DataFrame(rows), pd.DataFrame(pairs)


def aggregate(metrics, condition=False):
    keys = [
        "analysis",
        "validation",
        "comparison",
        "band",
        "model",
    ]

    if condition:
        keys += [
            "Condition_ID",
            "Temperature",
            "Rate",
            "Charge_Rate",
        ]

    s = metrics.groupby(keys, dropna=False).agg(
        n_planned_cells=("sample_int", "size"),
        n_evaluated_cells=("eligible", "sum"),
        n_grid=("n_grid", "sum"),
        n_target=("n_target", "sum"),
        n_paired=("n_paired", "sum"),
        n_used=("n_used", "sum"),
        n_missing_target=("n_missing_target", "sum"),
        n_target_not_paired=("n_target_not_paired", "sum"),
        n_pairs_discarded_min_nodes=(
            "n_pairs_discarded_min_nodes", "sum"
        ),
        MAE_mV=("MAE_mV", "mean"),
        mean_cell_RMSE_mV=("RMSE_mV", "mean"),
        bias_mV=("bias_mV", "mean"),
        worst_abs_mV=("max_abs_mV", "max"),
    ).reset_index()

    s["n_skipped_cells"] = (
        s.n_planned_cells - s.n_evaluated_cells
    )
    s["n_grid_not_used"] = s.n_grid - s.n_used

    s["target_coverage_pct"] = (
        100*s.n_used / s.n_target.replace(0, np.nan)
    )
    s["nominal_grid_coverage_pct"] = (
        100*s.n_used / s.n_grid.replace(0, np.nan)
    )
    s["cell_coverage_pct"] = (
        100*s.n_evaluated_cells / s.n_planned_cells
    )

    return s


def check_summary(summary, old, out):
    new = summary[
        summary.model.eq("condition_iso")
        & summary.comparison.eq("own_support")
    ]

    keys = ["analysis", "validation", "band"]

    columns = [
        "MAE_mV",
        "mean_cell_RMSE_mV",
        "bias_mV",
        "worst_abs_mV",
        "n_planned_cells",
        "n_evaluated_cells",
        "n_grid",
        "n_target",
        "n_paired",
        "n_used",
    ]

    q = new[keys + columns].merge(
        old[keys + columns],
        on=keys,
        how="outer",
        suffixes=("_new", "_ref"),
        validate="one_to_one",
        indicator=True,
    )

    require(
        q["_merge"].eq("both").all(),
        "Referans özet kayıtları farklı.",
    )

    for c in columns:
        x = q[c + "_new"]
        y = q[c + "_ref"]

        q[c + "_abs_difference"] = abs(x-y)

        require(
            np.allclose(
                x, y,
                atol=1e-6,
                rtol=0,
                equal_nan=True,
            ),
            "Referans özet uyuşmuyor: " + c,
        )

    csv_save(q, out, "reference_summary_check.csv")

    print(
        "Referans MAE, RMSE, bias, en kötü hata "
        "ve kapsam sayımları doğrulandı."
    )


# ============================================================
# EŞLEŞTİRİLMİŞ HÜCRE BOOTSTRAP'I
# ============================================================

def bootstrap_pairs(pairs, out, reps=BOOTSTRAP_REPS):
    """
    Sabit out-of-fold hata farklarında koşul-tabakalı
    fiziksel hücre bootstrap'ı.

    İki yöntem için aynı hücreler birlikte örneklenir.
    Modeller yeniden fit edilmez.
    CI'lar noktasaldır ve mevcut test tahminlerine koşulludur.
    """
    require(
        reps >= 2000,
        "Nihai rapor için en az 2000 bootstrap kullanın.",
    )

    rng = np.random.default_rng(SEED)
    results = []
    draw_export = {}

    keys = [
        "analysis",
        "validation",
        "comparison",
        "left_model",
        "right_model",
    ]

    for idx, (key, g) in enumerate(
        pairs.groupby(keys, sort=True)
    ):
        e = (
            g[g.eligible]
            .sort_values(["Condition_ID", "sample_int"])
            .reset_index(drop=True)
        )

        row = dict(zip(keys, key))

        row.update(
            n_planned_cells=len(g),
            n_evaluated_cells=len(e),
            n_skipped_cells=len(g)-len(e),
            n_grid=int(g.n_grid.sum()),
            n_target=int(g.n_target.sum()),
            n_used=int(g.n_used.sum()),
            n_conditions=int(e.Condition_ID.nunique()),
        )

        row["target_coverage_pct"] = (
            100*row["n_used"]/row["n_target"]
            if row["n_target"] else np.nan
        )

        if not len(e):
            row.update(
                status="no_evaluable_cells",
                gain_right_mV=np.nan,
                CI95_low_mV=np.nan,
                CI95_high_mV=np.nan,
                bootstrap_reps=0,
            )

        else:
            left = e.left_MAE_mV.to_numpy()
            right = e.right_MAE_mV.to_numpy()
            delta = left-right

            require(
                np.isfinite(delta).all(),
                "Bootstrap farkları sonlu değil.",
            )

            require(
                not e.sample_int.duplicated().any(),
                "Aynı deneyde hücre birden çok kez sayılmış.",
            )

            L = np.zeros(reps)
            R = np.zeros(reps)

            strata = [
                part.index.to_numpy()
                for _, part in e.groupby(
                    "Condition_ID",
                    sort=True,
                )
            ]

            for ids in strata:
                take = rng.choice(
                    ids,
                    size=(reps, len(ids)),
                    replace=True,
                )

                L += left[take].sum(axis=1)
                R += right[take].sum(axis=1)

            L /= len(e)
            R /= len(e)

            draws = L-R
            draws[abs(draws) <= NUMERIC_TIE_MV] = 0.0

            low, high = np.quantile(
                draws,
                [0.025, 0.975],
            )

            gain = float(delta.mean())
            denominator = float(left.mean())

            if abs(gain) <= NUMERIC_TIE_MV:
                gain = 0.0

            row.update(
                status="ok_conditional_fixed_predictions",
                left_MAE_mV=denominator,
                right_MAE_mV=float(right.mean()),
                gain_right_mV=gain,
                gain_right_percent=(
                    100*gain/denominator
                    if denominator else np.nan
                ),
                CI95_low_mV=float(low),
                CI95_high_mV=float(high),
                CI_excludes_zero=bool(
                    low > NUMERIC_TIE_MV
                    or high < -NUMERIC_TIE_MV
                ),
                bootstrap_reps=reps,
                bootstrap_SD_mV=float(
                    np.std(draws, ddof=1)
                ),
                degenerate_bootstrap=bool(
                    np.ptp(draws) <= NUMERIC_TIE_MV
                ),
                n_cells_right_better=int(
                    (delta > NUMERIC_TIE_MV).sum()
                ),
                n_cells_tied=int(
                    (abs(delta) <= NUMERIC_TIE_MV).sum()
                ),
                n_cells_left_better=int(
                    (delta < -NUMERIC_TIE_MV).sum()
                ),
                n_singleton_strata=sum(
                    len(ids) == 1 for ids in strata
                ),
                max_abs_cell_gain_mV=float(
                    abs(delta).max()
                ),
                bootstrap_fraction_right_better=float(
                    (draws > NUMERIC_TIE_MV).mean()
                ),
                bootstrap_fraction_tied=float(
                    (abs(draws) <= NUMERIC_TIE_MV).mean()
                ),
            )

            label = "|".join(map(str, key))
            draw_export[label] = draws

        results.append(row)

        if (idx+1) % 20 == 0:
            print(
                f"Eşleştirilmiş bootstrap: "
                f"{idx+1} karşılaştırma tamamlandı.",
                flush=True,
            )

    np.savez_compressed(
        Path(out) / "bootstrap_gain_draws_mV.npz",
        **draw_export,
    )

    result = pd.DataFrame(results)

    csv_save(
        result,
        out,
        "paired_bootstrap_summary.csv",
    )

    return result


# ============================================================
# NİHAİ TEMSİLLER VE LUT REFERANS KONTROLÜ
# ============================================================

def export_final(datasets, old_lut, old_support, out):
    rows = []

    for analysis, d in datasets.items():
        frame = d.pivot(
            index="sample_int",
            columns="SoC",
            values="Voltage",
        ).reindex(columns=GRID)

        metadata = (
            d.drop_duplicates("sample_int")
            .set_index("sample_int")[
                [
                    "Condition_ID",
                    "Temperature",
                    "Rate",
                    "Charge_Rate",
                ]
            ]
        )

        frame = frame.join(metadata)

        audit = []
        fold = Fold(
            frame,
            analysis,
            "all_available_cells",
            audit,
        )

        for cid in range(1, 25):
            for model in MODELS:
                p, a, b, w = fold.predict(
                    model,
                    cid,
                    "final",
                )

                for soc in GRID:
                    rows.append({
                        "analysis": analysis,
                        "Condition_ID": cid,
                        "model": model,
                        "SoC": int(soc),
                        "prediction_V": float(p[soc]),
                        "n_training_cells_at_node": int(
                            a["support"][soc]
                        ),
                        "n_train_cells": a["n_train_cells"],
                        "fit_status": a["status"],
                    })

    lut = pd.DataFrame(rows)

    csv_save(
        lut,
        out,
        "all_representations_0_100.csv",
    )

    csv_save(
        lut[lut.SoC <= 90],
        out,
        "all_representations_0_90.csv",
    )

    final = lut[
        lut.analysis.eq("full_history")
        & lut.model.eq("condition_iso")
        & lut.SoC.le(90)
    ]

    arr = final.pivot(
        index="Condition_ID",
        columns="SoC",
        values="prediction_V",
    ).to_numpy()

    support = final.pivot(
        index="Condition_ID",
        columns="SoC",
        values="n_training_cells_at_node",
    ).to_numpy()

    require(
        arr.shape == old_lut.shape
        and np.array_equal(
            np.isfinite(arr),
            np.isfinite(old_lut),
        ),
        "Nihai LUT şekli/sonlu maskesi değişti.",
    )

    require(
        np.allclose(
            arr,
            old_lut,
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        ),
        "Nihai LUT değerleri değişti.",
    )

    require(
        np.array_equal(support, old_support),
        "Nihai LUT hücre desteği değişti.",
    )

    np.save(
        Path(out) / "final_condition_LUT_float64.npy",
        arr,
    )

    np.save(
        Path(out) / "final_condition_LUT_float32.npy",
        arr.astype("float32"),
    )

    qa = {
        "final_lut_reference_max_difference_mV": float(
            np.nanmax(abs(arr-old_lut))*1000
        ),
        "final_lut_shape": list(arr.shape),
        "final_lut_finite_nodes": int(
            np.isfinite(arr).sum()
        ),
    }

    # İzotonik düzeltmenin düğüm bazlı büyüklüğü ve desteği.
    pair = lut[
        lut.model.isin(["condition_raw", "condition_iso"])
    ].pivot(
        index=["analysis", "Condition_ID", "SoC"],
        columns="model",
        values="prediction_V",
    ).reset_index()

    supports = lut[lut.model.eq("condition_iso")][
        [
            "analysis",
            "Condition_ID",
            "SoC",
            "n_training_cells_at_node",
        ]
    ]

    pair = pair.merge(
        supports,
        on=["analysis", "Condition_ID", "SoC"],
        validate="one_to_one",
    )

    pair["isotonic_minus_raw_mV"] = (
        1000*(pair.condition_iso-pair.condition_raw)
    )

    csv_save(
        pair,
        out,
        "isotonic_node_adjustments.csv",
    )

    (Path(out) / "LUT_reference_check.json").write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    return qa


# ============================================================
# ÇIKTI AÇIKLAMASI
# ============================================================

README = """
ADIM 5: KOŞUL BİLGİSİ VE İZOTONİK DÜZELTME KARŞILAŞTIRMASI

MODELLER
global_iso: Tek ortak izotonik eğri.
temperature_iso: Yalnız sıcaklığa bağlı izotonik eğriler.
rate_iso: Yalnız deşarj hızına bağlı izotonik eğriler.
condition_raw: Koşula özgü hücre medyanı; izotonik uygulanmaz.
condition_iso: Koşula özgü destek-ağırlıklı izotonik LUT.

Her eğitim satırı bir fiziksel hücrenin medyan profilidir.
Azaltılmış modeller uygun eğitim hücrelerini doğrudan havuzlar.
Koşullara eşit ağırlık uygulanmaz; her fiziksel hücre bir satırdır.
İlk üç model ve condition_iso aynı izotonik kurallarını kullanır.
condition_raw aynı eğitim hücrelerinin düğüm medyanını kullanır.
Ham/izotonik modellerinde minimum 2 destekli fit düğümü şartı aynıdır.

FİT VE DOĞRULAMA
Fit: 0..100. Değerlendirme: 0..90 dahil.
Düğümde en az bir eğitim hücresi gerekir; sıfır destek NaN kalır.
LOCO bütün eğitim havuzundan test hücresini çıkarır.
Faktör tutma bütün tutulan düzeyi bütün modellerin eğitiminden çıkarır.
25 ve 45 derece sıcaklık tutmaları ayrı deneylerdir.
Tutulan faktörü kullanan modeller iki komşu eğriyi doğrusal birleştirir.
Tutulan faktörü kullanmayan modeller kalan eğitimde kendi eğrisini kurar.
Komşulardan biri destek vermiyorsa taşıma veya ekstrapolasyon yapılmaz.

ANA SONUÇ
comparison=common_five, band=all_0_90_inclusive.
Beş model aynı hücre–SoC düğümlerinde puanlanır.
Her hücre/bant için en az 5 sonlu hedef–tahmin çifti gerekir.
own_support her yöntemin kendi kapsamını gösterir.
pairwise_common iki yöntemin ortak kapsamını gösterir.
Farklı maskelerdeki MAE'ler doğrudan iyileşme kanıtı sayılmaz.
MAE, değerlendirilebilir hücre MAE'lerinin aritmetik ortalamasıdır.
mean_cell_RMSE, hücre RMSE'lerinin ortalamasıdır; pooled RMSE değildir.
Hedef hücre medyan gerilim profilidir.
Bireysel çevrim veya çevrimiçi SoC kestirim hatası değildir.
# %80 bant kapsam filtresi LUT girdilerine ayrıca uygulanmamıştır.
Yardımcı hata bantları: [0,20), [20,60), [60,90].
Son hata bandı 90'ı içerir; kontrast analizindeki [60,90) farklıdır.

KAZANÇ YÖNÜ
gain_right_mV = MAE(left_model) - MAE(right_model).
Pozitif değer sağdaki yöntemin daha küçük MAE verdiğini gösterir.
condition_raw -> condition_iso karşılaştırması izotonik katkıdır.
condition_raw kaynak ham gerilim kayıtlarını ifade etmez.

BOOTSTRAP
2000 koşul-tabakalı, eşleştirilmiş fiziksel hücre örneklemesi.
Modeller bootstrap içinde yeniden fit edilmez.
Aralıklar mevcut out-of-fold tahminlere koşulludur.
Paylaşılan eğitim kümelerinden kaynaklanan bağımlılığı veya tüm eğitim
belirsizliğini kapsadıkları iddia edilmez.
Tek hücreli tabaka ek değişkenlik üretmez.
degenerate_bootstrap=True, bootstrap farklarının değişkenliğinin
yok/çok az olduğunu gösterir; sıfır genişlikli CI genel kesinlik kanıtı değildir.
Aralıklar noktasaldır; çoklu karşılaştırma kontrolü uygulanmaz.
Bootstrap işaret oranı p-değeri değildir.
Faktör tutmada koşullar sabit tutulur.
Aralıklar yeni faktör düzeylerine evrensel aktarımı veya iki sıcaklık
katının popülasyon belirsizliğini tahmin etmez.
Değerlendirilemeyen hücrelere sıfır hata atanmaz.

REFERANS VE KAPSAM
Önceki unified düğüm tahminleri, hedefleri, bütün bant özetleri ve nihai
LUT tolerans içinde yeniden üretilmeden karşılaştırma tamamlanmaz.
Girdi SHA256 kontrolü zorunludur.
Upstream çevrim yeniden kurma ve akım hedefi seçimi yeniden çalıştırılmaz.
Yalnız profil düzeyindeki eğitim/test ayrımı denetlenir.
Tüm upstream işlemlerin training-only olduğu iddia edilmez.
Nihai LUT tam geçmişten üretilir; 108'e yaş standardizasyonu yapılmaz.
SoC, çevrim sonu kapasitesine göre normalize edilmiş çevrimdışı koordinattır.

ANA DOSYALAR
MAIN_common_five.csv
MAIN_gains_vs_condition_iso.csv
summary.csv
summary_by_condition.csv
cell_metrics.csv
paired_cell_MAE_differences.csv
paired_bootstrap_summary.csv
bootstrap_gain_draws_mV.npz
node_predictions.csv.gz
node_exclusion_reasons.csv
training_fit_audit.csv
reference_node_check.csv
reference_summary_check.csv
LUT_reference_check.json
isotonic_node_adjustments.csv
"""


# ============================================================
# ANALİZİ ÇALIŞTIR
# ============================================================

def run(uploaded, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    require(
        int(BOOTSTRAP_REPS) == BOOTSTRAP_REPS
        and BOOTSTRAP_REPS >= 2000,
        "Bootstrap tekrar sayısı en az 2000 olmalı.",
    )

    (
        datasets,
        old_nodes,
        old_summary,
        old_lut,
        old_support,
        hashes,
        provenance,
        reference_hash,
    ) = load_inputs(uploaded)

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "models": list(MODELS),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": SEED,
        "min_eval_nodes": MIN_EVAL_POINTS,
        "input_sha256": hashes,
        "input_source": provenance,
        "reference_zip_sha256": reference_hash,
        "pool_weight": "one physical-cell median profile per row",
        "main_mask": "common_five",
        "fit_soc": [0, 100],
        "evaluate_soc": [0, 90],
        "bootstrap": (
            "condition-stratified paired cells; "
            "fixed out-of-fold errors; pointwise CI"
        ),
    }

    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "README.txt").write_text(
        README,
        encoding="utf-8",
    )

    csv_save(META, out, "condition_map.csv")

    inventory = []
    all_nodes = []
    all_audits = []

    for name, d in datasets.items():
        print(
            f"\n{name}: {d.sample_int.nunique()} hücre, "
            f"{d.Condition_ID.nunique()} koşul.",
            flush=True,
        )

        for cid in range(1, 25):
            g = d[d.Condition_ID.eq(cid)]

            inventory.append({
                "analysis": name,
                "Condition_ID": cid,
                "n_cells": g.sample_int.nunique(),
                "n_finite_nodes_0_90": int(
                    np.isfinite(
                        g.loc[g.SoC <= 90, "Voltage"]
                    ).sum()
                ),
            })

        nodes, audit = predict_dataset(d, name)
        all_nodes.append(nodes)
        all_audits.append(audit)

    nodes = pd.concat(all_nodes, ignore_index=True)

    csv_save(
        pd.DataFrame(inventory),
        out,
        "input_inventory.csv",
    )

    check_reference(nodes, old_nodes, out)

    print(
        "Hücre hata ve kapsam tabloları hesaplanıyor...",
        flush=True,
    )

    metrics, pairs = compute_metrics(nodes)
    summary = aggregate(metrics)

    check_summary(summary, old_summary, out)

    qa = export_final(
        datasets,
        old_lut,
        old_support,
        out,
    )

    csv_save(
        nodes,
        out,
        "node_predictions.csv.gz",
    )

    csv_save(
        pd.concat(all_audits, ignore_index=True),
        out,
        "training_fit_audit.csv",
    )

    csv_save(metrics, out, "cell_metrics.csv")
    csv_save(summary, out, "summary.csv")

    csv_save(
        aggregate(metrics, condition=True),
        out,
        "summary_by_condition.csv",
    )

    csv_save(
        pairs,
        out,
        "paired_cell_MAE_differences.csv",
    )

    exclusions = []

    for model in MODELS:
        reason_column = model + "_reason"

        g = (
            nodes.groupby([
                "analysis",
                "validation",
                "Condition_ID",
                reason_column,
            ])
            .size()
            .rename("n_nodes")
            .reset_index()
            .rename(columns={reason_column: "reason"})
        )

        g["model"] = model
        exclusions.append(g)

    csv_save(
        pd.concat(exclusions, ignore_index=True),
        out,
        "node_exclusion_reasons.csv",
    )

    bootstrap = bootstrap_pairs(
        pairs,
        out,
        reps=BOOTSTRAP_REPS,
    )

    main_summary = summary[
        summary.comparison.eq("common_five")
        & summary.band.eq("all_0_90_inclusive")
    ]

    csv_save(
        main_summary,
        out,
        "MAIN_common_five.csv",
    )

    main_gains = bootstrap[
        bootstrap.comparison.eq("common_five")
        & bootstrap.left_model.isin([
            "global_iso",
            "temperature_iso",
            "rate_iso",
            "condition_raw",
        ])
        & bootstrap.right_model.eq("condition_iso")
    ]

    csv_save(
        main_gains,
        out,
        "MAIN_gains_vs_condition_iso.csv",
    )

    print("\nANA KARŞILAŞTIRMA / AYNI HÜCRE VE DÜĞÜMLER:")

    print(
        main_summary[[
            "analysis",
            "validation",
            "model",
            "n_evaluated_cells",
            "n_used",
            "target_coverage_pct",
            "MAE_mV",
            "mean_cell_RMSE_mV",
        ]].to_string(index=False)
    )

    print(
        "\nCONDITION_ISO KAZANCI:"
        "\nPozitif değer = condition_iso daha düşük MAE."
    )

    print(
        main_gains[[
            "analysis",
            "validation",
            "left_model",
            "gain_right_mV",
            "CI95_low_mV",
            "CI95_high_mV",
            "n_evaluated_cells",
            "target_coverage_pct",
            "degenerate_bootstrap",
        ]].to_string(index=False)
    )

    print("\nNihai LUT referans kontrolü:", qa)

    return main_summary, main_gains


# ============================================================
# MANUEL YÜKLEME VE OTOMATİK ZIP İNDİRME
# ============================================================

def main():
    from google.colab import files

    print(
        "MANUEL YÜKLEME:\n"
        "1. Önceki SoC_unified_....zip dosyasını seçin (zorunlu).\n"
        "2. İsterseniz aynı anda iki CSV'yi de seçebilirsiniz:\n"
        "   cell_profiles_nominal_concordant.csv\n"
        "   broad_52_132_cell_profiles.csv\n\n"
        "CSV seçmezseniz ZIP içindeki girdiler kullanılacak.\n"
        "Bant/yaş duyarlılığı ZIP dosyalarını seçmeyin."
    )

    uploaded = files.upload()

    require(
        bool(uploaded),
        "Dosya seçilmedi.",
    )

    out = Path(
        tempfile.mkdtemp(prefix="SoC_step5_ablation_")
    )

    try:
        run(uploaded, out)

    except Exception as exc:
        (out / "ERROR.txt").write_text(
            type(exc).__name__ + ": " + str(exc),
            encoding="utf-8",
        )

        print(
            "\nÇalışma durduruldu."
            "\nKısmi sonuçlar nihai değildir."
            "\nTanı klasörü:",
            out,
        )

        raise

    archive = shutil.make_archive(
        str(out),
        "zip",
        root_dir=out,
    )

    print("\nTAMAMLANDI. Çıktı:", archive)
    print(
        "Önce bakılacak dosyalar:\n"
        "  MAIN_common_five.csv\n"
        "  MAIN_gains_vs_condition_iso.csv\n"
        "  README.txt"
    )

    files.download(archive)


# Incomplete exported guard omitted.

# ---------- LOCAL, NONDESTRUCTIVE STAGE RUNNER ----------
path=require_path('/content/SoC_unified_ConditionvCodes2.zip')
out=stage_dir('05_five_representations')
run({path.name:path.read_bytes()},out)
import pandas as pd
for name in ['MAIN_common_five.csv','MAIN_gains_vs_condition_iso.csv']:
    d=pd.read_csv(out/name)
    print(name,len(d),'rows')
stage_zip('05_five_representations','SoC_step5_ablation_')
