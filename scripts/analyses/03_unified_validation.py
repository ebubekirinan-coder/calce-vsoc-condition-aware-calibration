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

# SOURCE-ALIGNED SCIENTIFIC BLOCK: 03 — Unified LOCO, rate and temperature hold-outs
# Scientific block SHA256: 81bb0a3960fbfa90aaffb95da88087df78e738960b78d750372433237645fd46
# Original notebook cell 8 copied byte-for-byte below.

# GOOGLE COLAB: Bu dosyanin TAMAMINI tek hucreye yapistirip calistirin.
# Girdiler: cell_profiles_nominal_concordant.csv
#           broad_52_132_cell_profiles.csv
# Processed_Outputv3 ham dosyalari bu asamanin girdisi DEGILDIR.
import io, json, hashlib, warnings, shutil, tempfile, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

GRID = np.arange(101)
MIN_EVAL_POINTS = 5
FILES = {
    'full_history': 'cell_profiles_nominal_concordant.csv',
    'broad_52_132': 'broad_52_132_cell_profiles.csv',
}
METHODS = ('unified', 'legacy_R5')
KEYS = ['analysis', 'validation', 'Condition_ID', 'sample_int']
META = pd.DataFrame([
    (1 + 6*t + 2*r + k, T, cutoff, rate)
    for t, T in enumerate((10, 25, 45, 60))
    for r, rate in enumerate((0.7, 1.0, 2.0))
    for k, cutoff in enumerate(('C/5', 'C/40'))
], columns=['Condition_ID', 'Temperature', 'Rate', 'Charge_Rate'])
LOOKUP = {(r.Temperature, r.Rate, r.Charge_Rate): r.Condition_ID
          for r in META.itertuples()}


def load_profiles(blob, name):
    d = pd.read_csv(io.BytesIO(blob))
    d.columns = d.columns.str.strip()
    if 'V_cell_median' in d:
        if 'Voltage' in d:
            raise ValueError(f'{name}: iki voltaj sutunu var; belirsiz girdi.')
        d = d.rename(columns={'V_cell_median': 'Voltage'})
    required = ['Condition_ID', 'sample_int', 'SoC', 'Voltage']
    if not set(required).issubset(d.columns):
        raise ValueError(f'{name}: gerekli sutunlar: {required}')
    for c in required:
        d[c] = pd.to_numeric(d[c], errors='raise')
    for c in required[:3]:
        a = d[c].to_numpy(float)
        if not np.isfinite(a).all() or not np.allclose(a, np.round(a), rtol=0, atol=1e-8):
            raise ValueError(f'{name}: {c} tam sayi ve sonlu olmali.')
        d[c] = np.round(a).astype(int)
    if np.isinf(d.Voltage).any():
        raise ValueError(f'{name}: sonsuz voltaj degeri var.')
    if not d.Condition_ID.between(1, 24).all() or not d.SoC.between(0, 100).all():
        raise ValueError(f'{name}: kosul veya SoC araligi hatali.')
    if d.duplicated(required[:3]).any():
        raise ValueError(f'{name}: yinelenen hucre-kosul-SoC; sessizce birlestirilmedi.')
    if (d.groupby('sample_int').Condition_ID.nunique() > 1).any():
        raise ValueError(f'{name}: bir fiziksel hucre birden cok kosulda.')
    for (cid, sid), g in d.groupby(['Condition_ID', 'sample_int']):
        allowed = range(8*(cid-1)+1, 8*cid+1) if cid <= 22 else (
            range(177, 180) if cid == 23 else range(185, 188))
        if sid not in allowed:
            raise ValueError(f'{name}: nominal harita uyusmazligi: kosul={cid}, hucre={sid}')
        if set(g.SoC) != set(GRID):
            raise ValueError(f'{name}: hucre {sid}: 0..100 satirlari eksik. '
                             'Kirpilmamis ara ciktiyi yukleyin; eksik voltajlar NaN olabilir.')
    if name == 'full_history' and set(d.Condition_ID) != set(range(1, 25)):
        raise ValueError('Tam gecmis dosyasinda 24 kosulun tamami bulunmali.')
    # Varsa faktor bilgileri resmi kosul haritasiyla karsilastirilir.
    for c in ('Temperature', 'Rate', 'Charge_Rate'):
        if c not in d:
            continue
        expected = d.Condition_ID.map(META.set_index('Condition_ID')[c])
        if c == 'Rate':
            actual = d[c].astype(str).str.replace(' ', '', regex=False).str.upper()
            ok = actual.eq(expected)
        else:
            actual = d[c].astype(str).str.replace('°', '', regex=False)
            actual = actual.str.replace('º', '', regex=False).str.replace('C', '', regex=False)
            actual = pd.to_numeric(actual.str.strip(), errors='coerce')
            ok = np.isclose(actual, expected, rtol=0, atol=1e-8)
        if not np.asarray(ok).all():
            raise ValueError(f'{name}: {c} metadata uyusmazligi.')
    return d[required].merge(META, on='Condition_ID', validate='many_to_one')


def build_lut(A, method='unified'):
    """A: benzersiz egitim hucreleri x 101 SoC dugumu. Test verisi almaz.
    unified: >=1 hucre/dugum, destek agirlikli, 0..100'de fit,
    >=2 destekli fit dugumu; sifir destekli cikti kesinlikle NaN.
    legacy_R5: >=2 hucre/dugum, agirliksiz, 0..89'da fit,
    >=5 fit dugumu, ic bosluklarda eski R5 interpolasyonu.
    """
    if method not in METHODS:
        raise ValueError(method)
    A = np.asarray(A, float).reshape(-1, 101)
    n = np.isfinite(A).sum(axis=0)
    raw = np.full(101, np.nan)
    if len(A):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            raw = np.nanmedian(A, axis=0)
    legacy = method == 'legacy_R5'
    fit = (n >= (2 if legacy else 1)) & (GRID < (90 if legacy else 101))
    pred = np.full(101, np.nan)
    status = 'ok'
    if len(A) < (2 if legacy else 1):
        status = 'insufficient_training_cells'
    elif fit.sum() < (5 if legacy else 2):
        status = 'insufficient_fit_nodes'
    else:
        model = IsotonicRegression(increasing=True, out_of_bounds='nan')
        model.fit(GRID[fit], raw[fit], sample_weight=None if legacy else n[fit])
        if legacy:
            pred[:90] = model.predict(GRID[:90])
        else:
            # predict(GRID) YOK: desteksiz dugumler doldurulmaz.
            pred[fit] = model.predict(GRID[fit])
    return {'pred': pred, 'support': n, 'raw': raw, 'status': status,
            'n_fit_nodes': int(fit.sum()), 'n_train_cells': len(A)}


def cell_arrays(d):
    return {cid: g.pivot(index='sample_int', columns='SoC', values='Voltage')
            .reindex(columns=GRID).sort_index() for cid, g in d.groupby('Condition_ID')}


def validate(d, analysis):
    groups = cell_arrays(d)
    empty = np.empty((0, 101))
    cache = {(m, cid): build_lut(groups[cid].to_numpy() if cid in groups else empty, m)
             for m in METHODS for cid in range(1, 25)}
    rows = []
    for cid, cells in groups.items():
        meta = META.set_index('Condition_ID').loc[cid]
        T, cutoff, rate = meta.Temperature, meta.Rate, meta.Charge_Rate
        experiments = [('LOCO', None, None, np.nan)]
        if T in (25, 45):
            low, high = ((10, 45) if T == 25 else (25, 60))
            experiments.append(('temperature_holdout', LOOKUP[low, cutoff, rate],
                                LOOKUP[high, cutoff, rate], (T-low)/(high-low)))
        if rate == 1.0:
            experiments.append(('rate_holdout', LOOKUP[T, cutoff, 0.7],
                                LOOKUP[T, cutoff, 2.0], (1.0-0.7)/(2.0-0.7)))
        for validation, lowcid, highcid, w in experiments:
            for sid, true in cells.iterrows():
                for method in METHODS:
                    if validation == 'LOCO':
                        a = build_lut(cells.drop(index=sid).to_numpy(), method)
                        b = None
                        pred = a['pred'].copy()
                    else:
                        a, b = cache[method, lowcid], cache[method, highcid]
                        # Bir taraf NaN ise sonuc NaN: tek tarafa tasima yok.
                        pred = (1-w)*a['pred'] + w*b['pred']
                    for soc in range(91):
                        y, p = float(true.loc[soc]), float(pred[soc])
                        if not np.isfinite(y):
                            reason = 'missing_test_target'
                        elif method == 'legacy_R5' and soc == 90:
                            reason = 'outside_legacy_domain'
                        elif np.isfinite(p):
                            reason = 'evaluable'
                        elif a['support'][soc] == 0 or (b is not None and b['support'][soc] == 0):
                            reason = 'zero_training_support'
                        elif a['status'] != 'ok' or (b is not None and b['status'] != 'ok'):
                            reason = 'insufficient_training_curve'
                        else:
                            reason = 'legacy_support_or_domain_limit'
                        rows.append(dict(analysis=analysis, validation=validation,
                            Condition_ID=cid, sample_int=int(sid), Temperature=T,
                            Rate=cutoff, Charge_Rate=rate, method=method, SoC=soc,
                            target_V=y, prediction_V=p, reason=reason,
                            train_condition_a=cid if lowcid is None else lowcid,
                            train_condition_b=highcid, weight_b=w,
                            support_a=int(a['support'][soc]),
                            support_b=int(b['support'][soc]) if b is not None else np.nan,
                            fit_status_a=a['status'], fit_status_b=b['status'] if b else '',
                            n_train_cells_a=a['n_train_cells'],
                            n_train_cells_b=b['n_train_cells'] if b else np.nan))
        print(f'{analysis}: kosul {cid}/24 tamamlandi.')
    return pd.DataFrame(rows)


def metrics(nodes, common=False):
    rows = []
    for key, g in nodes.groupby(KEYS + ['method'], sort=False):
        for band, lo, hi in [('all_0_90_inclusive', 0, 91),
                             ('[0,20)', 0, 20), ('[20,60)', 20, 60), ('[60,90]', 60, 91)]:
            q = g[g.SoC.between(lo, hi-1)]
            target = np.isfinite(q.target_V)
            paired = target & np.isfinite(q.prediction_V)
            if common:
                paired &= q.common_support
            n = int(paired.sum())
            eligible = n >= MIN_EVAL_POINTS
            err = (q.loc[paired, 'prediction_V'] - q.loc[paired, 'target_V']).to_numpy()*1000
            r = dict(zip(KEYS + ['method'], key))
            r.update(band=band, comparison='common_support' if common else 'own_support',
                Temperature=g.Temperature.iloc[0], Rate=g.Rate.iloc[0],
                Charge_Rate=g.Charge_Rate.iloc[0], n_grid=len(q), n_target=int(target.sum()),
                n_paired=n, n_used=n if eligible else 0, eligible=eligible,
                status='ok' if eligible else ('no_test_target' if not target.any() else
                       'no_prediction' if n == 0 else 'fewer_than_5_paired_nodes'),
                MAE_mV=float(np.mean(abs(err))) if eligible else np.nan,
                RMSE_mV=float(np.sqrt(np.mean(err**2))) if eligible else np.nan,
                bias_mV=float(np.mean(err)) if eligible else np.nan,
                max_abs_mV=float(np.max(abs(err))) if eligible else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(m, by_condition=False):
    keys = ['analysis', 'validation', 'method', 'comparison', 'band']
    if by_condition:
        keys += ['Condition_ID', 'Temperature', 'Rate', 'Charge_Rate']
    s = m.groupby(keys, dropna=False).agg(
        n_planned_cells=('sample_int', 'size'), n_evaluated_cells=('eligible', 'sum'),
        n_grid=('n_grid', 'sum'), n_target=('n_target', 'sum'),
        n_paired=('n_paired', 'sum'), n_used=('n_used', 'sum'),
        MAE_mV=('MAE_mV', 'mean'), mean_cell_RMSE_mV=('RMSE_mV', 'mean'),
        bias_mV=('bias_mV', 'mean'), worst_abs_mV=('max_abs_mV', 'max')).reset_index()
    s['paired_target_fraction'] = s.n_paired / s.n_target.replace(0, np.nan)
    s['used_target_fraction'] = s.n_used / s.n_target.replace(0, np.nan)
    s['evaluated_cell_fraction'] = s.n_evaluated_cells / s.n_planned_cells
    return s


def export_luts(datasets, out):
    rows = []
    for analysis, d in datasets.items():
        groups = cell_arrays(d)
        for cid in range(1, 25):
            fit = build_lut(groups[cid].to_numpy() if cid in groups else np.empty((0, 101)))
            for soc in GRID:
                rows.append(dict(analysis=analysis, Condition_ID=cid, SoC=int(soc),
                    V_raw_median=fit['raw'][soc], V_lut=fit['pred'][soc],
                    n_cells_at_soc=int(fit['support'][soc]), fit_status=fit['status']))
    lut = pd.DataFrame(rows).merge(META, on='Condition_ID')
    lut.to_csv(out/'LUT_all_0_100.csv', index=False)
    lut[lut.SoC <= 90].to_csv(out/'LUT_all_0_90_inclusive.csv', index=False)
    final = lut[(lut.analysis == 'full_history') & (lut.SoC <= 90)]
    arr = final.pivot(index='Condition_ID', columns='SoC', values='V_lut').to_numpy()
    support = final.pivot(index='Condition_ID', columns='SoC', values='n_cells_at_soc').to_numpy()
    np.save(out/'final_LUT_float64.npy', arr)
    np.save(out/'final_LUT_float32.npy', arr.astype(np.float32))
    np.save(out/'final_LUT_float16.npy', arr.astype(np.float16))
    np.save(out/'final_LUT_valid_mask.npy', np.isfinite(arr))
    np.save(out/'final_LUT_support.npy', support.astype(np.int16))
    finite = np.isfinite(arr)
    qa = {'shape': list(arr.shape), 'finite_nodes': int(finite.sum()),
          'NaN_nodes': int((~finite).sum()), 'float32_array_bytes': arr.astype('float32').nbytes,
          'float16_array_bytes': arr.astype('float16').nbytes}
    for typ in ('float32', 'float16'):
        qa[typ+'_max_quantization_mV'] = (float(np.max(abs(arr[finite].astype(typ)
                 .astype(float)-arr[finite]))*1000) if finite.any() else None)
    (out/'LUT_QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    return qa


def run(uploaded, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing = set(FILES.values()) - set(uploaded)
    if missing:
        raise ValueError(f'Eksik dosyalar: {sorted(missing)}')
    datasets = {a: load_profiles(uploaded[f], a) for a, f in FILES.items()}
    full_pairs = set(map(tuple, datasets['full_history'][['Condition_ID', 'sample_int']].values))
    broad_pairs = set(map(tuple, datasets['broad_52_132'][['Condition_ID', 'sample_int']].values))
    if not broad_pairs.issubset(full_pairs):
        raise ValueError('52-132 hucreleri tam gecmis dosyasinin alt kumesi olmali.')
    inventory = []
    for a, d in datasets.items():
        for cid in range(1, 25):
            g = d[d.Condition_ID == cid]
            inventory.append(dict(analysis=a, Condition_ID=cid, n_cells=g.sample_int.nunique(),
                                  n_finite_cell_nodes=int(np.isfinite(g.Voltage).sum())))
    pd.DataFrame(inventory).to_csv(out/'input_inventory.csv', index=False)
    nodes = pd.concat([validate(d, a) for a, d in datasets.items()], ignore_index=True)
    key = KEYS + ['SoC']
    common = nodes.assign(valid=np.isfinite(nodes.target_V) & np.isfinite(nodes.prediction_V))
    common = common.groupby(key).agg(common_support=('valid', 'all'), methods=('method', 'nunique'))
    assert common.methods.eq(2).all()
    nodes = nodes.merge(common[['common_support']], left_on=key, right_index=True, validate='many_to_one')
    nodes.to_csv(out/'validation_node_predictions.csv', index=False)
    m = pd.concat([metrics(nodes), metrics(nodes, common=True)], ignore_index=True)
    m.to_csv(out/'validation_cell_metrics.csv', index=False)
    summary = summarize(m)
    summary.to_csv(out/'validation_summary.csv', index=False)
    summarize(m, True).to_csv(out/'validation_summary_by_condition.csv', index=False)
    nodes.groupby(['analysis', 'validation', 'method', 'Condition_ID', 'reason']).size().rename(
        'n_nodes').reset_index().to_csv(out/'node_exclusion_reasons.csv', index=False)
    paired = m[m.comparison == 'common_support'].pivot(
        index=KEYS+['band'], columns='method', values=['MAE_mV', 'RMSE_mV', 'n_used'])
    paired.columns = ['_'.join(c) for c in paired.columns]
    paired['delta_MAE_unified_minus_legacy_mV'] = paired.MAE_mV_unified - paired.MAE_mV_legacy_R5
    paired.to_csv(out/'paired_method_comparison.csv')
    qa = export_luts(datasets, out)
    META.to_csv(out/'condition_map.csv', index=False)
    (out/'inputs').mkdir(exist_ok=True)
    for f in FILES.values():
        (out/'inputs'/f).write_bytes(uploaded[f])
    config = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, min_eval_points=MIN_EVAL_POINTS,
        fit_grid=[0, 100], output_grid=[0, 90], grid_step=1, min_support=1,
        minimum_supported_fit_nodes=2, zero_support='NaN; no filling',
        isotonic_weight='number of finite training cells',
        input_sha256={f: hashlib.sha256(uploaded[f]).hexdigest() for f in FILES.values()})
    (out/'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out/'README.txt').write_text(
        'Unified R5/R9 validation. Voltages in V; errors in mV.\n'
        'Inputs must be untrimmed 0..100 cell-median profiles from the existing audited pipeline.\n'
        'Rate = charge cutoff; Charge_Rate = discharge rate.\n'
        'The full-history LUT is the final artifact; the 52..132 LUT is a sensitivity artifact.\n'
        'All folds use training-only medians, support counts and weighted isotonic fits.\n'
        'Zero-support nodes remain NaN. Two supported fit nodes are required.\n'
        'Factor interpolation requires valid predictions from both bracketing conditions.\n'
        'A cell/band needs at least five paired nodes for a reported error.\n'
        'MAE and RMSE summaries are arithmetic means of evaluable cell-level metrics.\n'
        'Planned cells are those present in each input/window; absent conditions are in input_inventory.\n'
        'n_target: finite held-cell targets; n_paired: finite target/prediction pairs;\n'
        'n_used: paired nodes in cells/bands satisfying the five-node rule.\n'
        'own_support compares separate evaluation supports; common_support uses identical nodes.\n'
        'legacy_R5 reconstructs the reviewed R5 rules; it does not import archived results.\n'
        'Legacy excludes 90; unified includes 90. Thus common_support cannot include 90.\n'
        'Bands are [0,20), [20,60), [60,90]; the last includes 90 for unified.\n'
        'New masking also revises old R9 internal-gap interpolation.\n'
        'Validation uses float64 curves; float32/float16 conversion errors are reported separately.\n'
        'Final arrays: rows Condition_ID=1..24, columns SoC=0..90; NaNs must be respected.\n'
        'These are offline cycle-normalized SoC targets, not online SoC estimates.\n'
        'Upstream current-target selection/reconstruction is not rerun here; this is not\n'
        'evidence that all upstream preprocessing is training-only.\n', encoding='utf-8')
    print('\nYENI YONTEM / ANA SONUCLAR (mV):')
    print(summary[(summary.method == 'unified') & (summary.comparison == 'own_support') &
                  (summary.band == 'all_0_90_inclusive')].to_string(index=False))
    print('\nNihai LUT:', qa)
    return nodes, m, summary


def main():
    from google.colab import files
    print('Iki dosyayi birlikte secin:\n' + '\n'.join(FILES.values()))
    uploaded = files.upload()
    out = Path(tempfile.mkdtemp(prefix='SoC_unified_'))
    run(uploaded, out)
    archive = shutil.make_archive(str(out), 'zip', root_dir=out)
    print('\nTamamlandi. Sonuc ZIP dosyasi:', archive)
    files.download(archive)


# Original auto-run omitted: the runner below supplies audited local inputs.

# ---------- LOCAL, NONDESTRUCTIVE STAGE RUNNER ----------
source1=require_path(INPUT_CELLS)
source2=require_path(INPUT_BROAD)
out=stage_dir('03_unified')
print('SOURCE FULL:', source1)
print('SOURCE WINDOW:', source2)
run({source1.name:source1.read_bytes(), source2.name:source2.read_bytes()},out)
import pandas as pd
summary=pd.read_csv(out/'validation_summary.csv')
p=summary[(summary.method.eq('unified')) & (summary.comparison.eq('own_support')) & (summary.band.eq('all_0_90_inclusive'))]
print('UNIFIED 6 MAIN RESULTS:')
print(p.to_string(index=False))
stage_zip('03_unified','SoC_unified_')
