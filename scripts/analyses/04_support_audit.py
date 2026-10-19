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

# SOURCE-ALIGNED SCIENTIFIC BLOCK: 04 — Fixed-prediction support audit
# Scientific block SHA256: 66144840a99cdd376676ee0a87d3181282ea9520d1fdbf7b439a3d5577536120
# Original notebook cell 10 copied byte-for-byte below.

# GOOGLE COLAB: Tamamini TEK hucreye yapistirin ve calistirin.
# Manuel girdi: onceki SoC_unified_....zip sonuc dosyasi.
# Sabit tahminler uzerinde kapsam denetimi; model YENIDEN EGITILMEZ.
import io, json, hashlib, tempfile, shutil, platform
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

CELL_KEYS = ['analysis', 'validation', 'Condition_ID', 'sample_int']
BANDS = {'all_0_90_inclusive': (0, 90), '[0,20)': (0, 19),
         '[20,60)': (20, 59), '[60,90]': (60, 90)}
THRESHOLDS = (1, 2, 3)
MIN_POINTS = 5
SCOPES = ('all', 'target_C5', 'without_target_C5',
          'C5_involved', 'without_C5_involvement')


def read_csv(z, name):
    # Arsiv klasor icerse de calisir; belirsiz ayni adlari reddeder.
    hits = [p for p in z.namelist() if p == name or p.endswith('/'+name)]
    if len(hits) != 1:
        raise ValueError(f'ZIP icinde tek bir {name} bekleniyor; bulunan: {hits}')
    return pd.read_csv(z.open(hits[0]))


def load_nodes(z):
    d = read_csv(z, 'validation_node_predictions.csv')
    required = CELL_KEYS + ['method', 'SoC', 'target_V', 'prediction_V',
        'support_a', 'support_b', 'train_condition_a', 'train_condition_b',
        'Temperature', 'Rate', 'Charge_Rate']
    if not set(required).issubset(d):
        raise ValueError(f'Dugum dosyasinda eksik sutunlar: {set(required)-set(d)}')
    d = d[d.method.eq('unified')].copy()
    if d.empty or set(d.analysis) != {'full_history', 'broad_52_132'}:
        raise ValueError('Iki analiz kapsaminda unified sonuclari bulunamadi.')
    numeric = ['Condition_ID', 'sample_int', 'SoC', 'target_V', 'prediction_V',
               'support_a', 'support_b', 'train_condition_a', 'train_condition_b']
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if d.duplicated(CELL_KEYS+['SoC']).any():
        raise ValueError('Yinelenen hucre-deney-SoC kaydi var.')
    if np.isinf(d[['target_V', 'prediction_V']].to_numpy()).any():
        raise ValueError('Sonsuz hedef/tahmin degeri var.')
    for key, g in d.groupby(CELL_KEYS):
        if set(g.SoC) != set(range(91)) or len(g) != 91:
            raise ValueError(f'{key}: 0..90 dugum kayitlari eksik.')
    factor = ~d.validation.eq('LOCO')
    if not set(d.validation).issubset({'LOCO', 'rate_holdout', 'temperature_holdout'}):
        raise ValueError('Bilinmeyen dogrulama turu.')
    for a in d.analysis.unique():
        if set(d.loc[d.analysis.eq(a), 'validation']) != {'LOCO','rate_holdout','temperature_holdout'}:
            raise ValueError(f'{a}: eksik dogrulama deneyi.')
    for col, mask in [('support_a', np.ones(len(d), dtype=bool)), ('support_b', factor)]:
        v=d.loc[mask,col].to_numpy(float)
        if not np.isfinite(v).all() or (v<0).any() or not np.allclose(v,np.round(v)):
            raise ValueError(f'{col}: gecersiz egitim destek sayisi.')
    d['effective_support'] = d.support_a
    d.loc[factor, 'effective_support'] = np.minimum(d.loc[factor,'support_a'], d.loc[factor,'support_b'])
    if (d.prediction_V.notna() & d.effective_support.eq(0)).any():
        raise ValueError('Sifir destekli dugumde tahmin var: ortak yontem kurali ihlal edilmis.')
    d['C5_involved'] = d.Condition_ID.eq(5) | d.train_condition_a.eq(5) | d.train_condition_b.eq(5)
    return d


def cell_metrics(d):
    rows=[]
    for key, g in d.groupby(CELL_KEYS, sort=True):
        for band, (lo, hi) in BANDS.items():
            q=g[g.SoC.between(lo,hi)]
            target=np.isfinite(q.target_V.to_numpy())
            pred=np.isfinite(q.prediction_V.to_numpy())
            paired=target & pred
            for threshold in THRESHOLDS:
                selected=paired & (q.effective_support.to_numpy() >= threshold)
                count=int(selected.sum())
                eligible=count >= MIN_POINTS
                err=(q.prediction_V.to_numpy()-q.target_V.to_numpy())[selected]*1000
                r=dict(zip(CELL_KEYS,key))
                r.update(band=band, support_threshold=threshold,
                    Temperature=g.Temperature.iloc[0], Rate=g.Rate.iloc[0],
                    Charge_Rate=g.Charge_Rate.iloc[0], C5_involved=bool(g.C5_involved.iloc[0]),
                    n_grid=len(q), n_target=int(target.sum()), n_missing_target=int((~target).sum()),
                    n_no_prediction=int((target & ~pred).sum()), n_paired=int(paired.sum()),
                    n_filtered_by_threshold=int((paired & ~selected).sum()), n_selected=count,
                    n_discarded_by_min_points=count if not eligible else 0,
                    n_used=count if eligible else 0, evaluated=eligible,
                    status='evaluated' if eligible else 'no_target' if not target.any() else
                           'no_prediction' if not paired.any() else
                           'no_nodes_at_threshold' if count==0 else 'fewer_than_5_nodes',
                    MAE_mV=float(np.mean(abs(err))) if eligible else np.nan,
                    RMSE_mV=float(np.sqrt(np.mean(err**2))) if eligible else np.nan,
                    bias_mV=float(np.mean(err)) if eligible else np.nan,
                    max_abs_mV=float(np.max(abs(err))) if eligible else np.nan)
                assert r['n_grid']==r['n_missing_target']+r['n_no_prediction']+r['n_filtered_by_threshold']+r['n_discarded_by_min_points']+r['n_used']
                rows.append(r)
    return pd.DataFrame(rows)


def scope_filter(m,scope):
    if scope=='all': return m
    if scope=='target_C5': return m[m.Condition_ID.eq(5)]
    if scope=='without_target_C5': return m[~m.Condition_ID.eq(5)]
    if scope=='C5_involved': return m[m.C5_involved]
    return m[~m.C5_involved]


def aggregate(m, by_condition=False, scopes=('all',)):
    rows=[]
    keys=['analysis','validation','band','support_threshold']
    if by_condition: keys+=['Condition_ID','Temperature','Rate','Charge_Rate']
    for scope in scopes:
        for key,g in scope_filter(m,scope).groupby(keys,sort=True):
            e=g[g.evaluated]
            r=dict(zip(keys,key));r['scope']=scope
            r.update(n_planned_cells=len(g),n_evaluated_cells=len(e),n_skipped_cells=len(g)-len(e))
            for c in ('n_grid','n_target','n_missing_target','n_no_prediction','n_paired',
                      'n_filtered_by_threshold','n_selected','n_discarded_by_min_points','n_used'):
                r[c]=int(g[c].sum())
            r.update(n_unscored_target=r['n_target']-r['n_used'],
                target_coverage_pct=100*r['n_used']/r['n_target'] if r['n_target'] else np.nan,
                grid_coverage_pct=100*r['n_used']/r['n_grid'],
                evaluated_cell_pct=100*len(e)/len(g),
                selected_nodes_min=int(g.n_selected.min()),
                selected_nodes_median=float(g.n_selected.median()),
                selected_nodes_max=int(g.n_selected.max()),
                evaluated_nodes_min=int(e.n_used.min()) if len(e) else np.nan,
                evaluated_nodes_median=float(e.n_used.median()) if len(e) else np.nan,
                evaluated_nodes_max=int(e.n_used.max()) if len(e) else np.nan,
                MAE_mV=e.MAE_mV.mean(), median_cell_MAE_mV=e.MAE_mV.median(),
                P90_cell_MAE_mV=e.MAE_mV.quantile(.9), mean_cell_RMSE_mV=e.RMSE_mV.mean(),
                bias_mV=e.bias_mV.mean(), worst_abs_mV=e.max_abs_mV.max())
            rows.append(r)
    return pd.DataFrame(rows)


def condition5_maps(z,d,out):
    allmaps=[]
    luts=read_csv(z,'LUT_all_0_100.csv')
    for analysis,filename in [('full_history','cell_profiles_nominal_concordant.csv'),
                              ('broad_52_132','broad_52_132_cell_profiles.csv')]:
        p=read_csv(z,'inputs/'+filename)
        vc='V_cell_median' if 'V_cell_median' in p else 'Voltage'
        p=p[p.Condition_ID.eq(5)]
        A=p.pivot(index='sample_int',columns='SoC',values=vc).reindex(columns=range(91)).sort_index()
        support=np.isfinite(A.to_numpy()).sum(axis=0)
        lut=luts[(luts.analysis==analysis)&(luts.Condition_ID==5)&(luts.SoC<=90)].sort_values('SoC')
        if len(lut)!=91 or not np.array_equal(support,lut.n_cells_at_soc.to_numpy()):
            raise ValueError(f'{analysis}: profil destegi ile LUT destegi uyusmuyor.')
        q=d[(d.analysis==analysis)&(d.validation=='LOCO')&(d.Condition_ID==5)].copy()
        q['all_cell_support']=q.SoC.map(dict(enumerate(support)))
        expected=q.all_cell_support - q.target_V.notna().astype(int)
        if not np.array_equal(expected.to_numpy(),q.support_a.to_numpy()):
            raise ValueError(f'{analysis}: LOCO egitim destegi beklenenden farkli.')
        q['evaluated_node']=q.target_V.notna() & q.prediction_V.notna()
        q['node_state']=np.where(q.target_V.isna(),0,np.where(q.prediction_V.isna(),1,2))
        allmaps.append(q)
        S=q.pivot(index='sample_int',columns='SoC',values='support_a').reindex(index=A.index,columns=range(91))
        C=q.pivot(index='sample_int',columns='SoC',values='node_state').reindex(index=A.index,columns=range(91))
        fig,axs=plt.subplots(3,1,figsize=(11,7.5),sharex=True,
                             gridspec_kw={'height_ratios':[1,1.5,1.5]},layout='constrained')
        axs[0].step(range(91),support,where='mid',color='#244e75')
        axs[0].set(ylabel='All-cell support',ylim=(-.3,len(A)+.5),
                   title=f'Condition 5: 10°C, C/5, 2C | {analysis}')
        im=axs[1].imshow(S,aspect='auto',interpolation='nearest',vmin=0,vmax=max(1,len(A)-1),cmap='viridis')
        fig.colorbar(im,ax=axs[1],label='Training cells',pad=.015)
        axs[1].set(ylabel='Held-out cell',title='Training support after excluding the held-out cell')
        cmap=ListedColormap(['#dadada','#d3624f','#3b8a78'])
        axs[2].imshow(C,aspect='auto',interpolation='nearest',cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
        axs[2].set(ylabel='Held-out cell',xlabel='SoC (%)',title='Target availability and LOCO evaluation')
        for ax in axs[1:]:
            ax.set_yticks(range(len(A)),[str(int(x)) for x in A.index])
        axs[2].legend(handles=[Patch(color=c,label=l) for c,l in
            zip(cmap.colors,['Missing target','Target present; no prediction','Evaluable pair'])],
            loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,frameon=False)
        axs[2].set_xticks(range(0,91,10));axs[2].set_xlim(-.5,90.5)
        fig.savefig(out/f'Condition5_support_{analysis}.png',dpi=250)
        plt.close(fig)
    pd.concat(allmaps).to_csv(out/'Condition5_node_support_audit.csv',index=False)


def run_audit(zip_bytes,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with ZipFile(io.BytesIO(zip_bytes)) as z:
        d=load_nodes(z)
        m=cell_metrics(d)
        s=aggregate(m,scopes=SCOPES)
        c=aggregate(m,by_condition=True)
        base=s[(s.scope=='all')&(s.support_threshold==1)]
        original=read_csv(z,'validation_summary.csv')
        original=original[(original.method=='unified')&(original.comparison=='own_support')]
        check=base.merge(original,on=['analysis','validation','band'],suffixes=('_audit','_original'),validate='one_to_one')
        if len(check)!=len(base) or len(check)!=len(original):
            raise ValueError('Referans ozet ile denetim gruplari uyusmuyor.')
        for field in ('n_planned_cells','n_evaluated_cells','n_grid','n_target','n_paired','n_used','MAE_mV','bias_mV'):
            if not np.allclose(check[field+'_audit'],check[field+'_original'],rtol=0,atol=1e-8,equal_nan=True):
                raise ValueError(f'Referans sonucla uyusmazlik: {field}')
        m.to_csv(out/'cell_coverage_and_threshold_metrics.csv',index=False)
        base.to_csv(out/'primary_coverage_all_bands.csv',index=False)
        base[base.band=='all_0_90_inclusive'].to_csv(out/'primary_coverage_overall.csv',index=False)
        c.to_csv(out/'coverage_by_condition_and_threshold.csv',index=False)
        s.to_csv(out/'support_threshold_and_C5_sensitivity.csv',index=False)
        m.groupby(['analysis','validation','band','support_threshold','status']).size().rename('n_cells').reset_index().to_csv(out/'cell_exclusion_reasons.csv',index=False)
        d.to_csv(out/'audited_unified_nodes.csv',index=False)
        condition5_maps(z,d,out)
        # Pencereye girmeyen hucreleri, dogrulamada atlanan hucrelerden ayir.
        fp=read_csv(z,'inputs/cell_profiles_nominal_concordant.csv')
        bp=read_csv(z,'inputs/broad_52_132_cell_profiles.csv')
        pairs=['Condition_ID','sample_int']
        inventory=fp[pairs].drop_duplicates().merge(bp[pairs].drop_duplicates().assign(present_in_broad=True),on=pairs,how='left')
        inventory['present_in_broad']=inventory.present_in_broad.eq(True)
        inventory.to_csv(out/'cell_window_inventory.csv',index=False)
    config={'source_zip_sha256':hashlib.sha256(zip_bytes).hexdigest(),
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'pandas':pd.__version__,'numpy':np.__version__,'support_thresholds':list(THRESHOLDS),
        'minimum_paired_nodes':MIN_POINTS,'refitted':False,'reference_summary_check':'passed'}
    (out/'audit_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'README.txt').write_text(
        'KAPSAM DENETIMI: mevcut unified tahminleri sabittir; yeniden egitim yoktur.\n'
        'Her hata degeri hucre bazinda hesaplanir; ozet MAE hucrelere esit agirlik verir.\n'
        'Egitim destegi LOCO icin support_a, faktor tutmada min(support_a,support_b).\n'
        'Esik 1 ana yontem; 2 ve 3 yalnizca degerlendirme kapsam duyarliligidir.\n'
        'Esik altindaki dugumler fitten silinmez; bu bir yeniden egitim/alternatif LUT deneyi degildir.\n'
        'Her hucre/bant/esik icin en az 5 nokta gerekir. Daha azinda MAE NaN, asla sifir degildir.\n'
        'n_grid = n_missing_target + n_no_prediction + n_filtered_by_threshold +\n'
        'n_discarded_by_min_points + n_used.\n'
        'target_coverage_pct paydasi n_target; grid_coverage_pct paydasi n_grid.\n'
        'Planlanan hucreler ilgili giriste bulunanlardir. Pencereye girmeyenler inventory dosyasindadir.\n'
        'n_paired esik oncesi; n_selected esik sonrasi; n_used bes-nokta kurali sonrasi sayidir.\n'
        'selected_nodes_* tum planlanan hucreler; evaluated_nodes_* yalnizca puanlanan hucreler.\n'
        'all: tum hedef kosullar; target_C5: yalniz C5; without_target_C5: C5 hedefi haric.\n'
        'C5_involved: C5 hedef veya egitim komsusu; without_C5_involvement: bunlar haric.\n'
        'C5 hedefi faktor tutmada yoktur; ilgili bos scope gruplari satir olarak uretilmez.\n'
        'C5 haric ortalama farki farkli hucre karisimidir; iyilesme veya nedensel etki degildir.\n'
        'Esikler degistikce dugumler ve bazi hucreler degisebilir; hatayi kapsamla birlikte okuyun.\n'
        'Condition5 haritalari girdi profillerinden R11 tipi destek sayisini yeniden uretir;\n'
        'arsivlenmis eski R11 dosyasi okunmamistir. Destek sayilari LUT ve LOCO kayitlariyla dogrulanir.\n'
        'Harita yesil: sonlu hedef/tahmin cifti. Puanlanabilirlik cell metric tablosunda ayrica bulunur.\n'
        'Hata birimi mV. Toplam 0..90 dahil; son hata bandi [60,90] dahil.\n'
        'Bu hata bantlari cikarimsal [60,90) bant kapsamiyla karistirilmamalidir.\n'
        'Tahmin uretilmeyen dugumlerdeki hata bu analizden hesaplanamaz.\n',encoding='utf-8')
    main=base[base.band=='all_0_90_inclusive']
    cols=['analysis','validation','n_planned_cells','n_evaluated_cells','n_skipped_cells',
          'n_grid','n_missing_target','n_target','n_no_prediction','n_used','target_coverage_pct',
          'evaluated_nodes_min','evaluated_nodes_median','evaluated_nodes_max','MAE_mV']
    print('\nANA KAPSAM TABLOSU\n',main[cols].to_string(index=False))
    q=s[(s.scope=='all')&(s.band=='all_0_90_inclusive')]
    print('\nSABIT TAHMINLER / DESTEK ESIGI DUYARLILIGI\n',q[['analysis','validation','support_threshold','n_evaluated_cells','n_used','target_coverage_pct','MAE_mV']].to_string(index=False))
    print('\nKontroller gecti: referans hatalar, dugum sayimlari, C5 LUT ve LOCO destekleri.')
    return m,s,c


def main():
    from google.colab import files
    print('Onceki SoC_unified_....zip sonuc dosyasini secin (tek ZIP).')
    uploaded=files.upload()
    if len(uploaded)!=1:
        raise ValueError('Yalnizca bir sonuc ZIP dosyasi yukleyin.')
    name,blob=next(iter(uploaded.items()))
    # Colab adina (1), (2) eklese de dogrudan yuklenen baytlar kullanilir.
    if not name.lower().endswith('.zip'):
        raise ValueError('Girdi bir ZIP dosyasi olmali.')
    out=Path(tempfile.mkdtemp(prefix='SoC_coverage_audit_'))
    run_audit(blob,out)
    archive=shutil.make_archive(str(out),'zip',root_dir=out)
    print('\nTamamlandi:',archive)
    files.download(archive)


# Original auto-run omitted: the runner below supplies audited local inputs.

# ---------- LOCAL, NONDESTRUCTIVE STAGE RUNNER ----------
path=require_path('/content/SoC_unified_ConditionvCodes2.zip')
out=stage_dir('04_unified_coverage_audit')
run_audit(path.read_bytes(),out)
print('AUDIT OUTPUTS:',[p.name for p in sorted(out.iterdir())])
stage_zip('04_unified_coverage_audit','SoC_coverage_audit_')
