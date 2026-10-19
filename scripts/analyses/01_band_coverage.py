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

# SOURCE-ALIGNED SCIENTIFIC BLOCK: 01 — Coverage 80 and band inference
# Scientific block SHA256: 76cdd40e541612aac1ad6ac48dc99e88bc059c2cba7b5e6fa3b8c4298a4e3b05
# Original notebook cell 4 copied byte-for-byte below.

# GOOGLE COLAB: Tamamini TEK hucrede calistirin.
# Girdi: kendi R2.2 cikti dosyaniz global_cycle_profiles.pkl.
# Istege bagli: eski R7 kontrast CSV'si (delta_age_adjusted_V sutunlu).
import io, json, hashlib, tempfile, shutil, warnings, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

N_BOOT=2000
SEED=20260927
REFERENCE_CYCLE=108
RULES={'legacy_1node':None,'coverage_60':0.60,'coverage_80':0.80,'coverage_100':1.0}
BANDS=[('0-20',0,20),('20-60',20,60),('60-90',60,90)]
BOOT_RULES={'legacy_1node','coverage_80'}
MIN_BOOT_VALID_FRACTION=0.95
# Ayni R7 modeli ve pratik siniflar; yeni yas modeli bu blokta YOK.
META=pd.DataFrame([(1+6*t+2*r+k,T,c,rate)
    for t,T in enumerate((10,25,45,60))
    for r,rate in enumerate(('0.7C','1C','2C'))
    for k,c in enumerate(('C/5','C/40'))],
    columns=['Condition_ID','Temperature','Rate','Charge_Rate'])
LOOKUP={(r.Temperature,r.Rate,r.Charge_Rate):r.Condition_ID for r in META.itertuples()}


def specifications():
    rows=[]
    def add(f,i,l,r):
        rows.append(dict(contrast_family=f,contrast_id=i,lhs_condition=LOOKUP[l],rhs_condition=LOOKUP[r]))
    for T in (10,25,45,60):
        add('cutoff_C40_minus_C5',f'cutoff_T{T}_DR1C',(T,'C/40','1C'),(T,'C/5','1C'))
    for T in (10,25,45,60):
        add('rate_0.7C_minus_1C',f'rate07minus1_T{T}_C40',(T,'C/40','0.7C'),(T,'C/40','1C'))
    for dr in ('0.7C','1C','2C'):
        add('temperature_60_minus_25',f'temp60minus25_C40_DR{dr}',(60,'C/40',dr),(25,'C/40',dr))
    for family,prefix,lr,rr in [('rate_2C_minus_1C','rate2minus1','2C','1C'),
                                ('rate_0.7C_minus_2C','rate07minus2','0.7C','2C')]:
        for T in (10,25,45,60):
            for c in ('C/5','C/40'):
                add(family,f'{prefix}_T{T}_{c.replace("/","")}',(T,c,lr),(T,c,rr))
    assert len(rows)==27
    return rows
SPECS=specifications()


def sign(x):
    return 'NA' if not np.isfinite(x) else 'positive' if x>1e-12 else 'negative' if x< -1e-12 else 'zero'


def bucket(x):
    if not np.isfinite(x): return 'NA'
    return ('negligible' if abs(x)<.05 else 'small' if abs(x)<.10 else 'moderate' if abs(x)<.15 else 'large')


def load_cycles(blob):
    # Pickle kendi uretilmis R2.2 dosyaniz olmalidir.
    d=pd.read_pickle(io.BytesIO(blob))
    required=['sample_int','Condition_ID','Cycle_Index','matches_nominal']+[f'V_{i:03d}' for i in range(90)]
    if not isinstance(d,pd.DataFrame) or not set(required).issubset(d):
        raise ValueError('R2.2 global_cycle_profiles.pkl semasi bekleniyor.')
    before=len(d)
    if d.matches_nominal.isna().any() or not d.matches_nominal.isin([True,False]).all():
        raise ValueError('matches_nominal boolean olmali.')
    d=d[d.matches_nominal.eq(True)].copy()
    if d.empty: raise ValueError('Nominal uyumlu cevrim yok.')
    for c in required[:3]:
        d[c]=pd.to_numeric(d[c],errors='raise')
        a=d[c].to_numpy(float)
        if not np.isfinite(a).all() or not np.allclose(a,np.round(a),rtol=0,atol=1e-8):
            raise ValueError(f'{c}: sonlu tamsayi bekleniyor.')
        d[c]=np.round(a).astype(int)
    if not d.Condition_ID.between(1,24).all() or (d.Cycle_Index<0).any():
        raise ValueError('Kosul veya cevrim araligi hatali.')
    if d.duplicated(required[:3]).any(): raise ValueError('Yinelenen hucre-kosul-cevrim var; otomatik silinmedi.')
    if (d.groupby('sample_int').Condition_ID.nunique()>1).any(): raise ValueError('Hucre birden cok kosulda.')
    for r in d[['sample_int','Condition_ID']].drop_duplicates().itertuples():
        allowed=range(8*(r.Condition_ID-1)+1,8*r.Condition_ID+1) if r.Condition_ID<=22 else (range(177,180) if r.Condition_ID==23 else range(185,188))
        if r.sample_int not in allowed: raise ValueError(f'Nominal hucre haritasi uyusmazligi: {r}')
    for c in required[4:]:
        d[c]=pd.to_numeric(d[c],errors='raise')
        if np.isinf(d[c]).any(): raise ValueError(f'{c}: sonsuz gerilim var.')
    print(f'Girdi: {before:,}; nominal uyumlu: {len(d):,} cevrim; {d.sample_int.nunique()} hucre.')
    return d.reset_index(drop=True),before


def band_data(d,band,lo,hi):
    A=d[[f'V_{i:03d}' for i in range(lo,hi)]].to_numpy(float)
    finite=np.isfinite(A);n=finite.sum(axis=1)
    q=d[['sample_int','Condition_ID','Cycle_Index']].copy()
    q['SoC_band']=band;q['n_band_nodes']=hi-lo;q['n_valid']=n
    q['coverage_fraction']=n/(hi-lo)
    q['Voltage_band']=np.divide(np.nansum(A,axis=1),n,out=np.full(len(A),np.nan),where=n>0)
    q['age_c']=(q.Cycle_Index-REFERENCE_CYCLE)/100
    for rule,frac in RULES.items():
        q[rule]=n >= (1 if frac is None else int(np.ceil(frac*(hi-lo))))
    return q


def coverage_tables(q):
    rows=[];cellrows=[]
    for rule in RULES:
        for cid in range(1,25):
            a=q[q.Condition_ID.eq(cid)];b=a[a[rule]];old=a[a.legacy_1node]
            rows.append(dict(rule=rule,SoC_band=q.SoC_band.iloc[0],Condition_ID=cid,
                n_cycles_input=len(a),n_cycles_legacy=len(old),n_cycles_kept=len(b),
                n_cycles_removed_vs_legacy=len(old)-len(b),n_cycles_no_band_nodes=int(a.n_valid.eq(0).sum()),
                n_cells_input=a.sample_int.nunique(),n_cells_legacy=old.sample_int.nunique(),n_cells_kept=b.sample_int.nunique(),
                n_cells_lost_vs_legacy=old.sample_int.nunique()-b.sample_int.nunique(),
                cycle_min=b.Cycle_Index.min(),cycle_median=b.Cycle_Index.median(),cycle_max=b.Cycle_Index.max(),
                coverage_min=b.coverage_fraction.min(),coverage_median=b.coverage_fraction.median()))
        for sid,a in q.groupby('sample_int'):
            b=a[a[rule]]
            cellrows.append(dict(rule=rule,SoC_band=q.SoC_band.iloc[0],sample_int=sid,
                Condition_ID=int(a.Condition_ID.iloc[0]),n_cycles_input=len(a),
                n_cycles_legacy=int(a.legacy_1node.sum()),n_cycles_kept=len(b),
                n_cycles_removed_vs_legacy=int(a.legacy_1node.sum())-len(b),
                affected=bool((a.legacy_1node & ~a[rule]).any()),
                completely_lost=bool(a.legacy_1node.any() and b.empty),
                cycle_min=b.Cycle_Index.min(),cycle_max=b.Cycle_Index.max()))
    return rows,cellrows


def bootstrap_effects(d,X,L,seed):
    # Kosul icinde fiziksel hucreler yeniden orneklenir; secilen hucrenin tum uygun cevrimleri korunur.
    groups=list(d.groupby('sample_int',sort=True).indices.items())
    nc=len(groups);p=X.shape[1];Sxx=np.zeros((nc,p,p));Sxy=np.zeros((nc,p));cids=[]
    y=d.Voltage_band.to_numpy(float)
    for k,(sid,pos) in enumerate(groups):
        Z=X[pos];Sxx[k]=Z.T@Z;Sxy[k]=Z.T@y[pos];cids.append(int(d.iloc[pos[0]].Condition_ID))
    rng=np.random.default_rng(seed);counts=np.zeros((N_BOOT,nc))
    cids=np.array(cids)
    for cid in np.unique(cids):
        idx=np.flatnonzero(cids==cid);m=len(idx)
        counts[:,idx]=rng.multinomial(m,np.full(m,1/m),size=N_BOOT)
    xx=(counts@Sxx.reshape(nc,-1)).reshape(N_BOOT,p,p);xy=counts@Sxy
    effects=np.full((N_BOOT,len(SPECS)),np.nan);invalid=0
    for b in range(N_BOOT):
        # Sahte katsayi ureten singular pseudoinverse fallback YOK.
        if np.linalg.matrix_rank(xx[b])<p:
            invalid+=1;continue
        try: beta=np.linalg.solve(xx[b],xy[b])
        except np.linalg.LinAlgError:
            invalid+=1;continue
        effects[b]=L@beta
    return effects,invalid


def fit_band(q,rule,band_index):
    d=q[q[rule]].copy().reset_index(drop=True)
    present=sorted(d.Condition_ID.unique());idx={c:i for i,c in enumerate(present)}
    counts=d.groupby('Condition_ID').sample_int.nunique().to_dict()
    cyc=d.groupby('Condition_ID').size().to_dict()
    L=np.full((27,len(present)+2),np.nan)
    for j,s in enumerate(SPECS):
        if s['lhs_condition'] in idx and s['rhs_condition'] in idx:
            L[j]=0
            L[j,idx[s['lhs_condition']]]=1;L[j,idx[s['rhs_condition']]]=-1
    fitted=None;boot=None;invalid=0;status='empty_band';diagnostics={}
    if len(d):
        X=np.column_stack([(d.Condition_ID.to_numpy()[:,None]==present).astype(float),d.age_c.to_numpy(),d.age_c.to_numpy()**2])
        rank=np.linalg.matrix_rank(X);p=X.shape[1];n=len(d);g=d.sample_int.nunique()
        status='ok' if rank==p and n>p and g>=2 else 'rank_or_cluster_deficiency'
        diagnostics=dict(n_cycles=n,n_cells=g,n_conditions=len(present),rank=rank,n_parameters=p,condition_number=float(np.linalg.cond(X)))
        if status=='ok':
            fitted=sm.OLS(d.Voltage_band.to_numpy(),X).fit(cov_type='cluster',
                cov_kwds={'groups':d.sample_int.to_numpy(),'use_correction':True},use_t=False)
            diagnostics.update(age_linear=fitted.params[-2],age_quadratic=fitted.params[-1],r_squared=fitted.rsquared)
            if rule in BOOT_RULES:
                print(f'  {rule}, {q.SoC_band.iloc[0]}: {N_BOOT} bootstrap...',flush=True)
                boot,invalid=bootstrap_effects(d,X,L,SEED+band_index*100+(0 if rule=='legacy_1node' else 1))
    rows=[]
    for j,s in enumerate(SPECS):
        l=s['lhs_condition'];r=s['rhs_condition'];nl=counts.get(l,0);nr=counts.get(r,0)
        rstatus='missing_condition_arm' if min(nl,nr)==0 else status
        row=dict(rule=rule,SoC_band=q.SoC_band.iloc[0],**s,n_cells_lhs=nl,n_cells_rhs=nr,
                 n_cycles_lhs=cyc.get(l,0),n_cycles_rhs=cyc.get(r,0),status=rstatus,
                 sparse_arm_lt5=min(nl,nr)<5,inference_eligible=False,
                 delta_age_adjusted_V=np.nan,delta_unadjusted_V=np.nan,SE_V=np.nan,
                 CI95_low_V=np.nan,CI95_high_V=np.nan,p_value=np.nan,
                 bootstrap_status='not_requested' if rule not in BOOT_RULES else 'model_unavailable',
                 bootstrap_valid=0,bootstrap_CI95_low_V=np.nan,bootstrap_CI95_high_V=np.nan,
                 bootstrap_sign_stability=np.nan,bootstrap_bucket_stability=np.nan)
        if rstatus=='ok':
            effect=float(L[j]@fitted.params)
            row['delta_age_adjusted_V']=effect
            row['delta_unadjusted_V']=float(d.loc[d.Condition_ID.eq(l),'Voltage_band'].mean()-d.loc[d.Condition_ID.eq(r),'Voltage_band'].mean())
            # Tek hucreli kolun belirsizligi fiziksel hucre bootstrap ile ogrenilemez.
            row['inference_eligible']=min(nl,nr)>=2
            if min(nl,nr)>=2:
                test=fitted.t_test(L[j]);ci=np.asarray(test.conf_int()).ravel()
                row.update(SE_V=float(np.asarray(test.sd).item()),CI95_low_V=float(ci[0]),
                    CI95_high_V=float(ci[1]),p_value=float(np.asarray(test.pvalue).item()))
            if boot is not None:
                vals=boot[:,j];vals=vals[np.isfinite(vals)];row['bootstrap_valid']=len(vals)
                bstatus='single_cell_arm' if min(nl,nr)<2 else 'insufficient_valid_replicates' if len(vals)<np.ceil(N_BOOT*MIN_BOOT_VALID_FRACTION) else 'ok'
                row['bootstrap_status']=bstatus
                if bstatus=='ok':
                    low,high=np.quantile(vals,[.025,.975]);row.update(bootstrap_CI95_low_V=low,
                        bootstrap_CI95_high_V=high,bootstrap_sign_stability=np.mean([sign(v)==sign(effect) for v in vals]),
                        bootstrap_bucket_stability=np.mean([bucket(v)==bucket(effect) for v in vals]))
        row['point_sign']=sign(row['delta_age_adjusted_V']);row['point_bucket']=bucket(row['delta_age_adjusted_V'])
        rows.append(row)
    diagnostics.update(rule=rule,SoC_band=q.SoC_band.iloc[0],status=status,bootstrap_invalid=invalid)
    return rows,diagnostics,boot


def run(blob,out,old_csv=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d,before=load_cycles(blob)
    allq=[];coverage=[];cells=[];contrasts=[];diagnostics=[];draws={}
    with threadpool_limits(limits=1):
        for bi,(band,lo,hi) in enumerate(BANDS):
            q=band_data(d,band,lo,hi);allq.append(q)
            a,b=coverage_tables(q);coverage+=a;cells+=b
            for rule in RULES:
                print(f'Analiz: {band}, {rule}',flush=True)
                a,diag,boot=fit_band(q,rule,bi);contrasts+=a;diagnostics.append(diag)
                if boot is not None: draws[f'{rule}_band_{lo}_{hi}']=boot
    c=pd.DataFrame(contrasts)
    assert len(c)==4*81
    c['q_BH81']=np.nan
    # Her kurala ayri 81 planli test ailesi. Hesaplanamayan testler 1 ile yer tutar;
    # raporda p ve q degerleri NaN kalir; aile boyutu sessizce azaltilmaz.
    for rule in RULES:
        ii=c.index[c.rule.eq(rule)];v=c.loc[ii,'p_value'].to_numpy();ok=np.isfinite(v)
        adjusted=multipletests(np.where(ok,v,1.0),method='fdr_bh')[1]
        c.loc[ii,'q_BH81']=np.where(ok,adjusted,np.nan)
    for name in ('delta_age_adjusted_V','delta_unadjusted_V','SE_V','CI95_low_V','CI95_high_V','bootstrap_CI95_low_V','bootstrap_CI95_high_V'):
        c[name.replace('_V','_mV')]=1000*c[name]
    c['bootstrap_CI_excludes_zero']=pd.Series(pd.NA,index=c.index,dtype='boolean')
    bok=c.bootstrap_status.eq('ok')
    c.loc[bok,'bootstrap_CI_excludes_zero']=(c.loc[bok,'bootstrap_CI95_low_V'].gt(0)|c.loc[bok,'bootstrap_CI95_high_V'].lt(0)).to_numpy()
    baseline=c[c.rule.eq('legacy_1node')][['SoC_band','contrast_id','delta_age_adjusted_mV','point_sign','point_bucket','q_BH81','status','bootstrap_CI_excludes_zero']]
    compare=c.merge(baseline,on=['SoC_band','contrast_id'],suffixes=('','_legacy'),validate='many_to_one')
    compare['comparable']=np.isfinite(compare.delta_age_adjusted_mV)&np.isfinite(compare.delta_age_adjusted_mV_legacy)
    compare['change_mV']=compare.delta_age_adjusted_mV-compare.delta_age_adjusted_mV_legacy
    compare['absolute_magnitude_change_mV']=abs(compare.delta_age_adjusted_mV)-abs(compare.delta_age_adjusted_mV_legacy)
    for dest,col in [('sign_changed','point_sign'),('bucket_changed','point_bucket')]:
        compare[dest]=pd.Series(pd.NA,index=compare.index,dtype='boolean')
        compare.loc[compare.comparable,dest]=(compare.loc[compare.comparable,col]!=compare.loc[compare.comparable,col+'_legacy']).to_numpy()
    compare['FDR_decision_changed']=pd.Series(pd.NA,index=compare.index,dtype='boolean')
    valid=compare.q_BH81.notna()&compare.q_BH81_legacy.notna()
    compare.loc[valid,'FDR_decision_changed']=(compare.loc[valid,'q_BH81'].lt(.05)!=compare.loc[valid,'q_BH81_legacy'].lt(.05)).to_numpy()
    compare['bootstrap_CI_decision_changed']=pd.Series(pd.NA,index=compare.index,dtype='boolean')
    bv=compare.bootstrap_CI_excludes_zero.notna()&compare.bootstrap_CI_excludes_zero_legacy.notna()
    compare.loc[bv,'bootstrap_CI_decision_changed']=(compare.loc[bv,'bootstrap_CI_excludes_zero']!=compare.loc[bv,'bootstrap_CI_excludes_zero_legacy']).to_numpy()
    c.to_csv(out/'contrasts_all_rules.csv',index=False)
    compare.to_csv(out/'contrast_changes_vs_legacy.csv',index=False)
    compare[~compare.rule.eq('legacy_1node') & (~compare.comparable | compare.sign_changed.fillna(False) | compare.bucket_changed.fillna(False))].to_csv(out/'changed_or_unavailable_contrasts.csv',index=False)
    pd.DataFrame(coverage).merge(META,on='Condition_ID').to_csv(out/'coverage_by_condition_band_rule.csv',index=False)
    cell=pd.DataFrame(cells);cell.to_csv(out/'coverage_by_cell_band_rule.csv',index=False)
    cell.groupby(['rule','SoC_band']).agg(n_cycles_input=('n_cycles_input','sum'),n_cycles_legacy=('n_cycles_legacy','sum'),n_cycles_kept=('n_cycles_kept','sum'),n_cycles_removed=('n_cycles_removed_vs_legacy','sum'),n_cells_affected=('affected','sum'),n_cells_lost=('completely_lost','sum')).reset_index().to_csv(out/'coverage_summary.csv',index=False)
    q=pd.concat(allq,ignore_index=True);q.to_csv(out/'cycle_band_coverage_audit.csv.gz',index=False,compression='gzip')
    q[q.coverage_80].to_csv(out/'primary80_cycle_band_data.csv.gz',index=False,compression='gzip')
    pd.DataFrame(diagnostics).to_csv(out/'model_diagnostics.csv',index=False)
    pd.DataFrame(SPECS).to_csv(out/'contrast_specifications.csv',index=False)
    np.savez_compressed(out/'bootstrap_draws.npz',**draws)
    summary=compare.groupby('rule').agg(n_planned=('contrast_id','size'),n_comparable=('comparable','sum'),n_sign_changes=('sign_changed','sum'),n_bucket_changes=('bucket_changed','sum'),n_FDR_changes=('FDR_decision_changed','sum'),max_abs_change_mV=('change_mV',lambda x:abs(x).max())).reset_index()
    summary.to_csv(out/'contrast_change_summary.csv',index=False)
    if old_csv is not None:
        old=pd.read_csv(io.BytesIO(old_csv));keys=['SoC_band','contrast_id']
        if not set(keys+['delta_age_adjusted_V']).issubset(old) or old.duplicated(keys).any():
            raise ValueError('Opsiyonel eski R7 CSV semasi veya kontrast anahtarlari gecersiz.')
        check=c[c.rule.eq('legacy_1node')].merge(old[keys+['delta_age_adjusted_V']],on=keys,suffixes=('_new','_old'),how='outer',indicator=True)
        check['difference_mV']=(check.delta_age_adjusted_V_new-check.delta_age_adjusted_V_old)*1000
        check.to_csv(out/'optional_old_R7_reproduction_check.csv',index=False)
        print('Eski R7 nokta tahminleriyle en buyuk fark (mV):',check.difference_mV.abs().max())
    cfg=dict(source_sha256=hashlib.sha256(blob).hexdigest(),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        rules=RULES,bands=BANDS,bootstrap_replicates=N_BOOT,seed=SEED,reference_cycle=REFERENCE_CYCLE,
        input_rows=before,nominal_cycles=len(d),nominal_cells=d.sample_int.nunique(),
        model='Voltage_band ~ 0 + C(Condition_ID) + age_c + age_c^2',
        covariance='physical-cell cluster CR1, normal reference',BH_family_size=81,
        bootstrap_rules=sorted(BOOT_RULES),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,statsmodels=sm.__version__)
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'README.txt').write_text(
        'BANT KAPSAMI REVIZYONU / tam cevrim gecmisi.\n'
        'Bantlar [0,20), [20,60), [60,90): 20,40,30 dugum; 90 dahil degildir.\n'
        'Ana %80 esikleri 16/32/24; %60:12/24/18; %100:20/40/30; legacy:1/1/1.\n'
        'Eksik dugumler doldurulmaz. Ortalama sadece sonlu dugumlerdendir.\n'
        'Ana kural sonuclara bakilarak secilmez. Kapsam esigi tam bant ortalamasini garanti etmez.\n'
        'R7 ortak yas modeli korunur; 108 merkezleme noktasidir, kontrastlar ortak yas terimlerinden bagimsizdir.\n'
        'Hucresi eksik kosul modelden cikar; bu kola ait kontrast NA kalir.\n'
        'Tek hucreli kol varsa nokta tahmini betimseldir; p, CI ve bootstrap belirsizligi raporlanmaz.\n'
        '2-4 hucreli kollar sparse_arm_lt5 olarak isaretlenir; belirsizlik temkinli yorumlanmalidir.\n'
        'CR1 global hucre-kume kovaryansidir; hucre katkilarini dengelemez.\n'
        'Her esik icin 81 planli kontrasta ayri BH uygulanir. Hesaplanamayan testler aile hesabinda p=1\n'
        'yer tutar; kendi p/q alanlari NA kalir. Farkli esikler tek bir 324 test ailesi diye birlestirilmez.\n'
        'Bootstrap legacy ve %80 icin kosul icinde hucre ornekleme; tum uygun cevrimler birlikte tasinir.\n'
        'Her bant ayri orneklenir. Bootstrap yuzdelik CI noktasaldir, eszamanli/FDR duzeltilmis degildir.\n'
        'Rank eksik bootstrap tekrarlarinda pseudoinverse kullanilmaz; gecerli tekrar orani <%95 ise CI verilmez.\n'
        'Pratik siniflar mutlak etki: <50 negligible, 50-100 small, 100-150 moderate, >=150 large (mV).\n'
        'Degisimler kapsam secimi ve tum modelin yeniden uyumunu birlikte yansitir; nedensel etki degildir.\n'
        'Bootstrap draw sutun sirasi contrast_specifications.csv satir sirasidir; degerler V biriminde.\n'
        'primary80_cycle_band_data.csv.gz sonraki yaslanma duyarliligi icin hazir girdidir.\n'
        'Opsiyonel eski R7 dosyasi yoksa arsivlenmis R7 sonucunun aynen yeniden uretildigi iddia edilmez.\n'
        'Eski R8 dosyasi karsilastirilmaz; legacy bootstrap ayni tasarimin yeni tohumla tekraridir.\n',encoding='utf-8')
    print('\n81 KONTRAST / KURAL KARSILASTIRMASI\n',summary.to_string(index=False))
    print('\nAna %80: hesaplanamayan veya tek hucreli kollar\n',c[c.rule.eq('coverage_80')&(~c.status.eq('ok')|~c.inference_eligible)][['SoC_band','contrast_id','n_cells_lhs','n_cells_rhs','status']].to_string(index=False))
    return c


def main():
    from google.colab import files
    print('global_cycle_profiles.pkl dosyasini secin. Isterseniz eski R7 kontrast CSV dosyasini da ekleyin.')
    uploaded=files.upload()
    pkls=[(n,b) for n,b in uploaded.items() if n.lower().endswith('.pkl')]
    csvs=[(n,b) for n,b in uploaded.items() if n.lower().endswith('.csv')]
    if len(pkls)!=1 or len(csvs)>1 or len(uploaded)!=len(pkls)+len(csvs):
        raise ValueError('Bir PKL ve en fazla bir eski R7 CSV yukleyin.')
    out=Path(tempfile.mkdtemp(prefix='SoC_band_coverage_'))
    run(pkls[0][1],out,csvs[0][1] if csvs else None)
    archive=shutil.make_archive(str(out),'zip',root_dir=out)
    print('\nTamamlandi:',archive);files.download(archive)


# Original auto-run omitted: the runner below supplies audited local inputs.

# ---------- LOCAL, NONDESTRUCTIVE STAGE RUNNER ----------
source = require_path(INPUT_CYCLES)
out = stage_dir('01_band_coverage')
print('SOURCE R2.2:', source)
run(source.read_bytes(), out)
import pandas as pd
x = pd.read_csv(out/'coverage_summary.csv')
print('COVERAGE 80:', x[x['rule'].eq('coverage_80')].to_string(index=False))
y = pd.read_csv(out/'contrasts_all_rules.csv')
assert len(y[y.rule.eq('coverage_80')]) == 81, 'Expected 81 planned primary contrasts'
print('Primary contrasts:',len(y[y.rule.eq('coverage_80')]),'eligible:',int(y[y.rule.eq('coverage_80')]['inference_eligible'].sum()))
stage_zip('01_band_coverage','SoC_band_coverage_')
