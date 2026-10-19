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

# SOURCE-ALIGNED SCIENTIFIC BLOCK: 02 — Four-model age sensitivity
# Scientific block SHA256: 99d38fe63027886e2d923ba49a87476c7da0f79a3d2ff9073609974feb351ef7
# Original notebook cell 6 copied byte-for-byte below.

# COLAB: Bu dosyanin tamamini TEK hucreye yapistirip calistirin.
# Manuel girdi: SoC_band_coverage_ts2_1sbp.zip (3. adimin sonuc ZIP'i).
# Ham veri gerekmez. Ana %80 kapsam kurali degistirilmez.
import io, json, hashlib, tempfile, shutil, platform
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

N_BOOT=2000
SEED=20260928
MIN_VALID_FRACTION=.95
REF=108
LOCAL_RADIUS=10
BANDS=('0-20','20-60','60-90')
MODELS=('reference','condition_slopes','cell_balanced','window_52_132')
TIMES=(52,108,132)
# condition_slopes icin destek kuralimiz:
# her kolda >=2 fiziksel hucrenin kendi cevrim min/max araligi sorgu cevrimini kapsar.
# +/-10 cevrimdeki hucre ve gozlem sayilari ayrica tani amaciyla raporlanir.


def read_member(z,name):
    hits=[x for x in z.namelist() if x==name or x.endswith('/'+name)]
    if len(hits)!=1: raise ValueError(f'Arsivde tek bir {name} gerekli: {hits}')
    return z.read(hits[0])


def read_inputs(blob):
    with ZipFile(io.BytesIO(blob)) as z:
        d=pd.read_csv(io.BytesIO(read_member(z,'primary80_cycle_band_data.csv.gz')),compression='gzip')
        specs=pd.read_csv(io.BytesIO(read_member(z,'contrast_specifications.csv')))
        previous=pd.read_csv(io.BytesIO(read_member(z,'contrasts_all_rules.csv')))
    cols=['sample_int','Condition_ID','Cycle_Index','SoC_band','Voltage_band','n_valid','n_band_nodes']
    if not set(cols).issubset(d): raise ValueError('primary80 verisinin semasi eksik.')
    for c in cols:
        if c!='SoC_band': d[c]=pd.to_numeric(d[c],errors='raise')
    if not np.isfinite(d[[c for c in cols if c!='SoC_band']].to_numpy()).all():
        raise ValueError('Ana %80 verisinde sonlu olmayan sayisal deger var.')
    for c in ['sample_int','Condition_ID','Cycle_Index','n_valid','n_band_nodes']:
        if not np.allclose(d[c],np.round(d[c]),rtol=0,atol=1e-8): raise ValueError(f'{c} tamsayi olmali.')
        d[c]=np.round(d[c]).astype(int)
    if set(d.SoC_band)!=set(BANDS) or not d.Condition_ID.between(1,24).all():
        raise ValueError('Bant/kosul degerleri gecersiz.')
    if d.duplicated(['sample_int','Condition_ID','Cycle_Index','SoC_band']).any():
        raise ValueError('Yinelenen cevrim-bant satiri var.')
    if (d.groupby('sample_int').Condition_ID.nunique()>1).any(): raise ValueError('Hucre birden cok kosulda.')
    expected=d.SoC_band.map({'0-20':20,'20-60':40,'60-90':30})
    if not (d.n_band_nodes.eq(expected)&d.n_valid.ge(np.ceil(.8*expected))&d.n_valid.le(expected)).all():
        raise ValueError('%80 kapsam kurali veya bant boyutu uyusmuyor.')
    needed=['contrast_id','contrast_family','lhs_condition','rhs_condition']
    if not set(needed).issubset(specs) or len(specs)!=27 or specs.contrast_id.duplicated().any():
        raise ValueError('27 benzersiz kontrast tanimi gerekli.')
    for col in ['lhs_condition','rhs_condition']:
        if not specs[col].isin(range(1,25)).all(): raise ValueError('Kontrast kosulu gecersiz.')
    previous=previous[previous.rule.eq('coverage_80')]
    if len(previous)!=81 or previous.duplicated(['SoC_band','contrast_id']).any():
        raise ValueError('3. adimin 81 adet %80 kontrasti gerekli.')
    d['age_c']=(d.Cycle_Index-REF)/100.0
    return d.reset_index(drop=True),specs,previous


def sign(v):
    return 'NA' if not np.isfinite(v) else 'positive' if v>1e-12 else 'negative' if v< -1e-12 else 'zero'


def bucket(v):
    return 'NA' if not np.isfinite(v) else 'negligible' if abs(v)<.05 else 'small' if abs(v)<.10 else 'moderate' if abs(v)<.15 else 'large'


def support(d,t):
    rows={}
    for cid in range(1,25):
        g=d[d.Condition_ID.eq(cid)]
        per=g.groupby('sample_int').Cycle_Index.agg(['min','max'])
        local=g[g.Cycle_Index.between(t-LOCAL_RADIUS,t+LOCAL_RADIUS)]
        rows[cid]=dict(n_cells=g.sample_int.nunique(),n_cycles=len(g),
            n_distinct_cycles=g.Cycle_Index.nunique(),cycle_min=g.Cycle_Index.min(),cycle_max=g.Cycle_Index.max(),
            n_cells_bracketing=int((per['min'].le(t)&per['max'].ge(t)).sum()),
            n_local_cells=local.sample_int.nunique(),n_local_cycles=len(local),
            inside_condition_range=bool(len(g) and g.Cycle_Index.min()<=t<=g.Cycle_Index.max()))
    return rows


def design(d,name):
    levels=sorted(d.Condition_ID.unique());k=len(levels)
    D=(d.Condition_ID.to_numpy()[:,None]==levels).astype(float)
    age=d.age_c.to_numpy()
    X=np.column_stack([D,D*age[:,None],age**2]) if name=='condition_slopes' else np.column_stack([D,age,age**2])
    w=(1/d.groupby('sample_int').sample_int.transform('size')).to_numpy() if name=='cell_balanced' else np.ones(len(d))
    return X,w,levels


def contrast_matrix(specs,levels,p,t,name):
    idx={cid:i for i,cid in enumerate(levels)};L=np.full((len(specs),p),np.nan)
    for j,s in enumerate(specs.itertuples()):
        if s.lhs_condition not in idx or s.rhs_condition not in idx: continue
        l,r=idx[s.lhs_condition],idx[s.rhs_condition];L[j]=0;L[j,l]=1;L[j,r]=-1
        if name=='condition_slopes':L[j,len(levels)+l]=(t-REF)/100;L[j,len(levels)+r]=-(t-REF)/100
    return L


def fit_one(d,name,specs,rng_seed):
    times=TIMES if name=='condition_slopes' else (108,)
    X,w,levels=design(d,name);p=X.shape[1];n=len(d)
    rank=np.linalg.matrix_rank(X*np.sqrt(w[:,None])) if n else 0
    status='ok' if n>p and rank==p and d.sample_int.nunique()>=2 else 'rank_or_sample_deficiency'
    diag=dict(model=name,n_cycles=n,n_cells=d.sample_int.nunique(),n_conditions=len(levels),
        n_parameters=p,rank=rank,status=status,condition_number=float(np.linalg.cond(X*np.sqrt(w[:,None]))) if n else np.nan,
        bootstrap_requested=N_BOOT,bootstrap_invalid=0)
    matrices={t:contrast_matrix(specs,levels,p,t,name) for t in times}
    supp={t:support(d,t) for t in times}
    effects={t:np.full((N_BOOT,len(specs)),np.nan) for t in times}
    unadjusted_boot=np.full((N_BOOT,len(specs)),np.nan)
    fit=None;raw=np.full(len(specs),np.nan);coeff=[]
    if status=='ok':
        fit=sm.WLS(d.Voltage_band.to_numpy(),X,weights=w).fit(cov_type='cluster',
            cov_kwds={'groups':d.sample_int.to_numpy(),'use_correction':True},use_t=False)
        diag['R_squared']=fit.rsquared
        names=[f'condition_{c}_intercept' for c in levels]
        names+=([f'condition_{c}_slope' for c in levels] if name=='condition_slopes' else ['common_linear'])+['common_quadratic']
        coeff=[dict(term=term,estimate_V=float(val)) for term,val in zip(names,fit.params)]
        groups=list(d.groupby('sample_int',sort=True).indices.items());nc=len(groups)
        sxx=np.zeros((nc,p,p));sxy=np.zeros((nc,p));cids=[]
        y=d.Voltage_band.to_numpy();k=len(levels)
        for j,(sid,pos) in enumerate(groups):
            Z=X[pos];sxx[j]=Z.T@(w[pos,None]*Z);sxy[j]=Z.T@(w[pos]*y[pos]);cids.append(int(d.iloc[pos[0]].Condition_ID))
        totalxx=sxx.sum(axis=0);totalxy=sxy.sum(axis=0)
        raw=matrices[108][:,:k]@(totalxy[:k]/np.diag(totalxx)[:k])
        rng=np.random.default_rng(rng_seed);counts=np.zeros((N_BOOT,nc));cids=np.asarray(cids)
        for cid in levels:
            ii=np.flatnonzero(cids==cid);m=len(ii)
            counts[:,ii]=rng.multinomial(m,np.full(m,1/m),size=N_BOOT)
        # Her bootstrap kopyasi kendi hucre agirligini korur: balanced toplam agirlik/kopya=1.
        bxx=(counts@sxx.reshape(nc,-1)).reshape(N_BOOT,p,p);bxy=counts@sxy
        for b in range(N_BOOT):
            if np.linalg.matrix_rank(bxx[b])<p:
                diag['bootstrap_invalid']+=1;continue
            try:beta=np.linalg.solve(bxx[b],bxy[b])
            except np.linalg.LinAlgError:
                diag['bootstrap_invalid']+=1;continue
            for t in times:effects[t][b]=matrices[t]@beta
            unadjusted_boot[b]=matrices[108][:,:k]@(bxy[b,:k]/np.diag(bxx[b])[:k])
    rows=[];support_rows=[]
    for t in times:
        for cid,sp in supp[t].items():support_rows.append(dict(reference_cycle=t,Condition_ID=cid,**sp))
        for j,s in enumerate(specs.to_dict('records')):
            a,b=supp[t][s['lhs_condition']],supp[t][s['rhs_condition']]
            missing=min(a['n_cells'],b['n_cells'])==0
            inside=a['inside_condition_range'] and b['inside_condition_range']
            replicated=min(a['n_cells_bracketing'],b['n_cells_bracketing'])>=2
            reason='missing_arm' if missing else status if status!='ok' else 'single_cell_arm' if min(a['n_cells'],b['n_cells'])<2 else 'reference_outside_condition_range' if name=='condition_slopes' and not inside else 'insufficient_bracketing_cells' if name=='condition_slopes' and not replicated else 'ok'
            r=dict(**s,reference_cycle=t,status=reason,inference_eligible=reason=='ok',
                support_inside_both_ranges=inside,support_two_bracketing_cells_each=replicated,
                n_cells_lhs=a['n_cells'],n_cells_rhs=b['n_cells'],n_cycles_lhs=a['n_cycles'],n_cycles_rhs=b['n_cycles'],
                n_bracketing_cells_lhs=a['n_cells_bracketing'],n_bracketing_cells_rhs=b['n_cells_bracketing'],
                n_local_cells_lhs=a['n_local_cells'],n_local_cells_rhs=b['n_local_cells'],
                sparse_arm_lt5=min(a['n_cells'],b['n_cells'])<5,
                delta_adjusted_V=np.nan,delta_unadjusted_V=np.nan,absolute_shrinkage_V=np.nan,
                p_value=np.nan,cluster_CI_low_V=np.nan,cluster_CI_high_V=np.nan,
                bootstrap_CI_low_V=np.nan,bootstrap_CI_high_V=np.nan,bootstrap_sign_stability=np.nan,
                shrinkage_CI_low_V=np.nan,shrinkage_CI_high_V=np.nan,bootstrap_shrinkage_fraction=np.nan,
                bootstrap_valid=0,bootstrap_status='ineligible')
            if not missing and fit is not None:
                L=matrices[t][j];point=float(L@fit.params)
                r.update(delta_adjusted_V=point,delta_unadjusted_V=float(raw[j]),absolute_shrinkage_V=abs(raw[j])-abs(point))
                if reason=='ok':
                    test=fit.t_test(L);ci=np.asarray(test.conf_int()).ravel()
                    r.update(p_value=float(np.asarray(test.pvalue).item()),cluster_CI_low_V=float(ci[0]),cluster_CI_high_V=float(ci[1]))
                    v=effects[t][:,j];u=unadjusted_boot[:,j];valid=np.isfinite(v)&np.isfinite(u);v=v[valid];u=u[valid]
                    r['bootstrap_valid']=len(v)
                    r['bootstrap_status']='ok' if len(v)>=np.ceil(N_BOOT*MIN_VALID_FRACTION) else 'too_few_valid_replicates'
                    if r['bootstrap_status']=='ok':
                        lo,hi=np.quantile(v,[.025,.975]);sh=abs(u)-abs(v);sl,shigh=np.quantile(sh,[.025,.975])
                        r.update(bootstrap_CI_low_V=lo,bootstrap_CI_high_V=hi,
                            bootstrap_sign_stability=np.mean([sign(x)==sign(point) for x in v]),
                            shrinkage_CI_low_V=sl,shrinkage_CI_high_V=shigh,bootstrap_shrinkage_fraction=np.mean(sh>0))
            r['point_sign']=sign(r['delta_adjusted_V']);r['point_bucket']=bucket(r['delta_adjusted_V'])
            rows.append(r)
    return rows,diag,coeff,support_rows,effects


def run(blob,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d,specs,previous=read_inputs(blob)
    rows=[];diagnostics=[];coef=[];sup=[];draws={}
    with threadpool_limits(limits=1):
        for bi,band in enumerate(BANDS):
            base=d[d.SoC_band.eq(band)].copy()
            for mi,name in enumerate(MODELS):
                use=base[base.Cycle_Index.between(52,132)].copy() if name=='window_52_132' else base.copy()
                use=use.reset_index(drop=True)
                print(f'{band} / {name}: {len(use):,} cevrim, {use.sample_int.nunique()} hucre; {N_BOOT} bootstrap',flush=True)
                rr,dd,cc,ss,ee=fit_one(use,name,specs,SEED+bi*100+mi)
                rows.extend(dict(model=name,SoC_band=band,**r) for r in rr)
                diagnostics.append(dict(SoC_band=band,**dd))
                coef.extend(dict(model=name,SoC_band=band,**r) for r in cc)
                sup.extend(dict(model=name,SoC_band=band,**r) for r in ss)
                for t,v in ee.items():draws[f'{name}_band{bi}_cycle{t}']=v
    c=pd.DataFrame(rows)
    assert len(c)==486 # 81*(reference + 3 slope times + balanced + window)
    c['q_BH81']=np.nan
    for _,g in c.groupby(['model','reference_cycle']):
        assert len(g)==81
        p=g.p_value.to_numpy();finite=np.isfinite(p)
        q=multipletests(np.where(finite,p,1),method='fdr_bh')[1]
        c.loc[g.index,'q_BH81']=np.where(finite,q,np.nan)
    vcols=[x for x in c if x.endswith('_V')]
    for col in vcols:c[col[:-2]+'_mV']=c[col]*1000
    c['bootstrap_excludes_zero']=pd.Series(pd.NA,index=c.index,dtype='boolean')
    ok=c.bootstrap_status.eq('ok')
    c.loc[ok,'bootstrap_excludes_zero']=(c.loc[ok,'bootstrap_CI_low_V'].gt(0)|c.loc[ok,'bootstrap_CI_high_V'].lt(0)).to_numpy()
    # Asama 3 modeline sayisal geri-dogrulama.
    check=c[c.model.eq('reference')].merge(previous[['SoC_band','contrast_id','delta_age_adjusted_V']],on=['SoC_band','contrast_id'],validate='one_to_one')
    check['difference_mV']=(check.delta_adjusted_V-check.delta_age_adjusted_V)*1000
    check.to_csv(out/'reference_reproduction_check.csv',index=False)
    if len(check)!=81 or not np.allclose(check.delta_adjusted_V,check.delta_age_adjusted_V,rtol=0,atol=1e-8,equal_nan=True):
        raise RuntimeError('Referans 3. adimla uyusmuyor; sonuclari kullanmayin. Kontrol CSVsine bakin.')
    ref=c[c.model.eq('reference')][['SoC_band','contrast_id','delta_adjusted_mV','point_sign','point_bucket','inference_eligible']]
    comparison=c.merge(ref,on=['SoC_band','contrast_id'],suffixes=('','_reference'),validate='many_to_one')
    comparison['change_vs_reference_mV']=comparison.delta_adjusted_mV-comparison.delta_adjusted_mV_reference
    comparison['primary_comparable']=comparison.inference_eligible & comparison.inference_eligible_reference
    for target,col in [('sign_changed','point_sign'),('bucket_changed','point_bucket')]:
        comparison[target]=pd.Series(pd.NA,index=comparison.index,dtype='boolean')
        mask=comparison.primary_comparable
        comparison.loc[mask,target]=(comparison.loc[mask,col]!=comparison.loc[mask,col+'_reference']).to_numpy()
    # Ayni destekli kontrast kumesi: 108'de dort analiz de uygun olmali.
    at108=comparison[comparison.reference_cycle.eq(108)].copy()
    common=at108.groupby(['SoC_band','contrast_id']).agg(all_eligible=('primary_comparable','all'),n_models=('model','nunique'))
    common['common_four_models']=common.all_eligible & common.n_models.eq(4)
    comparison=comparison.merge(common[['common_four_models']],on=['SoC_band','contrast_id'],validate='many_to_one')
    c.to_csv(out/'all_model_contrasts.csv',index=False)
    comparison.to_csv(out/'model_changes_vs_reference.csv',index=False)
    comparison[(comparison.reference_cycle==108)&comparison.common_four_models].to_csv(out/'common_supported_contrasts_at108.csv',index=False)
    c[c.contrast_family.eq('temperature_60_minus_25')].to_csv(out/'thermal_adjustment_and_shrinkage.csv',index=False)
    c[c.contrast_family.isin(['rate_2C_minus_1C','rate_0.7C_minus_2C'])].to_csv(out/'aggressive_rate_robustness.csv',index=False)
    c[~c.inference_eligible].to_csv(out/'unsupported_or_descriptive_contrasts.csv',index=False)
    pd.DataFrame(sup).to_csv(out/'condition_cycle_support.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'model_diagnostics.csv',index=False)
    pd.DataFrame(coef).to_csv(out/'model_coefficients.csv',index=False)
    np.savez_compressed(out/'bootstrap_contrast_draws.npz',**draws)
    specs.to_csv(out/'contrast_specifications.csv',index=False)
    summary=[]
    for (name,t),g in comparison.groupby(['model','reference_cycle']):
        good=g[g.primary_comparable]
        summary.append(dict(model=name,reference_cycle=t,n_planned=len(g),n_eligible=int(g.inference_eligible.sum()),
            n_comparable=len(good),sign_changes=int(good.sign_changed.sum()),bucket_changes=int(good.bucket_changed.sum()),
            max_abs_change_mV=good.change_vs_reference_mV.abs().max(),
            n_bootstrap_supported=int(g.bootstrap_excludes_zero.fillna(False).sum())))
    pd.DataFrame(summary).to_csv(out/'model_comparison_summary.csv',index=False)
    cfg=dict(source_sha256=hashlib.sha256(blob).hexdigest(),timestamp_utc=datetime.now(timezone.utc).isoformat(),
        bootstrap_replicates=N_BOOT,seed=SEED,reference_cycle=REF,local_radius=LOCAL_RADIUS,
        slope_times=TIMES,min_bracketing_cells_per_arm=2,min_valid_bootstrap_fraction=MIN_VALID_FRACTION,
        models=MODELS,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,statsmodels=sm.__version__,
        reference_max_difference_mV=float(check.difference_mV.abs().max()))
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'README.txt').write_text(
        'ASAMA 4: Sabit %80 bant kapsamli veride yaslanma modeli duyarliligi.\n'
        'reference: kosul sabitleri + ortak age + age^2, age=(Cycle_Index-108)/100.\n'
        'condition_slopes: kosul sabitleri + kosula ozgu dogrusal egimler + ortak age^2.\n'
        'cell_balanced: referans model, her satir agirligi=1/(ilgili hucrenin bu banttaki uygun cevrim sayisi).\n'
        'window_52_132: %80 uygun verinin 52..132 dahil alt kumesi, referans model.\n'
        'Tum modellerde fiziksel hucre CR1 kovaryansi ve normal referans dagilimi kullanilir.\n'
        'Duzeltilmemis fark, ayni modelin veri ve agirliklariyla kosul ortalamalarinin farkidir.\n'
        'absolute_shrinkage = abs(unadjusted)-abs(adjusted); pozitif deger mutlak kuculmedir.\n'
        'Egimli modelde unadjusted tum uygun cevrim gecmisi ortalamasidir; t cevrimine yerel ortalama degildir.\n'
        'Bootstrap kuculme araliklari ayni tekrar icindeki duzeltilmis/duzeltilmemis tahminlerden uretilir.\n'
        'Her kosul icinde hucreler yeniden orneklenir; secilen hucrenin tum uygun cevrimleri tasinir.\n'
        'Balanced bootstrap: her secilmis hucre kopyasinin toplam agirligi 1 olur; coklu secim bu agirligi carpandir.\n'
        'Tek hucreli kol: nokta betimsel; p/CI yok. Eksik kol: nokta da NA.\n'
        'Egimli model: her kolda en az iki hucrenin kendi min/max cevrim araligi t yi kapsamalidir.\n'
        'Bu destek kurali sonuclardan once sabitlenmistir; yerel +/-10 hucre sayilari ek tani olup ikinci esik degildir.\n'
        'Destek disi egimli nokta tahminleri denetim icin saklanir, primary_comparable=False; ana sonuca katilmaz.\n'
        'Ortak egimli modelin kontrasti t den bagimsizdir; support bilgisi raporlanir fakat t-destegi kontrasta ek kosul yapilmaz.\n'
        '52,108,132 egimli sonuclari ayri 81-test BH aileleridir. 108 ana, 52/132 kesifsel duyarliliktir.\n'
        'Eksik/uygunsuz testler BH hesabinda p=1 yer tutar; raporda q NA kalir. Modeller arasi ortak FDR iddiasi yoktur.\n'
        'Bootstrap CI yuzdelik ve noktasaldir; coklu karsilastirma duzeltilmis CI degildir.\n'
        'Rank eksik model/tekrarda pseudoinverse veya model degistirme yoktur. Gecerli tekrar <%95 ise CI verilmez.\n'
        'NPZ draw sutun sirasi contrast_specifications.csv; birim V. Destek disi ham drawlar ana CI icin kullanilmaz.\n'
        'common_supported_contrasts_at108 dort modelde de uygun ayni kontrastlarin karsilastirmasidir.\n'
        'Pencere analizi gozlem kumesini; balanced analiz hedeflenen agirliklandirmayi degistirir. Farklar salt model bicimine baglanamaz.\n'
        'Condition5 tek hucre/iki cevrimle temsil edildigi icin egimi guvenilir grup etkisi olarak yorumlanmaz.\n'
        'Yas etkisi cevrim gecmisi iliskisidir; elektrokimyasal yaslanmanin nedensel etkisi degildir.\n',encoding='utf-8')
    print('\nREFERANS KONTROLU: en buyuk fark (mV) =',cfg['reference_max_difference_mV'])
    print('\nMODEL KARSILASTIRMASI\n',pd.DataFrame(summary).to_string(index=False))
    print('\nTAMAM: Sonuclari model, destek ve bootstrap durumuyla birlikte okuyun.')
    return c,comparison


def main():
    from google.colab import files
    print('3. adimin sonuc ZIP dosyasini yukleyin: SoC_band_coverage_ts2_1sbp.zip')
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('Tek ZIP yukleyin.')
    name,blob=next(iter(uploaded.items()))
    if not name.lower().endswith('.zip'):raise ValueError('ZIP dosyasi gerekli.')
    out=Path(tempfile.mkdtemp(prefix='SoC_age_sensitivity_'))
    run(blob,out)
    archive=shutil.make_archive(str(out),'zip',root_dir=out)
    print('\nSonuc ZIP:',archive);files.download(archive)


# Original auto-run omitted: the runner below supplies audited local inputs.

# ---------- LOCAL, NONDESTRUCTIVE STAGE RUNNER ----------
path = require_path('/content/SoC_band_coverage_ConditionvCodes2.zip')
out = stage_dir('02_age_sensitivity')
print('SOURCE COVERAGE:',path)
run(path.read_bytes(),out)
import pandas as pd
c = pd.read_csv(out/'all_model_contrasts.csv')
print('AGE CONTRAST ROWS:',len(c))
print('REPRODUCTION QA:')
print(pd.read_csv(out/'reference_reproduction_check.csv').head(8).to_string(index=False))
stage_zip('02_age_sensitivity','SoC_age_sensitivity_')
