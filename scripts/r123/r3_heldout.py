"""R3 prospective OOF curve-compression evaluation on archived CALCE held-out cells.
Reference B0 is rebuilt from original per-cell inputs and checked against
archived unified node predictions. No held-out observations enter fitting.
"""
from __future__ import annotations
import hashlib, io, json, warnings
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

ROOT=Path('/content/SETA_R123_REPRO_20261025T184824Z/inputs')
INPUT=ROOT/'SoC_unified_ConditionvCodes2.zip'
OUT=Path('/content/SETA_R123_REPRO_20261025T184824Z/results/R3_Heldout'); OUT.mkdir(exist_ok=True)
GRID=np.arange(101); SCORE=np.arange(91)
DTYPES=('float32','float16'); STEPS=(1,2,5); INTERPS=('linear','pchip')
METHODS=[f'step{step}_{dtype}_{alg}' for step in STEPS for dtype in DTYPES for alg in INTERPS]
META={1+6*t+2*r+k:(T,cutoff,rate) for t,T in enumerate((10,25,45,60)) for r,rate in enumerate((0.7,1.,2.)) for k,cutoff in enumerate(('C/5','C/40'))}
INVERSE={v:k for k,v in META.items()}

def sha(data): return hashlib.sha256(data).hexdigest()

def load(csv,zip):
    d=pd.read_csv(io.BytesIO(zip.read(csv)))
    if 'V_cell_median' in d.columns: d=d.rename(columns={'V_cell_median':'Voltage'})
    if d.duplicated(['Condition_ID','sample_int','SoC']).any(): raise ValueError('Duplicates '+csv)
    return d

def arrays(d):
    groups={}
    for cid,g in d.groupby('Condition_ID'):
        q=g.pivot(index='sample_int',columns='SoC',values='Voltage').reindex(columns=GRID).sort_index()
        assert q.shape[1]==101 and q.index.is_unique
        groups[int(cid)]=q
    return groups

def fit(A):
    A=np.asarray(A,float).reshape(-1,101)
    mask=np.isfinite(A); n=mask.sum(axis=0)
    out=np.full(101,np.nan,float)
    fit_nodes=n>0
    if fit_nodes.sum()>=2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            raw=np.nanmedian(A,axis=0)
        iso=IsotonicRegression(increasing=True,out_of_bounds='nan')
        out[fit_nodes]=iso.fit_transform(GRID[fit_nodes],raw[fit_nodes],sample_weight=n[fit_nodes])
    return out

def reduce_curve(full,step,dtype,alg):
    # Preserve reference fine-grid unsupported nodes; no interpolation across
    # internal gaps of original supported nodes. Compression can reduce coverage.
    full=np.asarray(full,float)
    result=np.full(91,np.nan)
    ok=np.isfinite(full[:91]); knots=np.arange(0,91,step)
    if not ok.any(): return result
    starts=np.where(ok & ~np.r_[False,ok[:-1]])[0]
    ends=np.where(ok & ~np.r_[ok[1:],False])[0]
    for start,end in zip(starts,ends):
        kk=knots[(knots>=start)&(knots<=end)]
        if not len(kk):continue
        yy=full[kk].astype(dtype).astype(float)
        xx=np.arange(start,end+1)
        if len(kk)==1:
            result[int(kk[0])]=yy[0];continue
        xx=xx[(xx>=kk[0]) & (xx<=kk[-1])]
        if alg=='linear':result[xx]=np.interp(xx,kk,yy)
        else: result[xx]=PchipInterpolator(kk,yy,extrapolate=False)(xx)
    return result

def eval_one(y,p):
    mask=np.isfinite(y)&np.isfinite(p)
    if mask.sum()<5: return dict(eligible=False,n_paired=int(mask.sum()),MAE_mV=np.nan,RMSE_mV=np.nan,bias_mV=np.nan,max_abs_mV=np.nan)
    dif=(p[mask]-y[mask])*1000
    return dict(eligible=True,n_paired=int(mask.sum()),MAE_mV=float(np.mean(abs(dif))),RMSE_mV=float(np.sqrt(np.mean(dif*dif))),bias_mV=float(np.mean(dif)),max_abs_mV=float(np.max(abs(dif))))

with ZipFile(INPUT) as z:
    assert z.testzip() is None
    old=pd.read_csv(z.open('validation_node_predictions.csv'))
    old=old[(old.method=='unified')&(old.SoC<=90)]
    summary_old=pd.read_csv(z.open('validation_summary.csv'))
    profiles={'full_history':arrays(load('inputs/cell_profiles_nominal_concordant.csv',z)),
              'broad_52_132':arrays(load('inputs/broad_52_132_cell_profiles.csv',z))}
    source_hash={'full_history':sha(z.read('inputs/cell_profiles_nominal_concordant.csv')),
                 'broad_52_132':sha(z.read('inputs/broad_52_132_cell_profiles.csv'))}

old_lookup={(str(k[0]),str(k[1]),int(k[2]),int(k[3])):frame.sort_values('SoC') for k,frame in old.groupby(['analysis','validation','Condition_ID','sample_int'],sort=False)}
prediction_rows=[]; parity=[]
for analysis, groups in profiles.items():
    cache={cid:fit(g.to_numpy()) for cid,g in groups.items()}
    for cid in sorted(groups):
        T,cutoff,rate=META[cid]
        g=groups[cid]
        exps=[('LOCO',None,None,0.)]
        if T in (25,45):
            tlo,thi=((10,45) if T==25 else (25,60))
            exps.append(('temperature_holdout',INVERSE[(tlo,cutoff,rate)],INVERSE[(thi,cutoff,rate)],(T-tlo)/(thi-tlo)))
        if rate==1.:
            exps.append(('rate_holdout',INVERSE[(T,cutoff,0.7)],INVERSE[(T,cutoff,2.)],(rate-0.7)/(2.-0.7)))
        for val,lo,hi,w in exps:
            if val!='LOCO':
                ca=cache[lo]; cb=cache[hi]
                variants={m:(1-w)*reduce_curve(ca,int(m.split('_')[0][4:]),m.split('_')[1],m.split('_')[2]) + w*reduce_curve(cb,int(m.split('_')[0][4:]),m.split('_')[1],m.split('_')[2]) for m in METHODS}
                base=(1-w)*ca[:91]+w*cb[:91]
            for sid,series in g.iterrows():
                y=series.to_numpy(float)[:91]
                if val=='LOCO':
                    ca=fit(g.drop(index=sid).to_numpy())
                    base=ca[:91]
                    variants={m:reduce_curve(ca,int(m.split('_')[0][4:]),m.split('_')[1],m.split('_')[2]) for m in METHODS}
                key={'analysis':analysis,'validation':val,'Condition_ID':cid,'sample_int':int(sid)}
                archive=old_lookup[(analysis,val,int(cid),int(sid))]
                if len(archive)!=91:raise ValueError(f'Archive key not 91 {key}: {len(archive)}')
                if not np.allclose(archive.target_V.to_numpy(),y,equal_nan=True,atol=1e-12):raise ValueError('Test targets mismatch')
                archived=archive.prediction_V.to_numpy(float)
                if not np.allclose(archived,base,equal_nan=True,atol=1e-12):
                    dev=np.nanmax(np.abs(archived-base))
                    raise ValueError(f'B0 parity failed at {key}: {dev}')
                parity.append(np.nanmax(np.abs(archived-base)) if np.isfinite(base).any() else 0.)
                common=np.isfinite(y)&np.isfinite(base)
                for p in variants.values(): common &=np.isfinite(p)
                predict={'B0_original':base,**variants}
                for m,p in predict.items():
                    for kind,mask in [('own',np.isfinite(y)&np.isfinite(p)),('all_model_common',common)]:
                        for region,lo_soc,hi_soc in [('all_0_90',0,90),('low_0_19',0,19),('middle_20_59',20,59),('high_60_90',60,90)]:
                            rangemask=(SCORE>=lo_soc)&(SCORE<=hi_soc)
                            yq=np.where(mask&rangemask,y,np.nan)
                            pq=np.where(mask&rangemask,p,np.nan)
                            record=eval_one(yq,pq)
                            prediction_rows.append({**key,'model':m,'support':kind,'region':region,'n_target':int((np.isfinite(y)&rangemask).sum()),'n_nodes_in_region':int(rangemask.sum()),**record})
    print('COMPLETE:',analysis, flush=True)

metrics=pd.DataFrame(prediction_rows)
metrics.to_csv(OUT/'R3_OOF_cell_metrics.csv',index=False)
cols=['analysis','validation','model','support','region']
agg=metrics.groupby(cols,dropna=False).agg(n_planned_cells=('sample_int','size'),n_evaluated_cells=('eligible','sum'),n_finite_targets=('n_target','sum'),n_paired=('n_paired','sum'),MAE_mV=('MAE_mV','mean'),RMSE_mV=('RMSE_mV','mean'),bias_mV=('bias_mV','mean'),max_cell_abs_error_mV=('max_abs_mV','max')).reset_index()
agg['finite_target_coverage']=agg.n_paired/agg.n_finite_targets
agg[agg.support=='own'].to_csv(OUT/'R3_OOF_own_support.csv',index=False)
agg[agg.support=='all_model_common'].to_csv(OUT/'R3_OOF_common_support.csv',index=False)
cond=metrics.groupby(cols+['Condition_ID'],dropna=False).agg(n_planned_cells=('sample_int','size'),n_evaluated_cells=('eligible','sum'),n_paired=('n_paired','sum'),MAE_mV=('MAE_mV','mean'),max_cell_abs_error_mV=('max_abs_mV','max')).reset_index()
cond.to_csv(OUT/'R3_by_condition.csv',index=False)
storage=[]
for step in STEPS:
 for dtype in DTYPES:
  for alg in INTERPS:
   storage.append(dict(model=f'step{step}_{dtype}_{alg}',step_SoC=step,stored_values=24*len(np.arange(0,91,step)),bytes_per_value=(2 if dtype=='float16' else 4),payload_KiB=24*len(np.arange(0,91,step))*(2 if dtype=='float16' else 4)/1024,interpolation=alg,excludes_support_masks=True))
storage.append(dict(model='B0_original',step_SoC=1,stored_values=2184,bytes_per_value=4,payload_KiB=8.53125,interpolation='linear',excludes_support_masks=True))
pd.DataFrame(storage).to_csv(OUT/'R3_storage_budget.csv',index=False)
# Assert reconstructed original results match published within float precision
ref=agg[(agg.model=='B0_original')&(agg.support=='own')&(agg.region=='all_0_90')].copy()
for _,row in ref.iterrows():
    rr=summary_old[(summary_old.analysis==row.analysis)&(summary_old.validation==row.validation)&(summary_old.method=='unified')&(summary_old.comparison=='own_support')&(summary_old.band=='all_0_90_inclusive')]
    assert len(rr)==1
    if abs(rr.MAE_mV.iloc[0]-row.MAE_mV)>1e-8:raise ValueError('MAE parity failed')
manifest=dict(source_zip=INPUT.name,source_zip_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),source_csv_sha256=source_hash,method='original source code isotonic training-fold reconstruction, then fold-specific downsampling of learned curves',baseline_parity_max_abs_V=float(np.max(parity)),versions={'numpy':np.__version__,'pandas':pd.__version__},n_cell_metrics=int(len(metrics)),all_model_common_evaluation=True,comment='R3 exploratory OOF evaluation. The fine-grid support mask is preserved; storage budget excludes mask and runtime.')
(OUT/'R3_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('\nB0 REFERENCE PARITY:',max(parity))
print('\nR3 OOF (52-132, own support):')
print(agg[(agg.analysis=='broad_52_132')&(agg.support=='own')&(agg.region=='all_0_90')][['validation','model','MAE_mV','finite_target_coverage']].sort_values(['validation','MAE_mV']).to_string(index=False))
print('\nRESULTS:',OUT)
