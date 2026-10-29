#!/usr/bin/env python3
"""SETA R2/B3 fixed-fold ridge baseline — independent reproducibility script.

Protocol: 2026-10-21 frozen R2. Identical held-out physical cells and
finite target nodes as the unified-stage OOF archive. Inner CV splits only
outer training cells. All displayed comparison metrics are OOF cell-averaged.
"""
from __future__ import annotations
import io, json, hashlib, time, os, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

SOURCE=Path(os.environ.get('R2_SOURCE','/mnt/data/SoC_unified_ConditionvCodes2.zip'))
OUT=Path(os.environ.get('R2_OUTPUT','/mnt/data/SETA_R2_B3_Ridge_2026-10-23'))
OUT.mkdir(parents=True,exist_ok=True)
ALPHAS=np.array([0.01, 0.1, 1., 10., 100.], dtype=float)
SEED=20261021
GRID=np.arange(91)
FEATURE_NAMES=['1/T_K','log_discharge_Crate','cutoff_C40','log_rate/T_K','cutoff_C40/T_K','cutoff_C40*log_rate']

def read_csv(z,name):
    hits=[q for q in z.namelist() if q==name or q.endswith('/'+name)]
    if len(hits)!=1:raise ValueError(f'Expected exactly one {name}: {hits}')
    return pd.read_csv(z.open(hits[0]))

def make_design(md):
    T=np.asarray(md.Temperature,float)+273.15
    r=np.asarray(md.Charge_Rate,float)
    c=np.asarray(md.Rate.eq('C/40'),float)
    logr=np.log(r); inv=1/T
    return np.column_stack([inv,logr,c,logr*inv,c*inv,c*logr])

def fold_standardization(raw_training):
    mu=raw_training.mean(axis=0)
    sigma=raw_training.std(axis=0)
    sigma[sigma<1e-12]=1.
    return mu,sigma

def ridge_multialpha(train_raw,train_y, query_raw, alphas):
    # Feature scaling uses training cells only, once for all nodes.
    mu,sd=fold_standardization(train_raw)
    X=np.column_stack([np.ones(train_raw.shape[0]),(train_raw-mu)/sd])
    Z=np.column_stack([np.ones(query_raw.shape[0]),(query_raw-mu)/sd])
    yy=np.nan_to_num(train_y,nan=0.)
    valid=np.isfinite(train_y)
    n_support=valid.sum(axis=0)
    # Sufficient statistics per SoC node: (91,7,7), (91,7)
    gram=np.einsum('ni,nj,nm->mij',X,X,valid.astype(float),optimize=True)
    rhs=np.einsum('ni,nm->mi',X,yy,optimize=True)
    reg=np.zeros((len(alphas),7))
    reg[:,0]=1e-10  # machine-scale stabilizer, not practical penalization
    reg[:,1:]=alphas[:,None]
    A=gram[None,:,:,:]+np.eye(7)[None,None,:,:]*reg[:,None,None,:]
    B=np.broadcast_to(rhs,(len(alphas),*rhs.shape)).copy()
    beta=np.linalg.solve(A,B[...,None])[...,0]
    pred=np.einsum('ni,ami->anm',Z,beta,optimize=True)
    pred[:,:,n_support==0]=np.nan
    return pred,n_support

def group_cv(train_ids,cond_ids):
    # Unique cell is the unit; grouping by condition preserves representation.
    # StratifiedKFold can issue a warning for n=2 strata; assignment remains
    # deterministic and never splits a physical cell across train and validation.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        splits=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(train_ids,cond_ids))
    if len(splits)!=3:raise AssertionError('3-fold CV not constructed')
    return splits

def choose_alpha(tr_raw,tr_y,tr_ids,tr_condition):
    splits=group_cv(tr_ids,tr_condition)
    scores=[]; diagnostics=[]
    for j,(inside,validation) in enumerate(splits):
        if np.intersect1d(tr_ids[inside],tr_ids[validation]).size:
            raise AssertionError('Leakage in inner CV')
        pred,n_sup=ridge_multialpha(tr_raw[inside],tr_y[inside],tr_raw[validation],ALPHAS)
        target=tr_y[validation][None,:,:]
        mask=np.isfinite(pred)&np.isfinite(target)
        n=mask.sum(axis=2)
        abs_err=np.where(mask,abs(pred-target),0.)
        cell_mae=np.divide(abs_err.sum(axis=2),n,out=np.full(n.shape,np.nan,dtype=float),where=n>=5)*1000
        scores.append(cell_mae)
        diagnostics.append(dict(inner_fold=j,n_test_cells=len(validation),n_usable_cells=int(np.isfinite(cell_mae[0]).sum()),n_pairs=int(n[0].sum())))
    scores=np.concatenate(scores,axis=1)
    n_cells=np.isfinite(scores).sum(axis=1)
    score=np.nanmean(scores,axis=1) if np.any(n_cells) else np.full(len(ALPHAS),np.nan)
    if np.all(~np.isfinite(score)):
        chosen=1.; fallback=True
    else:
        ii=np.nanargmin(score)
        chosen=float(ALPHAS[ii]); fallback=False
    return chosen,fallback,dict(zip([str(a) for a in ALPHAS], [float(s) for s in score])),diagnostics

def metric_rows(rows):
    records=[]
    for (analysis,validation,model,support_type),gr in rows.groupby(['analysis','validation','model','support_type'],sort=True):
        # Equal weights per physical cell, not per target node.
        g=gr[gr.n_scored>=5]
        records.append(dict(analysis=analysis,validation=validation,model=model,support_type=support_type,
            n_cells=int(g.shape[0]),n_planned=int(gr.shape[0]),n_finite_target=int(gr.n_target.sum()),
            n_scored=int(g.n_scored.sum()),target_coverage=float(g.n_scored.sum()/gr.n_target.sum()) if gr.n_target.sum() else np.nan,
            mean_MAE_mV=float(g.MAE_mV.mean()),mean_RMSE_mV=float(g.RMSE_mV.mean()),
            mean_bias_mV=float(g.bias_mV.mean()),
            median_MAE_mV=float(g.MAE_mV.median()),p90_MAE_mV=float(g.MAE_mV.quantile(.90)),
            max_abs_error_mV=float(g.max_abs_error_mV.max())))
    return pd.DataFrame(records)

def main():
    t0=time.time()
    assert SOURCE.is_file(),SOURCE
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with zipfile.ZipFile(SOURCE) as z:
        assert z.testzip() is None
        nodal=read_csv(z,'validation_node_predictions.csv')
        meta=read_csv(z,'condition_map.csv')
        source_data={
            'full_history':read_csv(z,'inputs/cell_profiles_nominal_concordant.csv').rename(columns={'V_cell_median':'Voltage'}),
            'broad_52_132':read_csv(z,'inputs/broad_52_132_cell_profiles.csv')}
        config=json.loads(z.read('run_config.json'))
    nodal=nodal[(nodal.method.eq('unified')) & (nodal.SoC.between(0,90))].copy()
    nodal.sort_values(['analysis','validation','sample_int','SoC'],inplace=True)
    assert not nodal.duplicated(['analysis','validation','sample_int','SoC']).any()
    conditions=meta.set_index('Condition_ID')
    V={}; ids={}; X={}; labels={}; cond={}; idmap={}
    for scope,d in source_data.items():
        d=d[d.SoC.between(0,90)].copy()
        assert d.SoC.eq(d.SoC.round()).all()
        # Independent physical-cell row for each SoC node, NA preserved.
        key=d[['sample_int','Condition_ID']].drop_duplicates()
        assert not key.sample_int.duplicated().any()
        pivot=d.pivot(index='sample_int',columns='SoC',values='Voltage').reindex(columns=GRID)
        pivot=pivot.sort_index()
        u=pivot.index.to_numpy(dtype=int)
        cm=key.set_index('sample_int').loc[u,'Condition_ID'].to_numpy(dtype=int)
        cm_meta=conditions.loc[cm].reset_index(drop=True)
        raw=make_design(cm_meta)
        ids[scope]=u; cond[scope]=cm; V[scope]=pivot.to_numpy(float)
        X[scope]=raw
        idmap[scope]={int(k):int(i) for i,k in enumerate(u)}
        assert len(u)==(180 if scope=='full_history' else 179)
    outer=[]
    for scope,sub in nodal.groupby('analysis',sort=True):
        for validation,subv in sub.groupby('validation',sort=True):
            if validation=='LOCO':
                outer.extend((scope,'LOCO',f'cell_{int(sid)}',g) for sid,g in subv.groupby('sample_int',sort=True))
            elif validation=='rate_holdout':
                assert np.array_equal(np.unique(subv.Charge_Rate),np.array([1.]))
                outer.append((scope,'rate_holdout','rate_1C',subv))
            else:
                assert validation=='temperature_holdout'
                for T,g in subv.groupby('Temperature',sort=True):
                    assert T in (25,45)
                    outer.append((scope,'temperature_holdout',f'temperature_{int(T)}',g))
    print('Outer folds:',len(outer),flush=True)
    predicted=[]; registry=[]; selected=[]
    with threadpool_limits(limits=1):
        for fi,(scope,validation,foldid,g) in enumerate(outer):
            g=g.sort_values(['sample_int','SoC']).copy()
            held=np.unique(g.sample_int.to_numpy(dtype=int))
            pool_ids=ids[scope]
            if validation=='LOCO':
                train_mask=~np.isin(pool_ids,held)
            elif validation=='rate_holdout':
                held_condition=conditions.loc[cond[scope]]
                train_mask=~np.isclose(held_condition.Charge_Rate.to_numpy(float),1.)
            else:
                target_temp=int(foldid.split('_')[1])
                held_condition=conditions.loc[cond[scope]]
                train_mask=held_condition.Temperature.to_numpy(int)!=target_temp
            train_u=pool_ids[train_mask]
            if np.intersect1d(train_u,held).size:raise AssertionError(f'Leakage outer {scope}/{foldid}')
            if not np.array_equal(np.sort(pool_ids[~train_mask]),np.sort(held)):
                raise AssertionError(f'Outer fold split not matching archive: {scope}/{foldid}')
            trX=X[scope][train_mask];trY=V[scope][train_mask]
            trCond=cond[scope][train_mask]
            alpha, fallback, cvs, cv_folds=choose_alpha(trX,trY,train_u,trCond)
            qIdx=np.array([idmap[scope][int(v)] for v in g.sample_int],int)
            pred,support=ridge_multialpha(trX,trY,X[scope][qIdx],np.array([alpha]))
            pred=pred[0,np.arange(len(g)),g.SoC.to_numpy(int)]
            # Verify target provenance against original physical-cell source.
            source_targets=V[scope][qIdx,g.SoC.to_numpy(int)]
            if not np.allclose(source_targets,g.target_V.to_numpy(float),atol=1e-12,equal_nan=True):
                raise AssertionError(f'Target provenance mismatch {scope}/{foldid}')
            g['ridge_pred_V']=pred
            g['ridge_training_node_support']=support[g.SoC.to_numpy(int)]
            g['selected_alpha']=alpha
            g['outer_fold']=foldid
            g['test_unit_excluded_from_fit']=True
            predicted.append(g)
            registry.append(dict(analysis=scope,validation=validation,fold=foldid,
                n_train_cells=int(train_mask.sum()),n_test_cells=len(held),
                n_training_conditions=int(len(np.unique(trCond))),selected_alpha=alpha,
                inner_cv_fallback=fallback,test_cell_ids=';'.join(map(str,held)),
                test_temperature=(int(foldid.split('_')[1]) if validation=='temperature_holdout' else np.nan)))
            selected.append(dict(analysis=scope,validation=validation,fold=foldid,alpha=alpha,fallback=fallback,
                **{f'CV_alpha_{str(a).replace(".","p")}':cvs[str(a)] for a in ALPHAS},
                min_inner_n_usable_cells=min(x['n_usable_cells'] for x in cv_folds)))
            if (fi+1)%30==0:print(f'Processed {fi+1}/{len(outer)} outer folds in {time.time()-t0:.1f} s',flush=True)
    nd=pd.concat(predicted,ignore_index=True)
    assert len(nd)==len(nodal)
    nd['B0_original_pred_V']=nd.prediction_V
    if nd.ridge_pred_V.isna().sum()>0:print('ridge NaN nodes',nd.ridge_pred_V.isna().sum(),flush=True)
    # Need per-cell metrics: both own supports and each pair's identical common support.
    results=[]
    for (scope,validation,sid),g in nd.groupby(['analysis','validation','sample_int'],sort=True):
        y=g.target_V.to_numpy(float)
        p0=g.B0_original_pred_V.to_numpy(float)
        p3=g.ridge_pred_V.to_numpy(float)
        common=np.isfinite(y)&np.isfinite(p0)&np.isfinite(p3)
        for support_type in ('own_support','common_B0_B3'):
            for model,pred in (('B0_condition_isotonic_LUT',p0),('B3_ridge_7_coeff_per_soc',p3)):
                mm=(np.isfinite(y)&np.isfinite(pred)) if support_type=='own_support' else common
                n=int(mm.sum())
                e=(pred[mm]-y[mm])*1000
                results.append(dict(analysis=scope,validation=validation,sample_int=int(sid),
                    Condition_ID=int(g.Condition_ID.iloc[0]),model=model,support_type=support_type,
                    n_target=int(np.isfinite(y).sum()),n_scored=n,
                    MAE_mV=float(np.mean(abs(e))) if n>=5 else np.nan,
                    RMSE_mV=float(np.sqrt(np.mean(e**2))) if n>=5 else np.nan,
                    bias_mV=float(np.mean(e)) if n>=5 else np.nan,
                    max_abs_error_mV=float(np.max(abs(e))) if n>=5 else np.nan))
    metric_cells=pd.DataFrame(results)
    summary=metric_rows(metric_cells)
    baseline={('full_history','LOCO'):5.240437,('full_history','rate_holdout'):19.531124,
              ('full_history','temperature_holdout'):32.141306,
              ('broad_52_132','LOCO'):4.353667,('broad_52_132','rate_holdout'):18.041400,
              ('broad_52_132','temperature_holdout'):33.189926}
    for (sc,val),v in baseline.items():
        row=summary[(summary.analysis.eq(sc)) & (summary.validation.eq(val)) &
            (summary.model.eq('B0_condition_isotonic_LUT')) & (summary.support_type.eq('own_support'))]
        assert len(row)==1
        assert abs(float(row.mean_MAE_mV.iloc[0])-v)<1e-4, (sc,val,row.mean_MAE_mV.iloc[0],v)
    # n coefficients = 7*91 float32 (including intercept), can omit zeros only with separate encoding.
    storage=pd.DataFrame([
        dict(model='B0_condition_isotonic_LUT',float32_values=24*91,payload_KiB=24*91*4/1024,
             qualifications='Fixed 24-condition x 91-node voltage payload only; masks not counted'),
        dict(model='B3_ridge_7_coeff_per_soc',float32_values=7*91,payload_KiB=7*91*4/1024,
             qualifications='91-node x 7 coefficients; scaling/metadata/support masks omitted')])
    metrics=summary.sort_values(['analysis','validation','support_type','model'])
    outfiles={'R2_B3_prediction_nodes.csv':nd[['analysis','validation','outer_fold','Condition_ID','sample_int','Temperature','Rate','Charge_Rate','SoC','target_V','B0_original_pred_V','ridge_pred_V','ridge_training_node_support','selected_alpha','test_unit_excluded_from_fit']],
    'R2_B3_fold_registry.csv':pd.DataFrame(registry),
    'R2_B3_alpha_selected.csv':pd.DataFrame(selected),
    'R2_B3_cell_metrics.csv':metric_cells,
    'R2_B3_support_summary.csv':metrics,
    'R2_B3_storage_budget.csv':storage}
    for n,d in outfiles.items():d.to_csv(OUT/n,index=False)
    manifest={'source':str(SOURCE),'source_sha256':source_hash,'source_csv_sha256':config.get('input_sha256'),
        'script':Path(__file__).name,'fixed_protocol_date':'2026-10-21',
        'seed':SEED,'n_outer_folds':len(outer),'inner_cv':3,
        'alpha_grid':ALPHAS.tolist(),'features':FEATURE_NAMES,'model':'nodewise ridge; physical cell independent row',
        'intercept_regularized':False,'support_rule':'No SoC extrapolation: no finite training voltage at node => NaN',
        'n_source_nodes':len(nodal),'n_predicted_nodes':len(nd),'runtime_seconds':time.time()-t0,
        'all_original_OOF_metrics_matched':True,'research_scope':'offline retrospective-normalized voltage representation; not online BMS evaluation',
        'notes':['Outer factor-level holdouts exclude all withheld cells','Inner folds contain only outer training cells',
                 'Equal cell weighting in metric; alpha selected by cell-level 3-fold CV',
                 'Payload size excludes support masks and model metadata']}
    (OUT/'R2_B3_run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('\nB3 RIDGE — FINISHED',round(time.time()-t0,2),'seconds\n')
    print(summary[summary.support_type.eq('own_support')][['analysis','validation','model','n_cells','n_scored','target_coverage','mean_MAE_mV','mean_bias_mV']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nCOMMON\n',summary[summary.support_type.eq('common_B0_B3')][['analysis','validation','model','n_cells','n_scored','mean_MAE_mV']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Output:',OUT)

if __name__=='__main__':main()
