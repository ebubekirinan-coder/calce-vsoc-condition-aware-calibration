from pathlib import Path
from zipfile import ZipFile
import json, hashlib, warnings
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
import importlib.util

D=Path('/content/SETA_R123_REPRO_20261025T184824Z/results/R2_Ridge'); P=Path('/content/SETA_R123_REPRO_20261025T184824Z/inputs/SoC_unified_ConditionvCodes2.zip')
cells=pd.read_csv(D/'R2_B3_cell_metrics.csv')
alpha=pd.read_csv(D/'R2_B3_alpha_selected.csv')
manifest=json.loads((D/'R2_B3_run_manifest.json').read_text())
boot=[]
rng=np.random.default_rng(20261021)
for (scope,val),gr in cells[cells.support_type.eq('common_B0_B3')].groupby(['analysis','validation']):
    wide=gr.pivot(index=['Condition_ID','sample_int'],columns='model',values='MAE_mV').reset_index()
    diff=wide['B3_ridge_7_coeff_per_soc'].to_numpy()-wide['B0_condition_isotonic_LUT'].to_numpy()
    obs=float(diff.mean())
    bootstrap=np.empty(2000)
    strata=[grp.index.to_numpy() for _,grp in wide.groupby('Condition_ID',sort=True)]
    for b in range(2000):
        rr=np.concatenate([rng.choice(s,size=len(s),replace=True) for s in strata])
        bootstrap[b]=float(np.mean(diff[rr]))
    lo,hi=np.quantile(bootstrap,[.025,.975])
    boot.append(dict(analysis=scope,validation=val,n_cells=len(diff),
            mean_B3_minus_B0_MAE_mV=obs,pointwise_CI_low_mV=lo,pointwise_CI_high_mV=hi,
            B3_worse_than_B0_fraction=float(np.mean(diff>0)),
            bootstrap_positive_fraction=float(np.mean(bootstrap>0)),
            qualifier='Condition-stratified paired-cell fixed-OOF-error bootstrap; excludes training uncertainty and shared-training-fold dependence'))
bootstrap=pd.DataFrame(boot);bootstrap.to_csv(D/'R2_B3_conditional_paired_bootstrap.csv',index=False)

# Final full-history refit for engineering payload illustration; not an OOF result.
spec=importlib.util.spec_from_file_location('r2','/content/SETA_R123_REPRO_20261025T184824Z/source_code/r2_ridge.py')
r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
with ZipFile(P) as z:
    df=pd.read_csv(z.open('inputs/cell_profiles_nominal_concordant.csv'))
    meta=pd.read_csv(z.open('condition_map.csv')).set_index('Condition_ID')
mat=df[df.SoC.between(0,90)].pivot(index='sample_int',columns='SoC',values='V_cell_median').reindex(columns=np.arange(91)).sort_index()
ids=mat.index.to_numpy(dtype=int)
condition=df[['sample_int','Condition_ID']].drop_duplicates().set_index('sample_int').loc[ids,'Condition_ID'].to_numpy(dtype=int)
X=r2.make_design(meta.loc[condition].reset_index(drop=True));Y=mat.to_numpy(float)
with threadpool_limits(limits=1):
    a,fb,cv,diags=r2.choose_alpha(X,Y,ids,condition)
    mu,sigma=r2.fold_standardization(X)
    Z=np.column_stack([np.ones(X.shape[0]),(X-mu)/sigma])
    valid=np.isfinite(Y);support=valid.sum(axis=0)
    M=np.einsum('ni,nj,nm->mij',Z,Z,valid.astype(float),optimize=True)
    B=np.einsum('ni,nm->mi',Z,np.nan_to_num(Y),optimize=True)
    reg=np.array([1e-10]+[a]*6)
    coeff=np.linalg.solve(M+np.diag(reg)[None,:,:],B[...,None])[...,0] # (91,7)
    pred=np.einsum('ni,mi->nm',Z,coeff)
assert np.allclose(pred, r2.ridge_multialpha(X,Y,X,np.array([a]))[0][0],atol=1e-11,equal_nan=True)
np.savez(D/'R2_B3_final_full_history_fit.npz', coefficients_float32=coeff.astype(np.float32),
         feature_means_float32=mu.astype(np.float32),feature_stds_float32=sigma.astype(np.float32),
         grid=np.arange(91,dtype=np.int16),node_support_counts=support.astype(np.int16),alpha=np.array([a],np.float32))
raw_payload=coeff.size*4+(mu.size+sigma.size)*4
raw_info=dict(alpha=a,alpha_cv=cv,fallback=fb,coeff_shape=list(coeff.shape),
              coefficients_bytes=int(coeff.size*4),scaling_bytes=int((mu.size+sigma.size)*4),
              nominal_model_payload_bytes=int(raw_payload),nominal_model_payload_KiB=raw_payload/1024,
              exclusions='no software code, conditioning map, support mask, grid, or runtime memory',
              model_training='all 180 physical cells in full-history, 3-fold inner CV',
              not_part_of_oof_metrics=True)
(D/'R2_B3_final_model_manifest.json').write_text(json.dumps(raw_info,indent=2,ensure_ascii=False))
print(bootstrap[['analysis','validation','mean_B3_minus_B0_MAE_mV','pointwise_CI_low_mV','pointwise_CI_high_mV','B3_worse_than_B0_fraction']].round(3).to_string(index=False))
print('Final model alpha',a,'coeff bytes',raw_payload,'fallback',fb)
print('Alpha counts',alpha.groupby(['analysis','validation','alpha']).size().to_string())
