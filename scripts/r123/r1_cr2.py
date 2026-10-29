#!/usr/bin/env python3
"""R1 independent audit: reference CR1 reproduction, CR2 model-based adjustment,
Satterthwaite contrast-specific df (working Phi=I), condition-level LOO.
Do not present CR2 working-independence calibration as a fully robust frequentist guarantee
for intrinsically correlated longitudinal cycle data.
"""
import os, json, io, hashlib, zipfile, time
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np, pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.multitest import multipletests

IN=Path('/content/SETA_R123_REPRO_20261025T184824Z/inputs/SoC_band_coverage_ConditionvCodes2.zip')
OUT=Path('/content/SETA_R123_REPRO_20261025T184824Z/results/R1_CR2'); OUT.mkdir(exist_ok=True)
BANDS=('0-20','20-60','60-90'); MODELS=('reference','condition_slopes','cell_balanced','window_52_132')
TVAL={'reference':[108],'condition_slopes':[52,108,132],'cell_balanced':[108],'window_52_132':[108]}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

with zipfile.ZipFile(IN) as z:
    assert z.testzip() is None
    d=pd.read_csv(io.BytesIO(z.read('primary80_cycle_band_data.csv.gz')),compression='gzip')
    specs=pd.read_csv(io.BytesIO(z.read('contrast_specifications.csv')))
    prev=pd.read_csv(io.BytesIO(z.read('contrasts_all_rules.csv')))
    config=json.loads(z.read('run_config.json'))
assert len(d)==237659 and len(specs)==27 and len(prev[prev.rule=='coverage_80'])==81
assert not d.duplicated(['sample_int','Condition_ID','Cycle_Index','SoC_band']).any()
assert d.groupby('sample_int').Condition_ID.nunique().max()==1


def fit(band,model,exclude_c5=False, do_cr2=True):
    data=d[d.SoC_band.eq(band)].copy()
    if model=='window_52_132': data=data[data.Cycle_Index.between(52,132)]
    if exclude_c5:data=data[data.Condition_ID!=5]
    data=data.reset_index(drop=True)
    n=len(data); age=(data.Cycle_Index.to_numpy()-108)/100
    lev=np.sort(data.Condition_ID.unique()); k=len(lev)
    condid=data.Condition_ID.to_numpy(); D=(condid[:,None]==lev).astype(float)
    xx=np.column_stack([D,D*age[:,None],age**2]) if model=='condition_slopes' else np.column_stack([D,age,age**2])
    weights=(1/data.groupby('sample_int').sample_int.transform('size')).to_numpy() if model=='cell_balanced' else np.ones(n)
    x=xx*np.sqrt(weights)[:,None]; y=data.Voltage_band.to_numpy()*np.sqrt(weights)
    gram=x.T@x
    lamb,U=np.linalg.eigh(gram)
    if lamb.min()<=0: raise ValueError(('singular',model,band,lamb.min()))
    invsqrt=U@np.diag(lamb**(-.5))@U.T
    M=invsqrt@invsqrt
    beta=np.linalg.solve(gram,x.T@y)
    res=y-x@beta
    samples=data.sample_int.to_numpy(); ids,pos=np.unique(samples,return_inverse=True)
    nc=len(ids); p=x.shape[1]
    score=np.zeros((nc,p)); groups=[]; Xs=[]; Ys=[]; W=[]
    for j in range(nc):
        ix=np.flatnonzero(pos==j); Xi=x[ix]; e=res[ix]
        score[j]=Xi.T@e
        groups.append(ix)
    CR1_factor=nc/(nc-1)*(n-1)/(n-p)
    GCR=score.T@score
    covcr1=CR1_factor*M@GCR@M
    assert n>p and np.linalg.matrix_rank(gram)==p
    # evaluate all planned contrasts
    Ls=[]; tags=[]; ranks=[]; cellcounts=dict(data.groupby('Condition_ID').sample_int.nunique())
    brk={}
    if model=='condition_slopes':
        for cid,sub in data.groupby('Condition_ID'):
            brk[int(cid)]=sub.groupby('sample_int').Cycle_Index.agg(['min','max'])
    for ti in TVAL[model]:
        for bspec in specs.itertuples(index=False):
            C=np.zeros(p)
            lhs=int(bspec.lhs_condition); rhs=int(bspec.rhs_condition)
            if lhs in lev and rhs in lev:
                li=np.where(lev==lhs)[0][0];ri=np.where(lev==rhs)[0][0]
                C[li]=1;C[ri]=-1
                if model=='condition_slopes': C[k+li]=(ti-108)/100;C[k+ri]=-(ti-108)/100
            cell_l=cellcounts.get(lhs,0);cell_r=cellcounts.get(rhs,0)
            eligibility=cell_l>=2 and cell_r>=2
            if model=='condition_slopes' and eligibility:
                eligibility=all(sum((brk[cid]['min']<=ti)&(brk[cid]['max']>=ti))>=2 for cid in (lhs,rhs))
            Ls.append(C)
            tags.append((ti,bspec.contrast_family,bspec.contrast_id,lhs,rhs,cell_l,cell_r,bool(eligibility)))
    L=np.array(Ls).T  # p x contrasts
    estimate=beta@L
    # Absent-arm contrasts remain unavailable; no artificial zero effect.
    missing_arm=np.array([not (tag[3] in lev and tag[4] in lev) for tag in tags])
    estimate[missing_arm]=np.nan
    cr1_se=np.sqrt(np.einsum('it,ij,jt->t',L,covcr1,L).clip(min=0))
    with np.errstate(divide='ignore',invalid='ignore'):
        cr1_p=2*norm.sf(np.abs(estimate/cr1_se))
    # CR2 adjustment with cluster-internal IID working model.
    # Difference from original clustered CR1 covariance is explicit sensitivity.
    cr2_var=np.full(len(tags),np.nan);cr2_df=np.full(len(tags),np.nan)
    cl_problem=[]
    if do_cr2:
        # For each cell's whitened design, Hii eigenvalues from invsqrt @ X_i'X_i @ invsqrt.
        # Since each cell may lie in a unique saturated condition, Hii has eigenvalue 1.
        # Generalized inverse is only used on estimable residual subspace;
        # perfect-leverage cases flagged, and a C5-excluded check is primary.
        cs=invsqrt@L
        Uscore=np.zeros((nc,len(tags)))
        gList=[]; diagList=[]
        for j,ix in enumerate(groups):
            Xi=x[ix]; ei=res[ix]
            Q=Xi@invsqrt
            Gj=Q.T@Q
            Gj=(Gj+Gj.T)/2
            ev,V=np.linalg.eigh(Gj)
            if ev.max()>1+1e-7: raise RuntimeError(('leverage>1',band,model,j,ev.max()))
            sing=np.sum(ev>=1-1e-8)
            if sing:cl_problem.append({'sample_int':int(ids[j]),'Condition_ID':int(condid[ix[0]]),'n_eigen_ge1':int(sing),'largest_eigen':float(ev.max())})
            safe=ev<1-1e-8
            fs=np.where(safe,1/np.sqrt(np.clip(1-ev,1e-16,None)),0.)
            Hj=V.T@cs
            Fc=V@(fs[:,None]*Hj)
            Uscore[j]=np.einsum('ij,i->j',Fc,Q.T@ei)
            # calculation of K_ij for Satterthwaite under Phi=I
            Vi=V@((ev*fs)[:,None]*Hj)  # p x contrast, each column vi = Gi Fi Sci
            dj=np.sum((ev*fs**2)[:,None]*Hj**2,axis=0) # ||A_i Xi M c||**2
            gList.append(Vi.T)
            diagList.append(dj)
        cr2_var=(Uscore**2).sum(axis=0)
        gl=np.stack(gList,axis=0)  # n_cluster x contrasts x p
        dg=np.stack(diagList,axis=0) # n_cluster x contrasts
        for a in range(len(tags)):
            if not tags[a][-1]:continue
            ga=gl[:,a,:]
            B=-ga@ga.T
            B.flat[::nc+1]+=dg[:,a]
            tr=np.trace(B)
            den=np.sum(B*B)
            cr2_df[a]=tr*tr/den if den>0 and tr>0 else np.nan
    cr2_se=np.sqrt(cr2_var)
    with np.errstate(divide='ignore',invalid='ignore'):
        cr2_p=2*t.sf(np.abs(estimate/cr2_se),cr2_df)
    rows=[]
    for idx,(ti,fam,cid,lhs,rhs,cell_l,cell_r,eligible) in enumerate(tags):
        rows.append(dict(model=model,SoC_band=band,reference_cycle=ti,contrast_family=fam,contrast_id=cid,
                lhs_condition=lhs,rhs_condition=rhs,n_cells_lhs=cell_l,n_cells_rhs=cell_r,
                min_cells=min(cell_l,cell_r),eligible=eligible,
                estimate_mV=estimate[idx]*1000,CR1_SE_mV=cr1_se[idx]*1000,CR1_p_recomputed=cr1_p[idx] if eligible else np.nan,
                CR2_SE_mV=cr2_se[idx]*1000 if eligible else np.nan,CR2_df=cr2_df[idx] if eligible else np.nan,
                CR2_p=cr2_p[idx] if eligible else np.nan,
                CR2_CI_low_mV=(estimate[idx]-t.ppf(.975,cr2_df[idx])*cr2_se[idx])*1000 if eligible and np.isfinite(cr2_df[idx]) else np.nan,
                CR2_CI_high_mV=(estimate[idx]+t.ppf(.975,cr2_df[idx])*cr2_se[idx])*1000 if eligible and np.isfinite(cr2_df[idx]) else np.nan,
                C5_removed=exclude_c5,
            ))
    diag=dict(model=model,SoC_band=band,exclude_C5=exclude_c5,n_obs=n,n_cells=nc,p=p,
            rank=p,CR1_correction=CR1_factor,
            C5_singular_cluster_count=len(cl_problem),C5_singular_sample_ids=';'.join(str(q['sample_int']) for q in cl_problem),
            min_C2_df=float(np.nanmin(cr2_df)) if np.isfinite(cr2_df).any() else None,
            max_C2_df=float(np.nanmax(cr2_df)) if np.isfinite(cr2_df).any() else None)
    return pd.DataFrame(rows),diag,cl_problem,beta


def cluster_leaveone(band,model,cid,loo_sample):
    data=d[d.SoC_band.eq(band)].copy()
    if model=='window_52_132':data=data[data.Cycle_Index.between(52,132)]
    data=data[data.sample_int!=loo_sample]
    lev=np.sort(data.Condition_ID.unique());k=len(lev);age=(data.Cycle_Index.to_numpy()-108)/100
    D=(data.Condition_ID.to_numpy()[:,None]==lev).astype(float)
    X=np.column_stack([D,D*age[:,None],age**2]) if model=='condition_slopes' else np.column_stack([D,age,age**2])
    w=(1/data.groupby('sample_int').sample_int.transform('size')).to_numpy() if model=='cell_balanced' else np.ones(len(data))
    x=X*np.sqrt(w[:,None]);y=data.Voltage_band.to_numpy()*np.sqrt(w)
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    return {int(c):float(beta[i]) for i,c in enumerate(lev)},(beta,k)


def execute():
    started=time.time()
    with threadpool_limits(limits=1):
        allres=[];alldiags=[];anoms=[]
        for band in BANDS:
            for model in MODELS:
                for exclude in [False,True]:
                    st=time.time();res,diag,prob,beta=fit(band,model,exclude)
                    allres.append(res);alldiags.append(diag)
                    anoms.extend(dict(model=model,SoC_band=band,exclude_C5=exclude,**q) for q in prob)
                    print('fit',band,model,'exclude',exclude,'cells',diag['n_cells'],'badlev',len(prob),'time',round(time.time()-st,2),flush=True)
        table=pd.concat(allres,ignore_index=True)
        # 81-item BH by band*spec per model & time for each C5 variant
        table['CR2_q_BH81']=np.nan;table['CR1_q_BH81_recomputed']=np.nan
        for keys,inds in table.groupby(['model','reference_cycle','C5_removed']).groups.items():
            group=table.loc[list(inds)]
            assert len(group)==81,(keys,len(group))
            for src,dest in [('CR2_p','CR2_q_BH81'),('CR1_p_recomputed','CR1_q_BH81_recomputed')]:
                p=np.where(group[src].notna(),group[src],1)
                q=multipletests(p,method='fdr_bh')[1]
                table.loc[group.index,dest]=np.where(group[src].notna(),q,np.nan)
        # In original models compare CR1 exactly to source for reference
        orig=table[(table.model=='reference') & (~table.C5_removed) & (table.reference_cycle==108)].merge(
            prev[prev.rule=='coverage_80'][['SoC_band','contrast_id','delta_age_adjusted_mV','p_value','SE_mV']],on=['SoC_band','contrast_id'],validate='one_to_one')
        orig['delta_diff_mV']=orig.estimate_mV-orig.delta_age_adjusted_mV
        orig['SE_diff_mV']=orig.CR1_SE_mV-orig.SE_mV
        orig.to_csv(OUT/'R1_reference_reproduction_check.csv',index=False)
        maxe=orig.delta_diff_mV.abs().max();maxse=orig.SE_diff_mV.abs().max()
        print('CR1 original reproduction max delta and SE difference',maxe,maxse,flush=True)
        if maxe>1e-5 or maxse>1e-4:raise RuntimeError('Reference benchmark failed')
        # LOO of scarce conditions on 108 cycle, 75 eligible, across all four models
        lrows=[]
        for band in BANDS:
            for model in MODELS:
                source=d[d.SoC_band==band]
                if model=='window_52_132':source=source[source.Cycle_Index.between(52,132)]
                for cid in [23,24]:
                    for sid in source.loc[source.Condition_ID==cid,'sample_int'].unique():
                        coefficients,(beta,k)=cluster_leaveone(band,model,cid,sid)
                        # at t=108 only, both arms need >=2 cells after exclusion
                        for spec in specs.itertuples(index=False):
                            if cid not in (int(spec.lhs_condition),int(spec.rhs_condition)):continue
                            lhs=int(spec.lhs_condition);rhs=int(spec.rhs_condition)
                            if lhs not in coefficients or rhs not in coefficients:continue
                            est=(coefficients[lhs]-coefficients[rhs])*1000
                            if model=='condition_slopes':
                                # 108 -> age_c=0, no slope terms for contrast
                                pass
                            lrows.append(dict(model=model,SoC_band=band,omitted_sample=int(sid),omitted_condition=cid,
                                    contrast_id=spec.contrast_id,contrast_family=spec.contrast_family,
                                    estimate_loo_mV=est,estimate_loo_sign=np.sign(est)))
        loo=pd.DataFrame(lrows)
        base=table[(~table.C5_removed)&(table.reference_cycle==108)][['model','SoC_band','contrast_id','estimate_mV']]
        loo=loo.merge(base,on=['model','SoC_band','contrast_id'],how='left',validate='many_to_one')
        loo['changed_mV']=loo.estimate_loo_mV-loo.estimate_mV
        loo['sign_changed']=np.sign(loo.estimate_loo_mV)!=np.sign(loo.estimate_mV)
        loo.to_csv(OUT/'R1_small_cell_leave_one_out.csv',index=False)
        table.to_csv(OUT/'R1_CR2_all_models_972.csv',index=False)
        pd.DataFrame(alldiags).to_csv(OUT/'R1_model_diagnostics.csv',index=False)
        pd.DataFrame(anoms).to_csv(OUT/'R1_perfect_leverage_clusters.csv',index=False)
        stage=dict(source_zip_name=IN.name,sha256=sha(IN),source_processed_sha256=config['source_sha256'],
                   fit_models=list(MODELS),bands=list(BANDS),working_covariance='identity on weighted observations; longitudinal within-cell independence is a WORKING assumption, NOT verified',
                   adjustment='CR2 generalized inverse on perfectly leveraged singular cells; primary sensitivity C5_removed=True',
                   sat_df='trace(K)^2 / trace(K^2) with K_ij = d_i delta_ij - v_i.v_j',
                   BH_family=81,models_original_and_sensitivity='two fit variants per band/model',
                   n_raw_rows=len(d),CR1_ref_max_effect_repro_error_mV=float(maxe),
                   CR1_ref_max_se_repro_error_mV=float(maxse),
                   time_elapsed_sec=time.time()-started)
        (OUT/'R1_run_config.json').write_text(json.dumps(stage,indent=2),encoding='utf8')
        print('RESULT SUMMARY 108')
        for key,g in table[table.reference_cycle==108].groupby(['model','C5_removed']):
            ok=g[g.eligible]; aggressive=ok[ok.contrast_family.isin(['rate_2C_minus_1C','rate_0.7C_minus_2C'])]
            print(key,'eligible',len(ok),'CR2_FDR',sum(ok.CR2_q_BH81<.05),'CR1_FDR',sum(ok.CR1_q_BH81_recomputed<.05),
                  '2C_CR2',sum(aggressive.CR2_q_BH81<.05),'of',len(aggressive),
                  'df min median',np.nanmin(ok.CR2_df),np.nanmedian(ok.CR2_df))
        print('LOO rows',len(loo),'sign changes',int(loo.sign_changed.sum()), 'max abs effect shift',loo.changed_mV.abs().max())

if __name__=='__main__':execute()
