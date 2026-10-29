"""Secondary check: physical-cell adjusted means, Welch independent-cell t tests.
Distinct estimand/weighting from primary cycle-weighted CR1/CR2.
Age adjustment coefficients estimated globally and treated as fixed in this check;
thus NOT a replacement for uncertainty-aware CR2 primary inference.
"""
from pathlib import Path
import io,zipfile
import numpy as np,pandas as pd
from scipy.stats import t as studentt
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
root=Path('/content/SETA_R123_REPRO_20261025T184824Z/results/R1_CR2')
with zipfile.ZipFile('/content/SETA_R123_REPRO_20261025T184824Z/inputs/SoC_band_coverage_ConditionvCodes2.zip') as z:
 d=pd.read_csv(io.BytesIO(z.read('primary80_cycle_band_data.csv.gz')),compression='gzip')
 specs=pd.read_csv(io.BytesIO(z.read('contrast_specifications.csv')))
rows=[]
with threadpool_limits(limits=1):
 for band in ['0-20','20-60','60-90']:
  x=d[d.SoC_band==band].copy(); level=sorted(x.Condition_ID.unique()); D=(x.Condition_ID.to_numpy()[:,None]==level).astype(float)
  a=(x.Cycle_Index.to_numpy()-108)/100
  X=np.column_stack([D,a,a*a]);beta=np.linalg.lstsq(X,x.Voltage_band.to_numpy(),rcond=None)[0]
  x['adjusted']=x.Voltage_band.to_numpy()-beta[-2]*a-beta[-1]*a*a
  med=x.groupby(['Condition_ID','sample_int']).adjusted.mean().reset_index()
  for spec in specs.itertuples(index=False):
   lhs=med.loc[med.Condition_ID==spec.lhs_condition,'adjusted'].to_numpy();rhs=med.loc[med.Condition_ID==spec.rhs_condition,'adjusted'].to_numpy()
   if min(len(lhs),len(rhs))<2:rows.append(dict(SoC_band=band,contrast_id=spec.contrast_id,contrast_family=spec.contrast_family, n_lhs=len(lhs),n_rhs=len(rhs), delta_mV=np.nan,welch_df=np.nan,welch_p=np.nan));continue
   delta=lhs.mean()-rhs.mean();v1=lhs.var(ddof=1)/len(lhs);v2=rhs.var(ddof=1)/len(rhs)
   se=np.sqrt(v1+v2);df=(v1+v2)**2/(v1*v1/(len(lhs)-1)+v2*v2/(len(rhs)-1))
   p=2*studentt.sf(abs(delta/se),df)
   rows.append(dict(SoC_band=band,contrast_id=spec.contrast_id,contrast_family=spec.contrast_family,n_lhs=len(lhs),n_rhs=len(rhs),delta_mV=delta*1e3,welch_df=df,welch_p=p))
out=pd.DataFrame(rows);out['BH_q81']=np.where(out.welch_p.notna(),multipletests(out.welch_p.fillna(1),method='fdr_bh')[1],np.nan)
out.to_csv(root/'R1_cell_level_Welch_secondary.csv',index=False)
agg=out[out.contrast_family.isin(['rate_2C_minus_1C','rate_0.7C_minus_2C']) & out.welch_p.notna()]
print('2C associated',len(agg),'Welch BH supported',sum(agg.BH_q81<.05),'df min median',agg.welch_df.min(),agg.welch_df.median())
print('3cell contrasts',out[(out.n_lhs==3)|(out.n_rhs==3)][['SoC_band','contrast_id','delta_mV','welch_df','BH_q81']].to_string(index=False))
