"""R2 independent interpolation-axis controls; no ridge model in this module.
All training anchors are fitted on original condition-specific physical-cell
profiles; withheld-level cells are absent from the anchor conditions.
"""
from pathlib import Path
from zipfile import ZipFile
import io, hashlib, json, warnings
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression
P=Path('/content/SETA_R123_REPRO_20261025T184824Z/inputs/SoC_unified_ConditionvCodes2.zip')
OUT=Path('/content/SETA_R123_REPRO_20261025T184824Z/results/R2_Axis');OUT.mkdir(exist_ok=True)
GRID=np.arange(101)
def fit(A):
 a=np.asarray(A,float).reshape(-1,101)
 n=np.isfinite(a).sum(axis=0); good=n>0
 pred=np.full(101,np.nan)
 if good.sum()>=2:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   mean=np.nanmedian(a,axis=0)
  model=IsotonicRegression(increasing=True,out_of_bounds='nan')
  pred[good]=model.fit_transform(GRID[good],mean[good],sample_weight=n[good])
 return pred
with ZipFile(P) as z:
    old=pd.read_csv(z.open('validation_node_predictions.csv'))
    old=old[(old.method=='unified')&(old.validation.isin(['rate_holdout','temperature_holdout']))&(old.SoC<=90)].copy()
    profiles={}
    for key,name in [('full_history','inputs/cell_profiles_nominal_concordant.csv'),('broad_52_132','inputs/broad_52_132_cell_profiles.csv')]:
      d=pd.read_csv(z.open(name)); d=d.rename(columns={'V_cell_median':'Voltage'})
      profiles[key]={int(cid):fit(g.pivot(index='sample_int',columns='SoC',values='Voltage').reindex(columns=GRID).to_numpy()) for cid,g in d.groupby('Condition_ID')}
meta={1+6*t+2*r+k:(T,cutoff,rate) for t,T in enumerate((10,25,45,60)) for r,rate in enumerate((0.7,1.,2.)) for k,cutoff in enumerate(('C/5','C/40'))}
records=[]
for key,g in old.groupby(['analysis','validation','Condition_ID','sample_int'],sort=False):
 analysis,validation,cid,sid=key;g=g.sort_values('SoC')
 cida,cidb=int(g.train_condition_a.iloc[0]),int(g.train_condition_b.iloc[0])
 T,_,r=meta[cid];Ta,_,ra=meta[cida];Tb,_,rb=meta[cidb]
 if validation=='rate_holdout':
  w=(np.log(r)-np.log(ra))/(np.log(rb)-np.log(ra))
  name='B1_LUT_log_rate'
 else:
  w=(1/(T+273.15)-1/(Ta+273.15))/(1/(Tb+273.15)-1/(Ta+273.15))
  name='B2_LUT_inverse_temperature'
 ca,cb=profiles[analysis][cida][:91],profiles[analysis][cidb][:91]
 y=g.target_V.to_numpy(float)
 original=g.prediction_V.to_numpy(float)
 new=(1-w)*ca+w*cb
 # verify archived baseline was linear weights
 wo=float(g.weight_b.iloc[0])
 assert np.allclose((1-wo)*ca+wo*cb,original,atol=1e-12,equal_nan=True)
 assert np.array_equal(np.isfinite(new),np.isfinite(original))
 for label,pred in [('B0_original',original),(name,new)]:
  mask=np.isfinite(y)&np.isfinite(pred)
  diff=(pred[mask]-y[mask])*1000
  records.append(dict(analysis=analysis,validation=validation,Condition_ID=int(cid),sample_int=int(sid),model=label,w=w,n_target=int(np.isfinite(y).sum()),n_paired=int(mask.sum()),n_evaluated=int(mask.sum()>=5),MAE_mV=np.mean(abs(diff)) if mask.sum()>=5 else np.nan,RMSE_mV=np.sqrt(np.mean(diff**2)) if mask.sum()>=5 else np.nan,bias_mV=np.mean(diff) if mask.sum()>=5 else np.nan))
df=pd.DataFrame(records);df.to_csv(OUT/'R2_axis_controls_cell_metrics.csv',index=False)
s=df.groupby(['analysis','validation','model']).agg(n_cells=('n_evaluated','sum'),n_paired=('n_paired','sum'),n_target=('n_target','sum'),MAE_mV=('MAE_mV','mean'),RMSE_mV=('RMSE_mV','mean'),bias_mV=('bias_mV','mean')).reset_index()
s['coverage']=s.n_paired/s.n_target
s.to_csv(OUT/'R2_axis_controls_summary.csv',index=False)
(OUT/'R2_axis_controls_manifest.json').write_text(json.dumps(dict(source=P.name,sha256=hashlib.sha256(P.read_bytes()).hexdigest(),scope='B0/B1/B2 only; NO ridge B3/B4 experiment',note='Hold-out anchor profiles fitted on source physical-cell profiles; no training target leakage; predictions share identical finite support.'),indent=2),encoding='utf-8')
print(s.to_string(index=False))
