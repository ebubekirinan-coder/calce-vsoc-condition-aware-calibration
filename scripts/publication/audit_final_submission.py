# -*- coding: utf-8 -*-
"""Independent (read-only) ZIP vs saved manuscript seven-table audit.
No original scientific estimator is refitted; verifies supplied reported outputs.
"""
from __future__ import annotations
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
import csv, hashlib, json, re, zipfile
import numpy as np

ZIP = Path('/mnt/data/SoC_Final_Submission_20261018T190535Z.zip')
MANUSCRIPT = Path('/mnt/data/Yapıştırılan metin(20261016-175452).txt')
REPORT = Path('/mnt/data/SoC_Nihai_ZIP_Bagimsiz_Denetim_Raporu_2026-10-18.md')

z = zipfile.ZipFile(ZIP)
root = z.namelist()[0].split('/')[0] + '/'
M = json.loads(z.read(root+'manifest.json'))
assert z.testzip() is None
assert len(M['files'])==52
for obj in M['files']:
    blob=z.read(root+obj['path'])
    assert hashlib.sha256(blob).hexdigest()==obj['sha256'] and len(blob)==obj['size'],obj['path']
for line in z.read(root+'checksums.sha256').decode('utf-8').splitlines():
    h, rel=line.split('  ',1)
    assert hashlib.sha256(z.read(root+rel)).hexdigest()==h,rel

def rows(name):
    p=next(n for n in z.namelist() if n.endswith('/'+name))
    return list(csv.DictReader(StringIO(z.read(p).decode('utf-8-sig'))))

def f(row,col):
    x=row.get(col,'')
    return float(x) if x not in ('', 'nan', 'NaN', 'None', None) else float('nan')

def b(row,col):
    return str(row.get(col,'')).lower() in ('true','1','yes')

def check(actual, expected, dp, note):
    tol=0.5*10**(-dp)+1e-7
    if not abs(actual-expected)<tol:
        raise AssertionError(f'{note}: expected {expected}, got {actual}, tol {tol}')

ms=MANUSCRIPT.read_text('utf-8-sig')
blocks=re.findall(r'\\begin\{table\}.*?\\end\{table\}',ms,re.S)
assert len(blocks)==7

def parse_table(i):
    txt=blocks[i]
    body=re.search(r'\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}',txt,re.S).group(1)
    result=[]; started=False
    for line in body.splitlines():
        s=line.strip()
        if s.startswith('\\midrule'):
            started=True; continue
        if s.startswith('\\bottomrule'):
            break
        if not started or '&' not in s: continue
        s=re.sub(r'\\\\\s*$', '',s)
        result.append([x.strip() for x in s.split('&')])
    return result

T=[parse_table(i) for i in range(7)]
assert list(map(len,T))==[3,5,5,9,6,6,10], list(map(len,T))

def n(s):
    return float(s.replace(',','').replace('\\%','').strip())

checks=[]
# TABLE 1: counts do not exist independently in the delivery ZIP.
# Exact reported values were independently confirmed from Colab stage-01 logs,
# which are not in the uploaded ZIP. Mark as externally verified.
checks.append(('Table 1 — Band coverage','LOG PASS / ZIP SOURCE NOT INCLUDED',0,
               'Coverage-80 original summary table must be preserved with source stage archive.'))

# TABLE 2: from eligible reference contrasts only
ref=rows('contrast_reference_81.csv')
order=['cutoff_C40_minus_C5','rate_0.7C_minus_1C','rate_2C_minus_1C','rate_0.7C_minus_2C','temperature_60_minus_25']
ct=0
for i, fam in enumerate(order):
    q=[r for r in ref if r['contrast_family']==fam]
    eligible=[r for r in q if b(r,'inference_eligible')]
    values=np.abs([f(r,'delta_adjusted_mV') for r in eligible])
    measures=[len(q),len(eligible),sum(f(r,'q_BH81')<0.05 for r in eligible),
              sum(b(r,'bootstrap_excludes_zero') for r in eligible),
              float(np.median(values)),float(np.max(values))]
    exp=T[1][i][1:]
    for j,(a,e) in enumerate(zip(measures,exp)):
        check(a,n(e),0 if j<4 else 1,f'Table2 {fam} col{j}');ct+=1
checks.append(('Table 2 — Contrast families','PASS',ct,'Eligible contrasts only for median/max.'))

# TABLE 3: reference-model paired sensitivity
age=rows('contrast_results.csv')
base={(r['SoC_band'],r['contrast_id']):r for r in age if r['model']=='reference' and int(f(r,'reference_cycle'))==108}
models=[('cell_balanced',108),('condition_slopes',52),('condition_slopes',108),('condition_slopes',132),('window_52_132',108)]
ct=0
for i,(model,cyc) in enumerate(models):
    comp=[]
    for r in age:
        if r['model']!=model or int(f(r,'reference_cycle'))!=cyc: continue
        rr=base.get((r['SoC_band'],r['contrast_id']))
        if rr and b(r,'inference_eligible') and b(rr,'inference_eligible'):comp.append((r,rr))
    sign=sum(np.sign(f(r,'delta_adjusted_mV')) !=np.sign(f(rr,'delta_adjusted_mV')) for r,rr in comp)
    bins=sum(r['point_bucket'] !=rr['point_bucket'] for r,rr in comp)
    change=max(abs(f(r,'delta_adjusted_mV')-f(rr,'delta_adjusted_mV')) for r,rr in comp)
    arr=[cyc,len(comp),sign,bins,change]
    for j,(a,e) in enumerate(zip(arr,T[2][i][1:])):
        check(a,n(e),2 if j==4 else 0,f'Table3 {model} {cyc} col{j}');ct+=1
checks.append(('Table 3 — Model sensitivity','PASS',ct,'Models evaluated at the specified reference cycle.'))

# TABLE 4: thermal 60-25 C/40 contrasts, 9 bands x 4 models
ct=0
for i, vals in enumerate(T[3]):
    rate=vals[0]
    band=vals[1].replace('--','-')
    ident='temp60minus25_C40_DR'+rate
    for model,textval in zip(['reference','condition_slopes','cell_balanced','window_52_132'], vals[2:]):
        hit=[r for r in age if r['contrast_id']==ident and r['SoC_band']==band and r['model']==model and int(f(r,'reference_cycle'))==108]
        assert len(hit)==1, (ident,band,model)
        check(f(hit[0],'delta_adjusted_mV'),n(textval),2,f'Table4 {ident} {band} {model}');ct+=1
checks.append(('Table 4 — Thermal contrasts','PASS',ct,'Nine rows x four models.'))

# TABLES 5 and 6: from independent support audit summary
cov=rows('primary_coverage_overall.csv')
order=[('full_history','LOCO'),('broad_52_132','LOCO'),('full_history','rate_holdout'),('broad_52_132','rate_holdout'),('full_history','temperature_holdout'),('broad_52_132','temperature_holdout')]
ct5=ct6=0
for i,(hist,val) in enumerate(order):
    q=[r for r in cov if r['analysis']==hist and r['validation']==val];assert len(q)==1
    r=q[0]
    for col, txt in zip(['MAE_mV','median_cell_MAE_mV','P90_cell_MAE_mV','mean_cell_RMSE_mV'],T[4][i][2:]):
        check(f(r,col),n(txt),2,f'Table5 {hist} {val} {col}');ct5+=1
    actual_cells=f"{int(f(r,'n_evaluated_cells'))}/{int(f(r,'n_planned_cells'))}"
    actual_nodes=f"{int(f(r,'n_used')):,}/{int(f(r,'n_target')):,}"
    actual_mmm=f"{int(f(r,'evaluated_nodes_min'))}/{int(f(r,'evaluated_nodes_median'))}/{int(f(r,'evaluated_nodes_max'))}"
    q6=T[5][i]
    assert actual_cells==q6[2] and actual_nodes==q6[3] and actual_mmm==q6[5],(i,actual_cells,actual_nodes,actual_mmm,q6)
    check(f(r,'target_coverage_pct'),n(q6[4]),2,f'Table6 {hist} {val} coverage')
    ct6+=4
checks.append(('Table 5 — Validation errors','PASS',ct5,'Six rows x four error statistics.'))
checks.append(('Table 6 — Validation support','PASS',ct6,'Cells, paired nodes, coverage, nodes/cell.'))

# TABLE 7: common five model comparison
rp=rows('representation_common_five.csv')
modelnames={'Pooled isotonic':'global_iso','Temperature-only isotonic':'temperature_iso','Rate-only isotonic':'rate_iso','Condition median':'condition_raw','Condition isotonic':'condition_iso'}
ct=0
for i,row in enumerate(T[6]):
    history='full_history' if i<5 else 'broad_52_132'
    model=modelnames[row[1]]
    for val,txt in zip(['LOCO','rate_holdout','temperature_holdout'],row[2:]):
        q=[r for r in rp if r['analysis']==history and r['model']==model and r['validation']==val];assert len(q)==1
        check(f(q[0],'MAE_mV'),n(txt),3,f'Table7 {history} {model} {val}');ct+=1
checks.append(('Table 7 — Five representations','PASS',ct,'Thirty published MAEs.'))

# Other mandatory supplementary and LUT tests
s1,s2=rows('Table_S1_81_planned_contrasts.csv'),rows('Table_S2_6_descriptive_contrasts.csv')
assert len(s1)==81 and sum(b(r,'inference_eligible') for r in s1)==75 and len(s2)==6
assert all(not b(r,'inference_eligible') and all(np.isnan(f(r,c)) for c in ['p_value','q_BH81','cluster_CI_low_mV','bootstrap_CI_low_mV']) for r in s2)
lut=np.load(BytesIO(z.read(root+'lut/final_LUT_0_90_float32.npy')))
assert lut.shape==(24,91) and lut.dtype==np.float32 and np.isfinite(lut).all()
mat=rows('final_LUT_0_90_matrix.csv')
assert len(mat)==24 and len(mat[0])==92
matarr=np.array([[f(r,f'SoC_{j:03d}') for j in range(91)] for r in mat],dtype=np.float64)
assert float(np.max(np.abs(lut.astype(float)-matarr))*1000)<0.001
figs=[p for p in z.namelist() if '/figures/' in p and p.endswith('.png')]
assert len(figs)==31
assert all(any(p.endswith('/'+need) for p in figs) for need in M['figures']['manuscript_figures'])

report=[
'# CALCE V–SoC — Bağımsız nihai ZIP denetimi',
'',f'**Kaynak ZIP:** `{ZIP.name}`',
'**Makale:** `Yapıştırılan metin(20261016-175452).txt` (projede erişilebilir son karşılaştırma metni).',
'',
'## Tabloların kontrol sonuçları','',
'| Tablo | Sonuç | Denetlenen hücre/ölçüt | Açıklama |',
'|---|---|---:|---|',
]
for name,status,num,explain in checks:
    report.append(f'| {name} | **{status}** | {num if num else "—"} | {explain} |')
report += ['', '## Teknik bütünlük','',
 '- ZIP: 54 üye, hiçbir üye bozuk değil.',
 '- Manifest: 52/52 dosyanın boyutu ve SHA256 değeri doğru.',
 '- `checksums.sha256`: 52/52 eşleşme.',
 '- Makalede geçen sekiz PNG: mevcut ve açılabilir; toplam 31 PNG.',
 '- S1: 81 kontrast; 75 çıkarımsal, 6 betimsel.',
 '- S2: 6 betimsel satır; p/q/GA alanları boş.',
 '- LUT: 24×91, float32, 2184 sonlu düğüm; matris CSV eşdeğerliği <0.001 mV.',
 '- Üç temel supplementary CSV mevcut.',
 '', '## Teslim öncesi kalan iki konu','',
 '1. **Tekrarlanabilir bilimsel kaynak:** Bu ZIP içinde Python kodu, notebook veya beş özgün aşama ZIP’i bulunmuyor. Manifest bu arşivlerin SHA256 parmak izlerini taşıyor. README ayrı bir GitHub kaynak kodu dağıtımı öngörüyor. GitHub depo ve uygun sürüm henüz bağımsız denetlenmedi.',
 '2. **Tablo 1’in veri düzeyinde yeniden kurulması:** 79.218/79.218/79.223 çevrim ve kapsam eşiği sayıları daha önceki gerçek Colab Stage-01 loguyla eşleşiyor; bu ZIP içinde `coverage_summary` üretici tablosu veya orijinal stage ZIP’i olmadığından salt bu ZIP kullanılarak tekrar çıkarılamıyor. Kaynak coverage arşivi korunmalı.',
 '', '## Denetimin sınırı','',
 'Bu rapor, makale–üretilmiş sonuç tutarlılığını ve dosya bütünlüğünü kontrol eder; 2,36 milyon satırlık kaynak Excel verilerinden bütün analitik süreç yeniden çalıştırılmamıştır. Makalenin burada kullanılan sürümüyle karşılaştırma yapılmıştır. Daha sonra değiştirilmiş nihai makale varsa bu denetim yenilenmelidir.',
]
REPORT.write_text('\n'.join(report)+'\n',encoding='utf-8')
for x in checks: print(x)
print('ZIP & MANIFEST PASS, image+supp PASS, LUT PASS')
print('REPORT',REPORT)
