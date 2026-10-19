import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]

class ReproArtifactTests(unittest.TestCase):
    def test_stage_source_cells_exact(self):
        notebook=json.loads((REPO/'notebooks/01_to_05_scientific_rebuild.ipynb').read_text())
        provenance=json.loads((REPO/'docs/CODE_PROVENANCE.json').read_text())
        setup=''.join(notebook['cells'][2]['source'])
        for name,meta in provenance['scientific_stages'].items():
            path=REPO/'scripts/analyses'/name
            whole=path.read_text(encoding='utf8')
            body=''.join(notebook['cells'][meta['original_notebook_cell']]['source'])
            self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),meta['scientific_cell_sha256'])
            self.assertEqual(hashlib.sha256(setup.encode()).hexdigest(),meta['shared_setup_sha256'])
            self.assertIn(setup,whole)
            self.assertTrue(whole.endswith(body),name)
        snapshot=(REPO/'docs/source_snapshot/revisedcalce_colab_export.txt').read_bytes()
        self.assertEqual(hashlib.sha256(snapshot).hexdigest(),provenance['snapshot']['sha256'])

    def test_publication_artifact_hashes(self):
        d=json.loads((REPO/'data_manifest/audited_repo_payload_sha256.json').read_text())
        self.assertEqual(len(d['files']),17)
        for path,h in d['files'].items():
            blob=(REPO/path).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(),h,path)

    def test_42_contrast_check(self):
        d=pd.read_csv(REPO/'supplementary/contrast_results.csv')
        m=d[d['reference_cycle'].eq(108) & d['model'].isin(['reference','condition_slopes','cell_balanced','window_52_132']) & d['contrast_family'].isin(['rate_2C_minus_1C','rate_0.7C_minus_2C'])].copy()
        m['eligible']=m['inference_eligible'].astype(str).str.lower().eq('true')
        m['ci']=(m['bootstrap_CI_low_V']>0)|(m['bootstrap_CI_high_V']<0)
        z=m.groupby(['SoC_band','contrast_id'],dropna=False).agg(n=('model','nunique'),eligible=('eligible','all'),signs=('point_sign','nunique'),ci=('ci','all'))
        self.assertEqual(len(z),48)
        self.assertEqual(int((z['n']==4).sum()),48)
        self.assertEqual(int((z['eligible'] & (z['signs']==1) & z['ci']).sum()),42)

    def test_main_mae_and_gain(self):
        d=pd.read_csv(REPO/'supplementary/representation_common_five.csv')
        d=d[(d['comparison']=='common_five')&(d['band']=='all_0_90_inclusive')]
        self.assertEqual(len(d),30)
        expected={'LOCO':(103.688425,4.353667,95.80),
                  'rate_holdout':(99.451909,18.041400,81.86),
                  'temperature_holdout':(89.328318,33.189926,62.85)}
        for v,(global_mae,cond_mae,gain) in expected.items():
            t=d[(d['analysis']=='broad_52_132')&(d['validation']==v)]
            g=float(t.loc[t.model=='global_iso','MAE_mV'].iloc[0]);c=float(t.loc[t.model=='condition_iso','MAE_mV'].iloc[0])
            self.assertAlmostEqual(g,global_mae,places=5)
            self.assertAlmostEqual(c,cond_mae,places=5)
            self.assertEqual(round((g-c)/g*100,2),gain)

    def test_lut_matrix_identity_and_monotonicity(self):
        matrix=np.load(REPO/'lut/final_LUT_0_90_float32.npy',allow_pickle=False)
        self.assertEqual(matrix.shape,(24,91));self.assertEqual(matrix.dtype,np.float32)
        self.assertEqual(int(np.isfinite(matrix).sum()),2184)
        self.assertTrue((np.diff(matrix.astype(float),axis=1)>=-1e-6).all())
        c=pd.read_csv(REPO/'lut/final_LUT_0_90_matrix.csv').set_index('Condition_ID')
        self.assertTrue(np.allclose(matrix,c.to_numpy(dtype=float),atol=1e-6,rtol=0))

if __name__=='__main__':
    unittest.main()
