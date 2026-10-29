"""Byte integrity and manuscript-result consistency of archived R1/R2/R3 output.

Does NOT rerun the scientific pipeline or claim that missing input ZIPs are present.
"""
import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'supplementary' / 'R123'


class R123ArtifactTests(unittest.TestCase):
    def test_supplement_fingerprints(self):
        manifest = json.loads((REPO / 'data_manifest/r123_payload_sha256.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['files']), 23)
        for rel, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((REPO / rel).read_bytes()).hexdigest(), expected, rel)
        orig = json.loads((R / 'REPRO_MANIFEST.json').read_text(encoding='utf-8'))
        for name, expected in orig['source_code_sha256'].items():
            self.assertEqual(hashlib.sha256((REPO / 'scripts/r123' / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual({x['name'] for x in orig['inputs']},
                         {'SoC_unified_ConditionvCodes2.zip', 'SoC_band_coverage_ConditionvCodes2.zip'})

    def test_r1_cr2_counts_and_contrast_family(self):
        d = pd.read_csv(R / 'R1_summary_recomputed.csv').set_index('model')
        self.assertEqual(len(d), 4)
        self.assertEqual(set(d['eligible'].astype(int)), {75})
        self.assertEqual({k:int(d.loc[k,'CR2_BH_supported']) for k in
                         ['reference','condition_slopes','cell_balanced','window_52_132']},
                         {'reference':66,'condition_slopes':68,'cell_balanced':67,'window_52_132':73})
        self.assertEqual(set(d['2C_CR2_supported'].astype(int)), {42})
        raw = pd.read_csv(R / 'R1_CR2_all_models_972.csv')
        self.assertEqual(len(raw), 972)
        self.assertEqual(set(raw['contrast_family'].dropna()) &
                         {'rate_2C_minus_1C','rate_0.7C_minus_2C'},
                         {'rate_2C_minus_1C','rate_0.7C_minus_2C'})

    def test_r2_matched_lut_better_than_ridge(self):
        d = pd.read_csv(R / 'R2_compact_comparison.csv')
        self.assertEqual(len(d), 6)
        self.assertTrue((d['ridge_minus_lut_mV']>0).all())
        self.assertIn('full_history', set(d['analysis']))
        full = d[(d['analysis']=='full_history') & (d['validation']=='LOCO')].iloc[0]
        self.assertAlmostEqual(full['B0_condition_isotonic_LUT'],5.240436852444727,places=5)
        self.assertAlmostEqual(full['B3_ridge_7_coeff_per_soc'],25.58,places=2)
        temp = d[(d['analysis']=='full_history') & (d['validation']=='temperature_holdout')].iloc[0]
        self.assertAlmostEqual(temp['B0_condition_isotonic_LUT'],32.141306,places=5)
        axes = pd.read_csv(R / 'R2_axis_controls_summary.csv')
        subset = axes[(axes['analysis']=='full_history') & (axes['validation']=='temperature_holdout')]
        self.assertEqual(len(subset),2)
        self.assertLess(subset.loc[subset['model']!='B0_original','MAE_mV'].iloc[0],
                        subset.loc[subset['model']=='B0_original','MAE_mV'].iloc[0])

    def test_r3_common_node_storage_tradeoff(self):
        d = pd.read_csv(R / 'R3_LOCO_compact_comparison.csv').set_index('model')
        b = pd.read_csv(R / 'R3_storage_budget.csv').set_index('model')
        self.assertEqual(len(d),13)
        self.assertEqual(len(b),13)
        self.assertAlmostEqual(float(d.loc['B0_original','MAE_mV']),4.260269,places=5)
        self.assertAlmostEqual(float(d.loc['step1_float16_linear','MAE_mV']),4.301049,places=5)
        self.assertAlmostEqual(float(d.loc['step2_float32_pchip','MAE_mV']),4.3065,places=3)
        self.assertAlmostEqual(float(b.loc['step1_float16_linear','payload_KiB']),4.265625,places=6)
        self.assertAlmostEqual(float(b.loc['step2_float32_pchip','payload_KiB']),4.3125,places=6)
        self.assertEqual(set(b['excludes_support_masks'].astype(str).str.lower()),{'true'})


if __name__ == '__main__':
    unittest.main()
