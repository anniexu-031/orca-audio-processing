"""Data contracts and training guardrails, independent of audio/network access."""
import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from orca_s04.pipeline import fetch_object, folds_by_date, load_features, load_manifest
from orca_s04.training import train_baseline
from orca_s04.audio_features import describe_call


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.data = load_manifest(ROOT/'data/manifests/s04_calls.csv',only_s04=True)

    def test_real_subset_and_date_folds(self):
        self.assertEqual(len(self.data),137)
        self.assertEqual(self.data.groupby('recording_date').size().to_dict(),
            {'2022-07-10':39,'2022-07-11':14,'2022-07-13':6,'2022-07-25':52,'2022-07-29':26})
        folds = folds_by_date(self.data)
        tested = []
        for f in folds:
            self.assertFalse(set(f['train_call_ids']) & set(f['test_call_ids']))
            self.assertNotIn(f['heldout_date'],f['train_dates'])
            self.assertEqual(set(f['train_call_ids'])|set(f['test_call_ids']),set(self.data.call_id))
            tested += f['test_call_ids']
        self.assertEqual(len(tested),len(set(tested)))

    def test_included_features_match_by_key(self):
        features = load_features(ROOT/'data/features/s04_features.csv',self.data)
        self.assertEqual(features.index.tolist(),self.data.call_id.tolist())
        self.assertTrue(np.allclose(features.duration_s,self.data.end_s-self.data.start_s))
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_mismatched_features_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'features.csv'
            pd.read_csv(ROOT/'data/features/s04_features.csv').iloc[:-1].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'match exactly'):
                load_features(path,self.data)



    def test_modified_identity_or_date_fails(self):
        with tempfile.TemporaryDirectory() as td:
            data = self.data.copy(); data.loc[0,'recording_date']='2023-01-01'
            path = Path(td)/'bad.csv'; data.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_audio_descriptor_gain_invariance(self):
        sr = 64000; t = np.arange(32000)/sr
        x = np.sin(2*np.pi*(900*t+1400*t*t))+.2*np.sin(2*np.pi*5000*t)
        a,b = describe_call(x,sr),describe_call(.1*x+.2,sr)
        keys = [k for k in a if k.startswith(('mel_','linear_','cep_'))]
        self.assertTrue(np.allclose([a[k] for k in keys],[b[k] for k in keys],rtol=1e-4,atol=.003))

    def test_verified_cache_does_not_download(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'cached.flac'; path.write_bytes(b'fixture-original-bytes')
            obj = dict(size=path.stat().st_size,md5Hash=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode())
            # Missing URL fields would fail if a network request were attempted.
            fetch_object(obj,path)
            self.assertEqual(path.read_bytes(),b'fixture-original-bytes')

    def test_future_multiclass_training_split_contract(self):
        with tempfile.TemporaryDirectory() as td:
            rows,fs = [],[]
            for date in ['2024-01-01','2024-01-02','2024-01-03']:
                rid='fixture_'+date.replace('-','')+'T000000.flac'
                for selection,label in [(1,'S04'),(2,'S19')]:
                    key=Path(rid).stem+'__selection_'+str(selection)
                    rows.append(dict(call_id=key,recording_id=rid,recording_date=date,selection=selection,
                                     start_s=float(selection),end_s=selection+.5,channel=1,call_type=label))
                    fs.append(dict(call_id=key,duration_s=.5,spectral_centroid_hz=1000 if label=='S04' else 7000))
            td=Path(td); m,f=td/'manifest.csv',td/'features.csv'
            pd.DataFrame(rows).to_csv(m,index=False); pd.DataFrame(fs).to_csv(f,index=False)
            train_baseline(SimpleNamespace(manifest=m,features=f,out=td/'run'))
            splits=json.loads((td/'run/split_audit.json').read_text())
            self.assertEqual(len(splits),3)
            for split in splits:
                self.assertFalse(set(split['train_call_ids'])&set(split['test_call_ids']))
            self.assertTrue((td/'run/classifier.joblib').is_file())


if __name__ == '__main__':
    unittest.main()
