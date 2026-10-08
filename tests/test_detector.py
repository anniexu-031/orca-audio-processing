"""Detector labels, portable inference, timing, matching and annotation-free scan."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from orca_s04.detector import DEFAULT_CONFIG,detect,event_metrics,export_model,label_windows,merge_events,model_scores,window_features


class DetectorTests(unittest.TestCase):
    def test_edges_and_uncertain_s04_cannot_be_negative(self):
        bounds=pd.DataFrame({'start_s':[.8,1.4,3,4.2,6,8], 'end_s':[1.6,2.2,3.8,5,6.8,8.8]})
        positives=pd.DataFrame([dict(call_id='positive',start_s=1,end_s=1.5)])
        annotations=pd.DataFrame([
            dict(start_s=1,end_s=1.5,call_type='S04',species='KW',confidence='high'),
            dict(start_s=3.2,end_s=3.7,call_type='S19',species='KW',confidence='high'),
            dict(start_s=4.3,end_s=4.8,call_type='S04',species='KW',confidence='low'),
            dict(start_s=8.2,end_s=8.9,call_type='',species='KW',confidence='')])
        labelled=label_windows(bounds,positives,annotations,DEFAULT_CONFIG)
        self.assertEqual(labelled.label.tolist(),[1,-1,0,-1,0,-1])
        self.assertEqual(labelled.loc[4,'label_source'],'unannotated_background_proxy')
        self.assertEqual(labelled.loc[2,'label_source'],'annotated_non_s04')

    def test_json_model_preserves_sklearn_scores(self):
        x=pd.DataFrame({'f1':[0,1,2,4,5,6],'f2':[2,1,4,1,3,5]})
        y=np.array([0,0,0,1,1,1])
        fitted=make_pipeline(StandardScaler(),LogisticRegression(C=.3)).fit(x,y)
        model=json.loads(json.dumps(export_model(fitted,x.columns,DEFAULT_CONFIG,[])))
        self.assertTrue(np.allclose(model_scores(x,model),fitted.predict_proba(x)[:,1],atol=1e-12))

    def test_merge_threshold_gap_and_bounds(self):
        bounds=pd.DataFrame(dict(start_s=[0,.2,.4,1.5],end_s=[.8,1,1.2,2.3]))
        events=merge_events(bounds,np.array([.7,.1,.8,.9]),DEFAULT_CONFIG,2.3)
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['positive_windows'],2)
        self.assertTrue(all(0<=e['start_s']<e['end_s']<=2.3 for e in events))

    def test_one_to_one_event_match(self):
        events=[dict(start_s=1,end_s=2),dict(start_s=1,end_s=2)]
        reference=pd.DataFrame(dict(start_s=[1],end_s=[2]))
        metrics=event_metrics(events,reference,60,.3)
        self.assertEqual(metrics['matched_events'],1)
        self.assertEqual(metrics['annotation_recall'],1)
        self.assertEqual(metrics['annotation_precision'],.5)

    def test_short_silent_audio_has_finite_features(self):
        bounds,x=window_features(np.zeros(400,dtype=np.float32),32000,DEFAULT_CONFIG)
        self.assertEqual(len(bounds),1)
        self.assertEqual(len(x.columns),56)
        self.assertTrue(np.isfinite(x.to_numpy()).all())
        self.assertLessEqual(bounds.end_s.max(),400/32000)

    def test_scan_requires_audio_and_model_but_no_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); sr=32000; t=np.arange(sr*2)/sr
            x=.01*np.sin(2*np.pi*1100*t)
            wavfile.write(str(root/'new_recording.wav'),sr,x.astype(np.float32))
            args=SimpleNamespace(model=ROOT/'models/s04_detector.json',audio=[str(root/'new_recording.wav')],
                channel=1,threshold=None,preview_limit=0,save_windows=True,out=root/'scan')
            detect(args)
            candidates=pd.read_csv(root/'scan/candidates.csv')
            windows=pd.read_csv(root/'scan/window_scores.csv')
            self.assertTrue(len(windows)>0)
            self.assertTrue(np.isfinite(windows.score).all())
            self.assertTrue(((windows.score>=0)&(windows.score<=1)).all())
            self.assertTrue((candidates.start_s>=0).all() and (candidates.end_s<=2).all())
            self.assertFalse(candidates.training_audio.any())
            self.assertTrue((root/'scan/report.html').is_file())


if __name__=='__main__':
    unittest.main()
