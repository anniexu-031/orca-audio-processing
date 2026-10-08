"Classifier template for adding additional data/other labelled calls"
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .pipeline import DEFAULT_FEATURES, load_manifest, load_features, write_json


def train_baseline(args):
    data = load_manifest(args.manifest)
    if data.call_type.nunique() < 2:
        raise ValueError('not enough labels')
    features = load_features(args.features,data)
    if not set(DEFAULT_FEATURES) <= set(features):
        raise ValueError('Baseline needs duration_s and spectral_centroid_hz.')
    coverage = data.groupby('call_type').recording_date.nunique()
    if (coverage < 2).any() or data.recording_date.nunique() < 3:
        raise ValueError('Each class needs at least two dates; evaluation needs at least three dates. More dates are strongly preferred.')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory.')
    x, y, groups = features[DEFAULT_FEATURES], data.call_type.to_numpy(), data.recording_date.to_numpy()
    labels = sorted(set(y))
    result = data[['call_id','recording_date','call_type']].copy()
    result['baseline_prediction'] = ''; result['majority_prediction'] = ''
    splits = []
    def model():
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42))
    with threadpool_limits(limits=1):
        for train,test in LeaveOneGroupOut().split(x,y,groups):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError('Date leakage.')
            if len(set(y[train])) < 2 or set(y[test])-set(y[train]):
                raise ValueError('Held-out date leaves missing classes in training; add cross-date examples.')
            fit = model().fit(x.iloc[train],y[train])
            dummy = DummyClassifier(strategy='most_frequent').fit(x.iloc[train],y[train])
            result.loc[test,'baseline_prediction'] = fit.predict(x.iloc[test])
            result.loc[test,'majority_prediction'] = dummy.predict(x.iloc[test])
            splits.append(dict(heldout_date=str(groups[test][0]),train_call_ids=data.iloc[train].call_id.tolist(),test_call_ids=data.iloc[test].call_id.tolist()))
        fitted = model().fit(x,y)
    out.mkdir(parents=True,exist_ok=True)
    pooled, by_class = {}, []
    for name in ['baseline','majority']:
        pred = result[name+'_prediction']
        pooled[name] = dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)))
        pr,re,f1,n = precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
        by_class += [dict(model=name,call_type=label,precision=float(pr[i]),recall=float(re[i]),f1=float(f1[i]),support=int(n[i])) for i,label in enumerate(labels)]
    result.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(by_class).to_csv(out/'per_class_metrics.csv',index=False)
    write_json(out/'metrics.json',dict(status='fixed baseline on reviewed multi-class segments',
        evaluation='leave-one-recording-date-out development evaluation',features=DEFAULT_FEATURES,
        pooled_out_of_fold=pooled,classes=labels,calls=len(data),
        note='Not an independent final test after iterative development. No call detector or unknown-class rejection.'))
    write_json(out/'split_audit.json',splits)
    joblib.dump(dict(pipeline=fitted,features=DEFAULT_FEATURES,classes=labels,sklearn_version=sklearn.__version__,
                    training_call_ids=data.call_id.tolist(),training_dates=sorted(set(groups))),out/'classifier.joblib',compress=3)
    print(json.dumps(pooled,indent=2)); print('Saved:',out)
