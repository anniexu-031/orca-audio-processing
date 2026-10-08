"""Offline sliding-window S04 detector to do baseline locality estimation.
"""
import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.special import expit

from .audio_features import audio_index, read_audio, resolve_audio, sha256
from .pipeline import load_manifest, make_previews, md5, write_json

DEFAULT_CONFIG = dict(feature_version=1, sample_rate=32000, fft_samples=1024,
    spectrogram_hop=512, bands=24, min_hz=80.0, max_hz=15000.0,
    window_s=0.8, hop_s=0.2, threshold=0.5, merge_gap_s=0.2,
    boundary_pad_s=0.15, min_event_s=0.2, positive_coverage=0.5,
    background_guard_s=0.1, negatives_per_positive=5, C=0.3, seed=42,
    event_iou_threshold=0.3)
EVENT_COLUMNS = ['recording_id','channel','event_id','start_s','end_s','peak_score',
                 'mean_score','positive_windows','training_audio']


def validate_config(cfg):
    if cfg['feature_version'] != 1:
        raise ValueError('Unsupported detector feature version.')
    if not (0 < cfg['hop_s'] <= cfg['window_s'] and 0 <= cfg['threshold'] <= 1):
        raise ValueError('Require 0 < hop_s <= window_s and threshold in [0,1].')
    if not (0 < cfg['min_hz'] < cfg['max_hz'] < cfg['sample_rate']/2):
        raise ValueError('Invalid feature frequency range.')
    if not (cfg['fft_samples'] > cfg['spectrogram_hop'] > 0 and cfg['bands'] >= 2):
        raise ValueError('Invalid spectrogram/filterbank configuration.')
    if any(cfg[k] < 0 for k in ['merge_gap_s','boundary_pad_s','min_event_s','background_guard_s']):
        raise ValueError('Time tolerances cannot be negative.')
    if not (0 < cfg['positive_coverage'] <= 1 and cfg['negatives_per_positive'] >= 1 and cfg['C'] > 0):
        raise ValueError('Invalid label/sampling/model settings.')


def window_features(samples, sr, cfg):
    """One feature row per complete scanning window, plus start/end in seconds."""
    validate_config(cfg)
    x = np.asarray(samples,dtype=np.float32)
    if sr < cfg['sample_rate']:
        raise ValueError(f'Audio must be sampled at least at {cfg["sample_rate"]} Hz for this model.')
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite mono-channel samples.')
    duration = len(x)/sr
    target = cfg['sample_rate']
    if sr != target:
        g = math.gcd(sr,target)
        x = signal.resample_poly(x,target//g,sr//g)
    n = cfg['fft_samples']
    if len(x) < n:
        x = np.pad(x,(0,n-len(x)))
    f,t,power = signal.spectrogram(x,fs=target,nperseg=n,window='hann',
        noverlap=n-cfg['spectrogram_hop'],detrend='constant',scaling='density')
    lo,hi = 2595*np.log10(1+cfg['min_hz']/700),2595*np.log10(1+cfg['max_hz']/700)
    edges = 700*(10**(np.linspace(lo,hi,cfg['bands']+2)/2595)-1)
    weights = np.maximum(0,np.minimum((f[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],
                                      (edges[2:,None]-f[None,:])/(edges[2:]-edges[1:-1])[:,None]))
    band = weights@power
    total = band.sum(axis=0)
    fraction = band/np.maximum(total[None,:],1e-30)
    log_band = 10*np.log10(np.maximum(fraction,1e-8))
    centers = edges[1:-1]
    centroid = (centers[:,None]*fraction).sum(axis=0)
    entropy = -(fraction*np.log(np.maximum(fraction,1e-30))).sum(axis=0)/np.log(cfg['bands'])
    flatness = np.exp(np.log(np.maximum(fraction,1e-30)).mean(axis=0))/np.maximum(fraction.mean(axis=0),1e-30)
    log_power = 10*np.log10(np.maximum(total,1e-30))
    starts = np.arange(0,max(0,duration-cfg['window_s'])+1e-9,cfg['hop_s'])
    final = max(0,duration-cfg['window_s'])
    if final-starts[-1] > 1e-6:
        starts = np.append(starts,final)
    bounds,rows = [],[]
    for start in starts:
        end = min(duration,start+cfg['window_s'])
        mask = (t >= start)&(t < end)
        if not mask.any():
            mask[np.argmin(np.abs(t-(start+end)/2))] = True
        values = {}
        for i in range(cfg['bands']):
            values[f'mel_mean_{i:02d}'] = float(log_band[i,mask].mean())
            values[f'mel_std_{i:02d}'] = float(log_band[i,mask].std())
        for name,series in [('log_power',log_power),('centroid',centroid),('entropy',entropy),('flatness',flatness)]:
            values[name+'_mean'] = float(series[mask].mean())
            values[name+'_std'] = float(series[mask].std())
        rows.append(values); bounds.append(dict(start_s=float(start),end_s=float(end)))
    features = pd.DataFrame(rows)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite detector features.')
    return pd.DataFrame(bounds),features


def read_annotations(path, channel, duration):
    raw = pd.read_csv(path,sep='\t',keep_default_na=False)
    required = {'Channel','Begin Time (s)','End Time (s)','Call Type','Confidence','Sound ID Species'}
    if not required <= set(raw):
        raise ValueError('Original Raven table missing required fields: '+str(required-set(raw)))
    df = pd.DataFrame(dict(channel=pd.to_numeric(raw['Channel'],errors='raise'),
        start_s=pd.to_numeric(raw['Begin Time (s)'],errors='raise'),
        end_s=pd.to_numeric(raw['End Time (s)'],errors='raise'),
        call_type=raw['Call Type'].str.strip(),confidence=raw['Confidence'].str.lower().str.strip(),
        species=raw['Sound ID Species'].str.strip()))
    valid = np.isfinite(df[['channel','start_s','end_s']].to_numpy()).all(axis=1)
    valid &= (df.channel==channel)&(df.start_s>=0)&(df.end_s>df.start_s)&(df.end_s<=duration+1e-5)
    return df[valid].copy()


def overlaps(start,end,intervals,guard=0):
    if not len(intervals):
        return False
    return bool(((intervals.start_s-guard < end)&(intervals.end_s+guard > start)).any())


def label_windows(bounds, positives, annotations, cfg):
    rows = []
    # Exclude all S04-like annotations from negatives, including low-confidence
    # or variant labels outside the positive manifest.
    possible_s04 = annotations[annotations.call_type.str.upper().str.startswith('S04')]
    known_other = annotations[(annotations.species=='KW')&(annotations.confidence=='high')&
        annotations.call_type.str.fullmatch(r'(?i)(?:S\d+(?:i+)?|whistle|rasp)')&
        ~annotations.call_type.str.upper().str.startswith('S04')]
    for row in bounds.itertuples():
        start,end = row.start_s,row.end_s
        center = (start+end)/2
        hit = positives[(positives.start_s<=center)&(positives.end_s>center)]
        target = None
        for call in hit.itertuples():
            overlap = max(0,min(end,call.end_s)-max(start,call.start_s))
            if overlap >= cfg['positive_coverage']*min(end-start,call.end_s-call.start_s):
                target = call.call_id; break
        if target:
            rows.append(dict(label=1,label_source='annotated_s04',positive_call_id=target))
        elif overlaps(start,end,possible_s04) or overlaps(start,end,positives):
            rows.append(dict(label=-1,label_source='ambiguous_s04_edge_or_label',positive_call_id=''))
        elif ((known_other.start_s<=center)&(known_other.end_s>center)).any():
            rows.append(dict(label=0,label_source='annotated_non_s04',positive_call_id=''))
        elif not overlaps(start,end,annotations,cfg['background_guard_s']):
            rows.append(dict(label=0,label_source='unannotated_background_proxy',positive_call_id=''))
        else:
            rows.append(dict(label=-1,label_source='ambiguous_other_annotation',positive_call_id=''))
    return pd.DataFrame(rows)


def training_indices(meta,cfg):
    rng = np.random.default_rng(cfg['seed'])
    result = []
    for _,group in meta.groupby('recording_id',sort=True):
        positive = group.index[group.label==1].to_numpy()
        hard = group.index[(group.label==0)&(group.label_source=='annotated_non_s04')].to_numpy()
        proxy = group.index[(group.label==0)&(group.label_source=='unannotated_background_proxy')].to_numpy()
        cap = int(cfg['negatives_per_positive']*len(positive))
        if not len(positive):
            raise ValueError('A training recording has no positive windows; revise sampling or annotations.')
        hard = rng.choice(hard,min(len(hard),cap),replace=False)
        proxy = rng.choice(proxy,min(len(proxy),max(0,cap-len(hard))),replace=False)
        result.extend(positive.tolist()+hard.tolist()+proxy.tolist())
    result = np.array(sorted(result),dtype=int)
    if set(meta.loc[result,'label']) != {0,1}:
        raise ValueError('Training requires S04 and negative/proxy windows.')
    return result


def export_model(fitted,names,cfg,provenance):
    scale,clf = fitted.named_steps['standardscaler'],fitted.named_steps['logisticregression']
    if clf.classes_.tolist() != [0,1]:
        raise ValueError('Unexpected class encoding.')
    return dict(model_type='s04_window_logistic_v1',feature_names=list(names),config=cfg,
        scaler_mean=scale.mean_.tolist(),scaler_scale=scale.scale_.tolist(),
        coefficients=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),
        training_recordings=provenance,score_note='Uncalibrated model score; not a verified S04 probability.')


def model_scores(features,model):
    if model.get('model_type') != 's04_window_logistic_v1':
        raise ValueError('Unsupported model format.')
    names = model['feature_names']
    if len(set(names)) != len(names) or set(names)-set(features):
        raise ValueError('Model feature names do not match extractor.')
    x = features[names].to_numpy(dtype=float)
    mean,scale,coef = (np.asarray(model[k],dtype=float) for k in ['scaler_mean','scaler_scale','coefficients'])
    if not (mean.shape==scale.shape==coef.shape==(len(names),) and np.isfinite([mean,scale,coef]).all() and (scale>0).all()):
        raise ValueError('Invalid portable model weights/scaler.')
    return expit(((x-mean)/scale)@coef+model['intercept'])


def merge_events(bounds,scores,cfg,duration):
    events = []
    for row,value in zip(bounds.itertuples(),scores):
        if value < cfg['threshold']:
            continue
        center = (row.start_s+row.end_s)/2
        start,end = max(0,center-cfg['hop_s']/2),min(duration,center+cfg['hop_s']/2)
        if events and start <= events[-1]['end_s']+cfg['merge_gap_s']+1e-9:
            events[-1]['end_s'] = max(events[-1]['end_s'],end)
            events[-1]['values'].append(float(value))
        else:
            events.append(dict(start_s=start,end_s=end,values=[float(value)]))
    result = []
    for event in events:
        start = max(0,event['start_s']-cfg['boundary_pad_s'])
        end = min(duration,event['end_s']+cfg['boundary_pad_s'])
        if end-start >= cfg['min_event_s']:
            result.append(dict(start_s=float(start),end_s=float(end),peak_score=max(event['values']),
                mean_score=float(np.mean(event['values'])),positive_windows=len(event['values'])))
    return result


def event_metrics(events,reference,duration,iou_threshold):
    pairs = []
    for i,event in enumerate(events):
        for j,ref in enumerate(reference.itertuples()):
            intersection = max(0,min(event['end_s'],ref.end_s)-max(event['start_s'],ref.start_s))
            union = event['end_s']-event['start_s']+ref.end_s-ref.start_s-intersection
            iou = intersection/union if union else 0
            if iou>=iou_threshold:
                pairs.append((iou,i,j))
    seen_events,seen_refs = set(),set()
    for _,i,j in sorted(pairs,reverse=True):
        if i not in seen_events and j not in seen_refs:
            seen_events.add(i); seen_refs.add(j)
    matched = len(seen_events)
    return dict(reference_calls=len(reference),candidate_events=len(events),matched_events=matched,
        unmatched_events=len(events)-matched,missed_reference_calls=len(reference)-matched,
        annotation_precision=matched/len(events) if events else 0.0,
        annotation_recall=matched/len(reference) if len(reference) else 0.0,
        unmatched_events_per_hour=(len(events)-matched)/(duration/3600))


def annotation_path(record,roots,mapping):
    rid = record['recording_id']
    if rid in mapping:
        path = Path(mapping[rid]).resolve()
    else:
        name = Path(record['annotations']['name']).name
        matches = [Path(root)/name for root in roots if (Path(root)/name).is_file()]
        if not matches:
            raise ValueError('Missing original annotation '+name+'; download or supply --annotation-roots.')
        path = matches[0]
    obj = record['annotations']
    if not path.is_file() or path.stat().st_size!=int(obj['size']) or md5(path)!=obj['md5Hash']:
        raise ValueError('Annotation bytes differ from source catalog: '+str(path))
    return path


def fit_detector(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score,precision_recall_fscore_support
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    import sklearn
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        changes = json.loads(Path(args.config).read_text())
        if set(changes)-set(cfg):
            raise ValueError('Unknown detector config keys: '+str(set(changes)-set(cfg)))
        cfg.update(changes)
    validate_config(cfg)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory for detector training.')
    out.mkdir(parents=True,exist_ok=True)
    positives = load_manifest(args.manifest,only_s04=True)
    records = json.loads(Path(args.recordings).read_text())['records']
    if set(positives.recording_id)!=set(r['recording_id'] for r in records):
        raise ValueError('Positive manifest and recording catalog differ.')
    index = audio_index(args.audio_roots)
    aliases = json.loads(Path(args.audio_map).read_text()) if args.audio_map else {}
    annotation_map = json.loads(Path(args.annotation_map).read_text()) if args.annotation_map else {}
    meta_frames,feature_frames,provenance,record_info = [],[],[],{}
    for record in records:
        rid = record['recording_id']; reference = positives[positives.recording_id==rid]
        if reference.channel.nunique()!=1:
            raise ValueError('Starter training expects one annotated channel per recording.')
        channel = int(reference.channel.iloc[0]); path = resolve_audio(rid,index,aliases)
        if path.stat().st_size!=int(record['audio']['size']) or md5(path)!=record['audio']['md5Hash']:
            raise ValueError('Original audio checksum mismatch: '+str(path))
        annotations_path = annotation_path(record,args.annotation_roots,annotation_map)
        audio,sr = read_audio(path)
        if not 1<=channel<=audio.shape[1]:
            raise ValueError('Training annotation channel absent from audio.')
        duration = len(audio)/sr
        if (reference.end_s>duration).any():
            raise ValueError('Positive times exceed decoded recording.')
        annotations = read_annotations(annotations_path,channel,duration)
        print('Window extraction:',rid,flush=True)
        bounds,features = window_features(audio[:,channel-1],sr,cfg)
        labels = label_windows(bounds,reference,annotations,cfg)
        meta = pd.concat([bounds,labels],axis=1)
        meta['recording_id']=rid; meta['recording_date']=record['recording_date'];meta['channel']=channel
        meta['window_id']=[Path(rid).stem+'__window_'+str(i) for i in range(len(meta))]
        meta_frames.append(meta); feature_frames.append(features)
        provenance.append(dict(recording_id=rid,recording_date=record['recording_date'],channel=channel,
            duration_s=duration,audio_sha256=sha256(path),annotation_sha256=sha256(annotations_path)))
        record_info[rid]=dict(bounds=bounds,reference=reference,duration=duration,features=features,channel=channel)
        del audio
    meta = pd.concat(meta_frames,ignore_index=True);x=pd.concat(feature_frames,ignore_index=True)
    groups = meta.recording_date.to_numpy()
    if len(set(groups))<3:
        raise ValueError('At least three recording dates are required for development evaluation.')
    def build():
        return make_pipeline(StandardScaler(),LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=3000,random_state=cfg['seed']))
    meta['heldout_score']=np.nan
    fold_metrics,splits,event_rows = [],[],[]
    with threadpool_limits(limits=1):
        for train,test in LeaveOneGroupOut().split(x,meta.label,groups):
            training = training_indices(meta.iloc[train],cfg)
            if set(groups[training])&set(groups[test]):
                raise AssertionError('Training/test date overlap.')
            fitted = build().fit(x.loc[training],meta.loc[training,'label'])
            portable = export_model(fitted,x.columns,cfg,[])
            score = model_scores(x.iloc[test],portable)
            if not np.allclose(score,fitted.predict_proba(x.iloc[test])[:,1],atol=1e-12):
                raise AssertionError('Portable model differs from fitted classifier.')
            meta.loc[test,'heldout_score']=score
            for rid,test_meta in meta.iloc[test].groupby('recording_id',sort=True):
                info=record_info[rid]
                events=merge_events(info['bounds'],test_meta.heldout_score.to_numpy(),cfg,info['duration'])
                metric=event_metrics(events,info['reference'],info['duration'],cfg['event_iou_threshold'])
                fold_metrics.append(dict(recording_id=rid,heldout_date=str(test_meta.recording_date.iloc[0]),duration_s=info['duration'],**metric))
                for i,event in enumerate(events,1):
                    event_rows.append(dict(recording_id=rid,event_id=Path(rid).stem+'__candidate_'+str(i),**event))
            splits.append(dict(heldout_date=str(groups[test][0]),train_dates=sorted(set(groups[training])),
                train_window_ids=meta.loc[training,'window_id'].tolist(),test_window_ids=meta.iloc[test].window_id.tolist()))
            print('Held out:',str(groups[test][0]),flush=True)
        final_indices=training_indices(meta,cfg)
        final_fit=build().fit(x.loc[final_indices],meta.loc[final_indices,'label'])
    model=export_model(final_fit,x.columns,cfg,provenance)
    model['training_summary']=dict(reference_s04_calls=len(positives),dates=len(set(groups)),
        sampled_windows=len(final_indices),label_source_counts=meta.loc[final_indices,'label_source'].value_counts().to_dict(),
        note='Annotated non-S04 calls and unannotated background proxies supply negative windows; proxies are not verified negatives.')
    model['training_source_code_sha256']=sha256(Path(__file__))
    write_json(out/'detector.json',model)
    write_json(out/'split_audit.json',splits)
    meta.to_csv(out/'window_labels_and_scores.csv',index=False)
    summary=meta.groupby(['recording_date','label_source']).size().rename('windows').reset_index()
    summary.to_csv(out/'window_label_summary.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'event_metrics_by_date.csv',index=False)
    pd.DataFrame(event_rows,columns=['recording_id','event_id','start_s','end_s','peak_score','mean_score','positive_windows']).to_csv(out/'heldout_candidates.csv',index=False)
    evaluated=meta[meta.label>=0];pred=evaluated.heldout_score>=cfg['threshold']
    pr,re,f1,_=precision_recall_fscore_support(evaluated.label,pred,labels=[1],zero_division=0)
    total_events=sum(m['candidate_events'] for m in fold_metrics);matched=sum(m['matched_events'] for m in fold_metrics)
    event_pr=matched/total_events if total_events else 0.0;event_re=matched/len(positives)
    metrics=dict(scope='Exploratory fixed baseline; entire dates held out; previously inspected recordings; incomplete annotation reference.',
        reference_s04_calls=len(positives),recording_dates=len(set(groups)),features=len(x.columns),
        config=cfg,sklearn_version=sklearn.__version__,training_summary=model['training_summary'],
        window_metrics=dict(precision=float(pr[0]),recall=float(re[0]),f1=float(f1[0]),
            average_precision=float(average_precision_score(evaluated.label,evaluated.heldout_score)),
            evaluated_windows=len(evaluated),note='Weak window labels; ambiguous windows excluded.'),
        event_metrics=dict(annotation_precision=event_pr,annotation_recall=event_re,
            f1=2*event_pr*event_re/(event_pr+event_re) if event_pr+event_re else 0,
            candidate_events=total_events,matched_events=matched,unmatched_events=total_events-matched,
            reference_calls=len(positives),unmatched_events_per_hour=(total_events-matched)/(sum(m['duration_s'] for m in fold_metrics)/3600),
            matching='Greedy one-to-one interval IoU >= '+str(cfg['event_iou_threshold']),
            note='Unmatched detections can include unannotated true calls; this is annotation-referenced performance.'),
        warnings=['Background proxies are not confirmed negatives.','Scores are uncalibrated.','No independent final test or live-streaming benchmark.'])
    write_json(out/'metrics.json',metrics)
    write_training_report(out,metrics,pd.DataFrame(fold_metrics),summary)
    print(json.dumps(metrics['event_metrics'],indent=2),flush=True)
    print('Fitted detector:',out/'detector.json',flush=True)


def detect(args):
    model=json.loads(Path(args.model).read_text());cfg=dict(model['config']);validate_config(cfg)
    if args.threshold is not None:
        if not 0<=args.threshold<=1:raise ValueError('Threshold must be between zero and one.')
        cfg['threshold']=args.threshold
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory.')
    out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name in args.audio:
        path=Path(name).expanduser().resolve()
        if path.is_dir():paths+=sorted(p for p in path.iterdir() if p.suffix.lower() in {'.flac','.wav'})
        elif path.is_file():paths.append(path)
        else:raise ValueError('Missing audio input: '+str(path))
    paths=list(dict.fromkeys(paths))
    if not paths:raise ValueError('No FLAC/WAV inputs found.')
    if len({p.name for p in paths})!=len(paths):raise ValueError('Input filenames must be unique.')
    training_hashes={r['audio_sha256'] for r in model['training_recordings']}
    all_events,windows,run_inputs=[],[],[]
    for path in paths:
        print('Scanning:',path.name,flush=True)
        audio,sr=read_audio(path)
        if not 1<=args.channel<=audio.shape[1]:raise ValueError('Requested channel absent from '+path.name)
        duration=len(audio)/sr;hash_=sha256(path)
        bounds,features=window_features(audio[:,args.channel-1],sr,model['config'])
        score=model_scores(features,model)
        events=merge_events(bounds,score,cfg,duration)
        for i,event in enumerate(events,1):
            all_events.append(dict(recording_id=path.name,channel=args.channel,
                event_id=path.stem+'__candidate_'+str(i),training_audio=hash_ in training_hashes,**event))
        if args.save_windows:
            window=bounds.copy();window['recording_id']=path.name;window['channel']=args.channel;window['score']=score;windows.append(window)
        run_inputs.append(dict(recording_id=path.name,sha256=hash_,duration_s=duration,
            channel=args.channel,sample_rate=sr,training_audio=hash_ in training_hashes,candidates=len(events)))
        del audio
    events=pd.DataFrame(all_events,columns=EVENT_COLUMNS)
    events.to_csv(out/'candidates.csv',index=False)
    if windows:pd.concat(windows,ignore_index=True).to_csv(out/'window_scores.csv',index=False)
    write_json(out/'run.json',dict(model_sha256=sha256(Path(args.model)),inputs=run_inputs,config=cfg,
        note='Candidate S04 events; uncalibrated scores and approximate boundaries. Training-audio scans are demonstrations, not independent tests.'))
    previews={}
    if args.preview_limit and len(events):
        chosen=events.sort_values('peak_score',ascending=False).head(args.preview_limit).copy()
        chosen['call_id']=chosen.event_id
        preview_args=argparse.Namespace(audio_roots=list({str(p.parent) for p in paths}),audio_map=None,
                                         max_previews=args.preview_limit)
        previews=make_previews(chosen,preview_args,out)
    write_detection_report(out,events,previews,run_inputs)
    print(f'{len(events)} candidate S04 intervals. Open: {out/"report.html"}',flush=True)


def write_detection_report(out,events,previews,inputs):
    cards=[]
    for row in events.itertuples():
        if row.event_id in previews:
            key=html.escape(row.event_id,quote=True)
            cards.append(f'<article><h3>{key} · score {row.peak_score:.3f}</h3><audio controls preload="none" src="clips/{key}.wav"></audio><img src="plots/{key}.png" alt="Candidate spectrogram"></article>')
    text=f'<p>{len(events)} candidate intervals. Scores are uncalibrated; predictions require review.</p>'
    if any(r['training_audio'] for r in inputs):text+='<p><strong>Some inputs were used for fitting.</strong> This scan is a demonstration, not an independent performance test.</p>'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Candidate S04 detections</title>
    <style>body{{font:16px/1.55 system-ui;color:#17344c;max-width:1200px;margin:35px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccd8df;padding:6px}}.scroll{{overflow:auto}}img{{max-width:100%}}article{{background:#eff4f7;padding:18px;margin:24px 0}}</style>
    <h1>Candidate S04 detections</h1>{text}<p>Approximate timestamps are in seconds from the original recording start. Listening copies are independently normalized.</p>
    <p><a href="candidates.csv">Candidate intervals CSV</a> · <a href="run.json">Run settings</a></p>
    <div class="scroll">{events.to_html(index=False,float_format=lambda v:f'{v:.3f}')}</div><h2>Highest-scoring listening examples</h2>{''.join(cards)}</html>'''
    (out/'report.html').write_text(page,encoding='utf-8')


def write_training_report(out,metrics,folds,summary):
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>S04 detector baseline evaluation</title>
    <style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#17344c}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #cbd6df;padding:6px}}pre{{white-space:pre-wrap;background:#edf3f7;padding:15px}}.scroll{{overflow:auto}}</style>
    <h1>S04 continuous-audio detector baseline</h1><p>{metrics['reference_s04_calls']} reference S04 calls · {metrics['recording_dates']} dates · {metrics['features']} window features.</p>
    <p>Fixed logistic regression, 0.8-second windows, 0.2-second steps. Each evaluation recording was scanned by a model trained without its entire recording date.</p>
    <p><strong>Exploratory results.</strong> Unannotated regions supply background proxies, not verified negatives. Available annotations may omit calls.
    Event precision/recall are referenced to the available S04 intervals, with greedy one-to-one IoU matching. Scores are uncalibrated.</p>
    <h2>Annotation-referenced event results</h2><pre>{html.escape(json.dumps(metrics['event_metrics'],indent=2))}</pre>
    <h2>By held-out date</h2><div class="scroll">{folds.to_html(index=False,float_format=lambda v:f'{v:.3f}')}</div>
    <h2>Window label sources</h2>{summary.to_html(index=False)}
    <h2>Window-level development results</h2><pre>{html.escape(json.dumps(metrics['window_metrics'],indent=2))}</pre>
    <p>The final JSON model was refitted on all development recordings. Its performance on fresh dates has not been measured.</p>
    <p><a href="metrics.json">Metrics</a> · <a href="heldout_candidates.csv">Held-out candidate intervals</a> · <a href="window_label_summary.csv">Label counts</a></p></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')
