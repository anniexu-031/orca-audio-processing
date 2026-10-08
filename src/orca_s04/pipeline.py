
import argparse
import base64
import hashlib
import html
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal

from .audio_features import FEATURE_CONFIG, audio_index, extract_features, read_audio, resolve_audio

REQUIRED = {'call_id', 'recording_id', 'recording_date', 'selection', 'start_s', 'end_s', 'channel', 'call_type'}
DEFAULT_FEATURES = ['duration_s', 'spectral_centroid_hz']


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def load_manifest(path, only_s04=False):
    data = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED - set(data)
    if missing:
        raise ValueError('Missing manifest fields: ' + ', '.join(sorted(missing)))
    if data.empty or data.call_id.duplicated().any():
        raise ValueError('Manifest must be nonempty with unique call IDs.')
    if data.recording_id.astype(str).str.strip().eq('').any() or data.call_type.astype(str).str.strip().eq('').any():
        raise ValueError('Recording IDs and labels cannot be blank.')
    for col in ['selection', 'start_s', 'end_s', 'channel']:
        data[col] = pd.to_numeric(data[col], errors='raise')
    if not np.isfinite(data[['selection','start_s','end_s','channel']].to_numpy()).all():
        raise ValueError('Nonfinite annotation times/IDs/channels.')
    if ((data.selection % 1 != 0) | (data.selection < 1) | (data.channel % 1 != 0) | (data.channel < 1)).any():
        raise ValueError('Selection/channel numbers must be positive integers.')
    if ((data.start_s < 0) | (data.end_s <= data.start_s)).any():
        raise ValueError('Invalid segment onset/end.')
    data['selection'] = data.selection.astype(int)
    expected = [Path(str(r)).stem + '__selection_' + str(s) for r,s in zip(data.recording_id, data.selection)]
    if expected != data.call_id.tolist():
        raise ValueError('call_id must be recording filename stem + __selection_ + selection.')
    dates = pd.to_datetime(data.recording_date, format='%Y-%m-%d', errors='raise')
    if (data.groupby('recording_id').recording_date.nunique() != 1).any():
        raise ValueError('A recording cannot belong to multiple dates.')
    encoded = data.recording_id.str.extract(r'_(\d{8})T', expand=False)
    if encoded.notna().any():
        parsed = pd.to_datetime(encoded, format='%Y%m%d', errors='coerce')
        if (encoded.notna() & (parsed != dates)).any():
            raise ValueError('Recording date disagrees with its original filename.')
    if only_s04 and set(data.call_type) != {'S04'}:
        raise ValueError('The supplied starter manifest must contain S04 only.')
    if only_s04:
        if not {'species','ecotype','confidence'} <= set(data):
            raise ValueError('The S04 source manifest must preserve species, ecotype and confidence.')
        if not (data.species.eq('KW') & data.ecotype.str.upper().isin(['SR','SRKW']) & data.confidence.str.lower().eq('high')).all():
            raise ValueError('S04 starter rows must preserve confirmed, high-confidence Southern Resident KW labels.')
    return data


def load_features(path, manifest):
    features = pd.read_csv(path)
    if 'call_id' not in features or features.call_id.duplicated().any():
        raise ValueError('Feature rows need unique call IDs.')
    if set(features.call_id) != set(manifest.call_id):
        raise ValueError('Feature and manifest call IDs must match exactly; re-extract after edits.')
    features = features.set_index('call_id').loc[manifest.call_id]
    numeric = features.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Features must be finite numeric measurements.')
    return numeric


def folds_by_date(data):
    folds = []
    for date in sorted(data.recording_date.unique()):
        train, test = data[data.recording_date != date], data[data.recording_date == date]
        folds.append(dict(heldout_date=date, train_dates=sorted(train.recording_date.unique()),
                          train_call_ids=train.call_id.tolist(), test_call_ids=test.call_id.tolist()))
    return folds


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return base64.b64encode(h.digest()).decode('ascii')


def fetch_object(obj, destination):
    """Reuse verified cache or atomically fetch a pinned public archive object."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size == int(obj['size']) and md5(destination) == obj['md5Hash']:
        print('Verified cache:', destination.name, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.part')
    url = 'https://storage.googleapis.com/noaa-passive-bioacoustic/' + urllib.parse.quote(obj['name'], safe='/')
    url += '?' + urllib.parse.urlencode({'generation': obj['generation']})
    print('Downloading:', destination.name, flush=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'orca-s04-starter/0.1'}), timeout=90) as response, part.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if part.stat().st_size != int(obj['size']) or md5(part) != obj['md5Hash']:
            raise ValueError('Download checksum/size mismatch: ' + destination.name)
        part.replace(destination)
    finally:
        part.unlink(missing_ok=True)


def download(args):
    records = json.loads(Path(args.recordings).read_text())['records']
    selected = records[:args.limit] if args.limit is not None else records
    if not selected:
        raise ValueError('Select at least one recording.')
    total = sum(int(r[k]['size']) for r in selected for k in ['audio','annotations'])
    print(f'{len(selected)} recordings, {total/1e6:.1f} MB including original annotation tables.')
    for r in selected:
        print(r['recording_date'], r['recording_id'], r['s04_calls'])
    if args.plan:
        return
    out = Path(args.out)
    for r in selected:
        for key, folder in [('audio','audio'),('annotations','annotations')]:
            fetch_object(r[key], out/folder/Path(r[key]['name']).name)


def validate(args):
    data = load_manifest(args.manifest, only_s04=True)
    features = load_features(args.features, data)
    catalog = json.loads(Path(args.recordings).read_text())
    record_map = {r['recording_id']:r for r in catalog['records']}
    if set(record_map) != set(data.recording_id):
        raise ValueError('Recording catalog and manifest differ.')
    for rid, calls in data.groupby('recording_id'):
        if record_map[rid]['s04_calls'] != len(calls):
            raise ValueError('Catalog count disagrees with manifest for ' + rid)
    recorded_folds = json.loads(Path(args.folds).read_text())['folds']
    if recorded_folds != folds_by_date(data):
        raise ValueError('Date folds are stale; regenerate them after changing the manifest.')
    print(f'Validated {len(data)} S04 calls, {data.recording_date.nunique()} dates, {len(features.columns)} acoustic features.')
    print(data.groupby('recording_date').size().rename('calls').to_string())
    print('Source labels are preserved; no classification accuracy is reported for one-class data.')


def extract(args):
    data = load_manifest(args.manifest)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Feature output exists; use a new output filename.')
    aliases = json.loads(Path(args.audio_map).read_text()) if args.audio_map else None
    features, provenance = extract_features(data, args.audio_roots, FEATURE_CONFIG, aliases)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    write_json(out.with_suffix('.provenance.json'), dict(feature_config=FEATURE_CONFIG, audio=provenance))
    print('Saved:', out)


def profile(data, features):
    summaries = {}
    for column in ['duration_s', 'spectral_centroid_hz', 'spectral_peak_hz', 'rms_ac_digital']:
        values = features[column]
        summaries[column] = dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                                 median=float(values.median()), q10=float(values.quantile(.1)), q90=float(values.quantile(.9)))
    return dict(status='descriptive S04 profile; not a classifier', calls=len(data),
                labels=sorted(data.call_type.unique()), dates=sorted(data.recording_date.unique()),
                measurements=summaries)


def make_previews(data, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index = audio_index(args.audio_roots)
    aliases = json.loads(Path(args.audio_map).read_text()) if args.audio_map else None
    clips, plots = out/'clips', out/'plots'
    clips.mkdir(exist_ok=True); plots.mkdir(exist_ok=True)
    previews = {}
    chosen = data.head(args.max_previews)
    for rid, group in chosen.groupby('recording_id', sort=False):
        path = resolve_audio(rid, index, aliases)
        audio, sr = read_audio(path)
        for row in group.itertuples():
            if not (0 <= row.start_s < row.end_s <= len(audio)/sr and 1 <= row.channel <= audio.shape[1]):
                raise ValueError('Invalid original-audio segment: ' + row.call_id)
            a, b = max(0,int(round((row.start_s-.2)*sr))), min(len(audio),int(round((row.end_s+.2)*sr)))
            x = audio[a:b,int(row.channel)-1]
            listen = x.astype(float)-float(x.mean())
            if np.max(np.abs(listen)):
                listen = .95 * listen / np.max(np.abs(listen))
            wavfile.write(str(clips/(row.call_id+'.wav')),sr,np.round(listen*32767).astype(np.int16))
            n = min(2048,len(x))
            f,t,p = signal.spectrogram(x,fs=sr,nperseg=n,noverlap=3*n//4,detrend='constant')
            db = 10*np.log10(np.maximum(p,1e-20))
            fig,ax = plt.subplots(figsize=(9,3),constrained_layout=True)
            ax.pcolormesh(t+a/sr-row.start_s,f/1000,db,shading='auto',cmap='magma',vmin=db.max()-70,vmax=db.max())
            ax.axvline(0,color='cyan',linestyle='--'); ax.axvline(row.end_s-row.start_s,color='cyan',linestyle='--')
            ax.set(ylim=(0,min(24,sr/2000)),xlabel='Seconds relative to onset',ylabel='kHz',title=row.call_id)
            fig.savefig(plots/(row.call_id+'.png'),dpi=110); plt.close(fig)
            previews[row.call_id] = True
    return previews


def inspect(args):
    data = load_manifest(args.manifest)
    features = load_features(args.features,data)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Inspection output must be a new, empty directory.')
    out.mkdir(parents=True,exist_ok=True)
    summary = profile(data,features)
    write_json(out/'s04_profile.json',summary)
    joined = data.join(features[['duration_s','spectral_centroid_hz']],on='call_id',rsuffix='_feature')
    previews = make_previews(data,args,out) if args.with_audio else {}
    columns = ['call_id','recording_date','selection','call_type','start_s','end_s','duration_s','spectral_centroid_hz']
    table = joined[columns].to_html(index=False,float_format=lambda n:f'{n:.3f}')
    listening = []
    for row in data.itertuples():
        if row.call_id not in previews:
            continue
        key = html.escape(row.call_id,quote=True)
        listening.append(f'<article id="{key}"><h3>{key}</h3><audio controls preload="none" src="clips/{key}.wav"></audio><img src="plots/{key}.png" alt="S04 spectrogram"></article>')
    coverage = data.groupby('recording_date').size().rename('S04 calls').to_frame().to_html()
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>S04 dataset inspection</title>
    <style>body{{font:16px/1.55 system-ui;color:#17344c;margin:35px auto;max-width:1250px;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #d1dce3;padding:6px}}img{{max-width:100%}}article{{margin:24px 0;padding:18px;background:#f1f5f8}}.scroll{{overflow:auto}}</style>
    <h1>S04 acoustic research starter</h1><p>{len(data)} source-labeled calls · {data.recording_date.nunique()} dates.</p>
    <p>This is a one-class inspection and characterization dataset. It has no negative/other-type examples and no classifier performance estimate.</p>
    <p>Labels come from original annotations. Listening copies are DC-removed and independently normalized; quantitative features use original audio.</p>
    <h2>Coverage</h2>{coverage}<h2>Listening examples</h2>
    <p>{len(previews)} previews generated. To inspect every call, rerun with <code>--with-audio --max-previews 137</code> and original audio directories.</p>
    {''.join(listening)}<h2>All S04 calls</h2><p>Ordered by date, recording, onset and selection. The subset omits other call types and cannot represent a complete call sequence or identify callers.</p>
    <div class="scroll">{table}</div><p><a href="s04_profile.json">Descriptive measurements</a></p></html>'''
    (out/'inspection.html').write_text(page,encoding='utf-8')
    print('Open:',out/'inspection.html')


def train(args):
    from .training import train_baseline
    train_baseline(args)


def fit_detector(args):
    from .detector import fit_detector as run
    run(args)


def detect(args):
    from .detector import detect as run
    run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--manifest',default='data/manifests/s04_calls.csv')
    common.add_argument('--features',default='data/features/s04_features.csv')
    v = sub.add_parser('validate',parents=[common],help='Validate included metadata/features/folds; no audio needed')
    v.add_argument('--recordings',default='data/manifests/recordings.json')
    v.add_argument('--folds',default='data/manifests/date_folds.json'); v.set_defaults(action=validate)
    d = sub.add_parser('download',help='Download original audio and source annotation tables')
    d.add_argument('--recordings',default='data/manifests/recordings.json')
    d.add_argument('--out',default='data/raw'); d.add_argument('--plan',action='store_true')
    d.add_argument('--limit',type=int); d.set_defaults(action=download)
    e = sub.add_parser('extract',parents=[common],help='Recompute call features from original audio')
    e.add_argument('--audio-roots',nargs='+',default=['data/raw/audio'])
    e.add_argument('--audio-map'); e.add_argument('--out',default='outputs/s04_features.csv'); e.set_defaults(action=extract)
    i = sub.add_parser('inspect',parents=[common],help='Build descriptive report and optional listening previews')
    i.add_argument('--audio-roots',nargs='+',default=['data/raw/audio']); i.add_argument('--audio-map')
    i.add_argument('--with-audio',action='store_true'); i.add_argument('--max-previews',type=int,default=12)
    i.add_argument('--out',default='outputs/inspection'); i.set_defaults(action=inspect)
    t = sub.add_parser('train',parents=[common],help='Train a grouped baseline after adding at least one other class')
    t.add_argument('--out',default='outputs/baseline'); t.set_defaults(action=train)
    fd = sub.add_parser('fit-detector',help='Fit and date-evaluate the weakly supervised S04 window detector')
    fd.add_argument('--manifest',default='data/manifests/s04_calls.csv')
    fd.add_argument('--recordings',default='data/manifests/recordings.json')
    fd.add_argument('--audio-roots',nargs='+',default=['data/raw/audio']); fd.add_argument('--audio-map')
    fd.add_argument('--annotation-roots',nargs='+',default=['data/raw/annotations']); fd.add_argument('--annotation-map')
    fd.add_argument('--config',help='Optional JSON overrides for detector settings')
    fd.add_argument('--out',default='outputs/detector_v1'); fd.set_defaults(action=fit_detector)
    sc = sub.add_parser('detect',help='Scan continuous FLAC/WAV audio for candidate S04 events')
    sc.add_argument('--audio',nargs='+',required=True,help='Audio files or directories containing FLAC/WAV files')
    sc.add_argument('--model',default='models/s04_detector.json'); sc.add_argument('--channel',type=int,default=1)
    sc.add_argument('--threshold',type=float); sc.add_argument('--preview-limit',type=int,default=8)
    sc.add_argument('--save-windows',action='store_true'); sc.add_argument('--out',default='outputs/detections');sc.set_defaults(action=detect)
    args = p.parse_args()
    if getattr(args,'limit',None) is not None and args.limit < 1:
        p.error('--limit must be positive')
    if getattr(args,'max_previews',0) < 0:
        p.error('--max-previews cannot be negative')
    if getattr(args,'preview_limit',0) < 0 or getattr(args,'channel',1) < 1:
        p.error('--preview-limit must be nonnegative; --channel must be positive')
    try:
        args.action(args)
    except Exception as error:
        p.exit(1,f'Error: {error}\n')
