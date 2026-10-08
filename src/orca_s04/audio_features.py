"""Deterministic call descriptors from original audio, never inspection WAVs.

No learned transforms or annotation frequency boxes are used here. Changing
FEATURE_CONFIG changes the representation and requires a new training run.
"""
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct

FEATURE_CONFIG = dict(version=1, sample_rate=64000, window_samples=2048,
                      hop_samples=512, min_hz=80.0, max_hz=24000.0,
                      bands=32, cepstral_coefficients=16, floor_db=-80.0)
BASE_FEATURES = ['duration_s', 'rms_ac_digital', 'spectral_peak_hz', 'spectral_centroid_hz']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_audio(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype='float32', always_2d=True)
    except ImportError:
        if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
            raise RuntimeError('Install soundfile, or ffmpeg and ffprobe.')
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
            'stream=sample_rate,channels', '-of', 'json', str(path)]))['streams'][0]
        sr, channels = int(info['sample_rate']), int(info['channels'])
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
            '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        x = np.frombuffer(raw, dtype='<f4').reshape(-1, channels)
    if not np.isfinite(x).all() or not len(x):
        raise ValueError('Invalid decoded audio: ' + str(path))
    return x, int(sr)


def audio_index(roots):
    index = {}
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('Missing audio directory: ' + str(root))
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() in {'.flac', '.wav'} and path.is_file():
                index.setdefault(path.name, []).append(path)
    return index


def resolve_audio(recording_id, index, aliases=None):
    if aliases and recording_id in aliases:
        path = Path(aliases[recording_id]).resolve()
        if not path.is_file():
            raise ValueError('Missing explicitly mapped audio: ' + str(path))
        return path
    matches = list(dict.fromkeys(p.resolve() for p in index.get(Path(recording_id).name, [])))
    if not matches:
        raise ValueError('Cannot find original audio ' + recording_id + '. Supply --audio-roots.')
    if len(matches) > 1 and len({sha256(p) for p in matches}) > 1:
        raise ValueError('Different audio files share the name ' + recording_id)
    return matches[0]


def filterbank(frequencies, cfg, mel):
    if mel:
        lo = 2595 * np.log10(1 + cfg['min_hz'] / 700)
        hi = 2595 * np.log10(1 + cfg['max_hz'] / 700)
        edges = 700 * (10 ** (np.linspace(lo, hi, cfg['bands'] + 2) / 2595) - 1)
    else:
        edges = np.linspace(cfg['min_hz'], cfg['max_hz'], cfg['bands'] + 2)
    weights = np.maximum(0, np.minimum(
        (frequencies[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
        (edges[2:, None] - frequencies[None, :]) / (edges[2:] - edges[1:-1])[:, None]))
    if (weights.sum(axis=1) == 0).any():
        raise ValueError('Frequency resolution too low for configured bands.')
    return weights


def log_band_power(power, weights, cfg):
    band = weights @ power
    # Per-frame relative energy removes absolute recording gain. These are
    # fixed transforms on each call; no statistics from other calls are used.
    band /= np.maximum(band.sum(axis=0, keepdims=True), np.finfo(float).tiny)
    return 10 * np.log10(np.maximum(band, 10 ** (cfg['floor_db'] / 10)))


def describe_call(samples, sr, cfg=None):
    cfg = FEATURE_CONFIG if cfg is None else cfg
    raw = np.asarray(samples, dtype=np.float32)
    x = raw.astype(np.float64)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('A call must contain at least two finite samples.')
    duration = len(x) / sr
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if np.max(np.abs(x)) <= np.finfo(float).tiny:
        raise ValueError('Silent call segment; review its annotation before training.')
    # Recompute the original four descriptors, for future prediction without
    # needing ground-truth annotation-derived acoustic columns.
    f, p = signal.welch(raw, fs=sr, nperseg=min(2048, len(x)),
                         noverlap=min(2048, len(x)) // 2, detrend='constant')
    out = dict(duration_s=duration, rms_ac_digital=rms,
               spectral_peak_hz=float(f[np.argmax(p)]),
               spectral_centroid_hz=float(np.sum(f * p) / np.maximum(p.sum(), 1e-30)))
    # Gain-normalize before resampling for numerical stability.
    x /= np.max(np.abs(x))
    target = cfg['sample_rate']
    if sr != target:
        divisor = math.gcd(sr, target)
        x = signal.resample_poly(x, target // divisor, sr // divisor)
    n = cfg['window_samples']
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    f, _, power = signal.spectrogram(x, fs=target, window='hann', nperseg=n,
        noverlap=n - cfg['hop_samples'], detrend='constant', scaling='density', mode='psd')
    sections = np.array_split(np.arange(power.shape[1]), 3)
    sections = [s if len(s) else np.array([power.shape[1] - 1]) for s in sections]
    for name, mel in [('linear', False), ('mel', True)]:
        logpower = log_band_power(power, filterbank(f, cfg, mel), cfg)
        for i in range(cfg['bands']):
            out[f'{name}_mean_{i:02d}'] = float(logpower[i].mean())
            out[f'{name}_std_{i:02d}'] = float(logpower[i].std())
            for j, frames in enumerate(sections):
                out[f'{name}_part{j}_{i:02d}'] = float(logpower[i, frames].mean())
        if mel:
            # Relative-energy mel cepstra, not bitwise Librosa MFCCs. Coefficient
            # zero is omitted; duration and band evolution remain explicit.
            cepstra = dct(logpower, axis=0, norm='ortho')[1:cfg['cepstral_coefficients'] + 1]
            for i, values in enumerate(cepstra):
                out[f'cep_mean_{i:02d}'] = float(values.mean())
                out[f'cep_std_{i:02d}'] = float(values.std())
                out[f'cep_change_{i:02d}'] = float(values[sections[-1]].mean() - values[sections[0]].mean())
    return out


def extract_features(data, roots, cfg=None, aliases=None):
    cfg = FEATURE_CONFIG if cfg is None else cfg
    index, rows, provenance = audio_index(roots), [], []
    for recording_id, calls in data.groupby('recording_id', sort=True):
        path = resolve_audio(str(recording_id), index, aliases)
        print(f'Extracting {len(calls)} calls: {path.name}', flush=True)
        audio, sr = read_audio(path)
        provenance.append(dict(recording_id=str(recording_id), filename=path.name,
                               sha256=sha256(path), bytes=path.stat().st_size,
                               sample_rate=sr, decoded_samples=len(audio), channels=audio.shape[1]))
        for row in calls.itertuples():
            start_s, end_s, channel = float(row.start_s), float(row.end_s), float(row.channel)
            if not (np.isfinite([start_s, end_s, channel]).all() and channel.is_integer()
                    and 0 <= start_s < end_s <= len(audio) / sr and 1 <= channel <= audio.shape[1]):
                raise ValueError('Invalid audio bounds/channel for ' + str(row.call_id))
            # Match the original extractor's sample-boundary convention.
            start = int(round(start_s * sr))
            end = min(len(audio), int(round(end_s * sr)))
            values = describe_call(audio[start:end, int(channel) - 1], sr, cfg)
            values['duration_s'] = end_s - start_s
            rows.append(dict(call_id=row.call_id, **values))
    frame = pd.DataFrame(rows).set_index('call_id').loc[data.call_id].reset_index()
    if not np.isfinite(frame.drop(columns='call_id').to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite acoustic features; inspect segments.')
    return frame, provenance


def feature_sets(frame):
    def pref(prefix):
        return [c for c in frame.columns if c.startswith(prefix)]
    return {
        'baseline': BASE_FEATURES,
        'no_amplitude': [c for c in BASE_FEATURES if c != 'rms_ac_digital'],
        'duration_centroid': ['duration_s', 'spectral_centroid_hz'],
        'linear_shape': ['duration_s'] + pref('linear_mean_') + pref('linear_std_'),
        'linear_temporal': ['duration_s'] + pref('linear_'),
        'mel_temporal': ['duration_s'] + pref('mel_'),
        'cepstral_shape': ['duration_s'] + pref('cep_'),
    }
