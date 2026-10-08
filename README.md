# S04 Orca Detection Baseline

An experimental baseline that **scans continuous FLAC/WAV recordings and returns
candidate S04 call intervals**. Baseline for validating feature extraction before analyzing larger instances of audio.The repository also provides a reproducible
dataset of annotated S04 examples, feature extraction, and listening tools.

The detector is deliberately simple: overlapping 0.8-second windows, spectral
features, balanced logistic regression, and thresholding/merging. A small fitted
JSON model is included, so scanning new audio does not require annotation files
or retraining.

**Status:** working prototype with weak localization and many false candidates.
Review its output before using it as labels. It is an offline recording scanner;
live streaming and generalization to new sites/instruments have not been tested.

## Installation

Use a standalone checkout or extract this archive into its own directory. Run
commands from the repository root. Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```bash
python -m pip install -e .
python scripts/orca.py --help
```

After installation, `orca-s04` is an alternative to `python scripts/orca.py`.
If environment activation is unavailable, invoke the environment's Python
directly (`.venv/bin/python` or `.venv\Scripts\python.exe`).

## Scan continuous audio

Use the included model on a FLAC/WAV recording sampled at 32 kHz or higher:

```bash
python scripts/orca.py detect \
  --audio path/to/recording.flac \
  --out outputs/my_scan
```

`path/to/recording.flac` is a placeholder for an actual recording. To scan a
directory, pass that directory; the command reads its immediate FLAC/WAV files.
Multiple file or directory arguments are supported. Filenames must be unique
within a run. Channel numbers start at 1.

```bash
python scripts/orca.py detect \
  --audio path/to/audio_directory \
  --channel 1 \
  --threshold 0.6 \
  --preview-limit 8 \
  --save-windows \
  --out outputs/directory_scan
```

Open `outputs/my_scan/report.html` in a browser. Outputs include:

| File                | Contents                                                           |
| ------------------- | ------------------------------------------------------------------ |
| `candidates.csv`    | Approximate start/end times, peak/mean score and recording/channel |
| `report.html`       | Candidate table and listening/spectrogram previews                 |
| `run.json`          | Model hash, settings, input hashes and training-audio flags        |
| `window_scores.csv` | All scored windows, when `--save-windows` is supplied              |
| `clips/`, `plots/`  | Up to `--preview-limit` highest-scoring examples                   |

**Scores are uncalibrated model outputs, not verified probabilities.** Raising
`--threshold` demands stronger window scores; validate the tradeoff on reviewed
data. Boundaries are approximate because a 0.8-second context window is scored
every 0.2 seconds. Listening copies are independently normalized. Original audio
is unchanged. Use a new output directory for every run.

The scanner loads each recording into memory; five-minute recordings are the
tested scale. Longer files may require substantial memory. It does not require
source labels, selection numbers or pre-cut clips at inference time.

## Included model and evaluation

`models/s04_detector.json` is a portable fitted linear model containing feature
settings, scaler statistics, coefficients and training provenance. It is small
enough to commit to the repository and does not use pickle/joblib.

Training used five recordings containing **137 high-confidence S04 references**:

| Date       | S04 calls |
| ---------- | --------: |
| 2022-07-10 |        39 |
| 2022-07-11 |        14 |
| 2022-07-13 |         6 |
| 2022-07-25 |        52 |
| 2022-07-29 |        26 |

The fixed detector was evaluated by holding out entire recording dates. At the
default threshold of 0.5, the five held-out scans produced:

| Annotation-referenced event metric |   Result |
| ---------------------------------- | -------: |
| Matched S04 references             | 33 / 137 |
| Recall                             |    24.1% |
| Precision                          |    14.9% |
| Candidate events                   |      221 |
| Unmatched candidate events         |      188 |

See `reports/detector/report.html` have results per day
and label-source counts. The final model was then refitted on all development
recordings. These recordings have been used during development; the reported scores
are exploratory development results, not an independent final test. The scan
output marks exact audio files used for fitting by hash, even if renamed.

## Download the example recordings

Audio is not included in Git. The five original recordings and annotation tables
total approximately 201 MB. The downloader verifies pinned archive versions,
published byte counts and MD5 checksums.

```bash
python scripts/orca.py download --plan
python scripts/orca.py download --limit 1
python scripts/orca.py detect --audio data/raw/audio --out outputs/example_scan
```

## Reproduce detector training

After downloading all five recordings and original annotation tables:

```bash
python scripts/orca.py download
python scripts/orca.py fit-detector --out outputs/detector_v1
```

To use existing data directories:

```bash
python scripts/orca.py fit-detector \
  --audio-roots path/to/original_audio \
  --annotation-roots path/to/original_annotations \
  --out outputs/detector_v1
```

Optional `--audio-map` and `--annotation-map` JSON files map original recording
IDs to exact local paths when files were renamed. Training verifies original
source bytes against the included catalog. For a new annotated dataset, provide
an updated positive manifest and recording catalog rather than silently
substituting changed labels.

Training writes a new `detector.json`, held-out candidate intervals, per-date
event results, window labels/scores, split audit, and HTML report. To use the new
model:

```bash
python scripts/orca.py detect \
  --model outputs/detector_v1/detector.json \
  --audio path/to/new_recording.flac \
  --out outputs/new_scan
```

The settings are fixed rather than optimized against these test folds. Optional
`--config settings.json` overrides named defaults for a new development run.
Tune on separate development dates or use nested selection; preserve fresh
dates for final evaluation.

## Inspect known S04 examples

The repository retains `data/manifests/s04_calls.csv`, the precomputed per-call
feature table, source provenance and development date folds. These describe
known annotated calls, separately from the detector's fixed-window features.

```bash
python scripts/orca.py validate
python scripts/orca.py inspect \
  --with-audio --max-previews 137 \
  --out outputs/known_call_inspection
```

Open the generated HTML report in a browser. `reports/inspection.html` is also
included as a ready-to-open metadata-only report. Quantitative features use
original samples, not normalized preview WAVs. The older `train` command remains
a segmented-call classifier scaffold requiring at least two reviewed labels;
`fit-detector` is the continuous-audio baseline described here.

Next improvements are reviewed hard negatives, S04 examples from
additional independent dates, more stable acoustic features, and better event
boundaries. See `docs/ROADMAP.md`, `docs/FEATURES.md`, and
`docs/DATA_SOURCES.md`. Predictions are candidates for review; they should not
automatically replace source annotations or establish biological meaning.

## Tests and repository hygiene

```bash
python -m unittest discover -s tests -v
```

CI runs these tests and the dataset checks without downloading source audio.
Raw audio, environments, generated outputs and model binaries are ignored by
Git. The small JSON detector and evaluation summary are included. Keep one
checkout and reuse existing audio directories to avoid duplicating recordings.
