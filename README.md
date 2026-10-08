# S04 Orca Detection Baseline

An experimental baseline that **scans continuous FLAC/WAV recordings and returns
candidate S04 call intervals**. The current model is simple and relies really heavily on locality. Baseline for validating feature extraction before analyzing larger instances of audio.The repository also provides a reproducible
dataset of annotated S04 examples, feature extraction, and listening tools.

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

If environment activation is unavailable, invoke the environment's Python
directly (`.venv/bin/python` or `.venv\Scripts\python.exe`).

## Scan continuous audio

Use the included model on a FLAC/WAV recording sampled at 32 kHz or higher:

```bash
python scripts/orca.py detect \
  --audio path/to/recording.flac \
  --out outputs/my_scan
```

```bash
python scripts/orca.py detect \
  --audio path/to/audio_directory \
  --channel 1 \
  --threshold 0.6 \
  --preview-limit 8 \
  --save-windows \
  --out outputs/directory_scan
```

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

## Download the example recordings

Audio is not included in Git. The five original recordings and annotation tables
total approximately 201 MB. The downloader verifies pinned archive versions,
published byte counts and MD5 checksums.

```bash
python scripts/orca.py download --plan
python scripts/orca.py download --limit 1
python scripts/orca.py detect --audio data/raw/audio --out outputs/example_scan
```
