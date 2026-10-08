# Sources and provenance

The source recordings and annotations are provided through the public NOAA
passive-bioacoustic archive for the DCLDE workshop dataset. The current object
prefix used by this project is:

`dclde/2027/dclde_2027_killer_whales/simres/`

The original manuscript describes the broader dataset:

Palmer, K. J. et al. **A Public Dataset of Annotated Orcinus orca Acoustic Signals
for Detection and Ecotype Classification.** Scientific Data 12, 1137 (2025).
[https://doi.org/10.1038/s41597-025-05281-5](https://doi.org/10.1038/s41597-025-05281-5)

Data DOI: [https://doi.org/10.25921/15ey-mh50](https://doi.org/10.25921/15ey-mh50).
The old NOAA metadata display may fail; this project's downloader uses the
direct public archive objects listed in `data/manifests/recordings.json`.

This repository is an independent starter built from an S04 subset of original
SIMRES Raven selection tables, not the authors' codebase or a reproduction of
the full published study. Selection numbers and source labels are preserved.
The precomputed features were measured from original FLAC samples. Absolute
audio hashes and archive versions allow the source inputs to be identified.

The accompanying recordings contain other sound types, but only the 137 S04
segments are included as training metadata here. Downloaded original annotation
tables retain their full content and are not edited by the scripts.

The continuous-audio detector also derives training windows from the complete
annotation tables: high-confidence non-S04 calls provide contrast examples,
and unannotated regions provide explicitly identified background proxies.
These proxies are weak labels rather than manually confirmed negatives. Model
training provenance and label counts accompany the fitted JSON detector.

Archive objects for all five recordings were checked against published byte
counts and MD5 values during project validation. Precomputed descriptors were
verified by recomputing them from original audio; amplitude is stored in
original digital units. Source
file hashes and code hashes are in `data/features/provenance.json`.
