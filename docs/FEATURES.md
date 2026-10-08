# Feature definitions

The feature table contains only a join key (`call_id`) and numeric acoustic
features. Source labels and recording metadata are in a separate manifest.

| Columns | Definition |
|---|---|
| duration_s | Annotated end minus onset |
| rms_ac_digital | Standard deviation of the original samples, in digital units |
| spectral_peak_hz | Frequency at maximum full-band Welch PSD |
| spectral_centroid_hz | Full-band power-weighted mean frequency |
| linear_mean_00 ... 31 | Whole-call means of relative log energy in triangular linear-frequency bands |
| linear_std_00 ... 31 | Within-call standard deviations in those bands |
| linear_part0/1/2_00 ... 31 | Band means in successive thirds of the spectrogram frames |
| mel_mean / mel_std / mel_part0/1/2 | Corresponding summaries using HTK mel-spaced triangular bands |
| cep_mean_00 ... 15 | Means of relative-energy mel cepstral coefficients 1–16 |
| cep_std_00 ... 15 | Standard deviations of those coefficients |
| cep_change_00 ... 15 | Last-third minus first-third cepstral means |

Original four scalar descriptors use sample bounds rounded to the nearest
sample and a Welch window of up to 2048 original-rate samples with 50% overlap
and per-window mean removal, as implemented in `audio_features.py`.

Rich descriptors use call-level DC removal and gain normalization, polyphase
resampling to 64 kHz, 2048-sample Hann windows (32 ms), 512-sample hops (8 ms),
32 triangular bands from 80 Hz to 24 kHz, per-frame band-power normalization,
and a relative-energy log floor at -80 dB. Cepstra are an orthonormal DCT with
coefficient zero omitted. They are not bitwise Librosa MFCCs.

These band limits are a fixed candidate representation for this pilot, not a
claim about the biological range of S04. The baseline's full-band centroid
does not use that band cutoff. Annotation frequency boxes are never used to
choose a feature's frequency range.

## Continuous-audio detector features

The detector uses a separate 56-dimensional representation of **fixed windows**,
not the 372 per-annotated-call measurements above. No call boundaries, frequency
boxes, labels, dates or pod metadata are required to extract detector features.

Audio is resampled to 32 kHz. A 1024-sample Hann spectrogram window (32 ms) with
512-sample steps (16 ms) is mean-removed per frame. Twenty-four HTK mel-spaced
triangular bands cover 80 Hz–15 kHz. Per-frame normalized log band energy is
summarized by means/standard deviations in each 0.8-second scanning window.
Additional frame-series summaries are log band power, band centroid, normalized
spectral entropy and flatness. Windows advance by 0.2 seconds, with a final
end-aligned window to cover the recording tail.

Unlike the gain-normalized per-call shape descriptors, the detector includes
absolute log-power measurements, which can vary with recording conditions.
The fixed band range is an experimental representation rather than a physical
limit on S04. Source sampling rates below 32 kHz are rejected for this model.

Scaling and logistic coefficients are fitted only on training dates and
exported as small JSON arrays. Inference reconstructs the logistic score from
those arrays; tests compare it with scikit-learn predictions. The score is
uncalibrated. Thresholded window-center cells are merged, then padded by 0.15
seconds; output intervals remain inside the decoded recording bounds.

Implementation references:
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).
