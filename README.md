# SLEP Hypothesis IV: Kymata SOTO EEG/MEG + podcast audio

**Project version: 0.5.2**

This Poetry project runs an exploratory, coarse-grained test of the Semantic Thermodynamic Hypothesis on the Kymata English naturalistic-podcast dataset. It reads MNE Evoked FIF averages and the original experimental podcast WAV.

The expected stimulus directory is:

```text
slep_thermo_project/
├── src/
│   └── slep_thermo/
│       └── data/
│           ├── mfa_word.txt          # canonical bundled transcript
│           ├── mfa_word_stime.txt    # canonical bundled word onsets
│           └── mfa_manifest.json     # provenance and validation metadata
└── stimuli/
    └── audio/
        ├── stimulus.wav             # exact edited experimental podcast (add locally)
        └── README_audio.txt
```

The actual audio file is not included in this ZIP; copy the dataset's exact edited experimental stimulus to `stimuli/audio/stimulus.wav`, or specify another path with `--audio`. The MFA files are bundled once under `src/slep_thermo/data/`; do not duplicate them under `stimuli/audio/`.

## What the analysis does

For each participant and sensor type, the pipeline:

1. Reads `{subject}-ave.fif` with MNE and uses the FIF metadata to identify EEG, MEG magnetometers and planar gradiometers.
2. Applies an affine mapping between the WAV/MFA clock and the Evoked clock, including constant stimulus-delivery latency, progressive stimulus shift/drift and a separate neural response latency. It crops the neural data to the mapped podcast interval, not simply to the nominal WAV interval on the Evoked clock.
3. Downsamples the cropped sensor data from its original sampling rate (expected 1000 Hz) to 50 Hz using polyphase anti-alias resampling. The resulting EMEG sample times are then mapped back onto the audio timeline. No additional band-pass is applied unless `--bandpass` is specified.
4. Reads `stimuli/audio/stimulus.wav`, downmixes stereo to mono for acoustic controls, and extracts centred 25-ms frames at 50 Hz.
5. Computes low-level acoustic control features: 40 log-mel bands, log-RMS, zero-crossing rate, normalised spectral centroid, normalised spectral bandwidth, normalised 85%-rolloff, and spectral flux. The WAV is internally resampled to 16 kHz for feature extraction; the original file is not changed.
6. Aligns acoustic features and MFA-derived semantic targets to the corrected audio timeline. The affine inverse mapping already accounts for the delivery delay, drift and neural lag, so the lag is not applied a second time. Audio features are used as acoustic nuisance/control predictors, not as semantic energy.
7. Reads the single canonical MFA transcript and word-onset files from `src/slep_thermo/data/` (also included as package data when installed), removes `<sp>` rows with their matching onset entries, and creates causal 12-word contextual embeddings.
8. Uses purged contiguous folds over complete narrative windows. Within each fold, robust channel scaling, PCA and the GMM state codebook are fit on training samples only; held-out windows receive state assignments from that fold's codebook. Three semantic decoders are evaluated:
   - **Neural-only:** semantic target from neural latent state. Its held-out Gaussian NLL is the primary operational proxy for semantic potential, `V(z)`.
   - **Acoustic-only:** semantic target from audio acoustic features. This is the low-level acoustic baseline.
  - **Joint:** semantic target from neural state plus acoustic features. This tests whether neural information improves semantic prediction over and above measured acoustics.
  The semantic decoders, latent transforms and state assignments all use the same purged fold boundaries. The biological neural lag remains exploratory and should be assessed over an independently justified range, not selected to maximize the Hypothesis IV relationship.
9. Estimates coarse-grained neural-state occupancy `q`, self-information `-log(q)`, and within-window slopes against all three decoder NLL measures.
10. Reports windowed entropy, state transition probabilities and dwell times. Transition and dwell summaries are fold-specific: state IDs from independently fitted fold codebooks are not pooled or compared across folds.

The key acoustic-control comparison is:

```text
neural_incremental_information_nats = acoustic_only_NLL - joint_NLL
```

A positive value means that adding the neural representation improves held-out semantic prediction beyond the acoustic features. It is an incremental predictive measure, not a thermodynamic quantity.

**Interpretation:** this is not a direct measurement of biological or physical energy. The primary `V(z)` proxy remains the held-out neural-only decoder NLL, matching the operational form `-log p(x_semantic | z)`. The acoustic-only and joint models are controls/sensitivity analyses. The implementation does not calculate the paper's full Fisher–Rao geometry or prove SLEP.

## Expected neural data layout

The default participant IDs are stored in `config/participants.txt`:

```text
pilot_01
pilot_02
participant_01
participant_01b
participant_02
participant_03
participant_04
participant_05
participant_07
participant_08
participant_09
participant_10
participant_11
participant_12
participant_13
participant_14
participant_15
participant_16
participant_17
participant_18
participant_19
```

The code expects files such as:

```text
<data-dir>/participant_01-ave.fif
<data-dir>/participant_02-ave.fif
```

It does not assume channel order. For the stated 370-channel files, check that metadata identifies 64 EEG, 102 magnetometers and 204 planar gradiometers. These sensor groups are analysed separately because they have different units and noise properties.

## Create the Poetry environment

Use Python 3.11 if available on the cluster:

```bash
cd /path/to/slep_thermo_project
poetry config virtualenvs.in-project true --local
poetry env use python3.11
poetry install --with semantic
```

The environment is stored in `.venv/`. The `semantic` group installs SentenceTransformers and PyTorch dependencies. The first analysis run downloads `sentence-transformers/all-MiniLM-L6-v2` if it is not already cached. On clusters without internet access, cache the model on a connected machine or provide `--word-embeddings`.

Check the environment:

```bash
poetry run python -c "import mne, numpy, sklearn; print('MNE', mne.__version__); print('NumPy', numpy.__version__); print('scikit-learn', sklearn.__version__)"
```

## Inspect audio and MFA alignment

Copy the WAV to the expected location, then inspect it:

```bash
poetry run slep-inspect-audio
poetry run slep-inspect-mfa
```

The audio inspection reports duration, original sample rate, channel shape, extracted feature matrix dimensions and feature names. The MFA files each contain 1,656 aligned rows, including 260 `<sp>` markers; after paired pause removal, 1,396 lexical words remain. The final lexical word (`follow`) begins at 401.1 s and the final pause marker is at 401.5 s.

The pipeline checks that the last lexical onset is inside the WAV duration. If it is not, it stops with an error: confirm that the WAV is the exact edited experimental excerpt and that the WAV and MFA time origins agree. It does not silently pad or truncate the stimulus.

## Inspect one FIF file

Before running the full batch:

```bash
poetry run python scripts/inspect_evoked.py \
  /imaging/projects/cbu/kymata/data/dataset_4-english_narratives/interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/participant_01-ave.fif
```

Check the actual Evoked time axis and channel-type counts. The code assumes the podcast onset is at `Evoked.times == 0` by default; this must be verified from the preprocessing/epoching code or event alignment. The dataset descriptor identifies trigger value `3` on `STI101` as the conversation onset. In an averaged Evoked file, the trigger channel may no longer be present.

## Run a pilot and then all participants

Set paths:

```bash
DATA_ROOT=/imaging/projects/cbu/kymata/data/dataset_4-english_narratives
DATA_DIR="$DATA_ROOT/interim_preprocessing_files/3_trialwise_sensorspace/evoked_data"
OUT="outputs/slep_thermodynamics"
```

Run one participant first. The default audio path is `stimuli/audio/stimulus.wav`. The default MFA paths are the canonical package-data files under `src/slep_thermo/data/`. The analysis duration defaults to the actual WAV duration; do not set it to the longer FIF duration merely because those extra samples are present.

```bash
poetry run slep-run \
  --data-dir "$DATA_DIR" \
  --out "$OUT" \
  --subjects participant_01 \
  --stimulus-onset 0.0 \
  --stimulus-shift-correction 0.0005404 \
  --stimulus-delivery-latency 0.026 \
  --neural-lag-s 0.2
```

The drift and delivery-delay values above are the fixed corrections supplied for this recording setup. They are also the CLI defaults, so the two timing flags may be omitted without changing the result. `--neural-lag-s` remains separate from the hardware/presentation correction; 0.2 s is an exploratory biological response-latency starting value, not part of the supplied fixed calibration. If the stimulus WAV is stored elsewhere, add `--audio /path/to/stimulus.wav`. The `--words` and `--word-times` options can still override the bundled MFA files, but normally should not be needed.

Then run all supplied participants with the same global timing parameters:

```bash
poetry run slep-run \
  --data-dir "$DATA_DIR" \
  --out "$OUT" \
  --stimulus-onset 0.0 \
  --stimulus-shift-correction 0.0005404 \
  --stimulus-delivery-latency 0.026 \
  --neural-lag-s 0.2
```

For a repeatability analysis, the preprocessing directory must also contain at least two aligned `{subject}_repN.fif` files per subject. Leave-one-repetition-out CV fits sensor scaling, PCA, GMM and semantic decoders on all other repetitions, then evaluates the held-out repetition on the same narrative timeline:

```bash
poetry run slep-run \
  --data-dir "$DATA_DIR" \
  --out outputs/slep_thermodynamics/repetition_cv_lag_0.2 \
  --neural-lag-s 0.2 \
  --cv-scheme leave-one-repetition-out
```

This estimates repeatability for the same narrative, not generalization to new narratives. Repetition-level metrics are combined within participant; repetitions are not independent group-level observations. The default `--cv-scheme temporal` remains the purged contiguous-window analysis.

An optional participant-matched noise covariance can whiten MEG sensor groups before scaling and downsampling:

```bash
poetry run slep-run \
  --data-dir "$DATA_DIR" \
  --out outputs/slep_thermodynamics/repetition_cv_whitened \
  --neural-lag-s 0.2 \
  --cv-scheme leave-one-repetition-out \
  --noise-cov-dir /path/to/verified/noise_covariance_files \
  --noise-cov-pattern "{subject}-emptyroom-cov.fif"
```

Whitening is off by default, applies only to MEG magnetometers/gradiometers, and validates covariance channel names/order. Use only covariance files whose provenance confirms they estimate sensor noise for the corresponding participant and preprocessing; do not assume files named `fusion-cov` are empty-room noise estimates.

If a participant/session genuinely has a different calibrated timing value, create a CSV from `config/timing_corrections_template.csv` and pass `--timing-corrections /path/to/timing_corrections.csv`. Columns are `subject`, `stimulus_shift_correction`, `stimulus_delivery_latency_s`, and optionally `neural_lag_s`. Blank values fall back to the fixed global defaults above. The file supports per-subject overrides; it does not estimate the parameters.

Missing FIF files are reported and do not stop the batch. Subject-level failures are logged in `batch_report.json`; use `--stop-on-error` to stop at the first failure. Existing completed subjects are skipped unless `--overwrite` is supplied.

Useful options:

```text
--audio                    WAV path (default: project/stimuli/audio/stimulus.wav)
--words                    MFA transcript override (default: package data)
--word-times               MFA onset override (default: package data)
--audio-feature-cache      acoustic feature cache (default: OUT/acoustic_features_50hz.npz)
--target-sfreq             neural and acoustic feature rate (default: 50 Hz)
--n-mels                   log-mel bands (default: 40)
--audio-frame-ms           centred audio frame length (default: 25 ms)
--cv-scheme                 temporal (default) or leave-one-repetition-out
--noise-cov-dir             optional participant-matched covariance directory (MEG only)
--noise-cov-pattern         covariance filename template (default: {subject}-emptyroom-cov.fif)
--neural-lag-s             biological response latency (default: 0.2 s; validate empirically)
--stimulus-shift-correction progressive drift in seconds/second (default: 0.0005404)
--stimulus-delivery-latency constant stimulus delivery delay in seconds (default: 0.026 s)
--timing-corrections       optional per-subject timing override CSV
--duration                 podcast interval (default: actual WAV duration)
--max-subjects             optional quick pilot limit
--n-dwell-surrogates        per-window IAAFT dwell null replicates (default: 0, disabled)
--iaaft-iterations           amplitude-adjustment iterations per channel/window (default: 100)
--overwrite                rerun existing subject analyses
```

## Timing correction: delay and progressive shift/drift

The timing convention follows the supplied Kymata grid-search example. Let `t_audio` be a timestamp on the WAV/MFA clock and `t0` be the nominal stimulus-onset time in `Evoked.times`. The corresponding EMEG time at which a neural response is sampled is:

```text
t_evoked = t0 + neural_lag_s + stimulus_delivery_latency_s
           + (1 + stimulus_shift_correction) * t_audio
```

The inverse mapping used by the pipeline is:

```text
t_audio = (t_evoked - t0 - neural_lag_s - stimulus_delivery_latency_s)
          / (1 + stimulus_shift_correction)
```

This is applied continuously to every analysis sample. In the supplied `plain.py`, the same correction is applied to each split's start index as `i * split_duration * (1 + stimulus_shift_correction)` plus the constant delivery-latency offset. The continuous affine mapping is appropriate here because the thermodynamic analysis uses a continuous time series rather than split-wise grid-search windows.

**Sign convention:**

- Positive `stimulus_delivery_latency_s` moves the EMEG sampling time later relative to the nominal audio/trigger time.
- Positive `stimulus_shift_correction` makes the EMEG timeline stretch relative to the audio timeline; the discrepancy accumulates with audio time.
- Negative values correct in the opposite direction.
- `neural_lag_s` is a separate response-latency term (analogous to `start_latency` in the supplied example). It is not the hardware delivery delay. The existing 0.2-s default is an exploratory assumption and should be checked rather than treated as a measured parameter.

The fixed values configured for this project are `stimulus_shift_correction = 0.0005404 s/s` and `stimulus_delivery_latency_s = 0.026 s`. With the exploratory `neural_lag_s = 0.200 s` and `t0 = 0`, an event at 200 s on the audio timeline maps to:

```text
200 * (1 + 0.0005404) + 0.026 + 0.200 = 200.33408 s on the Evoked timeline
```

At the end of a 400-s podcast, drift contributes 0.21616 s and delivery latency contributes 0.026 s, for a total hardware-timing offset of 0.24216 s (before adding the separate neural response lag).

The pipeline crops the EMEG interval corresponding to `[0, WAV_duration)` under this mapping, resamples the neural data, then maps resampled sample times back to the original audio/MFA timeline. This means word onsets and audio features are not shifted a second time.

### Provenance and use of the fixed timing parameters

The supplied `plain.py` **applies** the correction parameters; it does not estimate them. For this project, the supplied fixed values are configured as the command-line defaults:

- `stimulus_shift_correction = 0.0005404` seconds per second.
- `stimulus_delivery_latency_s = 0.026` seconds.

The pipeline applies these values once in the affine audio-to-Evoked mapping. Do not also shift the WAV, MFA timestamps, or `Evoked.times` manually. The values can be overridden explicitly on the command line or for an individual participant through the timing CSV, but the default analysis uses the fixed values above.

The biological response latency `neural_lag_s` is separate. Its default of 0.2 s remains an exploratory starting value and should be checked using an independent auditory-response analysis. Do not tune it to maximise the thermodynamic relationship.

Before the full batch, verify from the preprocessing code that the FIF time axis has not already been corrected using these same drift and delivery-delay values. The dataset paper notes that presentation delays are handled in the experiment configuration; if the corrections have already been applied to the timing of the saved FIF data, do not apply them again. If the FIF data remain on the acquisition clock, use the defaults above as configured.

The same fixed hardware corrections are applied to EEG, magnetometers and gradiometers. Use the timing CSV only when acquisition records establish genuine participant/session-specific differences; blank CSV entries inherit the fixed global defaults.

## Group-level aggregation

```bash
poetry run slep-aggregate \
  --results-root "$OUT" \
  --out "$OUT/group_summary.csv"
```

The inferential unit is the participant, not the number of time points or GMM state/window cells. The group summary includes the primary neural-only beta, acoustic-only baseline beta, joint audio+neural beta, mean incremental neural predictive information, and mean normalised entropy (where available).

## Main outputs

```text
<OUT>/
  acoustic_features_50hz.npz
  acoustic_features_50hz.npz.json
  semantic_context_embeddings.npy
  semantic_context_embeddings.npy.json
  batch_report.json
  all_subject_summaries.csv
  group_summary.csv
  <subject>/
    subject_summary.csv
    run_metadata.json
    EEG/
      state_window_metrics.csv
      entropy_by_window.csv
      transition_matrix.csv
      dwell_summary.csv
      dwell_surrogate_null.csv  # only when --n-dwell-surrogates > 0
      EEG_boltzmann_fit.png
      EEG_entropy_over_time.png
      EEG_acoustic_control_comparison.png
      summary.json
    MEG_mag/...
    MEG_grad/...
```

The state/window table includes `q_k`, `self_information`, `p_kk` (same-state probability at the next sample within the same fold and window), neural-only semantic potential `V_k`, acoustic-only baseline NLL, joint NLL and incremental neural predictive information. `summary.json` reports a window-bootstrap regression of self-information on both `V_k` and `p_kk`, so the potential association is adjusted for persistence. The subject summary includes the primary `beta` and `T=1/beta`, plus acoustic-only, joint-model and persistence-adjusted coefficients.

When enabled, `dwell_surrogate_null.csv` compares observed per-window dwell statistics with IAAFT surrogates passed through the same fold's training-fitted codebook. Each channel's empirical marginal is preserved exactly and its Fourier amplitude spectrum approximately within each held-out window. The surrogate randomizes channels independently, so it does not preserve cross-channel dependence; interpret this as a temporal-smoothness null, not a complete neural null. The default is disabled because surrogate generation can be expensive. The empirical upper-tail p-value has minimum resolution `1 / (n_surrogates + 1)`; use at least 199 surrogates for a useful exploratory tail estimate, and more for confirmatory inference.

## Important methodological caveats

1. **Evoked averages:** the default `*-ave.fif` analysis estimates occupancy across the temporal trajectory of the average, not trial-to-trial equilibrium statistics. The optional repetition-CV mode uses `{subject}_repN.fif` single-repetition Evokeds, but they remain repeated presentations of one narrative and do not provide independent participants or new-story generalization.
2. **Non-stationarity:** the podcast is non-stationary. Occupancy and entropy are estimated within 20-second windows, and slope fitting removes each window's mean. Repeat with 10-, 20- and 30-second windows.
3. **Semantic state validity:** PCA+GMM states are initially neural states, not automatically semantic states. Validate semantic information in the latent representation and compare acoustic/phonological baselines.
4. **Decoder target:** the default target is a contextual text embedding. NLL depends on embedding model, decoder, regularisation and target scaling. Test alternative semantic targets.
5. **Acoustic features:** log-mel/RMS/spectral measures are controls, not complete speech representations. They do not fully model phonetic content, speaker identity, prosody or linguistic surprisal. Consider HuBERT/Wav2Vec-style features as a later, richer control.
6. **Timing:** apply the affine drift and delivery-delay correction only after verifying values from acquisition/presentation timing or independent calibration. Keep hardware delivery delay separate from biological neural latency. Do not estimate both from the same neural/audio correlation alone, because their constant offsets are confounded. Repeat the analysis over plausible calibrated parameter ranges.
7. **Temporal autocorrelation:** use contiguous held-out folds with a purge gap. Group inference is across participants; do not treat samples as independent participants.
8. **Sensor modality:** EEG, magnetometers and gradiometers are analysed separately.
9. **Preprocessing:** the code assumes the Evoked FIF has already been preprocessed. It applies no additional band-pass by default. Do not repeat SSS/Maxwell filtering on already processed data.
10. **Dwell null:** the optional IAAFT null preserves each channel's marginal distribution exactly and its within-window Fourier amplitudes approximately. It destroys cross-channel dependence and does not model nonstationarity within each window; any observed-vs-null difference is therefore not uniquely semantic.
11. **Thermodynamic interpretation:** a positive slope is compatible with a Boltzmann-like relationship, but does not uniquely establish thermodynamic organisation or SLEP. Interpret occupancy jointly with state persistence and compare against acoustic-only and autocorrelation-preserving nulls.

## Suggested first-pilot settings

Keep the defaults initially: 50 Hz analysis sampling, 20 PCA dimensions, 10 GMM states, 20-second windows, five contiguous CV folds and a 5-second purge gap. Use the supplied fixed drift (0.0005404 s/s) and delivery delay (0.026 s) after confirming that neither has already been applied during preprocessing; validate the 0.2-s neural lag separately. Then perform sensitivity analyses over PCA dimensions (10/20/30), state counts (5/10/15), windows (10/20/30 s), and plausible timing values around the calibration. These are starting settings, not theoretically privileged values.
