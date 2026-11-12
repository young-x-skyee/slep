# Repository Guidance

## Project Scope

- This Poetry project is an exploratory operationalization of SLEP Hypothesis IV on naturalistic-listening EEG/MEG. Read the [README](README.md) before changing analysis assumptions; it documents data provenance, timing, outputs, and limitations.
- Keep source changes in `src/slep_thermo/`, tests in `tests/`, and generated results under `outputs/slep_thermodynamics/`. Do not modify raw EEG/MEG, stimulus audio, canonical MFA annotations, or generated outputs unless explicitly asked.
- Use the existing Poetry environment and dependency groups in [pyproject.toml](pyproject.toml). Install test and embedding dependencies with `poetry install --with dev --with semantic` when needed.
- Run project commands from Bash; invoke Python through Poetry as `poetry run python ...`.

## Interpreting Hypothesis IV Results

- Verify run provenance before interpreting estimates: check `batch_report.json`, each participant's `run_metadata.json` and `subject_summary.csv`, plus the sensor-specific `state_window_metrics.csv` and `entropy_by_window.csv`. Confirm timing parameters, model settings, missing/failed subjects, and sample coverage.
- The current operational quantities are not physical energy or a direct Fisher-Rao implementation: `q` is GMM neural-state occupancy within a narrative window; `self_information` is `-log(q)`; `semantic_potential_nll` is held-out Gaussian NLL for predicting contextual text embeddings from neural states.
- `beta` is the weighted within-window slope of self-information against neural-only NLL. `temperature` is `1 / beta` only when `beta > 0`. Describe a positive slope as compatible with a Boltzmann-like relationship, not as proof of thermodynamic organization or SLEP.
- Treat participants as the group-level inferential unit. Do not count time samples, state-window cells, or channels as independent participants. Keep EEG, MEG magnetometers, and gradiometers separate unless a justified analysis explicitly combines them.
- These are Evoked averages and the podcast is non-stationary: the analysis does not measure trial-level equilibrium statistics. Check window-size, state-count, PCA, timing, decoder, and null-model sensitivity before making strong claims. Treat low-level acoustic features as incomplete controls, not a complete speech model.
- Keep hardware delivery/drift corrections separate from biological neural latency. Verify preprocessing provenance and apply the affine timing correction only once; do not tune timing to maximize the thermodynamic relationship.

## Implementation and Validation

- Use MNE channel metadata to select sensor types; do not assume channel order. Preserve the existing separation of sensor groups and held-out contiguous, purged temporal cross-validation.
- When `{subject}_repN.fif` files are available, use leave-one-repetition-out as a repeatability analysis; do not treat repetitions as independent participants or as generalization to new narratives.
- Optional MEG noise whitening must use participant-matched covariance with verified provenance and validated channel order. Do not assume a `fusion-cov` file is an empty-room noise covariance; leave whitening off when provenance is uncertain.
- Fit robust sensor scaling, PCA, and GMM state codebooks on each fold's training samples only. Keep complete narrative windows within a held-out fold so fold-specific state labels are not mixed within a metric window.
- For analysis or interpretation changes, state a falsifiable hypothesis and a focused check first. Add or update tests for changed behavior, especially timing, alignment, occupancy, decoding, or participant-level aggregation.
- Run a focused test while iterating, then the full suite with `poetry run python -m pytest`. The main test areas are [timing corrections](tests/test_timing_corrections.py), [audio controls](tests/test_audio_control_pipeline.py), [audio features](tests/test_audio_features.py), and [aggregation](tests/test_aggregate.py).
- Update the README when changing CLI behavior, output schemas, timing conventions, or interpretation of reported quantities. Keep conclusions tied to the measured proxy and identify unresolved assumptions.
