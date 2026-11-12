# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-11-11

### Added
- Modular architecture splitting the previous 2,023-line `pipeline.py` into focused submodules:
  - `timing.py`: continuous affine clock conversion and cropping
  - `neural.py`: sensor grouping, whitening, band-pass/resampling, and GMM/PCA representation fitting
  - `crossval.py`: purged contiguous window cross-validation and ridge decoders
  - `embeddings.py`: MFA transcript parsing, causal context generation, and temporal alignment
  - `metrics.py`: hard/soft state-window occupancy, persistence ($p_{kk}$), dynamics, and IAAFT dwell nulls
  - `slopes.py`: within-window Boltzmann slope ($\beta$) and persistence-adjusted joint regression
  - `plots.py`: diagnostic Boltzmann fit, entropy trajectory, and acoustic control comparison plots
- `--dry-run` flag in `slep-run` to validate audio, MFA, and subject FIF availability without starting model training.
- Incremental writing of `all_subject_summaries.csv` after each participant finishes (protects against aborted batch runs).
- Git provenance logging (`git_commit` in `run_metadata.json`).
- Participant-level beta forest plot in `slep-aggregate` (saving `.png` alongside group summary).
- `scripts/lag_sensitivity.py`: automated biological response latency ($\tau$) parameter sweep.
- New test suite `tests/test_modular_components.py` bringing total test coverage to 38 test cases.

### Changed
- `pipeline.py` is now a thin orchestration script maintaining 100% backwards compatibility for all imports and CLI commands.

## [0.5.1] - 2026-11-10
- Initial consolidated pipeline release with acoustic controls (log-mel, RMS, spectral descriptors) and joint predictive models.
- IAAFT surrogate dwell null model.
- Soft-state GMM responsibility occupancy estimation.
