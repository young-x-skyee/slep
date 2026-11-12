---
name: slep-hypothesis-iv
description: "Use when interpreting SLEP Hypothesis IV results, semantic thermodynamics, Boltzmann fits, beta or temperature, state occupancy or entropy from this naturalistic-listening EEG/MEG analysis, or when improving its analysis code."
---

# SLEP Hypothesis IV Analysis

Use this workflow for result interpretation and code changes in this repository. Read the [project README](../../../README.md) for dataset provenance, run commands, output inventory, and established caveats. Inspect the implementation in [pipeline.py](../../../src/slep_thermo/pipeline.py) before describing or changing a metric; do not infer definitions from column names alone.

## Interpret a Run

1. Identify the exact output root and run. Inspect `batch_report.json`, participant `run_metadata.json` and `subject_summary.csv`, then modality-specific `summary.json`, `state_window_metrics.csv`, and `entropy_by_window.csv`. Check completed, missing, and failed participants; timing parameters; model and window settings; and coverage before interpreting estimates.
2. Use the implementation's operational definitions:
   - `q` is the fraction of samples assigned to a GMM neural state within one narrative window; `self_information = -log(q)`.
   - `semantic_potential_nll` is the held-out Gaussian decoder NLL for contextual text embeddings predicted from neural states. It is the primary operational proxy for `V(z)`, not physical energy.
   - The primary `beta` is a sample-count-weighted, within-window slope of self-information against neural-only NLL. Confidence intervals resample narrative windows. `temperature = 1 / beta` is reported only for positive beta.
   - Acoustic-only NLL is a low-level acoustic baseline. Joint audio-plus-neural NLL is a sensitivity/control model. `neural_incremental_information_nats = acoustic_only_NLL - joint_NLL`; positive values indicate incremental held-out prediction, not thermodynamic energy.
3. Evaluate the Hypothesis IV prediction as a conditional, approximate affine relationship between state self-information and the operational potential proxy. Report slope, uncertainty, fit quality, participant consistency, sensor group, and relevant control comparisons. A positive slope is compatible with the prediction; it does not establish equilibrium, causality, or the principle.
4. Check whether the measurement supports the claim: Evoked averages do not provide trial-level variability, podcast statistics are non-stationary, and PCA/GMM states are neural states rather than validated semantic states by default. Decoder NLL also depends on the embedding target, decoder, regularization, and scaling. Low-level acoustic features do not fully control phonetic, speaker, prosodic, or linguistic structure.
5. Keep EEG, magnetometers, and gradiometers separate. Make group inference across participants, never across time points or state-window cells. Present participant counts and uncertainty; distinguish exploratory estimates from confirmatory evidence. Consider shuffled or matched nulls and sensitivity to window size, state count, PCA dimension, timing, and decoder choices.
6. Verify hardware timing provenance before interpreting results. The fixed stimulus drift/delivery correction and biological neural lag are distinct; confirm preprocessing has not already applied the hardware correction. Do not tune timing or analysis choices to maximize the hypothesized relationship.

## Improve the Code

- Start from one concrete result discrepancy or failing test, trace it to the code that computes the value, state a falsifiable local hypothesis, and make the smallest testable change.
- Preserve participant-level inference, per-sensor analysis, and explicit provenance unless the task specifically changes those design choices. Fit robust sensor scaling, PCA, and GMM state codebooks on training samples only within purged, window-aligned folds.
- Treat the default biological neural lag as exploratory. Assess a range justified independently of the Hypothesis IV effect; do not select a lag by maximizing beta or fit quality.
- For MEG whitening, use only participant-matched, provenance-verified noise covariance; do not assume `fusion-cov` means empty-room noise. Keep whitening optional and sensor groups separate.
- Run project commands from Bash and invoke Python through Poetry as `poetry run python ...`.
- Add or update focused tests for the changed calculation or behavior, then run `poetry run python -m pytest` for the full suite. Update the README if metric meaning, output columns, CLI behavior, or assumptions change.
- Do not silently rewrite prior outputs or alter raw data, canonical transcript annotations, or timing parameters. Make generated analysis outputs reproducible and keep them under `outputs/slep_thermodynamics/`.

## Useful References

- [Timing tests](../../../tests/test_timing_corrections.py)
- [Audio-control tests](../../../tests/test_audio_control_pipeline.py)
- [Aggregation tests](../../../tests/test_aggregate.py)
- [Poetry configuration](../../../pyproject.toml)
