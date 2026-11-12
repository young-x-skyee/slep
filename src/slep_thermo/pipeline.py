#!/usr/bin/env python3
"""Exploratory SLEP Hypothesis IV analysis for Kymata SOTO Evoked FIF files.

This is a coarse-grained empirical operationalisation, not a measurement of
physical energy and not a direct implementation of the paper's Fisher--Rao
geometry. See README.md for assumptions and limitations.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import subprocess
import warnings

import mne
import numpy as np
import pandas as pd

from .audio import align_audio_features_to_times, extract_acoustic_features_cached
from .crossval import (
    _decoder_nll,
    crossvalidated_repetition_states_and_nll,
    crossvalidated_semantic_nll,
    crossvalidated_sensor_states_and_nll,
    purged_block_splits,
    purged_window_splits,
)
from .embeddings import (
    align_embeddings_to_time,
    embedding_cache_key,
    load_or_create_embeddings,
    make_context_texts,
    read_mfa_words,
)
from .metrics import (
    dwell_run_lengths,
    iaaft_dwell_null,
    iaaft_surrogate,
    soft_state_window_metrics,
    state_dynamics,
    state_window_metrics,
)
from .neural import (
    apply_sensor_noise_whitener,
    fit_neural_state_transform,
    get_sensor_groups,
    list_repetition_evoked_paths,
    make_sensor_noise_whitener,
    preprocess_sensor_group,
    read_evoked,
    transform_neural_state_model,
)
from .plots import save_plots
from .slopes import (
    estimate_joint_state_association,
    estimate_within_window_slope,
)
from .timing import (
    DEFAULT_STIMULUS_DELIVERY_LATENCY_S,
    DEFAULT_STIMULUS_SHIFT_CORRECTION,
    audio_time_to_evoked_time,
    crop_evoked_to_mapped_stimulus,
    crop_evoked_to_stimulus,
    evoked_time_to_audio_time,
    load_timing_corrections,
)

LOG = logging.getLogger("slep_thermo")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STIMULI_AUDIO_DIR = PROJECT_ROOT / "stimuli" / "audio"
PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_AUDIO_PATH = STIMULI_AUDIO_DIR / "stimulus.wav"
# Canonical MFA annotations live inside the Python package so there is only
# one bundled copy and they remain available after package installation.
DEFAULT_WORDS_PATH = PACKAGE_DATA_DIR / "mfa_word.txt"
DEFAULT_WORD_TIMES_PATH = PACKAGE_DATA_DIR / "mfa_word_stime.txt"

DEFAULT_SUBJECTS = [
    "pilot_01", "pilot_02", "participant_01", "participant_01b",
    "participant_02", "participant_03", "participant_04", "participant_05",
    "participant_07", "participant_08", "participant_09", "participant_10",
    "participant_11", "participant_12", "participant_13", "participant_14",
    "participant_15", "participant_16", "participant_17", "participant_18",
    "participant_19",
]


def get_git_commit_hash() -> Optional[str]:
    """Retrieve git HEAD SHA if running within a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return None


def analyse_sensor_group(
    subject: str,
    group_name: str,
    group_data: np.ndarray,
    group_times_s: np.ndarray,
    relative_times_s: np.ndarray,
    embeddings_grid: np.ndarray,
    acoustic_grid: np.ndarray,
    raw_sfreq: float,
    args: argparse.Namespace,
    subject_out: Path,
    data_is_preprocessed: bool = False,
    noise_whitener: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    n_sensors = group_data.shape[1] if group_data.ndim == 3 else group_data.shape[0]
    LOG.info("%s | %s: %d sensors", subject, group_name, n_sensors)
    if noise_whitener is not None and not data_is_preprocessed:
        group_data = apply_sensor_noise_whitener(group_data, noise_whitener)
    processed = group_data if data_is_preprocessed else preprocess_sensor_group(
        group_data, raw_sfreq, args.target_sfreq, args.bandpass,
        args.bandpass_low, args.bandpass_high,
    )
    # resample_poly returns ceil(n/down) samples. Construct exact corresponding time grid.
    down = int(round(raw_sfreq / args.target_sfreq))
    n_repetitions = processed.shape[0] if processed.ndim == 3 else 1
    n = min(processed.shape[-1], len(relative_times_s), len(embeddings_grid), len(acoustic_grid))
    processed = processed[..., :n]
    times_rel = relative_times_s[:n]
    y = embeddings_grid[:n]
    acoustic = acoustic_grid[:n]

    if args.cv_scheme == "leave-one-repetition-out":
        labels, nll_by_model, fold_ids, fold_models, responsibilities, n_pca_actual, n_states_actual = (
            crossvalidated_repetition_states_and_nll(
                processed, y, acoustic, times_rel,
                n_pca=args.n_pca,
                n_states=args.n_states,
                window_s=args.window_s,
                ridge_alpha=args.ridge_alpha,
                random_state=args.random_state,
            )
        )
        metrics_times = np.tile(times_rel, n_repetitions)
    else:
        labels, nll_by_model, fold_ids, fold_models, responsibilities, n_pca_actual, n_states_actual = (
            crossvalidated_sensor_states_and_nll(
                processed, y, acoustic, times_rel,
                n_pca=args.n_pca,
                n_states=args.n_states,
                n_splits=args.cv_splits,
                gap_s=args.cv_gap_s,
                window_s=args.window_s,
                ridge_alpha=args.ridge_alpha,
                random_state=args.random_state,
            )
        )
        metrics_times = times_rel
    metrics, entropy = state_window_metrics(
        labels, nll_by_model, metrics_times, n_states_actual, args.window_s,
        args.min_state_window_samples, fold_ids=fold_ids,
    )
    soft_metrics, soft_entropy = soft_state_window_metrics(
        responsibilities, nll_by_model, metrics_times, args.window_s,
        float(args.min_state_window_samples), fold_ids=fold_ids,
    )
    summary = estimate_within_window_slope(metrics, args.bootstrap, args.random_state)
    joint_state_association = estimate_joint_state_association(
        metrics, args.bootstrap, args.random_state + 3,
    )
    soft_summary = estimate_within_window_slope(
        soft_metrics, args.bootstrap, args.random_state + 4,
    )
    soft_joint_state_association = estimate_joint_state_association(
        soft_metrics, args.bootstrap, args.random_state + 5,
    )
    acoustic_slope = estimate_within_window_slope(
        metrics, args.bootstrap, args.random_state + 1, potential_col="acoustic_baseline_nll",
    )
    joint_slope = estimate_within_window_slope(
        metrics, args.bootstrap, args.random_state + 2, potential_col="acoustic_controlled_joint_nll",
    )
    # `times_rel` is on the audio timeline after the affine inverse mapping.
    # Its effective sampling rate differs slightly from the EMEG output rate
    # when a non-zero stretch correction is applied.
    effective_audio_sfreq = (
        1.0 / float(np.median(np.diff(times_rel))) if len(times_rel) > 1 else args.target_sfreq
    )
    summary.update({
        "beta_acoustic_only_baseline": acoustic_slope["beta"],
        "beta_audio_controlled_joint": joint_slope["beta"],
        "temperature_audio_controlled_joint": joint_slope["temperature"],
        "soft_beta": soft_summary["beta"],
        "soft_temperature": soft_summary["temperature"],
        "soft_r2_within": soft_summary["r2_within"],
        "soft_beta_ci_low": soft_summary["ci_low"],
        "soft_beta_ci_high": soft_summary["ci_high"],
        "soft_mean_entropy_norm": float(soft_entropy["entropy_norm"].mean()) if len(soft_entropy) else np.nan,
        "soft_beta_potential_adjusted_for_persistence": soft_joint_state_association[
            "beta_potential_adjusted_for_persistence"
        ],
        "soft_beta_potential_ci_low": soft_joint_state_association["beta_potential_ci_low"],
        "soft_beta_potential_ci_high": soft_joint_state_association["beta_potential_ci_high"],
        **joint_state_association,
        "mean_neural_incremental_information_nats": float(
            np.average(metrics["neural_incremental_information_nats"], weights=metrics["n_samples"])
        ) if len(metrics) else np.nan,
    })
    summary.update({
        "subject": subject, "modality": group_name,
        "n_sensors": int(processed.shape[1] if processed.ndim == 3 else processed.shape[0]),
        "n_samples": int(n * n_repetitions), "sfreq_original": float(raw_sfreq),
        "sfreq_analysis": float(args.target_sfreq),
        "sfreq_analysis_audio_time": float(effective_audio_sfreq),
        "n_pca": int(n_pca_actual),
        "n_states": int(n_states_actual),
        "n_repetitions": int(n_repetitions),
        "cv_scheme": args.cv_scheme,
        "noise_whitened": noise_whitener is not None or bool(args.noise_cov_dir),
        "mean_entropy_norm": float(entropy["entropy_norm"].mean()) if len(entropy) else np.nan,
    })

    group_out = subject_out / group_name
    group_out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(group_out / "state_window_metrics.csv", index=False)
    entropy.to_csv(group_out / "entropy_by_window.csv", index=False)
    soft_metrics.to_csv(group_out / "state_window_soft_metrics.csv", index=False)
    soft_entropy.to_csv(group_out / "soft_entropy_by_window.csv", index=False)
    trans, dwell = state_dynamics(labels, effective_audio_sfreq, n_states_actual, fold_ids)
    trans.to_csv(group_out / "transition_matrix.csv", index=False)
    dwell.to_csv(group_out / "dwell_summary.csv", index=False)
    dwell_null = iaaft_dwell_null(
        processed, metrics_times, labels, fold_models, args.window_s, effective_audio_sfreq,
        args.n_dwell_surrogates, args.iaaft_iterations, args.random_state,
    )
    if not dwell_null.empty:
        dwell_null.to_csv(group_out / "dwell_surrogate_null.csv", index=False)
        summary["dwell_surrogate_windows"] = int(len(dwell_null))
        summary["dwell_surrogate_replicates"] = int(args.n_dwell_surrogates)
    save_plots(metrics, entropy, summary, group_out, group_name)
    (group_out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    return summary


def analyse_subject(
    subject: str,
    fif_path: Path,
    words: Sequence[str],
    word_onsets: np.ndarray,
    embeddings: np.ndarray,
    audio_features: np.ndarray,
    audio_feature_times: np.ndarray,
    audio_metadata: Dict[str, object],
    out_root: Path,
    args: argparse.Namespace,
    timing_params: Dict[str, float],
) -> List[Dict[str, object]]:
    repetition_paths: List[Path] = []
    repetition_sensor_data: Dict[str, List[np.ndarray]] = {}
    if args.cv_scheme == "leave-one-repetition-out":
        repetition_paths = list_repetition_evoked_paths(fif_path.parent, subject)
        evoked = read_evoked(repetition_paths[0], args.evoked_index)
    else:
        evoked = read_evoked(fif_path, args.evoked_index)
    LOG.info(
        "%s: Evoked shape=%s, sfreq=%.3f Hz, tmin=%.3f s, tmax=%.3f s, nchan=%d",
        subject, evoked.data.shape, evoked.info["sfreq"], evoked.times[0], evoked.times[-1], len(evoked.ch_names)
    )
    # MNE's Evoked channel metadata is used; no hard-coded assumption about channel order.
    groups = get_sensor_groups(evoked, include_bads=args.include_bads)
    group_counts = {name: int(len(picks)) for name, picks in groups.items()}
    LOG.info("%s channel groups (excluding bads unless requested): %s", subject, group_counts)
    noise_covariance_path = None
    noise_whiteners: Dict[str, np.ndarray] = {}
    if args.noise_cov_dir is not None:
        noise_covariance_path = args.noise_cov_dir / args.noise_cov_pattern.format(subject=subject)
        if not noise_covariance_path.is_file():
            raise FileNotFoundError(f"Requested MEG noise covariance not found: {noise_covariance_path}")
        noise_covariance = mne.read_cov(str(noise_covariance_path), verbose="ERROR")
        for group_name in ("MEG_mag", "MEG_grad"):
            if group_name in groups:
                noise_whiteners[group_name] = make_sensor_noise_whitener(
                    noise_covariance, evoked.info, groups[group_name],
                )

    if args.cv_scheme == "leave-one-repetition-out":
        repetition_sensor_data = {name: [] for name in groups}
        selected_times = None
        timing_meta = None
        raw_sfreq = float(evoked.info["sfreq"])
        for repetition_index, repetition_path in enumerate(repetition_paths):
            repetition_evoked = evoked if repetition_index == 0 else read_evoked(
                repetition_path, args.evoked_index,
            )
            if repetition_evoked.ch_names != evoked.ch_names:
                raise ValueError(f"Channel order differs in repetition file {repetition_path}.")
            repetition_data, repetition_times, repetition_sfreq, repetition_timing = (
                crop_evoked_to_mapped_stimulus(
                    repetition_evoked,
                    stimulus_onset_s=args.stimulus_onset,
                    duration_s=args.duration,
                    stimulus_shift_correction=timing_params["stimulus_shift_correction"],
                    stimulus_delivery_latency_s=timing_params["stimulus_delivery_latency_s"],
                    neural_lag_s=timing_params["neural_lag_s"],
                )
            )
            if selected_times is None:
                selected_times = repetition_times
                timing_meta = repetition_timing
                raw_sfreq = repetition_sfreq
            elif (
                not np.isclose(repetition_sfreq, raw_sfreq)
                or repetition_times.shape != selected_times.shape
                or not np.allclose(repetition_times, selected_times, rtol=0.0, atol=1e-9)
            ):
                raise ValueError(
                    f"Repetition time grids differ for {subject}; repetition-held-out CV requires alignment."
                )
            for group_name, picks in groups.items():
                repetition_group = repetition_data[picks]
                if group_name in noise_whiteners:
                    repetition_group = apply_sensor_noise_whitener(
                        repetition_group, noise_whiteners[group_name],
                    )
                repetition_sensor_data[group_name].append(preprocess_sensor_group(
                    repetition_group, raw_sfreq, args.target_sfreq, args.bandpass,
                    args.bandpass_low, args.bandpass_high,
                ))
            LOG.info("%s loaded repetition %d/%d", subject, repetition_index + 1, len(repetition_paths))
        repetition_sensor_data = {
            name: np.stack(repetitions, axis=0)
            for name, repetitions in repetition_sensor_data.items()
        }
        data_sample_count = len(selected_times)
    else:
        data, selected_times, raw_sfreq, timing_meta = crop_evoked_to_mapped_stimulus(
            evoked,
            stimulus_onset_s=args.stimulus_onset,
            duration_s=args.duration,
            stimulus_shift_correction=timing_params["stimulus_shift_correction"],
            stimulus_delivery_latency_s=timing_params["stimulus_delivery_latency_s"],
            neural_lag_s=timing_params["neural_lag_s"],
        )
        data_sample_count = data.shape[1]

    # Resample the neural data on its own clock, then invert the affine mapping
    # to obtain the matching timestamp on the original audio/MFA clock.
    down = int(round(raw_sfreq / args.target_sfreq))
    if not np.isclose(raw_sfreq / args.target_sfreq, down):
        raise ValueError("For now, raw_sfreq/target_sfreq must be an integer (e.g. 1000/50).")
    n_target = int(np.ceil(data_sample_count / down))
    # resample_poly returns ceil(N/down) samples; trim any final sample whose
    # nominal time would lie after the last available input sample.
    n_target = min(
        n_target,
        int(np.floor((selected_times[-1] - selected_times[0]) * args.target_sfreq + 1e-9)) + 1,
    )
    analysis_evoked_times = selected_times[0] + np.arange(n_target) / args.target_sfreq
    analysis_times = evoked_time_to_audio_time(
        analysis_evoked_times,
        stimulus_onset_s=args.stimulus_onset,
        stimulus_shift_correction=timing_params["stimulus_shift_correction"],
        stimulus_delivery_latency_s=timing_params["stimulus_delivery_latency_s"],
        neural_lag_s=timing_params["neural_lag_s"],
    )
    # The inverse mapping already accounts for the neural lag and delivery delay:
    # do not shift the transcript/audio a second time here.
    y_grid = align_embeddings_to_time(words, word_onsets, embeddings, analysis_times, neural_lag_s=0.0)
    audio_grid = align_audio_features_to_times(
        audio_features, audio_feature_times, analysis_times,
        audio_duration_s=float(audio_metadata["duration_s"]),
    )
    valid_audio = np.isfinite(audio_grid).all(axis=1)
    if not np.any(valid_audio):
        raise ValueError("No audio feature frames overlap the neural analysis time axis.")
    LOG.info(
        "%s: audio features overlap %.2f%% of neural analysis samples",
        subject, 100.0 * valid_audio.mean(),
    )

    subject_out = out_root / subject
    subject_out.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    for group_name, picks in groups.items():
        if len(picks) < 8:
            LOG.warning("%s | %s: skipping group with only %d channels", subject, group_name, len(picks))
            continue
        group_summary = analyse_sensor_group(
            subject, group_name,
            repetition_sensor_data[group_name] if repetition_paths else data[picks],
            selected_times, analysis_times, y_grid, audio_grid, raw_sfreq, args,
            subject_out, data_is_preprocessed=bool(repetition_paths),
            noise_whitener=noise_whiteners.get(group_name),
        )
        group_summary.update(timing_meta)
        summaries.append(group_summary)

    pd.DataFrame(summaries).to_csv(subject_out / "subject_summary.csv", index=False)
    metadata = {
        "subject": subject,
        "git_commit": get_git_commit_hash(),
        "fif_path": str(fif_path),
        "repetition_fif_paths": [str(path) for path in repetition_paths],
        "meg_noise_covariance_path": str(noise_covariance_path) if noise_covariance_path else None,
        "evoked_comment": getattr(evoked, "comment", ""),
        "evoked_shape": list(evoked.data.shape),
        "channel_names": evoked.ch_names,
        "channel_type_counts": {kind: int(sum(mne.channel_type(evoked.info, i) == kind for i in range(len(evoked.ch_names)))) for kind in sorted(set(mne.channel_type(evoked.info, i) for i in range(len(evoked.ch_names))))},
        "sfreq_hz": float(raw_sfreq),
        "evoked_tmin_s": float(evoked.times[0]),
        "evoked_tmax_s": float(evoked.times[-1]),
        "stimulus_onset_in_evoked_time_s": float(args.stimulus_onset),
        "stimulus_duration_s": float(args.duration),
        "timing_correction": timing_meta,
        "timing_mapping": "t_evoked = stimulus_onset + neural_lag + delivery_latency + (1 + shift_correction) * t_audio",
        "analysis_audio_time_start_s": float(analysis_times[0]),
        "analysis_audio_time_end_s": float(analysis_times[-1]),
        "channel_group_counts": group_counts,
        "n_words": int(len(words)),
        "semantic_model": args.sentence_model,
        "audio": audio_metadata,
        "audio_feature_names": audio_metadata.get("feature_names", []),
        "audio_feature_n_frames": int(audio_features.shape[0]),
        "audio_predictor_lag_s": float(timing_params["neural_lag_s"]),
        "audio_time_coverage_fraction": float(valid_audio.mean()),
        "n_pca": int(args.n_pca), "n_states": int(args.n_states),
        "window_s": float(args.window_s), "target_sfreq": float(args.target_sfreq),
        "decoder_cv": {
            "scheme": args.cv_scheme,
            "n_splits": int(args.cv_splits) if args.cv_scheme == "temporal" else int(len(repetition_paths)),
            "gap_s": float(args.cv_gap_s) if args.cv_scheme == "temporal" else 0.0,
            "ridge_alpha": float(args.ridge_alpha),
            "bootstrap_replicates": int(args.bootstrap),
            "random_state": int(args.random_state),
            "representation_fit": "training samples only within each fold",
            "n_repetitions": int(len(repetition_paths)) if repetition_paths else 1,
        },
        "dwell_surrogate_null": {
            "n_surrogates": int(args.n_dwell_surrogates),
            "iaaft_iterations": int(args.iaaft_iterations),
            "method": "per-channel IAAFT within held-out narrative windows through the fixed fold codebook",
            "cross_channel_dependence_preserved": False,
        },
        "notes": [
            (
                "Each single-repetition Evoked is held out once; this tests repeatability for this narrative, not generalization to new narratives."
                if repetition_paths else
                "Input is an MNE Evoked average; this run does not estimate repetition-level variability."
            ),
            "q is occupancy of GMM-defined coarse-grained neural states within narrative windows.",
            "V is held-out Gaussian decoder NLL for contextual text embeddings predicted from neural states.",
            "This tests a discrete/coarse-grained Boltzmann-like relation, not the paper's full Fisher-Rao formulation.",
            "Biological neural lag is exploratory and should be evaluated using an independent sensitivity range.",
            "Soft-state metrics use held-out GMM posterior responsibilities; they are a sensitivity analysis, not a replacement for hard-label results.",
        ],
    }
    (subject_out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing {subject}-ave.fif files")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH,
                        help=f"Original podcast WAV (default: {DEFAULT_AUDIO_PATH})")
    parser.add_argument("--audio-feature-cache", type=Path, default=None,
                        help="Optional cached acoustic feature .npz; defaults to OUT/acoustic_features.npz")
    parser.add_argument("--n-mels", type=int, default=40,
                        help="Number of log-mel acoustic control bands (default: 40)")
    parser.add_argument("--audio-frame-ms", type=float, default=25.0,
                        help="Centred audio feature frame length in ms (default: 25)")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS_PATH,
                        help=f"Word transcript (default: {DEFAULT_WORDS_PATH})")
    parser.add_argument("--word-times", type=Path, default=DEFAULT_WORD_TIMES_PATH,
                        help=f"Word onset times in seconds (default: {DEFAULT_WORD_TIMES_PATH})")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                        help="Participant IDs; defaults to the supplied Kymata participant list")
    parser.add_argument("--fif-pattern", default="{subject}-ave.fif",
                        help="Filename pattern within data-dir; default: {subject}-ave.fif")
    parser.add_argument("--evoked-index", type=int, default=0,
                        help="Index of Evoked object in FIF file (default: 0)")
    parser.add_argument("--stimulus-onset", type=float, default=0.0,
                        help="Podcast onset in Evoked.times coordinates; verify from epoching code")
    parser.add_argument("--duration", type=float, default=None,
                        help="Podcast duration in seconds; defaults to the actual WAV duration")
    parser.add_argument("--stimulus-shift-correction", type=float,
                        default=DEFAULT_STIMULUS_SHIFT_CORRECTION,
                        help=("Affine timing drift in seconds per second; positive values stretch "
                              "EMEG time relative to audio (default: 0.0005404)"))
    parser.add_argument("--stimulus-delivery-latency", "--stimulus-delivery-latency-s",
                        dest="stimulus_delivery_latency_s", type=float,
                        default=DEFAULT_STIMULUS_DELIVERY_LATENCY_S,
                        help=("Constant delivery delay in seconds; positive values move the EMEG "
                              "sampling time later (default: 0.026)"))
    parser.add_argument("--timing-corrections", type=Path, default=None,
                        help="Optional CSV with per-subject overrides: subject, stimulus_shift_correction, stimulus_delivery_latency_s, neural_lag_s")
    parser.add_argument("--target-sfreq", type=float, default=50.0)
    parser.add_argument("--n-pca", type=int, default=20)
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--cv-scheme", choices=("temporal", "leave-one-repetition-out"),
                        default="temporal",
                        help="CV scheme; repetition mode needs {subject}_repN.fif inputs")
    parser.add_argument("--noise-cov-dir", type=Path, default=None,
                        help="Optional directory of participant-matched MNE noise covariance FIFs; whitens MEG only")
    parser.add_argument("--noise-cov-pattern", default="{subject}-emptyroom-cov.fif",
                        help="Covariance filename pattern under --noise-cov-dir")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-gap-s", type=float, default=5.0,
                        help="Temporal purge around each held-out fold")
    parser.add_argument("--context-words", type=int, default=12)
    parser.add_argument("--neural-lag-s", type=float, default=0.2,
                        help="Biological response latency in seconds, separate from hardware delivery delay (default 0.2; validate empirically)")
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--min-state-window-samples", type=int, default=20)
    parser.add_argument("--n-dwell-surrogates", type=int, default=0,
                        help="Optional per-window IAAFT dwell null replicates (default: disabled; use >=199 for useful tail resolution)")
    parser.add_argument("--iaaft-iterations", type=int, default=100,
                        help="IAAFT amplitude-adjustment iterations per channel/window (default: 100)")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--word-embeddings", type=Path, default=None,
                        help="Optional precomputed (n_words, embedding_dim) .npy; bypasses model download")
    parser.add_argument("--embedding-cache", type=Path, default=None,
                        help="Cached contextual embeddings (.npy); defaults to OUT/semantic_embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bandpass", action="store_true",
                        help="Apply additional 0.5-30 Hz filter; default is no extra band-pass")
    parser.add_argument("--bandpass-low", type=float, default=0.5)
    parser.add_argument("--bandpass-high", type=float, default=30.0)
    parser.add_argument("--include-bads", action="store_true", help="Include channels marked bad in Evoked.info")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true", help="Re-run subjects even if subject_summary.csv exists")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first failed subject")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional quick pilot limit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate audio, transcript, and data file availability without executing the analysis")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_dwell_surrogates < 0:
        raise ValueError("--n-dwell-surrogates must be nonnegative.")
    if args.iaaft_iterations < 1:
        raise ValueError("--iaaft-iterations must be positive.")
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    audio_cache = args.audio_feature_cache or (args.out / f"acoustic_features_{args.target_sfreq:g}hz.npz")
    audio_features, audio_feature_times, audio_metadata = extract_acoustic_features_cached(
        args.audio, audio_cache, target_sfreq=args.target_sfreq,
        n_mels=args.n_mels, frame_ms=args.audio_frame_ms,
    )
    LOG.info(
        "Loaded stimulus audio: %.3f s, %d Hz original, %d acoustic features at %.2f Hz",
        float(audio_metadata["duration_s"]), int(audio_metadata["original_sample_rate_hz"]),
        audio_features.shape[1], args.target_sfreq,
    )
    if args.duration is None:
        args.duration = float(audio_metadata["duration_s"])
        LOG.info("Using actual WAV duration as analysis duration: %.6f s", args.duration)
    elif args.duration <= 0:
        raise ValueError("--duration must be positive.")

    words, word_onsets = read_mfa_words(args.words, args.word_times)
    LOG.info("Loaded %d non-pause words from MFA; onset range %.3f..%.3f s",
             len(words), float(word_onsets.min()), float(word_onsets.max()))
    if float(word_onsets.max()) >= args.duration:
        raise ValueError(
            f"Last lexical word onset ({word_onsets.max():.3f}s) is not inside the requested "
            f"analysis duration ({args.duration:.3f}s). Check WAV duration and MFA/audio origins."
        )
    if float(word_onsets.max()) >= float(audio_metadata["duration_s"]):
        raise ValueError(
            f"Last MFA word onset ({word_onsets.max():.3f}s) exceeds audio duration "
            f"({audio_metadata['duration_s']:.3f}s). Check that stimulus.wav is the exact "
            "edited experimental stimulus and that MFA/audio timing share the same origin."
        )
    if float(audio_metadata["duration_s"]) < args.duration:
        warnings.warn(
            f"WAV duration ({audio_metadata['duration_s']:.3f}s) is shorter than the "
            f"requested neural interval ({args.duration:.3f}s). Neural samples after the "
            "audio ends will be excluded from decoder analyses.", RuntimeWarning,
        )
    timing_overrides = load_timing_corrections(args.timing_corrections)
    cache = args.embedding_cache or (args.out / "semantic_context_embeddings.npy")
    if args.word_embeddings is not None:
        if not args.word_embeddings.is_file():
            raise FileNotFoundError(args.word_embeddings)
        embeddings = np.asarray(np.load(args.word_embeddings), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(words):
            raise ValueError(
                f"--word-embeddings must have shape (n_words, n_features); got {embeddings.shape}, "
                f"with {len(words)} transcript words."
            )
        LOG.info("Loaded precomputed semantic embeddings from %s", args.word_embeddings)
    else:
        embeddings = load_or_create_embeddings(
            words, args.sentence_model, args.context_words, cache, args.batch_size,
        )
    subjects = args.subjects[:args.max_subjects] if args.max_subjects else args.subjects

    if args.dry_run:
        LOG.info("Dry run requested. Checking subject FIF files...")
        existing_fif = []
        missing_fif = []
        for subject in subjects:
            fif_path = args.data_dir / args.fif_pattern.format(subject=subject)
            if fif_path.is_file():
                existing_fif.append(subject)
            else:
                missing_fif.append(subject)
        LOG.info(
            "Dry run complete: %d/%d subject FIFs found. Stimulus WAV duration: %.2f s, %d words.",
            len(existing_fif), len(subjects), float(audio_metadata["duration_s"]), len(words),
        )
        if missing_fif:
            LOG.warning("Missing subjects in %s: %s", args.data_dir, missing_fif)
        return

    all_rows: List[Dict[str, object]] = []
    missing: List[str] = []
    failed: Dict[str, str] = {}
    for subject in subjects:
        fif_path = args.data_dir / args.fif_pattern.format(subject=subject)
        subject_summary_path = args.out / subject / "subject_summary.csv"
        if subject_summary_path.exists() and not args.overwrite:
            LOG.info("Skipping %s: existing result found (use --overwrite to rerun).", subject)
            try:
                existing = pd.read_csv(subject_summary_path).to_dict(orient="records")
                all_rows.extend(existing)
            except Exception as exc:
                LOG.warning("Could not read existing summary for %s: %s", subject, exc)
            continue
        if not fif_path.is_file():
            LOG.warning("Missing FIF for %s: %s", subject, fif_path)
            missing.append(subject)
            continue
        try:
            timing_params = {
                "stimulus_shift_correction": float(args.stimulus_shift_correction),
                "stimulus_delivery_latency_s": float(args.stimulus_delivery_latency_s),
                "neural_lag_s": float(args.neural_lag_s),
            }
            timing_params.update(timing_overrides.get(subject, {}))
            if 1.0 + timing_params["stimulus_shift_correction"] <= 0:
                raise ValueError(
                    f"Invalid shift correction for {subject}: 1 + shift must be positive."
                )
            LOG.info(
                "%s timing: shift=%+.9f s/s, delivery=%+.6f s, neural lag=%+.6f s",
                subject, timing_params["stimulus_shift_correction"],
                timing_params["stimulus_delivery_latency_s"], timing_params["neural_lag_s"],
            )
            subject_summaries = analyse_subject(
                subject, fif_path, words, word_onsets, embeddings,
                audio_features, audio_feature_times, audio_metadata, args.out, args,
                timing_params,
            )
            all_rows.extend(subject_summaries)
            pd.DataFrame(all_rows).to_csv(args.out / "all_subject_summaries.csv", index=False)
        except Exception as exc:
            LOG.exception("Failed to analyse %s", subject)
            failed[subject] = f"{type(exc).__name__}: {exc}"
            if args.stop_on_error:
                raise

    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.out / "all_subject_summaries.csv", index=False)
    report = {
        "requested_subjects": subjects,
        "missing_subjects": missing,
        "completed_subjects": sorted({str(row["subject"]) for row in all_rows if "subject" in row}),
        "n_completed": len({str(row["subject"]) for row in all_rows if "subject" in row}),
        "missing_fif": missing,
        "failed_subjects": failed,
        "cv_scheme": args.cv_scheme,
        "audio_path": str(args.audio),
        "audio_metadata": audio_metadata,
        "audio_feature_cache": str(audio_cache),
        "global_timing_defaults": {
            "stimulus_shift_correction": float(args.stimulus_shift_correction),
            "stimulus_delivery_latency_s": float(args.stimulus_delivery_latency_s),
            "neural_lag_s": float(args.neural_lag_s),
        },
        "timing_corrections_csv": str(args.timing_corrections) if args.timing_corrections else None,
        "out": str(args.out),
    }
    (args.out / "batch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Finished batch. %d subjects completed; %d missing.", report["n_completed"], len(missing))


if __name__ == "__main__":
    main()
