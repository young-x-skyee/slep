"""State-window occupancy metrics, IAAFT surrogates and state dynamics for the SLEP pipeline."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .neural import transform_neural_state_model

LOG = logging.getLogger("slep_thermo.metrics")


def iaaft_surrogate(
    signal: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int = 100,
) -> np.ndarray:
    """Return an IAAFT surrogate with the exact marginal and approximate spectrum."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 4 or not np.isfinite(signal).all():
        raise ValueError("IAAFT input must be a finite 1-D signal with at least four samples.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    sorted_signal = np.sort(signal)
    target_amplitude = np.abs(np.fft.rfft(signal))
    target_norm = max(float(np.linalg.norm(target_amplitude)), np.finfo(float).eps)
    surrogate = rng.permutation(signal)
    best = surrogate.copy()
    best_error = np.inf

    for _ in range(n_iterations):
        spectrum = np.fft.rfft(surrogate)
        phase = np.ones_like(spectrum)
        nonzero = np.abs(spectrum) > 0
        phase[nonzero] = spectrum[nonzero] / np.abs(spectrum[nonzero])
        projected = np.fft.irfft(target_amplitude * phase, n=len(signal))
        ranks = np.argsort(np.argsort(projected, kind="mergesort"), kind="mergesort")
        updated = sorted_signal[ranks]
        error = float(np.linalg.norm(np.abs(np.fft.rfft(updated)) - target_amplitude) / target_norm)
        if error < best_error:
            best, best_error = updated.copy(), error
        if np.array_equal(updated, surrogate):
            break
        surrogate = updated
    return best


def dwell_run_lengths(labels: np.ndarray, sfreq: float) -> np.ndarray:
    """Return contiguous state-run durations for one uninterrupted label sequence."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        return np.empty(0, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
    return np.diff(boundaries).astype(float) / float(sfreq)


def iaaft_dwell_null(
    sensor_time: np.ndarray,
    times_s: np.ndarray,
    labels: np.ndarray,
    fold_models: Sequence[Dict[str, object]],
    window_s: float,
    sfreq: float,
    n_surrogates: int,
    n_iterations: int,
    random_state: int,
) -> pd.DataFrame:
    """Compare held-out dwell summaries with window-wise univariate IAAFT surrogates."""
    if n_surrogates <= 0:
        return pd.DataFrame()
    sensor_time = np.asarray(sensor_time, dtype=np.float32)
    times_s = np.asarray(times_s, dtype=float)
    labels = np.asarray(labels)
    if sensor_time.ndim == 2:
        n_channels, n_time = sensor_time.shape
        n_repetitions = 1
    elif sensor_time.ndim == 3:
        n_repetitions, n_channels, n_time = sensor_time.shape
    else:
        raise ValueError("sensor_time must be channels x time or repetitions x channels x time.")
    if n_repetitions * n_time != len(times_s) or labels.shape != times_s.shape:
        raise ValueError("Sensor, repeated time, and label arrays must have matching sample dimensions.")
    if window_s <= 0 or sfreq <= 0:
        raise ValueError("window_s and sfreq must be positive.")

    rng = np.random.default_rng(random_state)
    rows = []
    for fold_model in fold_models:
        fold_id = int(fold_model["fold"])
        sample_indices = np.asarray(fold_model["sample_indices"], dtype=int)
        window_origin_s = float(fold_model["window_origin_s"])
        window_ids = np.floor((times_s - window_origin_s) / window_s).astype(int)
        model = fold_model["model"]

        for window in np.unique(window_ids[sample_indices]):
            indices = sample_indices[window_ids[sample_indices] == window]
            if len(indices) < 4:
                continue
            observed_dwell = dwell_run_lengths(labels[indices], sfreq)
            surrogate_stats = []
            for _ in range(n_surrogates):
                if sensor_time.ndim == 2:
                    source = sensor_time[:, indices]
                else:
                    repetition = int(fold_model.get("repetition", fold_id))
                    local_indices = indices - repetition * n_time
                    if np.any((local_indices < 0) | (local_indices >= n_time)):
                        raise ValueError("Fold sample indices do not match repetition-stacked sensor data.")
                    source = sensor_time[repetition, :, local_indices]
                surrogate_sensor = np.column_stack([
                    iaaft_surrogate(source[channel], rng, n_iterations)
                    for channel in range(n_channels)
                ])
                _, surrogate_labels = transform_neural_state_model(model, surrogate_sensor)
                surrogate_dwell = dwell_run_lengths(surrogate_labels, sfreq)
                surrogate_stats.append([
                    float(np.mean(surrogate_dwell)),
                    float(np.median(surrogate_dwell)),
                    float(np.percentile(surrogate_dwell, 95)),
                ])
            surrogate_stats = np.asarray(surrogate_stats, dtype=float)
            observed_stats = [
                float(np.mean(observed_dwell)),
                float(np.median(observed_dwell)),
                float(np.percentile(observed_dwell, 95)),
            ]
            row = {
                "fold": fold_id,
                "window": int(window),
                "n_samples": int(len(indices)),
                "n_observed_runs": int(len(observed_dwell)),
                "n_surrogates": int(n_surrogates),
            }
            metric_names = ("mean", "median", "p95")
            for column, metric_name in enumerate(metric_names):
                null_values = surrogate_stats[:, column]
                row[f"observed_{metric_name}_dwell_s"] = observed_stats[column]
                row[f"null_{metric_name}_dwell_mean_s"] = float(np.mean(null_values))
                row[f"null_{metric_name}_dwell_ci_low_s"] = float(np.percentile(null_values, 2.5))
                row[f"null_{metric_name}_dwell_ci_high_s"] = float(np.percentile(null_values, 97.5))
                row[f"p_observed_{metric_name}_ge_null"] = float(
                    (1 + np.sum(null_values >= observed_stats[column])) / (n_surrogates + 1)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def state_window_metrics(
    labels: np.ndarray,
    nll_by_model: Dict[str, np.ndarray],
    times_s: np.ndarray,
    n_states: int,
    window_s: float,
    min_count: int,
    fold_ids: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate occupancy, semantic potential and acoustic-control NLL by state/window."""
    required = ("neural_only", "acoustic_only", "joint")
    if any(name not in nll_by_model for name in required):
        raise ValueError(f"nll_by_model must contain {required}.")
    nll_arrays = {name: np.asarray(nll_by_model[name], dtype=float) for name in required}
    valid = np.isfinite(times_s) & np.isfinite(labels) & (labels >= 0) & (labels < n_states)
    for arr in nll_arrays.values():
        valid &= np.isfinite(arr)
    if fold_ids is None:
        fold_ids = np.zeros(len(labels), dtype=int)
    else:
        fold_ids = np.asarray(fold_ids)
        if fold_ids.shape != np.asarray(labels).shape:
            raise ValueError("fold_ids must have the same shape as labels.")
        valid &= fold_ids >= 0
    labels_v = np.asarray(labels, dtype=int)[valid]
    time_v = np.asarray(times_s, dtype=float)[valid]
    fold_ids_v = np.asarray(fold_ids, dtype=int)[valid]
    valid_indices = np.flatnonzero(valid)
    nll_v = {name: arr[valid] for name, arr in nll_arrays.items()}
    if len(labels_v) == 0:
        raise ValueError("No valid out-of-fold NLL values; check audio/transcript/neural alignment.")

    window_origin = float(time_v.min())
    window_ids = np.floor((time_v - window_origin) / window_s).astype(int)
    state_rows, entropy_rows = [], []
    fold_window_pairs = sorted(set(zip(fold_ids_v.tolist(), window_ids.tolist())))
    for fold_id, window in fold_window_pairs:
        mask = (fold_ids_v == fold_id) & (window_ids == window)
        lab_w = labels_v[mask]
        sample_indices = valid_indices[mask]
        counts = np.bincount(lab_w, minlength=n_states)
        probs = counts / max(len(lab_w), 1)
        nonzero = probs > 0
        entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
        entropy_rows.append({
            "fold": int(fold_id),
            "window": int(window),
            "start_s": float(time_v[mask].min()),
            "end_s": float(time_v[mask].max()),
            "n_samples": int(len(lab_w)),
            "entropy_nats": float(entropy),
            "entropy_norm": float(entropy / np.log(n_states)) if n_states > 1 else np.nan,
            "n_occupied_states": int(nonzero.sum()),
        })
        pair_starts = sample_indices[sample_indices + 1 < len(labels)]
        pair_stops = pair_starts + 1
        pair_valid = (
            valid[pair_stops]
            & (np.asarray(fold_ids)[pair_stops] == fold_id)
            & (np.floor((np.asarray(times_s)[pair_stops] - window_origin) / window_s) == window)
        )
        from_states = np.asarray(labels, dtype=int)[pair_starts[pair_valid]]
        to_states = np.asarray(labels, dtype=int)[pair_stops[pair_valid]]
        p_kk = np.full(n_states, np.nan, dtype=float)
        for state in range(n_states):
            from_state = from_states == state
            if from_state.any():
                p_kk[state] = float(np.mean(to_states[from_state] == state))
        for state in range(n_states):
            selected = lab_w == state
            count = int(selected.sum())
            if count < min_count:
                continue
            q = float(probs[state])
            mean_neural = float(np.mean(nll_v["neural_only"][mask][selected]))
            mean_audio = float(np.mean(nll_v["acoustic_only"][mask][selected]))
            mean_joint = float(np.mean(nll_v["joint"][mask][selected]))
            state_rows.append({
                "fold": int(fold_id),
                "window": int(window),
                "state": int(state),
                "n_samples": count,
                "q": q,
                "self_information": float(-np.log(q)),
                "p_kk": float(p_kk[state]),
                # Primary V(z): held-out semantic NLL decoded from neural state.
                "semantic_potential_nll": mean_neural,
                "acoustic_baseline_nll": mean_audio,
                # Sensitivity/control model: joint audio + neural prediction.
                "acoustic_controlled_joint_nll": mean_joint,
                # Positive means adding neural state improves prediction over audio alone.
                "neural_incremental_information_nats": mean_audio - mean_joint,
            })
    return pd.DataFrame(state_rows), pd.DataFrame(entropy_rows)


def soft_state_window_metrics(
    responsibilities: np.ndarray,
    nll_by_model: Dict[str, np.ndarray],
    times_s: np.ndarray,
    window_s: float,
    min_effective_count: float,
    fold_ids: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate state occupancy, persistence and NLL using GMM responsibilities."""
    responsibilities = np.asarray(responsibilities, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    if responsibilities.ndim != 2 or responsibilities.shape[0] != len(times_s):
        raise ValueError("responsibilities must have shape (samples, states) matching times_s.")
    n_states = responsibilities.shape[1]
    required = ("neural_only", "acoustic_only", "joint")
    if any(name not in nll_by_model for name in required):
        raise ValueError(f"nll_by_model must contain {required}.")
    nll_arrays = {name: np.asarray(nll_by_model[name], dtype=float) for name in required}
    if any(array.shape != times_s.shape for array in nll_arrays.values()):
        raise ValueError("NLL arrays must have the same shape as times_s.")
    if fold_ids is None:
        fold_ids = np.zeros(len(times_s), dtype=int)
    else:
        fold_ids = np.asarray(fold_ids)
        if fold_ids.shape != times_s.shape:
            raise ValueError("fold_ids must have the same shape as times_s.")

    valid = (
        np.isfinite(times_s)
        & np.isfinite(responsibilities).all(axis=1)
        & (fold_ids >= 0)
    )
    for array in nll_arrays.values():
        valid &= np.isfinite(array)
    valid &= (responsibilities >= 0).all(axis=1) & (responsibilities.sum(axis=1) > 0)
    if not valid.any():
        raise ValueError("No valid posterior/NLL rows are available for soft-state metrics.")

    normalized = np.zeros_like(responsibilities)
    normalized[valid] = responsibilities[valid] / responsibilities[valid].sum(axis=1, keepdims=True)
    valid_indices = np.flatnonzero(valid)
    time_origin = float(times_s[valid].min())
    window_ids = np.floor((times_s - time_origin) / window_s).astype(int)
    rows, entropy_rows = [], []
    fold_window_pairs = sorted(set(zip(fold_ids[valid].tolist(), window_ids[valid].tolist())))
    for fold_id, window in fold_window_pairs:
        indices = valid_indices[(fold_ids[valid_indices] == fold_id) & (window_ids[valid_indices] == window)]
        posterior = normalized[indices]
        occupancy = posterior.mean(axis=0)
        nonzero = occupancy > 0
        entropy = -float(np.sum(occupancy[nonzero] * np.log(occupancy[nonzero])))
        entropy_rows.append({
            "fold": int(fold_id), "window": int(window),
            "start_s": float(times_s[indices].min()), "end_s": float(times_s[indices].max()),
            "n_samples": int(len(indices)), "entropy_nats": entropy,
            "entropy_norm": entropy / np.log(n_states) if n_states > 1 else np.nan,
            "n_occupied_states": int(nonzero.sum()),
        })

        pair_starts = indices[indices + 1 < len(times_s)]
        pair_stops = pair_starts + 1
        pair_mask = valid[pair_stops] & (fold_ids[pair_stops] == fold_id) & (window_ids[pair_stops] == window)
        pair_starts, pair_stops = pair_starts[pair_mask], pair_stops[pair_mask]
        p_kk = np.full(n_states, np.nan, dtype=float)
        if len(pair_starts):
            weights = normalized[pair_starts]
            next_weights = normalized[pair_stops]
            denominator = weights.sum(axis=0)
            p_kk = np.divide(
                np.sum(weights * next_weights, axis=0), denominator,
                out=np.full(n_states, np.nan), where=denominator > 0,
            )

        for state in range(n_states):
            state_weights = posterior[:, state]
            effective_count = float(state_weights.sum())
            if effective_count < min_effective_count:
                continue
            mean_neural = float(np.average(nll_arrays["neural_only"][indices], weights=state_weights))
            mean_audio = float(np.average(nll_arrays["acoustic_only"][indices], weights=state_weights))
            mean_joint = float(np.average(nll_arrays["joint"][indices], weights=state_weights))
            q = float(occupancy[state])
            rows.append({
                "fold": int(fold_id), "window": int(window), "state": int(state),
                "n_samples": effective_count, "n_observations": int(len(indices)),
                "q": q, "self_information": float(-np.log(q)), "p_kk": float(p_kk[state]),
                "semantic_potential_nll": mean_neural,
                "acoustic_baseline_nll": mean_audio,
                "acoustic_controlled_joint_nll": mean_joint,
                "neural_incremental_information_nats": mean_audio - mean_joint,
            })
    return pd.DataFrame(rows), pd.DataFrame(entropy_rows)


def state_dynamics(
    labels: np.ndarray,
    sfreq: float,
    n_states: int,
    fold_ids: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels)
    valid = np.isfinite(labels) & (labels >= 0) & (labels < n_states)
    if fold_ids is None:
        fold_ids = np.zeros(len(labels), dtype=int)
    else:
        fold_ids = np.asarray(fold_ids)
        if fold_ids.shape != labels.shape:
            raise ValueError("fold_ids must have the same shape as labels.")
        valid &= fold_ids >= 0
    labels_int = np.zeros(len(labels), dtype=int)
    labels_int[valid] = labels[valid].astype(int)
    transition_frames = []
    for fold_id in np.unique(fold_ids[valid]):
        transition = np.zeros((n_states, n_states), dtype=np.int64)
        fold_valid = valid & (fold_ids == fold_id)
        if len(labels) > 1:
            adjacent_valid = fold_valid[:-1] & fold_valid[1:]
            np.add.at(
                transition,
                (labels_int[:-1][adjacent_valid], labels_int[1:][adjacent_valid]),
                1,
            )
        row_sums = transition.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            transition, row_sums, out=np.zeros_like(transition, dtype=float), where=row_sums > 0
        )
        transition_frame = pd.DataFrame(
            probabilities, columns=[f"to_{i}" for i in range(n_states)]
        )
        transition_frame.insert(0, "from_state", np.arange(n_states))
        transition_frame.insert(0, "fold", int(fold_id))
        transition_frames.append(transition_frame)
    transition_columns = ["fold", "from_state", *(f"to_{i}" for i in range(n_states))]
    transition_df = (
        pd.concat(transition_frames, ignore_index=True)
        if transition_frames else pd.DataFrame(columns=transition_columns)
    )

    run_breaks = (valid[1:] != valid[:-1]) | (fold_ids[1:] != fold_ids[:-1]) | (
        valid[1:] & valid[:-1] & (labels_int[1:] != labels_int[:-1])
    )
    boundaries = np.r_[0, np.flatnonzero(run_breaks) + 1, len(labels)]
    runs = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if valid[start]:
            runs.append({"fold": int(fold_ids[start]), "state": int(labels_int[start]), "start_index": int(start),
                         "end_index": int(stop), "dwell_s": float((stop - start) / sfreq)})
    runs_df = pd.DataFrame(runs)
    if len(runs_df):
        dwell = runs_df.groupby(["fold", "state"])["dwell_s"].agg(
            n_runs="count", mean_dwell_s="mean", median_dwell_s="median"
        ).reset_index()
    else:
        dwell = pd.DataFrame(columns=["fold", "state", "n_runs", "mean_dwell_s", "median_dwell_s"])
    return transition_df, dwell
