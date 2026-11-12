"""Purged cross-validation splits and cross-validated semantic/acoustic decoding for the SLEP pipeline."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .neural import fit_neural_state_transform

LOG = logging.getLogger("slep_thermo.crossval")


def purged_block_splits(times_s: np.ndarray, n_splits: int, gap_s: float):
    """Contiguous held-out folds with a temporal buffer around each test block."""
    n = len(times_s)
    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    for fold in range(n_splits):
        test_start, test_stop = boundaries[fold], boundaries[fold + 1]
        t_start = times_s[test_start] - gap_s
        t_stop = times_s[test_stop - 1] + gap_s
        test_idx = np.arange(test_start, test_stop)
        train_mask = (times_s < t_start) | (times_s > t_stop)
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) > 100 and len(test_idx) > 0:
            yield train_idx, test_idx


def purged_window_splits(
    times_s: np.ndarray,
    n_splits: int,
    window_s: float,
    gap_s: float,
    window_origin_s: float,
):
    """Yield contiguous folds whose test blocks contain complete analysis windows."""
    times_s = np.asarray(times_s, dtype=float)
    if n_splits < 3 or window_s <= 0 or gap_s < 0:
        raise ValueError("Require at least three folds, a positive window, and a nonnegative gap.")
    window_ids = np.floor((times_s - float(window_origin_s)) / window_s).astype(int)
    unique_windows = np.unique(window_ids)
    if len(unique_windows) < 3:
        raise ValueError("At least three analysis windows are required for purged CV.")

    for window_group in np.array_split(unique_windows, min(n_splits, len(unique_windows))):
        test_idx = np.flatnonzero(np.isin(window_ids, window_group))
        if not len(test_idx):
            continue
        test_start = times_s[test_idx[0]]
        test_stop = times_s[test_idx[-1]]
        train_idx = np.flatnonzero(
            (times_s < test_start - gap_s) | (times_s > test_stop + gap_s)
        )
        if len(train_idx) > 100:
            yield train_idx, test_idx


def _decoder_nll(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    """Fit a standardised multi-output ridge decoder and return test Gaussian NLL."""
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)

    decoder = Ridge(alpha=ridge_alpha)
    decoder.fit(x_train_s, y_train_s)
    train_pred = decoder.predict(x_train_s)
    test_pred = decoder.predict(x_test_s)
    residual = y_train_s - train_pred
    # Per-target residual variance with a small floor prevents singular
    # Gaussian likelihoods. NLL is reported per standardised target dimension.
    sigma2 = np.maximum(np.mean(residual ** 2, axis=0), 1e-4)
    nll = 0.5 * (
        np.log(2.0 * np.pi * sigma2)[None, :]
        + ((y_test_s - test_pred) ** 2) / sigma2[None, :]
    )
    return np.mean(nll, axis=1).astype(np.float32)


def crossvalidated_semantic_nll(
    z: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    n_splits: int,
    gap_s: float,
    ridge_alpha: float,
    acoustic_features: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute blocked-CV semantic NLL for neural, acoustic and joint predictors.

    `neural_only` is the primary semantic-potential proxy V(z), preserving the
    paper-inspired definition p(x_semantic | z). `acoustic_only` is a low-level
    audio control baseline. `joint` asks whether neural activity predicts
    semantic targets over and above the measured acoustic features.
    """
    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    times_s = np.asarray(times_s, dtype=float)
    if acoustic_features is None:
        raise ValueError("Acoustic control features are required; provide the stimulus WAV.")
    acoustic_features = np.asarray(acoustic_features, dtype=np.float32)
    if z.ndim != 2 or y.ndim != 2 or acoustic_features.ndim != 2:
        raise ValueError("z, y and acoustic_features must all be 2-D arrays.")
    if not (len(z) == len(y) == len(times_s) == len(acoustic_features)):
        raise ValueError("Neural, semantic, acoustic and time arrays must have equal sample counts.")

    valid = (
        np.isfinite(y).all(axis=1)
        & np.isfinite(z).all(axis=1)
        & np.isfinite(acoustic_features).all(axis=1)
        & np.isfinite(times_s)
    )
    z_valid, y_valid = z[valid], y[valid]
    a_valid, t_valid = acoustic_features[valid], times_s[valid]
    model_names = ("neural_only", "acoustic_only", "joint")
    results_valid = {
        name: np.full(len(z_valid), np.nan, dtype=np.float32) for name in model_names
    }
    splits = list(purged_block_splits(t_valid, n_splits=n_splits, gap_s=gap_s))
    if len(splits) < 3:
        raise ValueError("Fewer than three valid purged CV folds. Reduce --cv-gap-s or --cv-splits.")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        predictors = {
            "neural_only": z_valid,
            "acoustic_only": a_valid,
            "joint": np.column_stack([z_valid, a_valid]),
        }
        for name, x in predictors.items():
            results_valid[name][test_idx] = _decoder_nll(
                x[train_idx], x[test_idx], y_valid[train_idx], y_valid[test_idx], ridge_alpha,
            )
        LOG.debug("Completed semantic/acoustic decoding fold %d/%d", fold, len(splits))

    results = {}
    for name in model_names:
        full = np.full(len(z), np.nan, dtype=np.float32)
        full[valid] = results_valid[name]
        results[name] = full
    return results


def crossvalidated_sensor_states_and_nll(
    sensor_time: np.ndarray,
    y: np.ndarray,
    acoustic_features: np.ndarray,
    times_s: np.ndarray,
    n_pca: int,
    n_states: int,
    n_splits: int,
    gap_s: float,
    window_s: float,
    ridge_alpha: float,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, List[Dict[str, object]], int, int]:
    """Cross-fit neural states and semantic decoders on purged window blocks."""
    sensor_time = np.asarray(sensor_time, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    acoustic_features = np.asarray(acoustic_features, dtype=np.float32)
    times_s = np.asarray(times_s, dtype=float)
    if sensor_time.ndim != 2 or y.ndim != 2 or acoustic_features.ndim != 2:
        raise ValueError("Sensor data, semantic targets, and acoustic features must be 2-D.")
    n_samples = sensor_time.shape[1]
    if not (len(y) == len(acoustic_features) == len(times_s) == n_samples):
        raise ValueError("Sensor, target, acoustic, and time arrays must have matching samples.")

    valid_sensor = np.isfinite(sensor_time).all(axis=0) & np.isfinite(times_s)
    valid_indices = np.flatnonzero(valid_sensor)
    if not len(valid_indices):
        raise ValueError("No finite neural samples are available for cross-validation.")
    t_valid = times_s[valid_indices]
    sensor_valid = sensor_time[:, valid_indices].T
    y_valid = y[valid_indices]
    acoustic_valid = acoustic_features[valid_indices]
    decoder_valid = np.isfinite(y_valid).all(axis=1) & np.isfinite(acoustic_valid).all(axis=1)
    if not decoder_valid.any():
        raise ValueError("No aligned finite semantic and acoustic samples are available.")
    window_origin_s = float(t_valid[decoder_valid].min())

    splits = list(purged_window_splits(
        t_valid, n_splits, window_s, gap_s, window_origin_s,
    ))
    if len(splits) < 3:
        raise ValueError("Fewer than three purged window folds. Reduce --cv-gap-s or --cv-splits.")

    labels_valid = np.full(len(valid_indices), np.nan, dtype=float)
    fold_ids_valid = np.full(len(valid_indices), -1, dtype=int)
    model_names = ("neural_only", "acoustic_only", "joint")
    nll_valid = {name: np.full(len(valid_indices), np.nan, dtype=np.float32) for name in model_names}
    responsibilities_valid = None
    fold_models: List[Dict[str, object]] = []
    actual_n_pca = None
    actual_n_states = None

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        z_train, z_test, test_labels, fold_n_pca, fold_n_states, model = fit_neural_state_transform(
            sensor_valid[train_idx], sensor_valid[test_idx], n_pca, n_states,
            random_state + fold - 1,
        )
        fold_models.append({
            "fold": fold - 1,
            "sample_indices": valid_indices[test_idx].copy(),
            "window_origin_s": window_origin_s,
            "model": model,
        })
        if actual_n_pca is not None and fold_n_pca != actual_n_pca:
            raise ValueError("The effective PCA dimension changed between CV folds.")
        if actual_n_states is not None and fold_n_states != actual_n_states:
            raise ValueError("The effective GMM state count changed between CV folds.")
        actual_n_pca, actual_n_states = fold_n_pca, fold_n_states
        labels_valid[test_idx] = test_labels
        fold_ids_valid[test_idx] = fold - 1
        test_responsibilities = model["gmm"].predict_proba(z_test)
        if responsibilities_valid is None:
            responsibilities_valid = np.full(
                (len(valid_indices), fold_n_states), np.nan, dtype=np.float32,
            )
        responsibilities_valid[test_idx] = test_responsibilities

        train_decode = decoder_valid[train_idx]
        test_decode = decoder_valid[test_idx]
        train_decode_idx = train_idx[train_decode]
        test_decode_idx = test_idx[test_decode]
        if not len(train_decode_idx) or not len(test_decode_idx):
            continue

        predictors_train = {
            "neural_only": z_train[train_decode],
            "acoustic_only": acoustic_valid[train_decode_idx],
            "joint": np.column_stack([z_train[train_decode], acoustic_valid[train_decode_idx]]),
        }
        predictors_test = {
            "neural_only": z_test[test_decode],
            "acoustic_only": acoustic_valid[test_decode_idx],
            "joint": np.column_stack([z_test[test_decode], acoustic_valid[test_decode_idx]]),
        }
        for name in model_names:
            nll_valid[name][test_decode_idx] = _decoder_nll(
                predictors_train[name], predictors_test[name],
                y_valid[train_decode_idx], y_valid[test_decode_idx], ridge_alpha,
            )
        LOG.debug("Completed cross-fitted neural-state fold %d/%d", fold, len(splits))

    labels = np.full(n_samples, np.nan, dtype=float)
    labels[valid_indices] = labels_valid
    fold_ids = np.full(n_samples, -1, dtype=int)
    fold_ids[valid_indices] = fold_ids_valid
    responsibilities = np.full((n_samples, int(actual_n_states)), np.nan, dtype=np.float32)
    responsibilities[valid_indices] = responsibilities_valid
    nll_by_model = {}
    for name in model_names:
        full = np.full(n_samples, np.nan, dtype=np.float32)
        full[valid_indices] = nll_valid[name]
        nll_by_model[name] = full
    if not np.isfinite(nll_by_model["neural_only"]).any():
        raise ValueError("Cross-validation produced no held-out semantic predictions.")
    return labels, nll_by_model, fold_ids, fold_models, responsibilities, int(actual_n_pca), int(actual_n_states)


def crossvalidated_repetition_states_and_nll(
    sensor_repetitions: np.ndarray,
    y: np.ndarray,
    acoustic_features: np.ndarray,
    times_s: np.ndarray,
    n_pca: int,
    n_states: int,
    window_s: float,
    ridge_alpha: float,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, List[Dict[str, object]], int, int]:
    """Leave one complete presentation out; fit representation on other repeats."""
    sensor_repetitions = np.asarray(sensor_repetitions, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    acoustic_features = np.asarray(acoustic_features, dtype=np.float32)
    times_s = np.asarray(times_s, dtype=float)
    if sensor_repetitions.ndim != 3:
        raise ValueError("sensor_repetitions must have shape (repetitions, channels, samples).")
    n_repetitions, _, n_samples = sensor_repetitions.shape
    if n_repetitions < 2:
        raise ValueError("Leave-one-repetition-out CV requires at least two repetitions.")
    if not (len(y) == len(acoustic_features) == len(times_s) == n_samples):
        raise ValueError("Targets, acoustics, times and each repetition must share a sample grid.")

    sensor_samples = sensor_repetitions.transpose(0, 2, 1).reshape(
        n_repetitions * n_samples, sensor_repetitions.shape[1]
    )
    y_repeated = np.tile(y, (n_repetitions, 1))
    acoustic_repeated = np.tile(acoustic_features, (n_repetitions, 1))
    valid_sensor = np.isfinite(sensor_repetitions).all(axis=1)
    decoder_valid = np.isfinite(y).all(axis=1) & np.isfinite(acoustic_features).all(axis=1)
    window_origin_s = float(times_s[decoder_valid].min()) if decoder_valid.any() else float(times_s.min())
    decoder_valid_flat = np.tile(decoder_valid, n_repetitions)
    labels = np.full(n_repetitions * n_samples, np.nan, dtype=float)
    fold_ids = np.full(n_repetitions * n_samples, -1, dtype=int)
    model_names = ("neural_only", "acoustic_only", "joint")
    nll_by_model = {
        name: np.full(n_repetitions * n_samples, np.nan, dtype=np.float32)
        for name in model_names
    }
    fold_models: List[Dict[str, object]] = []
    responsibilities = None
    actual_n_pca = None
    actual_n_states = None

    for held_out in range(n_repetitions):
        train_repetitions = [rep for rep in range(n_repetitions) if rep != held_out]
        train_indices = np.concatenate([
            rep * n_samples + np.flatnonzero(valid_sensor[rep]) for rep in train_repetitions
        ])
        test_indices = held_out * n_samples + np.flatnonzero(valid_sensor[held_out])
        z_train, z_test, test_labels, fold_n_pca, fold_n_states, model = fit_neural_state_transform(
            sensor_samples[train_indices], sensor_samples[test_indices], n_pca, n_states,
            random_state + held_out,
        )
        if actual_n_pca is not None and fold_n_pca != actual_n_pca:
            raise ValueError("The effective PCA dimension changed between repetition folds.")
        if actual_n_states is not None and fold_n_states != actual_n_states:
            raise ValueError("The effective GMM state count changed between repetition folds.")
        actual_n_pca, actual_n_states = fold_n_pca, fold_n_states
        labels[test_indices] = test_labels
        fold_ids[test_indices] = held_out
        test_responsibilities = model["gmm"].predict_proba(z_test)
        if responsibilities is None:
            responsibilities = np.full(
                (n_repetitions * n_samples, fold_n_states), np.nan, dtype=np.float32,
            )
        responsibilities[test_indices] = test_responsibilities
        fold_models.append({
            "fold": held_out,
            "sample_indices": test_indices.copy(),
            "window_origin_s": window_origin_s,
            "repetition": held_out,
            "model": model,
        })

        train_decode = decoder_valid_flat[train_indices]
        test_decode = decoder_valid_flat[test_indices]
        train_decode_indices = train_indices[train_decode]
        test_decode_indices = test_indices[test_decode]
        if not len(train_decode_indices) or not len(test_decode_indices):
            continue
        predictors_train = {
            "neural_only": z_train[train_decode],
            "acoustic_only": acoustic_repeated[train_decode_indices],
            "joint": np.column_stack([z_train[train_decode], acoustic_repeated[train_decode_indices]]),
        }
        predictors_test = {
            "neural_only": z_test[test_decode],
            "acoustic_only": acoustic_repeated[test_decode_indices],
            "joint": np.column_stack([z_test[test_decode], acoustic_repeated[test_decode_indices]]),
        }
        for name in model_names:
            nll_by_model[name][test_decode_indices] = _decoder_nll(
                predictors_train[name], predictors_test[name],
                y_repeated[train_decode_indices], y_repeated[test_decode_indices], ridge_alpha,
            )

    if not np.isfinite(nll_by_model["neural_only"]).any():
        raise ValueError("Leave-one-repetition-out CV produced no held-out semantic predictions.")
    return labels, nll_by_model, fold_ids, fold_models, responsibilities, int(actual_n_pca), int(actual_n_states)
