"""Neural data loading, preprocessing, and latent-state modelling for the SLEP pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import warnings

import mne
from mne.cov import compute_whitener
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("slep_thermo.neural")


def read_evoked(fif_path: Path, evoked_index: int = 0):
    """Read one Evoked object and return it with its metadata.

    `condition` is an index into the Evoked objects stored in the FIF file.
    `baseline=None` prevents MNE from applying an additional baseline correction.
    SSP projectors are applied by MNE (`proj=True`).
    """
    fif_path = Path(fif_path)
    if not fif_path.is_file():
        raise FileNotFoundError(f"Evoked FIF not found: {fif_path}")
    evoked = mne.read_evokeds(
        str(fif_path), condition=evoked_index, baseline=None, proj=True, verbose="ERROR"
    )
    if evoked.data.ndim != 2:
        raise ValueError(f"Expected Evoked data (channels, times), got {evoked.data.shape}")
    return evoked


def list_repetition_evoked_paths(data_dir: Path, subject: str) -> List[Path]:
    """Find and numerically sort single-repetition Evoked files for one subject."""
    pattern = re.compile(rf"{re.escape(subject)}_rep(\d+)\.fif")
    indexed_paths = []
    for path in Path(data_dir).glob(f"{subject}_rep*.fif"):
        match = pattern.fullmatch(path.name)
        if match:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort(key=lambda item: item[0])
    if len(indexed_paths) < 2:
        raise FileNotFoundError(
            f"Leave-one-repetition-out CV requires at least two files named "
            f"{subject}_repN.fif in {data_dir}. Found {len(indexed_paths)}."
        )
    return [path for _, path in indexed_paths]


def get_sensor_groups(evoked, include_bads: bool = False) -> Dict[str, np.ndarray]:
    """Use MNE channel metadata to separate EEG, MEG magnetometers and gradiometers."""
    exclude = [] if include_bads else "bads"
    info = evoked.info
    groups = {
        "EEG": np.asarray(mne.pick_types(info, meg=False, eeg=True, exclude=exclude), dtype=int),
        "MEG_mag": np.asarray(mne.pick_types(info, meg="mag", eeg=False, exclude=exclude), dtype=int),
        "MEG_grad": np.asarray(mne.pick_types(info, meg="grad", eeg=False, exclude=exclude), dtype=int),
    }
    groups = {name: picks for name, picks in groups.items() if len(picks) > 0}
    if not groups:
        raise ValueError("No EEG, MEG magnetometer or MEG gradiometer channels found in Evoked.info")
    return groups


def make_sensor_noise_whitener(noise_cov, info, picks: np.ndarray) -> np.ndarray:
    """Construct an MNE noise-covariance whitener for one sensor group."""
    channel_names = [info["ch_names"][int(pick)] for pick in picks]
    missing = sorted(set(channel_names) - set(noise_cov["names"]))
    if missing:
        raise ValueError(f"Noise covariance is missing selected channels: {missing[:8]}.")
    whitener, whitened_names = compute_whitener(
        noise_cov, info, picks=picks, rank=None, verbose="ERROR",
    )
    if list(whitened_names) != channel_names:
        raise ValueError("MNE returned noise-whitener channels in a different order than the sensor data.")
    if whitener.shape != (len(channel_names), len(channel_names)):
        raise ValueError(f"Unexpected noise whitener shape: {whitener.shape}.")
    return np.asarray(whitener, dtype=np.float64)


def apply_sensor_noise_whitener(data: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    """Apply a sensor-space whitener to channels x time or repeats x channels x time."""
    data = np.asarray(data, dtype=np.float32)
    whitener = np.asarray(whitener, dtype=np.float64)
    if data.ndim == 2:
        if whitener.shape != (data.shape[0], data.shape[0]):
            raise ValueError("Noise whitener dimensions must match sensor channels.")
        return (whitener @ data).astype(np.float32)
    if data.ndim == 3:
        if whitener.shape != (data.shape[1], data.shape[1]):
            raise ValueError("Noise whitener dimensions must match repetition sensor channels.")
        return np.einsum("ij,rjt->rit", whitener, data, optimize=True).astype(np.float32)
    raise ValueError("Sensor data must be 2-D or 3-D to apply a noise whitener.")


def preprocess_sensor_group(
    data: np.ndarray,
    raw_sfreq: float,
    target_sfreq: float,
    apply_bandpass: bool,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Optional band-pass and anti-aliased polyphase downsampling."""
    x = np.asarray(data, dtype=np.float32)
    if apply_bandpass:
        if not (0 < low_hz < high_hz < raw_sfreq / 2):
            raise ValueError("Bandpass must satisfy 0 < low < high < Nyquist.")
        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=raw_sfreq, output="sos")
        x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)

    ratio = raw_sfreq / target_sfreq
    down = int(round(ratio))
    if down < 1 or not np.isclose(ratio, down):
        raise ValueError(
            f"raw_sfreq/target_sfreq must be an integer; received {raw_sfreq}/{target_sfreq}."
        )
    if down > 1:
        x = resample_poly(x, up=1, down=down, axis=-1).astype(np.float32)
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_neural_state_transform(
    train_sensor_time: np.ndarray,
    test_sensor_time: np.ndarray,
    n_pca: int,
    n_states: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, Dict[str, object]]:
    """Fit channel scaling, PCA and GMM on training samples; transform held-out samples."""
    train_x = np.asarray(train_sensor_time, dtype=np.float32)
    test_x = np.asarray(test_sensor_time, dtype=np.float32)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and test sensor data must be 2-D with matching channels.")

    median = np.median(train_x, axis=0)
    mad = median_abs_deviation(train_x, axis=0, scale="normal", nan_policy="omit")
    std = np.std(train_x, axis=0)
    median_scale = float(np.median(std)) if np.median(std) > 0 else 1.0
    scale_floor = max(median_scale * 1e-4, 1e-25)
    scale = np.where(mad > scale_floor, mad, std)
    scale = np.where(scale > scale_floor, scale, 1.0)
    train_x = ((train_x - median) / scale).astype(np.float32)
    n_components = min(n_pca, train_x.shape[1], train_x.shape[0] - 1)
    if n_components < 2:
        raise ValueError("Insufficient samples/channels for PCA.")
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized",
              random_state=random_state)
    z_train = pca.fit_transform(train_x).astype(np.float32)
    z_scaler = StandardScaler().fit(z_train)
    z_train = z_scaler.transform(z_train).astype(np.float32)

    actual_states = min(n_states, max(2, len(z_train) // 100))
    gmm = GaussianMixture(
        n_components=actual_states, covariance_type="diag", reg_covar=1e-4,
        n_init=5, max_iter=300, random_state=random_state,
    )
    gmm.fit(z_train)
    model = {
        "median": median,
        "scale": scale,
        "pca": pca,
        "z_scaler": z_scaler,
        "gmm": gmm,
    }
    z_test, test_labels = transform_neural_state_model(model, test_sensor_time)
    return z_train, z_test, test_labels, n_components, gmm.n_components, model


def transform_neural_state_model(
    model: Dict[str, object],
    sensor_time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a training-fitted neural transform and GMM to new samples."""
    x = np.asarray(sensor_time, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(model["median"]):
        raise ValueError("Sensor data must be 2-D and match the fitted codebook channels.")
    x = ((x - model["median"]) / model["scale"]).astype(np.float32)
    z = model["pca"].transform(x).astype(np.float32)
    z = model["z_scaler"].transform(z).astype(np.float32)
    labels = model["gmm"].predict(z)
    return z, labels
