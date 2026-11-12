"""Audio loading and low-level acoustic feature extraction.

Features are intentionally acoustic (log-mel energy, RMS, zero-crossing rate,
spectral centroid/bandwidth/rolloff and spectral flux). They are used as
control predictors, not as the definition of semantic potential.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Dict, Tuple
import json
import logging

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from numpy.lib.stride_tricks import sliding_window_view

LOG = logging.getLogger("slep_thermo.audio")


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int, Dict[str, object]]:
    """Read a PCM/float WAV file and return a normalised mono float32 signal."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Podcast audio not found: {path}. Place the experimental stimulus at "
            "<project>/stimuli/audio/stimulus.wav or pass --audio explicitly."
        )
    sfreq, data = wavfile.read(path, mmap=False)
    original_dtype = str(data.dtype)
    original_shape = tuple(data.shape)

    if data.ndim == 2:
        # Conventional arithmetic downmix. The original stereo file is retained;
        # this mono signal is only used for low-level acoustic control features.
        data = data.astype(np.float64).mean(axis=1)
    elif data.ndim != 1:
        raise ValueError(f"Expected mono or stereo WAV, got shape {data.shape}.")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if np.issubdtype(data.dtype, np.unsignedinteger):
            midpoint = (info.max + 1) / 2.0
            signal = (data.astype(np.float64) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            signal = data.astype(np.float64) / scale
    else:
        signal = data.astype(np.float64)

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal -= np.mean(signal)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1.0:
        signal /= peak
    if not np.isfinite(signal).all() or signal.size < int(sfreq * 0.1):
        raise ValueError("WAV contains no usable finite audio or is shorter than 100 ms.")
    if sfreq <= 0:
        raise ValueError(f"Invalid WAV sampling rate: {sfreq}")

    metadata = {
        "path": str(path.resolve()),
        "original_sample_rate_hz": int(sfreq),
        "original_shape": list(original_shape),
        "original_dtype": original_dtype,
        "duration_s": float(len(signal) / sfreq),
        "mono_downmix": bool(len(original_shape) == 2),
    }
    return signal.astype(np.float32), int(sfreq), metadata


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                    fmin: float = 40.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filters without a librosa dependency."""
    if fmax is None:
        fmax = sample_rate / 2.0
    if not 0 <= fmin < fmax <= sample_rate / 2.0:
        raise ValueError("Mel filter frequency limits are invalid.")
    mel_points = np.linspace(_hz_to_mel(np.array([fmin]))[0],
                             _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    filters = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = hz_points[m:m + 3]
        rising = (fft_freqs - left) / max(centre - left, 1e-12)
        falling = (right - fft_freqs) / max(right - centre, 1e-12)
        filters[m] = np.maximum(0.0, np.minimum(rising, falling))
    # Slaney-style area normalisation keeps filters with different bandwidths
    # on a more comparable scale.
    enorm = 2.0 / np.maximum(hz_points[2:n_mels + 2] - hz_points[:n_mels], 1e-12)
    filters *= enorm[:, None]
    return filters.astype(np.float32)


def extract_acoustic_features(
    wav_path: Path,
    target_sfreq: float = 50.0,
    n_mels: int = 40,
    frame_ms: float = 25.0,
    analysis_sample_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Extract centred-frame acoustic features at the requested frame rate.

    Returns
    -------
    features : ndarray, shape (n_frames, n_features)
        40 log-mel bands by default plus log-RMS, zero-crossing rate, spectral
        centroid, bandwidth, 85%-rolloff and spectral flux.
    frame_times_s : ndarray, shape (n_frames,)
        Frame-centre times relative to the start of the WAV.
    metadata : dict
        Audio and feature extraction metadata.
    """
    if target_sfreq <= 0 or n_mels < 4 or frame_ms <= 0:
        raise ValueError("target_sfreq, n_mels and frame_ms must be positive; n_mels >= 4.")
    mono, original_sfreq, metadata = read_wav_mono(Path(wav_path))

    common = gcd(original_sfreq, int(analysis_sample_rate))
    up, down = int(analysis_sample_rate // common), int(original_sfreq // common)
    if up != down:
        mono = resample_poly(mono, up=up, down=down).astype(np.float32)

    fs = int(analysis_sample_rate)
    ratio = fs / float(target_sfreq)
    hop = int(round(ratio))
    if not np.isclose(ratio, hop, atol=1e-8):
        raise ValueError(
            f"analysis_sample_rate/target_sfreq must be integer; got {fs}/{target_sfreq}."
        )
    frame_len = max(16, int(round(frame_ms * fs / 1000.0)))
    n_fft = max(512, 2 ** int(np.ceil(np.log2(frame_len))))
    n_frames = int(np.ceil(len(mono) / hop))
    if n_frames < 2:
        raise ValueError("Audio is too short to construct acoustic features.")

    # Frame centres sit on 0, 1/target_sfreq, ...; padding makes the first and
    # final centred frames available without shifting the transcript timeline.
    half_left = frame_len // 2
    half_right = frame_len - half_left
    padded = np.pad(mono, (half_left, half_right), mode="reflect")
    all_windows = sliding_window_view(padded, frame_len)
    starts = np.arange(n_frames, dtype=np.int64) * hop
    frames = all_windows[starts]

    window = np.hanning(frame_len).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32) / max(float(np.sum(window ** 2)), 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mel_filters = _mel_filterbank(fs, n_fft, n_mels)
    mel_power = power @ mel_filters.T
    log_mel = np.log(np.maximum(mel_power, 1e-10)).astype(np.float32)

    eps = 1e-10
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + eps)
    log_rms = np.log(frame_rms + eps)
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    power_sum = np.sum(power, axis=1) + eps
    centroid = np.sum(power * freqs[None, :], axis=1) / power_sum
    bandwidth = np.sqrt(
        np.sum(power * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / power_sum
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= (0.85 * cumulative[:, -1:]), axis=1)
    rolloff = freqs[rolloff_idx]
    norm_mel = mel_power / (np.linalg.norm(mel_power, axis=1, keepdims=True) + eps)
    spectral_flux = np.zeros(n_frames, dtype=np.float32)
    if n_frames > 1:
        spectral_flux[1:] = np.mean(np.maximum(norm_mel[1:] - norm_mel[:-1], 0.0), axis=1)

    scalar_features = np.column_stack([
        log_rms,
        zcr,
        centroid / (fs / 2.0),
        bandwidth / (fs / 2.0),
        rolloff / (fs / 2.0),
        spectral_flux,
    ]).astype(np.float32)
    features = np.column_stack([log_mel, scalar_features]).astype(np.float32)
    frame_times = np.arange(n_frames, dtype=np.float64) / float(target_sfreq)

    feature_names = [f"logmel_{i:02d}" for i in range(n_mels)] + [
        "log_rms", "zero_crossing_rate", "spectral_centroid_norm",
        "spectral_bandwidth_norm", "spectral_rolloff85_norm", "spectral_flux",
    ]
    metadata.update({
        "analysis_sample_rate_hz": fs,
        "target_feature_rate_hz": float(target_sfreq),
        "frame_ms": float(frame_ms),
        "hop_samples": int(hop),
        "n_mels": int(n_mels),
        "n_frames": int(n_frames),
        "n_features": int(features.shape[1]),
        "feature_names": feature_names,
        "feature_type": "low-level acoustic control features; not semantic energy",
    })
    return features, frame_times, metadata


def extract_acoustic_features_cached(
    wav_path: Path,
    cache_path: Path,
    target_sfreq: float = 50.0,
    n_mels: int = 40,
    frame_ms: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Load or create a cache keyed by WAV identity and feature parameters."""
    wav_path, cache_path = Path(wav_path), Path(cache_path)
    if not wav_path.is_file():
        # Let read_wav_mono provide the detailed, actionable error.
        read_wav_mono(wav_path)
    stat = wav_path.stat()
    key = {
        "wav_path": str(wav_path.resolve()),
        "wav_size": int(stat.st_size),
        "wav_mtime_ns": int(stat.st_mtime_ns),
        "target_sfreq": float(target_sfreq),
        "n_mels": int(n_mels),
        "frame_ms": float(frame_ms),
    }
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.is_file() and metadata_path.is_file():
        try:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            if stored.get("cache_key") == key:
                with np.load(cache_path, allow_pickle=False) as data:
                    features = data["features"].astype(np.float32)
                    times = data["times"].astype(np.float64)
                LOG.info("Loaded cached audio features: %s", cache_path)
                return features, times, stored["feature_metadata"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            LOG.warning("Audio feature cache is unreadable; recomputing: %s", cache_path)

    features, times, metadata = extract_acoustic_features(
        wav_path, target_sfreq=target_sfreq, n_mels=n_mels, frame_ms=frame_ms,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, times=times)
    metadata_path.write_text(json.dumps({
        "cache_key": key,
        "feature_metadata": metadata,
    }, indent=2), encoding="utf-8")
    LOG.info("Wrote acoustic feature cache: %s", cache_path)
    return features, times, metadata


def align_audio_features_to_times(
    features: np.ndarray,
    feature_times_s: np.ndarray,
    target_times_s: np.ndarray,
    audio_duration_s: float,
) -> np.ndarray:
    """Interpolate audio features to neural times; mark samples outside WAV as NaN."""
    features = np.asarray(features, dtype=np.float32)
    feature_times_s = np.asarray(feature_times_s, dtype=float)
    target_times_s = np.asarray(target_times_s, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(feature_times_s):
        raise ValueError("Audio features and feature times have inconsistent shapes.")
    if len(feature_times_s) < 2 or np.any(np.diff(feature_times_s) <= 0):
        raise ValueError("Audio feature times must be strictly increasing and contain >=2 frames.")
    if np.any(~np.isfinite(target_times_s)):
        raise ValueError("Target audio alignment times must be finite.")

    aligned = np.empty((len(target_times_s), features.shape[1]), dtype=np.float32)
    for dim in range(features.shape[1]):
        aligned[:, dim] = np.interp(target_times_s, feature_times_s, features[:, dim])
    valid = (target_times_s >= 0.0) & (target_times_s <= float(audio_duration_s) + 1e-9)
    aligned[~valid] = np.nan
    return aligned
