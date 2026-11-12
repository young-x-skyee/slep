"""Affine timing correction functions for the SLEP thermodynamic pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# Fixed stimulus-delivery corrections supplied for this Kymata recording setup.
DEFAULT_STIMULUS_SHIFT_CORRECTION = 0.0005404  # seconds per second
DEFAULT_STIMULUS_DELIVERY_LATENCY_S = 0.026  # seconds


def crop_evoked_to_stimulus(
    evoked,
    stimulus_onset_s: float,
    duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Crop using Evoked.times (seconds relative to the event), not array indices.

    The default onset of 0 s assumes the Evoked time origin is the podcast trigger.
    This must be checked from the preprocessing/epoching code.
    """
    sfreq = float(evoked.info["sfreq"])
    all_times = np.asarray(evoked.times, dtype=float)
    start_time = float(stimulus_onset_s)
    end_time = start_time + float(duration_s)
    start = int(np.searchsorted(all_times, start_time - 1e-9, side="left"))
    stop = int(np.searchsorted(all_times, end_time - 1e-9, side="left"))
    if start >= len(all_times) or stop <= start:
        raise ValueError(
            f"Requested time interval [{start_time:.3f}, {end_time:.3f}) s is not present. "
            f"Evoked.times spans [{all_times[0]:.3f}, {all_times[-1]:.3f}] s. "
            "Check --stimulus-onset and --duration."
        )
    data = np.asarray(evoked.data[:, start:stop], dtype=np.float32)
    times = all_times[start:stop]
    expected = int(round(duration_s * sfreq))
    if abs(data.shape[1] - expected) > 2:
        warnings.warn(
            f"Cropped {data.shape[1]} samples, expected about {expected} for {duration_s}s "
            f"at {sfreq} Hz. Continuing with actual Evoked time axis.", RuntimeWarning,
        )
    return data, times, sfreq


def audio_time_to_evoked_time(
    audio_time_s: np.ndarray | float,
    stimulus_onset_s: float,
    stimulus_shift_correction: float,
    stimulus_delivery_latency_s: float,
    neural_lag_s: float,
) -> np.ndarray:
    """Map nominal audio time to the Evoked time at which its neural response is sampled.

    This follows the convention in the supplied Kymata grid-search example:
        t_evoked = t0 + neural_lag + delivery_latency + (1 + shift) * t_audio

    Positive delivery latency and positive shift move the corresponding EMEG
    sample later in the Evoked time axis. `neural_lag_s` is kept separate from
    the hardware/presentation delay and drift.
    """
    scale = 1.0 + float(stimulus_shift_correction)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("1 + stimulus_shift_correction must be finite and positive.")
    return (
        float(stimulus_onset_s)
        + float(neural_lag_s)
        + float(stimulus_delivery_latency_s)
        + scale * np.asarray(audio_time_s, dtype=float)
    )


def evoked_time_to_audio_time(
    evoked_time_s: np.ndarray | float,
    stimulus_onset_s: float,
    stimulus_shift_correction: float,
    stimulus_delivery_latency_s: float,
    neural_lag_s: float,
) -> np.ndarray:
    """Inverse of :func:`audio_time_to_evoked_time`. Returns time on the WAV/MFA axis."""
    scale = 1.0 + float(stimulus_shift_correction)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("1 + stimulus_shift_correction must be finite and positive.")
    return (
        np.asarray(evoked_time_s, dtype=float)
        - float(stimulus_onset_s)
        - float(neural_lag_s)
        - float(stimulus_delivery_latency_s)
    ) / scale


def crop_evoked_to_mapped_stimulus(
    evoked,
    stimulus_onset_s: float,
    duration_s: float,
    stimulus_shift_correction: float,
    stimulus_delivery_latency_s: float,
    neural_lag_s: float,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Crop neural samples corresponding to the warped and delayed audio interval.

    The returned sample times remain in Evoked coordinates. The caller must
    convert them back to the audio/MFA time axis with
    :func:`evoked_time_to_audio_time` after downsampling.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    sfreq = float(evoked.info["sfreq"])
    all_times = np.asarray(evoked.times, dtype=float)
    mapped_start = float(audio_time_to_evoked_time(
        0.0, stimulus_onset_s, stimulus_shift_correction,
        stimulus_delivery_latency_s, neural_lag_s,
    ))
    mapped_stop = float(audio_time_to_evoked_time(
        duration_s, stimulus_onset_s, stimulus_shift_correction,
        stimulus_delivery_latency_s, neural_lag_s,
    ))
    # End is exclusive, matching a stimulus interval [0, duration).
    start = int(np.searchsorted(all_times, mapped_start - 1e-9, side="left"))
    stop = int(np.searchsorted(all_times, mapped_stop - 1e-9, side="left"))
    if start >= len(all_times) or stop <= start:
        raise ValueError(
            f"Mapped audio interval [{mapped_start:.6f}, {mapped_stop:.6f}) s is not present. "
            f"Evoked.times spans [{all_times[0]:.6f}, {all_times[-1]:.6f}] s. "
            "Check the stimulus onset, delivery latency, shift correction and neural lag."
        )
    if mapped_start < all_times[0] - 0.5 / sfreq or mapped_stop > all_times[-1] + 1.5 / sfreq:
        raise ValueError(
            f"The requested mapped neural interval [{mapped_start:.6f}, {mapped_stop:.6f}) s "
            f"extends beyond Evoked.times [{all_times[0]:.6f}, {all_times[-1]:.6f}] s. "
            "Reduce the requested duration or verify the timing parameters."
        )
    data = np.asarray(evoked.data[:, start:stop], dtype=np.float32)
    times = all_times[start:stop]
    if len(times) < 2:
        raise ValueError("Mapped stimulus crop contains fewer than two neural samples.")
    meta = {
        "mapped_evoked_start_s": float(mapped_start),
        "mapped_evoked_stop_exclusive_s": float(mapped_stop),
        "actual_cropped_evoked_start_s": float(times[0]),
        "actual_cropped_evoked_end_s": float(times[-1]),
        "stimulus_shift_correction_s_per_s": float(stimulus_shift_correction),
        "stimulus_delivery_latency_s": float(stimulus_delivery_latency_s),
        "neural_lag_s": float(neural_lag_s),
    }
    return data, times, sfreq, meta


def load_timing_corrections(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Load optional per-participant timing overrides from CSV.

    Required column: subject. Optional columns:
      stimulus_shift_correction, stimulus_delivery_latency_s, neural_lag_s

    Blank values are ignored and fall back to global CLI values.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Timing-corrections CSV not found: {path}")
    frame = pd.read_csv(path)
    if "subject" not in frame.columns:
        raise ValueError("Timing-corrections CSV must contain a 'subject' column.")
    if frame["subject"].duplicated().any():
        duplicates = frame.loc[frame["subject"].duplicated(), "subject"].astype(str).tolist()
        raise ValueError(f"Duplicate subjects in timing-corrections CSV: {duplicates}")
    allowed = {
        "stimulus_shift_correction",
        "stimulus_delivery_latency_s",
        "neural_lag_s",
    }
    unknown = set(frame.columns) - {"subject"} - allowed
    if unknown:
        raise ValueError(f"Unknown columns in timing-corrections CSV: {sorted(unknown)}")
    result: Dict[str, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        subject = str(row["subject"]).strip()
        if not subject:
            continue
        result[subject] = {
            key: float(row[key])
            for key in allowed
            if key in frame.columns and pd.notna(row[key])
        }
    return result
