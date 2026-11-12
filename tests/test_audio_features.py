from pathlib import Path

import numpy as np
from scipy.io import wavfile

from slep_thermo.audio import (
    align_audio_features_to_times,
    extract_acoustic_features,
    extract_acoustic_features_cached,
)


def _write_test_wav(path: Path, duration_s: float = 2.0, sfreq: int = 16000):
    t = np.arange(int(duration_s * sfreq)) / sfreq
    x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t)
    wavfile.write(path, sfreq, np.asarray(np.clip(x, -1, 1) * 32767, dtype=np.int16))


def test_extract_acoustic_features_has_expected_shape_and_is_finite(tmp_path):
    wav = tmp_path / "stimulus.wav"
    _write_test_wav(wav, duration_s=2.0)
    features, times, metadata = extract_acoustic_features(wav, target_sfreq=50, n_mels=20)
    assert features.shape == (100, 26)
    assert times.shape == (100,)
    assert np.isfinite(features).all()
    assert metadata["duration_s"] == 2.0
    assert metadata["n_features"] == 26


def test_align_audio_features_marks_out_of_range_as_nan():
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    feature_times = np.arange(10) / 10
    target_times = np.array([0.0, 0.35, 0.9, 1.0])
    aligned = align_audio_features_to_times(features, feature_times, target_times, audio_duration_s=0.95)
    assert np.isfinite(aligned[:3]).all()
    assert np.isnan(aligned[3]).all()


def test_audio_feature_cache_reuses_same_wav(tmp_path):
    wav = tmp_path / "stimulus.wav"
    cache = tmp_path / "features.npz"
    _write_test_wav(wav, duration_s=1.0)
    f1, t1, m1 = extract_acoustic_features_cached(wav, cache, target_sfreq=50, n_mels=12)
    f2, t2, m2 = extract_acoustic_features_cached(wav, cache, target_sfreq=50, n_mels=12)
    np.testing.assert_allclose(f1, f2)
    np.testing.assert_allclose(t1, t2)
    assert m1["feature_names"] == m2["feature_names"]
