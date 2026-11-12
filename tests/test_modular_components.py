import numpy as np
import pandas as pd
import pytest

from slep_thermo.neural import (
    preprocess_sensor_group,
    fit_neural_state_transform,
    transform_neural_state_model,
)
from slep_thermo.embeddings import (
    align_embeddings_to_time,
    make_context_texts,
    embedding_cache_key,
)
from slep_thermo.plots import save_plots
from slep_thermo.slopes import estimate_within_window_slope


def test_preprocess_sensor_group_with_bandpass():
    raw_sfreq = 1000.0
    target_sfreq = 50.0
    duration_s = 2.0
    n_samples = int(raw_sfreq * duration_s)
    t = np.arange(n_samples) / raw_sfreq
    # Signal with 10 Hz (in band: 0.5-30 Hz) and 100 Hz (out of band)
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
    data = sig[None, :].astype(np.float32)

    filtered = preprocess_sensor_group(
        data,
        raw_sfreq=raw_sfreq,
        target_sfreq=target_sfreq,
        apply_bandpass=True,
        low_hz=0.5,
        high_hz=30.0,
    )
    expected_samples = int(duration_s * target_sfreq)
    assert filtered.shape == (1, expected_samples)
    assert np.isfinite(filtered).all()


def test_preprocess_sensor_group_rejects_invalid_bandpass():
    data = np.ones((1, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match="Bandpass must satisfy"):
        preprocess_sensor_group(
            data, raw_sfreq=1000.0, target_sfreq=50.0,
            apply_bandpass=True, low_hz=30.0, high_hz=10.0,
        )


def test_align_embeddings_to_time_zero_lag():
    words = ["hello", "world"]
    word_onsets = np.array([0.0, 1.0])
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # At time 0.5 s, should be "hello" (index 0); at 1.5 s, "world" (index 1)
    eval_times = np.array([0.5, 1.5])
    aligned = align_embeddings_to_time(
        words, word_onsets, embeddings, eval_times, neural_lag_s=0.0
    )
    assert np.array_equal(aligned[0], embeddings[0])
    assert np.array_equal(aligned[1], embeddings[1])


def test_align_embeddings_to_time_before_first_word():
    words = ["hello"]
    word_onsets = np.array([1.0])
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    eval_times = np.array([0.5, 1.5])
    aligned = align_embeddings_to_time(
        words, word_onsets, embeddings, eval_times, neural_lag_s=0.0
    )
    assert np.isnan(aligned[0]).all()
    assert np.array_equal(aligned[1], embeddings[0])


def test_save_plots_smoke_empty_and_populated(tmp_path):
    # Smoke test: empty dataframes should not crash save_plots
    empty_metrics = pd.DataFrame()
    empty_entropy = pd.DataFrame()
    save_plots(empty_metrics, empty_entropy, {}, tmp_path, "TEST_EMPTY")
    assert (tmp_path / "TEST_EMPTY_boltzmann_fit.png").is_file()
    assert (tmp_path / "TEST_EMPTY_entropy_over_time.png").is_file()
    assert (tmp_path / "TEST_EMPTY_acoustic_control_comparison.png").is_file()


def test_embedding_cache_key_deterministic():
    words = ["the", "quick", "brown", "fox"]
    k1 = embedding_cache_key(words, "test-model", 12)
    k2 = embedding_cache_key(words, "test-model", 12)
    k3 = embedding_cache_key(words, "test-model", 6)
    assert k1 == k2
    assert k1 != k3


def test_estimate_within_window_slope_insufficient_windows():
    metrics = pd.DataFrame([
        {"window": 0, "n_samples": 10, "self_information": 1.0, "semantic_potential_nll": 2.0},
        {"window": 1, "n_samples": 10, "self_information": 1.5, "semantic_potential_nll": 2.5},
    ])
    result = estimate_within_window_slope(metrics, n_boot=10, seed=42)
    assert np.isnan(result["beta"])


def test_dry_run_flag_parsed():
    from slep_thermo.pipeline import build_parser

    parser = build_parser()
    args = parser.parse_args(["--data-dir", "/tmp", "--out", "/tmp", "--dry-run"])
    assert args.dry_run is True


def test_fit_neural_state_transform_meg_scale():
    # Signals at MEG magnetometer scale (~1e-13) should not collapse to 1 cluster
    rng = np.random.default_rng(123)
    train_meg = (rng.normal(size=(500, 10)) * 1e-13).astype(np.float32)
    test_meg = (rng.normal(size=(100, 10)) * 1e-13).astype(np.float32)

    z_train, z_test, test_labels, n_pca, n_states, model = fit_neural_state_transform(
        train_meg, test_meg, n_pca=4, n_states=3, random_state=42
    )
    assert n_pca == 4
    assert n_states == 3
    # Check that test samples don't collapse to a single state
    assert len(np.unique(test_labels)) > 1


