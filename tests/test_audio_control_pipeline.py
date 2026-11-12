import numpy as np
import pandas as pd
import pytest

from slep_thermo.pipeline import (
    build_parser,
    apply_sensor_noise_whitener,
    crossvalidated_sensor_states_and_nll,
    crossvalidated_repetition_states_and_nll,
    crossvalidated_semantic_nll,
    estimate_joint_state_association,
    fit_neural_state_transform,
    iaaft_dwell_null,
    iaaft_surrogate,
    list_repetition_evoked_paths,
    purged_window_splits,
    state_dynamics,
    soft_state_window_metrics,
    state_window_metrics,
)


def test_crossvalidated_nll_returns_neural_acoustic_and_joint_models():
    rng = np.random.default_rng(3)
    n = 900
    times = np.arange(n) / 50.0
    acoustic = rng.normal(size=(n, 8)).astype(np.float32)
    neural = rng.normal(size=(n, 5)).astype(np.float32)
    # Construct targets with both acoustic and neural contributions.
    y = (0.25 * acoustic[:, :4] + 0.4 * neural[:, :4] +
         0.1 * rng.normal(size=(n, 4))).astype(np.float32)
    nll = crossvalidated_semantic_nll(
        neural, y, times, n_splits=3, gap_s=0.2, ridge_alpha=1.0,
        acoustic_features=acoustic,
    )
    assert set(nll) == {"neural_only", "acoustic_only", "joint"}
    for values in nll.values():
        assert values.shape == (n,)
        assert np.isfinite(values).all()


def test_neural_state_transform_is_fit_without_held_out_samples():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(400, 8)).astype(np.float32)
    test = rng.normal(size=(120, 8)).astype(np.float32)
    extreme_test = test + 1000.0
    train_z, _, _, _, _, _ = fit_neural_state_transform(train, test, 4, 3, 11)
    extreme_train_z, _, _, _, _, _ = fit_neural_state_transform(train, extreme_test, 4, 3, 11)
    np.testing.assert_allclose(train_z, extreme_train_z)


def test_iaaft_surrogate_preserves_marginal_and_temporal_spectrum():
    rng = np.random.default_rng(42)
    innovations = rng.normal(size=512)
    signal = np.empty_like(innovations)
    signal[0] = innovations[0]
    for index in range(1, len(signal)):
        signal[index] = 0.85 * signal[index - 1] + innovations[index]

    surrogate = iaaft_surrogate(signal, np.random.default_rng(9), n_iterations=200)

    np.testing.assert_array_equal(np.sort(surrogate), np.sort(signal))
    target_spectrum = np.abs(np.fft.rfft(signal))
    surrogate_spectrum = np.abs(np.fft.rfft(surrogate))
    assert np.corrcoef(target_spectrum, surrogate_spectrum)[0, 1] > 0.95


def test_purged_window_splits_keep_complete_windows_and_gap():
    times = np.arange(2000) / 50.0
    window_ids = np.floor(times / 4.0).astype(int)
    splits = list(purged_window_splits(
        times, n_splits=4, window_s=4.0, gap_s=0.5, window_origin_s=0.0,
    ))

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert not set(window_ids[train_idx]).intersection(window_ids[test_idx])
        assert np.all(
            (times[train_idx] < times[test_idx[0]] - 0.5)
            | (times[train_idx] > times[test_idx[-1]] + 0.5)
        )


def test_state_dynamics_skips_invalid_labels_and_splits_dwell_runs():
    labels = np.array([0.0, 0.0, np.nan, 1.0, 1.0, 2.0, -1.0, 2.0])
    transition, dwell = state_dynamics(labels, sfreq=2.0, n_states=3)

    assert transition.loc[0, "to_0"] == 1.0
    assert transition.loc[1, "to_1"] == 0.5
    assert transition.loc[1, "to_2"] == 0.5
    assert dwell.set_index("state").loc[0, "n_runs"] == 1
    assert dwell.set_index("state").loc[2, "n_runs"] == 2


def test_state_dynamics_does_not_join_fold_specific_state_labels():
    labels = np.array([0, 0, 1, 1, 0, 0])
    fold_ids = np.array([0, 0, 0, 0, 1, 1])

    transition, dwell = state_dynamics(labels, sfreq=2.0, n_states=2, fold_ids=fold_ids)

    fold_zero_transition = transition.query("fold == 0 and from_state == 1").iloc[0]
    assert fold_zero_transition["to_0"] == 0.0
    assert set(dwell.loc[dwell["state"] == 0, "fold"]) == {0, 1}
    assert (dwell.loc[dwell["state"] == 0, "n_runs"] == 1).all()


def test_crossvalidated_sensor_states_and_nll_returns_aligned_held_out_values():
    rng = np.random.default_rng(8)
    n = 900
    times = np.arange(n) / 50.0
    sensor_time = rng.normal(size=(6, n)).astype(np.float32)
    acoustic = rng.normal(size=(n, 5)).astype(np.float32)
    y = (0.3 * sensor_time[:4].T + 0.2 * acoustic[:, :4] +
         0.1 * rng.normal(size=(n, 4))).astype(np.float32)

    labels, nll, fold_ids, fold_models, responsibilities, n_pca, n_states = crossvalidated_sensor_states_and_nll(
        sensor_time, y, acoustic, times,
        n_pca=3, n_states=3, n_splits=3, gap_s=0.2, window_s=2.0,
        ridge_alpha=1.0, random_state=5,
    )

    assert labels.shape == (n,)
    assert np.isfinite(labels).all()
    assert fold_ids.shape == (n,)
    assert set(np.unique(fold_ids)) == {0, 1, 2}
    assert len(fold_models) == 3
    assert responsibilities.shape == (n, n_states)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0, atol=1e-6)
    assert n_pca == 3
    assert n_states == 3
    assert set(nll) == {"neural_only", "acoustic_only", "joint"}
    for values in nll.values():
        assert values.shape == (n,)
        assert np.isfinite(values).all()

    dwell_null = iaaft_dwell_null(
        sensor_time, times, labels, fold_models, window_s=2.0, sfreq=50.0,
        n_surrogates=2, n_iterations=10, random_state=12,
    )
    assert len(dwell_null) == 9
    assert {"observed_mean_dwell_s", "null_mean_dwell_ci_low_s", "p_observed_mean_ge_null"}.issubset(
        dwell_null.columns
    )
    assert dwell_null["p_observed_mean_ge_null"].between(0.0, 1.0).all()


def test_leave_one_repetition_out_predicts_each_repeat_with_other_repeats():
    rng = np.random.default_rng(16)
    n_repetitions = 4
    n = 300
    times = np.arange(n) / 50.0
    base = rng.normal(size=(6, n)).astype(np.float32)
    sensor_repetitions = np.stack([
        base + rng.normal(scale=0.1, size=base.shape).astype(np.float32)
        for _ in range(n_repetitions)
    ])
    acoustic = rng.normal(size=(n, 4)).astype(np.float32)
    y = (0.25 * sensor_repetitions[0, :3].T + 0.2 * acoustic[:, :3] +
         0.1 * rng.normal(size=(n, 3))).astype(np.float32)

    labels, nll, fold_ids, fold_models, responsibilities, n_pca, n_states = crossvalidated_repetition_states_and_nll(
        sensor_repetitions, y, acoustic, times,
        n_pca=3, n_states=3, window_s=2.0, ridge_alpha=1.0, random_state=4,
    )

    assert labels.shape == (n_repetitions * n,)
    assert np.isfinite(labels).all()
    assert set(np.unique(fold_ids)) == set(range(n_repetitions))
    assert len(fold_models) == n_repetitions
    assert responsibilities.shape == (n_repetitions * n, n_states)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0, atol=1e-6)
    assert n_pca == 3
    assert n_states == 3
    for values in nll.values():
        assert np.isfinite(values).all()
    repetition_null = iaaft_dwell_null(
        sensor_repetitions, np.tile(times, n_repetitions), labels, fold_models,
        window_s=2.0, sfreq=50.0, n_surrogates=2, n_iterations=5, random_state=8,
    )
    assert len(repetition_null) == n_repetitions * 3
    assert set(repetition_null["fold"]) == set(range(n_repetitions))


def test_soft_state_metrics_recover_occupancy_when_hard_labels_collapse():
    responsibilities = np.tile(np.array([[0.55, 0.45]]), (100, 1))
    nll = {
        "neural_only": np.linspace(0.5, 1.5, 100),
        "acoustic_only": np.linspace(0.6, 1.6, 100),
        "joint": np.linspace(0.4, 1.4, 100),
    }
    metrics, entropy = soft_state_window_metrics(
        responsibilities, nll, np.arange(100) / 50.0,
        window_s=2.0, min_effective_count=10,
    )

    assert metrics["state"].nunique() == 2
    np.testing.assert_allclose(metrics.groupby("state").q.first().sort_index(), [0.55, 0.45])
    np.testing.assert_allclose(metrics.groupby("state").p_kk.first().sort_index(), [0.55, 0.45])
    assert entropy.entropy_norm.iloc[0] > 0.9


def test_repetition_file_discovery_sorts_numeric_ids_and_requires_two(tmp_path):
    for name in ("p_rep10.fif", "p_rep2.fif", "p_rep0.fif", "other_rep1.fif"):
        (tmp_path / name).touch()
    paths = list_repetition_evoked_paths(tmp_path, "p")
    assert [path.name for path in paths] == ["p_rep0.fif", "p_rep2.fif", "p_rep10.fif"]
    with pytest.raises(FileNotFoundError):
        list_repetition_evoked_paths(tmp_path, "other")


def test_cv_scheme_defaults_to_temporal_and_accepts_repetition_mode():
    parser = build_parser()
    required = ["--data-dir", "/tmp/data", "--out", "/tmp/out"]
    assert parser.parse_args(required).cv_scheme == "temporal"
    assert parser.parse_args(
        required + ["--cv-scheme", "leave-one-repetition-out"]
    ).cv_scheme == "leave-one-repetition-out"


def test_noise_whitener_applies_to_temporal_and_repetition_data():
    whitener = np.diag([2.0, 0.5])
    temporal = np.ones((2, 4), dtype=np.float32)
    repetitions = np.ones((3, 2, 4), dtype=np.float32)

    np.testing.assert_allclose(apply_sensor_noise_whitener(temporal, whitener), [[2] * 4, [0.5] * 4])
    whitened_repetitions = apply_sensor_noise_whitener(repetitions, whitener)
    assert whitened_repetitions.shape == repetitions.shape
    np.testing.assert_allclose(whitened_repetitions[:, 0], 2.0)
    np.testing.assert_allclose(whitened_repetitions[:, 1], 0.5)


def test_state_metrics_include_acoustic_baseline_and_incremental_neural_information():
    rng = np.random.default_rng(4)
    n = 1000
    labels = np.tile(np.repeat(np.arange(5), 40), 5)
    labels = labels[:n]
    times = np.arange(n) / 50.0
    nll = {
        "neural_only": rng.normal(1.0, 0.1, n),
        "acoustic_only": rng.normal(1.2, 0.1, n),
        "joint": rng.normal(0.9, 0.1, n),
    }
    metrics, entropy = state_window_metrics(
        labels, nll, times, n_states=5, window_s=4.0, min_count=5,
    )
    required = {
        "semantic_potential_nll", "acoustic_baseline_nll",
        "acoustic_controlled_joint_nll", "neural_incremental_information_nats",
    }
    assert required.issubset(metrics.columns)
    assert not entropy.empty


def test_state_metrics_include_fold_window_self_transition_probability():
    labels = np.array([0, 0, 1, 1, 1, 0, 0])
    times = np.arange(len(labels), dtype=float)
    fold_ids = np.zeros(len(labels), dtype=int)
    nll = {
        "neural_only": np.arange(len(labels), dtype=float),
        "acoustic_only": np.arange(len(labels), dtype=float) + 1,
        "joint": np.arange(len(labels), dtype=float) + 2,
    }

    metrics, _ = state_window_metrics(
        labels, nll, times, n_states=2, window_s=4.0, min_count=1, fold_ids=fold_ids,
    )
    first_window = metrics.query("window == 0").set_index("state")
    assert first_window.loc[0, "p_kk"] == 0.5
    assert first_window.loc[1, "p_kk"] == 1.0
    second_window = metrics.query("window == 1").set_index("state")
    assert second_window.loc[1, "p_kk"] == 0.0
    assert second_window.loc[0, "p_kk"] == 1.0


def test_joint_state_association_separates_potential_and_persistence_effects():
    rows = []
    potential = np.array([0.1, 0.4, 0.8])
    persistence = np.array([0.2, 0.8, 0.3])
    for window in range(8):
        for state in range(3):
            rows.append({
                "window": window,
                "state": state,
                "n_samples": 100,
                "semantic_potential_nll": potential[state],
                "p_kk": persistence[state],
                "self_information": 1.5 * potential[state] + 0.7 * persistence[state] + window,
            })

    result = estimate_joint_state_association(pd.DataFrame(rows), n_boot=100, seed=3)

    assert result["beta_potential_adjusted_for_persistence"] == pytest.approx(1.5)
    assert result["beta_persistence_adjusted_for_potential"] == pytest.approx(0.7)
