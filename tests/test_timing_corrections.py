from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slep_thermo.pipeline import (
    audio_time_to_evoked_time,
    evoked_time_to_audio_time,
    crop_evoked_to_mapped_stimulus,
    load_timing_corrections,
)


def test_affine_timing_map_matches_kymata_convention():
    audio_t = np.array([0.0, 100.0, 200.0, 400.0])
    mapped = audio_time_to_evoked_time(
        audio_t,
        stimulus_onset_s=0.0,
        stimulus_shift_correction=0.0005,
        stimulus_delivery_latency_s=0.08,
        neural_lag_s=0.2,
    )
    expected = 0.28 + 1.0005 * audio_t
    np.testing.assert_allclose(mapped, expected)
    np.testing.assert_allclose(
        evoked_time_to_audio_time(
            mapped, 0.0, 0.0005, 0.08, 0.2
        ),
        audio_t,
        atol=1e-12,
    )


def test_zero_corrections_are_identity_plus_neural_lag():
    audio_t = np.array([0.0, 1.0, 399.98])
    mapped = audio_time_to_evoked_time(audio_t, 0.0, 0.0, 0.0, 0.2)
    np.testing.assert_allclose(mapped, audio_t + 0.2)


def test_reject_invalid_shift_scale():
    with pytest.raises(ValueError):
        audio_time_to_evoked_time(1.0, 0.0, -1.0, 0.0, 0.0)


def test_crop_uses_mapped_start_and_end():
    sfreq = 1000.0
    times = np.arange(-0.2, 402.801, 1.0 / sfreq)
    data = np.zeros((2, len(times)), dtype=np.float32)
    evoked = SimpleNamespace(
        info={"sfreq": sfreq}, times=times, data=data
    )
    cropped, cropped_times, returned_sfreq, meta = crop_evoked_to_mapped_stimulus(
        evoked,
        stimulus_onset_s=0.0,
        duration_s=400.0,
        stimulus_shift_correction=0.0005,
        stimulus_delivery_latency_s=0.08,
        neural_lag_s=0.2,
    )
    assert returned_sfreq == sfreq
    assert cropped.shape[0] == 2
    assert cropped_times[0] >= 0.28 - 1.0 / sfreq
    assert cropped_times[-1] < 400.28 + 400.0 * 0.0005
    assert meta["mapped_evoked_start_s"] == pytest.approx(0.28)
    assert meta["mapped_evoked_stop_exclusive_s"] == pytest.approx(400.48)


def test_load_subject_timing_overrides(tmp_path: Path):
    csv = tmp_path / "timing.csv"
    pd.DataFrame([
        {"subject": "participant_01", "stimulus_shift_correction": 0.0004,
         "stimulus_delivery_latency_s": 0.075, "neural_lag_s": 0.18},
        {"subject": "participant_02", "stimulus_shift_correction": np.nan,
         "stimulus_delivery_latency_s": 0.08, "neural_lag_s": np.nan},
    ]).to_csv(csv, index=False)
    result = load_timing_corrections(csv)
    assert result["participant_01"]["stimulus_shift_correction"] == pytest.approx(0.0004)
    assert result["participant_02"] == {"stimulus_delivery_latency_s": pytest.approx(0.08)}


def test_cli_defaults_use_supplied_fixed_delivery_corrections():
    from slep_thermo.pipeline import build_parser

    args = build_parser().parse_args([
        "--data-dir", "/tmp/evoked",
        "--out", "/tmp/slep-out",
    ])
    assert args.stimulus_shift_correction == pytest.approx(0.0005404)
    assert args.stimulus_delivery_latency_s == pytest.approx(0.026)


def test_fixed_timing_values_map_audio_end_correctly():
    # At 400 s, drift contributes 0.21616 s; delivery adds another 0.026 s.
    mapped = audio_time_to_evoked_time(
        400.0,
        stimulus_onset_s=0.0,
        stimulus_shift_correction=0.0005404,
        stimulus_delivery_latency_s=0.026,
        neural_lag_s=0.0,
    )
    assert mapped == pytest.approx(400.24216)
