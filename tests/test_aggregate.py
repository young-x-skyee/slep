import pandas as pd
from slep_thermo.aggregate import summarise_group


def test_group_summary_uses_participant_betas():
    frame = pd.DataFrame({"beta": [0.5, 1.0, -0.25]})
    out = summarise_group(frame)
    assert out["n_participants"] == 3
    assert abs(out["mean_beta"] - (1.25 / 3)) < 1e-12
    assert out["proportion_beta_positive"] == 2 / 3
    assert out["p_value_vs_zero"] >= 0


def test_additional_audio_control_metrics_are_summarised():
    from slep_thermo.aggregate import summarise_numeric_column
    result = summarise_numeric_column(pd.Series([0.1, 0.2, 0.3]))
    assert result["n"] == 3
    assert abs(result["mean"] - 0.2) < 1e-12
    assert result["p_value_vs_zero"] >= 0


def test_save_forest_plot(tmp_path):
    from slep_thermo.aggregate import save_forest_plot
    frame = pd.DataFrame([
        {"subject": "sub-01", "modality": "EEG", "beta": 0.5, "ci_low": 0.2, "ci_high": 0.8},
        {"subject": "sub-02", "modality": "EEG", "beta": 1.0, "ci_low": 0.7, "ci_high": 1.3},
        {"subject": "sub-01", "modality": "MEG_mag", "beta": 0.2, "ci_low": -0.1, "ci_high": 0.5},
    ])
    out_png = tmp_path / "forest.png"
    save_forest_plot(frame, out_png)
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000

