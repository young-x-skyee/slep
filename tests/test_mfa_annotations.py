from pathlib import Path
import numpy as np
from slep_thermo.pipeline import read_mfa_words, DEFAULT_WORDS_PATH, DEFAULT_WORD_TIMES_PATH


def test_bundled_mfa_annotations_are_aligned_and_valid():
    words, times = read_mfa_words(DEFAULT_WORDS_PATH, DEFAULT_WORD_TIMES_PATH)
    assert len(words) == 1396
    assert len(times) == 1396
    assert words[0] == "there's"
    assert words[-1] == "follow"
    assert times[0] == 0.04
    assert times[-1] == 401.1
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) >= 0)


def test_mfa_defaults_use_single_package_data_copy():
    project_root = Path(__file__).resolve().parents[1]
    assert DEFAULT_WORDS_PATH == project_root / "src" / "slep_thermo" / "data" / "mfa_word.txt"
    assert DEFAULT_WORD_TIMES_PATH == project_root / "src" / "slep_thermo" / "data" / "mfa_word_stime.txt"
    assert not (project_root / "stimuli" / "audio" / "mfa_word.txt").exists()
    assert not (project_root / "stimuli" / "audio" / "mfa_word_stime.txt").exists()
