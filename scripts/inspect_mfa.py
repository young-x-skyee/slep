#!/usr/bin/env python3
"""Validate and summarise the bundled Kymata English MFA word files."""
from pathlib import Path
import argparse
import numpy as np


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    package_data = project_root / "src" / "slep_thermo" / "data"
    parser.add_argument("--words", type=Path, default=package_data / "mfa_word.txt")
    parser.add_argument("--word-times", type=Path, default=package_data / "mfa_word_stime.txt")
    args = parser.parse_args()

    word_lines = [x.strip() for x in args.words.read_text(encoding="utf-8-sig").splitlines()]
    time_lines = [x.strip() for x in args.word_times.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    times = np.asarray([float(x) for x in time_lines], dtype=float)
    if len(word_lines) != len(times):
        raise SystemExit(f"Mismatch: {len(word_lines)} word rows vs {len(times)} time rows")
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise SystemExit("Onset times must be finite and non-decreasing")

    is_pause = np.asarray([w.lower() in {"<sp>", "sp"} for w in word_lines])
    lexical_times = times[~is_pause]
    lexical_words = [w for w, pause in zip(word_lines, is_pause) if not pause]
    print(f"Transcript:       {args.words}")
    print(f"Onsets:           {args.word_times}")
    print(f"Rows in each:     {len(word_lines)}")
    print(f"Pause markers:    {int(is_pause.sum())}")
    print(f"Lexical words:    {len(lexical_words)}")
    print(f"First row:        {word_lines[0]!r} @ {times[0]:.3f}s")
    print(f"First word:       {lexical_words[0]!r} @ {lexical_times[0]:.3f}s")
    print(f"Last word:        {lexical_words[-1]!r} @ {lexical_times[-1]:.3f}s")
    print(f"Last row:         {word_lines[-1]!r} @ {times[-1]:.3f}s")
    print(f"Onsets after 400: {int(np.sum(lexical_times > 400.0))}")


if __name__ == "__main__":
    main()
