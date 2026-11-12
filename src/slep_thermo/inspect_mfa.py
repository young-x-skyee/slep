"""Inspect and validate the bundled English podcast MFA word annotations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .pipeline import read_mfa_words, DEFAULT_WORDS_PATH, DEFAULT_WORD_TIMES_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS_PATH)
    parser.add_argument("--word-times", type=Path, default=DEFAULT_WORD_TIMES_PATH)
    args = parser.parse_args()
    words, times = read_mfa_words(args.words, args.word_times)
    raw_words = [x.strip() for x in args.words.read_text(encoding="utf-8-sig").splitlines()]
    raw_times = np.asarray([float(x.strip()) for x in args.word_times.read_text(encoding="utf-8-sig").splitlines() if x.strip()])
    n_pauses = sum(x.lower() in {"<sp>", "sp"} for x in raw_words)
    print(f"Transcript:             {args.words}")
    print(f"Word onsets:            {args.word_times}")
    print(f"Original aligned rows:  {len(raw_words)}")
    print(f"Pause markers removed:  {n_pauses}")
    print(f"Lexical words retained: {len(words)}")
    print(f"Onsets monotonic:       {bool(np.all(np.diff(raw_times) >= 0))}")
    print(f"First lexical word:     {words[0]!r} at {times[0]:.3f}s")
    print(f"Last lexical word:      {words[-1]!r} at {times[-1]:.3f}s")
    print(f"Last raw row:           {raw_words[-1]!r} at {raw_times[-1]:.3f}s")
    print(f"Lexical onsets >400s:   {int(np.sum(times > 400.0))}")


if __name__ == "__main__":
    main()
