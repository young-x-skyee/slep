#!/usr/bin/env python3
"""Inspect the podcast WAV and report acoustic feature extraction settings."""
from __future__ import annotations

import argparse
from pathlib import Path

from slep_thermo.audio import extract_acoustic_features


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_audio = project_root / "stimuli" / "audio" / "stimulus.wav"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, default=default_audio)
    parser.add_argument("--target-sfreq", type=float, default=50.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    args = parser.parse_args()

    features, times, metadata = extract_acoustic_features(
        args.audio, target_sfreq=args.target_sfreq,
        n_mels=args.n_mels, frame_ms=args.frame_ms,
    )
    print(f"Audio:                 {metadata['path']}")
    print(f"Original sample rate:  {metadata['original_sample_rate_hz']} Hz")
    print(f"Original shape:        {metadata['original_shape']}")
    print(f"Original dtype:        {metadata['original_dtype']}")
    print(f"Duration:              {metadata['duration_s']:.3f} s")
    print(f"Feature frame rate:    {metadata['target_feature_rate_hz']:.3f} Hz")
    print(f"Feature matrix:        {features.shape}")
    print(f"Feature time range:    {times[0]:.3f}..{times[-1]:.3f} s")
    print("Features:")
    for name in metadata["feature_names"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
