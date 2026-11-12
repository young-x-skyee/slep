#!/usr/bin/env python3
"""Print timing and channel metadata for an MNE Evoked FIF file."""
import argparse
from collections import Counter
from pathlib import Path
import mne


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fif", type=Path)
    p.add_argument("--condition", type=int, default=0)
    args = p.parse_args()
    evoked = mne.read_evokeds(str(args.fif), condition=args.condition,
                              baseline=None, proj=True, verbose="ERROR")
    types = [mne.channel_type(evoked.info, i) for i in range(len(evoked.ch_names))]
    print(f"File:       {args.fif}")
    print(f"Comment:    {evoked.comment}")
    print(f"Shape:      {evoked.data.shape} (channels, samples)")
    print(f"Sampling:   {evoked.info['sfreq']} Hz")
    print(f"Time span:  {evoked.times[0]:.6f} to {evoked.times[-1]:.6f} s")
    print(f"Bads:       {evoked.info['bads']}")
    print(f"Types:      {dict(Counter(types))}")
    print("First channels:", evoked.ch_names[:12])
    print("Last channels: ", evoked.ch_names[-12:])
    print("\nAssuming event-time zero is podcast onset, verify that Evoked.times includes 0..400 s.")


if __name__ == "__main__":
    main()
