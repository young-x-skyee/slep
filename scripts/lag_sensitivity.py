#!/usr/bin/env python3
"""Run neural response latency sensitivity analysis for SLEP Hypothesis IV.

Sweeps exploratory biological neural lag (e.g. 0.0 to 0.4 s) while keeping
fixed hardware delivery delay and drift constant. Evaluates how Boltzmann slope
(beta) varies as a function of neural lag across sensor modalities.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slep_thermo.aggregate import summarise_group

LOG = logging.getLogger("slep_lag_sensitivity")


def plot_lag_sensitivity(summary_df: pd.DataFrame, out_png: Path) -> None:
    """Plot mean Boltzmann beta (+/- 95% CI) vs biological neural latency."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    modalities = sorted(summary_df["modality"].unique())

    colors = {"EEG": "tab:blue", "MEG_mag": "tab:orange", "MEG_grad": "tab:green"}
    for mod in modalities:
        m_df = summary_df[summary_df["modality"] == mod].sort_values("lag_s")
        lags = m_df["lag_s"].to_numpy()
        betas = m_df["mean_beta"].to_numpy()
        ci_low = m_df["beta_ci95_low"].to_numpy()
        ci_high = m_df["beta_ci95_high"].to_numpy()

        color = colors.get(mod, "tab:gray")
        ax.plot(lags, betas, marker="o", label=mod, color=color, linewidth=2)
        if np.isfinite(ci_low).any() and np.isfinite(ci_high).any():
            ax.fill_between(lags, ci_low, ci_high, color=color, alpha=0.15)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Neural response latency (seconds)")
    ax.set_ylabel("Group mean Boltzmann slope (β)")
    ax.set_title("Neural latency sensitivity: Boltzmann slope vs response lag")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    LOG.info("Saved lag sensitivity curve to %s", out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing {subject}-ave.fif files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for sensitivity run results")
    parser.add_argument("--lags", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                        help="Neural response lag values in seconds to test")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Optional subset of subjects to run")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Optional limit on number of subjects")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for lag in sorted(args.lags):
        lag_out = args.out / f"lag_{lag:.3f}s"
        cmd = [
            sys.executable, "-m", "slep_thermo.pipeline",
            "--data-dir", str(args.data_dir),
            "--out", str(lag_out),
            "--neural-lag-s", str(lag),
        ]
        if args.audio:
            cmd.extend(["--audio", str(args.audio)])
        if args.subjects:
            cmd.extend(["--subjects", *args.subjects])
        if args.max_subjects:
            cmd.extend(["--max-subjects", str(args.max_subjects)])
        if args.overwrite:
            cmd.append("--overwrite")

        LOG.info("Running pipeline for lag %.3f s...", lag)
        subject_summaries_file = lag_out / "all_subject_summaries.csv"
        if not subject_summaries_file.is_file() or args.overwrite:
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                LOG.error("Run failed for lag %.3f s (exit code %d)", lag, res.returncode)
                continue

        if not subject_summaries_file.is_file():
            LOG.warning("No summary found for lag %.3f s at %s", lag, subject_summaries_file)
            continue

        df = pd.read_csv(subject_summaries_file)
        for mod, group in df.groupby("modality"):
            mod_summary = summarise_group(group)
            mod_summary["lag_s"] = float(lag)
            mod_summary["modality"] = str(mod)
            summary_rows.append(mod_summary)

    if not summary_rows:
        LOG.error("No valid lag summaries collected.")
        sys.exit(1)

    combined_df = pd.DataFrame(summary_rows)
    summary_csv = args.out / "lag_sensitivity_summary.csv"
    combined_df.to_csv(summary_csv, index=False)
    LOG.info("Saved lag sensitivity summary to %s", summary_csv)

    plot_png = args.out / "lag_sensitivity_curve.png"
    plot_lag_sensitivity(combined_df, plot_png)


if __name__ == "__main__":
    main()
