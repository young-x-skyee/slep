"""Diagnostic plots for the SLEP thermodynamic analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_plots(metrics: pd.DataFrame, entropy: pd.DataFrame, summary: Dict[str, float], out: Path, name: str):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    if not metrics.empty:
        sizes = 15 + 0.15 * metrics["n_samples"].to_numpy()
        ax.scatter(metrics["semantic_potential_nll"], metrics["self_information"], s=sizes, alpha=0.65)
        beta = summary.get("beta", np.nan)
        if np.isfinite(beta):
            x = metrics["semantic_potential_nll"].to_numpy()
            y = metrics["self_information"].to_numpy()
            x_grid = np.linspace(x.min(), x.max(), 100)
            # Descriptive line; fit itself uses within-window centring.
            ax.plot(x_grid, np.mean(y) + beta * (x_grid - np.mean(x)), "--", linewidth=2)
    ax.set_xlabel("Semantic potential proxy: held-out neural-to-text NLL")
    ax.set_ylabel("State self-information: -log q")
    ax.set_title(f"{name}: coarse-grained Boltzmann relationship")
    fig.tight_layout()
    fig.savefig(out / f"{name}_boltzmann_fit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    if not entropy.empty:
        if "fold" in entropy.columns and entropy["fold"].nunique() > 1:
            for fold, fold_entropy in entropy.groupby("fold", sort=True):
                ax.plot(
                    fold_entropy["start_s"], fold_entropy["entropy_norm"],
                    marker="o", label=f"fold {fold}",
                )
            ax.legend(frameon=False, ncol=2)
        else:
            ax.plot(entropy["start_s"], entropy["entropy_norm"], marker="o")
    ax.set_xlabel("Time from podcast onset (s)")
    ax.set_ylabel("Normalised state entropy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}: coarse-grained neural-state entropy")
    fig.tight_layout()
    fig.savefig(out / f"{name}_entropy_over_time.png", dpi=160)
    plt.close(fig)

    # Compare semantic decoders with and without acoustic controls.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if not metrics.empty:
        series = [
            ("Neural only", "semantic_potential_nll"),
            ("Acoustic only", "acoustic_baseline_nll"),
            ("Neural + acoustic", "acoustic_controlled_joint_nll"),
        ]
        for label, column in series:
            window_ids, window_values = [], []
            for window_id, group in metrics.groupby("window", sort=True):
                value = np.average(group[column], weights=group["n_samples"])
                window_ids.append(window_id)
                window_values.append(value)
            # Plot against the numbered analysis windows (same windows for all models).
            ax.plot(window_ids, window_values, marker="o", label=label)
    ax.set_xlabel("Narrative window index")
    ax.set_ylabel("Held-out Gaussian NLL (nats / target dimension)")
    ax.set_title(f"{name}: acoustic-control decoder comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{name}_acoustic_control_comparison.png", dpi=160)
    plt.close(fig)
