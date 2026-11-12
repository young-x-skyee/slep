"""Aggregate participant-level SLEP thermodynamic estimates."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("slep_thermo.aggregate")


def summarise_group(frame: pd.DataFrame) -> dict:
    """Summarise participant-level Boltzmann slope (beta) estimates and compute one-sample t-test."""
    beta = pd.to_numeric(frame.get("beta"), errors="coerce")
    beta = beta[np.isfinite(beta)]
    n = int(beta.size)
    result = {
        "n_participants": n,
        "mean_beta": float(beta.mean()) if n else np.nan,
        "median_beta": float(beta.median()) if n else np.nan,
        "sd_beta": float(beta.std(ddof=1)) if n > 1 else np.nan,
        "proportion_beta_positive": float((beta > 0).mean()) if n else np.nan,
        "t_statistic_vs_zero": np.nan,
        "p_value_vs_zero": np.nan,
        "beta_ci95_low": np.nan,
        "beta_ci95_high": np.nan,
        "median_temperature_positive_beta": np.nan,
    }
    if n > 1:
        se = stats.sem(beta)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        result["beta_ci95_low"] = float(beta.mean() - tcrit * se)
        result["beta_ci95_high"] = float(beta.mean() + tcrit * se)
        test = stats.ttest_1samp(beta, popmean=0.0, nan_policy="omit")
        result["t_statistic_vs_zero"] = float(test.statistic)
        result["p_value_vs_zero"] = float(test.pvalue)
    positive = beta[beta > 0]
    if len(positive):
        result["median_temperature_positive_beta"] = float(np.median(1.0 / positive))
    return result


def summarise_numeric_column(values: pd.Series) -> dict:
    """Participant-level descriptive and one-sample summaries for one metric."""
    x = pd.to_numeric(values, errors="coerce")
    x = x[np.isfinite(x)]
    n = int(x.size)
    result = {
        "n": n, "mean": float(x.mean()) if n else np.nan,
        "median": float(x.median()) if n else np.nan,
        "sd": float(x.std(ddof=1)) if n > 1 else np.nan,
        "ci95_low": np.nan, "ci95_high": np.nan, "p_value_vs_zero": np.nan,
    }
    if n > 1:
        se = stats.sem(x)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        result["ci95_low"] = float(x.mean() - tcrit * se)
        result["ci95_high"] = float(x.mean() + tcrit * se)
        result["p_value_vs_zero"] = float(stats.ttest_1samp(x, popmean=0.0).pvalue)
    return result


def save_forest_plot(frame: pd.DataFrame, out_path: Path) -> None:
    """Generate a forest plot of individual participant beta estimates per sensor modality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modalities = sorted(frame["modality"].dropna().unique())
    if not modalities:
        return
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.0 * len(modalities), max(4.0, 0.3 * frame["subject"].nunique() + 2)), sharey=False)
    if len(modalities) == 1:
        axes = [axes]

    for ax, mod in zip(axes, modalities):
        sub_df = frame[frame["modality"] == mod].copy()
        sub_df["beta"] = pd.to_numeric(sub_df["beta"], errors="coerce")
        sub_df = sub_df.dropna(subset=["beta"]).sort_values("beta", ascending=True)
        if sub_df.empty:
            continue
        y_pos = np.arange(len(sub_df))
        betas = sub_df["beta"].to_numpy()
        ci_low = pd.to_numeric(sub_df.get("ci_low", np.nan), errors="coerce").to_numpy()
        ci_high = pd.to_numeric(sub_df.get("ci_high", np.nan), errors="coerce").to_numpy()

        has_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
        if has_ci.any():
            xerr = [betas - ci_low, ci_high - betas]
            ax.errorbar(betas, y_pos, xerr=xerr, fmt="o", color="steelblue", ecolor="lightsteelblue", elinewidth=2, capsize=3)
        else:
            ax.plot(betas, y_pos, "o", color="steelblue")

        ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        mean_beta = float(betas.mean())
        ax.axvline(mean_beta, color="firebrick", linestyle="-", linewidth=1.5, label=f"Mean β = {mean_beta:.3f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_df["subject"])
        ax.set_xlabel("Boltzmann slope (β)")
        ax.set_title(f"{mod} (n={len(sub_df)})")
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    LOG.info("Saved group beta forest plot to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None,
                        help="Optional output path for group forest plot PNG; defaults to OUT.with_suffix('.png')")
    parser.add_argument("--input", type=Path, default=None,
                        help="Optional all_subject_summaries.csv; defaults to RESULTS_ROOT/all_subject_summaries.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    source = args.input or (args.results_root / "all_subject_summaries.csv")
    if not source.is_file():
        raise FileNotFoundError(f"Participant summary CSV not found: {source}")
    frame = pd.read_csv(source)
    required = {"subject", "modality", "beta"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input summary is missing columns: {sorted(missing)}")

    rows = []
    additional_metrics = [
        "beta_acoustic_only_baseline",
        "beta_audio_controlled_joint",
        "beta_potential_adjusted_for_persistence",
        "beta_persistence_adjusted_for_potential",
        "soft_beta",
        "soft_beta_potential_adjusted_for_persistence",
        "soft_mean_entropy_norm",
        "mean_neural_incremental_information_nats",
        "mean_entropy_norm",
    ]
    for modality, group in frame.groupby("modality", sort=True):
        row = {"modality": modality}
        row.update(summarise_group(group))
        for column in additional_metrics:
            if column in group.columns:
                for statistic, value in summarise_numeric_column(group[column]).items():
                    row[f"{column}_{statistic}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    LOG.info("Wrote participant-level group summary to %s", args.out)

    plot_path = args.plot if args.plot is not None else args.out.with_suffix(".png")
    try:
        save_forest_plot(frame, plot_path)
    except Exception as exc:
        LOG.warning("Could not generate forest plot: %s", exc)

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
