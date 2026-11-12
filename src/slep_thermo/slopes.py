"""Within-window Boltzmann-slope estimation with window-bootstrap confidence intervals."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def estimate_joint_state_association(
    metrics: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Regress within-window self-information on potential and persistence jointly."""
    result = {
        "beta_potential_adjusted_for_persistence": np.nan,
        "beta_potential_ci_low": np.nan,
        "beta_potential_ci_high": np.nan,
        "beta_persistence_adjusted_for_potential": np.nan,
        "beta_persistence_ci_low": np.nan,
        "beta_persistence_ci_high": np.nan,
        "n_joint_state_cells": int(len(metrics)),
        "n_joint_state_windows": int(metrics["window"].nunique()) if not metrics.empty else 0,
    }
    required = {"window", "n_samples", "self_information", "semantic_potential_nll", "p_kk"}
    if metrics.empty or not required.issubset(metrics.columns):
        return result

    def fit(frame: pd.DataFrame) -> np.ndarray:
        x_rows, y_rows, weight_rows = [], [], []
        group_columns = ["fold", "window"] if "fold" in frame.columns else ["window"]
        for _, group in frame.groupby(group_columns):
            group = group.dropna(subset=["self_information", "semantic_potential_nll", "p_kk"])
            if len(group) < 2:
                continue
            weights = group["n_samples"].to_numpy(dtype=float)
            predictors = group[["semantic_potential_nll", "p_kk"]].to_numpy(dtype=float)
            outcome = group["self_information"].to_numpy(dtype=float)
            predictors -= np.average(predictors, axis=0, weights=weights)
            outcome -= np.average(outcome, weights=weights)
            x_rows.extend(predictors)
            y_rows.extend(outcome)
            weight_rows.extend(weights)
        x = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        weights = np.asarray(weight_rows, dtype=float)
        if len(y) < 3 or np.linalg.matrix_rank(x) < 2:
            return np.array([np.nan, np.nan])
        sqrt_weights = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(x * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
        return coefficients

    coefficients = fit(metrics)
    result["beta_potential_adjusted_for_persistence"] = float(coefficients[0])
    result["beta_persistence_adjusted_for_potential"] = float(coefficients[1])
    windows = metrics["window"].unique()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(windows, size=len(windows), replace=True)
        pieces = []
        for new_id, old_id in enumerate(sampled):
            piece = metrics.loc[metrics["window"] == old_id].copy()
            piece["window"] = new_id
            pieces.append(piece)
        coefficients = fit(pd.concat(pieces, ignore_index=True))
        if np.isfinite(coefficients).all():
            boot.append(coefficients)
    if boot:
        intervals = np.percentile(np.asarray(boot), [2.5, 97.5], axis=0)
        result["beta_potential_ci_low"] = float(intervals[0, 0])
        result["beta_potential_ci_high"] = float(intervals[1, 0])
        result["beta_persistence_ci_low"] = float(intervals[0, 1])
        result["beta_persistence_ci_high"] = float(intervals[1, 1])
    return result


def estimate_within_window_slope(
    metrics: pd.DataFrame,
    n_boot: int,
    seed: int,
    potential_col: str = "semantic_potential_nll",
) -> Dict[str, float]:
    """Weighted within-window slope; bootstrap resamples whole narrative windows."""
    empty = {"beta": np.nan, "temperature": np.nan, "r2_within": np.nan,
             "ci_low": np.nan, "ci_high": np.nan, "n_cells": int(len(metrics)),
             "n_windows": int(metrics["window"].nunique()) if not metrics.empty else 0}
    if metrics.empty or metrics["window"].nunique() < 3:
        return empty

    def fit(frame: pd.DataFrame) -> Tuple[float, float]:
        xs, ys, ws = [], [], []
        group_columns = ["fold", "window"] if "fold" in frame.columns else ["window"]
        for _, group in frame.groupby(group_columns):
            if len(group) < 2:
                continue
            weight = group["n_samples"].to_numpy(dtype=float)
            x = group[potential_col].to_numpy(dtype=float)
            y = group["self_information"].to_numpy(dtype=float)
            xbar, ybar = np.average(x, weights=weight), np.average(y, weights=weight)
            xs.extend(x - xbar)
            ys.extend(y - ybar)
            ws.extend(weight)
        x, y, w = np.asarray(xs), np.asarray(ys), np.asarray(ws)
        denom = np.sum(w * x * x)
        if len(x) < 3 or denom <= 1e-12:
            return np.nan, np.nan
        beta = float(np.sum(w * x * y) / denom)
        y_denom = np.sum(w * y * y)
        r2 = float(1 - np.sum(w * (y - beta * x) ** 2) / y_denom) if y_denom > 1e-12 else np.nan
        return beta, r2

    beta, r2 = fit(metrics)
    windows = metrics["window"].unique()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(windows, size=len(windows), replace=True)
        pieces = []
        for new_id, old_id in enumerate(sampled):
            piece = metrics.loc[metrics["window"] == old_id].copy()
            piece["window"] = new_id
            pieces.append(piece)
        b, _ = fit(pd.concat(pieces, ignore_index=True))
        if np.isfinite(b):
            boot.append(b)
    ci = np.percentile(boot, [2.5, 97.5]) if boot else [np.nan, np.nan]
    return {
        "beta": beta,
        "temperature": 1.0 / beta if np.isfinite(beta) and beta > 0 else np.nan,
        "r2_within": r2,
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "n_cells": int(len(metrics)), "n_windows": int(len(windows)),
    }
