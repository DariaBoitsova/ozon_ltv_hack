"""Common-OOF model selection and conservative blending."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import rmsle_from_log


def search_two_model_blend(
    oof: pd.DataFrame,
    *,
    target_column: str,
    control_column: str,
    candidate_column: str,
    fold_column: str = "anchor_date",
    grid_size: int = 1001,
) -> dict[str, object]:
    """Fit one nonnegative log-space weight on a shared OOF matrix."""
    required = {target_column, control_column, candidate_column, fold_column}
    missing = required.difference(oof.columns)
    if missing:
        raise KeyError(sorted(missing))

    folds = list(oof.groupby(fold_column, sort=True))
    control_scores = [
        rmsle_from_log(group[target_column].to_numpy(), group[control_column].to_numpy())
        for _, group in folds
    ]
    candidate_scores = [
        rmsle_from_log(group[target_column].to_numpy(), group[candidate_column].to_numpy())
        for _, group in folds
    ]
    rows: list[tuple[float, float, list[float]]] = []
    for weight in np.linspace(0.0, 1.0, grid_size):
        scores = []
        for _, group in folds:
            prediction = (
                (1.0 - weight) * group[control_column].to_numpy()
                + weight * group[candidate_column].to_numpy()
            )
            scores.append(rmsle_from_log(group[target_column].to_numpy(), prediction))
        rows.append((float(np.mean(scores)), float(weight), scores))
    mean_score, candidate_weight, fold_scores = min(rows, key=lambda row: row[0])
    gains = [control - blend for control, blend in zip(control_scores, fold_scores)]
    residual_correlations = []
    for _, group in folds:
        truth = np.log1p(group[target_column].to_numpy())
        residual_correlations.append(
            float(
                np.corrcoef(
                    truth - group[control_column].to_numpy(),
                    truth - group[candidate_column].to_numpy(),
                )[0, 1]
            )
        )
    return {
        "candidate_weight": candidate_weight,
        "control_fold_scores": control_scores,
        "candidate_fold_scores": candidate_scores,
        "blend_fold_scores": fold_scores,
        "standalone_gain": float(np.mean(control_scores) - np.mean(candidate_scores)),
        "blend_gain": float(np.mean(control_scores) - mean_score),
        "worst_fold_gain": float(min(gains)),
        "residual_correlation": float(np.mean(residual_correlations)),
    }


def diversity_gate(
    result: dict[str, object],
    *,
    standalone_gain: float = 0.002,
    blend_gain: float = 0.003,
    max_residual_correlation: float = 0.995,
    max_fold_regression: float = 0.003,
) -> bool:
    """Apply the pre-registered standalone-or-diversity acceptance rule."""
    return bool(
        float(result["standalone_gain"]) >= standalone_gain
        or (
            float(result["blend_gain"]) >= blend_gain
            and float(result["residual_correlation"]) <= max_residual_correlation
            and float(result["worst_fold_gain"]) >= -max_fold_regression
        )
    )


def mean_log_predictions(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Average same-family predictions in metric-aligned log space."""
    if not columns:
        raise ValueError("At least one prediction vector is required")
    shapes = {np.asarray(column).shape for column in columns}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes differ: {shapes}")
    return np.mean(np.stack(columns, axis=0), axis=0)
