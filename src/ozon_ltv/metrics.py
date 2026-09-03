"""Metrics aligned with the competition's RMSLE objective."""

from __future__ import annotations

import numpy as np


def rmsle_from_log(y_true: np.ndarray, pred_log: np.ndarray) -> float:
    """Compute RMSLE when the model already predicts ``log1p(target)``."""
    y_log = np.log1p(np.maximum(np.asarray(y_true, dtype=np.float64), 0.0))
    prediction = np.maximum(np.asarray(pred_log, dtype=np.float64), 0.0)
    if y_log.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: {y_log.shape} != {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions contain NaN or infinity")
    return float(np.sqrt(np.mean(np.square(y_log - prediction))))


def score_segments(y_true: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Report overall, zero-target and positive-target RMSLE."""
    target = np.asarray(y_true)
    positive = target > 0
    result = {
        "rmsle": rmsle_from_log(target, pred_log),
        "positive_share": float(positive.mean()),
    }
    result["rmsle_zero"] = (
        rmsle_from_log(target[~positive], np.asarray(pred_log)[~positive])
        if (~positive).any()
        else float("nan")
    )
    result["rmsle_positive"] = (
        rmsle_from_log(target[positive], np.asarray(pred_log)[positive])
        if positive.any()
        else float("nan")
    )
    return result
