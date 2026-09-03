"""Validation and reproducible writing of competition artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a potentially large file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(frame: pd.DataFrame, sample_ids: np.ndarray) -> None:
    """Enforce the exact two-column competition submission contract."""
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"Unexpected columns: {list(frame.columns)}")
    expected = np.asarray(sample_ids)
    if len(frame) != len(expected) or not frame["user_id"].is_unique:
        raise AssertionError("Submission rows or IDs are invalid")
    if not np.array_equal(frame["user_id"].to_numpy(), expected):
        raise AssertionError("Submission order differs from sample submission")
    prediction = frame["predict"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError("Predictions must be finite and nonnegative")


def write_submission(
    user_ids: np.ndarray,
    pred_log: np.ndarray,
    output_path: str | Path,
) -> dict[str, object]:
    """Convert log predictions, validate the output and return its manifest."""
    output = Path(output_path)
    frame = pd.DataFrame(
        {
            "user_id": np.asarray(user_ids),
            "predict": np.expm1(np.maximum(np.asarray(pred_log), 0.0)),
        }
    )
    validate_submission(frame, np.asarray(user_ids))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {
        "file": output.name,
        "rows": len(frame),
        "mean": float(frame["predict"].mean()),
        "median": float(frame["predict"].median()),
        "zero_share": float((frame["predict"] == 0).mean()),
        "maximum": float(frame["predict"].max()),
        "sha256": sha256(output),
    }
