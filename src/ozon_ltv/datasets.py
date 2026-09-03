"""Reading the compressed anchor cache and enforcing temporal eligibility."""

from __future__ import annotations

import gc
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from .config import (
    ANCHORS,
    CHANNEL_WINDOWS,
    KEY_COLUMNS,
    METRICS,
    TARGET_COLUMN,
    WINDOWS,
)


def anchor_directory(cache_root: str | Path, anchor: date) -> Path:
    return Path(cache_root) / f"anchor_{anchor:%Y%m%d}"


def anchor_paths(cache_root: str | Path, anchor: date) -> list[Path]:
    """Return the five expected compressed batches for an anchor."""
    paths = sorted(anchor_directory(cache_root, anchor).glob("batch_*.parquet"))
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected five batches for {anchor}, found {len(paths)}")
    return paths


def keep_compact_feature(name: str) -> bool:
    """Drop monotonic or exact raw duplicates while retaining 168 features."""
    raw_main = any(
        name == f"{metric}_sum_{days}d" for metric in METRICS for days in WINDOWS
    )
    raw_all = any(name == f"{metric}_sum_all" for metric in METRICS)
    channels = (
        "gmv_search",
        "gmv_cat",
        "search_to_ord",
        "cat_to_ord",
        "search_to_cart",
        "cat_to_cart",
    )
    raw_channel = any(
        name == f"{metric}_sum_{days}d"
        for metric in channels
        for days in CHANNEL_WINDOWS
    )
    return not (
        raw_main
        or raw_all
        or raw_channel
        or "_prior_" in name
        or "_trend_diff_" in name
        or "_trend_ratio_" in name
    )


def feature_columns(cache_root: str | Path) -> list[str]:
    """Discover and validate the model feature view from cache schema."""
    schema = pl.scan_parquet(anchor_paths(cache_root, ANCHORS[0])[0]).collect_schema()
    all_features = [
        column for column in schema if column not in (*KEY_COLUMNS, TARGET_COLUMN)
    ]
    selected = [column for column in all_features if keep_compact_feature(column)]
    if len(all_features) != 256 or len(selected) != 168:
        raise AssertionError(
            f"Feature contract changed: {len(all_features)} raw, {len(selected)} selected"
        )
    return selected


def load_anchors(
    cache_root: str | Path,
    anchors: Sequence[date],
    *,
    selected_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load ordered float32 features, target and keyed metadata."""
    features = list(selected_features or feature_columns(cache_root))
    paths = [
        str(path)
        for anchor in anchors
        for path in anchor_paths(cache_root, anchor)
    ]
    selected = [*KEY_COLUMNS, TARGET_COLUMN, *features]
    data = (
        pl.scan_parquet(paths)
        .select(selected)
        .collect(engine="streaming")
        .sort(["anchor_date", "sample_order"])
    )
    metadata = data.select(KEY_COLUMNS).to_pandas()
    target = data[TARGET_COLUMN].to_numpy().astype(np.float32, copy=False)
    matrix = data.select(features).to_pandas(use_pyarrow_extension_array=False)
    del data
    gc.collect()
    return matrix, target, metadata


def eligible_train_anchors(validation_anchor: date) -> list[date]:
    """Select anchors whose 30-day target ends before validation history."""
    eligible = [
        anchor
        for anchor in ANCHORS
        if anchor + timedelta(days=30) <= validation_anchor
    ]
    if not eligible:
        raise ValueError(f"No leakage-safe train anchors for {validation_anchor}")
    return eligible
