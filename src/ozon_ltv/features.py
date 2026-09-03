"""Leakage-safe feature engineering for sparse customer activity histories."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from .config import (
    CHANNEL_WINDOWS,
    DECAY_HALF_LIVES,
    KEY_COLUMNS,
    METRICS,
    TARGET_COLUMN,
    WINDOWS,
)


def _window_mask(anchor: date, days: int) -> pl.Expr:
    return pl.col("event_date") >= pl.lit(
        anchor - timedelta(days=days - 1), dtype=pl.Date
    )


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    """Use additive-one smoothing and never mix raw and log totals."""
    return (pl.col(numerator) / (pl.col(denominator) + 1.0)).alias(alias)


def _window_expressions(anchor: date) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for days in WINDOWS:
        mask = _window_mask(anchor, days)
        for metric in METRICS:
            expressions.append(
                pl.when(mask)
                .then(pl.col(metric))
                .otherwise(0)
                .sum()
                .alias(f"{metric}_sum_{days}d")
            )
        expressions.extend(
            [
                pl.when(mask).then(1).otherwise(0).sum().alias(f"active_days_{days}d"),
                pl.when(mask & (pl.col("search") > 0))
                .then(1)
                .otherwise(0)
                .sum()
                .alias(f"search_days_{days}d"),
                pl.when(mask & (pl.col("to_cart") > 0))
                .then(1)
                .otherwise(0)
                .sum()
                .alias(f"cart_days_{days}d"),
                pl.when(mask & (pl.col("to_ord") > 0))
                .then(1)
                .otherwise(0)
                .sum()
                .alias(f"order_days_{days}d"),
                pl.when(mask & (pl.col("gmv") > 0))
                .then(1)
                .otherwise(0)
                .sum()
                .alias(f"purchase_days_{days}d"),
                pl.col("gmv")
                .filter(mask & (pl.col("gmv") > 0))
                .mean()
                .alias(f"gmv_pos_mean_{days}d"),
                pl.col("gmv")
                .filter(mask & (pl.col("gmv") > 0))
                .max()
                .alias(f"gmv_pos_max_{days}d"),
                pl.col("gmv")
                .filter(mask & (pl.col("gmv") > 0))
                .std()
                .alias(f"gmv_pos_std_{days}d"),
            ]
        )

    channels = (
        "gmv_search",
        "gmv_cat",
        "search_to_ord",
        "cat_to_ord",
        "search_to_cart",
        "cat_to_cart",
    )
    for days in CHANNEL_WINDOWS:
        mask = _window_mask(anchor, days)
        for metric in channels:
            expressions.append(
                pl.when(mask)
                .then(pl.col(metric))
                .otherwise(0)
                .sum()
                .alias(f"{metric}_sum_{days}d")
            )

    for days in (7, 14, 30):
        prior_mask = pl.col("event_date").is_between(
            anchor - timedelta(days=2 * days - 1),
            anchor - timedelta(days=days),
        )
        for metric in METRICS:
            expressions.append(
                pl.when(prior_mask)
                .then(pl.col(metric))
                .otherwise(0)
                .sum()
                .alias(f"{metric}_prior_{days}d")
            )
    return expressions


def _all_history_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = [
        pl.len().alias("active_days_all"),
        (pl.col("gmv") > 0).sum().alias("purchase_days_all"),
        pl.col("event_date").min().alias("first_event_date"),
        pl.col("event_date").max().alias("last_event_date"),
        pl.col("event_date").filter(pl.col("search") > 0).max().alias("last_search_date"),
        pl.col("event_date").filter(pl.col("to_cart") > 0).max().alias("last_cart_date"),
        pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias("last_order_date"),
        pl.col("event_date").filter(pl.col("gmv") > 0).max().alias("last_purchase_date"),
        pl.col("gmv").filter(pl.col("gmv") > 0).mean().alias("gmv_pos_mean_all"),
        pl.col("gmv").filter(pl.col("gmv") > 0).max().alias("gmv_pos_max_all"),
        pl.col("gmv").filter(pl.col("gmv") > 0).std().alias("gmv_pos_std_all"),
    ]
    expressions.extend(
        pl.col(metric).sum().alias(f"{metric}_sum_all") for metric in METRICS
    )
    return expressions


def _gap_features(history: pl.DataFrame) -> pl.DataFrame:
    purchase_days = (
        history.filter(pl.col("gmv") > 0)
        .select("user_id", "event_date")
        .sort(["user_id", "event_date"])
        .with_columns(
            pl.col("event_date")
            .diff()
            .over("user_id")
            .dt.total_days()
            .alias("purchase_gap")
        )
    )
    return purchase_days.group_by("user_id").agg(
        pl.col("purchase_gap").count().alias("purchase_gap_count_180d"),
        pl.col("purchase_gap").mean().alias("purchase_gap_mean_180d"),
        pl.col("purchase_gap").std().alias("purchase_gap_std_180d"),
        pl.col("purchase_gap").median().alias("purchase_gap_median_180d"),
        pl.col("purchase_gap").max().alias("purchase_gap_max_180d"),
        pl.col("purchase_gap").drop_nulls().last().alias("purchase_gap_last_180d"),
    )


def _decay_features(history: pl.DataFrame, anchor: date) -> pl.DataFrame:
    aged = history.with_columns(
        (pl.lit(anchor, dtype=pl.Date) - pl.col("event_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("age_days")
    )
    expressions: list[pl.Expr] = []
    for half_life in DECAY_HALF_LIVES:
        weight = (-math.log(2.0) * pl.col("age_days") / half_life).exp()
        expressions.extend(
            [
                weight.sum().alias(f"decay_occ_{half_life}d"),
                (weight * pl.col("gmv")).sum().alias(f"decay_gmv_{half_life}d"),
                (weight * (pl.col("to_ord") > 0).cast(pl.Float64))
                .sum()
                .alias(f"decay_order_occ_{half_life}d"),
            ]
        )
    return aged.group_by("user_id").agg(expressions)


def _global_features(events: pl.DataFrame, anchor: date) -> dict[str, float]:
    current = events.filter(
        pl.col("event_date").is_between(anchor - timedelta(days=29), anchor)
    )
    previous = events.filter(
        pl.col("event_date").is_between(
            anchor - timedelta(days=59), anchor - timedelta(days=30)
        )
    )
    result: dict[str, float] = {}
    for name, frame in (("current", current), ("previous", previous)):
        result[f"global_users_{name}_30d"] = float(frame["user_id"].n_unique())
        result[f"global_gmv_{name}_30d"] = float(frame["gmv"].sum())
        result[f"global_searches_{name}_30d"] = float(frame["searches"].sum())
    for metric in ("users", "gmv", "searches"):
        result[f"global_{metric}_ratio_30d"] = (
            result[f"global_{metric}_current_30d"] + 1.0
        ) / (result[f"global_{metric}_previous_30d"] + 1.0)
    return result


def build_anchor_frame(
    events: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    *,
    target_available: bool,
) -> pl.DataFrame:
    """Build one point-in-time-correct user snapshot and optional 30-day target."""
    history_all = events.filter(pl.col("event_date") <= anchor)
    history_180 = history_all.filter(
        pl.col("event_date") >= anchor - timedelta(days=179)
    )
    frame = users.with_columns(pl.lit(anchor, dtype=pl.Date).alias("anchor_date"))
    frame = (
        frame.join(
            history_180.group_by("user_id").agg(_window_expressions(anchor)),
            on="user_id",
            how="left",
        )
        .join(
            history_all.group_by("user_id").agg(_all_history_expressions()),
            on="user_id",
            how="left",
        )
        .join(_gap_features(history_180), on="user_id", how="left")
        .join(_decay_features(history_180, anchor), on="user_id", how="left")
    )

    date_columns = [
        column
        for column in frame.columns
        if column.endswith("_date") and column != "anchor_date"
    ]
    numeric_raw = [
        column for column in frame.columns if column not in (*KEY_COLUMNS, *date_columns)
    ]
    frame = frame.with_columns([pl.col(column).fill_null(0) for column in numeric_raw])
    recency = {
        "last_event_date": "recency_activity",
        "last_search_date": "recency_search",
        "last_cart_date": "recency_cart",
        "last_order_date": "recency_order",
        "last_purchase_date": "recency_purchase",
    }
    frame = frame.with_columns(
        [
            (pl.lit(anchor, dtype=pl.Date) - pl.col(source))
            .dt.total_days()
            .fill_null(9999)
            .alias(alias)
            for source, alias in recency.items()
        ]
        + [
            (pl.lit(anchor, dtype=pl.Date) - pl.col("first_event_date"))
            .dt.total_days()
            .fill_null(0)
            .alias("cohort_age_days"),
            (pl.col("last_event_date") - pl.col("first_event_date"))
            .dt.total_days()
            .fill_null(0)
            .alias("activity_span_days"),
            pl.col("first_event_date").dt.month().fill_null(0).alias("cohort_month"),
        ]
    ).drop(date_columns)

    derived: list[pl.Expr] = []
    for days in WINDOWS:
        for metric in METRICS:
            derived.append(
                pl.col(f"{metric}_sum_{days}d")
                .log1p()
                .alias(f"{metric}_log1p_{days}d")
            )
        derived.extend(
            [
                (1.0 - pl.col(f"active_days_{days}d") / float(days)).alias(
                    f"inactive_share_{days}d"
                ),
                _safe_ratio(
                    f"to_cart_sum_{days}d",
                    f"searches_sum_{days}d",
                    f"cart_per_search_{days}d",
                ),
                _safe_ratio(
                    f"to_ord_sum_{days}d",
                    f"to_cart_sum_{days}d",
                    f"orders_per_cart_{days}d",
                ),
                _safe_ratio(
                    f"gmv_sum_{days}d",
                    f"to_ord_sum_{days}d",
                    f"gmv_per_order_{days}d",
                ),
            ]
        )
    for metric in METRICS:
        derived.append(pl.col(f"{metric}_sum_all").log1p().alias(f"{metric}_log1p_all"))
    for days in CHANNEL_WINDOWS:
        derived.extend(
            [
                _safe_ratio(
                    f"gmv_search_sum_{days}d",
                    f"gmv_sum_{days}d",
                    f"gmv_search_share_{days}d",
                ),
                _safe_ratio(
                    f"search_to_ord_sum_{days}d",
                    f"to_ord_sum_{days}d",
                    f"order_search_share_{days}d",
                ),
                _safe_ratio(
                    f"search_to_cart_sum_{days}d",
                    f"to_cart_sum_{days}d",
                    f"cart_search_share_{days}d",
                ),
            ]
        )
    for days in (7, 14, 30):
        for metric in METRICS:
            current = f"{metric}_sum_{days}d"
            previous = f"{metric}_prior_{days}d"
            derived.extend(
                [
                    (pl.col(current) - pl.col(previous)).alias(
                        f"{metric}_trend_diff_{days}d"
                    ),
                    ((pl.col(current) + 1.0) / (pl.col(previous) + 1.0)).alias(
                        f"{metric}_trend_ratio_{days}d"
                    ),
                    (pl.col(current).log1p() - pl.col(previous).log1p()).alias(
                        f"{metric}_trend_logratio_{days}d"
                    ),
                ]
            )
    frame = frame.with_columns(derived)

    globals_ = _global_features(events, anchor)
    frame = frame.with_columns(
        [pl.lit(value).alias(name) for name, value in globals_.items()]
        + [
            pl.lit((anchor - date(2025, 1, 1)).days).alias("anchor_day_index"),
            pl.lit(anchor.month).alias("anchor_month"),
            pl.lit(math.sin(2 * math.pi * anchor.timetuple().tm_yday / 365.25)).alias(
                "anchor_doy_sin"
            ),
            pl.lit(math.cos(2 * math.pi * anchor.timetuple().tm_yday / 365.25)).alias(
                "anchor_doy_cos"
            ),
        ]
    )

    if target_available:
        target = (
            events.filter(
                pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=30)
                )
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias(TARGET_COLUMN))
        )
        frame = frame.join(target, on="user_id", how="left").with_columns(
            pl.col(TARGET_COLUMN).fill_null(0.0)
        )
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float32).alias(TARGET_COLUMN))

    feature_columns = [
        column for column in frame.columns if column not in (*KEY_COLUMNS, TARGET_COLUMN)
    ]
    frame = frame.with_columns(
        [
            pl.col(column).fill_nan(0).fill_null(0).cast(pl.Float32)
            for column in feature_columns
        ]
    )
    if target_available:
        frame = frame.with_columns(pl.col(TARGET_COLUMN).cast(pl.Float32))
    return frame.sort("sample_order")


def validate_anchor_frame(
    frame: pl.DataFrame,
    users: pl.DataFrame,
    *,
    target_available: bool,
) -> None:
    """Validate row identity, finite features and target constraints."""
    if frame.height != users.height or frame["user_id"].n_unique() != users.height:
        raise AssertionError("Anchor must contain one row per requested user")
    expected = users.sort("sample_order")["user_id"].to_list()
    if frame["user_id"].to_list() != expected:
        raise AssertionError("Anchor user order differs from sample order")
    feature_columns = [
        column for column in frame.columns if column not in (*KEY_COLUMNS, TARGET_COLUMN)
    ]
    if any(frame[column].null_count() for column in feature_columns):
        raise AssertionError("Feature matrix contains nulls")
    float_columns = [
        column
        for column, dtype in frame.schema.items()
        if dtype in (pl.Float32, pl.Float64) and column != TARGET_COLUMN
    ]
    infinite = frame.select(
        [pl.col(column).is_infinite().sum() for column in float_columns]
    ).sum_horizontal().item()
    if infinite:
        raise AssertionError("Feature matrix contains infinity")
    if target_available and (
        frame[TARGET_COLUMN].null_count()
        or frame.select((pl.col(TARGET_COLUMN) < 0).sum()).item()
    ):
        raise AssertionError("Target contains null or negative values")
