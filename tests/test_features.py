from datetime import date, timedelta

import numpy as np
import polars as pl

from ozon_ltv.config import TARGET_COLUMN
from ozon_ltv.features import build_anchor_frame, validate_anchor_frame


def event_row(event_date: date, gmv: float) -> dict[str, int | float | date]:
    purchased = int(gmv > 0)
    return {
        "event_date": event_date,
        "user_id": 1,
        "search": 1,
        "cat": 0,
        "has_search_to_cart": purchased,
        "has_search_to_ord": purchased,
        "has_cat_to_cart": 0,
        "has_cat_to_ord": 0,
        "search_to_cart": purchased,
        "search_to_ord": purchased,
        "cat_to_cart": 0,
        "cat_to_ord": 0,
        "gmv_search": gmv,
        "gmv_cat": 0.0,
        "to_cart": purchased,
        "to_ord": purchased,
        "gmv": gmv,
        "searches": 1,
    }


def test_anchor_boundary_and_cold_user() -> None:
    anchor = date(2025, 7, 2)
    events = pl.DataFrame(
        [
            event_row(anchor - timedelta(days=1), 1.0),
            event_row(anchor, 2.0),
            event_row(anchor + timedelta(days=1), 3.0),
            event_row(anchor + timedelta(days=30), 4.0),
            event_row(anchor + timedelta(days=31), 5.0),
        ]
    ).with_columns(pl.col("event_date").cast(pl.Date))
    users = pl.DataFrame({"sample_order": [0, 1], "user_id": [1, 2]})

    frame = build_anchor_frame(events, users, anchor, target_available=True)
    validate_anchor_frame(frame, users, target_available=True)
    first = frame.filter(pl.col("user_id") == 1).row(0, named=True)
    cold = frame.filter(pl.col("user_id") == 2).row(0, named=True)

    assert np.isclose(first["gmv_sum_7d"], 3.0)
    assert np.isclose(first[TARGET_COLUMN], 7.0)
    assert first["recency_activity"] == 0
    assert cold["gmv_sum_30d"] == 0
    assert cold["recency_activity"] == 9999


def test_ratios_use_additive_one_smoothing() -> None:
    anchor = date(2025, 7, 2)
    events = pl.DataFrame([event_row(anchor, 2.0)]).with_columns(
        pl.col("event_date").cast(pl.Date)
    )
    users = pl.DataFrame({"sample_order": [0], "user_id": [1]})
    frame = build_anchor_frame(events, users, anchor, target_available=True)
    row = frame.row(0, named=True)
    assert np.isclose(row["gmv_per_order_7d"], 1.0)
    assert np.isclose(row["gmv_search_share_7d"], 2.0 / 3.0)
