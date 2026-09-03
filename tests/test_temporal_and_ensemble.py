from datetime import date, timedelta

import numpy as np
import pandas as pd

from ozon_ltv.config import ANCHORS
from ozon_ltv.datasets import eligible_train_anchors, keep_compact_feature
from ozon_ltv.ensemble import diversity_gate, mean_log_predictions, search_two_model_blend


def test_training_targets_end_before_validation_history() -> None:
    validation = date(2025, 12, 3)
    anchors = eligible_train_anchors(validation)
    assert anchors
    assert all(anchor + timedelta(days=30) <= validation for anchor in anchors)
    assert set(anchors).issubset(ANCHORS)


def test_compact_view_drops_monotonic_duplicates() -> None:
    assert not keep_compact_feature("gmv_sum_30d")
    assert not keep_compact_feature("gmv_prior_7d")
    assert not keep_compact_feature("gmv_trend_ratio_7d")
    assert keep_compact_feature("gmv_log1p_30d")
    assert keep_compact_feature("recency_purchase")


def test_common_oof_blend_prefers_better_candidate() -> None:
    target = np.array([0.0, 1.0, 3.0, 7.0] * 2)
    truth = np.log1p(target)
    frame = pd.DataFrame(
        {
            "anchor_date": [date(2025, 1, 1)] * 4 + [date(2025, 2, 1)] * 4,
            "target": target,
            "control": truth + 0.2,
            "candidate": truth + 0.02,
        }
    )
    result = search_two_model_blend(
        frame,
        target_column="target",
        control_column="control",
        candidate_column="candidate",
        grid_size=101,
    )
    assert result["candidate_weight"] > 0.9
    assert result["standalone_gain"] > 0
    assert diversity_gate(result)


def test_seed_predictions_are_averaged_in_log_space() -> None:
    first = np.array([0.0, 1.0])
    second = np.array([2.0, 3.0])
    np.testing.assert_allclose(mean_log_predictions([first, second]), [1.0, 2.0])
