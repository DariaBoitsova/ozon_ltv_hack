import numpy as np
import pandas as pd
import pytest

from ozon_ltv.artifacts import validate_submission, write_submission
from ozon_ltv.metrics import rmsle_from_log, score_segments


def test_log_prediction_metric_is_exact() -> None:
    target = np.array([0.0, 3.0, 8.0])
    prediction = np.log1p(target)
    assert rmsle_from_log(target, prediction) == 0.0
    segments = score_segments(target, prediction)
    assert segments["rmsle_zero"] == 0.0
    assert segments["rmsle_positive"] == 0.0


def test_submission_contract_and_manifest(tmp_path) -> None:
    user_ids = np.array([10, 20, 30])
    output = tmp_path / "submission.csv"
    manifest = write_submission(user_ids, np.log1p([0.0, 2.0, 5.0]), output)
    frame = pd.read_csv(output)
    validate_submission(frame, user_ids)
    assert manifest["rows"] == 3
    assert len(manifest["sha256"]) == 64


def test_submission_rejects_changed_order() -> None:
    frame = pd.DataFrame({"user_id": [2, 1], "predict": [0.0, 1.0]})
    with pytest.raises(AssertionError):
        validate_submission(frame, np.array([1, 2]))
