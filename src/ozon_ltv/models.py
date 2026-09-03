"""Model factories and metric-aligned training helpers."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .metrics import score_segments


@dataclass(frozen=True)
class FoldResult:
    """Serializable validation output for one temporal fold."""

    best_iteration: int
    runtime_seconds: float
    rmsle: float
    rmsle_zero: float
    rmsle_positive: float
    positive_share: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def lightgbm_parameters(
    *,
    seed: int = 42,
    n_estimators: int = 4000,
    n_jobs: int = 4,
    num_leaves: int = 31,
    min_child_samples: int = 2000,
) -> dict[str, object]:
    """Return the public-score champion parameterization."""
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.03,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": -1,
    }


def fit_lightgbm_fold(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
    *,
    parameters: dict[str, object],
    early_stopping_rounds: int = 200,
) -> tuple[np.ndarray, FoldResult]:
    """Fit one early-stopped log-target LightGBM fold."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**parameters)
    started = time.perf_counter()
    model.fit(
        train_features,
        np.log1p(train_target),
        eval_set=[(validation_features, np.log1p(validation_target))],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    prediction = np.maximum(
        0.0,
        model.predict(validation_features, num_iteration=model.best_iteration_),
    )
    scores = score_segments(validation_target, prediction)
    result = FoldResult(
        best_iteration=int(model.best_iteration_),
        runtime_seconds=time.perf_counter() - started,
        rmsle=scores["rmsle"],
        rmsle_zero=scores["rmsle_zero"],
        rmsle_positive=scores["rmsle_positive"],
        positive_share=scores["positive_share"],
    )
    del model
    gc.collect()
    return prediction, result


def fit_fixed_lightgbm(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    test_features: pd.DataFrame,
    *,
    parameters: dict[str, object],
) -> np.ndarray:
    """Fit a final log-target LightGBM and return clipped log predictions."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**parameters)
    model.fit(
        train_features,
        np.log1p(train_target),
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = np.maximum(0.0, model.predict(test_features))
    del model
    gc.collect()
    return prediction


def xgboost_gpu_parameters(
    *, seed: int = 42, n_estimators: int = 5000
) -> dict[str, object]:
    """Return the pre-registered CUDA diversity configuration."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "random_state": seed,
        "n_jobs": -1,
    }
