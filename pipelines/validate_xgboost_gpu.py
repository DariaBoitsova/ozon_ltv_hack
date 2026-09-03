"""Run the all-eligible-anchor XGBoost CUDA diversity experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ozon_ltv.artifacts import sha256
from ozon_ltv.config import VALIDATION_ANCHORS, ProjectPaths
from ozon_ltv.datasets import eligible_train_anchors, feature_columns, load_anchors
from ozon_ltv.metrics import score_segments
from ozon_ltv.models import xgboost_gpu_parameters


def main() -> None:
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "artifacts" / "xgb_oof")
    arguments = parser.parse_args()
    paths = ProjectPaths.from_root(arguments.project_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    features = feature_columns(paths.feature_cache)
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []

    for validation_anchor in VALIDATION_ANCHORS:
        stem = f"xgb_{validation_anchor:%Y%m%d}"
        prediction_path = output / f"{stem}.parquet"
        metric_path = output / f"{stem}.json"
        if prediction_path.exists() and metric_path.exists():
            frame = pd.read_parquet(prediction_path)
            row = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            train_anchors = eligible_train_anchors(validation_anchor)
            train_x, train_y, _ = load_anchors(
                paths.feature_cache, train_anchors, selected_features=features
            )
            validation_x, validation_y, metadata = load_anchors(
                paths.feature_cache, [validation_anchor], selected_features=features
            )
            model = xgb.XGBRegressor(
                **xgboost_gpu_parameters(), early_stopping_rounds=200
            )
            started = time.perf_counter()
            model.fit(
                train_x,
                np.log1p(train_y),
                eval_set=[(validation_x, np.log1p(validation_y))],
                verbose=100,
            )
            prediction = np.maximum(0.0, model.predict(validation_x))
            row = {
                "validation_anchor": str(validation_anchor),
                "train_anchors": [str(anchor) for anchor in train_anchors],
                "best_iteration": int(model.best_iteration),
                "runtime_seconds": time.perf_counter() - started,
                **score_segments(validation_y, prediction),
            }
            frame = metadata.copy()
            frame["target"] = validation_y.astype(np.float32)
            frame["pred_xgb_log"] = prediction.astype(np.float32)
            temporary_prediction = prediction_path.with_suffix(".tmp.parquet")
            temporary_metric = metric_path.with_suffix(".tmp.json")
            frame.to_parquet(temporary_prediction, index=False)
            temporary_metric.write_text(json.dumps(row, indent=2), encoding="utf-8")
            temporary_prediction.replace(prediction_path)
            temporary_metric.replace(metric_path)
            model.save_model(output / f"{stem}.model.json")
            del model, train_x, train_y, validation_x, validation_y
            gc.collect()
        frames.append(frame)
        rows.append(row)
        print(json.dumps(row))

    oof_path = output / "oof_xgboost.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(oof_path, index=False)
    scores = np.asarray([float(row["rmsle"]) for row in rows])
    summary = {
        "versions": {"python": sys.version, "xgboost": xgb.__version__},
        "parameters": xgboost_gpu_parameters(),
        "feature_count": len(features),
        "folds": rows,
        "mean_rmsle": float(scores.mean()),
        "std_rmsle": float(scores.std()),
        "worst_rmsle": float(scores.max()),
        "suggested_final_iterations": int(
            np.median([int(row["best_iteration"]) for row in rows]) * 1.10
        ),
        "oof_sha256": sha256(oof_path),
    }
    (output / "metrics_xgboost.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
