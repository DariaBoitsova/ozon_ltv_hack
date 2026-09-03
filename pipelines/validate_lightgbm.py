"""Run four leakage-safe frozen folds and save keyed OOF checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ozon_ltv.config import VALIDATION_ANCHORS, ProjectPaths
from ozon_ltv.datasets import eligible_train_anchors, feature_columns, load_anchors
from ozon_ltv.models import fit_lightgbm_fold, lightgbm_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "artifacts" / "lgbm_oof")
    arguments = parser.parse_args()
    paths = ProjectPaths.from_root(arguments.project_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    features = feature_columns(paths.feature_cache)
    fold_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []

    for validation_anchor in VALIDATION_ANCHORS:
        stem = f"fold_{validation_anchor:%Y%m%d}"
        prediction_path = output / f"{stem}.parquet"
        metric_path = output / f"{stem}.json"
        if prediction_path.exists() and metric_path.exists():
            frame = pd.read_parquet(prediction_path)
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            train_anchors = eligible_train_anchors(validation_anchor)
            train_x, train_y, _ = load_anchors(
                paths.feature_cache, train_anchors, selected_features=features
            )
            validation_x, validation_y, metadata = load_anchors(
                paths.feature_cache, [validation_anchor], selected_features=features
            )
            prediction, result = fit_lightgbm_fold(
                train_x,
                train_y,
                validation_x,
                validation_y,
                parameters=lightgbm_parameters(seed=42, n_jobs=arguments.threads),
            )
            frame = metadata.copy()
            frame["target"] = validation_y.astype(np.float32)
            frame["pred_lgbm_log"] = prediction.astype(np.float32)
            metric = {
                "validation_anchor": str(validation_anchor),
                "train_anchors": [str(anchor) for anchor in train_anchors],
                **result.to_dict(),
            }
            temporary_prediction = prediction_path.with_suffix(".tmp.parquet")
            temporary_metric = metric_path.with_suffix(".tmp.json")
            frame.to_parquet(temporary_prediction, index=False)
            temporary_metric.write_text(json.dumps(metric, indent=2), encoding="utf-8")
            temporary_prediction.replace(prediction_path)
            temporary_metric.replace(metric_path)
        if len(frame) != 250_000 or frame[["sample_order", "user_id", "anchor_date"]].duplicated().any():
            raise AssertionError(f"Invalid OOF checkpoint: {prediction_path}")
        fold_frames.append(frame)
        fold_metrics.append(metric)
        print(json.dumps(metric))

    oof = pd.concat(fold_frames, ignore_index=True)
    oof.to_parquet(output / "oof_lightgbm.parquet", index=False)
    scores = np.asarray([float(metric["rmsle"]) for metric in fold_metrics])
    summary = {
        "model": "LightGBM log-target champion family",
        "feature_count": len(features),
        "folds": fold_metrics,
        "mean_rmsle": float(scores.mean()),
        "std_rmsle": float(scores.std()),
        "worst_rmsle": float(scores.max()),
    }
    (output / "metrics_lightgbm.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
