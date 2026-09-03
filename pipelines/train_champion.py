"""Reproduce the immutable 965-tree LightGBM champion submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ozon_ltv.artifacts import write_submission
from ozon_ltv.config import ANCHORS, TEST_ANCHOR, ProjectPaths
from ozon_ltv.datasets import feature_columns, load_anchors
from ozon_ltv.models import fit_fixed_lightgbm, lightgbm_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=965)
    parser.add_argument("--output", default="ltv_portfolio_champion.csv")
    arguments = parser.parse_args()
    paths = ProjectPaths.from_root(arguments.project_root)

    features = feature_columns(paths.feature_cache)
    train_x, train_y, _ = load_anchors(
        paths.feature_cache, ANCHORS, selected_features=features
    )
    test_x, _, test_meta = load_anchors(
        paths.feature_cache, [TEST_ANCHOR], selected_features=features
    )
    parameters = lightgbm_parameters(
        seed=42,
        n_estimators=arguments.iterations,
        n_jobs=arguments.threads,
    )
    started = time.perf_counter()
    prediction = fit_fixed_lightgbm(
        train_x, train_y, test_x, parameters=parameters
    )
    order = np.argsort(test_meta["sample_order"].to_numpy())
    sample_ids = pl.read_csv(paths.sample_submission)["user_id"].to_numpy()
    ordered_ids = test_meta["user_id"].to_numpy()[order]
    if not np.array_equal(ordered_ids, sample_ids):
        raise AssertionError("Test cache and sample submission order differ")
    output = paths.submissions / arguments.output
    manifest = {
        "model": "LightGBM log-target champion",
        "feature_count": len(features),
        "anchor_count": len(ANCHORS),
        "parameters": parameters,
        "runtime_seconds": time.perf_counter() - started,
        "submission": write_submission(ordered_ids, prediction[order], output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
