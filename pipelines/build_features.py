"""Generate compressed point-in-time anchor tables from the immutable source."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ozon_ltv.config import ANCHORS, BATCH_SIZE, TEST_ANCHOR, ProjectPaths
from ozon_ltv.features import build_anchor_frame, validate_anchor_frame


def cache_paths(cache_root: Path, anchor) -> list[Path]:
    folder = cache_root / f"anchor_{anchor:%Y%m%d}"
    return [folder / f"batch_{index:04d}.parquet" for index in range(5)]


def cache_is_complete(cache_root: Path, anchor) -> bool:
    paths = cache_paths(cache_root, anchor)
    if not all(path.exists() for path in paths):
        return False
    rows = sum(
        pl.scan_parquet(path).select(pl.len()).collect().item() for path in paths
    )
    return rows == 250_000


def write_cache(frame: pl.DataFrame, cache_root: Path, anchor) -> None:
    folder = cache_root / f"anchor_{anchor:%Y%m%d}"
    folder.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, frame.height, BATCH_SIZE)):
        frame.slice(start, BATCH_SIZE).write_parquet(
            folder / f"batch_{index:04d}.parquet",
            compression="zstd",
            statistics=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    paths = ProjectPaths.from_root(arguments.project_root)

    users = (
        pl.read_csv(paths.sample_submission)
        .with_row_index("sample_order")
        .select("sample_order", "user_id")
    )
    events = pl.read_parquet(paths.train).sort(["user_id", "event_date"])
    if users.height != 250_000 or users["user_id"].n_unique() != users.height:
        raise AssertionError("Unexpected sample submission identity")

    started = time.perf_counter()
    for anchor in (*ANCHORS, TEST_ANCHOR):
        if not arguments.force and cache_is_complete(paths.feature_cache, anchor):
            print(f"{anchor}: reuse")
            continue
        target_available = anchor in ANCHORS
        frame = build_anchor_frame(
            events, users, anchor, target_available=target_available
        )
        validate_anchor_frame(frame, users, target_available=target_available)
        write_cache(frame, paths.feature_cache, anchor)
        print(
            f"{anchor}: {frame.height:,} rows, {len(frame.columns)} columns, "
            f"{(time.perf_counter() - started) / 60:.1f} min"
        )
        del frame
        gc.collect()


if __name__ == "__main__":
    main()
