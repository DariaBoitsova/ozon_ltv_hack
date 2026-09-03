"""Project-wide paths and temporal validation constants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


BATCH_SIZE = 50_000
WINDOWS = (7, 14, 30, 60, 90, 180)
CHANNEL_WINDOWS = (7, 30, 90, 180)
DECAY_HALF_LIVES = (7, 14, 30, 60)
METRICS = ("gmv", "to_ord", "to_cart", "searches")
ANCHORS = tuple(date(2025, 7, 2) + timedelta(days=14 * index) for index in range(15))
VALIDATION_ANCHORS = (
    date(2025, 12, 3),
    date(2025, 12, 17),
    date(2025, 12, 31),
    date(2026, 1, 14),
)
TEST_ANCHOR = date(2026, 2, 13)
TARGET_COLUMN = "target_30d"
KEY_COLUMNS = ("sample_order", "user_id", "anchor_date")


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve data and artifacts without machine-specific absolute paths."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def train(self) -> Path:
        return self.root / "data" / "train.parquet"

    @property
    def sample_submission(self) -> Path:
        return self.root / "data" / "sample_submit.csv"

    @property
    def feature_cache(self) -> Path:
        return self.root / "fe" / "clean_v1"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def results(self) -> Path:
        return self.root / "artifacts"


assert len(ANCHORS) == 15 and ANCHORS[-1] == VALIDATION_ANCHORS[-1]
