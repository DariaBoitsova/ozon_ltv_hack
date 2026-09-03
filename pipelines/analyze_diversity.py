"""Join keyed OOF predictions and apply the conservative diversity gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ozon_ltv.config import KEY_COLUMNS
from ozon_ltv.ensemble import diversity_gate, search_two_model_blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-oof", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--control-column", default="pred_lgbm_log")
    parser.add_argument("--candidate-column", default="pred_xgb_log")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    control = pd.read_parquet(arguments.control_oof)
    candidate = pd.read_parquet(arguments.candidate_oof)
    keys = list(KEY_COLUMNS)
    if control[keys].duplicated().any() or candidate[keys].duplicated().any():
        raise AssertionError("OOF keys must be unique")
    frame = control.merge(
        candidate[keys + ["target", arguments.candidate_column]],
        on=keys,
        how="inner",
        suffixes=("", "_candidate"),
        validate="one_to_one",
    )
    if len(frame) != len(control) or not np.allclose(
        frame["target"], frame["target_candidate"]
    ):
        raise AssertionError("Candidate OOF rows or targets differ from control")
    result = search_two_model_blend(
        frame,
        target_column="target",
        control_column=arguments.control_column,
        candidate_column=arguments.candidate_column,
    )
    result["accepted"] = diversity_gate(result)
    result["control_oof"] = str(arguments.control_oof.resolve())
    result["candidate_oof"] = str(arguments.candidate_oof.resolve())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
