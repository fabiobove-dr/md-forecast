"""Verify diagnostic artifacts and summarize equal-trajectory, equal-system errors."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from run_followup_diagnostics import DiagnosticConfig

from md_forecast.core.constants import ModelId
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata
from md_forecast.features.structural import BASE_FEATURE_IDS

KEYS = ["model_hash", "feature_id", "context_frames", "horizon_frames"]


def error_summary(table: pa.Table, *, leads: bool) -> list[dict[str, object]]:
    """RMSE takes its root after each trajectory's mean squared error."""
    keys = KEYS + (["lead"] if leads else [])
    trajectories = table.group_by(keys + ["group_id", "trajectory_id"]).aggregate(
        [
            ("lead_absolute_error", "mean"),
            ("lead_squared_error", "mean"),
            ("lead_bias", "mean"),
        ]
    )
    trajectories = trajectories.set_column(
        trajectories.schema.get_field_index("lead_squared_error_mean"),
        "rmse",
        pc.sqrt(trajectories["lead_squared_error_mean"]),
    )
    metrics = ["lead_absolute_error_mean", "rmse", "lead_bias_mean"]
    groups = trajectories.group_by(keys + ["group_id"]).aggregate(
        [(key, "mean") for key in metrics]
    )
    averages = groups.group_by(keys).aggregate(
        [(key + "_mean", "mean") for key in metrics]
    )
    result = []
    for row in averages.to_pylist():
        result.append(
            {key: row[key] for key in keys}
            | {
                "lead": row.get("lead", 0),
                "mae": row["lead_absolute_error_mean_mean_mean"],
                "rmse": row["rmse_mean_mean"],
                "bias": row["lead_bias_mean_mean_mean"],
            }
        )
    return result


def spread_summary(table: pa.Table) -> list[dict[str, object]]:
    """Undefined constant-future ratios stay missing; retain their explicit count."""
    metrics = [
        "predicted_std",
        "actual_std",
        "spread_ratio",
        "predicted_change_from_origin",
        "actual_change_from_origin",
    ]
    trajectories = table.group_by(KEYS + ["group_id", "trajectory_id"]).aggregate(
        [(key, "mean") for key in metrics]
    )
    groups = trajectories.group_by(KEYS + ["group_id"]).aggregate(
        [(key + "_mean", "mean") for key in metrics]
    )
    average = groups.group_by(KEYS).aggregate(
        [(key + "_mean_mean", "mean") for key in metrics]
    )
    counts = table.group_by(KEYS).aggregate(
        [("spread_ratio", "count"), ("actual_std", "count")]
    )
    count_map = {tuple(row[key] for key in KEYS): row for row in counts.to_pylist()}
    result = []
    for row in average.to_pylist():
        count = count_map[tuple(row[key] for key in KEYS)]
        result.append(
            {key: row[key] for key in KEYS}
            | {key: row[key + "_mean_mean_mean"] for key in metrics}
            | {
                "windows": count["actual_std_count"],
                "undefined_spread_ratios": count["actual_std_count"]
                - count["spread_ratio_count"],
            }
        )
    return result


def summarize(root: Path, output: Path) -> None:
    """Select AR/VAR on native VAL and transfer unchanged to seen VAL."""
    provenance = json.loads((root / "provenance.json").read_text())
    for name, checksum in provenance["file_sha256"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
        ):
            raise DataContractError("diagnostic artifact checksum/path differs")
    config = read_metadata(root / "config.json", DiagnosticConfig)
    if metadata_hash(config) != provenance["config_hash"]:
        raise DataContractError("diagnostic config differs from provenance")
    models = {metadata_hash(c): c for c in config.candidates}
    summaries = {}
    selected = {}
    for task in ("native", "seen"):
        run = json.loads((root / task / "run.json").read_text())
        cells: dict[str, dict[str, object]] = {}
        for name in run["files"]:
            if not name.endswith("-windows.parquet"):
                continue
            cell = name.removesuffix("-windows.parquet")
            windows = pq.read_table(root / task / name)
            leads = pq.read_table(root / task / (cell + "-leads.parquet"))
            point = error_summary(leads, leads=False)
            cells[cell] = {
                "errors": point,
                "per_lead": error_summary(leads, leads=True),
                "spread": spread_summary(windows),
            }
            if task == "native":
                scale = dict(
                    zip(
                        BASE_FEATURE_IDS,
                        run["training_scalers"][cell]["scale"],
                        strict=True,
                    )
                )
                scores = {
                    key: float(
                        np.mean(
                            [
                                cast(float, row["mae"]) / scale[str(row["feature_id"])]
                                for row in point
                                if row["model_hash"] == key
                            ]
                        )
                    )
                    for key in models
                }
                selected[cell] = {
                    model: min(
                        (key for key, c in models.items() if c.model_id == model),
                        key=lambda key: (scores[key], key),
                    )
                    for model in (ModelId.AR, ModelId.VAR)
                }
                cells[cell]["normalized_validation_mae"] = scores
            cells[cell]["native_selected_regularized"] = selected[cell]
        summaries[task] = cells
    result = {
        "diagnostic_provenance": provenance,
        "selection": (
            "native VAL equal-system MAE divided by native TRAIN-context std; "
            "spread is descriptive only; AR/VAR transfer unchanged to seen VAL"
        ),
        "tasks": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"published {output}")


def main() -> None:
    """Read the immutable diagnostic bundle and write a separate local summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
