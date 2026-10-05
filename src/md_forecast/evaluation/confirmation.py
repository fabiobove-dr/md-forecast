"""Frozen confirmation provenance and corrected, independent-system decisions."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import Field, model_validator

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import read_metadata
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, DatasetConfig
from md_forecast.data.series import FloatArray
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    group_interval,
    interval_status,
)
from md_forecast.evaluation.overview import confined, file_hash
from md_forecast.models.chronos import ChronosConfig


class ConfirmationPlan(BoundaryModel):
    """Every choice fixed before confirmation outcomes, with portable source pins."""

    grid: WindowConfig
    primary: WindowConfig
    inference: BenchmarkConfig
    descriptive: BenchmarkConfig
    settings: ChronosConfig
    training_dataset: DatasetConfig
    native_cohort_hash: ArtifactHash
    external_reserve_hash: ArtifactHash
    seen_reserve_hash: ArtifactHash
    probability_config_hash: ArtifactHash
    input_conditions: dict[str, Literal["joint", "target-only"]]
    fine_checkpoints: dict[str, str]
    compact_states: dict[str, str]
    residual_states: dict[str, dict[str, str]]
    statistical_choices: dict[str, dict[str, ArtifactHash]]
    source_files: dict[str, ArtifactHash]
    expected_groups: dict[str, Annotated[int, Field(gt=0, strict=True)]]
    worthwhile_reduction: Annotated[float, Field(gt=0, lt=1)]
    nominal_coverage: Annotated[float, Field(ge=0.8, le=0.8)] = 0.8
    coverage_tolerance: Annotated[float, Field(gt=0, lt=0.2)]
    calibration_scope: Literal["TRAIN-residuals-only; no confirmation refitting"]
    confirmation_state: Literal["no reserved outcomes decoded"]
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]

    @model_validator(mode="after")
    def complete_choices(self) -> Self:
        """Reject missing feature/cell bindings or a changed primary policy."""
        features = set(self.input_conditions)
        if features != set(self.fine_checkpoints):
            raise ValueError(
                "fine-tuned checkpoints differ from declared input features"
            )
        _validate_reference_cells(self)
        self._validate_policy()
        return self

    def _validate_policy(self) -> None:
        if len(self.primary.contexts) != 1 or len(self.primary.horizons) != 1:
            raise ValueError("confirmation requires one predeclared primary cell")
        if self.primary.horizons[0] < 2:
            raise ValueError("primary confirmation horizon must be nontrivial")
        if self.inference.comparison_family_size != 126:
            raise ValueError(
                "confirmation must preserve the 126-interval primary family"
            )


def _validate_reference_cells(plan: ConfirmationPlan) -> None:
    cells = {
        f"c{int(c)}-h{int(h)}" for c in plan.grid.contexts for h in plan.grid.horizons
    }
    for reference in (
        plan.compact_states,
        plan.residual_states,
        plan.statistical_choices,
    ):
        if cells != set(reference):
            raise ValueError(
                "confirmation references must cover every frozen grid cell"
            )


def read_confirmation_plan(path: Path, root: Path = Path(".")) -> ConfirmationPlan:
    """Verify the immutable plan and every frozen input before reserved access."""
    plan = read_metadata(path, ConfirmationPlan)
    for relative, expected in plan.source_files.items():
        if file_hash(confined(root, relative)) != expected:
            raise DataContractError("frozen confirmation source changed: " + relative)
    return plan


def system_metrics(table: pa.Table) -> tuple[tuple[str, ...], dict[str, FloatArray]]:
    """Equal trajectory then equal complex weights; dependent windows stay together."""
    columns = _metric_columns(table)
    trajectory = table.group_by(["group_id", "trajectory_id"]).aggregate(
        [(name, "mean") for name in columns]
    )
    trajectory = trajectory.set_column(
        trajectory.schema.get_field_index("lead_squared_error_mean"),
        "lead_squared_error_mean",
        pc.sqrt(trajectory["lead_squared_error_mean"]),
    )
    group = trajectory.group_by(["group_id"]).aggregate(
        [(name + "_mean", "mean") for name in columns]
    )
    return _ordered_values(group, columns)


def _metric_columns(table: pa.Table) -> list[str]:
    return ["lead_absolute_error", "lead_squared_error", "lead_bias"] + [
        name
        for name in table.column_names
        if name.startswith(("pinball-", "coverage-", "width-"))
    ]


def _ordered_values(
    group: pa.Table, columns: list[str]
) -> tuple[tuple[str, ...], dict[str, FloatArray]]:
    rows = sorted(group.to_pylist(), key=lambda row: row["group_id"])
    names: dict[str, str] = {
        "lead_absolute_error": "mae",
        "lead_squared_error": "rmse",
        "lead_bias": "bias",
    }
    values = {
        names.get(name, name): np.asarray(
            [row[name + "_mean_mean"] for row in rows], dtype=np.float64
        )
        for name in columns
    }
    _add_mean_pinball(values)
    return tuple(row["group_id"] for row in rows), values


def _add_mean_pinball(values: dict[str, FloatArray]) -> None:
    pinball = [value for name, value in values.items() if name.startswith("pinball-")]
    if pinball:
        values["mean-pinball"] = np.mean(pinball, axis=0)


def paired_effect(
    left: tuple[tuple[str, ...], dict[str, FloatArray]],
    right: tuple[tuple[str, ...], dict[str, FloatArray]],
    metric: str,
    config: BenchmarkConfig,
    *,
    factor: float = 1.0,
    corrected: bool = True,
) -> dict[str, object]:
    """Paired model-minus-reference effects with explicit group identity matching."""
    if left[0] != right[0]:
        raise DataContractError("paired confirmation groups differ")
    values = left[1][metric] - factor * right[1][metric]
    return {
        "metric": metric,
        "reference_factor": factor,
        "difference": float(values.mean()),
        "ci": group_interval(values, config, corrected=corrected),
        "independent_groups": len(values),
        "interval_status": interval_status(len(values), config, corrected=corrected),
        "corrected": corrected,
    }


def upper_below(effect: dict[str, Any], boundary: float = 0.0) -> bool:
    """Missing or unresolved intervals cannot establish superiority."""
    upper = effect["ci"][1]
    return upper is not None and bool(upper < boundary)


def calibrated(effect: dict[str, Any], nominal: float, tolerance: float) -> bool:
    """Require the entire coverage interval inside the frozen acceptable range."""
    lower, upper = effect["ci"]
    return (
        lower is not None
        and upper is not None
        and bool(lower >= nominal - tolerance and upper <= nominal + tolerance)
    )


def confirmation_decision(
    effects: dict[str, Any], nominal: float, tolerance: float
) -> dict[str, bool]:
    """Evaluate every frozen condition; point gains alone never imply useful skill."""
    references = ("persistence", "selected-statistic")
    point = all(upper_below(effects["mae"][name]) for name in references)
    worthwhile = all(upper_below(effects["worthwhile"][name]) for name in references)
    probability = all(upper_below(effects["pinball"][name]) for name in references)
    coverage = calibrated(effects["coverage"], nominal, tolerance)
    width = _width_not_worse(effects["width"])
    useful = all((point, worthwhile, probability, coverage, width))
    return {
        "measurable_point_gain": point,
        "worthwhile_point_gain": worthwhile,
        "pinball_gain": probability,
        "calibrated": coverage,
        "width_not_worse": width,
        "useful_minimum": useful,
        "useful_stronger": useful and upper_below(effects["worthwhile"]["compact"]),
    }


def _width_not_worse(effect: dict[str, Any]) -> bool:
    upper = effect["ci"][1]
    return upper is not None and bool(upper <= 0)
