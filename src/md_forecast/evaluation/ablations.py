"""Frozen input sets and paired common-target effects for controlled ablations."""

from typing import Annotated, Self

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.schemas import BoundaryModel, Identifier
from md_forecast.data.splits import SplitConfig
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    group_interval,
    interval_status,
)
from md_forecast.models.chronos import ChronosConfig


class AblationConfig(BoundaryModel):
    """Predeclare input sets, common scored target and complete comparison family."""

    target: Identifier
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    split: SplitConfig
    grid: WindowConfig
    evaluation: BenchmarkConfig
    chronos: ChronosConfig
    variants: dict[Identifier, tuple[Identifier, ...]]

    @model_validator(mode="after")
    def validate_variants(self) -> Self:
        """Require a target-only reference and comparisons differing only in inputs."""
        _validate_cell(self.grid)
        selections = tuple(self.variants.values())
        _validate_reference(selections, self.target)
        _validate_selections(selections, self.target)
        return self


def _validate_cell(grid: WindowConfig) -> None:
    if (len(grid.contexts), len(grid.horizons)) != (1, 1):
        raise ValueError("this experiment requires one frozen C/H cell")


def _validate_reference(selections: tuple[tuple[str, ...], ...], target: str) -> None:
    if len(selections) < 2 or selections[0] != (target,):
        raise ValueError("first variant must be the common target alone")


def _validate_selections(selections: tuple[tuple[str, ...], ...], target: str) -> None:
    if len(set(selections)) != len(selections):
        raise ValueError("duplicate input sets")
    for features in selections:
        if target not in features or len(set(features)) != len(features):
            raise ValueError("each input set requires unique channels and target")


def target_rows(table: pa.Table, target: str, model_hash: str) -> pa.Table:
    """Select and order exact future labels so mismatched comparisons fail."""
    mask = np.asarray(table["feature_id"].to_numpy() == target) & np.asarray(
        table["model_hash"].to_numpy() == model_hash
    )
    return (
        table.filter(pa.array(mask))
        .select(
            [
                "trajectory_id",
                "system_id",
                "group_id",
                "start",
                "step",
                "target_frame",
                "target",
            ]
        )
        .sort_by([(key, "ascending") for key in ("trajectory_id", "start", "step")])
    )


def paired_effects(
    reference: pa.Table,
    candidate: pa.Table,
    target: str,
    model_hash: str,
    policy: BenchmarkConfig,
) -> list[dict[str, object]]:
    """Pair group MAE per lead; resample complexes rather than overlapping windows."""
    base = _group_mae(reference, target, model_hash)
    variant = _group_mae(candidate, target, model_hash)
    if base.keys() != variant.keys():
        raise DataContractError("ablation groups/lead steps differ")
    return [
        _step_effect(step, base, variant, target, policy)
        for step in sorted({key[0] for key in base})
    ]


def _step_effect(
    step: int,
    base: dict[tuple[int, str], float],
    variant: dict[tuple[int, str], float],
    target: str,
    policy: BenchmarkConfig,
) -> dict[str, object]:
    keys = sorted(key for key in base if key[0] == step)
    baseline = np.asarray([base[key] for key in keys], dtype=np.float64)
    candidate = np.asarray([variant[key] for key in keys], dtype=np.float64)
    delta = candidate - baseline
    lower, upper = group_interval(delta, policy, corrected=True)
    marginal = group_interval(delta, policy)
    return dict(
        step=step,
        target=target,
        groups=len(keys),
        reference_mae=float(baseline.mean()),
        variant_mae=float(candidate.mean()),
        mae_difference=float(delta.mean()),
        ci_lower=lower,
        ci_upper=upper,
        marginal_ci_lower=marginal[0],
        marginal_ci_upper=marginal[1],
        interval_status=interval_status(len(keys), policy, corrected=True),
    )


def _group_mae(
    table: pa.Table, target: str, model_hash: str
) -> dict[tuple[int, str], float]:
    rows = [
        r
        for r in table.to_pylist()
        if (r["level"], r["metric"], r["feature_id"], r["model_hash"])
        == ("group", "mae", target, model_hash)
    ]
    result = {(r["step"], r["identity"]): r["value"] for r in rows}
    if len(result) != max(1, len(rows)):
        raise DataContractError("missing or duplicate group MAE rows")
    return result
