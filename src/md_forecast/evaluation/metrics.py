"""Native point/probabilistic losses and deterministic independent-group intervals."""

from enum import StrEnum
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.exceptions import ForecastError
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.series import FloatArray


class MetricId(StrEnum):
    """Closed benchmark vocabulary; persistence ratios are paired comparisons."""

    MAE = "mae"
    RMSE = "rmse"
    SCALED_MAE = "scaled-mae"
    PINBALL = "pinball"
    COVERAGE = "coverage"
    WIDTH = "width"


type MetricKey = tuple[MetricId, float | None, float | None]


class BenchmarkConfig(BoundaryModel):
    """Freeze group-level percentile CI policy and resource bounds before scoring."""

    seed: Annotated[int, Field(strict=True, ge=0)]
    bootstrap_samples: Annotated[int, Field(strict=True, ge=100)]
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    min_groups: Annotated[int, Field(strict=True, ge=2)] = 5
    min_tail_samples: Annotated[int, Field(strict=True, gt=0)] = 10
    bootstrap_unit: Literal["split-group"] = "split-group"
    correction: Literal["bonferroni"] = "bonferroni"
    comparison_family_size: Annotated[int, Field(strict=True, gt=0)]
    intervals: tuple[tuple[float, float], ...] = ((0.1, 0.9),)
    max_prediction_rows: Annotated[int, Field(strict=True, gt=0)]
    max_bootstrap_bytes: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def validate_intervals(self) -> Self:
        """Do not label invalid or repeated endpoints as calibrated intervals."""
        if len(set(self.intervals)) != len(self.intervals):
            raise ValueError("duplicate intervals")
        if any(not 0 < lower < upper < 1 for lower, upper in self.intervals):
            raise ValueError("interval endpoints must satisfy 0 < lower < upper < 1")
        return self


def point_losses(
    points: FloatArray,
    targets: FloatArray,
    scale: FloatArray,
) -> dict[MetricKey, FloatArray]:
    """Return elemental losses; RMSE requires square root after trajectory means."""
    _validate_point_inputs(points, targets, scale)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = points - targets
            result: dict[MetricKey, FloatArray] = {
                (MetricId.MAE, None, None): np.abs(error),
                (MetricId.RMSE, None, None): error**2,
                (MetricId.SCALED_MAE, None, None): np.abs(error) / scale,
            }
    except FloatingPointError as error:
        raise ForecastError("point metric arithmetic overflowed") from error
    return result


def _validate_point_inputs(
    points: FloatArray, targets: FloatArray, scale: FloatArray
) -> None:
    if points.shape != targets.shape or scale.shape != (points.shape[-1],):
        raise ForecastError("metric dimensions differ")
    _validate_scale(scale)
    if not np.isfinite(points).all() or not np.isfinite(targets).all():
        raise ForecastError("point metrics require finite inputs")


def _validate_scale(scale: FloatArray) -> None:
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ForecastError("metric scale must be finite, positive and frozen on TRAIN")


def quantile_losses(
    values: FloatArray,
    targets: FloatArray,
    levels: tuple[float, ...],
    intervals: tuple[tuple[float, float], ...],
) -> dict[MetricKey, FloatArray]:
    """Un-doubled pinball; inclusive interval coverage and nonnegative native width."""
    _validate_quantile_inputs(values, targets, levels)
    try:
        with np.errstate(over="raise", invalid="raise"):
            return _quantile_losses(values, targets, levels, intervals)
    except FloatingPointError as error:
        raise ForecastError("quantile metric arithmetic overflowed") from error


def _validate_quantile_inputs(
    values: FloatArray, targets: FloatArray, levels: tuple[float, ...]
) -> None:
    if values.shape != (*targets.shape, len(levels)):
        raise ForecastError("quantile metric dimensions differ")
    _validate_level_range(levels)
    if not np.isfinite(values).all() or np.any(np.diff(values, axis=-1) < 0):
        raise ForecastError("nonfinite or crossing quantiles cannot define intervals")
    if not np.isfinite(targets).all():
        raise ForecastError("quantile targets must be finite")


def _validate_level_range(levels: tuple[float, ...]) -> None:
    if tuple(sorted(set(levels))) != levels or not all(0 < q < 1 for q in levels):
        raise ForecastError("invalid quantile labels")


def _quantile_losses(
    values: FloatArray,
    targets: FloatArray,
    levels: tuple[float, ...],
    intervals: tuple[tuple[float, float], ...],
) -> dict[MetricKey, FloatArray]:
    result: dict[MetricKey, FloatArray] = {}
    for index, level in enumerate(levels):
        error = targets - values[..., index]
        result[(MetricId.PINBALL, level, level)] = np.maximum(
            level * error, (level - 1) * error
        )
    for lower, upper in intervals:
        if lower not in levels or upper not in levels:
            raise ForecastError("configured interval endpoints absent from prediction")
        lo, hi = values[..., levels.index(lower)], values[..., levels.index(upper)]
        result[(MetricId.COVERAGE, lower, upper)] = (
            (targets >= lo) & (targets <= hi)
        ).astype(np.float64)
        result[(MetricId.WIDTH, lower, upper)] = hi - lo
    return result


def group_interval(
    values: FloatArray,
    config: BenchmarkConfig,
    *,
    corrected: bool = False,
) -> tuple[float | None, float | None]:
    """Resample whole independent group values; never resample overlapping windows."""
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ForecastError("bootstrap expects one finite value per independent group")
    if (
        interval_status(len(values), config, corrected=corrected)
        != "descriptive-bootstrap"
    ):
        return None, None
    means = _bootstrap_means(values, config)
    family = config.comparison_family_size if corrected else 1
    tail = (1 - config.confidence_level) / (2 * family)
    bounds = np.quantile(means, (tail, 1 - tail), method="linear")
    return float(bounds[0]), float(bounds[1])


def interval_status(
    groups: int, config: BenchmarkConfig, *, corrected: bool = False
) -> str:
    """Distinguish inadequate independent sampling from unresolved corrected tails."""
    if groups < config.min_groups:
        return "insufficient-groups"
    family = config.comparison_family_size if corrected else 1
    tail_samples = (
        config.bootstrap_samples * (1 - config.confidence_level) / (2 * family)
    )
    if tail_samples < config.min_tail_samples:
        return "insufficient-bootstrap-resolution"
    return "descriptive-bootstrap"


def _bootstrap_means(values: FloatArray, config: BenchmarkConfig) -> FloatArray:
    estimated = config.bootstrap_samples * len(values) * 16
    if estimated > config.max_bootstrap_bytes:
        raise ForecastError("bootstrap arrays exceed configured memory budget")
    draws = np.random.default_rng(config.seed).integers(
        len(values), size=(config.bootstrap_samples, len(values))
    )
    return np.asarray(values[draws].mean(axis=1), dtype=np.float64)
