"""TRAIN-only, lead-specific residual quantiles around unchanged statistical points."""

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import Split, TimeUnit
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch, ForecastSpec, iter_forecasts
from md_forecast.data.predictions import (
    QuantileForecast,
    RuntimeStats,
    validate_quantile_levels,
)
from md_forecast.data.preprocessing import SeriesLoader
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, Identifier
from md_forecast.data.series import FloatArray
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, iter_windows
from md_forecast.models.base import ModelConfig
from md_forecast.models.baselines import StatisticalBaseline


class ResidualConfig(BoundaryModel):
    """Bounded empirical residual fitting; this does not promise nominal coverage."""

    levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    quantile_method: Literal["linear"] = "linear"
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    max_windows: Annotated[int, Field(strict=True, gt=0)]
    max_residual_bytes: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def valid_levels(self) -> Self:
        """Require unambiguous probabilities including the preserved point median."""
        validate_quantile_levels(self.levels)
        return self


class ResidualState(BoundaryModel):
    """Portable native-unit (H,F,Q) offsets and complete TRAIN fitting identity."""

    spec: ForecastSpec
    point_config: ModelConfig
    fitting: ResidualConfig
    offsets: tuple[tuple[tuple[float, ...], ...], ...]
    input_hash: ArtifactHash
    training_window_count: Annotated[int, Field(strict=True, gt=0)]
    training_trajectory_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    training_group_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    fit_scope: Literal["train-window-future-residuals"] = (
        "train-window-future-residuals"
    )
    support: Literal["unconstrained-no-point-clipping"] = (
        "unconstrained-no-point-clipping"
    )

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Reject nonfinite/crossing offsets or any modification of the point median."""
        StatisticalBaseline(self.point_config)
        _check_offset_dimensions(self)
        _check_offset_order(self)
        if self.training_window_count > self.fitting.max_windows:
            raise ValueError("residual fitting exceeds window budget")
        return self


def _check_offset_dimensions(state: ResidualState) -> None:
    values = np.asarray(state.offsets, dtype=np.float64)
    shape = (
        state.spec.horizon_frames,
        len(state.spec.feature_ids),
        len(state.fitting.levels),
    )
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("residual offsets differ from finite forecast dimensions")


def _check_offset_order(state: ResidualState) -> None:
    values = np.asarray(state.offsets, dtype=np.float64)
    if np.any(np.diff(values, axis=-1) < 0) or np.any(
        values[..., state.fitting.levels.index(0.5)] != 0
    ):
        raise ValueError("residual quantiles must be ordered and preserve the point")


def _fit_budget(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    config: ResidualConfig,
) -> int:
    if (grid.unit, len(grid.contexts), len(grid.horizons)) != (TimeUnit.FRAME, 1, 1):
        raise ForecastError("fit residuals on one explicit frame cell")
    count = sum(1 for _ in iter_windows(split, grid, Split.TRAIN))
    if not 0 < count <= config.max_windows:
        raise ForecastError("residual training windows are absent or exceed budget")
    estimated = count * int(grid.horizons[0]) * len(features) * 8 * 4
    if estimated > config.max_residual_bytes:
        raise ForecastError("residual arrays exceed configured memory budget")
    return count


def fit_residual_quantiles(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    loader: SeriesLoader,
    point: StatisticalBaseline,
    config: ResidualConfig,
) -> ResidualState:
    """Fit pooled TRAIN-window residuals separately for every lead and channel.

    No partition option permits VAL/TEST fitting. Quantiles below/above the median
    are bounded by zero offset, and the median offset is exactly zero. This keeps
    the original point while allowing asymmetric empirical intervals. Overlapping
    windows are training samples, never independent uncertainty replicates.
    """
    count = _fit_budget(split, grid, features, config)
    residuals = []
    trajectories: set[str] = set()
    groups: set[str] = set()
    digest = hashlib.sha256()
    for batch, targets in iter_forecasts(
        split,
        grid,
        Split.TRAIN,
        features,
        loader,
        batch_size=config.batch_size,
        with_targets=True,
    ):
        assert targets is not None
        with np.errstate(over="ignore", invalid="ignore"):
            residuals.append(targets - point.predict(batch))
        inputs, ids, group_ids = _batch_inputs(batch, targets)
        digest.update(inputs)
        trajectories.update(ids)
        groups.update(group_ids)

    values = np.concatenate(residuals)
    if not np.isfinite(values).all():
        raise ForecastError("training residuals must be finite")
    offsets = _fit_offsets(values, config)
    return ResidualState.model_validate(
        {
            "spec": batch.spec,
            "point_config": point.config,
            "fitting": config,
            "offsets": offsets.tolist(),
            "input_hash": "sha256:" + digest.hexdigest(),
            "training_window_count": count,
            "training_trajectory_ids": tuple(sorted(trajectories)),
            "training_group_ids": tuple(sorted(groups)),
        }
    )


def _batch_inputs(
    batch: ForecastBatch, targets: FloatArray
) -> tuple[bytes, set[str], set[str]]:
    identities = []
    trajectories: set[str] = set()
    groups: set[str] = set()
    for index in batch.indices:
        identities.append((index.trajectory_id, index.start))
        trajectories.add(index.trajectory_id)
        groups.add(index.group_id)
    inputs = (
        json.dumps(identities, separators=(",", ":")).encode()
        + batch.context.astype("<f8").tobytes()
        + targets.astype("<f8").tobytes()
    )
    return inputs, trajectories, groups


def _fit_offsets(values: FloatArray, config: ResidualConfig) -> FloatArray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            offsets = np.quantile(
                values, config.levels, axis=0, method=config.quantile_method
            ).transpose(1, 2, 0)
    except FloatingPointError as error:
        raise ForecastError("residual quantile fitting overflowed") from error
    for index, level in enumerate(config.levels):
        offsets[..., index] = (
            np.minimum(offsets[..., index], 0)
            if level < 0.5
            else np.maximum(offsets[..., index], 0)
        )
    offsets[..., config.levels.index(0.5)] = 0
    return offsets


@dataclass(frozen=True)
class ResidualBaseline:
    """Statistical points plus TRAIN-fitted offsets and strict source/grid binding."""

    state: ResidualState

    def __post_init__(self) -> None:
        """Validate loaded state at the adapter boundary without filesystem effects."""
        object.__setattr__(
            self,
            "state",
            ResidualState.model_validate_json(self.state.model_dump_json()),
        )

    @property
    def config(self) -> ModelConfig:
        """Point configuration; the artifact hash additionally binds interval state."""
        return self.state.point_config

    @property
    def artifact_hash(self) -> str:
        """Bind residual offsets, source inputs and all fitting settings."""
        return metadata_hash(self.state)

    def forecast(self, batch: ForecastBatch) -> QuantileForecast:
        """Use context-only points and fixed offsets without on-the-fly calibration."""
        if batch.spec != self.state.spec:
            raise ForecastError(
                "residual forecast differs from fitted source/features/grid"
            )
        start = perf_counter()
        points = StatisticalBaseline(self.config).predict(batch)
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = points[..., None] + np.asarray(self.state.offsets)[None, ...]
        except FloatingPointError as error:
            raise ForecastError("residual forecast overflowed") from error
        return QuantileForecast(
            batch.spec,
            batch.indices,
            self.state.fitting.levels,
            values,
            RuntimeStats(
                seconds=perf_counter() - start,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
            ),
        )

    def predict(self, batch: ForecastBatch) -> FloatArray:
        """Return the unchanged statistical point through its explicit median view."""
        return self.forecast(batch).median
