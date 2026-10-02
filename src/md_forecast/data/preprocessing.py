"""Context-scoped standardization with explicit fitting provenance."""

from collections.abc import Callable
from itertools import groupby
from typing import Annotated, Literal, Self

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    CONSTANT_FEATURE_SCALE,
    PS_PER_NS,
    SHA256_PATTERN,
    TIME_ATOL,
    TIME_COLUMN,
    TIME_RTOL,
    Split,
    TimeUnit,
)
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.schemas import (
    BoundaryModel,
    DatasetConfig,
    ExperimentConfig,
    Identifier,
    SchemaVersion,
    TrajectoryManifest,
)
from md_forecast.data.series import FloatArray, validate_series_structure
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import (
    WindowConfig,
    WindowIndex,
    iter_context_regions,
    validate_window,
    window_lengths,
)

type SeriesLoader = Callable[[TrajectoryManifest], pa.Table]


class ScalingConfig(BoundaryModel):
    """Explicit fitting scope and channels, using population standard deviation."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    scope: Literal["training-contexts", "context-local"]
    feature_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_features(self) -> Self:
        """Do not ambiguously repeat a channel."""
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("duplicate scaling feature IDs")
        return self


class FitRegion(BoundaryModel):
    """A half-open observed-frame interval used for fitting, never forecast targets."""

    trajectory_id: Identifier
    start: Annotated[int, Field(strict=True, ge=0)]
    stop: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def nonempty(self) -> Self:
        """Only nonempty observed intervals contribute statistics."""
        if self.stop <= self.start:
            raise ValueError("fitting region must be nonempty")
        return self


class ScalerMetadata(BoundaryModel):
    """Reusable moments with split/window/feature and observed-frame provenance."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    config: ScalingConfig
    dataset: DatasetConfig
    split_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    window_config_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    fit_regions: Annotated[tuple[FitRegion, ...], Field(min_length=1)]
    sample_count: Annotated[int, Field(strict=True, gt=0)]
    center: tuple[float, ...]
    scale: tuple[Annotated[float, Field(gt=0)], ...]

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        """Validate channel dimensions and the declared fitting observations."""
        _validate_features(self.config, self.dataset)
        if len(self.center) != len(self.config.feature_ids) or len(self.scale) != len(
            self.center
        ):
            raise ValueError(
                "scaler statistic dimensions differ from feature selection"
            )
        _validate_fit_regions(self)
        return self


def _validate_features(config: ScalingConfig, dataset: DatasetConfig) -> None:
    known = {feature.feature_id for feature in dataset.features}
    if not set(config.feature_ids) <= known:
        raise ValueError("scaling selection contains unknown feature IDs")


def _validate_fit_regions(metadata: ScalerMetadata) -> None:
    if (
        sum(region.stop - region.start for region in metadata.fit_regions)
        != metadata.sample_count
    ):
        raise ValueError("sample_count differs from fitted regions")
    if metadata.config.scope == "context-local" and len(metadata.fit_regions) != 1:
        raise ValueError("context-local scaling fits exactly one observed context")
    _validate_region_order(metadata.fit_regions)


def _validate_region_order(regions: tuple[FitRegion, ...]) -> None:
    previous: dict[str, int] = {}
    for region in regions:
        if region.start < previous.get(region.trajectory_id, 0):
            raise ValueError("fitting regions overlap or are out of order")
        previous[region.trajectory_id] = region.stop


def _training_regions(
    manifest: SplitManifest, config: WindowConfig
) -> tuple[FitRegion, ...]:
    return tuple(
        FitRegion(trajectory_id=trajectory, start=start, stop=stop)
        for trajectory, start, stop in iter_context_regions(
            manifest, config, Split.TRAIN
        )
    )


def _validate_source(
    table: pa.Table,
    record: TrajectoryManifest,
    dataset: DatasetConfig,
) -> None:
    registry = validate_series_structure(table)
    if registry.trajectories != (record,) or registry.dataset != dataset:
        raise DataContractError(
            "loaded series differs from the split's canonical source metadata"
        )


def _region_values(
    table: pa.Table,
    record: TrajectoryManifest,
    region: FitRegion,
    features: tuple[str, ...],
) -> FloatArray:
    observed = table.slice(region.start, region.stop - region.start)
    time = observed[TIME_COLUMN].to_numpy()
    values = np.column_stack([observed[feature].to_numpy() for feature in features])
    if not np.isfinite(time).all() or not np.isfinite(values).all():
        raise DataContractError("observed fitting values must be finite")
    _validate_observed_grid(time, record, region)
    return values


def _validate_observed_grid(
    time: FloatArray, record: TrajectoryManifest, region: FitRegion
) -> None:
    _validate_observed_order(time, region.start)
    if record.time_unit == TimeUnit.FRAME:
        if not np.array_equal(time, np.arange(region.start, region.stop)):
            raise DataContractError("observed frame indices are inconsistent")
        return
    if record.frame_interval_ps is not None:
        _validate_observed_interval(time, record, region, record.frame_interval_ps)


def _validate_observed_interval(
    time: FloatArray, record: TrajectoryManifest, region: FitRegion, interval_ps: float
) -> None:
    factor = PS_PER_NS if record.time_unit == TimeUnit.NS else 1.0
    expected = np.arange(region.start, region.stop) * interval_ps / factor
    if not np.allclose(time, expected, rtol=TIME_RTOL, atol=TIME_ATOL):
        raise DataContractError("observed sampling interval is inconsistent")


def _validate_observed_order(time: FloatArray, start: int) -> None:
    if not np.all(np.diff(time) > 0) or (start == 0 and time[0] != 0):
        raise DataContractError("observed time axis is inconsistent")


def _combine_moments(
    values: FloatArray, count: int, mean: FloatArray, m2: FloatArray
) -> tuple[int, FloatArray, FloatArray]:
    batch_mean = values.mean(axis=0)
    batch_m2 = np.square(values - batch_mean).sum(axis=0)
    if count == 0:
        return len(values), batch_mean, batch_m2
    total = count + len(values)
    delta = batch_mean - mean
    return (
        total,
        mean + delta * len(values) / total,
        m2 + batch_m2 + np.square(delta) * count * len(values) / total,
    )


def _fit(
    manifest: SplitManifest,
    windows: WindowConfig,
    config: ScalingConfig,
    regions: tuple[FitRegion, ...],
    loader: SeriesLoader,
) -> ScalerMetadata:
    if not regions:
        raise DataContractError("no training contexts available for scaler fitting")
    records = {
        record.trajectory_id: record for record in manifest.registry.trajectories
    }
    count, mean, scale = _fit_moments(
        regions, records, manifest.registry.dataset, config, loader
    )
    return ScalerMetadata(
        config=config,
        dataset=manifest.registry.dataset,
        split_hash=metadata_hash(manifest),
        window_config_hash=metadata_hash(windows),
        fit_regions=regions,
        sample_count=count,
        center=tuple(mean),
        scale=tuple(scale),
    )


def _fit_moments(
    regions: tuple[FitRegion, ...],
    records: dict[str, TrajectoryManifest],
    dataset: DatasetConfig,
    config: ScalingConfig,
    loader: SeriesLoader,
) -> tuple[int, FloatArray, FloatArray]:
    count = 0
    mean = np.zeros(len(config.feature_ids), dtype=np.float64)
    m2 = np.zeros_like(mean)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for trajectory, intervals in groupby(
                regions, key=lambda region: region.trajectory_id
            ):
                record = records[trajectory]
                table = loader(record)
                _validate_source(table, record, dataset)
                for region in intervals:
                    values = _region_values(table, record, region, config.feature_ids)
                    count, mean, m2 = _combine_moments(values, count, mean, m2)
            scale = np.sqrt(m2 / count)
    except FloatingPointError as error:
        raise DataContractError(
            "scaler moments overflowed; check observable magnitudes"
        ) from error
    scale = np.where(scale == 0, CONSTANT_FEATURE_SCALE, scale)
    return count, mean, scale


def fit_training_scaler(
    manifest: SplitManifest,
    windows: WindowConfig,
    config: ScalingConfig,
    loader: SeriesLoader,
) -> ScalerMetadata:
    """Fit unique observed training-context frames; never load held-out trajectories."""
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    _check_scope(config, "training-contexts", snapshot.registry.dataset)
    return _fit(snapshot, windows, config, _training_regions(snapshot, windows), loader)


def fit_context_scaler(
    manifest: SplitManifest,
    windows: WindowConfig,
    window: WindowIndex,
    config: ScalingConfig,
    loader: SeriesLoader,
) -> ScalerMetadata:
    """Fit only the supplied validated context, even on validation/test trajectories."""
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    _check_scope(config, "context-local", snapshot.registry.dataset)
    validate_window(snapshot, windows, window)
    region = FitRegion(
        trajectory_id=window.trajectory_id,
        start=window.context_slice.start,
        stop=window.context_slice.stop,
    )
    return _fit(snapshot, windows, config, (region,), loader)


def _check_scope(config: ScalingConfig, expected: str, dataset: DatasetConfig) -> None:
    if config.scope != expected:
        raise DataContractError(f"scaling operation requires scope={expected}")
    try:
        _validate_features(config, dataset)
    except ValueError as error:
        raise DataContractError(str(error)) from error


def transform(
    values: FloatArray, metadata: ScalerMetadata, *, inverse: bool = False
) -> FloatArray:
    """Apply frozen statistics without fitting; inverse restores the original units."""
    if values.ndim != 2 or values.shape[1] != len(metadata.config.feature_ids):
        raise DataContractError("scaling array shape differs from ordered feature IDs")
    if not np.isfinite(values).all():
        raise DataContractError("scaling values must be finite")
    center = np.asarray(metadata.center, dtype=np.float64)
    scale = np.asarray(metadata.scale, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = _transform(values, center, scale, inverse)
    except FloatingPointError as error:
        raise DataContractError(
            "scaling overflowed; check values and fitted statistics"
        ) from error
    return result


def _transform(
    values: FloatArray, center: FloatArray, scale: FloatArray, inverse: bool
) -> FloatArray:
    if inverse:
        return np.asarray(values * scale + center, dtype=np.float64)
    return np.asarray((values - center) / scale, dtype=np.float64)


def bind_experiment(
    experiment: ExperimentConfig,
    manifest: SplitManifest,
    windows: WindowConfig,
    scaler: ScalerMetadata,
) -> ExperimentConfig:
    """Record all data-preparation hashes after verifying their linkage."""
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    if experiment.dataset != snapshot.registry.dataset:
        raise DataContractError("experiment dataset differs from split registry")
    validate_scaler(scaler, snapshot, windows)
    return ExperimentConfig.model_validate(
        experiment.model_dump()
        | {
            "split_hash": metadata_hash(snapshot),
            "window_config_hash": metadata_hash(windows),
            "preprocessing_hash": metadata_hash(scaler),
        }
    )


def validate_scaler(
    metadata: ScalerMetadata, manifest: SplitManifest, windows: WindowConfig
) -> None:
    """Verify reused statistics belong to the data/partition/window protocol."""
    if metadata.split_hash != metadata_hash(
        manifest
    ) or metadata.window_config_hash != metadata_hash(windows):
        raise DataContractError("scaler references a different split/window protocol")
    if metadata.dataset != manifest.registry.dataset:
        raise DataContractError("scaler dataset differs from split registry")
    _validate_scaler_regions(metadata, manifest, windows)


def _validate_scaler_regions(
    metadata: ScalerMetadata, manifest: SplitManifest, windows: WindowConfig
) -> None:
    if metadata.config.scope == "training-contexts":
        if metadata.fit_regions != _training_regions(manifest, windows):
            raise DataContractError(
                "scaler fitting regions are not the declared training contexts"
            )
    else:
        _validate_context_region(metadata.fit_regions[0], manifest, windows)


def _validate_context_region(
    region: FitRegion, manifest: SplitManifest, windows: WindowConfig
) -> None:
    records = {
        record.trajectory_id: record for record in manifest.registry.trajectories
    }
    try:
        record = records[region.trajectory_id]
    except KeyError as error:
        raise DataContractError(
            "scaler context references an unknown trajectory"
        ) from error
    cells = window_lengths(record, windows)
    _check_context_region(region, record, windows, cells)


def _check_context_region(
    region: FitRegion,
    record: TrajectoryManifest,
    windows: WindowConfig,
    cells: tuple[tuple[int, int], ...],
) -> None:
    valid = any(
        region.stop - region.start == context
        and region.stop + horizon <= record.frame_count
        for context, horizon in cells
    )
    if not valid or region.start % windows.stride_frames:
        raise DataContractError("scaler fitting region is not a declared context")
