"""Canonical context-only batches; future labels are carried separately."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import groupby, islice
from typing import Annotated, Self

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.preprocessing import FitRegion, SeriesLoader, observed_values
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    DatasetConfig,
    Identifier,
    SchemaVersion,
    TrajectoryManifest,
)
from md_forecast.data.series import FloatArray, validate_series_structure
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, WindowIndex, iter_windows


class ForecastSpec(BoundaryModel):
    """One grid cell and ordered native-unit channels, shared by every model."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    dataset: DatasetConfig
    feature_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    context_frames: Annotated[int, Field(strict=True, gt=0)]
    horizon_frames: Annotated[int, Field(strict=True, gt=0)]
    split_hash: ArtifactHash
    window_config_hash: ArtifactHash

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        """Preserve explicit order and reject unknown or repeated channels."""
        known = {feature.feature_id for feature in self.dataset.features}
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("duplicate forecast feature IDs")
        if not set(self.feature_ids) <= known:
            raise ValueError("unknown forecast feature IDs")
        return self


@dataclass(frozen=True)
class ForecastBatch:
    """Immutable observed context (B,C,F) with validated source-window identities."""

    spec: ForecastSpec
    indices: tuple[WindowIndex, ...]
    context: FloatArray

    def __post_init__(self) -> None:
        """Validate once and detach read-only numerical inputs from their caller."""
        spec = ForecastSpec.model_validate_json(self.spec.model_dump_json())
        _validate_indices(self.indices, spec)
        validate_array(
            self.context,
            (len(self.indices), spec.context_frames, len(spec.feature_ids)),
        )
        context = self.context.copy()
        context.flags.writeable = False
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "context", context)


def _validate_indices(indices: tuple[WindowIndex, ...], spec: ForecastSpec) -> None:
    if not indices:
        raise DataContractError("forecast batch must contain windows")
    expected = (
        spec.context_frames,
        spec.horizon_frames,
        spec.split_hash,
        spec.window_config_hash,
        indices[0].split,
    )
    for index in indices:
        actual = (
            index.context_frames,
            index.horizon_frames,
            index.split_hash,
            index.config_hash,
            index.split,
        )
        if actual != expected:
            raise DataContractError(
                "forecast indices differ from batch protocol/partition"
            )


def validate_array(values: FloatArray, shape: tuple[int, ...]) -> None:
    """Reject malformed/nonfinite arrays at the shared numerical boundary."""
    if values.dtype != np.float64 or values.shape != shape:
        raise DataContractError(
            f"expected float64 forecast array {shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise DataContractError("forecast values must be finite")


def iter_forecasts(
    split: SplitManifest,
    grid: WindowConfig,
    partition: Split,
    feature_ids: tuple[str, ...],
    loader: SeriesLoader,
    *,
    batch_size: int,
    with_targets: bool = False,
) -> Iterator[tuple[ForecastBatch, FloatArray | None]]:
    """Read one source series at a time, never targets unless explicitly requested.

    The caller's loader may independently perform whole-file publication QC.
    This iterator checks structure/source metadata and reads only requested slices.
    """
    if batch_size < 1:
        raise DataContractError("forecast batch size must be positive")
    snapshot = SplitManifest.model_validate_json(split.model_dump_json())
    records = {
        record.trajectory_id: record for record in snapshot.registry.trajectories
    }
    windows = iter_windows(snapshot, grid, partition)
    for trajectory, indices in groupby(windows, key=lambda index: index.trajectory_id):
        record = records[trajectory]
        table = _load_table(loader, record, snapshot.registry.dataset)
        yield from _table_batches(
            snapshot,
            grid,
            feature_ids,
            table,
            record,
            indices,
            batch_size,
            with_targets,
        )


def _load_table(
    loader: SeriesLoader, record: TrajectoryManifest, dataset: DatasetConfig
) -> pa.Table:
    table = loader(record)
    metadata = validate_series_structure(table)
    if metadata.dataset != dataset or metadata.trajectories != (record,):
        raise DataContractError("forecast series differs from canonical source")
    return table


def _table_batches(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    table: pa.Table,
    record: TrajectoryManifest,
    indices: Iterator[WindowIndex],
    batch_size: int,
    with_targets: bool,
) -> Iterator[tuple[ForecastBatch, FloatArray | None]]:
    for lengths, cell_indices in groupby(
        indices, key=lambda index: (index.context_frames, index.horizon_frames)
    ):
        spec = ForecastSpec(
            dataset=split.registry.dataset,
            feature_ids=features,
            context_frames=lengths[0],
            horizon_frames=lengths[1],
            split_hash=metadata_hash(split),
            window_config_hash=metadata_hash(grid),
        )
        while batch_indices := tuple(islice(cell_indices, batch_size)):
            context = np.stack(
                [
                    _slice_values(table, index.context_slice, features, record)
                    for index in batch_indices
                ]
            )
            batch = ForecastBatch(spec, batch_indices, context)
            targets = (
                _targets(table, batch_indices, features, spec, record)
                if with_targets
                else None
            )
            yield batch, targets


def _slice_values(
    table: pa.Table,
    interval: slice,
    features: tuple[str, ...],
    record: TrajectoryManifest,
) -> FloatArray:
    region = FitRegion(
        trajectory_id=record.trajectory_id, start=interval.start, stop=interval.stop
    )
    return observed_values(table, record, region, features)


def _targets(
    table: pa.Table,
    indices: tuple[WindowIndex, ...],
    features: tuple[str, ...],
    spec: ForecastSpec,
    record: TrajectoryManifest,
) -> FloatArray:
    values = np.stack(
        [
            _slice_values(table, index.target_slice, features, record)
            for index in indices
        ]
    )
    validate_array(values, (len(indices), spec.horizon_frames, len(features)))
    return values
