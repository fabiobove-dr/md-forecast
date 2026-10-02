"""Lazy, deterministic forecast indices from a validated pre-window split."""

from collections.abc import Iterator
from dataclasses import dataclass
from heapq import merge
from math import isclose
from typing import Annotated, Self

from pydantic import Field, model_validator

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    PS_PER_NS,
    TIME_ATOL,
    TIME_RTOL,
    SamplingStatus,
    Split,
    TimeUnit,
)
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.schemas import BoundaryModel, SchemaVersion, TrajectoryManifest
from md_forecast.data.splits import SplitManifest


class WindowConfig(BoundaryModel):
    """A predeclared grid; physical contexts mean first-to-last context span."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    contexts: Annotated[
        tuple[Annotated[float, Field(strict=True, ge=0)], ...], Field(min_length=1)
    ]
    horizons: Annotated[
        tuple[Annotated[float, Field(strict=True, gt=0)], ...], Field(min_length=1)
    ]
    unit: TimeUnit = TimeUnit.FRAME
    stride_frames: Annotated[int, Field(strict=True, gt=0)]
    allow_assumed_time: bool = False

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Reject duplicate cells and fractional/zero frame contexts."""
        if len(set(self.contexts)) != len(self.contexts) or len(
            set(self.horizons)
        ) != len(self.horizons):
            raise ValueError("window grid lengths must be unique")
        if self.unit == TimeUnit.FRAME:
            _validate_frame_grid(self.contexts, self.horizons)
        return self


def _validate_frame_grid(
    contexts: tuple[float, ...], horizons: tuple[float, ...]
) -> None:
    if any(not value.is_integer() for value in (*contexts, *horizons)):
        raise ValueError("frame lengths must be integers")
    if min(contexts) < 1:
        raise ValueError("frame contexts must contain at least one sample")


@dataclass(frozen=True)
class WindowIndex:
    """Half-open context/target slices, with source-group and experiment identity."""

    trajectory_id: str
    system_id: str
    group_id: str
    split: Split
    start: int
    context_frames: int
    horizon_frames: int
    split_hash: str
    config_hash: str
    sampling_status: SamplingStatus
    context_span_ps: float | None
    horizon_span_ps: float | None

    @property
    def context_slice(self) -> slice:
        """Observed samples only, excluding every forecast target."""
        return slice(self.start, self.start + self.context_frames)

    @property
    def target_slice(self) -> slice:
        """Immediately following forecast samples."""
        origin = self.start + self.context_frames
        return slice(origin, origin + self.horizon_frames)


def _physical_interval(record: TrajectoryManifest, config: WindowConfig) -> float:
    if record.frame_interval_ps is None:
        raise DataContractError(
            f"physical windows require uniform sampling: {record.trajectory_id}"
        )
    if (
        record.sampling_status != SamplingStatus.VERIFIED
        and not config.allow_assumed_time
    ):
        raise DataContractError(
            "physical windows require verified time or explicit assumed-time opt-in"
        )
    return record.frame_interval_ps


def _sample_steps(duration_ps: float, interval_ps: float) -> int:
    count = duration_ps / interval_ps
    rounded = round(count)
    if not isclose(count, rounded, rel_tol=TIME_RTOL, abs_tol=TIME_ATOL):
        raise DataContractError(
            "physical window length is not aligned to the sampling interval"
        )
    return rounded


def window_lengths(
    record: TrajectoryManifest, config: WindowConfig
) -> tuple[tuple[int, int], ...]:
    """Resolve and reject every grid cell before any indices are emitted."""
    if config.unit == TimeUnit.FRAME:
        contexts, horizons = _frame_lengths(config)
    else:
        contexts, horizons = _physical_lengths(record, config)
    cells = tuple((context, horizon) for context in contexts for horizon in horizons)
    _validate_lengths(record, cells)
    return cells


def _frame_lengths(config: WindowConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(int(value) for value in sorted(config.contexts)),
        tuple(int(value) for value in sorted(config.horizons)),
    )


def _physical_lengths(
    record: TrajectoryManifest, config: WindowConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    interval = _physical_interval(record, config)
    factor = PS_PER_NS if config.unit == TimeUnit.NS else 1.0
    contexts = tuple(
        _sample_steps(value * factor, interval) + 1 for value in sorted(config.contexts)
    )
    horizons = tuple(
        _sample_steps(value * factor, interval) for value in sorted(config.horizons)
    )
    return contexts, horizons


def _validate_lengths(
    record: TrajectoryManifest, cells: tuple[tuple[int, int], ...]
) -> None:
    for context, horizon in cells:
        if horizon < 1 or context + horizon > record.frame_count:
            raise DataContractError(
                f"impossible C={context}, H={horizon} for {record.trajectory_id} "
                f"({record.frame_count} frames)"
            )


def iter_windows(
    manifest: SplitManifest, config: WindowConfig, partition: Split
) -> Iterator[WindowIndex]:
    """Yield indices only after a complete, validated partition and feasible grid.

    All selected trajectory metadata is checked before yielding the first window.
    No numerical arrays are read, duplicated, or used to choose grid cells.
    """
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    records = _partition_records(snapshot, partition)
    lengths = {
        record.trajectory_id: window_lengths(record, config) for record, _ in records
    }
    split_hash, config_hash = metadata_hash(snapshot), metadata_hash(config)
    for record, group in records:
        yield from _record_windows(
            record,
            group,
            partition,
            config,
            lengths[record.trajectory_id],
            split_hash,
            config_hash,
        )


def _partition_records(
    manifest: SplitManifest, partition: Split
) -> list[tuple[TrajectoryManifest, str]]:
    assignments = {entry.trajectory_id: entry for entry in manifest.assignments}
    records = [
        (record, assignments[record.trajectory_id].group_id)
        for record in manifest.registry.trajectories
        if assignments[record.trajectory_id].split == partition
    ]
    return sorted(records, key=lambda item: item[0].trajectory_id)


def _record_windows(
    record: TrajectoryManifest,
    group: str,
    partition: Split,
    config: WindowConfig,
    lengths: tuple[tuple[int, int], ...],
    split_hash: str,
    config_hash: str,
) -> Iterator[WindowIndex]:
    for context, horizon in lengths:
        for start in range(
            0, record.frame_count - context - horizon + 1, config.stride_frames
        ):
            yield _window_index(
                record,
                group,
                partition,
                start,
                context,
                horizon,
                split_hash,
                config_hash,
            )


def _window_index(
    record: TrajectoryManifest,
    group: str,
    partition: Split,
    start: int,
    context: int,
    horizon: int,
    split_hash: str,
    config_hash: str,
) -> WindowIndex:
    interval = record.frame_interval_ps
    context_span = None if interval is None else (context - 1) * interval
    horizon_span = None if interval is None else horizon * interval
    return WindowIndex(
        record.trajectory_id,
        record.system_id,
        group,
        partition,
        start,
        context,
        horizon,
        split_hash,
        config_hash,
        record.sampling_status,
        context_span,
        horizon_span,
    )


def validate_window(
    manifest: SplitManifest, config: WindowConfig, window: WindowIndex
) -> TrajectoryManifest:
    """Reject forged, stale, or out-of-bounds indices before reading feature values."""
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    records = {
        record.trajectory_id: record for record in snapshot.registry.trajectories
    }
    assignments = {entry.trajectory_id: entry for entry in snapshot.assignments}
    try:
        record = records[window.trajectory_id]
        assignment = assignments[window.trajectory_id]
    except KeyError as error:
        raise DataContractError("window references an unknown trajectory") from error
    _validate_window_position(record, config, window)
    expected = _window_index(
        record,
        assignment.group_id,
        assignment.split,
        window.start,
        window.context_frames,
        window.horizon_frames,
        metadata_hash(snapshot),
        metadata_hash(config),
    )
    if window != expected:
        raise DataContractError(
            "window identity/split/time/hash differs from its declared protocol"
        )
    return record


def _validate_window_position(
    record: TrajectoryManifest, config: WindowConfig, window: WindowIndex
) -> None:
    if window.start < 0 or window.start % config.stride_frames:
        raise DataContractError("window start is negative or not aligned to the stride")
    if (window.context_frames, window.horizon_frames) not in window_lengths(
        record, config
    ):
        raise DataContractError("window lengths are not in the declared grid")
    if window.target_slice.stop > record.frame_count:
        raise DataContractError("window exceeds trajectory bounds")


def iter_context_regions(
    manifest: SplitManifest, config: WindowConfig, partition: Split
) -> Iterator[tuple[str, int, int]]:
    """Merge observed context intervals with O(grid size) working memory per series."""
    snapshot = SplitManifest.model_validate_json(manifest.model_dump_json())
    records = _partition_records(snapshot, partition)
    lengths = {
        record.trajectory_id: window_lengths(record, config) for record, _ in records
    }
    for record, _ in records:
        streams = [
            _context_intervals(record, config, cell)
            for cell in lengths[record.trajectory_id]
        ]
        for start, stop in _merged_intervals(merge(*streams)):
            yield record.trajectory_id, start, stop


def _context_intervals(
    record: TrajectoryManifest, config: WindowConfig, cell: tuple[int, int]
) -> Iterator[tuple[int, int]]:
    context, horizon = cell
    for start in range(
        0, record.frame_count - context - horizon + 1, config.stride_frames
    ):
        yield start, start + context


def _merged_intervals(
    intervals: Iterator[tuple[int, int]],
) -> Iterator[tuple[int, int]]:
    current: tuple[int, int] | None = None
    for start, stop in intervals:
        if current is None:
            current = start, stop
        elif start <= current[1]:
            current = current[0], max(stop, current[1])
        else:
            yield current
            current = start, stop
    if current is not None:
        yield current
