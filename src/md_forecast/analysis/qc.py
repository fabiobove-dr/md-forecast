"""Reproducible training-group selection and descriptive trajectory diagnostics."""

import math
import random
from typing import Annotated, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.preprocessing import SeriesLoader
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    Identifier,
    SchemaVersion,
    TrajectoryManifest,
)
from md_forecast.data.series import FloatArray, validate_series
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, window_lengths

DECAY_LEVEL = math.exp(-1)
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
type Nonnegative = Annotated[float, Field(ge=0)]


class DevelopmentConfig(BoundaryModel):
    """Sample whole official training dependency groups without reading values."""

    seed: Annotated[int, Field(strict=True, ge=0)]
    max_groups: Annotated[int, Field(strict=True, gt=0)]


class DevelopmentManifest(BoundaryModel):
    """Self-contained source split and deterministic development identities."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    split: SplitManifest
    config: DevelopmentConfig
    trajectory_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Never permit held-out or partial dependency groups into development."""
        if self.trajectory_ids != _selected_ids(self.split, self.config):
            raise ValueError("development identities differ from declared selection")
        return self


def _selected_ids(split: SplitManifest, config: DevelopmentConfig) -> tuple[str, ...]:
    if split.config.mode != "official":
        raise ValueError("development QC requires official source partitions")
    groups = _training_groups(split)
    if not groups:
        raise ValueError("no official training groups available")
    random.Random(config.seed).shuffle(groups)
    selected = set(groups[: config.max_groups])
    return tuple(
        sorted(
            entry.trajectory_id
            for entry in split.assignments
            if entry.group_id in selected
        )
    )


def _training_groups(split: SplitManifest) -> list[str]:
    return sorted(
        {entry.group_id for entry in split.assignments if entry.split == Split.TRAIN}
    )


def select_development(
    split: SplitManifest, config: DevelopmentConfig
) -> DevelopmentManifest:
    """Freeze a training-only subset before any numerical series is loaded."""
    snapshot = SplitManifest.model_validate_json(split.model_dump_json())
    try:
        return DevelopmentManifest(
            split=snapshot,
            config=config,
            trajectory_ids=_selected_ids(snapshot, config),
        )
    except ValueError as error:
        raise DataContractError(f"cannot select development subset: {error}") from error


class QCConfig(BoundaryModel):
    """Explicit analysis thresholds in each observable's native units."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    development: DevelopmentConfig
    near_constant_std: dict[Identifier, Nonnegative]
    max_lag_frames: Annotated[int, Field(strict=True, gt=0)]
    event_sigma: Annotated[float, Field(gt=0)]
    windows: WindowConfig


class FeatureQC(BoundaryModel):
    """Per-trajectory distribution, dependence and exploratory drift/event summaries."""

    feature_id: Identifier
    missing_count: Annotated[int, Field(strict=True, ge=0)]
    quantiles: tuple[float, ...]
    mean: float
    variance: Nonnegative
    std: Nonnegative
    constant: bool
    near_constant: bool
    acf: tuple[float, ...] | None
    decay_frames: Annotated[int, Field(strict=True, gt=0)] | None
    decay_censored: bool
    linear_drift_per_frame: float
    half_mean_shift_sd: float | None
    jump_count: Annotated[int, Field(strict=True, ge=0)]
    increment_count: Annotated[int, Field(strict=True, ge=0)]


class TrajectoryQC(BoundaryModel):
    """One independent source trajectory, never a pool of overlapping windows."""

    trajectory_id: Identifier
    features: tuple[FeatureQC, ...]
    cross_correlation: tuple[tuple[float | None, ...], ...]


class WindowFeasibility(BoundaryModel):
    """One requested cell and source trajectory; unsupported cells remain explicit."""

    trajectory_id: Identifier
    context: float
    horizon: float
    window_count: Annotated[int, Field(strict=True, ge=0)]
    context_frames: Annotated[int, Field(strict=True, gt=0)] | None = None
    horizon_frames: Annotated[int, Field(strict=True, gt=0)] | None = None
    reason: str | None = None


class QCReport(BoundaryModel):
    """Portable diagnostics with complete source/config/code identity."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    development: DevelopmentManifest
    config: QCConfig
    split_hash: ArtifactHash
    development_hash: ArtifactHash
    config_hash: ArtifactHash
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    lockfile_hash: ArtifactHash
    trajectories: tuple[TrajectoryQC, ...]
    feasibility: tuple[WindowFeasibility, ...]

    @model_validator(mode="after")
    def validate_linkage(self) -> Self:
        """Reject stale hashes and unexpected or held-out diagnostic records."""
        _validate_report_hashes(self)
        if (
            tuple(row.trajectory_id for row in self.trajectories)
            != self.development.trajectory_ids
        ):
            raise ValueError("QC records differ from development identities")
        if self.config.development != self.development.config:
            raise ValueError("QC configuration differs from development selection")
        _validate_report_rows(self)
        return self


def _validate_report_hashes(report: QCReport) -> None:
    expected = (
        metadata_hash(report.development.split),
        metadata_hash(report.development),
        metadata_hash(report.config),
    )
    if (report.split_hash, report.development_hash, report.config_hash) != expected:
        raise ValueError("QC provenance hashes differ from their payloads")


def _validate_report_rows(report: QCReport) -> None:
    records = _qc_records(report.development, report.config)
    names = tuple(
        feature.feature_id
        for feature in report.development.split.registry.dataset.features
    )
    for row, record in zip(report.trajectories, records, strict=True):
        _validate_trajectory_row(row, record, names, report.config)
    expected = _all_feasibility(records, report.config.windows)
    if report.feasibility != expected:
        raise ValueError("QC feasibility differs from declared source/grid")


def _validate_trajectory_row(
    row: TrajectoryQC,
    record: TrajectoryManifest,
    names: tuple[str, ...],
    config: QCConfig,
) -> None:
    if tuple(feature.feature_id for feature in row.features) != names:
        raise ValueError("QC feature order differs from canonical definitions")
    _validate_correlation_dimensions(row.cross_correlation, len(names))
    for feature in row.features:
        _validate_feature_row(feature, record.frame_count, config.max_lag_frames)


def _validate_correlation_dimensions(
    matrix: tuple[tuple[float | None, ...], ...], size: int
) -> None:
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("QC cross-correlation matrix has incorrect dimensions")


def _validate_feature_row(feature: FeatureQC, frames: int, max_lag: int) -> None:
    if (len(feature.quantiles), feature.increment_count) != (
        len(QUANTILES),
        frames - 1,
    ):
        raise ValueError("QC summary dimensions differ from declared samples")
    if feature.missing_count != 0:
        raise ValueError("canonical published series cannot contain missing values")
    if feature.acf is not None and len(feature.acf) != min(max_lag, frames - 1) + 1:
        raise ValueError("QC ACF length differs from declared lag limit")


def autocorrelation(values: FloatArray, max_lag: int) -> FloatArray | None:
    """Return globally demeaned, fixed-denominator sample ACF; constants are null.

    This is sample-index dependence, not evidence of verified physical spacing.
    """
    centered = values - values.mean()
    energy = float(centered @ centered)
    if energy == 0:
        return None
    # ponytail: O(T * max_lag) for short MD exports; use FFT for long-series QC.
    return np.array(
        [
            float(centered[: len(values) - lag] @ centered[lag:]) / energy
            for lag in range(min(max_lag, len(values) - 1) + 1)
        ],
        dtype=np.float64,
    )


def _decay(acf: FloatArray | None) -> tuple[int | None, bool]:
    if acf is None:
        return None, False
    crossings = np.flatnonzero(acf[1:] <= DECAY_LEVEL)
    if not len(crossings):
        return None, True
    return int(crossings[0]) + 1, False


def _feature_qc(
    name: str, values: FloatArray, cutoff: float, config: QCConfig
) -> FeatureQC:
    std = float(values.std())
    acf = autocorrelation(values, config.max_lag_frames)
    decay, censored = _decay(acf)
    frame = np.arange(len(values), dtype=np.float64)
    frame -= frame.mean()
    drift = float(frame @ (values - values.mean()) / (frame @ frame))
    shift = _half_shift(values, std)
    increments = np.diff(values)
    jump_count = int(
        np.count_nonzero(
            np.abs(increments - increments.mean())
            > config.event_sigma * increments.std()
        )
    )
    return FeatureQC(
        feature_id=name,
        missing_count=0,
        quantiles=tuple(np.quantile(values, QUANTILES)),
        mean=float(values.mean()),
        variance=std * std,
        std=std,
        constant=std == 0,
        near_constant=std <= cutoff,
        acf=None if acf is None else tuple(acf),
        decay_frames=decay,
        decay_censored=censored,
        linear_drift_per_frame=drift,
        half_mean_shift_sd=shift,
        jump_count=jump_count,
        increment_count=len(increments),
    )


def _half_shift(values: FloatArray, std: float) -> float | None:
    if std == 0:
        return None
    midpoint = len(values) // 2
    return float((values[midpoint:].mean() - values[:midpoint].mean()) / std)


def _cross_correlation(values: FloatArray) -> tuple[tuple[float | None, ...], ...]:
    centered = values - values.mean(axis=0)
    norms = np.sqrt(np.square(centered).sum(axis=0))
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator != 0,
    )
    return tuple(
        tuple(None if np.isnan(value) else float(value) for value in row)
        for row in correlation
    )


def _analyze_record(
    record: TrajectoryManifest,
    development: DevelopmentManifest,
    config: QCConfig,
    loader: SeriesLoader,
) -> TrajectoryQC:
    if record.frame_count < 2:
        raise DataContractError("QC requires at least two samples per trajectory")
    names = tuple(
        feature.feature_id for feature in development.split.registry.dataset.features
    )
    values = _load_values(record, development, names, loader)
    return TrajectoryQC(
        trajectory_id=record.trajectory_id,
        features=tuple(
            _feature_qc(name, values[:, index], config.near_constant_std[name], config)
            for index, name in enumerate(names)
        ),
        cross_correlation=_cross_correlation(values),
    )


def _load_values(
    record: TrajectoryManifest,
    development: DevelopmentManifest,
    names: tuple[str, ...],
    loader: SeriesLoader,
) -> FloatArray:
    table = loader(record)
    registry = validate_series(table)
    if (
        registry.trajectories != (record,)
        or registry.dataset != development.split.registry.dataset
    ):
        raise DataContractError("QC series differs from canonical source metadata")
    return np.column_stack([table[name].to_numpy() for name in names])


def _feasibility(
    record: TrajectoryManifest, grid: WindowConfig
) -> list[WindowFeasibility]:
    result = []
    for context in sorted(grid.contexts):
        for horizon in sorted(grid.horizons):
            cell = WindowConfig.model_validate(
                grid.model_dump() | {"contexts": (context,), "horizons": (horizon,)}
            )
            result.append(_cell_feasibility(record, cell, context, horizon))
    return result


def _cell_feasibility(
    record: TrajectoryManifest, cell: WindowConfig, context: float, horizon: float
) -> WindowFeasibility:
    try:
        context_frames, horizon_frames = window_lengths(record, cell)[0]
    except DataContractError as error:
        return WindowFeasibility(
            trajectory_id=record.trajectory_id,
            context=context,
            horizon=horizon,
            window_count=0,
            reason=str(error),
        )
    count = (
        record.frame_count - context_frames - horizon_frames
    ) // cell.stride_frames + 1
    return WindowFeasibility(
        trajectory_id=record.trajectory_id,
        context=context,
        horizon=horizon,
        context_frames=context_frames,
        horizon_frames=horizon_frames,
        window_count=count,
    )


def analyze_development(
    split: SplitManifest,
    config: QCConfig,
    loader: SeriesLoader,
    *,
    code_commit: str,
    lockfile_hash: str,
) -> QCReport:
    """Analyze only selected official training groups, one trajectory at a time."""
    config = QCConfig.model_validate_json(config.model_dump_json())
    development = select_development(split, config.development)
    records = _qc_records(development, config)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rows = tuple(
                _analyze_record(record, development, config, loader)
                for record in records
            )
    except FloatingPointError as error:
        raise DataContractError(
            "QC arithmetic overflow; check feature magnitudes"
        ) from error
    return QCReport(
        development=development,
        config=config,
        code_commit=code_commit,
        lockfile_hash=lockfile_hash,
        split_hash=metadata_hash(development.split),
        development_hash=metadata_hash(development),
        config_hash=metadata_hash(config),
        trajectories=rows,
        feasibility=_all_feasibility(records, config.windows),
    )


def _all_feasibility(
    records: list[TrajectoryManifest], grid: WindowConfig
) -> tuple[WindowFeasibility, ...]:
    return tuple(row for record in records for row in _feasibility(record, grid))


def _qc_records(
    development: DevelopmentManifest, config: QCConfig
) -> list[TrajectoryManifest]:
    features = {
        feature.feature_id for feature in development.split.registry.dataset.features
    }
    if set(config.near_constant_std) != features:
        raise DataContractError(
            "near-constant thresholds must cover every canonical feature"
        )
    selected = set(development.trajectory_ids)
    records = [
        record
        for record in development.split.registry.trajectories
        if record.trajectory_id in selected
    ]
    records.sort(key=lambda record: record.trajectory_id)
    return records
