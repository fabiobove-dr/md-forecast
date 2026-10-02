"""One-trajectory Arrow/Parquet storage with vectorized numerical validation."""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from md_forecast.core.constants import (
    PS_PER_NS,
    SERIES_METADATA_KEY,
    TIME_ATOL,
    TIME_COLUMN,
    TIME_RTOL,
    TimeUnit,
)
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.registry import Registry, atomic_output
from md_forecast.data.schemas import DatasetConfig, TrajectoryManifest

type FloatArray = npt.NDArray[np.float64]


def to_arrow(
    manifest: TrajectoryManifest,
    dataset: DatasetConfig,
    time: FloatArray,
    values: FloatArray,
) -> pa.Table:
    """Normalize an (n_frames, n_features) matrix without per-row models."""
    registry = Registry(dataset=dataset, trajectories=(manifest,))
    expected_shape = (manifest.frame_count, len(dataset.features))
    if values.shape != expected_shape or time.shape != (manifest.frame_count,):
        raise DataContractError(
            f"expected time ({manifest.frame_count},) and values {expected_shape}"
        )
    columns = {TIME_COLUMN: pa.array(time, type=pa.float64())}
    columns.update(
        {
            feature.feature_id: pa.array(values[:, index], type=pa.float64())
            for index, feature in enumerate(dataset.features)
        }
    )
    table = pa.table(columns).replace_schema_metadata(
        {SERIES_METADATA_KEY: registry.model_dump_json().encode("utf-8")}
    )
    validate_series(table)
    return table


def series_metadata(table: pa.Table) -> Registry:
    """Decode one versioned metadata boundary for an entire trajectory."""
    try:
        encoded = (table.schema.metadata or {})[SERIES_METADATA_KEY]
        registry = Registry.model_validate_json(encoded)
    except (KeyError, ValueError) as error:
        raise DataContractError(
            "series requires valid canonical schema metadata"
        ) from error
    if len(registry.trajectories) != 1:
        raise DataContractError("a series must describe exactly one trajectory")
    return registry


def validate_series(table: pa.Table) -> Registry:
    """Reject malformed tables, nonfinite observables, and inconsistent times."""
    registry = series_metadata(table)
    manifest = registry.trajectories[0]
    names = [
        TIME_COLUMN,
        *(feature.feature_id for feature in registry.dataset.features),
    ]
    if table.column_names != names or table.num_rows != manifest.frame_count:
        raise DataContractError("series columns or frame count differ from metadata")
    _validate_columns(table)
    time = cast(FloatArray, table[TIME_COLUMN].to_numpy())
    _validate_time(time, manifest)
    return registry


def _validate_columns(table: pa.Table) -> None:
    for column in table.columns:
        if column.type != pa.float64() or column.null_count:
            raise DataContractError("series columns must be non-null float64")
        if not np.isfinite(column.to_numpy()).all():
            raise DataContractError(
                "series values must be finite; handle missing data before publication"
            )


def _validate_time(time: FloatArray, manifest: TrajectoryManifest) -> None:
    if time[0] != 0 or not np.all(np.diff(time) > 0):
        raise DataContractError("time must start at zero and be strictly increasing")
    if manifest.time_unit == TimeUnit.FRAME:
        if not np.array_equal(time, np.arange(manifest.frame_count)):
            raise DataContractError("frame axes must contain consecutive frame indices")
    else:
        _validate_physical_time(time, manifest)


def _validate_physical_time(time: FloatArray, manifest: TrajectoryManifest) -> None:
    factor = PS_PER_NS if manifest.time_unit == TimeUnit.PS else 1.0
    if not np.isclose(
        time[-1] / factor,
        cast(float, manifest.duration_ns),
        rtol=TIME_RTOL,
        atol=TIME_ATOL,
    ):
        raise DataContractError("time axis differs from duration_ns")
    if manifest.frame_interval_ps is not None:
        interval = manifest.frame_interval_ps * factor / PS_PER_NS
        if not np.allclose(np.diff(time), interval, rtol=TIME_RTOL, atol=TIME_ATOL):
            raise DataContractError("time axis differs from frame_interval_ps")


def write_series(path: Path, table: pa.Table) -> None:
    """Validate then atomically publish one trajectory, with embedded metadata."""
    validate_series(table)
    try:
        with atomic_output(path) as temporary:
            pq.write_table(table, temporary)
    except (OSError, pa.ArrowException) as error:
        raise DataContractError(f"cannot write series {path}: {error}") from error


def read_series(path: Path) -> pa.Table:
    """Read and validate one trajectory, never the full dataset at once."""
    try:
        table = cast(pa.Table, pq.ParquetFile(path).read())
    except (OSError, pa.ArrowException) as error:
        raise DataContractError(f"cannot read series {path}: {error}") from error
    validate_series(table)
    return table
