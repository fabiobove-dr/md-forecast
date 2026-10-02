"""Frozen single-partition grid cells, paired adapters and group-level scoring."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Literal, Self, cast

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, Split
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch, ForecastSpec, validate_array
from md_forecast.data.predictions import QuantileForecast
from md_forecast.data.preprocessing import ScalerMetadata
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, SchemaVersion
from md_forecast.data.series import FloatArray
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, WindowIndex, iter_windows
from md_forecast.evaluation.baselines import _reference_key
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    MetricId,
    MetricKey,
    group_interval,
    interval_status,
    point_losses,
    quantile_losses,
)
from md_forecast.models.base import ForecastModel, ModelConfig

type Totals = dict[str, dict[str, dict[MetricKey, FloatArray]]]


class CellManifest(BoundaryModel):
    """Full frozen inputs, including trained-state identities, not measured timings."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    spec: ForecastSpec
    split: SplitManifest
    grid: WindowConfig
    partition: Split
    config: BenchmarkConfig
    scaler: ScalerMetadata
    models: Annotated[tuple[ModelConfig, ...], Field(min_length=1)]
    model_artifacts: tuple[ArtifactHash, ...]
    window_selection: Literal["all", "first-per-trajectory"] = "all"
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    lockfile_hash: ArtifactHash
    hardware: Annotated[str, Field(min_length=1)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Reject cross-source normalization, ambiguous models and incomplete policy."""
        _validate_manifest_links(self)
        keys = tuple(metadata_hash(config) for config in self.models)
        if len(set(keys)) != len(keys) or len(self.model_artifacts) != len(keys):
            raise ValueError(
                "models/artifacts must be paired and configurations unique"
            )
        _reference_key(self.models, keys)
        _validate_family(self, len(keys))
        return self


def _validate_family(manifest: CellManifest, model_count: int) -> None:
    family = (
        (model_count - 1)
        * len(manifest.spec.feature_ids)
        * (manifest.spec.horizon_frames + 1)
    )
    if manifest.config.comparison_family_size < max(1, family):
        raise ValueError("declared comparison family is smaller than this cell")


def _validate_manifest_links(manifest: CellManifest) -> None:
    _validate_scaler_links(manifest)
    _validate_source_links(manifest)
    _validate_training_regions(manifest)


def _validate_scaler_links(manifest: CellManifest) -> None:
    spec, scaler = manifest.spec, manifest.scaler
    expected = (
        spec.dataset,
        spec.feature_ids,
        spec.split_hash,
        spec.window_config_hash,
    )
    actual = (
        scaler.dataset,
        scaler.config.feature_ids,
        scaler.split_hash,
        scaler.window_config_hash,
    )
    if actual != expected or scaler.config.scope != "training-contexts":
        raise ValueError(
            "benchmark scaling must match this cell and use TRAIN contexts"
        )


def _validate_source_links(manifest: CellManifest) -> None:
    spec = manifest.spec
    if (metadata_hash(manifest.split), metadata_hash(manifest.grid)) != (
        spec.split_hash,
        spec.window_config_hash,
    ) or manifest.split.registry.dataset != spec.dataset:
        raise ValueError("manifest split/grid/source linkage differs")
    if len(manifest.grid.contexts) != 1 or len(manifest.grid.horizons) != 1:
        raise ValueError("evaluate one explicit context/horizon cell at a time")


def _validate_training_regions(manifest: CellManifest) -> None:
    train_ids = {
        a.trajectory_id for a in manifest.split.assignments if a.split == Split.TRAIN
    }
    if (
        not {region.trajectory_id for region in manifest.scaler.fit_regions}
        <= train_ids
    ):
        raise ValueError("normalization includes held-out trajectories")


class ModelRuntime(BoundaryModel):
    """Operational metadata excluded from reproducible scientific artifact identity."""

    config_hash: ArtifactHash
    seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    peak_allocated_bytes: Annotated[int, Field(strict=True, ge=0)] | None
    peak_reserved_bytes: Annotated[int, Field(strict=True, ge=0)] | None


class GridManifest(BoundaryModel):
    """Predeclared complete context/horizon family; never pool different partitions."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    cells: Annotated[tuple[CellManifest, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Require unique cells, identical analysis policy and grid-wide correction."""
        _validate_grid_cells(self.cells)
        return self


def _grid_signature(cell: CellManifest) -> tuple[object, ...]:
    return (
        cell.spec.dataset,
        cell.spec.feature_ids,
        cell.spec.split_hash,
        cell.partition,
        cell.config,
        cell.models,
        cell.window_selection,
        cell.code_commit,
        cell.lockfile_hash,
        cell.hardware,
        cell.batch_size,
    )


def _validate_grid_cells(cells: tuple[CellManifest, ...]) -> None:
    signature = _grid_signature(cells[0])
    dimensions = set()
    for cell in cells:
        if _grid_signature(cell) != signature:
            raise ValueError(
                "grid cells cannot mix sources, partitions, models or policies"
            )
        dimension = (cell.spec.context_frames, cell.spec.horizon_frames)
        if dimension in dimensions:
            raise ValueError("duplicate grid cell")
        dimensions.add(dimension)
    _validate_grid_family(cells)


def _validate_grid_family(cells: tuple[CellManifest, ...]) -> None:
    first = cells[0]
    size = (
        (len(first.models) - 1)
        * len(first.spec.feature_ids)
        * sum(c.spec.horizon_frames + 1 for c in cells)
    )
    if first.config.comparison_family_size < max(1, size):
        raise ValueError("comparison correction family must cover the entire grid")


@dataclass(frozen=True)
class CellResult:
    """Efficient Arrow tables with a Pydantic persistence boundary in report.py."""

    manifest: CellManifest
    predictions: pa.Table
    metrics: pa.Table
    comparisons: pa.Table
    runtimes: tuple[ModelRuntime, ...]
    window_count: int
    trajectory_count: int
    system_count: int
    group_count: int


def _expected_windows(manifest: CellManifest) -> set[WindowIndex]:
    expected, trajectories = set(), set()
    for index in _selected_windows(manifest):
        if (index.context_frames, index.horizon_frames) != (
            manifest.spec.context_frames,
            manifest.spec.horizon_frames,
        ):
            raise ForecastError("cell contains inconsistent frame dimensions")
        expected.add(index)
        trajectories.add(index.trajectory_id)
        rows = (
            len(expected)
            * manifest.spec.horizon_frames
            * len(manifest.spec.feature_ids)
            * len(manifest.models)
        )
        if rows > manifest.config.max_prediction_rows:
            raise ForecastError("prediction table exceeds declared row budget")
    if not expected:
        raise ForecastError("requested partition/cell contains no windows")
    return expected


def _selected_windows(manifest: CellManifest) -> Iterable[WindowIndex]:
    trajectories: set[str] = set()
    for index in iter_windows(manifest.split, manifest.grid, manifest.partition):
        if (
            manifest.window_selection == "first-per-trajectory"
            and index.trajectory_id in trajectories
        ):
            continue
        trajectories.add(index.trajectory_id)
        yield index


def _validate_models(
    models: tuple[ForecastModel, ...], manifest: CellManifest
) -> tuple[str, ...]:
    if tuple(model.config for model in models) != manifest.models:
        raise ForecastError("adapters differ from frozen model configurations")
    if tuple(model.artifact_hash for model in models) != manifest.model_artifacts:
        raise ForecastError("adapter weights/settings differ from frozen artifacts")
    return tuple(map(metadata_hash, manifest.models))


def evaluate_cell(
    models: tuple[ForecastModel, ...],
    pairs: Iterable[tuple[ForecastBatch, FloatArray | None]],
    manifest: CellManifest,
) -> CellResult:
    """Evaluate exactly the frozen windows; labels are never passed to an adapter."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _evaluate_cell(models, pairs, manifest)
    except FloatingPointError as error:
        raise ForecastError("benchmark aggregation overflowed") from error


def _evaluate_cell(
    models: tuple[ForecastModel, ...],
    pairs: Iterable[tuple[ForecastBatch, FloatArray | None]],
    manifest: CellManifest,
) -> CellResult:
    manifest = CellManifest.model_validate_json(manifest.model_dump_json())
    keys = _validate_models(models, manifest)
    expected = _expected_windows(manifest)
    totals: Totals = {key: {} for key in keys}
    counts: dict[str, int] = {}
    identities: dict[str, tuple[str, str]] = {}
    tables: list[pa.Table] = []
    runtimes: list[ModelRuntime] = []
    seen: set[WindowIndex] = set()
    for batch, targets in pairs:
        _check_batch(batch, targets, manifest, expected, seen)
        assert targets is not None
        _count_batch(batch, counts, identities)
        _evaluate_models(
            models, keys, batch, targets, manifest, totals, tables, runtimes
        )
    if seen != expected:
        raise ForecastError(
            "evaluation is missing frozen windows; no silent subsampling"
        )
    metrics, groups = _metric_rows(totals, counts, identities, manifest)
    comparisons = _comparisons(groups, keys, manifest)
    systems, groups_count = _identity_counts(identities)
    return CellResult(
        manifest,
        pa.concat_tables(tables).sort_by(
            [
                ("model_hash", "ascending"),
                ("trajectory_id", "ascending"),
                ("start", "ascending"),
                ("step", "ascending"),
                ("feature_id", "ascending"),
            ]
        ),
        pa.Table.from_pylist(metrics),
        pa.Table.from_pylist(comparisons),
        tuple(runtimes),
        len(seen),
        len(counts),
        systems,
        groups_count,
    )


def evaluate_grid(
    models: tuple[tuple[ForecastModel, ...], ...],
    pairs: tuple[Iterable[tuple[ForecastBatch, FloatArray | None]], ...],
    manifest: GridManifest,
) -> tuple[CellResult, ...]:
    """Evaluate every frozen cell, with cell-specific learned weights allowed."""
    snapshot = GridManifest.model_validate_json(manifest.model_dump_json())
    if len(models) != len(snapshot.cells) or len(pairs) != len(snapshot.cells):
        raise ForecastError("grid requires adapters and data for every frozen cell")
    return tuple(
        evaluate_cell(m, p, c)
        for m, p, c in zip(models, pairs, snapshot.cells, strict=True)
    )


def _identity_counts(identities: dict[str, tuple[str, str]]) -> tuple[int, int]:
    return len({s for s, _ in identities.values()}), len(
        {g for _, g in identities.values()}
    )


def _evaluate_models(
    models: tuple[ForecastModel, ...],
    keys: tuple[str, ...],
    batch: ForecastBatch,
    targets: FloatArray,
    manifest: CellManifest,
    totals: Totals,
    tables: list[pa.Table],
    runtimes: list[ModelRuntime],
) -> None:
    for model, key in zip(models, keys, strict=True):
        points, probabilistic, runtime = _predict(model, key, batch)
        losses = point_losses(points, targets, np.asarray(manifest.scaler.scale))
        _probabilistic_losses(losses, probabilistic, targets, manifest.config)
        _accumulate(totals[key], batch, losses)
        tables.append(_prediction_table(key, batch, targets, points, probabilistic))
        runtimes.append(runtime)


def _check_batch(
    batch: ForecastBatch,
    targets: FloatArray | None,
    manifest: CellManifest,
    expected: set[WindowIndex],
    seen: set[WindowIndex],
) -> None:
    if batch.spec != manifest.spec or batch.indices[0].split != manifest.partition:
        raise ForecastError("benchmark cannot mix sources/cells/partitions")
    if targets is None:
        raise ForecastError("benchmark requires separate target labels")
    if len(batch.indices) > manifest.batch_size:
        raise ForecastError("batch exceeds frozen inference batch size")
    validate_array(
        targets,
        (
            len(batch.indices),
            manifest.spec.horizon_frames,
            len(manifest.spec.feature_ids),
        ),
    )
    _check_indices(batch.indices, expected, seen)


def _check_indices(
    indices: tuple[WindowIndex, ...], expected: set[WindowIndex], seen: set[WindowIndex]
) -> None:
    for index in indices:
        if index not in expected or index in seen:
            raise ForecastError("unexpected, forged or duplicate evaluation window")
        seen.add(index)


def _count_batch(
    batch: ForecastBatch, counts: dict[str, int], identities: dict[str, tuple[str, str]]
) -> None:
    for index in batch.indices:
        counts[index.trajectory_id] = counts.get(index.trajectory_id, 0) + 1
        identities[index.trajectory_id] = (index.system_id, index.group_id)


def _predict(
    model: ForecastModel, key: str, batch: ForecastBatch
) -> tuple[FloatArray, QuantileForecast | None, ModelRuntime]:
    start = perf_counter()
    forecast = getattr(model, "forecast", None)
    result = None
    if callable(forecast):
        result = cast(Callable[[ForecastBatch], QuantileForecast], forecast)(batch)
        _validate_output(result, batch)
        points = result.median
    else:
        points = model.predict(batch)
    validate_array(
        points,
        (len(batch.indices), batch.spec.horizon_frames, len(batch.spec.feature_ids)),
    )
    return (
        points,
        result,
        _model_runtime(key, start, result),
    )


def _validate_output(result: QuantileForecast, batch: ForecastBatch) -> None:
    if (
        not isinstance(result, QuantileForecast)
        or result.spec != batch.spec
        or result.indices != batch.indices
    ):
        raise ForecastError(
            "probabilistic adapter returned incompatible canonical output"
        )


def _model_runtime(
    key: str, start: float, result: QuantileForecast | None
) -> ModelRuntime:
    return ModelRuntime(
        config_hash=key,
        seconds=perf_counter() - start,
        peak_allocated_bytes=None
        if result is None
        else result.runtime.peak_allocated_bytes,
        peak_reserved_bytes=None
        if result is None
        else result.runtime.peak_reserved_bytes,
    )


def _probabilistic_losses(
    losses: dict[MetricKey, FloatArray],
    result: QuantileForecast | None,
    targets: FloatArray,
    config: BenchmarkConfig,
) -> None:
    if result is not None:
        losses.update(
            quantile_losses(
                result.values, targets, result.quantile_levels, config.intervals
            )
        )


def _accumulate(
    totals: dict[str, dict[MetricKey, FloatArray]],
    batch: ForecastBatch,
    losses: dict[MetricKey, FloatArray],
) -> None:
    for key, values in losses.items():
        steps = np.concatenate((values.mean(axis=1, keepdims=True), values), axis=1)
        for index, value in zip(batch.indices, steps, strict=True):
            entry = totals.setdefault(index.trajectory_id, {})
            entry[key] = entry.get(key, np.zeros_like(value)) + value


def _prediction_table(
    key: str,
    batch: ForecastBatch,
    targets: FloatArray,
    points: FloatArray,
    result: QuantileForecast | None,
) -> pa.Table:
    b, h, f = targets.shape
    levels = () if result is None else result.quantile_levels
    quantiles = (
        [[] for _ in range(b * h * f)]
        if result is None
        else result.values.reshape(-1, len(levels)).tolist()
    )
    identity = _prediction_identity(batch, h * f)
    return pa.table(
        {
            "model_hash": [key] * (b * h * f),
            **identity,
            "step": np.tile(np.repeat(np.arange(1, h + 1), f), b),
            "target_frame": np.asarray(identity["start"])
            + batch.spec.context_frames
            + np.tile(np.repeat(np.arange(h), f), b),
            "feature_id": np.tile(batch.spec.feature_ids, b * h),
            "target": targets.ravel(),
            "point": points.ravel(),
            "quantile_levels": pa.array(
                [list(levels)] * (b * h * f), type=pa.list_(pa.float64())
            ),
            "quantiles": pa.array(quantiles, type=pa.list_(pa.float64())),
        }
    )


def _prediction_identity(batch: ForecastBatch, repeats: int) -> dict[str, object]:
    return {
        "trajectory_id": np.repeat([i.trajectory_id for i in batch.indices], repeats),
        "system_id": np.repeat([i.system_id for i in batch.indices], repeats),
        "group_id": np.repeat([i.group_id for i in batch.indices], repeats),
        "start": np.repeat([i.start for i in batch.indices], repeats),
    }


def _trajectory_means(
    totals: dict[str, dict[MetricKey, FloatArray]], counts: dict[str, int]
) -> dict[str, dict[MetricKey, FloatArray]]:
    result = {}
    for trajectory, metrics in sorted(totals.items()):
        values = {key: value / counts[trajectory] for key, value in metrics.items()}
        for key in values:
            if key[0] == MetricId.RMSE:
                values[key] = np.sqrt(values[key])
        result[trajectory] = values
    return result


def _merge_entities(
    entities: dict[str, dict[MetricKey, FloatArray]], labels: dict[str, str]
) -> dict[str, dict[MetricKey, FloatArray]]:
    groups: dict[str, list[dict[MetricKey, FloatArray]]] = {}
    for identity, metrics in entities.items():
        groups.setdefault(labels[identity], []).append(metrics)
    return {
        group: {
            key: np.stack([entry[key] for entry in entries]).mean(axis=0)
            for key in entries[0]
        }
        for group, entries in sorted(groups.items())
    }


def _metric_rows(
    totals: Totals,
    counts: dict[str, int],
    identities: dict[str, tuple[str, str]],
    manifest: CellManifest,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[MetricKey, FloatArray]]]]:
    rows: list[dict[str, object]] = []
    all_groups = {}
    for model, values in totals.items():
        trajectories = _trajectory_means(values, counts)
        levels = _aggregation_levels(trajectories, identities)
        groups = levels["group"]
        for level, entities in levels.items():
            rows.extend(_entity_rows(model, level, entities, groups, manifest))
        all_groups[model] = groups
    return rows, all_groups


def _aggregation_levels(
    trajectories: dict[str, dict[MetricKey, FloatArray]],
    identities: dict[str, tuple[str, str]],
) -> dict[str, dict[str, dict[MetricKey, FloatArray]]]:
    systems = _merge_entities(trajectories, {t: s for t, (s, _) in identities.items()})
    groups = _merge_entities(systems, {s: g for s, g in identities.values()})
    aggregate = _merge_entities(groups, dict.fromkeys(groups, "all"))
    return dict(
        trajectory=trajectories, system=systems, group=groups, aggregate=aggregate
    )


def _entity_rows(
    model: str,
    level: str,
    entities: dict[str, dict[MetricKey, FloatArray]],
    groups: dict[str, dict[MetricKey, FloatArray]],
    manifest: CellManifest,
) -> list[dict[str, object]]:
    rows = []
    for identity, metrics in entities.items():
        for (metric, lower, upper), values in metrics.items():
            for step, feature_index in np.ndindex(values.shape):
                bounds = _metric_interval(
                    level,
                    groups,
                    (metric, lower, upper),
                    step,
                    feature_index,
                    manifest.config,
                )
                rows.append(
                    dict(
                        model_hash=model,
                        level=level,
                        identity=identity,
                        feature_id=manifest.spec.feature_ids[feature_index],
                        step=step,
                        metric=metric.value,
                        lower_quantile=lower,
                        upper_quantile=upper,
                        value=float(values[step, feature_index]),
                        ci_lower=bounds[0],
                        ci_upper=bounds[1],
                    )
                )
    return rows


def _metric_interval(
    level: str,
    groups: dict[str, dict[MetricKey, FloatArray]],
    key: MetricKey,
    step: int,
    feature: int,
    config: BenchmarkConfig,
) -> tuple[float | None, float | None]:
    if level != "aggregate":
        return None, None
    return group_interval(
        np.array([g[key][step, feature] for g in groups.values()]), config
    )


def _comparisons(
    groups: dict[str, dict[str, dict[MetricKey, FloatArray]]],
    keys: tuple[str, ...],
    manifest: CellManifest,
) -> list[dict[str, object]]:
    reference = _reference_key(manifest.models, keys)
    key: MetricKey = (MetricId.MAE, None, None)
    base = np.stack([value[key] for value in groups[reference].values()])
    rows = []
    for model, model_groups in groups.items():
        values = np.stack([value[key] for value in model_groups.values()])
        for step, feature in np.ndindex(base.shape[1:]):
            rows.append(
                _comparison_row(
                    model,
                    reference,
                    values[:, step, feature],
                    base[:, step, feature],
                    step,
                    feature,
                    manifest,
                )
            )
    return rows


def _comparison_row(
    model: str,
    reference: str,
    values: FloatArray,
    base: FloatArray,
    step: int,
    feature: int,
    manifest: CellManifest,
) -> dict[str, object]:
    differences = values - base
    bounds = group_interval(differences, manifest.config, corrected=True)
    baseline = float(base.mean())
    ratio = None if baseline == 0 else float(values.mean()) / baseline
    return dict(
        model_hash=model,
        reference_hash=reference,
        feature_id=manifest.spec.feature_ids[feature],
        step=step,
        mae_difference=float(differences.mean()),
        ratio_to_persistence=ratio,
        relative_improvement=None if ratio is None else 1 - ratio,
        ci_lower=bounds[0],
        ci_upper=bounds[1],
        independent_groups=len(base),
        uncertainty_status=interval_status(len(base), manifest.config, corrected=True),
    )
