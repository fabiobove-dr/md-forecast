"""Verified saved forecasts to a standalone HTML overview; no model inference."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from md_forecast.core.constants import SamplingStatus
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata
from md_forecast.data.registry import atomic_output
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, DatasetConfig
from md_forecast.data.splits import SplitManifest
from md_forecast.evaluation.benchmark import CellManifest, _expected_windows
from md_forecast.evaluation.mvp import read_mvp
from md_forecast.evaluation.report import _horizon_axis, _interval_axis, read_benchmark
from md_forecast.models.residuals import ResidualState


class OverviewSource(BoundaryModel):
    """Explicit role and pinned saved result, separate from scientific success."""

    title: str = Field(min_length=1)
    role: str = Field(min_length=1)
    note: str = Field(min_length=1)
    kind: Literal["benchmark", "probability"]
    root: Path
    expected_hash: ArtifactHash
    dataset: DatasetConfig | None = None
    model_labels: dict[ArtifactHash, str] = Field(default_factory=dict)
    statistical_references: dict[str, ArtifactHash] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_definitions(self) -> Self:
        """Columnar follow-up exports need their admitted feature definitions."""
        if self.kind == "probability" and self.dataset is None:
            raise ValueError("probability sources require dataset definitions")
        return self


@dataclass(frozen=True, slots=True)
class SavedPoint:
    """Small immutable projection after vectorized saved-table validation."""

    trajectory_id: str
    group_id: str
    start: int
    lead: int
    point: float
    target: float
    origin: float | None = None
    quantiles: tuple[float, ...] = ()
    levels: tuple[float, ...] = ()


def _point(data: dict[str, Any]) -> SavedPoint:
    levels, quantiles = tuple(data["levels"]), tuple(data["quantiles"])
    _validate_quantiles(levels, quantiles, data["point"])
    return SavedPoint(**(data | {"levels": levels, "quantiles": quantiles}))


def _validate_quantiles(
    levels: tuple[float, ...], values: tuple[float, ...], point: float
) -> None:
    if len(levels) != len(values) or levels != tuple(sorted(set(levels))):
        raise DataContractError("saved quantile levels/values differ or repeat")
    if any(not 0 < level < 1 for level in levels):
        raise DataContractError("saved quantile levels outside (0,1)")
    _validate_median(levels, values, point)


def _validate_median(
    levels: tuple[float, ...], values: tuple[float, ...], point: float
) -> None:
    if not np.isfinite(values).all() or values != tuple(sorted(values)):
        raise DataContractError("saved quantiles are nonfinite or cross")
    if 0.5 in levels and point != values[levels.index(0.5)]:
        raise DataContractError("saved median and point differ")


class OverviewWindow(BoundaryModel):
    """Aligned complete window; reference origin is optional in legacy bundles."""

    trajectory_id: str
    group_id: str
    start: int
    origin: float | None
    target: tuple[float, ...]
    points: dict[str, tuple[float, ...]]
    bands: dict[str, tuple[tuple[float, float], ...]]


class OverviewPanel(BoundaryModel):
    """One observable/cell with saved scores and exact matched windows."""

    title: str
    role: str
    note: str
    source_hash: str
    feature_id: str
    unit: str
    definition: str
    context_frames: int
    horizon_frames: int
    lead_interval: float = 1.0
    lead_unit: str = "frames"
    statistical_reference: str | None = None
    models: dict[str, str]
    windows: tuple[OverviewWindow, ...]
    metrics: list[dict[str, Any]]
    evidence: dict[str, object]


class OverviewSynthesis(BoundaryModel):
    """Pinned confirmation inference, separate from saved prediction tables."""

    path: Path
    expected_hash: ArtifactHash


class OverviewConfig(BoundaryModel):
    """Bounded offline inputs; structure embedding is optional and explicit."""

    title: str = "MD Forecast Performance Overview"
    conclusion: str = Field(min_length=1)
    default_panel: int = Field(default=0, ge=0, strict=True)
    sources: Annotated[tuple[OverviewSource, ...], Field(min_length=1)]
    max_rows: int = Field(default=2_000_000, gt=0, strict=True)
    max_input_bytes: int = Field(default=1_073_741_824, gt=0, strict=True)
    structure_config: Path | None = None
    mvp_root: Path | None = None
    confirmation_summary: OverviewSynthesis | None = None


def file_hash(path: Path) -> str:
    """Stream SHA-256 without loading large saved tables."""
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def confined(root: Path, relative: str) -> Path:
    """Reject bundle paths escaping their declared root, including symlinks."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DataContractError("saved bundle path escapes root")
    return path


def _table(path: Path, config: OverviewConfig) -> list[dict[str, Any]]:
    metadata = pq.read_metadata(path)
    if metadata.num_rows > config.max_rows:
        raise DataContractError("saved table exceeds overview max_rows")
    rows = cast(list[dict[str, Any]], pq.read_table(path).to_pylist())
    if not rows:
        raise DataContractError("saved forecast or metric table is empty")
    return rows


def _match(
    tables: dict[str, list[SavedPoint]], horizon: int
) -> tuple[OverviewWindow, ...]:
    _validate_matching(tables)
    grouped: dict[tuple[str, int], dict[str, list[SavedPoint]]] = defaultdict(dict)
    for model, points in tables.items():
        for p in points:
            grouped[(p.trajectory_id, p.start)].setdefault(model, []).append(p)
    return tuple(_window(key, rows, horizon) for key, rows in sorted(grouped.items()))


def _validate_matching(tables: dict[str, list[SavedPoint]]) -> None:
    identities = None
    for points in tables.values():
        _validate_points(points)
        current = _identities(points)
        if identities is not None and identities != current:
            raise DataContractError(
                "saved models have unmatched windows or future labels"
            )
        identities = current


def _identities(points: list[SavedPoint]) -> set[tuple[object, ...]]:
    keys = [(p.trajectory_id, p.start, p.lead) for p in points]
    if len(set(keys)) != len(keys):
        raise DataContractError("duplicate saved window/lead")
    return {
        (p.trajectory_id, p.start, p.lead, p.group_id, p.target, p.origin)
        for p in points
    }


def _validate_points(points: list[SavedPoint]) -> None:
    if not points:
        raise DataContractError("saved model has no forecasts")
    numeric = np.array([(p.point, p.target) for p in points], dtype=np.float64)
    origins = _origins(points)
    _finite(numeric)
    _finite(np.array(origins, dtype=np.float64))
    for point in points:
        _validate_identity(point)


def _origins(points: list[SavedPoint]) -> list[float]:
    return [p.origin for p in points if p.origin is not None]


def _finite(values: np.ndarray[Any, Any]) -> None:
    if not np.isfinite(values).all():
        raise DataContractError("saved forecasts must be finite")


def _validate_identity(point: SavedPoint) -> None:
    _identifier(point.trajectory_id)
    _identifier(point.group_id)
    _frame(point.start, 0)
    _frame(point.lead, 1)


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise DataContractError("invalid saved identity")


def _frame(value: object, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise DataContractError("invalid saved frame/lead")


def _window(
    key: tuple[str, int], rows: dict[str, list[SavedPoint]], horizon: int
) -> OverviewWindow:
    for points in rows.values():
        points.sort(key=lambda p: p.lead)
        _validate_window(points, horizon)
    reference = next(iter(rows.values()))
    return OverviewWindow(
        trajectory_id=key[0],
        group_id=reference[0].group_id,
        start=key[1],
        origin=reference[0].origin,
        target=tuple(p.target for p in reference),
        points=_window_points(rows),
        bands=_window_bands(rows),
    )


def _window_points(rows: dict[str, list[SavedPoint]]) -> dict[str, tuple[float, ...]]:
    return {model: tuple(p.point for p in points) for model, points in rows.items()}


def _window_bands(
    rows: dict[str, list[SavedPoint]],
) -> dict[str, tuple[tuple[float, float], ...]]:
    return {model: _bands(points) for model, points in rows.items()}


def _validate_window(points: list[SavedPoint], horizon: int) -> None:
    if [p.lead for p in points] != list(range(1, horizon + 1)):
        raise DataContractError("saved window has missing or extra future leads")
    _single({p.origin for p in points}, "origin within window")
    _single({p.group_id for p in points}, "group within window")


def _single(values: set[Any], label: str) -> Any:
    if len(values) != 1:
        raise DataContractError("inconsistent saved " + label)
    return next(iter(values))


def _bands(points: list[SavedPoint]) -> tuple[tuple[float, float], ...]:
    if not all(0.1 in p.levels and 0.9 in p.levels for p in points):
        return ()
    return tuple(
        (p.quantiles[p.levels.index(0.1)], p.quantiles[p.levels.index(0.9)])
        for p in points
    )


def _probability(source: OverviewSource, config: OverviewConfig) -> list[OverviewPanel]:
    path = source.root / "summary.json"
    _check_hash(path, source.expected_hash)
    summary = json.loads(path.read_text())
    _verify_files(source.root, summary["file_sha256"], config)
    assert source.dataset is not None
    definitions = _definitions(source.dataset)
    _followup_definitions(source, summary)
    lead_interval, lead_unit = _followup_axis(source, summary)
    panels = []
    for cell, feature, models in _result_cells(summary["results"]):
        definition = definitions[feature]
        tables, raw = _followup_tables(
            source, config, summary["file_sha256"], cell, feature, models
        )
        sample = next(iter(tables.values()))[0]
        panels.append(
            OverviewPanel(
                title=f"{source.title} · {cell}",
                role=source.role,
                note=source.note,
                source_hash=source.expected_hash,
                feature_id=feature,
                unit=definition.unit.value,
                definition=definition.definition,
                context_frames=int(raw["context_frames"]),
                horizon_frames=int(raw["horizon_frames"]),
                models={m: m for m in models},
                lead_interval=lead_interval,
                lead_unit=lead_unit,
                statistical_reference=_followup_reference(models),
                windows=_match(tables, int(raw["horizon_frames"])),
                metrics=_followup_metrics(models),
                evidence={
                    "saved_results": models,
                    "code_commit": summary["code_commit"],
                    "split_hash": summary["split_hash"],
                    "config_hash": summary["config_hash"],
                    "sample_origin": sample.origin,
                    "task": summary.get("task"),
                    "deviations": summary.get("deviations", []),
                    "training_dataset": summary.get("training_dataset"),
                    "evaluation_dataset": summary.get("evaluation_dataset"),
                },
            )
        )
    return panels


def _followup_axis(
    source: OverviewSource, summary: dict[str, Any]
) -> tuple[float, str]:
    if "split.json" not in summary["file_sha256"]:
        return _interval_axis(None)
    split = read_metadata(source.root / "split.json", SplitManifest)
    _same(metadata_hash(split), summary["split_hash"], "confirmation split hash")
    _same(split.registry.dataset, source.dataset, "confirmation split dataset")
    records = split.registry.trajectories
    if not all(record.sampling_status == SamplingStatus.VERIFIED for record in records):
        return _interval_axis(None)
    intervals = {record.frame_interval_ps for record in records}
    return _interval_axis(intervals.pop() if len(intervals) == 1 else None)


def _followup_reference(models: dict[str, Any]) -> str | None:
    return "selected-statistic" if "selected-statistic" in models else None


def _definitions(dataset: DatasetConfig) -> dict[str, Any]:
    return {f.feature_id: f for f in dataset.features}


def _result_cells(results: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (cell, feature, models)
        for cell, features in results.items()
        for feature, models in features.items()
    ]


def _followup_definitions(source: OverviewSource, summary: dict[str, Any]) -> None:
    if "evaluation_dataset" in summary:
        _same(
            DatasetConfig.model_validate(summary["evaluation_dataset"]),
            source.dataset,
            "confirmation evaluation dataset",
        )
    # TRAIN residual states preserve the actual dataset semantics and full spec.
    for cell, states in summary.get("residual_state_hashes", {}).items():
        _state_definitions(source, cell, states, summary["file_sha256"])


def _state_definitions(
    source: OverviewSource, cell: str, states: dict[str, str], files: dict[str, str]
) -> None:
    for model_hash in states:
        name = f"{cell}/{model_hash.removeprefix('sha256:')}.json"
        if name not in files:
            raise DataContractError("residual state definition is not pinned")
        state = read_metadata(confined(source.root, name), ResidualState)
        _same(metadata_hash(state), states[model_hash], "residual state hash")
        if state.spec.dataset != source.dataset:
            raise DataContractError(
                "overview feature definitions differ from fitted dataset"
            )


def _followup_tables(
    source: OverviewSource,
    config: OverviewConfig,
    hashes: dict[str, str],
    cell: str,
    feature: str,
    models: dict[str, Any],
) -> tuple[dict[str, list[SavedPoint]], dict[str, Any]]:
    tables = {}
    dimension = None
    for model in models:
        name = f"{cell}/{feature}-{model}-leads.parquet"
        if name not in hashes:
            raise DataContractError("follow-up forecast table is not pinned")
        rows = _table(confined(source.root, name), config)
        current = _dimensions(rows)
        if dimension is not None and current != dimension:
            raise DataContractError("saved model context/horizon dimensions differ")
        dimension = current
        tables[model] = _followup_points(rows, feature)
    return tables, rows[0]


def _dimensions(rows: list[dict[str, Any]]) -> tuple[int, int]:
    context, horizon = _single(
        {(r["context_frames"], r["horizon_frames"]) for r in rows},
        "context/horizon dimensions",
    )
    _frame(context, 1)
    _frame(horizon, 1)
    return context, horizon


def _verify_files(root: Path, files: dict[str, str], config: OverviewConfig) -> None:
    paths = [confined(root, name) for name in files]
    _budget(paths, config)
    for name, digest in files.items():
        if file_hash(confined(root, name)) != "sha256:" + digest:
            raise DataContractError(f"saved bundle checksum mismatch: {name}")


def _check_hash(path: Path, expected: str) -> None:
    if file_hash(path) != expected:
        raise DataContractError("saved bundle checksum mismatch: " + path.name)


def _budget(paths: list[Path], config: OverviewConfig) -> None:
    if sum(p.stat().st_size for p in paths) > config.max_input_bytes:
        raise DataContractError("saved bundle exceeds overview max_input_bytes")


def _followup_points(rows: list[dict[str, Any]], feature: str) -> list[SavedPoint]:
    points = []
    for row in rows:
        if row["feature_id"] != feature:
            raise DataContractError("saved follow-up feature does not match file")
        levels = _levels(row)
        points.append(
            _point(
                {
                    **{
                        k: row[k]
                        for k in (
                            "trajectory_id",
                            "group_id",
                            "start",
                            "lead",
                            "point",
                            "target",
                            "origin",
                        )
                    },
                    "levels": levels,
                    "quantiles": tuple(row[f"quantile-{q}"] for q in levels),
                }
            )
        )
    return points


def _levels(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        sorted(
            float(k.removeprefix("quantile-")) for k in row if k.startswith("quantile-")
        )
    )


def _followup_metrics(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for model, result in models.items() for row in _model_metrics(model, result)
    ]


def _model_metrics(model: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in result["errors"] + result["per_lead_error"]:
        rows.extend(
            {
                "model": model,
                "lead": raw["lead"],
                "metric": metric,
                "value": raw[metric],
            }
            for metric in ("mae", "rmse", "bias")
        )
    for raw in result.get("probability", []) + result.get("per_lead_probability", []):
        rows.extend(_probability_metrics(model, raw))
    return rows


def _probability_metrics(model: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "lead": raw["lead"],
            "metric": metric,
            "value": value,
            "ci": raw["marginal_ci"][metric],
            "status": raw["interval_status"],
        }
        for metric, value in raw["metrics"].items()
    ]


def _benchmark(source: OverviewSource, config: OverviewConfig) -> list[OverviewPanel]:
    report = read_benchmark(source.root)
    if report.artifact_id != source.expected_hash:
        raise DataContractError("benchmark identity differs from pinned overview input")
    panels = []
    for index, cell in enumerate(report.manifest.cells):
        directory = source.root / f"cell-{index}"
        raw = _table(directory / "predictions.parquet", config)
        scores = _table(directory / "metrics.parquet", config)
        _check_benchmark_rows(raw, cell)
        models = _model_labels(cell.models, source.model_labels)
        multiplier, lead_unit = _horizon_axis(cell)
        for feature, definition in _selected_definitions(cell):
            tables = {model: _benchmark_points(raw, model, feature) for model in models}
            panels.append(
                OverviewPanel(
                    title=f"{source.title} · "
                    f"c{cell.spec.context_frames}-h{cell.spec.horizon_frames}",
                    role=source.role,
                    note=source.note,
                    source_hash=report.artifact_id,
                    feature_id=feature,
                    unit=definition.unit.value,
                    definition=definition.definition,
                    context_frames=cell.spec.context_frames,
                    horizon_frames=cell.spec.horizon_frames,
                    models=models,
                    lead_interval=multiplier,
                    lead_unit=lead_unit,
                    statistical_reference=source.statistical_references.get(feature),
                    windows=_match(tables, cell.spec.horizon_frames),
                    metrics=_benchmark_metrics(scores, feature),
                    evidence={
                        "manifest": cell.model_dump(mode="json"),
                        "comparisons": _table(
                            directory / "comparisons.parquet", config
                        ),
                    },
                )
            )
    return panels


def _selected_definitions(cell: CellManifest) -> list[tuple[str, Any]]:
    definitions = {f.feature_id: f for f in cell.spec.dataset.features}
    return [(feature, definitions[feature]) for feature in cell.spec.feature_ids]


def _check_benchmark_rows(rows: list[dict[str, Any]], cell: CellManifest) -> None:
    expected = {(w.trajectory_id, w.group_id, w.start) for w in _expected_windows(cell)}
    actual = {(r["trajectory_id"], r["group_id"], r["start"]) for r in rows}
    if actual != expected:
        raise DataContractError("saved windows differ from frozen benchmark manifest")
    _check_benchmark_columns(rows, cell)


def _check_benchmark_columns(rows: list[dict[str, Any]], cell: CellManifest) -> None:
    _same({r["feature_id"] for r in rows}, set(cell.spec.feature_ids), "feature set")
    _same(
        {r["model_hash"] for r in rows},
        {metadata_hash(m) for m in cell.models},
        "model set",
    )
    _check_future_frames(rows, cell.spec.context_frames)


def _same(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise DataContractError("saved " + label + " differs from frozen manifest")


def _check_future_frames(rows: list[dict[str, Any]], context: int) -> None:
    if any(r["target_frame"] != r["start"] + context + r["step"] - 1 for r in rows):
        raise DataContractError("saved future frame alignment differs from manifest")


def _benchmark_metrics(
    scores: list[dict[str, Any]], feature: str
) -> list[dict[str, Any]]:
    return [
        {
            "model": r["model_hash"],
            "lead": r["step"],
            "metric": _metric_name(r),
            "value": r["value"],
            "ci": [r["ci_lower"], r["ci_upper"]],
        }
        for r in scores
        if r["level"] == "aggregate" and r["feature_id"] == feature
    ]


def _model_labels(models: tuple[Any, ...], labels: dict[str, str]) -> dict[str, str]:
    result = {metadata_hash(m): m.model_id.value for m in models}
    if not labels.keys() <= result.keys():
        raise DataContractError("overview model labels refer to absent models")
    return result | labels


def _metric_name(row: dict[str, object]) -> str:
    if row["lower_quantile"] is None:
        return str(row["metric"])
    return f"{row['metric']}-{row['lower_quantile']}-{row['upper_quantile']}"


def _benchmark_points(
    rows: list[dict[str, Any]], model: str, feature: str
) -> list[SavedPoint]:
    return [
        _point(
            {
                **{
                    k: r[k]
                    for k in ("trajectory_id", "group_id", "start", "point", "target")
                },
                "lead": r["step"],
                "levels": r["quantile_levels"],
                "quantiles": r["quantiles"],
            }
        )
        for r in rows
        if r["model_hash"] == model and r["feature_id"] == feature
    ]


def load_overview(config: OverviewConfig) -> tuple[OverviewPanel, ...]:
    """Read pinned scientific tables, validating matching before writing output."""
    try:
        _check_inputs(config)
        return tuple(
            panel for source in config.sources for panel in _load_source(source, config)
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        raise DataContractError(
            f"cannot load saved overview inputs: {error}"
        ) from error


def _check_inputs(config: OverviewConfig) -> None:
    if config.mvp_root is not None:
        read_mvp(config.mvp_root)
    _budget(
        [p for source in config.sources for p in source.root.rglob("*") if p.is_file()],
        config,
    )


def _load_source(source: OverviewSource, config: OverviewConfig) -> list[OverviewPanel]:
    return (
        _benchmark(source, config)
        if source.kind == "benchmark"
        else _probability(source, config)
    )


def generate_overview(
    config: OverviewConfig, output: Path
) -> tuple[OverviewPanel, ...]:
    """Write a verified report atomically; failed renders preserve existing output."""
    try:
        return _generate_overview(config, output)
    except (OSError, ValueError) as error:
        raise DataContractError(f"cannot render saved overview: {error}") from error


def _protect_inputs(config: OverviewConfig, output: Path) -> None:
    roots = [source.root.resolve() for source in config.sources]
    if config.mvp_root is not None:
        roots.append(config.mvp_root.resolve())
    if any(output.resolve().is_relative_to(root) for root in roots):
        raise DataContractError(
            "overview output must be outside immutable input bundles"
        )


def _generate_overview(
    config: OverviewConfig, output: Path
) -> tuple[OverviewPanel, ...]:
    """Atomically write embedded data and package-native HTML/JS, without models."""
    from html import escape

    from md_forecast.evaluation.confirmation_view import render_confirmation
    from md_forecast.evaluation.structure_view import render_structure

    _protect_inputs(config, output)
    panels = load_overview(config)
    if config.default_panel >= len(panels):
        raise DataContractError("default overview panel is outside saved results")
    payload = json.dumps(
        [p.model_dump(mode="json") for p in panels], allow_nan=False
    ).replace("<", r"\u003c")
    assets = Path(__file__).parent / "overview_assets"
    page = (assets / "overview.html").read_text()
    replacements = {
        "__TITLE__": escape(config.title),
        "__CONCLUSION__": escape(config.conclusion),
        "__DATA__": payload,
        "__CONFIRMATION__": render_confirmation(config),
        "__SCRIPT__": (assets / "overview.js").read_text(),
        "__STRUCTURE__": render_structure(config.structure_config)
        if config.structure_config
        else (
            "<p>No structure supplied. "
            "Scalar forecasts do not specify atom coordinates.</p>"
        ),
    }
    replacements["__DEFAULT__"] = str(config.default_panel)
    page = re.sub(r"__[A-Z]+__", lambda match: replacements[match[0]], page)
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(output) as temporary:
        temporary.write_text(page, encoding="utf-8")
    return panels
