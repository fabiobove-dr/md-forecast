"""Atomic benchmark bundles, deterministic scientific IDs and native SVG curves."""

import hashlib
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, ModelId, SamplingStatus
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    SchemaVersion,
    TrajectoryManifest,
)
from md_forecast.evaluation.benchmark import CellManifest, CellResult, GridManifest

type TableName = Literal["predictions", "metrics", "comparisons"]
TABLE_NAMES: tuple[TableName, ...] = ("predictions", "metrics", "comparisons")
COLORS = ("#222222", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


class PlotRow(TypedDict):
    """Projection of metric table columns used by the native SVG renderer."""

    model_hash: str
    level: str
    feature_id: str
    metric: str
    step: int
    value: float
    lower_quantile: float
    upper_quantile: float


class CellSummary(BoundaryModel):
    """Typed portable linkage to columnar outputs and independent sample counts."""

    manifest_hash: ArtifactHash
    table_hashes: dict[TableName, ArtifactHash]
    window_count: Annotated[int, Field(strict=True, gt=0)]
    trajectory_count: Annotated[int, Field(strict=True, gt=0)]
    system_count: Annotated[int, Field(strict=True, gt=0)]
    group_count: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        """Do not accept incomplete outputs or inverted hierarchy counts."""
        if set(self.table_hashes) != set(TABLE_NAMES):
            raise ValueError("benchmark summary requires all three scientific tables")
        if (
            not self.group_count
            <= self.system_count
            <= self.trajectory_count
            <= self.window_count
        ):
            raise ValueError("invalid window/trajectory/system/group counts")
        return self


class BenchmarkReport(BoundaryModel):
    """Content-based scientific identity excludes nondeterministic runtime telemetry."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    manifest: GridManifest
    cells: tuple[CellSummary, ...]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Reject missing cells and mismatched manifests at the persistence boundary."""
        hashes = tuple(metadata_hash(cell) for cell in self.manifest.cells)
        if tuple(cell.manifest_hash for cell in self.cells) != hashes:
            raise ValueError("report cells differ from the frozen complete grid")
        return self

    @property
    def artifact_id(self) -> str:
        """Stable hash of inputs and actual prediction/metric/comparison contents."""
        return metadata_hash(self)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _refuse_existing(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise DataContractError("benchmark output exists; choose a new directory")


def write_benchmark(
    output: Path, manifest: GridManifest, results: tuple[CellResult, ...]
) -> BenchmarkReport:
    """Publish every frozen cell together; no overwrite or partial output promotion."""
    snapshot = GridManifest.model_validate_json(manifest.model_dump_json())
    if tuple(result.manifest for result in results) != snapshot.cells:
        raise DataContractError("results do not match the complete frozen grid")
    _refuse_existing(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        return _publish(output, snapshot, results)
    except (OSError, ValueError) as error:
        raise DataContractError(
            f"benchmark failed before publication: {error}"
        ) from error


def _publish(
    output: Path, manifest: GridManifest, results: tuple[CellResult, ...]
) -> BenchmarkReport:
    with TemporaryDirectory(dir=output.parent, prefix=".benchmark-") as directory:
        staging = Path(directory) / "output"
        staging.mkdir()
        summaries = tuple(
            _write_cell(staging / f"cell-{index}", result)
            for index, result in enumerate(results)
        )
        report = BenchmarkReport(manifest=manifest, cells=summaries)
        write_metadata(staging / "report.json", report)
        _refuse_existing(output)
        # ponytail: single writer; add no-replace rename for concurrent publishers.
        staging.rename(output)
        return report


def _write_cell(root: Path, result: CellResult) -> CellSummary:
    root.mkdir()
    hashes = {}
    for name in TABLE_NAMES:
        path = root / f"{name}.parquet"
        pq.write_table(getattr(result, name), path)
        hashes[name] = _file_hash(path)
    # Wall-clock timings deliberately have no role in report.artifact_id.
    pq.write_table(
        pa.Table.from_pylist([r.model_dump() for r in result.runtimes]),
        root / "runtime.parquet",
    )
    _write_figures(root, result)
    return CellSummary(
        manifest_hash=metadata_hash(result.manifest),
        table_hashes=hashes,
        window_count=result.window_count,
        trajectory_count=result.trajectory_count,
        system_count=result.system_count,
        group_count=result.group_count,
    )


def read_benchmark(root: Path) -> BenchmarkReport:
    """Verify integrity of every scientific table before accepting a saved report."""
    report = read_metadata(root / "report.json", BenchmarkReport)
    try:
        for index, summary in enumerate(report.cells):
            _verify_tables(root / f"cell-{index}", summary)
    except OSError as error:
        raise DataContractError(f"cannot read benchmark tables: {error}") from error
    return report


def _verify_tables(root: Path, summary: CellSummary) -> None:
    for name, expected in summary.table_hashes.items():
        if _file_hash(root / f"{name}.parquet") != expected:
            raise DataContractError("benchmark scientific table checksum mismatch")


def _write_figures(root: Path, result: CellResult) -> None:
    rows = cast(list[PlotRow], result.metrics.to_pylist())
    units = {f.feature_id: f.unit.value for f in result.manifest.spec.dataset.features}
    for index, feature in enumerate(result.manifest.spec.feature_ids):
        curves = _horizon_curves(rows, feature, result.manifest)
        multiplier, unit = _horizon_axis(result.manifest)
        path = root / f"horizon-{index}.svg"
        path.write_text(
            _svg(
                curves,
                f"{feature}: MAE ({units[feature]})",
                f"lead time ({unit})",
                multiplier,
            ),
            encoding="utf-8",
        )
        calibration = _calibration_curves(rows, feature, result.manifest)
        if calibration:
            calibration["ideal coverage"] = [(0, 0), (1, 1)]
            (root / f"calibration-{index}.svg").write_text(
                _svg(
                    calibration,
                    f"{feature}: interval calibration",
                    "nominal coverage",
                    1,
                ),
                encoding="utf-8",
            )


def _horizon_curves(
    rows: list[PlotRow], feature: str, manifest: CellManifest
) -> dict[str, list[tuple[float, float]]]:
    keys = dict(zip(map(metadata_hash, manifest.models), manifest.models, strict=True))
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if _is_curve_row(row, feature, "mae") and int(row["step"]) > 0:
            key = str(row["model_hash"])
            label = f"{keys[key].model_id} ({key[-8:]})"
            curves.setdefault(label, []).append(
                (float(row["step"]), float(row["value"]))
            )
    label = _persistence_label(manifest)
    if label not in curves:
        raise DataContractError("horizon figure requires persistence")
    return curves


def _persistence_label(manifest: CellManifest) -> str:
    reference = next(
        config for config in manifest.models if config.model_id == ModelId.PERSISTENCE
    )
    return f"{reference.model_id} ({metadata_hash(reference)[-8:]})"


def _is_curve_row(row: PlotRow, feature: str, metric: str) -> bool:
    return (row["level"], row["feature_id"], row["metric"]) == (
        "aggregate",
        feature,
        metric,
    )


def _calibration_curves(
    rows: list[PlotRow], feature: str, manifest: CellManifest
) -> dict[str, list[tuple[float, float]]]:
    keys = dict(zip(map(metadata_hash, manifest.models), manifest.models, strict=True))
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if _is_curve_row(row, feature, "coverage") and row["step"] == 0:
            key = str(row["model_hash"])
            label = f"{keys[key].model_id} ({key[-8:]})"
            nominal = float(row["upper_quantile"]) - float(row["lower_quantile"])
            curves.setdefault(label, []).append((nominal, float(row["value"])))
    return curves


def _horizon_axis(manifest: CellManifest) -> tuple[float, str]:
    records = _evaluation_records(manifest)
    if not all(r.sampling_status == SamplingStatus.VERIFIED for r in records):
        return _interval_axis(None)
    intervals = {r.frame_interval_ps for r in records}
    if len(intervals) != 1:
        return _interval_axis(None)
    return _interval_axis(intervals.pop())


def _interval_axis(interval: float | None) -> tuple[float, str]:
    if interval is not None:
        return interval, "ps"
    return 1, "frames; physical sampling not verified/uniform"


def _evaluation_records(manifest: CellManifest) -> tuple[TrajectoryManifest, ...]:
    ids = {
        a.trajectory_id
        for a in manifest.split.assignments
        if a.split == manifest.partition
    }
    return tuple(
        r for r in manifest.split.registry.trajectories if r.trajectory_id in ids
    )


def _svg(
    curves: dict[str, list[tuple[float, float]]],
    title: str,
    axis: str,
    multiplier: float,
) -> str:
    xmax, ymax = _curve_limits(curves, multiplier)
    height = 340 + 22 * len(curves)
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" '
        f'height="{height}" viewBox="0 0 760 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="60" y="24">{escape(title)}</text>',
        '<path d="M60 50 V280 H700" fill="none" stroke="black"/>',
        f'<text x="60" y="46">max y: {ymax:.6g}</text>',
        f'<text x="60" y="310">{escape(axis)}; range 0–{xmax:.6g}</text>',
    ]
    for index, (label, values) in enumerate(curves.items()):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(
            f"{60 + 640 * x * multiplier / xmax:.3f},{280 - 220 * y / ymax:.3f}"
            for x, y in sorted(values)
        )
        content.append(
            f'<polyline points="{coordinates}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        content.extend(
            f'<circle cx="{60 + 640 * x * multiplier / xmax:.3f}" '
            f'cy="{280 - 220 * y / ymax:.3f}" r="3" fill="{color}"/>'
            for x, y in values
        )
        content.append(
            f'<text x="60" y="{340 + index * 22}" fill="{color}">{escape(label)}</text>'
        )
    return "\n".join((*content, "</svg>"))


def _curve_limits(
    curves: dict[str, list[tuple[float, float]]], multiplier: float
) -> tuple[float, float]:
    points = [point for values in curves.values() for point in values]
    return max(1e-12, max(x * multiplier for x, _ in points)), max(
        1e-12, max(y for _, y in points)
    )
