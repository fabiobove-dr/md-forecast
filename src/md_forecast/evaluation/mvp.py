"""Immutable, evidence-linked MVP synthesis from persisted scientific artifacts."""

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

from md_forecast.analysis.qc import FeatureQC, QCReport
from md_forecast.core.constants import DatasetId, ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, Identifier
from md_forecast.evaluation.ablations import AblationConfig
from md_forecast.evaluation.benchmark import CellManifest, ModelRuntime
from md_forecast.evaluation.external import BaselineSelection, ExternalConfig
from md_forecast.evaluation.report import (
    BenchmarkReport,
    read_benchmark,
    write_metric_figures,
)
from md_forecast.models.chronos import ChronosConfig
from md_forecast.models.finetuning import (
    FineTuneManifest,
    FineTuneResult,
    TrainingCheckpoint,
    checkpoint_artifact_hash,
)

METADATA_TYPES: dict[str, type[BaseModel]] = {
    "chronos": ChronosConfig,
    "training": FineTuneManifest,
    "training-result": FineTuneResult,
    "checkpoint": TrainingCheckpoint,
    "qc": QCReport,
    "ablations": AblationConfig,
    "external-selection": BaselineSelection,
    "external-config": ExternalConfig,
}
PACKAGES = (
    "md-forecast",
    "numpy",
    "pyarrow",
    "h5py",
    "pydantic",
    "chronos-forecasting",
    "torch",
    "mdtraj",
)
SCOPE_NOTES = {
    "integration-train": (
        "TRAIN integration evidence; does not establish held-out forecast skill."
    ),
    "development-validation": (
        "Small development validation; selection reuse and unresolved "
        "native semantics limit scientific claims."
    ),
    "external-complex": (
        "Untouched exact-PDB-disjoint external test; six complexes do not "
        "establish target/chemotype or pretraining independence."
    ),
    "replica-test": (
        "Held-out replicas of seen complexes; local supervised baseline "
        "adaptation is disclosed."
    ),
}


class ArtifactInput(BoundaryModel):
    """Portable input identity; an expected checksum prevents accidental run mixing."""

    key: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")]
    path: Path
    expected_hash: ArtifactHash

    @model_validator(mode="after")
    def portable_path(self) -> Self:
        """Keep versioned input paths relative and free of parent traversal."""
        if self.path.is_absolute() or ".." in self.path.parts:
            raise ValueError("report inputs require portable relative paths")
        return self


class BenchmarkInput(ArtifactInput):
    """One verified benchmark, with its actual scientific evaluation role."""

    role: Literal[
        "integration-train",
        "development-validation",
        "external-complex",
        "replica-test",
    ]
    model_labels: dict[ArtifactHash, Identifier] = {}
    headline_features: tuple[Identifier, ...] = ()


class MetadataInput(ArtifactInput):
    """A typed integrity-enveloped configuration or provenance artifact."""

    kind: Literal[
        "chronos",
        "training",
        "training-result",
        "checkpoint",
        "qc",
        "ablations",
        "external-selection",
        "external-config",
    ]


class MVPConfig(BoundaryModel):
    """Version the input cohort and bounded reporting resources, never tune models."""

    report_version: Identifier
    benchmarks: Annotated[tuple[BenchmarkInput, ...], Field(min_length=1)]
    metadata: tuple[MetadataInput, ...] = ()
    effects: tuple[ArtifactInput, ...] = ()
    limitations: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    max_metadata_bytes: Annotated[int, Field(strict=True, gt=0)] = 33554432
    max_table_rows: Annotated[int, Field(strict=True, gt=0)] = 1000000

    @model_validator(mode="after")
    def distinct_inputs(self) -> Self:
        """Prevent named evidence from shadowing another experiment or file."""
        keys = [s.key for s in (*self.benchmarks, *self.metadata, *self.effects)]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate report input key")
        return self


class Evidence(BoundaryModel):
    """Every displayed metric retains a saved table checksum and exact selector."""

    source: Identifier
    cell: Annotated[int, Field(strict=True, ge=0)]
    table_hash: ArtifactHash
    selector: dict[str, str | int | float | None]
    value: float
    unit: Identifier


class MVPSnapshot(BoundaryModel):
    """Paper-ready manifest with complete source manifests and output fingerprints."""

    config: MVPConfig
    config_hash: ArtifactHash
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    lockfile_hash: ArtifactHash
    python_version: Identifier
    packages: dict[str, str | None]
    benchmarks: dict[str, BenchmarkReport]
    metadata_payloads: dict[str, dict[str, Any]]
    operational_hashes: dict[str, ArtifactHash]
    evidence: tuple[Evidence, ...]
    output_hashes: dict[str, ArtifactHash]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Require the declared cohort/config to match the embedded sources."""
        if self.config_hash != metadata_hash(self.config):
            raise ValueError("MVP config identity differs")
        if set(self.benchmarks) != {s.key for s in self.config.benchmarks}:
            raise ValueError("MVP benchmark cohort differs")
        return self

    @property
    def artifact_id(self) -> str:
        """Version the scientific inputs, environment and generated report contents."""
        return metadata_hash(self)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _budget(path: Path, limit: int) -> None:
    if not path.is_file() or path.stat().st_size > limit:
        raise DataContractError(f"missing or oversized report input: {path}")


def _match(actual: str, source: ArtifactInput) -> None:
    if actual != source.expected_hash:
        raise DataContractError(f"report source identity differs: {source.key}")


def _read_source(source: BenchmarkInput, config: MVPConfig) -> BenchmarkReport:
    _budget(source.path / "report.json", config.max_metadata_bytes)
    report = read_benchmark(source.path)
    _match(report.artifact_id, source)
    for cell in report.manifest.cells:
        _validate_role(source, cell)
    return report


def _validate_role(source: BenchmarkInput, cell: CellManifest) -> None:
    expected = {
        "integration-train": Split.TRAIN,
        "development-validation": Split.VALIDATION,
        "external-complex": Split.TEST,
        "replica-test": Split.TEST,
    }
    if cell.partition != expected[source.role]:
        raise DataContractError("report role differs from the actual scored partition")
    if source.role in ("external-complex", "replica-test"):
        _validate_external_role(source, cell)
    _validate_labels(source, cell)


def _validate_labels(source: BenchmarkInput, cell: CellManifest) -> None:
    if not set(source.headline_features) <= set(cell.spec.feature_ids):
        raise DataContractError("report headline features are absent from this cell")
    known = {metadata_hash(c) for c in cell.models}
    if not set(source.model_labels) <= known:
        raise DataContractError("report model labels refer to unknown configurations")


def _validate_external_role(source: BenchmarkInput, cell: CellManifest) -> None:
    if cell.spec.dataset.dataset_id != DatasetId.MDBIND:
        raise DataContractError("external report role requires MDbind")
    replica = cell.split.config.mode == "unseen-replica"
    if replica != (source.role == "replica-test"):
        raise DataContractError("external complex/replica tasks cannot be conflated")


def _metadata_sources(config: MVPConfig) -> dict[str, dict[str, Any]]:
    payloads = {}
    for source in config.metadata:
        _budget(source.path, config.max_metadata_bytes)
        value = read_metadata(source.path, METADATA_TYPES[source.kind])
        _match(metadata_hash(value), source)
        payloads[source.key] = value.model_dump(mode="json")
    return payloads


def _read_table(path: Path, limit: int, **options: Any) -> pa.Table:
    file = pq.ParquetFile(path)
    if file.metadata.num_rows > limit:
        raise DataContractError(f"report table exceeds row budget: {path}")
    return pq.read_table(path, **options)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _refuse_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise DataContractError("MVP output exists; choose a new versioned directory")


def generate_mvp(
    config: MVPConfig, output: Path, *, code_commit: str, lockfile: Path
) -> MVPSnapshot:
    """Verify all inputs, regenerate tables/figures and atomically freeze a report."""
    config = MVPConfig.model_validate_json(config.model_dump_json())
    _refuse_output(output)
    try:
        return _generate_mvp(config, output, code_commit, lockfile)
    except (OSError, ValueError, KeyError, pa.ArrowException) as error:
        raise DataContractError(
            f"MVP report failed before publication: {error}"
        ) from error


def _generate_mvp(
    config: MVPConfig, output: Path, code_commit: str, lockfile: Path
) -> MVPSnapshot:
    reports = {s.key: _read_source(s, config) for s in config.benchmarks}
    metadata = _metadata_sources(config)
    _verify_model_provenance(reports, _chronos_provenance(config, metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".mvp-") as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        evidence, operational, sections = _benchmark_sections(staging, config, reports)
        sections.extend(_effect_sections(staging, config))
        sections.extend(_metadata_sections(config, metadata))
        text = _report_text(config, sections)
        (staging / "report.md").write_text(text, encoding="utf-8")
        pq.write_table(
            pa.Table.from_pylist([e.model_dump() for e in evidence]),
            staging / "headline-evidence.parquet",
        )
        hashes = _output_hashes(staging)
        snapshot = MVPSnapshot(
            config=config,
            config_hash=metadata_hash(config),
            code_commit=code_commit,
            lockfile_hash=_file_hash(lockfile),
            python_version=platform.python_version(),
            packages=_package_versions(),
            benchmarks=reports,
            metadata_payloads=metadata,
            operational_hashes=operational,
            evidence=tuple(evidence),
            output_hashes=hashes,
        )
        write_metadata(staging / "experiment-manifest.json", snapshot)
        _refuse_output(output)
        staging.rename(output)
        return snapshot


def _output_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _package_versions() -> dict[str, str | None]:
    return {p: _package_version(p) for p in PACKAGES}


def read_mvp(root: Path) -> MVPSnapshot:
    """Reject edited/missing report contents before accepting a final snapshot."""
    snapshot = read_metadata(root / "experiment-manifest.json", MVPSnapshot)
    for name, digest in snapshot.output_hashes.items():
        _safe_output_name(name)
        if _file_hash(root / name) != digest:
            raise DataContractError(f"MVP output checksum differs: {name}")
    return snapshot


def _safe_output_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise DataContractError("unsafe MVP output name")


def _benchmark_sections(
    root: Path, config: MVPConfig, reports: dict[str, BenchmarkReport]
) -> tuple[list[Evidence], dict[str, str], list[str]]:
    evidence, operational, sections = [], {}, []
    for source in config.benchmarks:
        report = reports[source.key]
        sections.append(
            f"## {source.key}\n\n{SCOPE_NOTES[source.role]}\n\nReport: `"
            f"{report.artifact_id}`.\n"
        )
        for index, cell in enumerate(report.manifest.cells):
            folder = root / source.key / f"cell-{index}"
            folder.mkdir(parents=True)
            metrics_path = source.path / f"cell-{index}" / "metrics.parquet"
            metrics = _read_table(
                metrics_path,
                config.max_table_rows,
                filters=[("level", "=", "aggregate")],
            )
            write_metric_figures(folder, cell, metrics)
            pq.write_table(metrics, folder / "aggregate-metrics.parquet")
            values = _metric_evidence(source, report, index, metrics)
            evidence.extend(values)
            details = _cell_section(source, report, index, values)
            sections.append(_cell_overview(source, report, index, values))
            runtime = source.path / f"cell-{index}" / "runtime.parquet"
            operational[f"{source.key}/cell-{index}/runtime"] = _file_hash(runtime)
            details += _runtime_section(runtime, cell, config.max_table_rows)
            details += _comparison_section(source, index, folder, config.max_table_rows)
            (folder / "details.md").write_text(
                details.replace(f"]({source.key}/cell-{index}/", "](")
            )
    return evidence, operational, sections


def _cell_overview(
    source: BenchmarkInput, report: BenchmarkReport, index: int, values: list[Evidence]
) -> str:
    cell, summary = report.manifest.cells[index], report.cells[index]
    text = (
        f"### C{cell.spec.context_frames}/H{cell.spec.horizon_frames}: "
        f"{summary.window_count} windows, {summary.group_count} groups\n\n"
    )
    text += (
        f"[Full metrics, calibration, comparisons, provenance and figures]("
        f"{source.key}/cell-{index}/details.md).\n\n"
    )
    if source.role == "integration-train":
        return text
    text += (
        "| Model/config | Feature | Unit | Metric | Value |\n| --- | --- |"
        " --- | --- | ---: |\n"
    )
    labels = {
        metadata_hash(m): source.model_labels.get(metadata_hash(m), m.model_id.value)
        for m in cell.models
    }
    return text + "".join(
        _evidence_row(e, labels) for e in _headline_mae(source, values)
    )


def _headline_mae(source: BenchmarkInput, values: list[Evidence]) -> list[Evidence]:
    allowed = set(source.headline_features)
    rows = [e for e in values if e.selector["metric"] == "mae"]
    if allowed:
        return _selected_features(rows, allowed)
    return rows


def _selected_features(rows: list[Evidence], allowed: set[str]) -> list[Evidence]:
    return [e for e in rows if e.selector["feature_id"] in allowed]


def _comparison_section(
    source: BenchmarkInput, index: int, root: Path, limit: int
) -> str:
    table = _read_table(source.path / f"cell-{index}" / "comparisons.parquet", limit)
    pq.write_table(table, root / "comparisons.parquet")
    text = (
        "\nPaired model-minus-persistence whole-horizon effects; the "
        "declared corrected group policy governs uncertainty. Unavailable "
        "intervals remain unavailable.\n\n| Model | Feature | MAE "
        "difference | Ratio | Corrected CI | Status |\n| --- | --- | ---: "
        "| ---: | --- | --- |\n"
    )
    for row in table.to_pylist():
        if row["step"] == 0:
            text += (
                f"| `{row['model_hash'][-8:]}` | {row['feature_id']} | "
                f"{row['mae_difference']:.8g} | {row['ratio_to_persistence']} | ["
                f"{row['ci_lower']}, {row['ci_upper']}] | "
                f"{row['uncertainty_status']} |\n"
            )
    return text


def _metric_evidence(
    source: BenchmarkInput, report: BenchmarkReport, index: int, metrics: pa.Table
) -> list[Evidence]:
    cell = report.manifest.cells[index]
    units = {f.feature_id: f.unit.value for f in cell.spec.dataset.features}
    rows = [r for r in metrics.to_pylist() if r["step"] == 0]
    return [
        Evidence(
            source=source.key,
            cell=index,
            table_hash=report.cells[index].table_hashes["metrics"],
            selector=_metric_selector(r),
            value=r["value"],
            unit=_metric_unit(r, units),
        )
        for r in rows
    ]


def _metric_selector(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: row[k]
        for k in (
            "level",
            "model_hash",
            "feature_id",
            "step",
            "metric",
            "lower_quantile",
            "upper_quantile",
        )
    }


def _metric_unit(row: dict[str, Any], units: dict[str, str]) -> str:
    if row["metric"] in ("coverage", "scaled-mae"):
        return "dimensionless"
    return units[row["feature_id"]]


def _cell_section(
    source: BenchmarkInput, report: BenchmarkReport, index: int, values: list[Evidence]
) -> str:
    cell, summary = report.manifest.cells[index], report.cells[index]
    records = cell.split.registry.trajectories
    datasets = sorted(
        {
            (
                f"{r.dataset_id.value} {r.dataset_version}: {r.license_id}, "
                f"{r.source_checksum}"
            )
            for r in records
        }
    )
    text = (
        f"### C{cell.spec.context_frames}/H{cell.spec.horizon_frames}, "
        f"{cell.partition.value}\n\n"
    )
    text += (
        f"Windows / trajectories / systems / groups: {summary.window_count}"
        f" / {summary.trajectory_count} / {summary.system_count} / "
        f"{summary.group_count}.\n\n"
    )
    text += (
        f"Split `{cell.spec.split_hash}`; preprocessing `"
        f"{metadata_hash(cell.scaler)}`; features `"
        f"{cell.spec.dataset.feature_set_version}`; config `"
        f"{metadata_hash(cell.config)}`; seed {cell.config.seed}; hardware "
        f"{cell.hardware}.\n\n"
    )
    text += "Sources: " + "; ".join(datasets) + ".\n\n"
    text += (
        "Scalers contain exactly declared TRAIN context regions. "
        "Biological identity/grouping and source/grid/feature linkage are "
        "validated by CellManifest. Final snapshot verification does not "
        "retrospectively prove undocumented upstream processing.\n\n"
    )
    text += (
        "| Model/config | Feature | Unit | Metric (whole horizon) | Value "
        "|\n| --- | --- | --- | --- | ---: |\n"
    )
    labels = {
        metadata_hash(m): source.model_labels.get(metadata_hash(m), m.model_id.value)
        for m in cell.models
    }
    text += "".join(_evidence_row(e, labels) for e in values)
    return text + _figure_markdown(source.key, index, cell.spec.feature_ids, values)


def _figure_markdown(
    key: str, index: int, features: tuple[str, ...], values: list[Evidence]
) -> str:
    text = ""
    for feature_index, feature in enumerate(features):
        prefix = f"{key}/cell-{index}"
        text += (
            f"\n![{feature} error versus horizon]({prefix}/horizon-"
            f"{feature_index}.svg)\n"
        )
        if any(
            e.selector["metric"] == "coverage" and e.selector["feature_id"] == feature
            for e in values
        ):
            text += (
                f"\n![{feature} calibration]({prefix}/calibration-{feature_index}"
                f".svg)\n"
            )
    return text


def _evidence_row(evidence: Evidence, labels: dict[str, str]) -> str:
    selector = evidence.selector
    key = str(selector["model_hash"])
    metric = str(selector["metric"])
    if selector["lower_quantile"] is not None:
        metric += f" ({selector['lower_quantile']}, {selector['upper_quantile']})"
    return (
        f"| {labels[key]} `{key[-8:]}` | {selector['feature_id']} | "
        f"{evidence.unit} | {metric} | {evidence.value:.8g} |\n"
    )


def _runtime_section(path: Path, cell: CellManifest, limit: int) -> str:
    rows = _read_table(path, limit).to_pylist()
    runtimes = [ModelRuntime.model_validate(r) for r in rows]
    text = (
        "\nAdapter inference/host-conversion telemetry (not full "
        "process/training cost):\n\n| Model hash | Total seconds | Peak "
        "allocated bytes | Peak reserved bytes |\n| --- | ---: | ---: | "
        "---: |\n"
    )
    for model in cell.models:
        key = metadata_hash(model)
        text += _runtime_row(runtimes, key)
    return text


def _runtime_row(runtimes: list[ModelRuntime], key: str) -> str:
    values = [r for r in runtimes if r.config_hash == key]
    return (
        f"| `{key[-8:]}` | {sum(r.seconds for r in values):.6g} | "
        f"{_peak(values, 'peak_allocated_bytes')} | "
        f"{_peak(values, 'peak_reserved_bytes')} |\n"
    )


def _peak(values: list[ModelRuntime], field: str) -> int | str:
    numbers = [getattr(r, field) for r in values if getattr(r, field) is not None]
    return max(numbers) if numbers else "unmeasured"


def _effect_sections(root: Path, config: MVPConfig) -> list[str]:
    sections = []
    for source in config.effects:
        _match(_file_hash(source.path), source)
        table = _read_table(source.path, config.max_table_rows)
        pq.write_table(table, root / f"{source.key}.parquet")
        text = (
            f"## {source.key}\n\nSaved effect table `{source.expected_hash}`; "
            f"complete leads/intervals retained beside this report. Negative "
            f"MAE differences favor the tested variant/model. Missing corrected"
            f" bounds remain unavailable, not significant.\n\n"
        )
        text += "| Field | Whole-horizon effect row |\n| --- | --- |\n"
        for row in table.to_pylist():
            if row["step"] == 0:
                text += (
                    "".join(f"| {key} | {value} |\n" for key, value in row.items())
                    + "\n"
                )
        sections.append(text)
    return sections


def _metadata_sections(
    config: MVPConfig, metadata: dict[str, dict[str, Any]]
) -> list[str]:
    sections = []
    for source in config.metadata:
        payload = metadata[source.key]
        text = (
            f"## {source.key}\n\nTyped {source.kind} artifact `"
            f"{source.expected_hash}` is embedded in "
            f"experiment-manifest.json.\n"
        )
        if source.kind == "qc":
            text += _qc_summary(QCReport.model_validate(payload))
        if source.kind == "training-result":
            text += (
                f"\nSelected step {payload['selected_step']}, completed "
                f"{payload['completed_steps']}; validation loss "
                f"{payload['validation_loss']}; runtime {payload['runtime']}. "
                f"Validation selection is not an independent test.\n"
            )
        sections.append(text)
    return sections


def _qc_summary(report: QCReport) -> str:
    features = report.trajectories[0].features
    text = (
        "\nTRAIN-only source QC; native cadence remains frames. Median "
        "e-folding crossing is a noisy finite-sample statistic, not an "
        "established molecular timescale.\n\n| Feature | Median lag-1 ACF "
        "| Median first 1/e crossing (frames) | Censored trajectories |\n|"
        " --- | ---: | ---: | ---: |\n"
    )
    for feature in features:
        values = _qc_values(report, feature.feature_id)
        text += _qc_row(feature.feature_id, values)
    return text


def _qc_values(report: QCReport, feature: str) -> list[FeatureQC]:
    return [
        next(f for f in row.features if f.feature_id == feature)
        for row in report.trajectories
    ]


def _qc_row(feature: str, values: list[FeatureQC]) -> str:
    rho = _qc_rho(values)
    decay = [v.decay_frames for v in values if v.decay_frames is not None]
    return (
        f"| {feature} | {_median(rho)} | {_median(decay)} | "
        f"{sum(v.decay_censored for v in values)} |\n"
    )


def _qc_rho(values: list[FeatureQC]) -> list[float]:
    return [v.acf[1] for v in values if v.acf is not None]


def _median(values: list[float] | list[int]) -> str:
    return f"{float(np.median(values)):.6g}" if values else "unavailable"


def _report_text(config: MVPConfig, sections: list[str]) -> str:
    text = (
        f"# md-forecast MVP — {config.report_version}\n\nScientific success"
        f" is **not established** by this artifact cohort. Engineering "
        f"completion, training integration, development selection and "
        f"external tests are reported separately. Negative/null effects and"
        f" unavailable corrected uncertainty remain visible.\n\n"
    )
    text += "## Frozen limits\n\n" + "".join(
        f"- {limit}\n" for limit in config.limitations
    )
    text += (
        "\nEvery metric row maps to its original saved table through "
        "[headline-evidence.parquet](headline-evidence.parquet) and "
        "[experiment-manifest.json](experiment-manifest.json) selectors/hashes."
        " Figures are regenerated from saved metrics. Source manifests "
        "retain dataset versions/checksums/terms, feature "
        "definitions/units, split/preprocessing/config hashes, model "
        "configuration/artifact identities, seeds, code/lock/hardware, and"
        " bootstrap policy. Typed model/training settings record exact "
        "revisions. No model runs or tuning occurs during synthesis.\n\n"
    )
    return text + "\n".join(sections)


def _chronos_provenance(
    config: MVPConfig, metadata: dict[str, dict[str, Any]]
) -> set[str]:
    settings, checkpoints = [], []
    for source in config.metadata:
        value = METADATA_TYPES[source.kind].model_validate(metadata[source.key])
        setting = _metadata_settings(value)
        if setting is not None:
            settings.append(setting)
        if isinstance(value, TrainingCheckpoint):
            checkpoints.append(value)
    return _chronos_artifacts(settings, checkpoints)


def _metadata_settings(value: BaseModel) -> ChronosConfig | None:
    if isinstance(value, ChronosConfig):
        return value
    if isinstance(value, FineTuneManifest):
        return value.settings
    if isinstance(value, TrainingCheckpoint):
        return value.manifest.settings
    if isinstance(value, (AblationConfig, ExternalConfig)):
        return value.chronos
    return None


def _chronos_artifacts(
    settings: list[ChronosConfig], checkpoints: list[TrainingCheckpoint]
) -> set[str]:
    known = {metadata_hash(s) for s in settings}
    known.update(checkpoint_artifact_hash(c, s) for c in checkpoints for s in settings)
    return known


def _verify_model_provenance(
    reports: dict[str, BenchmarkReport], known: set[str]
) -> None:
    for report in reports.values():
        for cell in report.manifest.cells:
            _verify_cell_models(cell, known)


def _verify_cell_models(cell: CellManifest, known: set[str]) -> None:
    for model, artifact in zip(cell.models, cell.model_artifacts, strict=True):
        if model.model_id == ModelId.CHRONOS2 and artifact not in known:
            raise DataContractError(
                "missing linked Chronos revision/checkpoint settings"
            )


def load_mvp_config(path: Path) -> MVPConfig:
    """Read a frozen synthesis specification with actionable boundary errors."""
    try:
        return MVPConfig.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise DataContractError(
            f"cannot load MVP configuration {path}: {error}"
        ) from error
