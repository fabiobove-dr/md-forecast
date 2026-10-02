"""Publish complete local QC bundles with canonical provenance and ACF curves."""

import csv
import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import yaml

from md_forecast.analysis.qc import QCConfig, QCReport, analyze_development
from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import write_metadata
from md_forecast.data.public.misato import ExtractionReport
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import Provenance, TrajectoryManifest
from md_forecast.data.series import read_series
from md_forecast.data.splits import SplitConfig, build_split

logger = logging.getLogger(__name__)


def load_qc_config(path: Path) -> QCConfig:
    """Read versioned analysis settings without touching any numerical data."""
    try:
        return QCConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise DataContractError(
            f"cannot load QC configuration {path}: {error}"
        ) from error


def _validate_extraction(root: Path, extraction: ExtractionReport) -> None:
    registry = read_registry(root / "registry.json")
    if set(extraction.exported) != {
        record.trajectory_id for record in registry.trajectories
    }:
        raise DataContractError(
            "extraction file mapping differs from registry identities"
        )
    for record in registry.trajectories:
        source = Provenance.model_validate(
            record.model_dump(include=set(Provenance.model_fields))
        )
        if source != extraction.source:
            raise DataContractError("extraction provenance differs from registry")


def _series_path(
    root: Path, extraction: ExtractionReport, record: TrajectoryManifest
) -> Path:
    filename = extraction.exported[record.trajectory_id]
    path = root / filename
    if Path(filename).name != filename or path.resolve().parent != root.resolve():
        raise DataContractError("QC series must be a direct local extraction file")
    return path


def _write_curves(path: Path, report: QCReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trajectory_id", "feature_id", "lag_frames", "acf"))
        for trajectory in report.trajectories:
            for feature in trajectory.features:
                writer.writerows(
                    _curve_rows(
                        trajectory.trajectory_id, feature.feature_id, feature.acf
                    )
                )


def _curve_rows(
    trajectory: str,
    feature: str,
    curve: tuple[float, ...] | None,
) -> Iterator[tuple[str, str, int, float | str]]:
    if curve is None:
        yield trajectory, feature, 0, ""
        return
    yield from ((trajectory, feature, lag, value) for lag, value in enumerate(curve))


def _refuse_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise DataContractError(
            f"QC output already exists: {path}; choose a new directory"
        )


def run_misato_qc(
    root: Path, output: Path, config: QCConfig, *, code_commit: str, lockfile: Path
) -> QCReport:
    """Publish a train-only report and complete curves; never overwrite a bundle.

    Input is an already verified MISATO extraction, not an HDF5 acquisition.
    Complete publication uses the same single-writer directory pattern as extraction.
    """
    _refuse_existing(output)
    try:
        extraction = ExtractionReport.model_validate_json(
            (root / "qc.json").read_text(encoding="utf-8")
        )
        _validate_extraction(root, extraction)
        split = build_split(
            read_registry(root / "registry.json"),
            SplitConfig(mode="official", seed=config.development.seed),
        )
        lockfile_hash = "sha256:" + hashlib.sha256(lockfile.read_bytes()).hexdigest()

        def loader(record: TrajectoryManifest) -> pa.Table:
            if record.split != Split.TRAIN:
                raise DataContractError("QC must never open held-out series")
            expected = Registry(dataset=split.registry.dataset, trajectories=(record,))
            return read_series(
                _series_path(root, extraction, record), expected=expected
            )

        report = analyze_development(
            split, config, loader, code_commit=code_commit, lockfile_hash=lockfile_hash
        )
        _publish(output, report, extraction)
    except (OSError, UnicodeError, ValueError) as error:
        raise DataContractError(f"QC failed before publication: {error}") from error
    logger.info(
        "development QC: trajectories=%d report=%s", len(report.trajectories), output
    )
    return report


def _publish(output: Path, report: QCReport, extraction: ExtractionReport) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".qc-") as directory:
        staging = Path(directory) / "output"
        staging.mkdir()
        write_metadata(staging / "report.json", report)
        write_metadata(staging / "development.json", report.development)
        write_metadata(staging / "config.json", report.config)
        write_metadata(staging / "extraction.json", extraction)
        _write_curves(staging / "acf.csv", report)
        _refuse_existing(output)
        # ponytail: one writer; add no-replace rename for concurrent publishers.
        staging.rename(output)
