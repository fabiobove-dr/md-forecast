"""Checksum-verified, one-system-at-a-time MISATO native observable extraction."""

import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self

import h5py
import numpy as np
import pyarrow as pa
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    MISATO_CONFIG,
    DatasetId,
    FeatureUnit,
    Split,
)
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import (
    AcquisitionSource,
    SourceArtifact,
    validate_artifact,
)
from md_forecast.data.registry import Registry, atomic_output, write_registry
from md_forecast.data.schemas import (
    BoundaryModel,
    DatasetConfig,
    FeatureDefinition,
    Identifier,
    Provenance,
    SchemaVersion,
    TrajectoryManifest,
    stable_system_id,
    stable_trajectory_id,
)
from md_forecast.data.series import FloatArray, to_arrow, write_series

logger = logging.getLogger(__name__)
NATIVE_BINDINGS = {
    "ligand_rmsd": ("frames_rmsd_ligand", FeatureUnit.ANGSTROM),
    "ligand_receptor_com_distance": ("frames_distance", FeatureUnit.ANGSTROM),
    "buried_sasa": ("frames_bSASA", FeatureUnit.ANGSTROM_SQUARED),
    "interaction_energy": ("frames_interaction_energy", FeatureUnit.KCAL_PER_MOL),
}
SPLIT_FILES = {
    Split.TRAIN: "train_MD.txt",
    Split.VALIDATION: "val_MD.txt",
    Split.TEST: "test_MD.txt",
}
PDB_ID_PATTERN = r"[0-9][A-Z0-9]{3}"
NATIVE_REPLICA = "native"
REGISTRY_FILE = "registry.json"
QC_FILE = "qc.json"


class NativeFeature(BaseModel):
    """Audited source channel, including explicit limits on its definition."""

    model_config = ConfigDict(extra="ignore")
    source_key: str
    unit: FeatureUnit
    shape: tuple[int, ...]
    dtype: Literal["float64"]
    time_resolved: Literal[True]
    description: Identifier
    definition: Identifier


class NativeTime(BaseModel):
    """Current audited export has frame order, but no verified physical times."""

    model_config = ConfigDict(extra="ignore")
    frame_count: Annotated[int, Field(gt=0)]
    timestamp_key: None
    verified_frame_interval_ps: None


class SourceLicense(BaseModel):
    """Record license, without inventing upstream rights clearance."""

    model_config = ConfigDict(extra="ignore")
    id: Identifier


class SampleEvidence(BaseModel):
    """Pinned sample identity is distinct from the full MD artifact."""

    model_config = ConfigDict(extra="ignore")
    sample_url: HttpUrl
    sample_size_bytes: Annotated[int, Field(gt=0)]
    sample_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class MisatoSource(AcquisitionSource):
    """Adapter subset of the versioned audit, not a new scientific audit."""

    dataset_id: Literal["misato"]
    feature_set_version: Identifier
    license: SourceLicense
    time_axis: NativeTime
    features: dict[str, NativeFeature]
    audit_evidence: SampleEvidence

    @model_validator(mode="after")
    def validate_native_channels(self) -> Self:
        """Fail closed if keys, units, or sampling assumptions change."""
        if set(self.features) != set(NATIVE_BINDINGS):
            raise ValueError(
                "MISATO native features differ from the audited four channels"
            )
        for name, binding in NATIVE_BINDINGS.items():
            _validate_feature(self.features[name], binding, self.time_axis.frame_count)
        return self


def _validate_feature(
    feature: NativeFeature, binding: tuple[str, FeatureUnit], frames: int
) -> None:
    if (feature.source_key, feature.unit) != binding or feature.shape != (frames,):
        raise ValueError(
            "MISATO source key/unit/shape differs from the audited contract"
        )


class ExtractionConfig(BoundaryModel):
    """Local runtime paths and explicit system selection; paths are not persisted."""

    input_path: Path
    output_dir: Path
    splits_dir: Path
    artifact: Literal["md", "sample"] = "md"
    system_ids: Annotated[tuple[str, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Reject duplicates and unsafe IDs before opening data or creating output."""
        if len(self.system_ids) != len(set(self.system_ids)):
            raise ValueError("duplicate requested system IDs")
        if any(not re.fullmatch(PDB_ID_PATTERN, system) for system in self.system_ids):
            raise ValueError("system IDs must be uppercase four-character PDB IDs")
        return self


class ExtractionIssue(BoundaryModel):
    """One requested source system excluded without interpolation or guessing."""

    source_system_id: str
    status: Literal["missing", "dropped"]
    reason: str


class ExtractionReport(BoundaryModel):
    """Portable QC and exact source/split provenance for a published subset."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    source: Provenance
    split_checksums: dict[Split, str]
    requested: tuple[str, ...]
    exported: dict[str, str]
    issues: tuple[ExtractionIssue, ...]


def load_misato_source(path: Path = MISATO_CONFIG) -> MisatoSource:
    """Validate scientific source metadata in the audited YAML configuration."""
    try:
        return MisatoSource.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise DataContractError(f"cannot load MISATO source {path}: {error}") from error


def _dataset_config(source: MisatoSource) -> DatasetConfig:
    features = tuple(
        FeatureDefinition(
            feature_id=name,
            unit=source.features[name].unit,
            description=source.features[name].description,
            definition=source.features[name].definition,
        )
        for name in NATIVE_BINDINGS
    )
    return DatasetConfig(
        dataset_id=DatasetId.MISATO,
        dataset_version=source.dataset_version,
        feature_set_version=source.feature_set_version,
        features=features,
    )


def _input_artifact(source: MisatoSource, kind: str) -> SourceArtifact:
    if kind == "sample":
        sample = source.audit_evidence
        return SourceArtifact(
            mode="md",
            url=sample.sample_url,
            size_bytes=sample.sample_size_bytes,
            checksum=f"sha256:{sample.sample_sha256}",
        )
    try:
        return source.artifacts["MD.hdf5"]
    except KeyError as error:
        raise DataContractError("MISATO config lacks MD.hdf5 artifact") from error


def _official_splits(source: MisatoSource, directory: Path) -> dict[str, Split]:
    assignments: dict[str, Split] = {}
    for split, filename in SPLIT_FILES.items():
        path = directory / filename
        try:
            artifact = source.artifacts[filename]
        except KeyError as error:
            raise DataContractError(
                f"MISATO config lacks official split {filename}"
            ) from error
        validate_artifact(path, artifact)
        _add_split(assignments, path.read_text(encoding="utf-8").splitlines(), split)
    return assignments


def _add_split(assignments: dict[str, Split], ids: list[str], split: Split) -> None:
    for system in ids:
        if not re.fullmatch(PDB_ID_PATTERN, system) or system in assignments:
            raise DataContractError(
                f"invalid/duplicate/overlapping official split ID: {system!r}"
            )
        assignments[system] = split


def _native_values(group: h5py.Group, source: MisatoSource) -> FloatArray:
    columns = []
    for name in NATIVE_BINDINGS:
        feature = source.features[name]
        key = feature.source_key
        if not isinstance(group.get(key, getlink=True), h5py.HardLink):
            raise DataContractError(f"missing or linked native channel {key}")
        dataset = group[key]
        _validate_native_dataset(dataset, feature)
        columns.append(dataset[:])
    return np.column_stack(columns)


def _validate_native_dataset(dataset: h5py.Dataset, feature: NativeFeature) -> None:
    if not isinstance(dataset, h5py.Dataset):
        raise DataContractError(f"{feature.source_key} is not a dataset")
    if dataset.shape != feature.shape or dataset.dtype != np.dtype(feature.dtype):
        raise DataContractError(f"invalid shape/dtype for {feature.source_key}")


def _manifest(
    system: str, split: Split, source: MisatoSource, provenance: Provenance
) -> TrajectoryManifest:
    system_id = stable_system_id(DatasetId.MISATO, system)
    return TrajectoryManifest.model_validate(
        provenance.model_dump()
        | {
            "system_id": system_id,
            "pdb_id": system,
            "protein_id": None,
            "ligand_id": None,
            "trajectory_id": stable_trajectory_id(system_id, system, NATIVE_REPLICA),
            "replicate_id": NATIVE_REPLICA,
            "split_group_id": system_id,
            "split": split,
            "frame_count": source.time_axis.frame_count,
            "feature_set_version": source.feature_set_version,
        }
    )


def _extract_one(
    file: h5py.File,
    system: str,
    source: MisatoSource,
    dataset: DatasetConfig,
    manifest: TrajectoryManifest,
) -> pa.Table:
    if not isinstance(file.get(system, getlink=True), h5py.HardLink):
        raise DataContractError("source system is linked rather than a local group")
    group = file[system]
    if not isinstance(group, h5py.Group):
        raise DataContractError("source system is not a group")
    time = np.arange(manifest.frame_count, dtype=np.float64)
    return to_arrow(manifest, dataset, time, _native_values(group, source))


def _export(
    file: h5py.File,
    config: ExtractionConfig,
    source: MisatoSource,
    splits: dict[str, Split],
    provenance: Provenance,
    staging: Path,
) -> ExtractionReport:
    dataset = _dataset_config(source)
    records: list[TrajectoryManifest] = []
    issues: list[ExtractionIssue] = []
    exported: dict[str, str] = {}
    for system in sorted(config.system_ids):
        entry = _export_system(
            file, system, source, dataset, splits, provenance, staging
        )
        if isinstance(entry, ExtractionIssue):
            issues.append(entry)
        else:
            records.append(entry)
            exported[entry.trajectory_id] = f"{system}.parquet"
    write_registry(
        staging / REGISTRY_FILE, Registry(dataset=dataset, trajectories=tuple(records))
    )
    return ExtractionReport(
        source=provenance,
        split_checksums={
            split: source.artifacts[name].checksum
            for split, name in SPLIT_FILES.items()
        },
        requested=tuple(sorted(config.system_ids)),
        exported=exported,
        issues=tuple(issues),
    )


def _export_system(
    file: h5py.File,
    system: str,
    source: MisatoSource,
    dataset: DatasetConfig,
    splits: dict[str, Split],
    provenance: Provenance,
    staging: Path,
) -> TrajectoryManifest | ExtractionIssue:
    if system not in splits:
        return ExtractionIssue(
            source_system_id=system, status="dropped", reason="not in official splits"
        )
    if system not in file:
        return ExtractionIssue(
            source_system_id=system, status="missing", reason="absent from input HDF5"
        )
    record = _manifest(system, splits[system], source, provenance)
    try:
        table = _extract_one(file, system, source, dataset, record)
    except DataContractError as error:
        return ExtractionIssue(
            source_system_id=system, status="dropped", reason=str(error)
        )
    write_series(staging / f"{system}.parquet", table)
    return record


def _check_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise DataContractError(
            f"output already exists: {path}; choose a new directory"
        )


def extract_misato(source: MisatoSource, config: ExtractionConfig) -> ExtractionReport:
    """Verify inputs, extract native arrays, and publish a complete directory.

    Scientific invalidity excludes that system with a QC reason. Storage/input
    integrity failures abort publication; an existing output is never overwritten.
    """
    _check_output(config.output_dir)
    try:
        artifact = _input_artifact(source, config.artifact)
        validate_artifact(config.input_path, artifact)
        splits = _official_splits(source, config.splits_dir)
        provenance = Provenance(
            dataset_id=DatasetId.MISATO,
            dataset_version=source.dataset_version,
            source_record=source.source_record,
            source_checksum=artifact.checksum,
            license_id=source.license.id,
            provenance_uri=artifact.url,
        )
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=config.output_dir.parent, prefix=".misato-"
        ) as directory:
            staging = Path(directory) / "output"
            staging.mkdir()
            with h5py.File(config.input_path, "r") as file:
                report = _export(file, config, source, splits, provenance, staging)
            with atomic_output(staging / QC_FILE) as temporary:
                temporary.write_text(
                    report.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
            _check_output(config.output_dir)
            # ponytail: one writer; add no-replace rename for concurrent publishers.
            staging.rename(config.output_dir)
    except (OSError, UnicodeError) as error:
        raise DataContractError(
            f"MISATO extraction failed before publication: {error}"
        ) from error
    logger.info(
        "MISATO extraction: exported=%d excluded=%d",
        len(report.exported),
        len(report.issues),
    )
    return report
