"""Canonical registry serialization and atomic publication."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self

from pydantic import ValidationError, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.schemas import (
    BoundaryModel,
    DatasetConfig,
    SchemaVersion,
    TrajectoryManifest,
)


class Registry(BoundaryModel):
    """Small trajectory-level index, not an in-memory table of all frames."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    dataset: DatasetConfig
    trajectories: tuple[TrajectoryManifest, ...]

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        """Reject duplicates and records inconsistent with the declared dataset."""
        ids = [record.trajectory_id for record in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate trajectory IDs")
        for record in self.trajectories:
            _validate_dataset(record, self.dataset)
        _validate_systems(self.trajectories)
        _validate_groups(self.trajectories)
        return self


def _validate_dataset(record: TrajectoryManifest, config: DatasetConfig) -> None:
    fields = ("dataset_id", "dataset_version", "feature_set_version")
    if any(getattr(record, field) != getattr(config, field) for field in fields):
        raise ValueError("trajectory dataset/version/feature set differs from registry")


def _validate_systems(records: tuple[TrajectoryManifest, ...]) -> None:
    identities: dict[str, tuple[str | None, ...]] = {}
    for record in records:
        identity = (record.pdb_id, record.protein_id, record.ligand_id)
        if identities.setdefault(record.system_id, identity) != identity:
            raise ValueError(f"conflicting identity for system {record.system_id}")


def _validate_groups(records: tuple[TrajectoryManifest, ...]) -> None:
    assignments: dict[str, str] = {}
    for record in records:
        if record.split is not None:
            if (
                assignments.setdefault(record.split_group_id, record.split)
                != record.split
            ):
                raise ValueError(
                    f"split group {record.split_group_id} crosses partitions"
                )


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Publish completed files only; preserve previous output on failure."""
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_registry(path: Path, registry: Registry) -> None:
    """Write stable trajectory order and portable JSON metadata atomically."""
    records = sorted(registry.trajectories, key=lambda record: record.trajectory_id)
    ordered = Registry(dataset=registry.dataset, trajectories=tuple(records))
    try:
        with atomic_output(path) as temporary:
            temporary.write_text(
                ordered.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
    except OSError as error:
        raise DataContractError(f"cannot write registry {path}: {error}") from error


def read_registry(path: Path) -> Registry:
    """Fail closed for invalid records or incompatible schema versions."""
    try:
        return Registry.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise DataContractError(f"cannot read registry {path}: {error}") from error
