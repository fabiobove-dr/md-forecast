"""Metadata-only cohort selection and purpose-scoped canonical series access."""

import hashlib
from collections.abc import Collection, Mapping
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal, Self

import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.constants import CHECKSUM_PATTERN, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.public.mdbind import MDBindSubset, ReplicaSource
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    Identifier,
    TrajectoryManifest,
)
from md_forecast.data.series import read_series

type CohortRole = Literal["train", "validation", "calibration", "confirmation"]
type CohortPurpose = Literal["tuning", "calibration", "confirmation"]
type PDBId = Annotated[str, Field(pattern=r"^[0-9][A-Z0-9]{3}$")]
ROLE_SPLITS: dict[CohortRole, Split] = {
    "train": Split.TRAIN,
    "validation": Split.VALIDATION,
    "calibration": Split.VALIDATION,
    "confirmation": Split.TEST,
}
PURPOSE_ROLES: dict[CohortPurpose, frozenset[CohortRole]] = {
    "tuning": frozenset({"train", "validation"}),
    "calibration": frozenset({"calibration"}),
    "confirmation": frozenset({"confirmation"}),
}

type MDbindAccession = Annotated[str, Field(pattern=r"^MD-[A-Z0-9]+$")]


class ExternalReserve(BoundaryModel):
    """Catalog identities for separate replica and unseen-complex tasks."""

    seed: Annotated[int, Field(strict=True, ge=0)]
    catalog_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    eligible_unique_complexes: Annotated[int, Field(strict=True, gt=0)]
    all_official_misato_identity_excluded: Annotated[int, Field(strict=True, gt=0)]
    excluded_previous_ids: tuple[PDBId, ...]
    selection_rule: Identifier
    confirmation: Annotated[
        tuple[tuple[PDBId, MDbindAccession], ...], Field(min_length=1)
    ]
    development_seen_replica: Annotated[
        tuple[tuple[PDBId, MDbindAccession], ...], Field(min_length=1)
    ]
    seen_replica_roles: dict[
        CohortRole, tuple[Annotated[int, Field(strict=True, ge=1, le=10)], ...]
    ]
    original_confirmation_count: Annotated[int, Field(strict=True, gt=0)] | None = (
        Field(default=None, exclude_if=lambda value: value is None)
    )
    parent_reserve_hash: ArtifactHash | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    payload_status: Identifier

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Reject intersecting external tasks, excluded systems and replica roles."""
        pairs = self.confirmation + self.development_seen_replica
        ids = tuple(pdb for pdb, _ in pairs)
        if len(set(ids)) != len(ids) or set(ids) & set(self.excluded_previous_ids):
            raise ValueError(
                "external tasks overlap or contain previously used systems"
            )
        self._validate_extension()
        self._validate_population()
        self._validate_replicas()
        return self

    def _validate_extension(self) -> None:
        if (self.original_confirmation_count is None) != (
            self.parent_reserve_hash is None
        ):
            raise ValueError(
                "reserve extension requires original count and parent hash"
            )
        if (
            self.original_confirmation_count is not None
            and self.original_confirmation_count >= len(self.confirmation)
        ):
            raise ValueError("reserve extension must add confirmation identities")

    def _validate_population(self) -> None:
        if (
            len(self.confirmation + self.development_seen_replica)
            > self.eligible_unique_complexes
        ):
            raise ValueError("external selection exceeds eligible population")

    def _validate_replicas(self) -> None:
        if set(self.seen_replica_roles) != set(ROLE_SPLITS):
            raise ValueError("seen-replica protocol requires all four roles")
        replicas = tuple(chain.from_iterable(self.seen_replica_roles.values()))
        if sorted(replicas) != list(range(1, 11)):
            raise ValueError("seen-replica roles must partition all ten replicas")
        if any(not ids for ids in self.seen_replica_roles.values()):
            raise ValueError("seen-replica roles cannot be empty")

    def role(self, pdb_id: str, accession: str, replica: int) -> CohortRole:
        """Bind exact catalog accession before a raw source may be decoded."""
        self._validate_identity(pdb_id, accession, replica)
        if pdb_id in dict(self.confirmation):
            return "confirmation"
        return next(
            role for role, ids in self.seen_replica_roles.items() if replica in ids
        )

    def _validate_identity(self, pdb_id: str, accession: str, replica: int) -> None:
        selected = dict(self.confirmation + self.development_seen_replica)
        base = selected.get(pdb_id)
        expected = base if replica == 1 else f"{base}.{replica}"
        if base is None or accession != expected or not 1 <= replica <= 10:
            raise DataContractError("foreign or altered reserved external source")

    def permitted_sources(
        self, subset: MDBindSubset, purpose: CohortPurpose
    ) -> tuple[ReplicaSource, ...]:
        """Select allowed whole roles before any raw payload read."""
        self._check_source_binding(subset)
        sources = tuple(
            source
            for source in subset.replicas
            if self.role(source.pdb_id, source.accession, source.replica)
            in PURPOSE_ROLES[purpose]
        )
        if not sources:
            raise DataContractError(
                "external subset contains no permitted role for this purpose"
            )
        return sources

    def _check_source_binding(self, subset: MDBindSubset) -> None:
        if subset.reserve_hash != metadata_hash(self):
            raise DataContractError("external source differs from frozen role policy")


def permitted_external_sources(
    subset: MDBindSubset, reserve: ExternalReserve | None, purpose: CohortPurpose
) -> tuple[ReplicaSource, ...]:
    """Require the frozen policy for reserved sources, including QC callers."""
    if purpose not in PURPOSE_ROLES:
        raise DataContractError("unknown external cohort process purpose")
    if reserve is None:
        if subset.reserve_hash is not None:
            raise DataContractError(
                "reserved source requires an explicit role access policy"
            )
        return subset.replicas
    snapshot = ExternalReserve.model_validate_json(reserve.model_dump_json())
    return snapshot.permitted_sources(subset, purpose)


class CohortConfig(BoundaryModel):
    """System counts and ranking seed, declared before outcomes or windows."""

    seed: Annotated[int, Field(strict=True, ge=0)]
    counts: dict[CohortRole, Annotated[int, Field(strict=True, gt=0)]]

    @model_validator(mode="after")
    def complete_roles(self) -> Self:
        """Require selection, separate calibration and untouched confirmation."""
        if set(self.counts) != set(ROLE_SPLITS):
            raise ValueError("cohort requires all four roles")
        return self


class CohortManifest(BoundaryModel):
    """Portable source-linked system assignments without decoded outcomes."""

    config: CohortConfig
    source_hash: ArtifactHash
    source_checksum: Annotated[str, Field(pattern=f"^{CHECKSUM_PATTERN}$")]
    geometry_hash: ArtifactHash
    dataset_hash: ArtifactHash
    official_list_hashes: dict[Split, ArtifactHash]
    roles: dict[CohortRole, tuple[PDBId, ...]]
    original_splits: dict[PDBId, Split]
    excluded_ids: tuple[PDBId, ...]
    eligible_counts: dict[Split, Annotated[int, Field(strict=True, ge=0)]]

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        """Reject intersecting roles, altered official splits or declared counts."""
        if set(self.roles) != set(ROLE_SPLITS):
            raise ValueError("cohort assignments require all four roles")
        if set(self.official_list_hashes) != set(Split):
            raise ValueError("cohort requires all official list identities")
        if set(self.eligible_counts) != set(Split):
            raise ValueError("cohort requires all eligible population counts")
        self._validate_membership()
        self._validate_populations()
        for role, expected in ROLE_SPLITS.items():
            self._validate_role(role, expected)
        return self

    def _validate_membership(self) -> None:
        ids = tuple(chain.from_iterable(self.roles.values()))
        if len(set(ids)) != len(ids) or set(ids) & set(self.excluded_ids):
            raise ValueError("cohort roles overlap or contain excluded systems")
        if set(self.original_splits) != set(ids):
            raise ValueError("original splits must cover exactly assigned systems")

    def _validate_populations(self) -> None:
        for split in Split:
            assigned = sum(
                self.config.counts[role]
                for role, label in ROLE_SPLITS.items()
                if label == split
            )
            if assigned > self.eligible_counts[split]:
                raise ValueError("assigned systems exceed eligible source population")

    def _validate_role(self, role: CohortRole, expected: Split) -> None:
        if len(self.roles[role]) != self.config.counts[role]:
            raise ValueError("cohort counts differ from declared selection")
        if any(self.original_splits[pdb] != expected for pdb in self.roles[role]):
            raise ValueError("role differs from original official split")


def select_systems(
    config: CohortConfig,
    official: Mapping[str, Split],
    available: Collection[str],
    excluded: Collection[str],
) -> dict[CohortRole, tuple[str, ...]]:
    """Rank only source identity metadata; never accept windows or future values."""
    config = CohortConfig.model_validate_json(config.model_dump_json())
    pools = _eligible_pools(official, available, excluded, config.seed)
    roles = {}
    for role, split in ROLE_SPLITS.items():
        count = config.counts[role]
        if len(pools[split]) < count:
            raise DataContractError(f"insufficient eligible systems for {role}")
        roles[role] = tuple(pools[split][:count])
        pools[split] = pools[split][count:]
    return roles


def _eligible_pools(
    official: Mapping[str, Split],
    available: Collection[str],
    excluded: Collection[str],
    seed: int,
) -> dict[Split, list[str]]:
    pools: dict[Split, list[str]] = {split: [] for split in Split}
    for pdb in set(available) - set(excluded):
        split = official.get(pdb)
        if split is not None:
            pools[split].append(pdb)
    return {
        split: sorted(
            ids,
            key=lambda pdb: hashlib.sha256(f"{seed}|{split}|{pdb}".encode()).digest(),
        )
        for split, ids in pools.items()
    }


class CohortLoader:
    """Reject foreign/calibration/confirmation records before any file read."""

    def __init__(
        self,
        root: Path,
        registry: Registry,
        manifest: CohortManifest,
        exported: Mapping[str, str],
        *,
        purpose: CohortPurpose,
    ) -> None:
        """Bind exact records and explicit permitted roles for this process."""
        if purpose not in PURPOSE_ROLES:
            raise DataContractError("unknown cohort process purpose")
        self.root = root
        self.registry = Registry.model_validate_json(registry.model_dump_json())
        self.manifest = CohortManifest.model_validate_json(manifest.model_dump_json())
        self._check_dataset()
        self.exported = dict(exported)
        self.roles = PURPOSE_ROLES[purpose]
        self.records = {r.trajectory_id: r for r in self.registry.trajectories}
        self.pdb_roles = {
            pdb: role for role, ids in self.manifest.roles.items() for pdb in ids
        }

    def _check_dataset(self) -> None:
        if metadata_hash(self.registry.dataset) != self.manifest.dataset_hash:
            raise DataContractError(
                "cohort dataset differs from frozen feature contract"
            )

    def __call__(self, record: TrajectoryManifest) -> pa.Table:
        """Validate scope before opening a canonical table, including forged labels."""
        self._validate_access(record)
        name = self.exported.get(record.trajectory_id)
        if name is None:
            raise DataContractError("permitted cohort record has no exported table")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise DataContractError("cohort table path escapes its source directory")
        expected = Registry(dataset=self.registry.dataset, trajectories=(record,))
        return read_series(path, expected)

    def _validate_access(self, record: TrajectoryManifest) -> None:
        if record.pdb_id is None or self.records.get(record.trajectory_id) != record:
            raise DataContractError("foreign or altered cohort record")
        role = self.pdb_roles.get(record.pdb_id)
        if role not in self.roles:
            raise DataContractError(
                "cohort record is outside this process's role scope"
            )
        self._validate_provenance(record, record.pdb_id)

    def _validate_provenance(self, record: TrajectoryManifest, pdb_id: str) -> None:
        if record.source_checksum != self.manifest.source_checksum:
            raise DataContractError("cohort record differs from frozen source bytes")
        if record.split != self.manifest.original_splits[pdb_id]:
            raise DataContractError("cohort record changed its original official split")
