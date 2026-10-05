"""Metadata-only selection and role guards prevent tuning/test contamination."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.cohort import (
    ROLE_SPLITS,
    CohortConfig,
    CohortLoader,
    CohortManifest,
    CohortPurpose,
    ExternalReserve,
    permitted_external_sources,
    select_systems,
)
from md_forecast.data.public.mdbind import MDBindSubset, ReplicaSource, SubsetSelection
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import DatasetConfig, TrajectoryManifest
from md_forecast.data.series import to_arrow, write_series


def selection() -> CohortManifest:
    """A tiny independent official-TRAIN/VAL/TEST selection."""
    roles = dict(
        zip(ROLE_SPLITS, (("1A01",), ("1A02",), ("1A03",), ("1A04",)), strict=True)
    )
    return CohortManifest(
        config=CohortConfig(seed=42, counts=dict.fromkeys(ROLE_SPLITS, 1)),
        source_hash="sha256:" + "a" * 64,
        source_checksum="sha256:" + "a" * 64,
        geometry_hash="sha256:" + "b" * 64,
        dataset_hash=metadata_hash(fixture_dataset()),
        official_list_hashes=dict.fromkeys(Split, "sha256:" + "c" * 64),
        roles=roles,
        original_splits={ids[0]: ROLE_SPLITS[role] for role, ids in roles.items()},
        excluded_ids=(),
        eligible_counts={Split.TRAIN: 1, Split.VALIDATION: 2, Split.TEST: 1},
    )


def fixture_dataset() -> DatasetConfig:
    """Small synthetic canonical feature contract."""
    return DatasetConfig.model_validate(
        {
            "dataset_id": "misato",
            "dataset_version": "fixture",
            "feature_set_version": "fixture-v1",
            "features": [
                {
                    "feature_id": "signal",
                    "unit": "dimensionless",
                    "description": "Synthetic signal",
                    "definition": "Synthetic four-frame test signal",
                }
            ],
        }
    )


def source_registry(manifest: CohortManifest) -> Registry:
    """Canonical records preserve original source splits, including calibration."""
    dataset = fixture_dataset()
    records = tuple(
        TrajectoryManifest.model_validate(
            {
                "dataset_id": "misato",
                "dataset_version": "fixture",
                "source_record": "fixture",
                "source_checksum": "sha256:" + "a" * 64,
                "license_id": "MIT",
                "provenance_uri": "https://example.org/fixture",
                "system_id": pdb,
                "pdb_id": pdb,
                "protein_id": None,
                "ligand_id": None,
                "trajectory_id": pdb,
                "replicate_id": "native",
                "split_group_id": pdb,
                "frame_count": 4,
                "feature_set_version": "fixture-v1",
                "split": split,
            }
        )
        for pdb, split in manifest.original_splits.items()
    )
    return Registry(dataset=dataset, trajectories=records)


def test_selection_depends_only_on_metadata_and_rejects_small_population() -> None:
    manifest = selection()
    official = manifest.original_splits | {"1A05": Split.TRAIN}
    first = select_systems(manifest.config, official, tuple(official), ("1A05",))
    assert first == select_systems(
        manifest.config,
        dict(reversed(tuple(official.items()))),
        set(official),
        ("1A05",),
    )
    assert first == manifest.roles
    with pytest.raises(DataContractError, match="insufficient"):
        select_systems(manifest.config, official, (), ())
    with pytest.raises(ValidationError, match="all four roles"):
        CohortConfig(seed=42, counts={"train": 1})


@pytest.mark.parametrize(
    "change",
    [
        {"roles": {"train": ("1A01",)}},
        {"excluded_ids": ("1A01",)},
        {"original_splits": {}},
        {"official_list_hashes": {}},
        {"eligible_counts": {}},
        {"eligible_counts": {"train": 0, "validation": 2, "test": 1}},
        {
            "original_splits": {
                "1A01": "test",
                "1A02": "validation",
                "1A03": "validation",
                "1A04": "test",
            }
        },
        {
            "roles": {
                "train": ("1A01", "1A05"),
                "validation": ("1A02",),
                "calibration": ("1A03",),
                "confirmation": ("1A04",),
            },
            "original_splits": {
                "1A01": "train",
                "1A05": "train",
                "1A02": "validation",
                "1A03": "validation",
                "1A04": "test",
            },
            "eligible_counts": {"train": 2, "validation": 2, "test": 1},
        },
        {
            "roles": {
                "train": ("1A01",),
                "validation": ("1A02",),
                "calibration": ("1A02",),
                "confirmation": ("1A04",),
            }
        },
    ],
)
def test_manifest_rejects_unsafe_roles(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CohortManifest.model_validate(selection().model_dump() | change)


def test_loader_blocks_forbidden_and_forged_records_before_filesystem_read(
    tmp_path: Path,
) -> None:
    manifest = selection()
    registry = source_registry(manifest)
    exports = {r.trajectory_id: f"{r.pdb_id}.parquet" for r in registry.trajectories}
    loader = CohortLoader(tmp_path, registry, manifest, exports, purpose="tuning")
    for record in registry.trajectories[2:]:
        with pytest.raises(DataContractError, match="role scope"):
            loader(record)
    forged = registry.trajectories[-1].model_copy(update={"split": Split.TRAIN})
    with pytest.raises(DataContractError, match="altered"):
        loader(forged)
    record = registry.trajectories[0]
    table = to_arrow(record, registry.dataset, np.arange(4.0), np.arange(4.0)[:, None])
    write_series(tmp_path / exports[record.trajectory_id], table)
    assert loader(record).equals(table)
    manifest.roles["train"] = ("1A04",)
    with pytest.raises(DataContractError, match="role scope"):
        loader(registry.trajectories[-1])


@pytest.mark.parametrize(
    ("purpose", "index"), [("tuning", 1), ("calibration", 2), ("confirmation", 3)]
)
def test_loader_only_reads_its_declared_role(
    tmp_path: Path, purpose: CohortPurpose, index: int
) -> None:
    manifest = selection()
    registry = source_registry(manifest)
    record = registry.trajectories[index]
    table = to_arrow(record, registry.dataset, np.arange(4.0), np.arange(4.0)[:, None])
    write_series(tmp_path / "series.parquet", table)
    loader = CohortLoader(
        tmp_path,
        registry,
        manifest,
        {record.trajectory_id: "series.parquet"},
        purpose=purpose,
    )
    assert loader(record).equals(table)
    for other in registry.trajectories:
        if other != record and (purpose != "tuning" or other.split != Split.TRAIN):
            with pytest.raises(DataContractError, match="role scope"):
                loader(other)


def test_loader_rejects_unknown_purpose_and_missing_identity(tmp_path: Path) -> None:
    manifest = selection()
    registry = source_registry(manifest)
    with pytest.raises(DataContractError, match="purpose"):
        CohortLoader(tmp_path, registry, manifest, {}, purpose="unknown")  # type: ignore[arg-type]
    loader = CohortLoader(tmp_path, registry, manifest, {}, purpose="tuning")
    record = registry.trajectories[0].model_copy(update={"pdb_id": None})
    with pytest.raises(DataContractError, match="foreign"):
        loader(record)


def test_loader_rejects_changed_dataset_and_payload_source(tmp_path: Path) -> None:
    manifest = selection()
    registry = source_registry(manifest)
    changed = registry.model_copy(
        update={
            "dataset": registry.dataset.model_copy(
                update={"dataset_version": "changed"}
            )
        }
    )
    with pytest.raises((DataContractError, ValidationError)):
        CohortLoader(tmp_path, changed, manifest, {}, purpose="tuning")
    record = registry.trajectories[0].model_copy(
        update={"source_checksum": "sha256:" + "d" * 64}
    )
    changed = registry.model_copy(update={"trajectories": (record,)})
    loader = CohortLoader(tmp_path, changed, manifest, {}, purpose="tuning")
    with pytest.raises(DataContractError, match="source bytes"):
        loader(record)


def test_loader_missing_exports_escaped_paths_and_changed_original_split(
    tmp_path: Path,
) -> None:
    manifest = selection()
    registry = source_registry(manifest)
    record = registry.trajectories[0]
    loader = CohortLoader(tmp_path, registry, manifest, {}, purpose="tuning")
    with pytest.raises(DataContractError, match="no exported"):
        loader(record)
    loader.exported[record.trajectory_id] = "../outside.parquet"
    with pytest.raises(DataContractError, match="escapes"):
        loader(record)
    changed = record.model_copy(update={"split": Split.TEST})
    altered_registry = Registry(dataset=registry.dataset, trajectories=(changed,))
    loader = CohortLoader(tmp_path, altered_registry, manifest, {}, purpose="tuning")
    with pytest.raises(DataContractError, match="original official split"):
        loader(changed)


def external_reserve() -> ExternalReserve:
    """Two distinct tasks with explicit, complete replica roles."""
    return ExternalReserve(
        seed=42,
        catalog_sha256="c" * 64,
        eligible_unique_complexes=2,
        all_official_misato_identity_excluded=10,
        excluded_previous_ids=(),
        selection_rule="Synthetic identity-only rule",
        confirmation=(("1B01", "MD-A001AA"),),
        development_seen_replica=(("1B02", "MD-A002AA"),),
        seen_replica_roles={
            "train": (1, 2, 3, 4, 5, 6),
            "validation": (7, 8),
            "calibration": (9,),
            "confirmation": (10,),
        },
        payload_status="No future payloads decoded",
    )


def test_external_reservation_keeps_tasks_and_replica_roles_distinct() -> None:
    reserve = external_reserve()
    for replica in range(1, 11):
        suffix = "" if replica == 1 else f".{replica}"
        assert reserve.role("1B01", "MD-A001AA" + suffix, replica) == "confirmation"
        expected = next(
            role for role, ids in reserve.seen_replica_roles.items() if replica in ids
        )
        assert reserve.role("1B02", "MD-A002AA" + suffix, replica) == expected
    for pdb, accession, replica in (
        ("1B03", "MD-A003AA", 1),
        ("1B02", "MD-A001AA", 1),
        ("1B02", "MD-A002AA.7", 1),
        ("1B02", "MD-A002AA.11", 11),
    ):
        with pytest.raises(DataContractError, match="foreign or altered"):
            reserve.role(pdb, accession, replica)


@pytest.mark.parametrize(
    "change",
    [
        {"excluded_previous_ids": ("1B01",)},
        {"eligible_unique_complexes": 1},
        {"development_seen_replica": (("1B01", "MD-A001AA"),)},
        {"seen_replica_roles": {}},
        {
            "seen_replica_roles": {
                "train": (1, 2, 3, 4, 5, 6),
                "validation": (6, 8),
                "calibration": (9,),
                "confirmation": (10,),
            }
        },
        {
            "seen_replica_roles": {
                "train": tuple(range(1, 8)),
                "validation": (8,),
                "calibration": (),
                "confirmation": (9, 10),
            }
        },
    ],
)
def test_external_reservation_rejects_overlap_and_incomplete_roles(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ExternalReserve.model_validate(external_reserve().model_dump() | change)


def reserved_subset(reserve: ExternalReserve) -> MDBindSubset:
    """Frozen identities with no payload files; role filtering must not read them."""
    files = {
        name: {
            "mode": "metadata",
            "url": f"https://example.org/{name}",
            "size_bytes": 1,
            "checksum": "sha256:" + "a" * 64,
        }
        for name in (
            "record.json",
            "structure.pdb",
            "topology.prmtop",
            "trajectory.xtc",
        )
    }
    sources = tuple(
        ReplicaSource.model_validate(
            {
                "accession": base if replica == 1 else f"{base}.{replica}",
                "pdb_id": pdb,
                "replica": replica,
                "atoms": 2,
                "ligand_residue_index": 1,
                "ligand_name": "LIG",
                "interaction_selection": "resid 2",
                "files": files,
            }
        )
        for pdb, base in reserve.confirmation + reserve.development_seen_replica
        for replica in range(1, 11)
    )
    return MDBindSubset(
        selection=SubsetSelection(
            selection_rule="Synthetic identity-only rule",
            catalog_sha256="c" * 64,
            selected=reserve.confirmation + reserve.development_seen_replica,
        ),
        replicas=sources,
        reserve_hash=metadata_hash(reserve),
    )


def test_external_payload_preparation_requires_exact_policy_and_whole_roles() -> None:
    reserve = external_reserve()
    subset = reserved_subset(reserve)
    expected = {"tuning": set(range(1, 9)), "calibration": {9}, "confirmation": {10}}
    for purpose, replicas in expected.items():
        allowed = permitted_external_sources(subset, reserve, purpose)  # type: ignore[arg-type]
        seen = {s.replica for s in allowed if s.pdb_id == "1B02"}
        assert seen == replicas
        assert sum(s.pdb_id == "1B01" for s in allowed) == (
            10 if purpose == "confirmation" else 0
        )
    with pytest.raises(DataContractError, match="explicit role access policy"):
        permitted_external_sources(subset, None, "tuning")
    changed = reserve.model_copy(update={"seed": 43})
    with pytest.raises(DataContractError, match="frozen role policy"):
        permitted_external_sources(subset, changed, "tuning")
    with pytest.raises(DataContractError, match="purpose"):
        permitted_external_sources(subset, reserve, "unknown")  # type: ignore[arg-type]
    legacy = subset.model_copy(update={"reserve_hash": None})
    assert permitted_external_sources(legacy, None, "tuning") == legacy.replicas
    unseen = MDBindSubset(
        selection=subset.selection.model_copy(
            update={"selected": reserve.confirmation}
        ),
        replicas=subset.replicas[:10],
        reserve_hash=subset.reserve_hash,
    )
    with pytest.raises(DataContractError, match="no permitted role"):
        permitted_external_sources(unseen, reserve, "tuning")
