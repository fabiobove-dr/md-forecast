"""Audited raw MDbind replicas mapped to the common dry-system geometry."""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import pyarrow as pa
from pydantic import ConfigDict, Field, model_validator

from md_forecast.core.constants import DatasetId, SamplingStatus, TimeUnit
from md_forecast.core.exceptions import AcquisitionError, DataContractError
from md_forecast.data.acquisition import SourceArtifact, validate_artifact
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.schemas import (
    BoundaryModel,
    DatasetConfig,
    FeatureDefinition,
    TrajectoryManifest,
    stable_system_id,
    stable_trajectory_id,
)
from md_forecast.data.series import FloatArray, to_arrow
from md_forecast.features.structural import (
    ANGSTROM_PER_NM,
    BASE_FEATURE_IDS,
    AtomPair,
    StructuralConfig,
    contact_geometry,
    validate_sasa_backend,
)

RAW_FILES = frozenset(
    {"record.json", "structure.pdb", "topology.prmtop", "trajectory.xtc"}
)
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
COMMON_VERSION = "common-dry-geometry-v1"


class ReplicaSource(BoundaryModel):
    """Frozen payload identities and a reviewed final-residue ligand selection."""

    accession: Annotated[str, Field(pattern=r"^MD-[A-Z0-9]+(?:\.[2-9]|\.10)?$")]
    pdb_id: Annotated[str, Field(pattern=r"^[A-Z0-9]{4}$")]
    replica: Annotated[int, Field(strict=True, ge=1, le=10)]
    atoms: Annotated[int, Field(strict=True, gt=0)]
    frames: Literal[50] = 50
    ligand_residue_index: Annotated[int, Field(strict=True, ge=0)]
    ligand_name: Annotated[str, Field(min_length=1)]
    interaction_selection: Annotated[str, Field(min_length=1)]
    license_id: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    frame_interval_ps: Annotated[float, Field(ge=200, le=200)] = 200.0
    files: dict[str, SourceArtifact]

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Require all source bytes and one-based replica/accession agreement."""
        if set(self.files) != RAW_FILES:
            raise ValueError("raw replica requires record, PDB, AMBER topology and XTC")
        suffix = int(self.accession.split(".")[1]) if "." in self.accession else 1
        if suffix != self.replica:
            raise ValueError("accession suffix differs from replica number")
        return self


class RawMetadata(BoundaryModel):
    """Only the upstream fields needed for source/selection admission."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    PDBIDS: tuple[str, ...]
    SYSTATS: Annotated[int, Field(strict=True, gt=0)]
    SOLVATS: Literal[0]
    LINKCENSE: Literal["https://creativecommons.org/licenses/by/4.0/"]
    INTERACTIONS: tuple[dict[str, str], ...]


class RawRecord(BoundaryModel):
    """Preserve raw replica numbering rather than parsing a display name."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    metadata: RawMetadata
    mdNumber: Annotated[int, Field(strict=True, ge=1, le=10)]
    mdIndex: Annotated[int, Field(strict=True, ge=0, le=9)]
    mdcount: Literal[10]


def common_features(config: StructuralConfig) -> tuple[FeatureDefinition, ...]:
    """Same physical formula; source-specific atom identities remain provenance.

    Receptor is every dry-source atom outside the reviewed final ligand segment,
    including retained ions/cofactors. Heavy means atomic number greater than one.
    """
    selection = (
        "Receptor: all dry-source atoms before the final ligand segment, including "
        "retained ions/cofactors; ligand: the reviewed final segment. Heavy atoms "
        "have atomic number > 1. Raw Cartesian Angstrom distances; no PBC repair."
    )
    original = config.feature_definitions()[: len(BASE_FEATURE_IDS)]
    formulas = (
        "Count receptor-ligand heavy atom pairs at distance <= "
        f"{config.contact_cutoff_angstrom!r} Angstrom.",
        "Fraction of first-retained-frame heavy atom-pair contacts remaining "
        f"at distance <= {config.contact_cutoff_angstrom!r} Angstrom; "
        "reject zero reference contacts.",
        "Minimum receptor-ligand heavy atom-pair distance.",
    )
    return tuple(
        feature.model_copy(update={"definition": f"{formula} {selection}"})
        for feature, formula in zip(original, formulas, strict=True)
    )


def common_dataset(config: StructuralConfig, dataset_id: DatasetId) -> DatasetConfig:
    """Declare a source-independent common feature version for each dataset."""
    features = common_features(config)
    identity = metadata_hash(config).split(":")[1]
    if config.regions or config.ligand_sasa is not None:
        raise DataContractError(
            "common geometry admits exactly the three core channels"
        )
    return DatasetConfig(
        dataset_id=dataset_id,
        dataset_version="audited-2026-10-03",
        feature_set_version=f"{COMMON_VERSION}-{identity}",
        features=features,
    )


def verify_files(directory: Path, source: ReplicaSource) -> None:
    """Stream-check every frozen source before any topology or target is read."""
    for name, artifact in source.files.items():
        try:
            validate_artifact(directory / name, artifact)
        except AcquisitionError as error:
            raise DataContractError(
                f"MDbind source integrity failed: {error}"
            ) from error


def _verify_metadata(directory: Path, source: ReplicaSource) -> None:
    record = RawRecord.model_validate_json((directory / "record.json").read_text())
    metadata = record.metadata
    if (
        metadata.PDBIDS != (source.pdb_id,)
        or metadata.SYSTATS != source.atoms
        or (record.mdNumber, record.mdIndex) != (source.replica, source.replica - 1)
    ):
        raise DataContractError("MDbind source identity/atom count differs")
    _verify_interaction(metadata, source)


def _verify_interaction(metadata: RawMetadata, source: ReplicaSource) -> None:
    interactions = [
        item for item in metadata.INTERACTIONS if item.get("type") == "protein-ligand"
    ]
    if len(interactions) != 1:
        raise DataContractError("MDbind requires one reviewed ligand interaction")
    if interactions[0].get("selection_2") != source.interaction_selection:
        raise DataContractError("MDbind reviewed ligand selection differs")


def _topology_selection(directory: Path, source: ReplicaSource) -> tuple[Any, AtomPair]:
    import mdtraj

    topology = mdtraj.load_topology(str(directory / "structure.pdb"))
    amber = mdtraj.load_topology(str(directory / "topology.prmtop"))
    _verify_topologies(topology, amber, source.atoms)
    return topology, _heavy_selection(topology, source)


def _verify_topologies(topology: Any, amber: Any, atoms: int) -> None:
    if topology.n_atoms != atoms or amber.n_atoms != atoms:
        raise DataContractError("MDbind PDB/AMBER atom counts differ")
    if any(
        _atom_signature(a) != _atom_signature(b)
        for a, b in zip(topology.atoms, amber.atoms, strict=True)
    ):
        raise DataContractError("MDbind PDB/AMBER atom order or elements differ")


def _heavy_selection(topology: Any, source: ReplicaSource) -> AtomPair:
    ligand = topology.residue(source.ligand_residue_index)
    if ligand.index != topology.n_residues - 1 or ligand.name != source.ligand_name:
        raise DataContractError("MDbind ligand must match reviewed final residue")
    _verify_selection(topology, ligand, source.interaction_selection)
    return _heavy_atoms(topology, ligand)


def _heavy_atoms(topology: Any, ligand: Any) -> AtomPair:
    atoms = list(topology.atoms)
    _verify_elements(atoms)
    left = _heavy_indices(a for a in atoms if a.residue != ligand)
    right = _heavy_indices(ligand.atoms)
    return AtomPair(left=left, right=right)


def _heavy_indices(atoms: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(a.index) for a in atoms if a.element.atomic_number > 1)


def _verify_elements(atoms: list[Any]) -> None:
    if any(atom.element is None for atom in atoms):
        raise DataContractError("MDbind topology has missing elements")


def _atom_signature(atom: Any) -> tuple[Any, ...]:
    return atom.name, atom.residue.index, atom.residue.name, atom.element


def _verify_selection(topology: Any, ligand: Any, selection: str) -> None:
    words = selection.split()
    selected = [r.index for r in topology.residues if _selection_matches(r, words)]
    if selected != [ligand.index]:
        raise DataContractError(
            "MDbind source selection must identify exactly the reviewed ligand"
        )


def _selection_matches(residue: Any, words: list[str]) -> bool:
    if words[0] == "resid" and len(words) == 2:
        return str(residue.resSeq) == words[1]
    if words[0] == "chain":
        return residue.chain.chain_id in words[1:]
    raise DataContractError("unsupported reviewed MDbind ligand selection")


def extract_replica(
    directory: Path, source: ReplicaSource, config: StructuralConfig
) -> pa.Table:
    """Extract bounded geometry and actual XTC ps timestamps without fitting."""
    try:
        return _extract_replica(directory, source, config)
    except (OSError, ValueError, IndexError) as error:
        raise DataContractError(
            f"cannot extract MDbind {source.accession}: {error}"
        ) from error


def _extract_replica(
    directory: Path, source: ReplicaSource, config: StructuralConfig
) -> pa.Table:
    validate_sasa_backend()

    source = ReplicaSource.model_validate_json(source.model_dump_json())
    if source.atoms > config.max_atoms or source.frames > config.max_frames:
        raise DataContractError("MDbind source exceeds atom/frame budget")
    verify_files(directory, source)
    _verify_metadata(directory, source)
    topology, selections = _topology_selection(directory, source)
    time, values = _read_geometry(directory, source, config, topology, selections)
    return _canonical_replica(source, config, time, values)


def _read_geometry(
    directory: Path,
    source: ReplicaSource,
    config: StructuralConfig,
    topology: Any,
    selections: AtomPair,
) -> tuple[FloatArray, FloatArray]:
    import mdtraj

    values, timestamps = [], []
    reference = None
    frames = 0
    for chunk in mdtraj.iterload(
        str(directory / "trajectory.xtc"), top=topology, chunk=config.frame_chunk_size
    ):
        frames += chunk.n_frames
        if frames > source.frames or chunk.n_atoms != source.atoms:
            raise DataContractError("MDbind XTC dimensions differ")
        coordinates = np.asarray(chunk.xyz, dtype=np.float64) * ANGSTROM_PER_NM
        if reference is None:
            reference = coordinates[:1].copy()
        values.append(contact_geometry(coordinates, reference, selections, config))
        timestamps.append(np.asarray(chunk.time, dtype=np.float64))
    return _finish_geometry(source, frames, timestamps, values)


def _finish_geometry(
    source: ReplicaSource,
    frames: int,
    timestamps: list[FloatArray],
    values: list[FloatArray],
) -> tuple[FloatArray, FloatArray]:
    if frames != source.frames:
        raise DataContractError("MDbind XTC frame count differs")
    time = np.concatenate(timestamps)
    if not np.allclose(np.diff(time), source.frame_interval_ps, rtol=0, atol=0):
        raise DataContractError(
            "MDbind XTC timestamps differ from verified 200 ps grid"
        )
    return time, np.concatenate(values)


def _canonical_replica(
    source: ReplicaSource,
    config: StructuralConfig,
    time: FloatArray,
    values: FloatArray,
) -> pa.Table:
    dataset = common_dataset(config, DatasetId.MDBIND)
    system = stable_system_id(DatasetId.MDBIND, source.pdb_id)
    record = TrajectoryManifest(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        source_record=source.accession,
        source_checksum=source.files["trajectory.xtc"].checksum,
        license_id=source.license_id,
        provenance_uri=source.files["trajectory.xtc"].url,
        system_id=system,
        pdb_id=source.pdb_id,
        protein_id=None,
        ligand_id=source.ligand_name,
        trajectory_id=stable_trajectory_id(
            system, source.accession.split(".")[0], str(source.replica)
        ),
        replicate_id=str(source.replica),
        split_group_id=system,
        frame_count=source.frames,
        time_unit=TimeUnit.PS,
        sampling_status=SamplingStatus.VERIFIED,
        frame_interval_ps=source.frame_interval_ps,
        duration_ns=float(time[-1] - time[0]) / 1000,
        sampling_note=(
            f"XTC stored origin {float(time[0])!r} ps; "
            "relative axis subtracts this origin; "
            "MDTraj nm coordinates converted to Angstrom. "
            "PDB/AMBER atom order verified."
        ),
        feature_set_version=dataset.feature_set_version,
    )
    return to_arrow(record, dataset, time - time[0], values)


class SubsetSelection(BoundaryModel):
    """Catalog identity and deterministic metadata-only selection rule."""

    selection_rule: Annotated[str, Field(min_length=1)]
    catalog_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    selected: Annotated[tuple[tuple[str, str], ...], Field(min_length=1)]


class MDBindSubset(BoundaryModel):
    """An explicit complex cohort with all ten distinct source replicas."""

    selection: SubsetSelection
    replicas: Annotated[tuple[ReplicaSource, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_cohort(self) -> Self:
        """Do not admit duplicate complexes, missing replicas or unseen sources."""
        _validate_subset(self)
        return self


def _validate_subset(subset: MDBindSubset) -> None:
    selected = dict(subset.selection.selected)
    if len(selected) != len(subset.selection.selected):
        raise ValueError("subset requires distinct PDB identities")
    expected = _cohort_ids(selected)
    actual = {(r.pdb_id, r.accession) for r in subset.replicas}
    if actual != expected or len(actual) != len(subset.replicas):
        raise ValueError("subset must contain each selected complex's ten replicas")


def _cohort_ids(selected: dict[str, str]) -> set[tuple[str, str]]:
    return {
        (pdb, base if replica == 1 else f"{base}.{replica}")
        for pdb, base in selected.items()
        for replica in range(1, 11)
    }
