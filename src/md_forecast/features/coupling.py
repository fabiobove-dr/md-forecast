"""Frozen static pocket/distal C-alpha regions and unweighted radius series."""

from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import FeatureUnit
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.public.mdbind import (
    MDBindSubset,
    ReplicaSource,
    verified_replica_topology,
)
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, FeatureDefinition
from md_forecast.data.series import FloatArray
from md_forecast.features.structural import ANGSTROM_PER_NM


class RegionRules(BoundaryModel):
    """Static replica-one PDB thresholds, fixed before reading trajectory values."""

    pocket_cutoff_angstrom: Annotated[float, Field(gt=0)] = 6.0
    distal_cutoff_angstrom: Annotated[float, Field(gt=0)] = 12.0
    min_region_atoms: Annotated[int, Field(strict=True, ge=3)] = 3
    max_atoms: Annotated[int, Field(strict=True, gt=0)] = 25000
    max_frames: Annotated[int, Field(strict=True, gt=0)] = 50
    frame_chunk_size: Annotated[int, Field(strict=True, gt=0)] = 10

    @model_validator(mode="after")
    def distinct_shells(self) -> Self:
        """Require a gap between the pocket and distal residue selections."""
        if self.pocket_cutoff_angstrom >= self.distal_cutoff_angstrom:
            raise ValueError("pocket and distal thresholds must be ordered")
        return self


class RegionSelection(BoundaryModel):
    """Portable explicit atom indices and reference identities for one complex."""

    reference_accession: str
    pdb_checksum: ArtifactHash
    topology_checksum: ArtifactHash
    pocket: Annotated[tuple[int, ...], Field(min_length=3)]
    distal: Annotated[tuple[int, ...], Field(min_length=3)]

    @model_validator(mode="after")
    def disjoint_atoms(self) -> Self:
        """Prevent duplicates, negative indices and overlapping regions."""
        for atoms in (self.pocket, self.distal):
            _validate_indices(atoms)
        if set(self.pocket) & set(self.distal):
            raise ValueError("pocket and distal regions overlap")
        return self


def _validate_indices(atoms: tuple[int, ...]) -> None:
    if min(atoms) < 0 or len(set(atoms)) != len(atoms):
        raise ValueError("region requires distinct nonnegative atom indices")


class RegionManifest(BoundaryModel):
    """Freeze rules/cohort and each replica-one static selection before inference."""

    source_hash: ArtifactHash
    rules: RegionRules
    selections: Annotated[dict[str, RegionSelection], Field(min_length=1)]

    def feature_definitions(self) -> tuple[FeatureDefinition, ...]:
        """Declare rotation/translation invariant, unweighted C-alpha radii in Å."""
        return tuple(
            FeatureDefinition(
                feature_id=f"{name}_rg",
                unit=FeatureUnit.ANGSTROM,
                description=f"Unweighted {name} C-alpha radius of gyration",
                definition=(
                    "sqrt(mean_i ||x_i - mean_j x_j||^2), raw Cartesian Angstrom "
                    "coordinates without PBC repair; protein C-alpha indices frozen "
                    "from replica-one source PDB using residue minimum heavy-atom "
                    "distance to the reviewed ligand. Region manifest "
                    f"{metadata_hash(self)}."
                ),
            )
            for name in ("distal", "pocket")
        )


def freeze_regions(
    subset: MDBindSubset, raw: Path, rules: RegionRules
) -> RegionManifest:
    """Read only static PDB coordinates; never select atoms from test dynamics."""
    import mdtraj

    selections = {}
    for source in subset.replicas:
        if source.replica == 1:
            root = raw / source.accession
            topology = verified_replica_topology(root, source)
            xyz = (
                np.asarray(
                    mdtraj.load_pdb(str(root / "structure.pdb")).xyz[0],
                    dtype=np.float64,
                )
                * ANGSTROM_PER_NM
            )
            pocket, distal = _select_regions(topology, xyz, source, rules)
            selections[source.pdb_id] = RegionSelection(
                reference_accession=source.accession,
                pdb_checksum=source.files["structure.pdb"].checksum,
                topology_checksum=source.files["topology.prmtop"].checksum,
                pocket=pocket,
                distal=distal,
            )
    return RegionManifest(
        source_hash=metadata_hash(subset), rules=rules, selections=selections
    )


def _select_regions(
    topology: Any, xyz: FloatArray, source: ReplicaSource, rules: RegionRules
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ligand = _ligand_indices(topology, source.ligand_residue_index)
    distances = _residue_distances(topology, xyz, ligand, source.ligand_residue_index)
    pocket = tuple(i for i, d in distances.items() if d <= rules.pocket_cutoff_angstrom)
    distal = tuple(i for i, d in distances.items() if d >= rules.distal_cutoff_angstrom)
    _validate_region_count(pocket, distal, rules)
    return pocket, distal


def _validate_region_count(
    pocket: tuple[int, ...], distal: tuple[int, ...], rules: RegionRules
) -> None:
    if min(len(pocket), len(distal)) < rules.min_region_atoms:
        raise DataContractError("static region selection has too few C-alpha atoms")


def _ligand_indices(topology: Any, residue_index: int) -> list[int]:
    return [
        a.index
        for a in topology.residue(residue_index).atoms
        if a.element.atomic_number > 1
    ]


def _residue_distances(
    topology: Any, xyz: FloatArray, ligand: list[int], limit: int
) -> dict[int, float]:
    distances = {}
    for residue in topology.residues:
        if residue.index < limit and residue.is_protein:
            index, distance = _residue_distance(residue, xyz, ligand)
            distances[index] = distance
    return distances


def _residue_distance(
    residue: Any, xyz: FloatArray, ligand: list[int]
) -> tuple[int, float]:
    ca = _protein_ca(residue)
    heavy = [a.index for a in residue.atoms if a.element.atomic_number > 1]
    return ca, float(np.linalg.norm(xyz[heavy, None, :] - xyz[ligand, :], axis=2).min())


def _protein_ca(residue: Any) -> int:
    ca = [a.index for a in residue.atoms if a.name == "CA"]
    if len(ca) != 1:
        raise DataContractError("protein residue must have exactly one C-alpha")
    return int(ca[0])


def region_radii(coordinates: FloatArray, selection: RegionSelection) -> FloatArray:
    """Compute B=distal then A=pocket without future fitting or mass assumptions."""
    values = np.column_stack(
        [
            _radius(coordinates[:, atoms, :])
            for atoms in (selection.distal, selection.pocket)
        ]
    )
    if not np.isfinite(values).all():
        raise DataContractError("region radius contains nonfinite coordinates")
    return values


def _radius(xyz: FloatArray) -> FloatArray:
    centered = xyz - xyz.mean(axis=1, keepdims=True)
    return np.asarray(
        np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1)), dtype=np.float64
    )


def extract_region_series(
    directory: Path, source: ReplicaSource, regions: RegionManifest
) -> tuple[FloatArray, FloatArray]:
    """Verify atom identity, bounded XTC dimensions and the exact retained ps grid."""
    try:
        return _extract_region_series(directory, source, regions)
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise DataContractError(
            f"cannot extract region series {source.accession}: {error}"
        ) from error


def _extract_region_series(
    directory: Path, source: ReplicaSource, regions: RegionManifest
) -> tuple[FloatArray, FloatArray]:
    import mdtraj

    rules = regions.rules
    selection = _validate_region_source(source, regions)
    topology = verified_replica_topology(directory, source)
    _validate_region_atoms(topology, selection)
    times, values = [], []
    frames = 0
    for chunk in mdtraj.iterload(
        str(directory / "trajectory.xtc"), top=topology, chunk=rules.frame_chunk_size
    ):
        frames += chunk.n_frames
        if frames > source.frames or chunk.n_atoms != source.atoms:
            raise DataContractError("region XTC dimensions differ from source")
        values.append(
            region_radii(
                np.asarray(chunk.xyz, dtype=np.float64) * ANGSTROM_PER_NM, selection
            )
        )
        times.append(np.asarray(chunk.time, dtype=np.float64))
    return _finish_series(source, frames, times, values)


def _validate_region_source(
    source: ReplicaSource, regions: RegionManifest
) -> RegionSelection:
    rules = regions.rules
    if source.atoms > rules.max_atoms or source.frames > rules.max_frames:
        raise DataContractError("region source exceeds atom/frame budget")
    selection = regions.selections[source.pdb_id]
    if source.files["topology.prmtop"].checksum != selection.topology_checksum:
        raise DataContractError("replica topology differs from frozen region reference")
    return selection


def _validate_region_atoms(topology: Any, selection: RegionSelection) -> None:
    for index in (*selection.pocket, *selection.distal):
        if index >= topology.n_atoms:
            raise DataContractError("region index exceeds atom count")
        atom = topology.atom(index)
        if atom.name != "CA" or not atom.residue.is_protein:
            raise DataContractError("frozen region atom is not protein C-alpha")


def _finish_series(
    source: ReplicaSource,
    frames: int,
    times: list[FloatArray],
    values: list[FloatArray],
) -> tuple[FloatArray, FloatArray]:
    if frames != source.frames:
        raise DataContractError("region XTC frame count differs")
    time = np.concatenate(times)
    if not np.array_equal(
        np.diff(time), np.full(source.frames - 1, source.frame_interval_ps)
    ):
        raise DataContractError("region XTC cadence differs from verified ps grid")
    return time - time[0], np.concatenate(values)
