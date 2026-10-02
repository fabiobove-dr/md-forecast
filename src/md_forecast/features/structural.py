"""Bounded, raw-Cartesian heavy-atom geometry for the audited MISATO layout."""

from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal, Self

import h5py
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator

from md_forecast.core.constants import FeatureUnit
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import validate_array
from md_forecast.data.schemas import BoundaryModel, FeatureDefinition, Identifier
from md_forecast.data.series import FloatArray

COORDINATES = "trajectory_coordinates"
ATOMIC_NUMBERS = "atoms_number"
SEGMENT_STARTS = "molecules_begin_atom_index"
CONTACT_COUNT = "protein_ligand_contact_count"
REFERENCE_FRACTION = "fraction_reference_contacts"
MIN_DISTANCE = "minimum_protein_ligand_heavy_distance"
LIGAND_SASA = "isolated_ligand_sasa"
BASE_FEATURE_IDS = (CONTACT_COUNT, REFERENCE_FRACTION, MIN_DISTANCE)
HYDROGEN_ATOMIC_NUMBER = 1
MAX_ATOMIC_NUMBER = 118
GEOMETRY_VERSION: Literal["misato-geometry-v1"] = "misato-geometry-v1"

type AtomIndices = Annotated[
    tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=1)
]
type IntArray = NDArray[np.int64]
SASA_BACKEND = "1.11.1.post2"
ANGSTROM_PER_NM = 10.0


class SASAConfig(BoundaryModel):
    """Pinned isolated all-atom ligand SASA; receptor is not an occluder."""

    backend: Literal["mdtraj-1.11.1.post2"] = "mdtraj-1.11.1.post2"
    radii: Literal["mdtraj-defaults-1.11.1.post2"] = "mdtraj-defaults-1.11.1.post2"
    probe_radius_angstrom: Annotated[float, Field(gt=0)]
    sphere_points: Annotated[int, Field(strict=True, ge=12)]
    max_atoms: Annotated[int, Field(strict=True, gt=0)] = 512


class AtomPair(BoundaryModel):
    """Explicit zero-based source atom indices, never guessed residue numbers."""

    left: AtomIndices
    right: AtomIndices

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        """Require sorted unique, non-overlapping atom sets."""
        for indices in (self.left, self.right):
            if indices != tuple(sorted(set(indices))):
                raise ValueError("atom indices must be sorted and unique")
        if set(self.left) & set(self.right):
            raise ValueError("region atom sets must not overlap")
        return self


class RegionDistance(BoundaryModel):
    """One named minimum-distance channel with per-system reviewed atom sets."""

    feature_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    description: Identifier
    selections: Annotated[dict[str, AtomPair], Field(min_length=1)]


class StructuralConfig(BoundaryModel):
    """Exact geometry definition and explicit resource ceilings; no imaging repair."""

    algorithm_version: Literal["misato-geometry-v1"] = GEOMETRY_VERSION
    coordinate_unit: Literal["angstrom"] = "angstrom"
    distance_policy: Literal["raw-cartesian-no-pbc"] = "raw-cartesian-no-pbc"
    reference_frame: Literal[0] = 0
    contact_cutoff_angstrom: Annotated[float, Field(gt=0)]
    frame_chunk_size: Annotated[int, Field(strict=True, gt=0)]
    atom_tile_size: Annotated[int, Field(strict=True, gt=0)]
    max_atoms: Annotated[int, Field(strict=True, gt=0)]
    max_frames: Annotated[int, Field(strict=True, gt=0)]
    max_working_bytes: Annotated[int, Field(strict=True, gt=0)]
    regions: tuple[RegionDistance, ...] = ()
    ligand_sasa: SASAConfig | None = None

    @model_validator(mode="after")
    def validate_regions(self) -> Self:
        """Prevent a region from shadowing a core feature or the time column."""
        ids = (
            *BASE_FEATURE_IDS,
            "time",
            LIGAND_SASA,
            *(r.feature_id for r in self.regions),
        )
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate or reserved structural feature ID")
        return self

    @property
    def feature_set_version(self) -> str:
        """Full configuration identity prevents silent definition changes."""
        return f"{self.algorithm_version}-{metadata_hash(self).removeprefix('sha256:')}"

    def feature_definitions(self) -> tuple[FeatureDefinition, ...]:
        """Declare selections, inclusive cutoff, reference and native units."""
        selection = (
            "Heavy atoms (atoms_number > 1); receptor indices before the last "
            "molecules_begin_atom_index, ligand indices from that boundary onward. "
            "Raw Cartesian Angstrom distances, no periodic imaging correction."
        )
        cutoff = f"distance <= {self.contact_cutoff_angstrom!r} Angstrom"
        core = (
            (
                CONTACT_COUNT,
                FeatureUnit.COUNT,
                "Heavy-atom pair contact count",
                f"Count receptor-ligand atom pairs with {cutoff}. {selection}",
            ),
            (
                REFERENCE_FRACTION,
                FeatureUnit.DIMENSIONLESS,
                "Initial contact retention",
                f"Fraction of frame-0 atom-pair contacts satisfying {cutoff} in each "
                f"frame; zero reference contacts excludes the system. {selection}",
            ),
            (
                MIN_DISTANCE,
                FeatureUnit.ANGSTROM,
                "Minimum heavy-atom distance",
                f"Minimum receptor-ligand heavy-atom pair distance. {selection}",
            ),
        )
        features = tuple(
            FeatureDefinition(
                feature_id=name,
                unit=unit,
                description=description,
                definition=definition,
            )
            for name, unit, description, definition in core
        )
        features += tuple(
            FeatureDefinition(
                feature_id=region.feature_id,
                unit=FeatureUnit.ANGSTROM,
                description=region.description,
                definition=(
                    "Minimum raw Cartesian distance between explicitly selected "
                    "heavy-atom sets; zero-based per-system indices are persisted "
                    f"in config {metadata_hash(self)}; no automatic residue mapping."
                ),
            )
            for region in self.regions
        )
        if self.ligand_sasa is not None:
            features += (_sasa_definition(self.ligand_sasa),)
        return features


def _sasa_definition(config: SASAConfig) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=LIGAND_SASA,
        unit=FeatureUnit.ANGSTROM_SQUARED,
        description="Isolated all-atom ligand solvent accessible surface area",
        definition=(
            f"MDTraj {SASA_BACKEND} Shrake-Rupley; all atoms in final segment, "
            f"including hydrogen; receptor excluded as occluder; default "
            f"backend element radii, probe {config.probe_radius_angstrom!r} "
            f"Angstrom, {config.sphere_points} sphere points; raw coordinates "
            "converted Angstrom to nm (float32), area nm^2 to Angstrom^2. "
            "One-frame backend calls (batch-independent); no PBC correction; "
            "not bound ligand SASA or buried SASA."
        ),
    )


def load_structural_config(path: Path) -> StructuralConfig:
    """Validate a versioned YAML geometry configuration before reading data."""
    try:
        return StructuralConfig.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise DataContractError(
            f"cannot load structural config {path}: {error}"
        ) from error


def _dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    if not isinstance(group.get(key, getlink=True), h5py.HardLink):
        raise DataContractError(f"missing or linked structural field {key}")
    value = group[key]
    if not isinstance(value, h5py.Dataset):
        raise DataContractError(f"structural field {key} is not a dataset")
    return value


def _coordinate_source(
    group: h5py.Group, frames: int, config: StructuralConfig
) -> h5py.Dataset:
    coordinates = _dataset(group, COORDINATES)
    _validate_coordinate_shape(coordinates, frames)
    _validate_budget(coordinates.shape[1], frames, config)
    return coordinates


def _validate_coordinate_shape(coordinates: h5py.Dataset, frames: int) -> None:
    if (
        coordinates.ndim != 3
        or coordinates.shape[0] != frames
        or coordinates.shape[2] != 3
        or coordinates.dtype != np.dtype("float64")
    ):
        raise DataContractError(
            "structural coordinates require float64 (frames,atoms,3)"
        )


def _validate_budget(atoms: int, frames: int, config: StructuralConfig) -> None:
    if not (0 < atoms <= config.max_atoms and 0 < frames <= config.max_frames):
        raise DataContractError("structural source exceeds atom/frame budget")
    chunk = min(frames, config.frame_chunk_size)
    tile = min(atoms, config.atom_tile_size)
    # Conservative array estimate: coordinates/copies, pair workspace, output.
    estimated = 8 * (
        6 * (chunk + 1) * atoms
        + 8 * chunk * tile**2
        + frames * len(config.feature_definitions())
    )
    if config.ligand_sasa is not None:
        # Bound a conservative neighbor workspace and the sphere point mesh.
        estimated += 8 * (
            min(atoms, config.ligand_sasa.max_atoms) ** 2
            + 3 * config.ligand_sasa.sphere_points
        )
    if estimated > config.max_working_bytes:
        raise DataContractError(f"structural arrays exceed byte budget ({estimated})")


def _heavy_sets(group: h5py.Group, atoms: int) -> tuple[IntArray, IntArray, IntArray]:
    numbers, starts = _dataset(group, ATOMIC_NUMBERS), _dataset(group, SEGMENT_STARTS)
    atomic_numbers = _atomic_numbers(numbers, atoms)
    boundaries = _boundaries(starts, atoms)
    heavy = np.flatnonzero(atomic_numbers > HYDROGEN_ATOMIC_NUMBER)
    boundary = boundaries[-1]
    receptor, ligand = heavy[heavy < boundary], heavy[heavy >= boundary]
    if not len(receptor) or not len(ligand):
        raise DataContractError("empty receptor or ligand heavy-atom selection")
    return atomic_numbers, receptor, ligand


def _atomic_numbers(numbers: h5py.Dataset, atoms: int) -> IntArray:
    if numbers.shape != (atoms,) or numbers.dtype != np.dtype("int64"):
        raise DataContractError("invalid atomic number shape/dtype")
    atomic_numbers = numbers[:]
    if np.any((atomic_numbers < 1) | (atomic_numbers > MAX_ATOMIC_NUMBER)):
        raise DataContractError("invalid atomic numbers")
    return np.asarray(atomic_numbers, dtype=np.int64)


def _boundaries(starts: h5py.Dataset, atoms: int) -> IntArray:
    if (starts.ndim, starts.dtype) != (1, np.dtype("int64")) or not 2 <= len(
        starts
    ) <= atoms:
        raise DataContractError("invalid molecule segment shape/dtype")
    boundaries = starts[:]
    if boundaries[0] != 0 or not np.all(
        (np.diff(boundaries) > 0) & (boundaries[1:] < atoms)
    ):
        raise DataContractError("invalid molecule segment boundaries")
    return np.asarray(boundaries, dtype=np.int64)


def _region_sets(
    config: StructuralConfig, system: str, atomic_numbers: IntArray
) -> tuple[tuple[IntArray, IntArray], ...]:
    result = []
    for region in config.regions:
        if system not in region.selections:
            raise DataContractError(f"missing explicit region selection for {system}")
        pair = region.selections[system]
        if max((*pair.left, *pair.right)) >= len(atomic_numbers):
            raise DataContractError("region atom index exceeds source bounds")
        if np.any(
            atomic_numbers[list((*pair.left, *pair.right))] <= HYDROGEN_ATOMIC_NUMBER
        ):
            raise DataContractError("region selections must contain heavy atoms only")
        result.append((np.asarray(pair.left), np.asarray(pair.right)))
    return tuple(result)


def _distances(left: FloatArray, right: FloatArray) -> FloatArray:
    delta = left[:, :, None, :] - right[:, None, :, :]
    return np.asarray(np.linalg.norm(delta, axis=-1), dtype=np.float64)


def _tiles(
    left: IntArray, right: IntArray, size: int
) -> Iterator[tuple[IntArray, IntArray]]:
    for start in range(0, len(left), size):
        for other in range(0, len(right), size):
            yield left[start : start + size], right[other : other + size]


def _contact_geometry(
    values: FloatArray,
    reference: FloatArray,
    left: IntArray,
    right: IntArray,
    config: StructuralConfig,
) -> FloatArray:
    count = np.zeros(len(values))
    retained = np.zeros(len(values))
    minimum = np.full(len(values), np.inf)
    reference_count = 0
    for a, b in _tiles(left, right, config.atom_tile_size):
        distances = _distances(values[:, a], values[:, b])
        contacts = distances <= config.contact_cutoff_angstrom
        initial = (
            _distances(reference[:, a], reference[:, b])[0]
            <= config.contact_cutoff_angstrom
        )
        count += contacts.sum(axis=(1, 2))
        retained += (contacts & initial).sum(axis=(1, 2))
        reference_count += int(initial.sum())
        minimum = np.minimum(minimum, distances.min(axis=(1, 2)))
    if reference_count == 0:
        raise DataContractError("no frame-0 reference contacts; fraction is undefined")
    return np.column_stack((count, retained / reference_count, minimum))


def _region_geometry(
    values: FloatArray, left: IntArray, right: IntArray, tile_size: int
) -> FloatArray:
    minimum = np.full(len(values), np.inf)
    for a, b in _tiles(left, right, tile_size):
        minimum = np.minimum(
            minimum, _distances(values[:, a], values[:, b]).min(axis=(1, 2))
        )
    return minimum


def _finite_coordinates(coordinates: h5py.Dataset, start: int, stop: int) -> FloatArray:
    values = np.asarray(coordinates[start:stop], dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataContractError("nonfinite structural coordinates")
    return values


def validate_sasa_backend() -> None:
    """Fail before extraction if the explicitly requested optional backend differs."""
    try:
        installed = version("mdtraj")
    except PackageNotFoundError as error:
        raise DataContractError("ligand SASA requires the structural extra") from error
    if installed != SASA_BACKEND:
        raise DataContractError(
            f"ligand SASA requires MDTraj {SASA_BACKEND}, got {installed}"
        )


def _ligand_sasa(
    values: FloatArray, numbers: IntArray, config: SASAConfig
) -> FloatArray:
    validate_sasa_backend()
    import mdtraj

    topology = mdtraj.Topology()
    residue = topology.add_residue("MOL", topology.add_chain())
    for index, number in enumerate(numbers):
        try:
            element = mdtraj.element.Element.getByAtomicNumber(int(number))
        except KeyError as error:
            raise DataContractError(
                "SASA backend does not support source element"
            ) from error
        topology.add_atom(f"A{index}", element, residue)
    # MDTraj's kernel consumes float32 nm. Translation reduces precision loss.
    coordinates = (values - values[:, :1, :]) / ANGSTROM_PER_NM
    trajectory = mdtraj.Trajectory(coordinates.astype(np.float32), topology)
    try:
        # ponytail: isolate frames; batch only after analytic invariance holds.
        areas = np.concatenate(
            [
                mdtraj.shrake_rupley(
                    frame,
                    probe_radius=config.probe_radius_angstrom / ANGSTROM_PER_NM,
                    n_sphere_points=config.sphere_points,
                    mode="atom",
                )
                for frame in trajectory
            ]
        )
    except (KeyError, ValueError) as error:
        raise DataContractError(
            "SASA backend rejected source elements/geometry"
        ) from error
    result = np.asarray(areas.sum(axis=1), dtype=np.float64) * ANGSTROM_PER_NM**2
    validate_array(result, (len(values),))
    return result


def structural_values(
    group: h5py.Group, system: str, frames: int, config: StructuralConfig
) -> FloatArray:
    """Read frame chunks and pair tiles; frame 0 is the only reference.

    Working arrays scale with chunk*atoms and chunk*tile², never all frames*atoms².
    The small output series is bounded by max_frames. Numerical faults exclude
    the source system, rather than creating invented contacts or imaged coordinates.
    """
    config = StructuralConfig.model_validate_json(config.model_dump_json())
    coordinates = _coordinate_source(group, frames, config)
    atomic_numbers, receptor, ligand = _heavy_sets(group, coordinates.shape[1])
    regions = _region_sets(config, system, atomic_numbers)
    boundary = int(_dataset(group, SEGMENT_STARTS)[-1])
    _validate_sasa_atoms(config.ligand_sasa, coordinates.shape[1] - boundary)
    reference = _finite_coordinates(coordinates, 0, 1)
    output = np.empty((frames, len(config.feature_definitions())), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for start in range(0, frames, config.frame_chunk_size):
                stop = min(start + config.frame_chunk_size, frames)
                values = _finite_coordinates(coordinates, start, stop)
                output[start:stop, : len(BASE_FEATURE_IDS)] = _contact_geometry(
                    values, reference, receptor, ligand, config
                )
                for index, (left, right) in enumerate(
                    regions, start=len(BASE_FEATURE_IDS)
                ):
                    output[start:stop, index] = _region_geometry(
                        values, left, right, config.atom_tile_size
                    )
                if config.ligand_sasa is not None:
                    output[start:stop, -1] = _ligand_sasa(
                        values[:, boundary:],
                        atomic_numbers[boundary:],
                        config.ligand_sasa,
                    )
    except FloatingPointError as error:
        raise DataContractError("structural distance calculation overflowed") from error
    return output


def _validate_sasa_atoms(config: SASAConfig | None, atoms: int) -> None:
    if config is not None and atoms > config.max_atoms:
        raise DataContractError("ligand SASA atom budget exceeded")
