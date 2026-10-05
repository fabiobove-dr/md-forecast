"""Compare raw/periodic geometry on verified dry MDbind source replicas."""

import argparse
import json
from pathlib import Path
from typing import Annotated

import mdtraj
import numpy as np
from pydantic import Field

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.public.mdbind import (
    MDBindSubset,
    ReplicaSource,
    verified_replica_topology,
)
from md_forecast.data.schemas import BoundaryModel
from md_forecast.features.structural import (
    ANGSTROM_PER_NM,
    StructuralConfig,
    load_structural_config,
    validate_sasa_backend,
)


class AuditBudget(BoundaryModel):
    """Bound the all-pair diagnostic; each admitted source has exactly 50 frames."""

    max_pairs: Annotated[int, Field(strict=True, gt=0)] = 400_000


def audit_replica(
    raw: Path, source: ReplicaSource, budget: AuditBudget, geometry: StructuralConfig
) -> dict[str, object]:
    """Verify source identity, compare policies, and retain measured time/box data."""
    directory = raw / source.accession
    topology = verified_replica_topology(directory, source)
    heavy = [atom for atom in topology.atoms if atom.element.atomic_number > 1]
    ligand = [a.index for a in heavy if a.residue.index == source.ligand_residue_index]
    receptor = [
        a.index for a in heavy if a.residue.index != source.ligand_residue_index
    ]
    if len(ligand) * len(receptor) > budget.max_pairs:
        raise DataContractError(f"{source.accession}: pair diagnostic exceeds budget")
    pairs = np.column_stack(
        (np.repeat(receptor, len(ligand)), np.tile(ligand, len(receptor)))
    ).astype(np.int32)
    trajectory = mdtraj.load_xtc(str(directory / "trajectory.xtc"), top=topology)
    if trajectory.n_frames != source.frames or trajectory.n_atoms != source.atoms:
        raise DataContractError(f"{source.accession}: XTC dimensions differ")
    if trajectory.unitcell_vectors is None:
        raise DataContractError(f"{source.accession}: periodic comparison needs boxes")
    if (
        not np.isfinite(trajectory.unitcell_vectors).all()
        or not (trajectory.unitcell_lengths > 0).all()
    ):
        raise DataContractError(f"{source.accession}: invalid periodic box")
    if not np.isfinite(trajectory.xyz).all():
        raise DataContractError(f"{source.accession}: nonfinite coordinates")
    raw_contacts, periodic_contacts = [], []
    minimum_differences = []
    cutoff_nm = geometry.contact_cutoff_angstrom / ANGSTROM_PER_NM
    for frame in trajectory:
        raw_distances = mdtraj.compute_distances(frame, pairs, periodic=False)[0]
        periodic_distances = mdtraj.compute_distances(frame, pairs, periodic=True)[0]
        # Same float32 kernel for both policies; canonical extraction stays float64.
        raw_contacts.append(raw_distances <= cutoff_nm)
        periodic_contacts.append(periodic_distances <= cutoff_nm)
        minimum_differences.append(
            float(abs(raw_distances.min() - periodic_distances.min()) * ANGSTROM_PER_NM)
        )
    raw_contacts = np.asarray(raw_contacts)
    periodic_contacts = np.asarray(periodic_contacts)
    raw_reference = raw_contacts[0]
    periodic_reference = periodic_contacts[0]
    if not raw_reference.any() or not periodic_reference.any():
        raise DataContractError(f"{source.accession}: undefined reference fraction")
    raw_fraction = (raw_contacts & raw_reference).sum(axis=1) / raw_reference.sum()
    periodic_fraction = (periodic_contacts & periodic_reference).sum(axis=1)
    periodic_fraction = periodic_fraction / periodic_reference.sum()
    return {
        "accession": source.accession,
        "pdb_id": source.pdb_id,
        "frames": trajectory.n_frames,
        "origin_ps": float(trajectory.time[0]),
        "last_ps": float(trajectory.time[-1]),
        "spacing_ps": np.unique(np.diff(trajectory.time)).tolist(),
        "minimum_box_length_nm": float(trajectory.unitcell_lengths.min()),
        "heavy_receptor": len(receptor),
        "heavy_ligand": len(ligand),
        "count_different_frames": int(
            np.count_nonzero(raw_contacts.sum(axis=1) != periodic_contacts.sum(axis=1))
        ),
        "fraction_different_frames": int(
            np.count_nonzero(raw_fraction != periodic_fraction)
        ),
        "max_min_distance_difference_angstrom": max(minimum_differences),
    }


def main() -> None:
    """Read the reviewed cohort; save derived diagnostics without source bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("configs/datasets/mdbind-common.json")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/external/mdbind-common-raw")
    )
    parser.add_argument("--max-pairs", type=int, default=400_000)
    parser.add_argument(
        "--geometry", type=Path, default=Path("configs/features/common-geometry.yaml")
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    budget = AuditBudget(max_pairs=args.max_pairs)
    geometry = load_structural_config(args.geometry)
    validate_sasa_backend()
    subset = MDBindSubset.model_validate_json(args.source.read_text())
    rows = [
        audit_replica(args.raw, source, budget, geometry) for source in subset.replicas
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
