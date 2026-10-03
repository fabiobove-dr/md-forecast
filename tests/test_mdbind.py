"""Synthetic raw replicas: units, atom identities, integrity and split leakage."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from test_structural import config

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.public.mdbind import (
    MDBindSubset,
    ReplicaSource,
    extract_replica,
)
from md_forecast.data.series import series_metadata
from md_forecast.features.structural import AtomPair, contact_geometry


def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: Any
) -> tuple[ReplicaSource, SimpleNamespace]:
    receptor = SimpleNamespace(index=0, name="ALA", resSeq=1)
    ligand = SimpleNamespace(index=1, name="LIG", resSeq=2)
    ligand.chain = SimpleNamespace(chain_id="H")
    carbon = SimpleNamespace(atomic_number=6)
    atoms = [
        SimpleNamespace(index=i, name=f"C{i}", residue=r, element=carbon)
        for i, r in enumerate((receptor, ligand))
    ]
    ligand.atoms = [atoms[1]]
    topology = SimpleNamespace(
        n_atoms=2,
        n_residues=2,
        atoms=atoms,
        residues=[receptor, ligand],
        residue=lambda index: ligand,
    )
    chunk = SimpleNamespace(
        xyz=np.zeros((50, 2, 3), dtype=np.float32),
        time=np.arange(50, dtype=np.float64) * 200 + 210,
        n_frames=50,
        n_atoms=2,
    )
    chunk.xyz[:, 1, 0] = 0.3  # nm -> 3 Angstrom, not a 0.3-A distance.
    for key, value in changes.items():
        setattr(chunk, key, value)

    def chunks(*args: Any, **kwargs: Any) -> list[SimpleNamespace]:
        size = kwargs["chunk"]
        return [
            SimpleNamespace(
                xyz=chunk.xyz[start : start + size],
                time=chunk.time[start : start + size],
                n_atoms=chunk.n_atoms,
                n_frames=min(size, chunk.n_frames - start),
            )
            for start in range(0, chunk.n_frames, size)
        ]

    monkeypatch.setitem(
        sys.modules,
        "mdtraj",
        SimpleNamespace(load_topology=lambda path: topology, iterload=chunks),
    )
    monkeypatch.setattr(
        "md_forecast.data.public.mdbind.validate_sasa_backend", lambda: None
    )
    record = {
        "metadata": {
            "PDBIDS": ["1AAA"],
            "SYSTATS": 2,
            "SOLVATS": 0,
            "LINKCENSE": "https://creativecommons.org/licenses/by/4.0/",
            "INTERACTIONS": [{"type": "protein-ligand", "selection_2": "resid 2"}],
        },
        "mdNumber": 1,
        "mdIndex": 0,
        "mdcount": 10,
    }
    files = {}
    for name in ("structure.pdb", "topology.prmtop", "trajectory.xtc", "record.json"):
        payload = json.dumps(record).encode() if name == "record.json" else b"synthetic"
        (tmp_path / name).write_bytes(payload)
        files[name] = {
            "mode": "metadata",
            "url": f"https://example.org/{name}",
            "size_bytes": len(payload),
            "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    return ReplicaSource(
        accession="MD-A000AA",
        pdb_id="1AAA",
        replica=1,
        atoms=2,
        ligand_residue_index=1,
        ligand_name="LIG",
        interaction_selection="resid 2",
        files=files,
    ), chunk


def test_actual_origin_units_and_common_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = fixture(tmp_path, monkeypatch)
    table = extract_replica(tmp_path, source, config())
    record = series_metadata(table).trajectories[0]
    assert record.frame_interval_ps == 200
    assert record.duration_ns == 9.8
    assert "210.0 ps" in str(record.sampling_note)
    np.testing.assert_array_equal(table["time"].to_numpy(), np.arange(50) * 200)
    np.testing.assert_allclose(
        table["minimum_protein_ligand_heavy_distance"].to_numpy(), 3
    )
    assert table["protein_ligand_contact_count"].to_pylist() == [1.0] * 50
    assert table["fraction_reference_contacts"].to_pylist() == [1.0] * 50


@pytest.mark.parametrize(
    "changes",
    [
        {"time": np.arange(50) * 0.2},
        {"time": np.full(50, np.nan)},
        {"n_frames": 49},
        {"n_frames": 51},
        {"n_atoms": 3},
        {"xyz": np.full((50, 2, 3), np.nan)},
    ],
)
def test_bad_raw_trajectory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any]
) -> None:
    source, _ = fixture(tmp_path, monkeypatch, **changes)
    with pytest.raises(DataContractError):
        extract_replica(tmp_path, source, config())


@pytest.mark.parametrize(
    "mutation", ["corrupt", "missing", "selection", "budget", "name"]
)
def test_integrity_selection_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source, _ = fixture(tmp_path, monkeypatch)
    settings = config()
    if mutation == "corrupt":
        (tmp_path / "trajectory.xtc").write_bytes(b"wrongsize")
    elif mutation == "missing":
        (tmp_path / "structure.pdb").unlink()
    elif mutation == "selection":
        source = source.model_copy(update={"interaction_selection": "resid 9"})
    elif mutation == "name":
        source = source.model_copy(update={"ligand_name": "OTHER"})
    else:
        settings = config(max_atoms=1)
    with pytest.raises(DataContractError):
        extract_replica(tmp_path, source, settings)


def test_geometry_public_boundary() -> None:
    pair = AtomPair(left=(0,), right=(1,))
    values = np.zeros((2, 2, 3), dtype=np.float64)
    values[:, 1, 0] = [3, 5]
    actual = contact_geometry(values, values[:1], pair, config())
    np.testing.assert_array_equal(actual, [[1, 1, 3], [0, 0, 5]])
    for bad in (values[:, :, 0], values.astype(np.float32)):
        with pytest.raises(DataContractError):
            contact_geometry(bad, values[:1], pair, config())  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="bounds"):
        contact_geometry(values, values[:1], AtomPair(left=(0,), right=(3,)), config())


def test_subset_requires_all_named_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = fixture(tmp_path, monkeypatch)
    replicas = tuple(
        source.model_copy(
            update={
                "accession": "MD-A000AA" if r == 1 else f"MD-A000AA.{r}",
                "replica": r,
            }
        )
        for r in range(1, 11)
    )
    selection = {
        "selection_rule": "synthetic cohort",
        "catalog_sha256": "a" * 64,
        "selected": (("1AAA", "MD-A000AA"),),
    }
    subset = MDBindSubset(selection=selection, replicas=replicas)
    assert len(subset.replicas) == 10
    with pytest.raises(ValidationError, match="ten replicas"):
        MDBindSubset(selection=selection, replicas=replicas[:-1])
    with pytest.raises(ValidationError, match="suffix"):
        ReplicaSource.model_validate(source.model_dump() | {"replica": 2})


@pytest.mark.parametrize("fault", ["atom-order", "atom-count", "elements"])
def test_topology_faults_block_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, _ = fixture(tmp_path, monkeypatch)
    backend = sys.modules["mdtraj"]
    topology = backend.load_topology("fixture")
    amber = SimpleNamespace(n_atoms=2, atoms=list(topology.atoms))
    if fault == "atom-order":
        amber.atoms = list(reversed(amber.atoms))
    elif fault == "atom-count":
        amber.n_atoms = 3
    else:
        for atom in topology.atoms:
            atom.element = None
    monkeypatch.setattr(
        backend,
        "load_topology",
        lambda path: amber if path.endswith("prmtop") else topology,
    )
    with pytest.raises(DataContractError, match="atom|element"):
        extract_replica(tmp_path, source, config())


@pytest.mark.parametrize("fault", ["license", "solvent", "identity", "selection"])
def test_metadata_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, _ = fixture(tmp_path, monkeypatch)
    record = json.loads((tmp_path / "record.json").read_text())
    if fault == "license":
        record["metadata"]["LINKCENSE"] = "https://example.org/unknown"
    elif fault == "solvent":
        record["metadata"]["SOLVATS"] = 10
    elif fault == "identity":
        record["mdNumber"] = 2
    else:
        record["metadata"]["INTERACTIONS"] = []
    payload = json.dumps(record).encode()
    (tmp_path / "record.json").write_bytes(payload)
    artifact = source.files["record.json"].model_copy(
        update={
            "size_bytes": len(payload),
            "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
        }
    )
    source = source.model_copy(
        update={"files": source.files | {"record.json": artifact}}
    )
    with pytest.raises(DataContractError):
        extract_replica(tmp_path, source, config())
