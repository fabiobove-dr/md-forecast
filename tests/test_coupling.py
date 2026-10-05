"""Static selections, invariant geometry and context-only conventional controls."""

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from test_baselines import pairs
from test_mdbind import fixture

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata
from md_forecast.data.forecast import ForecastBatch
from md_forecast.evaluation.coupling import (
    CouplingConfig,
    _correlation,
    _lagged_fit_gain,
    lag_diagnostics,
)
from md_forecast.features import coupling
from md_forecast.features.coupling import (
    RegionManifest,
    RegionRules,
    RegionSelection,
    extract_region_series,
    region_radii,
)


def selection(**changes: Any) -> RegionSelection:
    return RegionSelection.model_validate(
        dict(
            reference_accession="MD-A000AA",
            pdb_checksum="sha256:" + "a" * 64,
            topology_checksum="sha256:" + "b" * 64,
            pocket=(0, 1, 2),
            distal=(3, 4, 5),
        )
        | changes
    )


def config(**changes: Any) -> CouplingConfig:
    values = tomllib.loads(
        Path("configs/benchmarks/predictive-coupling.toml").read_text()
    )
    return CouplingConfig.model_validate(values | changes)


def test_portable_frozen_config() -> None:
    regions = read_metadata(
        Path("configs/features/coupling-selection.json"), RegionManifest
    )
    assert config().regions_hash == metadata_hash(regions)
    assert len(regions.selections) == 6
    assert all(set(s.pocket).isdisjoint(s.distal) for s in regions.selections.values())
    assert [f.feature_id for f in regions.feature_definitions()] == [
        "distal_rg",
        "pocket_rg",
    ]
    with pytest.raises(ValidationError, match="ordered"):
        RegionRules(pocket_cutoff_angstrom=12)
    for change in (
        {"pocket": (0, 0, 1)},
        {"pocket": (-1, 0, 1)},
        {"distal": (1, 3, 4)},
    ):
        with pytest.raises(ValidationError):
            selection(**change)
    with pytest.raises(ValidationError, match="distal B"):
        config(
            target="pocket_rg",
            variants={"b": ["pocket_rg"], "a": ["pocket_rg", "distal_rg"]},
        )
    with pytest.raises(ValidationError, match="replicas"):
        config(split={"mode": "grouped", "seed": 42, "ratios": (0.8, 0.0, 0.2)})


def test_radius_native_unit_and_rigid_invariance() -> None:
    xyz = np.zeros((2, 6, 3))
    xyz[:, :, 0] = [0, 1, 2, 10, 12, 14]
    expected = np.array([[np.sqrt(8 / 3), np.sqrt(2 / 3)]] * 2)
    assert np.allclose(region_radii(xyz, selection()), expected)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(region_radii(xyz @ rotation + 100, selection()), expected)
    xyz[0, 0, 0] = np.nan
    with pytest.raises(DataContractError, match="nonfinite"):
        region_radii(xyz, selection())


def test_static_selection_uses_residues_not_dynamic_frames() -> None:
    carbon = SimpleNamespace(atomic_number=6)
    residues = []
    for i in range(7):
        atoms = [SimpleNamespace(index=i, name="CA", element=carbon)]
        residues.append(SimpleNamespace(index=i, is_protein=i < 6, atoms=atoms))
    topology = SimpleNamespace(residues=residues, residue=lambda i: residues[i])
    xyz = np.zeros((7, 3))
    xyz[:, 0] = [1, 2, 3, 12, 13, 14, 0]
    source: Any = SimpleNamespace(ligand_residue_index=6)
    assert coupling._select_regions(topology, xyz, source, RegionRules()) == (
        (0, 1, 2),
        (3, 4, 5),
    )
    with pytest.raises(DataContractError, match="too few"):
        coupling._select_regions(topology, xyz, source, RegionRules(min_region_atoms=4))
    residues[0].atoms[0].name = "N"
    with pytest.raises(DataContractError, match="C-alpha"):
        coupling._select_regions(topology, xyz, source, RegionRules())


def extraction_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, RegionManifest]:
    source, chunk = fixture(tmp_path, monkeypatch)
    source = source.model_copy(update={"atoms": 6})
    atoms = [
        SimpleNamespace(name="CA", residue=SimpleNamespace(is_protein=True))
        for _ in range(6)
    ]
    topology = SimpleNamespace(n_atoms=6, atom=lambda i: atoms[i])
    monkeypatch.setattr(coupling, "verified_replica_topology", lambda *_: topology)
    chunk.xyz = np.zeros((50, 6, 3))
    chunk.xyz[:, :, 0] = np.arange(6) * 0.1
    chunk.n_atoms = 6
    monkeypatch.setitem(
        sys.modules, "mdtraj", SimpleNamespace(iterload=lambda *_, **__: [chunk])
    )
    regions = RegionManifest(
        source_hash="sha256:" + "a" * 64,
        rules=RegionRules(),
        selections={
            source.pdb_id: selection(
                topology_checksum=source.files["topology.prmtop"].checksum
            )
        },
    )
    return source, chunk, regions


def test_extraction_units_cadence_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, chunk, regions = extraction_fixture(tmp_path, monkeypatch)
    time, values = extract_region_series(tmp_path, source, regions)
    assert np.array_equal(time, np.arange(50) * 200)
    assert np.allclose(values, np.sqrt(2 / 3))  # Coordinates above were nm.
    chunk.time = np.arange(50) * 201.0
    with pytest.raises(DataContractError, match="cadence"):
        extract_region_series(tmp_path, source, regions)
    chunk.n_frames = 49
    with pytest.raises(DataContractError, match="frame count"):
        extract_region_series(tmp_path, source, regions)
    chunk.n_frames = 51
    with pytest.raises(DataContractError, match="dimensions"):
        extract_region_series(tmp_path, source, regions)
    with pytest.raises(DataContractError, match="budget"):
        extract_region_series(
            tmp_path,
            source,
            regions.model_copy(update={"rules": RegionRules(max_atoms=2)}),
        )
    with pytest.raises(DataContractError, match="topology differs"):
        extract_region_series(
            tmp_path,
            source,
            regions.model_copy(update={"selections": {source.pdb_id: selection()}}),
        )


def test_context_lag_direction_and_zero_variance() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=80)
    b = np.r_[0.0, a[:-1]]
    values = np.column_stack((b, a))
    assert _correlation(values, 1) == pytest.approx(1)
    backward = _correlation(values, -1)
    gain = _lagged_fit_gain(values, 1, 0)
    assert backward is not None and abs(backward) < 0.5
    assert gain is not None and gain > 0.99
    assert _correlation(np.ones((10, 2)), 0) is None
    assert _lagged_fit_gain(np.ones((10, 2)), 1, 0) is None
    with pytest.raises(ValueError, match="too few"):
        _lagged_fit_gain(values[:3], 2, 0)


def test_diagnostics_api_never_receives_targets() -> None:
    batch, targets = next(iter(pairs()))
    definitions = tuple(
        f.model_copy(update={"feature_id": name})
        for f, name in zip(
            batch.spec.dataset.features, ("distal_rg", "pocket_rg"), strict=True
        )
    )
    dataset = batch.spec.dataset.model_copy(update={"features": definitions})
    spec = batch.spec.model_copy(
        update={"dataset": dataset, "feature_ids": ("distal_rg", "pocket_rg")}
    )
    observed = ForecastBatch(spec, batch.indices, batch.context)
    settings = config(max_correlation_lag=2)
    rows = lag_diagnostics(observed, settings)
    assert len(rows) == len(batch.indices) * 5
    assert all(r["a_leads_b_correlation"] is None for r in rows)
    assert targets is not None
    targets[:] = 999999
    assert lag_diagnostics(observed, settings) == rows
    with pytest.raises(ValueError, match="ordered"):
        lag_diagnostics(batch, settings)
    with pytest.raises(ValueError, match="two paired"):
        lag_diagnostics(observed, config(max_correlation_lag=5))


def test_freezing_never_reads_trajectory_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from md_forecast.data.public.mdbind import MDBindSubset

    subset = MDBindSubset.model_validate_json(
        Path("configs/datasets/mdbind-common.json").read_text()
    )
    visited = []

    def static_topology(root: Path, source: Any) -> None:
        assert source.replica == 1
        visited.append(source.accession)

    monkeypatch.setattr(coupling, "verified_replica_topology", static_topology)
    monkeypatch.setattr(coupling, "_select_regions", lambda *_: ((0, 1, 2), (3, 4, 5)))
    # Backend provides PDB only. A trajectory request would fail immediately.
    monkeypatch.setitem(
        sys.modules,
        "mdtraj",
        SimpleNamespace(load_pdb=lambda _: SimpleNamespace(xyz=np.zeros((1, 6, 3)))),
    )
    result = coupling.freeze_regions(subset, tmp_path, RegionRules())
    assert len(visited) == 6
    assert result.source_hash == metadata_hash(subset)
    assert len(result.selections) == 6
