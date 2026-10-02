"""Independently checkable geometry, bounded reads and canonical extraction."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import ValidationError
from test_misato import fixture_source, refresh_artifact

import md_forecast.features.structural as structural
from md_forecast.cli import main
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.public.misato import extract_misato
from md_forecast.data.registry import read_registry
from md_forecast.data.series import read_series, series_metadata
from md_forecast.features.structural import (
    AtomPair,
    StructuralConfig,
    load_structural_config,
    structural_values,
)


def config(**changes: object) -> StructuralConfig:
    return StructuralConfig.model_validate(
        {
            "contact_cutoff_angstrom": 4.0,
            "frame_chunk_size": 2,
            "atom_tile_size": 1,
            "max_atoms": 10,
            "max_frames": 100,
            "max_working_bytes": 100000,
        }
        | changes
    )


def test_isolated_ligand_sasa_analytic_sphere(tmp_path: Path) -> None:
    pytest.importorskip("mdtraj")
    with h5py.File(tmp_path / "sasa.h5", "w") as file:
        group = file.create_group("1AAA")
        values = np.zeros((3, 2, 3), dtype=np.float64)
        values[:, 1, 0] = [2, 3, 4]
        group.create_dataset("trajectory_coordinates", data=values)
        group.create_dataset("atoms_number", data=np.array([6, 6], dtype=np.int64))
        group.create_dataset(
            "molecules_begin_atom_index", data=np.array([0, 1], dtype=np.int64)
        )
        settings = config(
            ligand_sasa={"probe_radius_angstrom": 1.4, "sphere_points": 960}
        )
        output = structural_values(group, "1AAA", 3, settings)
        # MDTraj carbon radius 1.7 Angstrom + probe 1.4: a complete isolated sphere.
        # Protein is deliberately close: including it as an occluder would fail.
        np.testing.assert_allclose(output[:, -1], 4 * np.pi * 3.1**2, rtol=1e-6)
        np.testing.assert_array_equal(
            output,
            structural_values(
                group,
                "1AAA",
                3,
                config(
                    frame_chunk_size=1,
                    ligand_sasa={"probe_radius_angstrom": 1.4, "sphere_points": 960},
                ),
            ),
        )
        assert settings.feature_definitions()[-1].unit == "angstrom_squared"


def test_sasa_backend_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(structural, "version", missing)
    with pytest.raises(DataContractError, match="structural extra"):
        structural.validate_sasa_backend()
    raw, extraction = fixture_source(tmp_path)
    with pytest.raises(DataContractError, match="structural extra"):
        extract_misato(
            refresh_artifact(raw, extraction),
            extraction,
            structural=config(
                ligand_sasa={"probe_radius_angstrom": 1.4, "sphere_points": 960}
            ),
        )
    assert not extraction.output_dir.exists()
    monkeypatch.setattr(structural, "version", lambda name: "wrong")
    with pytest.raises(DataContractError, match="requires MDTraj"):
        structural.validate_sasa_backend()


def test_sasa_nonfinite_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mdtraj = pytest.importorskip("mdtraj")
    monkeypatch.setattr(
        mdtraj, "shrake_rupley", lambda *args, **kwargs: np.full((1, 3), np.nan)
    )
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        with pytest.raises(DataContractError, match="finite"):
            structural_values(
                group,
                "1AAA",
                3,
                config(
                    ligand_sasa={"probe_radius_angstrom": 1.4, "sphere_points": 960}
                ),
            )


@pytest.mark.parametrize("fault", ["element", "kernel"])
def test_sasa_backend_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    mdtraj = pytest.importorskip("mdtraj")

    def failed(*args: object, **kwargs: object) -> Any:
        raise KeyError("unsupported source")

    if fault == "element":
        monkeypatch.setattr(mdtraj.element.Element, "getByAtomicNumber", failed)
    else:
        monkeypatch.setattr(mdtraj, "shrake_rupley", failed)
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        with pytest.raises(DataContractError, match="SASA backend"):
            structural_values(
                group,
                "1AAA",
                3,
                config(
                    ligand_sasa={"probe_radius_angstrom": 1.4, "sphere_points": 960}
                ),
            )


def geometry(group: h5py.Group, frames: int = 3) -> None:
    # Protein heavy x=0,10; ligand heavy x=4,14. Hydrogen decoys are ignored.
    values = np.zeros((frames, 6, 3), dtype=np.float64)
    values[:, :, 0] = [0, 10, 4, 4, 14, 0]
    for frame in range(frames):
        values[frame, 3:5, 0] += min(frame, 2)
    group.create_dataset("trajectory_coordinates", data=values)
    group.create_dataset(
        "atoms_number", data=np.array([6, 8, 1, 6, 7, 1], dtype=np.int64)
    )
    group.create_dataset(
        "molecules_begin_atom_index", data=np.array([0, 3], dtype=np.int64)
    )


def test_exact_geometry_and_chunk_tile_invariance(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        expected = np.array([[2, 1, 4], [0, 0, 5], [1, 0, 4]], dtype=np.float64)
        # At frame 2 the new contact is NOT one of the two initial contacts.
        np.testing.assert_array_equal(
            structural_values(group, "1AAA", 3, config()), expected
        )
        np.testing.assert_array_equal(
            structural_values(
                group, "1AAA", 3, config(frame_chunk_size=3, atom_tile_size=4)
            ),
            expected,
        )
        settings = config(
            regions=[
                {
                    "feature_id": "selected_pair_distance",
                    "description": "Explicit source atoms 0 and 3",
                    "selections": {"1AAA": {"left": [0], "right": [3]}},
                }
            ]
        )
        output = structural_values(group, "1AAA", 3, settings)
        np.testing.assert_array_equal(output[:, -1], [4, 5, 6])
        np.testing.assert_array_equal(output[:, :3], expected)
        # Rigid translation does not alter distances or contacts.
        group["trajectory_coordinates"][:] += 1000
        np.testing.assert_array_equal(
            structural_values(group, "1AAA", 3, settings), output
        )


def test_no_future_reference_or_unselected_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        file.create_group("1BBB")
        original = h5py.Dataset.__getitem__
        coordinate_reads = []

        def tracked(dataset: h5py.Dataset, selection: Any) -> Any:
            assert dataset.name.startswith("/1AAA/")
            if dataset.name.endswith("trajectory_coordinates"):
                assert isinstance(selection, slice)
                assert selection.stop - selection.start <= 2
                coordinate_reads.append((selection.start, selection.stop))
            return original(dataset, selection)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
        before = structural_values(group, "1AAA", 3, config())[0].copy()
        assert coordinate_reads == [(0, 1), (0, 2), (2, 3)]
        # A later frame cannot change initial atom selections or contact identities.
        group["trajectory_coordinates"][2] = np.full((6, 3), 1000.0)
        np.testing.assert_array_equal(
            structural_values(group, "1AAA", 3, config())[0], before
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("coordinate-shape", "coordinates require"),
        ("coordinate-dtype", "coordinates require"),
        ("number-shape", "atomic number shape"),
        ("number-dtype", "atomic number shape"),
        ("number-range", "invalid atomic numbers"),
        ("segments-dtype", "segment shape"),
        ("segments-shape", "segment shape"),
        ("segments-values", "segment boundaries"),
        ("no-heavy", "empty receptor"),
        ("no-reference", "reference contacts"),
        ("nonfinite", "nonfinite"),
        ("missing", "missing or linked"),
        ("link", "missing or linked"),
        ("group", "not a dataset"),
        ("overflow", "overflowed"),
    ],
)
def test_invalid_geometry(tmp_path: Path, fault: str, reason: str) -> None:
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        if fault == "coordinate-shape":
            del group["trajectory_coordinates"]
            group.create_dataset(
                "trajectory_coordinates", shape=(3, 6, 2), dtype="float64"
            )
        elif fault == "coordinate-dtype":
            del group["trajectory_coordinates"]
            group.create_dataset(
                "trajectory_coordinates", shape=(3, 6, 3), dtype="float32"
            )
        elif fault.startswith("number-"):
            del group["atoms_number"]
            values = {
                "number-shape": [6],
                "number-dtype": [6.0] * 6,
                "number-range": [0] * 6,
            }[fault]
            group.create_dataset("atoms_number", data=np.array(values))
        elif fault.startswith("segments-"):
            del group["molecules_begin_atom_index"]
            values = {
                "segments-dtype": [0.0, 3.0],
                "segments-shape": [0],
                "segments-values": [1, 3],
            }[fault]
            group.create_dataset("molecules_begin_atom_index", data=np.array(values))
        elif fault == "no-heavy":
            group["atoms_number"][:] = 1
        elif fault == "no-reference":
            group["trajectory_coordinates"][:, 3:5, :] = 1000
        elif fault == "nonfinite":
            group["trajectory_coordinates"][2, 0, 0] = np.nan
        elif fault == "overflow":
            group["trajectory_coordinates"][2, 0, 0] = 1e308
        else:
            del group["atoms_number"]
            if fault == "link":
                group["atoms_number"] = h5py.SoftLink("/untrusted")
            elif fault == "group":
                group.create_group("atoms_number")
        with pytest.raises(DataContractError, match=reason):
            structural_values(group, "1AAA", 3, config())


def test_limits_reject_before_coordinate_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)

        def no_read(*args: object) -> None:
            raise AssertionError("budget failure loaded payload")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", no_read)
        for changes in ({"max_atoms": 5}, {"max_frames": 2}, {"max_working_bytes": 1}):
            with pytest.raises(DataContractError, match="budget"):
                structural_values(group, "1AAA", 3, config(**changes))


def test_sasa_limits_reject_before_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        original = h5py.Dataset.__getitem__

        def tracked(dataset: h5py.Dataset, selection: Any) -> Any:
            assert not dataset.name.endswith("trajectory_coordinates")
            return original(dataset, selection)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
        for changes in (
            {"max_atoms": 2, "sphere_points": 960},
            {"max_atoms": 512, "sphere_points": 100000000},
        ):
            with pytest.raises(DataContractError, match="budget"):
                structural_values(
                    group,
                    "1AAA",
                    3,
                    config(ligand_sasa={"probe_radius_angstrom": 1.4, **changes}),
                )


def test_config_identity_and_region_validation(tmp_path: Path) -> None:
    settings = load_structural_config(Path("configs/features/misato-geometry.yaml"))
    assert settings.feature_set_version != config().feature_set_version
    assert (
        config(contact_cutoff_angstrom=4.1).feature_set_version
        != config().feature_set_version
    )
    assert all(
        "reference" in feature.definition or "Heavy atoms" in feature.definition
        for feature in config().feature_definitions()
    )
    for changes in (
        {"contact_cutoff_angstrom": 0},
        {"reference_frame": 1},
        {"distance_policy": "minimum-image"},
        {"frame_chunk_size": 0},
        {
            "regions": [
                {
                    "feature_id": "time",
                    "description": "x",
                    "selections": {"1AAA": {"left": [0], "right": [1]}},
                }
            ]
        },
    ):
        with pytest.raises(ValidationError):
            config(**changes)
    for indices in (
        {"left": [0, 0], "right": [1]},
        {"left": [1], "right": [1]},
        {"left": [-1], "right": [1]},
    ):
        with pytest.raises(ValidationError):
            AtomPair.model_validate(indices)
    bad = tmp_path / "invalid.yaml"
    bad.write_text("[")
    for path in (bad, tmp_path / "missing.yaml"):
        with pytest.raises(DataContractError, match="cannot load"):
            load_structural_config(path)
    with h5py.File(tmp_path / "coordinates.h5", "w") as file:
        group = file.create_group("1AAA")
        geometry(group)
        for selections, reason in (
            ({"1BBB": {"left": [0], "right": [3]}}, "missing explicit"),
            ({"1AAA": {"left": [0], "right": [7]}}, "source bounds"),
            ({"1AAA": {"left": [0], "right": [2]}}, "heavy atoms"),
        ):
            setting = config(
                regions=[
                    {
                        "feature_id": "region_distance",
                        "description": "test",
                        "selections": selections,
                    }
                ]
            )
            with pytest.raises(DataContractError, match=reason):
                structural_values(group, "1AAA", 3, setting)


def test_canonical_structural_export_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, extraction = fixture_source(tmp_path)
    with h5py.File(extraction.input_path, "a") as file:
        del file["1AAA/trajectory_coordinates"]
        geometry(file["1AAA"], 100)
    source = refresh_artifact(raw, extraction)
    report = extract_misato(source, extraction, structural=config())
    assert report.structural_config == config()
    registry = read_registry(extraction.output_dir / "registry.json")
    assert registry.dataset.feature_set_version == config().feature_set_version
    table = read_series(extraction.output_dir / "1AAA.parquet")
    assert series_metadata(table) == registry
    assert registry.trajectories[0].feature_set_version == config().feature_set_version
    np.testing.assert_array_equal(
        table["protein_ligand_contact_count"].to_numpy()[:3], [2, 0, 1]
    )
    import yaml

    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(raw))
    config_path = tmp_path / "geometry.yaml"
    config_path.write_text(yaml.safe_dump(config().model_dump(mode="json")))
    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "md-forecast",
            "extract-misato",
            "--source",
            str(source_path),
            "--input",
            str(extraction.input_path),
            "--output",
            str(output),
            "--splits",
            str(tmp_path),
            "--systems",
            "1AAA",
            "--structural-config",
            str(config_path),
        ],
    )
    main()
    assert read_registry(output / "registry.json") == registry
