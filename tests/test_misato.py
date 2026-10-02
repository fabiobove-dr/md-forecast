"""Synthetic MISATO boundary, extraction, failure, and bounded-read checks."""

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from md_forecast.cli import main
from md_forecast.core.exceptions import AcquisitionError, DataContractError
from md_forecast.data.public import misato
from md_forecast.data.public.misato import (
    ExtractionConfig,
    ExtractionReport,
    MisatoSource,
    extract_misato,
    load_misato_source,
)
from md_forecast.data.registry import read_registry
from md_forecast.data.series import read_series, series_metadata


def artifact(path: Path) -> dict[str, object]:
    return {
        "mode": "md",
        "url": f"https://example.org/{path.name}",
        "size_bytes": path.stat().st_size,
        "checksum": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def fixture_source(tmp_path: Path) -> tuple[dict[str, Any], ExtractionConfig]:
    path = tmp_path / "input.hdf5"
    with h5py.File(path, "w") as file:
        for system in ("1AAA", "1BBB"):
            group = file.create_group(system)
            for index, key in enumerate(
                (
                    "frames_rmsd_ligand",
                    "frames_distance",
                    "frames_bSASA",
                    "frames_interaction_energy",
                )
            ):
                group.create_dataset(
                    key, data=np.arange(100, dtype=np.float64) + index * 100
                )
            # No payload allocation: reading these coordinates would materialize 24 GB.
            group.create_dataset(
                "trajectory_coordinates", shape=(100, 10_000_000, 3), dtype="float64"
            )
    raw = yaml.safe_load(Path("configs/datasets/misato.yaml").read_text())
    for filename, ids in (
        ("train_MD.txt", "1AAA\n1BBB\n"),
        ("val_MD.txt", "1CCC\n"),
        ("test_MD.txt", "1DDD\n"),
    ):
        split = tmp_path / filename
        split.write_text(ids)
        raw["artifacts"][filename] = artifact(split)
    raw["artifacts"]["MD.hdf5"] = artifact(path)
    raw["audit_evidence"] |= {
        "sample_url": "https://example.org/sample.hdf5",
        "sample_size_bytes": path.stat().st_size,
        "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    config = ExtractionConfig(
        input_path=path,
        splits_dir=tmp_path,
        output_dir=tmp_path / "output",
        system_ids=("1AAA", "1CCC", "9ZZZ"),
    )
    return raw, config


def refresh_artifact(raw: dict[str, Any], config: ExtractionConfig) -> MisatoSource:
    raw["artifacts"]["MD.hdf5"] = artifact(config.input_path)
    return MisatoSource.model_validate(raw)


def test_extract_subset_without_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, config = fixture_source(tmp_path)
    original = h5py.Dataset.__getitem__
    reads: list[str] = []

    def tracked(dataset: h5py.Dataset, selection: object) -> Any:
        assert dataset.name.startswith("/1AAA/frames_")
        reads.append(dataset.name)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
    report = extract_misato(MisatoSource.model_validate(raw), config)
    assert len(reads) == 4
    assert len(report.exported) == 1
    assert {(issue.source_system_id, issue.status) for issue in report.issues} == {
        ("1CCC", "missing"),
        ("9ZZZ", "dropped"),
    }
    assert (
        ExtractionReport.model_validate_json(
            (config.output_dir / "qc.json").read_text()
        )
        == report
    )
    assert str(tmp_path) not in report.model_dump_json()
    registry = read_registry(config.output_dir / "registry.json")
    record = registry.trajectories[0]
    assert record.pdb_id == "1AAA" and record.split == "train"
    assert record.system_id == record.split_group_id
    assert record.protein_id is None and record.ligand_id is None
    assert record.frame_interval_ps is None and record.duration_ns is None
    series = read_series(config.output_dir / "1AAA.parquet")
    assert series_metadata(series) == registry
    np.testing.assert_array_equal(series["time"].to_numpy(), np.arange(100))
    np.testing.assert_array_equal(
        series["buried_sasa"].to_numpy(), np.arange(100) + 200
    )
    assert set(path.name for path in config.output_dir.iterdir()) == {
        "registry.json",
        "qc.json",
        "1AAA.parquet",
    }
    with pytest.raises(DataContractError, match="already exists"):
        extract_misato(MisatoSource.model_validate(raw), config)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing", "missing"),
        ("shape", "shape/dtype"),
        ("dtype", "shape/dtype"),
        ("nan", "finite"),
        ("group", "not a dataset"),
        ("link", "linked"),
        ("system-dataset", "not a group"),
        ("system-link", "linked"),
    ],
)
def test_invalid_system_is_excluded(tmp_path: Path, fault: str, reason: str) -> None:
    raw, config = fixture_source(tmp_path)
    with h5py.File(config.input_path, "a") as file:
        key = "/1AAA/frames_rmsd_ligand"
        del file[key]
        if fault == "shape":
            file.create_dataset(key, data=np.ones(99))
        elif fault == "dtype":
            file.create_dataset(key, data=np.arange(100))
        elif fault == "nan":
            values = np.zeros(100)
            values[2] = np.nan
            file.create_dataset(key, data=values)
        elif fault == "group":
            file.create_group(key)
        elif fault == "link":
            file[key] = h5py.ExternalLink("untrusted.hdf5", "values")
        elif fault == "system-dataset":
            del file["1AAA"]
            file.create_dataset("1AAA", data=np.ones(1))
        elif fault == "system-link":
            del file["1AAA"]
            file["1AAA"] = h5py.SoftLink("/1BBB")
    report = extract_misato(refresh_artifact(raw, config), config)
    assert not report.exported
    assert report.issues[0].status == "dropped"
    assert reason in report.issues[0].reason
    assert read_registry(config.output_dir / "registry.json").trajectories == ()
    assert not list(config.output_dir.glob("*.parquet"))


def test_sample_provenance(tmp_path: Path) -> None:
    raw, config = fixture_source(tmp_path)
    sample_config = ExtractionConfig.model_validate(
        config.model_dump() | {"artifact": "sample"}
    )
    report = extract_misato(MisatoSource.model_validate(raw), sample_config)
    assert str(report.source.provenance_uri) == raw["audit_evidence"]["sample_url"]
    assert (
        report.source.source_checksum
        == "sha256:" + raw["audit_evidence"]["sample_sha256"]
    )
    for record in read_registry(config.output_dir / "registry.json").trajectories:
        assert record.source_checksum == report.source.source_checksum


def test_integrity_and_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, config = fixture_source(tmp_path)
    source = MisatoSource.model_validate(raw)
    config.input_path.write_bytes(config.input_path.read_bytes() + b"corrupt")
    with pytest.raises(AcquisitionError):
        extract_misato(source, config)
    assert not config.output_dir.exists()
    source = refresh_artifact(raw, config)

    def failed_write(*args: object) -> None:
        raise DataContractError("disk full")

    monkeypatch.setattr(misato, "write_series", failed_write)
    with pytest.raises(DataContractError, match="disk full"):
        extract_misato(source, config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".misato-*"))
    config.output_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(DataContractError, match="already exists"):
        extract_misato(source, config)


@pytest.mark.parametrize("ids", ["1AAA\n1AAA\n", "1AAA\n1CCC\n", "bad\n", "\n"])
def test_invalid_official_splits(tmp_path: Path, ids: str) -> None:
    raw, config = fixture_source(tmp_path)
    path = tmp_path / "train_MD.txt"
    path.write_text(ids)
    raw["artifacts"][path.name] = artifact(path)
    with pytest.raises(DataContractError, match="split ID"):
        extract_misato(MisatoSource.model_validate(raw), config)
    assert not config.output_dir.exists()


def test_source_and_selection_validation(tmp_path: Path) -> None:
    raw, config = fixture_source(tmp_path)
    for changes in (
        {"system_ids": ()},
        {"system_ids": ("1AAA", "1AAA")},
        {"system_ids": ("../bad",)},
        {"artifact": "unknown"},
    ):
        with pytest.raises(ValidationError):
            ExtractionConfig.model_validate(config.model_dump() | changes)
    for field, value in (
        ("source_key", "new"),
        ("unit", "count"),
        ("shape", [99]),
        ("dtype", "float32"),
        ("time_resolved", False),
    ):
        changed = json.loads(json.dumps(raw))
        changed["features"]["ligand_rmsd"][field] = value
        with pytest.raises(ValidationError):
            MisatoSource.model_validate(changed)
    changed = json.loads(json.dumps(raw))
    del changed["features"]["ligand_rmsd"]
    with pytest.raises(ValidationError):
        MisatoSource.model_validate(changed)
    for key, time_value in (
        ("timestamp_key", "time"),
        ("verified_frame_interval_ps", 80.0),
    ):
        changed = json.loads(json.dumps(raw))
        changed["time_axis"][key] = time_value
        with pytest.raises(ValidationError):
            MisatoSource.model_validate(changed)
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_misato_source(path) == MisatoSource.model_validate(raw)
    for content in ("[invalid", "{}"):
        path.write_text(content)
        with pytest.raises(DataContractError):
            load_misato_source(path)
    with pytest.raises(DataContractError):
        load_misato_source(tmp_path / "missing")


def test_cli_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, config = fixture_source(tmp_path)
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw))
    args = [
        "md-forecast",
        "extract-misato",
        "--source",
        str(path),
        "--input",
        str(config.input_path),
        "--output",
        str(config.output_dir),
        "--splits",
        str(config.splits_dir),
        "--systems",
        "1AAA",
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    assert (config.output_dir / "1AAA.parquet").is_file()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err


def test_missing_artifact_contracts_and_hdf_failure(tmp_path: Path) -> None:
    raw, config = fixture_source(tmp_path)
    for name in ("MD.hdf5", "train_MD.txt"):
        changed = json.loads(json.dumps(raw))
        del changed["artifacts"][name]
        with pytest.raises(DataContractError, match="lacks"):
            extract_misato(MisatoSource.model_validate(changed), config)
    config.input_path.write_text("not HDF5")
    with pytest.raises(DataContractError, match="before publication"):
        extract_misato(refresh_artifact(raw, config), config)
    assert not config.output_dir.exists()
    assert not list(tmp_path.glob(".misato-*"))
