"""Synthetic regression checks for canonical metadata and vectorized storage."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from md_forecast.core.constants import SERIES_METADATA_KEY, DatasetId
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.registry import (
    Registry,
    atomic_output,
    read_registry,
    write_registry,
)
from md_forecast.data.schemas import (
    DatasetConfig,
    ExperimentConfig,
    TrajectoryManifest,
    stable_system_id,
    stable_trajectory_id,
)
from md_forecast.data.series import (
    read_series,
    series_metadata,
    to_arrow,
    validate_series,
    write_series,
)


def config(dataset: str = "misato") -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "dataset_id": dataset,
            "dataset_version": "1.0",
            "feature_set_version": "native-v1",
            "features": [
                {
                    "feature_id": "rmsd",
                    "unit": "angstrom",
                    "description": "Native RMSD",
                    "definition": "Source-native ligand RMSD; alignment unspecified",
                }
            ],
        }
    )


def manifest(**changes: object) -> TrajectoryManifest:
    data: dict[str, object] = {
        "dataset_id": "misato",
        "dataset_version": "1.0",
        "source_record": "record-1",
        "source_checksum": "sha256:" + "a" * 64,
        "license_id": "CC-BY-4.0",
        "provenance_uri": "https://example.org/source.hdf5",
        "system_id": "misato:system:10GS",
        "pdb_id": "10GS",
        "protein_id": None,
        "ligand_id": None,
        "trajectory_id": "trajectory-1",
        "replicate_id": "1",
        "split_group_id": "10GS",
        "frame_count": 3,
        "feature_set_version": "native-v1",
    }
    return TrajectoryManifest.model_validate(data | changes)


def table(**changes: object) -> pa.Table:
    return to_arrow(
        manifest(**changes),
        config(),
        np.array([0.0, 1.0, 2.0]),
        np.array([[1.0], [2.0], [3.0]]),
    )


@pytest.mark.parametrize("dataset", ["misato", "mdbind"])
def test_metadata_and_parquet_round_trip(tmp_path: Path, dataset: str) -> None:
    record = manifest(dataset_id=dataset)
    registry = Registry(dataset=config(dataset), trajectories=(record,))
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, registry)
    assert read_registry(registry_path) == registry
    assert TrajectoryManifest.model_validate_json(record.model_dump_json()) == record
    series = to_arrow(
        record,
        config(dataset),
        np.arange(3, dtype=np.float64),
        np.array([[1.0], [2.0], [3.0]]),
    )
    series_path = tmp_path / "trajectory.parquet"
    write_series(series_path, series)
    loaded = read_series(series_path)
    assert loaded.equals(series, check_metadata=True)
    assert series_metadata(loaded) == registry
    assert "frame_interval_ps" in json.loads(record.model_dump_json())
    assert record.frame_interval_ps is None
    assert record.duration_ns is None
    assert str(tmp_path) not in registry_path.read_text()


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": "1"},
        {"dataset_id": "other"},
        {"frame_count": 0},
        {"frame_count": True},
        {"duration_ns": float("nan")},
        {"time_unit": "seconds"},
        {"time_unit": "ps"},
        {"frame_interval_ps": 80},
        {"duration_ns": 8},
        {"sampling_status": "verified"},
        {"trajectory_id": " "},
        {"source_checksum": "sha256:bad"},
        {"license_id": ""},
        {"provenance_uri": "/home/user/data.hdf5"},
        {"provenance_uri": "file:///tmp/data"},
        {"provenance_uri": "https://user:secret@example.org/data"},
        {"absolute_path": "/home/user/data.hdf5"},
    ],
)
def test_invalid_manifest(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        manifest(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"sampling_status": "assumed"},
        {"frame_interval_ps": -1},
        {"duration_ns": 8},
        {"duration_ns": float("inf")},
    ],
)
def test_invalid_physical_metadata(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        manifest(
            **(
                {
                    "time_unit": "ps",
                    "sampling_status": "verified",
                    "frame_interval_ps": 80.0,
                    "duration_ns": 0.16,
                }
                | changes
            )
        )


@pytest.mark.parametrize(
    "unit,time", [("ps", [0.0, 80.0, 160.0]), ("ns", [0.0, 0.08, 0.16])]
)
def test_physical_sampling(unit: str, time: list[float]) -> None:
    record = manifest(
        time_unit=unit,
        sampling_status="assumed",
        sampling_note="Nominal source interval",
        frame_interval_ps=80.0,
        duration_ns=0.16,
    )
    series = to_arrow(record, config(), np.array(time), np.ones((3, 1)))
    assert validate_series(series).trajectories[0] == record
    irregular = manifest(time_unit=unit, sampling_status="verified", duration_ns=0.16)
    to_arrow(
        irregular, config(), np.array([0.0, time[1] / 2, time[2]]), np.ones((3, 1))
    )
    with pytest.raises(DataContractError, match="frame_interval_ps"):
        to_arrow(
            record, config(), np.array([0.0, time[1] / 2, time[2]]), np.ones((3, 1))
        )
    with pytest.raises(DataContractError, match="duration_ns"):
        to_arrow(
            record, config(), np.array([0.0, time[1], time[2] * 2]), np.ones((3, 1))
        )


def test_single_frame() -> None:
    record = manifest(
        frame_count=1,
        time_unit="ps",
        sampling_status="verified",
        frame_interval_ps=80.0,
        duration_ns=0.0,
    )
    to_arrow(record, config(), np.array([0.0]), np.ones((1, 1)))


def test_stable_ids() -> None:
    system = stable_system_id(DatasetId.MISATO, "10GS")
    assert system == "misato:system:10GS"
    assert stable_system_id(DatasetId.MDBIND, "10GS") != system
    assert stable_trajectory_id(system, "source:a", "1") == stable_trajectory_id(
        system, "source:a", "1"
    )
    assert stable_trajectory_id(system, "a:b", "c") != stable_trajectory_id(
        system, "a", "b:c"
    )
    assert stable_trajectory_id(system, "a", "1") != stable_trajectory_id(
        system, "a", "2"
    )
    for identity in ("", " a", "a "):
        with pytest.raises(ValueError):
            stable_system_id(DatasetId.MISATO, identity)


def test_registry_uniqueness_and_consistency(tmp_path: Path) -> None:
    first = manifest()
    second = manifest(trajectory_id="trajectory-2", replicate_id="2")
    registry = Registry(dataset=config(), trajectories=(second, first))
    path = tmp_path / "registry.json"
    write_registry(path, registry)
    assert read_registry(path).trajectories == (first, second)
    before = path.read_bytes()
    write_registry(path, Registry(dataset=config(), trajectories=(first, second)))
    assert path.read_bytes() == before
    for records in (
        (first, first),
        (first, manifest(trajectory_id="2", protein_id="different")),
        (manifest(dataset_version="2"),),
        (manifest(feature_set_version="2"),),
        (
            first,
            manifest(trajectory_id="2", split="train"),
            manifest(trajectory_id="3", split="test"),
        ),
    ):
        with pytest.raises(ValidationError):
            Registry(dataset=config(), trajectories=records)
    Registry(
        dataset=config(),
        trajectories=(
            first,
            manifest(trajectory_id="2", split="train"),
            manifest(trajectory_id="3", split="train"),
        ),
    )


def test_config_validation() -> None:
    data = config().model_dump(mode="json")
    invalid_configs: list[dict[str, object]] = [
        {"schema_version": 2},
        {"features": []},
        {"features": data["features"] * 2},
    ]
    for changes in invalid_configs:
        with pytest.raises(ValidationError):
            DatasetConfig.model_validate(data | changes)
    for feature_changes in (
        {"unit": "meters"},
        {"feature_id": "time"},
        {"feature_id": "bad:id"},
    ):
        feature = data["features"][0] | feature_changes
        with pytest.raises(ValidationError):
            DatasetConfig.model_validate(data | {"features": [feature]})
    experiment = ExperimentConfig(dataset=config(), seed=42, trajectory_ids=("a", "b"))
    assert (
        ExperimentConfig.model_validate_json(experiment.model_dump_json()) == experiment
    )
    invalid_experiments: list[dict[str, object]] = [
        {"seed": -1},
        {"schema_version": 2},
        {"trajectory_ids": ["a", "a"]},
    ]
    for changes in invalid_experiments:
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(experiment.model_dump() | changes)


@pytest.mark.parametrize(
    "time",
    [
        [0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
        [0.0, float("nan"), 2.0],
        [0.0, 1.0, 3.0],
    ],
)
def test_invalid_axes(time: list[float]) -> None:
    with pytest.raises(DataContractError):
        to_arrow(manifest(), config(), np.array(time), np.ones((3, 1)))


def test_invalid_tables() -> None:
    series = table()
    broken = [
        series.replace_schema_metadata(None),
        series.slice(0, 2),
        series.rename_columns(["time", "other"]),
        series.set_column(1, "rmsd", pa.array([1.0, None, 3.0])),
        series.set_column(1, "rmsd", pa.array([1, 2, 3])),
        series.set_column(1, "rmsd", pa.array([1.0, float("inf"), 3.0])),
        series.replace_schema_metadata({SERIES_METADATA_KEY: b"invalid"}),
    ]
    empty_metadata = (
        Registry(dataset=config(), trajectories=()).model_dump_json().encode()
    )
    broken.append(series.replace_schema_metadata({SERIES_METADATA_KEY: empty_metadata}))
    future = json.loads(series.schema.metadata[SERIES_METADATA_KEY])
    future["schema_version"] = 2
    broken.append(
        series.replace_schema_metadata(
            {SERIES_METADATA_KEY: json.dumps(future).encode()}
        )
    )
    for invalid in broken:
        with pytest.raises(DataContractError):
            validate_series(invalid)
    for time, values in ((np.ones(2), np.ones((3, 1))), (np.ones(3), np.ones((3, 2)))):
        with pytest.raises(DataContractError, match="expected time"):
            to_arrow(manifest(), config(), time, values)


def test_failed_publication_and_read_errors(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    path.write_text("valuable existing output")
    with pytest.raises(RuntimeError):
        with atomic_output(path) as temporary:
            temporary.write_text("partial")
            raise RuntimeError("interrupted")
    assert path.read_text() == "valuable existing output"
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(DataContractError):
        write_series(path, table().slice(0, 1))
    assert path.read_text() == "valuable existing output"
    for reader in (read_registry, read_series):
        with pytest.raises(DataContractError):
            reader(path)
        with pytest.raises(DataContractError):
            reader(tmp_path / "missing")
    with pytest.raises(DataContractError):
        write_registry(
            tmp_path / "missing" / "registry.json",
            Registry(dataset=config(), trajectories=()),
        )
    with pytest.raises(DataContractError):
        write_series(tmp_path / "missing" / "series.parquet", table())
