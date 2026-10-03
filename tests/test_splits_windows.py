"""Split dependencies, deterministic lazy grids, and deliberate leakage regressions."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data import preprocessing
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.preprocessing import (
    FitRegion,
    ScalerMetadata,
    ScalingConfig,
    bind_experiment,
    fit_context_scaler,
    fit_training_scaler,
    transform,
    validate_scaler,
)
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import DatasetConfig, ExperimentConfig, TrajectoryManifest
from md_forecast.data.series import to_arrow
from md_forecast.data.splits import SplitConfig, SplitManifest, build_split
from md_forecast.data.windows import WindowConfig, iter_windows, validate_window


def dataset() -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "dataset_id": "misato",
            "dataset_version": "test",
            "feature_set_version": "synthetic-v1",
            "features": [
                {
                    "feature_id": name,
                    "unit": "dimensionless",
                    "description": name,
                    "definition": "Synthetic test channel",
                }
                for name in ("position", "constant")
            ],
        }
    )


def record(index: int, **changes: object) -> TrajectoryManifest:
    values: dict[str, object] = {
        "dataset_id": "misato",
        "dataset_version": "test",
        "source_record": "synthetic",
        "source_checksum": "sha256:" + "a" * 64,
        "license_id": "MIT",
        "provenance_uri": "https://example.org/fixture",
        "system_id": f"system-{index}",
        "pdb_id": None,
        "protein_id": None,
        "ligand_id": None,
        "trajectory_id": f"trajectory-{index}",
        "replicate_id": "native",
        "split_group_id": f"group-{index}",
        "frame_count": 6,
        "feature_set_version": "synthetic-v1",
        "split": ("train" if index < 3 else "validation" if index < 5 else "test"),
    }
    return TrajectoryManifest.model_validate(values | changes)


def registry() -> Registry:
    return Registry(
        dataset=dataset(), trajectories=tuple(record(index) for index in range(6))
    )


def official() -> SplitManifest:
    return build_split(registry(), SplitConfig(mode="official", seed=42))


def grid() -> WindowConfig:
    return WindowConfig(contexts=(1.0, 3.0), horizons=(1.0, 2.0), stride_frames=2)


def tables(manifest: SplitManifest) -> dict[str, pa.Table]:
    result: dict[str, pa.Table] = {}
    for index, source in enumerate(manifest.registry.trajectories):
        values = np.column_stack(
            [np.arange(6, dtype=np.float64) + index * 10, np.full(6, 7.0)]
        )
        result[source.trajectory_id] = to_arrow(
            source, manifest.registry.dataset, np.arange(6, dtype=np.float64), values
        )
    return result


def test_official_and_grouped_determinism(tmp_path: Path) -> None:
    source = registry()
    manifest = official()
    assert [entry.split for entry in manifest.assignments] == [
        record.split for record in source.trajectories
    ]
    config = SplitConfig(mode="grouped", seed=7, ratios=(0.5, 1 / 3, 1 / 6))
    grouped = build_split(source, config)
    reversed_source = Registry(
        dataset=source.dataset, trajectories=tuple(reversed(source.trajectories))
    )
    assert build_split(reversed_source, config) == grouped
    assert metadata_hash(build_split(reversed_source, config)) == metadata_hash(grouped)
    assert [
        sum(entry.split == split for entry in grouped.assignments) for split in Split
    ] == [3, 2, 1]
    assert (
        build_split(source, SplitConfig(mode="grouped", seed=8, ratios=config.ratios))
        != grouped
    )
    path = tmp_path / "split.json"
    checksum = write_metadata(path, grouped)
    assert checksum == metadata_hash(grouped)
    assert read_metadata(path, SplitManifest) == grouped
    raw = json.loads(path.read_text())
    raw["payload"]["config"]["seed"] = 99
    path.write_text(json.dumps(raw))
    with pytest.raises(DataContractError, match="checksum"):
        read_metadata(path, SplitManifest)


def test_transitive_dependencies_and_external_families() -> None:
    sources = (
        record(0),
        record(1, system_id="system-0"),
        record(2, split_group_id="group-1"),
        record(3, split="train"),
        record(4),
        record(5),
    )
    source = Registry(dataset=dataset(), trajectories=sources)
    labels = {f"trajectory-{index}": f"family-{index}" for index in range(6)}
    labels["trajectory-3"] = labels["trajectory-2"]
    config = SplitConfig(
        mode="grouped",
        seed=3,
        ratios=(1 / 3, 1 / 3, 1 / 3),
        group_field="structural_family",
    )
    manifest = build_split(source, config, labels)
    entries = {entry.trajectory_id: entry for entry in manifest.assignments}
    assert len({entries[f"trajectory-{index}"].group_id for index in range(4)}) == 1
    for field in ("system_id", "split_group_id"):
        groups: dict[str, set[Split]] = {}
        for source_record in sources:
            groups.setdefault(getattr(source_record, field), set()).add(
                entries[source_record.trajectory_id].split
            )
        assert all(len(partitions) == 1 for partitions in groups.values())
    for split in Split:
        assert all(
            window.split == entries[window.trajectory_id].split
            for window in iter_windows(manifest, grid(), split)
        )


def test_invalid_split_contracts() -> None:
    for changes in (
        {"ratios": None},
        {"ratios": (0.5, 0.5, 0.5)},
        {"ratios": (-0.1, 0.5, 0.6)},
        {"ratios": (float("nan"), 0.5, 0.5)},
        {"seed": -1},
        {"mode": "unknown"},
    ):
        with pytest.raises(ValidationError):
            SplitConfig.model_validate(
                {"mode": "grouped", "seed": 1, "ratios": [0.5, 0.25, 0.25]} | changes
            )
    with pytest.raises(ValidationError):
        SplitConfig(mode="official", seed=1, ratios=(1.0, 0.0, 0.0))
    for source in (
        Registry(dataset=dataset(), trajectories=()),
        Registry(dataset=dataset(), trajectories=(record(0, split=None),)),
        Registry(
            dataset=dataset(), trajectories=(record(0), record(3, system_id="system-0"))
        ),
    ):
        with pytest.raises(DataContractError):
            build_split(source, SplitConfig(mode="official", seed=1))
    with pytest.raises(DataContractError, match="too few"):
        build_split(
            Registry(dataset=dataset(), trajectories=(record(0),)),
            SplitConfig(mode="grouped", seed=1, ratios=(0.8, 0.1, 0.1)),
        )
    with pytest.raises(DataContractError, match="missing protein_id"):
        build_split(
            registry(), SplitConfig(mode="official", seed=1, group_field="protein_id")
        )
    with pytest.raises(DataContractError, match="cannot be overridden"):
        build_split(registry(), SplitConfig(mode="official", seed=1), {})
    custom = SplitConfig(
        mode="grouped", seed=1, ratios=(1.0, 0.0, 0.0), group_field="target"
    )
    for labels in (
        None,
        {},
        {entry.trajectory_id: " " for entry in registry().trajectories},
    ):
        with pytest.raises(DataContractError):
            build_split(registry(), custom, labels)
    raw = official().model_dump()
    raw["assignments"][0]["split"] = "test"
    with pytest.raises(ValidationError):
        SplitManifest.model_validate(raw)
    raw = official().model_dump()
    raw["group_labels"]["trajectory-0"] = "wrong"
    with pytest.raises(ValidationError):
        SplitManifest.model_validate(raw)


def test_lazy_variable_windows_and_bounds() -> None:
    manifest = official()
    windows = iter_windows(manifest, grid(), Split.TRAIN)
    assert iter(windows) is windows
    emitted = list(windows)
    assert len(emitted) == 24
    assert emitted == list(iter_windows(manifest, grid(), Split.TRAIN))
    assert all(
        window.context_slice.stop == window.target_slice.start for window in emitted
    )
    assert all(window.target_slice.stop <= 6 for window in emitted)
    assert all(window.context_span_ps is None for window in emitted)
    identities = [
        {window.trajectory_id for window in iter_windows(manifest, grid(), split)}
        for split in Split
    ]
    assert not identities[0] & identities[1] and not identities[0] & identities[2]
    exact = WindowConfig(contexts=(4.0,), horizons=(2.0,), stride_frames=1)
    assert len(list(iter_windows(manifest, exact, Split.TEST))) == 1
    impossible = WindowConfig(contexts=(3.0, 5.0), horizons=(2.0,), stride_frames=1)
    with pytest.raises(DataContractError, match="impossible"):
        next(iter_windows(manifest, impossible, Split.TRAIN))
    short = build_split(
        Registry(dataset=dataset(), trajectories=(record(0, frame_count=1),)),
        SplitConfig(mode="official", seed=1),
    )
    with pytest.raises(DataContractError):
        next(iter_windows(short, grid(), Split.TRAIN))
    invalid_windows: list[dict[str, object]] = [
        {"contexts": []},
        {"contexts": [0]},
        {"contexts": [1.5]},
        {"horizons": [0]},
        {"contexts": [1, 1]},
        {"stride_frames": 0},
        {"contexts": [True]},
    ]
    for changes in invalid_windows:
        with pytest.raises(ValidationError):
            WindowConfig.model_validate(grid().model_dump() | changes)


@pytest.mark.parametrize("unit,factor", [("ps", 1.0), ("ns", 0.001)])
def test_physical_windows(unit: str, factor: float) -> None:
    source = record(
        0,
        time_unit=unit,
        sampling_status="verified",
        frame_interval_ps=80.0,
        duration_ns=0.4,
    )
    manifest = build_split(
        Registry(dataset=dataset(), trajectories=(source,)),
        SplitConfig(mode="official", seed=1),
    )
    config = WindowConfig(
        contexts=(160.0 * factor,),
        horizons=(160.0 * factor,),
        unit=unit,
        stride_frames=1,
    )
    windows = list(iter_windows(manifest, config, Split.TRAIN))
    assert len(windows) == 2
    assert windows[0].context_frames == 3 and windows[0].horizon_frames == 2
    assert windows[0].context_span_ps == 160 and windows[0].horizon_span_ps == 160
    table = to_arrow(
        source,
        dataset(),
        np.arange(6, dtype=np.float64) * 80 * factor,
        np.column_stack([np.arange(6, dtype=np.float64), np.full(6, 7.0)]),
    )
    scaling = ScalingConfig(scope="context-local", feature_ids=("position",))
    assert fit_context_scaler(
        manifest, config, windows[0], scaling, lambda _: table
    ).center == (1.0,)
    altered = table.set_column(0, "time", pa.array(np.arange(6) * 90.0 * factor))
    with pytest.raises(DataContractError, match="sampling interval"):
        fit_context_scaler(manifest, config, windows[0], scaling, lambda _: altered)
    bad = WindowConfig(contexts=(100.0,), horizons=(80.0,), unit="ps", stride_frames=1)
    with pytest.raises(DataContractError, match="aligned"):
        next(iter_windows(manifest, bad, Split.TRAIN))
    with pytest.raises(DataContractError, match="uniform"):
        next(iter_windows(official(), config, Split.TRAIN))
    assumed = record(
        0,
        time_unit="ps",
        sampling_status="assumed",
        sampling_note="Explicit fixture assumption",
        frame_interval_ps=80.0,
        duration_ns=0.4,
    )
    assumed_split = build_split(
        Registry(dataset=dataset(), trajectories=(assumed,)),
        SplitConfig(mode="official", seed=1),
    )
    with pytest.raises(DataContractError, match="verified"):
        next(iter_windows(assumed_split, config, Split.TRAIN))
    opted_in = WindowConfig.model_validate(
        config.model_dump() | {"allow_assumed_time": True}
    )
    assert (
        next(iter_windows(assumed_split, opted_in, Split.TRAIN)).sampling_status
        == "assumed"
    )


def test_forged_window_rejected() -> None:
    manifest = official()
    window = next(iter_windows(manifest, grid(), Split.TRAIN))
    assert (
        validate_window(manifest, grid(), window).trajectory_id == window.trajectory_id
    )
    for changes in (
        {"split": Split.TEST},
        {"split_hash": "bad"},
        {"start": -1},
        {"start": 1},
        {"start": 6},
        {"context_frames": 4},
        {"trajectory_id": "unknown"},
    ):
        with pytest.raises(DataContractError):
            validate_window(manifest, grid(), replace(window, **changes))


def test_training_and_context_scaling_no_future(tmp_path: Path) -> None:
    manifest = official()
    series = tables(manifest)
    loaded: list[str] = []

    def loader(source: TrajectoryManifest) -> pa.Table:
        loaded.append(source.trajectory_id)
        return series[source.trajectory_id]

    config = ScalingConfig(
        scope="training-contexts", feature_ids=("position", "constant")
    )
    fitted = fit_training_scaler(manifest, grid(), config, loader)
    assert loaded == ["trajectory-0", "trajectory-1", "trajectory-2"]
    assert fitted.sample_count == 15
    assert all(region.stop == 5 for region in fitted.fit_regions)
    expected = np.column_stack(
        [
            np.concatenate([np.arange(5) + shift for shift in (0, 10, 20)]),
            np.full(15, 7.0),
        ]
    )
    np.testing.assert_allclose(fitted.center, expected.mean(axis=0))
    np.testing.assert_allclose(fitted.scale, [expected[:, 0].std(), 1.0])
    np.testing.assert_allclose(
        transform(transform(expected, fitted), fitted, inverse=True), expected
    )
    for source in manifest.registry.trajectories:
        values = np.column_stack(
            [
                np.arange(6, dtype=np.float64) + int(source.trajectory_id[-1]) * 10,
                np.full(6, 7.0),
            ]
        )
        if source.split == Split.TRAIN:
            values[-1, 0] = 1e9
        else:
            values[:, 0] = -1e9
        series[source.trajectory_id] = to_arrow(
            source, dataset(), np.arange(6, dtype=np.float64), values
        )
    assert fit_training_scaler(manifest, grid(), config, loader) == fitted
    path = tmp_path / "scaler.json"
    assert write_metadata(path, fitted) == metadata_hash(fitted)
    assert read_metadata(path, ScalerMetadata) == fitted
    context_config = ScalingConfig(scope="context-local", feature_ids=("position",))
    window = next(
        window
        for window in iter_windows(manifest, grid(), Split.VALIDATION)
        if window.context_frames == 3
    )
    local = fit_context_scaler(manifest, grid(), window, context_config, loader)
    assert local.sample_count == 3 and local.center == (-1e9,)
    assert len(local.fit_regions) == 1
    source = next(
        source
        for source in manifest.registry.trajectories
        if source.trajectory_id == window.trajectory_id
    )
    values = np.column_stack([np.full(6, -1e9), np.full(6, 7.0)])
    values[window.target_slice, 0] = 1e12
    series[source.trajectory_id] = to_arrow(
        source, dataset(), np.arange(6, dtype=np.float64), values
    )
    assert fit_context_scaler(manifest, grid(), window, context_config, loader) == local
    # Fitting must not even numerically validate unavailable forecast values.
    values[window.target_slice, 0] = np.nan
    series[source.trajectory_id] = series[source.trajectory_id].set_column(
        1, "position", pa.array(values[:, 0])
    )
    assert fit_context_scaler(manifest, grid(), window, context_config, loader) == local
    values[window.context_slice, 0] = np.nan
    series[source.trajectory_id] = series[source.trajectory_id].set_column(
        1, "position", pa.array(values[:, 0])
    )
    with pytest.raises(DataContractError, match="must be finite"):
        fit_context_scaler(manifest, grid(), window, context_config, loader)


@pytest.mark.parametrize("time", [[0, 0, 2, 3, 4, 5], [0, 1, 4, 5, 6, 7]])
def test_invalid_observed_time(time: list[int]) -> None:
    manifest = official()
    series = tables(manifest)
    source = manifest.registry.trajectories[0]
    series[source.trajectory_id] = series[source.trajectory_id].set_column(
        0, "time", pa.array(time, type=pa.float64())
    )
    with pytest.raises(DataContractError, match="observed"):
        fit_training_scaler(
            manifest,
            grid(),
            ScalingConfig(scope="training-contexts", feature_ids=("position",)),
            lambda record: series[record.trajectory_id],
        )


def test_leakage_guard_detects_deliberately_bad_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = official()
    series = tables(manifest)
    config = ScalingConfig(scope="training-contexts", feature_ids=("position",))
    correct = fit_training_scaler(
        manifest, grid(), config, lambda source: series[source.trajectory_id]
    )
    # Mutation: fit complete trajectories, including their withheld final target.
    monkeypatch.setattr(
        preprocessing,
        "_training_regions",
        lambda *_: tuple(
            FitRegion(trajectory_id=f"trajectory-{index}", start=0, stop=6)
            for index in range(3)
        ),
    )
    leaked = fit_training_scaler(
        manifest, grid(), config, lambda source: series[source.trajectory_id]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(leaked.center, correct.center)


def test_scaling_boundary_failures(tmp_path: Path) -> None:
    manifest = official()
    series = tables(manifest)

    def loader(source: TrajectoryManifest) -> pa.Table:
        return series[source.trajectory_id]

    config = ScalingConfig(scope="training-contexts", feature_ids=("position",))
    fitted = fit_training_scaler(manifest, grid(), config, loader)
    invalid_metadata: list[dict[str, object]] = [
        {"schema_version": 2},
        {"center": ()},
        {"scale": (0.0,)},
        {"sample_count": 1},
        {"config": {"scope": "context-local", "feature_ids": ["position"]}},
        {
            "fit_regions": [
                {"trajectory_id": "trajectory-0", "start": 0, "stop": 5},
                {"trajectory_id": "trajectory-0", "start": 2, "stop": 7},
            ],
            "sample_count": 10,
        },
    ]
    for changes in invalid_metadata:
        with pytest.raises(ValidationError):
            ScalerMetadata.model_validate(fitted.model_dump() | changes)
    with pytest.raises(ValidationError):
        FitRegion(trajectory_id="trajectory-0", start=2, stop=1)
    with pytest.raises(ValidationError):
        ScalingConfig(scope="training-contexts", feature_ids=("position", "position"))
    for values in (np.ones((3, 2)), np.array([[float("nan")]])):
        with pytest.raises(DataContractError):
            transform(values, fitted)
    with pytest.raises(DataContractError, match="overflowed"):
        transform(np.array([[1e308]]), fitted, inverse=True)
    local = ScalingConfig(scope="context-local", feature_ids=("position",))
    with pytest.raises(DataContractError, match="requires scope"):
        fit_training_scaler(manifest, grid(), local, loader)
    with pytest.raises(DataContractError, match="unknown feature"):
        fit_training_scaler(
            manifest,
            grid(),
            ScalingConfig(scope="training-contexts", feature_ids=("unknown",)),
            loader,
        )
    with pytest.raises(DataContractError, match="source metadata"):
        fit_training_scaler(manifest, grid(), config, lambda _: series["trajectory-5"])
    no_train = build_split(
        Registry(dataset=dataset(), trajectories=(record(4),)),
        SplitConfig(mode="official", seed=1),
    )
    with pytest.raises(DataContractError, match="no training"):
        fit_training_scaler(no_train, grid(), config, loader)
    huge = to_arrow(
        record(0), dataset(), np.arange(6, dtype=np.float64), np.full((6, 2), 1e308)
    )
    with pytest.raises(DataContractError, match="moments overflowed"):
        fit_training_scaler(manifest, grid(), config, lambda _: huge)
    with pytest.raises(DataContractError, match="cannot write"):
        write_metadata(tmp_path / "missing" / "scaler.json", fitted)
    with pytest.raises(DataContractError, match="cannot read"):
        read_metadata(tmp_path / "missing", ScalerMetadata)


def test_context_leakage_guard_detects_bad_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = official()
    series = tables(manifest)
    window = next(
        window
        for window in iter_windows(manifest, grid(), Split.TEST)
        if window.context_frames == 3
    )
    config = ScalingConfig(scope="context-local", feature_ids=("position",))

    def loader(source: TrajectoryManifest) -> pa.Table:
        return series[source.trajectory_id]

    correct = fit_context_scaler(manifest, grid(), window, config, loader)
    original = preprocessing._fit

    def future_fit(
        split: SplitManifest,
        windows: WindowConfig,
        selection: ScalingConfig,
        regions: tuple[FitRegion, ...],
        read: preprocessing.SeriesLoader,
    ) -> ScalerMetadata:
        context = regions[0]
        # Mutation: include the forecast region in the scaler's context fit.
        leaked = FitRegion(
            trajectory_id=context.trajectory_id,
            start=context.start,
            stop=window.target_slice.stop,
        )
        return original(split, windows, selection, (leaked,), read)

    monkeypatch.setattr(preprocessing, "_fit", future_fit)
    incorrect = fit_context_scaler(manifest, grid(), window, config, loader)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(incorrect.center, correct.center)


def test_experiment_linkage_and_versions() -> None:
    manifest = official()
    series = tables(manifest)
    fitted = fit_training_scaler(
        manifest,
        grid(),
        ScalingConfig(scope="training-contexts", feature_ids=("position",)),
        lambda source: series[source.trajectory_id],
    )
    experiment = bind_experiment(
        ExperimentConfig(dataset=dataset(), seed=17), manifest, grid(), fitted
    )
    assert experiment.split_hash == metadata_hash(manifest)
    assert experiment.preprocessing_hash == metadata_hash(fitted)
    assert experiment.window_config_hash == metadata_hash(grid())
    changed_dataset = DatasetConfig.model_validate(
        dataset().model_dump() | {"dataset_version": "different"}
    )
    with pytest.raises(DataContractError, match="experiment dataset"):
        bind_experiment(
            ExperimentConfig(dataset=changed_dataset, seed=17), manifest, grid(), fitted
        )
    invalid_scalers: list[dict[str, object]] = [
        {"split_hash": "sha256:" + "0" * 64},
        {"dataset": changed_dataset},
        {
            "fit_regions": [FitRegion(trajectory_id="trajectory-5", start=0, stop=5)],
            "sample_count": 5,
        },
    ]
    for changes in invalid_scalers:
        invalid = ScalerMetadata.model_validate(fitted.model_dump() | changes)
        with pytest.raises(DataContractError):
            validate_scaler(invalid, manifest, grid())
    window = next(
        window
        for window in iter_windows(manifest, grid(), Split.TEST)
        if window.context_frames == 3
    )
    local = fit_context_scaler(
        manifest,
        grid(),
        window,
        ScalingConfig(scope="context-local", feature_ids=("position",)),
        lambda source: series[source.trajectory_id],
    )
    validate_scaler(local, manifest, grid())
    for region in (
        FitRegion(trajectory_id="unknown", start=0, stop=3),
        FitRegion(trajectory_id=window.trajectory_id, start=1, stop=4),
    ):
        invalid = ScalerMetadata.model_validate(
            local.model_dump() | {"fit_regions": [region]}
        )
        with pytest.raises(DataContractError):
            validate_scaler(invalid, manifest, grid())
    for config in (SplitConfig(mode="official", seed=1), grid(), fitted.config):
        with pytest.raises(ValidationError):
            type(config).model_validate(config.model_dump() | {"schema_version": 2})
    with pytest.raises(ValidationError):
        record(0, frame_count=None)


def test_disjoint_training_contexts() -> None:
    manifest = official()
    source = tables(manifest)
    windows = WindowConfig(contexts=(1.0,), horizons=(1.0,), stride_frames=2)
    fitted = fit_training_scaler(
        manifest,
        windows,
        ScalingConfig(scope="training-contexts", feature_ids=("position",)),
        lambda record: source[record.trajectory_id],
    )
    assert fitted.sample_count == 9
    assert [(region.start, region.stop) for region in fitted.fit_regions[:3]] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


def test_replica_protocol_preserves_system_identity_and_dependence() -> None:
    records = tuple(
        record(
            system,
            trajectory_id=f"s{system}-r{replica}",
            replicate_id=str(replica),
            split=None,
        )
        for system in range(2)
        for replica in range(1, 4)
    )
    source = Registry(dataset=dataset(), trajectories=records)
    config = SplitConfig(
        mode="unseen-replica",
        seed=42,
        replica_partitions={"1": Split.TRAIN, "2": Split.TRAIN, "3": Split.TEST},
    )
    split = build_split(source, config)
    assert split.registry.trajectories == records
    for system in range(2):
        selected = [
            a for a in split.assignments if a.trajectory_id.startswith(f"s{system}-")
        ]
        assert len({a.group_id for a in selected}) == 1
        assert {a.split for a in selected} == {Split.TRAIN, Split.TEST}
    with pytest.raises(DataContractError, match="named replicas"):
        build_split(Registry(dataset=dataset(), trajectories=records[:-1]), config)
    with pytest.raises(DataContractError, match="official"):
        build_split(
            Registry(
                dataset=dataset(),
                trajectories=tuple(
                    r.model_copy(update={"split": Split.TRAIN}) for r in records
                ),
            ),
            config,
        )
    with pytest.raises(ValidationError, match="replica partitions"):
        SplitConfig(mode="official", seed=42, replica_partitions={"1": Split.TRAIN})
