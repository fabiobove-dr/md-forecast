"""Hand-calculated losses, all adapters, group weighting and reproducible bundles."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError
from test_baselines import config, grid, series, source, training
from test_chronos import install_backend, settings

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch, iter_forecasts
from md_forecast.data.preprocessing import ScalingConfig, fit_training_scaler
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import FloatArray, to_arrow
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.evaluation.benchmark import (
    CellManifest,
    GridManifest,
    evaluate_cell,
    evaluate_grid,
)
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    MetricId,
    group_interval,
    interval_status,
    point_losses,
    quantile_losses,
)
from md_forecast.evaluation.report import (
    read_benchmark,
    write_benchmark,
)
from md_forecast.models.base import ForecastModel
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.learned import fit_nlinear


def policy(**changes: object) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        dict(
            seed=42,
            bootstrap_samples=200,
            min_groups=2,
            min_tail_samples=1,
            comparison_family_size=100,
            max_prediction_rows=100000,
            max_bootstrap_bytes=10000000,
        )
        | changes
    )


def setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[
    tuple[ForecastModel, ...],
    CellManifest,
    list[tuple[ForecastBatch, FloatArray | None]],
]:
    split = source()
    tables = series(split)

    def loader(entry: TrajectoryManifest) -> pa.Table:
        return tables[entry.trajectory_id]

    features = ("position", "constant")
    scaler = fit_training_scaler(
        split,
        grid(),
        ScalingConfig(scope="training-contexts", feature_ids=features),
        loader,
    )
    install_backend(monkeypatch)
    models: tuple[ForecastModel, ...] = (
        *(
            StatisticalBaseline(
                config(m, lags=1) if m in (ModelId.AR, ModelId.VAR) else config(m)
            )
            for m in ModelId
            if m not in (ModelId.NLINEAR, ModelId.CHRONOS2)
        ),
        fit_nlinear(
            split, grid(), features, loader, config(ModelId.NLINEAR), training()
        ),
        Chronos2Adapter(settings(), cache_dir=tmp_path),
    )
    pairs = list(
        iter_forecasts(
            split,
            grid(),
            Split.TRAIN,
            features,
            loader,
            batch_size=2,
            with_targets=True,
        )
    )
    manifest = CellManifest(
        spec=pairs[0][0].spec,
        split=split,
        grid=grid(),
        partition=Split.TRAIN,
        config=policy(),
        scaler=scaler,
        models=tuple(m.config for m in models),
        model_artifacts=tuple(m.artifact_hash for m in models),
        code_commit="a" * 40,
        lockfile_hash="sha256:" + "b" * 64,
        hardware="synthetic CPU/fake GPU",
        batch_size=2,
    )
    return models, manifest, pairs


def test_reference_metrics_and_edge_cases() -> None:
    target = np.array([[[0.0], [0.0]]])
    points = np.array([[[3.0], [4.0]]])
    losses = point_losses(points, target, np.array([2.0]))
    assert losses[(MetricId.MAE, None, None)].mean() == 3.5
    assert np.sqrt(losses[(MetricId.RMSE, None, None)].mean()) == np.sqrt(12.5)
    assert losses[(MetricId.SCALED_MAE, None, None)].mean() == 1.75
    quantiles = np.array([[[[-1.0, 0.0, 1.0]], [[1.0, 2.0, 3.0]]]])
    probabilistic = quantile_losses(quantiles, target, (0.1, 0.5, 0.9), ((0.1, 0.9),))
    assert probabilistic[(MetricId.PINBALL, 0.1, 0.1)].mean() == 0.5
    assert probabilistic[(MetricId.PINBALL, 0.5, 0.5)].mean() == 0.5
    assert probabilistic[(MetricId.COVERAGE, 0.1, 0.9)].mean() == 0.5
    assert probabilistic[(MetricId.WIDTH, 0.1, 0.9)].mean() == 2
    assert group_interval(np.array([-2.0, -2.0]), policy()) == (-2, -2)
    assert group_interval(np.array([1.0]), policy()) == (None, None)
    values = np.arange(19, dtype=np.float64)
    lo, hi = group_interval(values, policy())
    clo, chi = group_interval(values, policy(comparison_family_size=2), corrected=True)
    assert clo is not None and chi is not None and lo is not None and hi is not None
    assert clo <= lo <= hi <= chi
    assert group_interval(values, policy()) == (lo, hi)


@pytest.mark.parametrize("failure", ["shape", "scale", "nan", "overflow"])
def test_point_failures(failure: str) -> None:
    points, targets, scale = np.ones((1, 2, 1)), np.zeros((1, 2, 1)), np.ones(1)
    if failure == "shape":
        targets = targets[:, :1]
    elif failure == "scale":
        scale[0] = 0
    elif failure == "nan":
        targets[0, 0, 0] = np.nan
    else:
        points[:] = 1e308
    with pytest.raises(ForecastError):
        point_losses(points, targets, scale)


def test_empty_and_scalar_metric_inputs() -> None:
    for values in (np.array(1.0), np.empty((0, 1, 1))):
        with pytest.raises(ForecastError, match="nonempty"):
            point_losses(values, values, np.ones(1))
        with pytest.raises(ForecastError, match="nonempty"):
            quantile_losses(
                np.ones((*values.shape, 3)), values, (0.1, 0.5, 0.9), ((0.1, 0.9),)
            )


@pytest.mark.parametrize(
    "failure", ["shape", "crossing", "level", "endpoint", "nan", "overflow"]
)
def test_quantile_failures(failure: str) -> None:
    values = np.array([[[[-1.0, 0.0, 1.0]]]])
    targets = np.zeros((1, 1, 1))
    levels, intervals = (0.1, 0.5, 0.9), ((0.1, 0.9),)
    if failure == "shape":
        values = values[..., :1]
    elif failure == "crossing":
        values = values[..., ::-1]
    elif failure == "level":
        levels = (0.5, 0.1, 0.9)
    elif failure == "endpoint":
        intervals = ((0.05, 0.95),)
    elif failure == "nan":
        targets[:] = np.nan
    else:
        values[:] = -1e308
        targets[:] = 1e308
    with pytest.raises(ForecastError):
        quantile_losses(values, targets, levels, intervals)


def test_every_adapter_and_reproducible_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models, manifest, pairs = setup(monkeypatch, tmp_path)
    frozen = GridManifest(cells=(manifest,))
    results = evaluate_grid((models,), (pairs,), frozen)
    result = results[0]
    assert (
        result.window_count,
        result.trajectory_count,
        result.system_count,
        result.group_count,
    ) == (15, 3, 3, 3)
    assert result.predictions.num_rows == 420
    reference = metadata_hash(models[0].config)
    rows = [
        r
        for r in result.metrics.to_pylist()
        if r["level"] == "aggregate"
        and r["feature_id"] == "position"
        and r["step"] == 0
        and r["model_hash"] == reference
    ]
    values = {r["metric"]: r["value"] for r in rows}
    assert values["mae"] == 1.875
    assert values["rmse"] == pytest.approx(np.sqrt(2.5) * 1.25)
    assert {r["metric"] for r in result.metrics.to_pylist()} == set(MetricId)
    constant_ratios = [
        r["ratio_to_persistence"]
        for r in result.comparisons.to_pylist()
        if r["feature_id"] == "constant"
    ]
    assert all(ratio is None for ratio in constant_ratios)
    first = write_benchmark(tmp_path / "first", frozen, results)
    rerun = evaluate_grid((models,), (pairs,), frozen)
    second = write_benchmark(tmp_path / "second", frozen, rerun)
    assert first.artifact_id == second.artifact_id
    assert read_benchmark(tmp_path / "first") == first
    assert "persistence" in (tmp_path / "first/cell-0/horizon-0.svg").read_text()
    assert (tmp_path / "first/cell-0/calibration-0.svg").exists()
    changed = tuple(r.model_copy(update={"seconds": 999.0}) for r in result.runtimes)
    third = write_benchmark(
        tmp_path / "third", frozen, (replace(result, runtimes=changed),)
    )
    assert third.artifact_id == first.artifact_id
    with pytest.raises(DataContractError, match="exists"):
        write_benchmark(tmp_path / "first", frozen, results)
    (tmp_path / "first/cell-0/predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(DataContractError, match="checksum"):
        read_benchmark(tmp_path / "first")


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "partition",
        "targets",
        "model",
        "weights",
        "rows",
        "batch",
    ],
)
def test_evaluation_rejects_protocol_drift(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models, manifest, pairs = setup(monkeypatch, tmp_path)
    if failure == "missing":
        pairs = pairs[:-1]
    elif failure == "duplicate":
        pairs = pairs + pairs[:1]
    elif failure == "partition":
        manifest = manifest.model_copy(update={"partition": Split.VALIDATION})
    elif failure == "targets":
        pairs = [(pairs[0][0], None)]
    elif failure == "model":
        models = models[::-1]
    elif failure == "weights":
        manifest = manifest.model_copy(
            update={"model_artifacts": ("sha256:" + "c" * 64,) * len(models)}
        )
    elif failure == "rows":
        manifest = manifest.model_copy(update={"config": policy(max_prediction_rows=1)})
    else:
        manifest = manifest.model_copy(update={"batch_size": 1})
    with pytest.raises(ForecastError):
        evaluate_cell(models, pairs, manifest)


def test_manifest_and_bootstrap_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models, manifest, pairs = setup(monkeypatch, tmp_path)
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(
            policy().model_dump() | {"intervals": [[0.9, 0.1]]}
        )
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(
            policy().model_dump() | {"intervals": [[0.1, 0.9], [0.1, 0.9]]}
        )
    with pytest.raises(ValidationError):
        CellManifest.model_validate(
            manifest.model_dump()
            | {"config": policy(comparison_family_size=1).model_dump()}
        )
    with pytest.raises(ValidationError):
        GridManifest(cells=(manifest, manifest))
    different = manifest.model_copy(update={"partition": Split.TEST})
    with pytest.raises(ValidationError):
        GridManifest(cells=(manifest, different))
    with pytest.raises(ForecastError):
        evaluate_grid((), (), GridManifest(cells=(manifest,)))
    with pytest.raises(ForecastError):
        group_interval(np.arange(5, dtype=np.float64), policy(max_bootstrap_bytes=1))
    with pytest.raises(ForecastError):
        group_interval(np.array([np.nan]), policy())
    with pytest.raises(DataContractError):
        write_benchmark(tmp_path / "partial", GridManifest(cells=(manifest,)), ())
    result = evaluate_cell(models, pairs, manifest)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("md_forecast.evaluation.report.pq.write_table", fail)
    with pytest.raises(DataContractError):
        write_benchmark(tmp_path / "failed", GridManifest(cells=(manifest,)), (result,))
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".benchmark-*"))


def test_statistical_resolution_and_scaling_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, manifest, _ = setup(monkeypatch, tmp_path)
    assert interval_status(1, policy()) == "insufficient-groups"
    assert (
        interval_status(19, policy(), corrected=True)
        == "insufficient-bootstrap-resolution"
    )
    assert group_interval(
        np.arange(19, dtype=np.float64), policy(), corrected=True
    ) == (None, None)
    payload = manifest.model_dump()
    with pytest.raises(ValidationError):
        CellManifest.model_validate(payload | {"model_artifacts": []})
    with pytest.raises(ValidationError):
        CellManifest.model_validate(
            payload
            | {
                "models": [manifest.models[0], manifest.models[0]],
                "model_artifacts": manifest.model_artifacts[:2],
            }
        )
    wrong = manifest.scaler.model_copy(
        update={"window_config_hash": "sha256:" + "c" * 64}
    )
    with pytest.raises(ValidationError):
        CellManifest.model_validate(payload | {"scaler": wrong})
    wrong_spec = manifest.spec.model_copy(update={"split_hash": "sha256:" + "c" * 64})
    with pytest.raises(ValidationError):
        CellManifest.model_validate(payload | {"spec": wrong_spec})
    regions = tuple(
        region.model_copy(update={"trajectory_id": "trajectory-5"})
        for region in manifest.scaler.fit_regions
    )
    wrong = manifest.scaler.model_copy(
        update={
            "fit_regions": regions[:1],
            "sample_count": regions[0].stop - regions[0].start,
        }
    )
    with pytest.raises(ValidationError, match="held-out"):
        CellManifest.model_validate(payload | {"scaler": wrong})


def test_replica_grouping_not_window_independence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models, original, _ = setup(monkeypatch, tmp_path)
    records = list(original.split.registry.trajectories)
    records[1] = records[1].model_copy(
        update={
            "system_id": records[0].system_id,
            "split_group_id": records[0].split_group_id,
        }
    )
    split = build_split(
        Registry(dataset=original.spec.dataset, trajectories=tuple(records)),
        SplitConfig(mode="official", seed=42),
    )
    tables = {}
    for index, record in enumerate(records):
        time = np.arange(12, dtype=np.float64)
        tables[record.trajectory_id] = to_arrow(
            record,
            split.registry.dataset,
            time,
            np.column_stack((index * 100 + time * (1 + index / 4), np.full(12, 7.0))),
        )

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    pairs = list(
        iter_forecasts(
            split,
            grid(),
            Split.TRAIN,
            original.spec.feature_ids,
            loader,
            batch_size=2,
            with_targets=True,
        )
    )
    scaler = fit_training_scaler(
        split,
        grid(),
        ScalingConfig(scope="training-contexts", feature_ids=original.spec.feature_ids),
        loader,
    )
    models = (models[0],)
    manifest = CellManifest.model_validate(
        original.model_dump()
        | dict(
            spec=pairs[0][0].spec,
            split=split,
            scaler=scaler,
            models=[models[0].config],
            model_artifacts=[models[0].artifact_hash],
        )
    )
    result = evaluate_cell(models, pairs, manifest)
    assert (
        result.window_count == 15
        and result.trajectory_count == 3
        and result.group_count == 2
    )
    row = next(
        r
        for r in result.metrics.to_pylist()
        if r["level"] == "aggregate"
        and r["metric"] == "mae"
        and r["feature_id"] == "position"
        and r["step"] == 0
    )
    assert row["value"] == 1.96875
    assert {r["independent_groups"] for r in result.comparisons.to_pylist()} == {2}


def test_explicit_first_window_selector_and_point_only_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models, original, pairs = setup(monkeypatch, tmp_path)
    models = models[:1]
    manifest = CellManifest.model_validate(
        original.model_dump()
        | dict(
            models=[models[0].config],
            model_artifacts=[models[0].artifact_hash],
            window_selection="first-per-trajectory",
        )
    )
    selected = []
    for batch, targets in pairs:
        assert targets is not None
        if batch.indices[0].start == 0:
            selected.append(
                (
                    ForecastBatch(batch.spec, batch.indices[:1], batch.context[:1]),
                    targets[:1],
                )
            )
    result = evaluate_cell(models, selected, manifest)
    assert result.window_count == result.trajectory_count == 3
    assert set(result.predictions["start"].to_pylist()) == {0}
    frozen = GridManifest(cells=(manifest,))
    write_benchmark(tmp_path / "point", frozen, (result,))
    assert not (tmp_path / "point/cell-0/calibration-0.svg").exists()


def test_overflow_and_incompatible_probabilistic_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models, manifest, pairs = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(models[-1], "forecast", lambda batch: None)
    with pytest.raises(ForecastError, match="canonical output"):
        evaluate_cell(models, pairs, manifest)

    def huge(self: StatisticalBaseline, batch: ForecastBatch) -> FloatArray:
        return np.full(
            (
                len(batch.indices),
                batch.spec.horizon_frames,
                len(batch.spec.feature_ids),
            ),
            1e154,
        )

    monkeypatch.setattr(StatisticalBaseline, "predict", huge)
    with pytest.raises(ForecastError, match="aggregation overflowed"):
        evaluate_cell(models, pairs, manifest)


def test_external_scaler_requires_audited_source_and_equivalent_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from md_forecast.core.constants import DatasetId

    _, original, _ = setup(monkeypatch, tmp_path)
    dataset = original.spec.dataset.model_copy(update={"dataset_id": DatasetId.MDBIND})
    records = tuple(
        r.model_copy(
            update={
                "dataset_id": DatasetId.MDBIND,
                "trajectory_id": "external-" + r.trajectory_id,
                "split": Split.TEST,
            }
        )
        for r in original.split.registry.trajectories
    )
    split = build_split(
        Registry(dataset=dataset, trajectories=records),
        SplitConfig(mode="official", seed=42),
    )
    spec = original.spec.model_copy(
        update={"dataset": dataset, "split_hash": metadata_hash(split)}
    )
    payload = original.model_dump() | {
        "spec": spec,
        "split": split,
        "partition": Split.TEST,
    }
    with pytest.raises(ValidationError, match="scaling must match"):
        CellManifest.model_validate(payload)
    external = CellManifest.model_validate(
        payload | {"scaler_source_split": original.split}
    )
    assert external.scaler == original.scaler
    bad = dataset.model_copy(
        update={
            "features": tuple(
                f.model_copy(update={"definition": "different physical quantity"})
                for f in dataset.features
            )
        }
    )
    bad_registry = Registry(dataset=bad, trajectories=records)
    bad_split = build_split(bad_registry, SplitConfig(mode="official", seed=42))
    with pytest.raises(ValidationError, match="equivalent feature"):
        CellManifest.model_validate(
            payload
            | {
                "spec": spec.model_copy(
                    update={"dataset": bad, "split_hash": metadata_hash(bad_split)}
                ),
                "split": bad_split,
                "scaler_source_split": original.split,
            }
        )
    bad_scaler = original.scaler.model_copy(
        update={
            "fit_regions": tuple(
                r.model_copy(
                    update={
                        "trajectory_id": original.split.assignments[-1].trajectory_id
                    }
                )
                for r in original.scaler.fit_regions
            )
        }
    )
    with pytest.raises(ValidationError):
        CellManifest.model_validate(
            payload | {"scaler_source_split": original.split, "scaler": bad_scaler}
        )
