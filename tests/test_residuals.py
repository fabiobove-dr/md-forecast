"""Residual references preserve points, fit TRAIN only, and reject stale state."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError
from test_baselines import config, grid, series, source
from test_benchmark import policy

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import to_arrow
from md_forecast.evaluation.metrics import (
    MetricId,
    group_interval,
    interval_status,
    quantile_losses,
)
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.residuals import (
    ResidualBaseline,
    ResidualConfig,
    ResidualState,
    fit_residual_quantiles,
)


def fitting(**changes: object) -> ResidualConfig:
    return ResidualConfig.model_validate(
        dict(batch_size=2, max_windows=100, max_residual_bytes=1_000_000) | changes
    )


def test_lead_specific_residuals_fit_train_only_and_preserve_points(
    tmp_path: Path,
) -> None:
    split = source()
    tables = series(split)
    for i, record in enumerate(split.registry.trajectories):
        time = np.arange(12, dtype=np.float64)
        values = np.column_stack([time + i * 100, np.full(12, 7.0)])
        tables[record.trajectory_id] = to_arrow(
            record, split.registry.dataset, time, values
        )
    seen = []

    def loader(record: TrajectoryManifest) -> pa.Table:
        assert record.split == Split.TRAIN
        seen.append(record.trajectory_id)
        return tables[record.trajectory_id]

    point = StatisticalBaseline(config(ModelId.PERSISTENCE))
    state = fit_residual_quantiles(
        split, grid(), ("position", "constant"), loader, point, fitting()
    )
    np.testing.assert_array_equal(
        state.offsets, [[[0, 0, 1], [0, 0, 0]], [[0, 0, 2], [0, 0, 0]]]
    )
    assert set(seen) == set(state.training_trajectory_ids)
    assert len(state.training_group_ids) == len(state.training_trajectory_ids)
    path = tmp_path / "state.json"
    write_metadata(path, state)
    model = ResidualBaseline(read_metadata(path, ResidualState))
    assert model.artifact_hash == metadata_hash(state)
    batch, targets = next(
        iter_forecasts(
            split,
            grid(),
            Split.VALIDATION,
            ("position", "constant"),
            lambda r: tables[r.trajectory_id],
            batch_size=2,
            with_targets=True,
        )
    )
    forecast = model.forecast(batch)
    np.testing.assert_array_equal(forecast.median, point.predict(batch))
    np.testing.assert_array_equal(model.predict(batch), forecast.median)
    assert forecast.runtime.peak_reserved_bytes == 0
    assert not forecast.values.flags.writeable
    assert targets is not None
    metrics = quantile_losses(
        forecast.values, targets, forecast.quantile_levels, ((0.1, 0.9),)
    )
    assert metrics[(MetricId.COVERAGE, 0.1, 0.9)].mean() == 1
    assert metrics[(MetricId.WIDTH, 0.1, 0.9)].mean() == 0.75
    # Changing every held-out label does not affect a fitted TRAIN artifact.
    for record in split.registry.trajectories:
        if record.split != Split.TRAIN:
            tables[record.trajectory_id] = to_arrow(
                record,
                split.registry.dataset,
                np.arange(12, dtype=np.float64),
                np.full((12, 2), 9999.0),
            )
    repeated = fit_residual_quantiles(
        split, grid(), ("position", "constant"), loader, point, fitting()
    )
    assert metadata_hash(repeated) == metadata_hash(state)
    altered = replace(
        batch,
        spec=batch.spec.model_copy(update={"split_hash": "sha256:" + "f" * 64}),
        indices=tuple(
            replace(i, split_hash="sha256:" + "f" * 64) for i in batch.indices
        ),
    )
    with pytest.raises(ForecastError, match="source/features/grid"):
        model.forecast(altered)


def test_residual_bounds_fail_before_read_and_reject_bad_state() -> None:
    split = source()
    tables = series(split)
    point = StatisticalBaseline(config(ModelId.PERSISTENCE))

    def forbidden(record: TrajectoryManifest) -> pa.Table:
        raise AssertionError("budget must fail before any read")

    for settings in [fitting(max_windows=1), fitting(max_residual_bytes=1)]:
        with pytest.raises(ForecastError, match="budget"):
            fit_residual_quantiles(
                split, grid(), ("position", "constant"), forbidden, point, settings
            )
    for levels in [(0.9, 0.5, 0.1), (0.1, 0.9)]:
        with pytest.raises(ValidationError):
            fitting(levels=levels)
    state = fit_residual_quantiles(
        split,
        grid(),
        ("position", "constant"),
        lambda r: tables[r.trajectory_id],
        point,
        fitting(),
    )
    for offsets in [
        [],
        [[[0, 1, 2]]],
        [[[0, 0, float("nan")], [0, 0, 0]], [[0, 0, 2], [0, 0, 0]]],
        [[[0, 1, 2], [0, 0, 0]], [[0, 0, 2], [0, 0, 0]]],
        [[[1, 0, 2], [0, 0, 0]], [[0, 0, 2], [0, 0, 0]]],
    ]:
        with pytest.raises(ValidationError):
            ResidualState.model_validate(state.model_dump() | {"offsets": offsets})
    with pytest.raises(ValidationError, match="window budget"):
        ResidualState.model_validate(
            state.model_dump() | {"training_window_count": 101}
        )
    with pytest.raises(ForecastError, match="one explicit"):
        fit_residual_quantiles(
            split,
            grid().model_copy(update={"contexts": (4.0, 6.0)}),
            ("position", "constant"),
            forbidden,
            point,
            fitting(),
        )


def test_residual_numeric_overflow_is_a_domain_error() -> None:
    from md_forecast.models.residuals import _fit_offsets

    with pytest.raises(ForecastError, match="quantile fitting overflowed"):
        _fit_offsets(np.array([[[-1e308]], [[1e308]]]), fitting())
    split = source()
    tables = series(split)
    point = StatisticalBaseline(config(ModelId.PERSISTENCE))
    state = fit_residual_quantiles(
        split,
        grid(),
        ("position", "constant"),
        lambda r: tables[r.trajectory_id],
        point,
        fitting(),
    )
    offsets = np.asarray(state.offsets)
    offsets[:, :, 2] = 1e308
    model = ResidualBaseline(
        state.model_copy(
            update={"offsets": tuple(tuple(tuple(v) for v in lead) for lead in offsets)}
        )
    )
    batch, _ = next(
        iter_forecasts(
            split,
            grid(),
            Split.VALIDATION,
            ("position", "constant"),
            lambda r: tables[r.trajectory_id],
            batch_size=2,
        )
    )
    with pytest.raises(ForecastError, match="forecast overflowed"):
        model.forecast(replace(batch, context=np.full_like(batch.context, 1e308)))


def test_prospective_corrected_tail_resolution_and_small_group_limit() -> None:
    plan = policy(
        bootstrap_samples=60000,
        comparison_family_size=126,
        min_groups=20,
        min_tail_samples=10,
        max_bootstrap_bytes=128 * 1024**2,
    )
    assert interval_status(100, plan, corrected=True) == "descriptive-bootstrap"
    assert interval_status(50, plan, corrected=True) == "descriptive-bootstrap"
    assert interval_status(6, plan, corrected=True) == "insufficient-groups"
    values = np.linspace(-2, -1, 50)
    low, high = group_interval(values, plan, corrected=True)
    assert low is not None and high is not None and low <= high < 0
    assert (
        interval_status(
            100, plan.model_copy(update={"bootstrap_samples": 1000}), corrected=True
        )
        == "insufficient-bootstrap-resolution"
    )
