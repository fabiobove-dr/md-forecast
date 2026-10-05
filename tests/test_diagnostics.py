"""Hand-calculated spread/error behavior and dependent-replica uncertainty."""

import numpy as np
import pytest
from test_baselines import pairs
from test_benchmark import policy

from md_forecast.analysis.diagnostics import dependence_summary, forecast_diagnostics
from md_forecast.core.exceptions import DataContractError, ForecastError


def test_spread_is_descriptive_and_constants_are_undefined() -> None:
    batch, targets = pairs()[0]
    assert targets is not None
    points = np.repeat(batch.context[:, -1:, :], 2, axis=1)
    rows = forecast_diagnostics(batch, points, targets)
    np.testing.assert_allclose(rows["spread_ratio"][:, 0], 0)
    assert np.isnan(rows["spread_ratio"][:, 1]).all()
    np.testing.assert_allclose(rows["mae"][:, 0], [1.5, 1.5])
    np.testing.assert_allclose(rows["mean_bias"][:, 0], [-1.5, -1.5])
    np.testing.assert_allclose(rows["lead_absolute_error"][0, :, 0], [1, 2])
    np.testing.assert_allclose(rows["rmse"][0, 0], np.sqrt(2.5))
    np.testing.assert_allclose(rows["change_from_origin_mae"], rows["mae"])
    perfect = forecast_diagnostics(batch, targets, targets)
    np.testing.assert_allclose(perfect["spread_ratio"][:, 0], 1)
    np.testing.assert_allclose(perfect["mae"], 0)
    np.testing.assert_allclose(
        perfect["predicted_change_from_origin"], perfect["actual_change_from_origin"]
    )


@pytest.mark.parametrize("problem", ["shape", "nan", "dtype", "overflow"])
def test_invalid_diagnostic_inputs(problem: str) -> None:
    batch, targets = pairs()[0]
    assert targets is not None
    points = targets.copy()
    if problem == "shape":
        points = points[:, :1]
    elif problem == "nan":
        points[0, 0, 0] = np.nan
    elif problem == "dtype":
        points = points.astype(np.float32)
    else:
        points[:] = 1e308
    with pytest.raises((DataContractError, ForecastError)):
        forecast_diagnostics(batch, points, targets)


def test_dependence_counts_systems_and_replica_means() -> None:
    alternating = np.array([0.0, 1.0] * 20)
    groups: dict[str, tuple[np.ndarray, ...]] = {
        str(i): (alternating, alternating.copy()) for i in range(6)
    }
    result = dependence_summary(groups, 10, policy(comparison_family_size=1))
    assert result.groups == 6 and result.trajectories == 12
    assert result.restricted_decay_mean_frames == 1
    assert result.restricted_decay_interval == (1, 1)
    assert result.positive_integrated_acf_mean_frames == 0.5
    assert result.censored_fraction == 0
    # Replicating one dependent trajectory never creates independent samples.
    groups["0"] = (alternating,) * 30
    repeated = dependence_summary(groups, 10, policy(comparison_family_size=1))
    assert repeated.groups == 6
    np.testing.assert_allclose(
        np.asarray(repeated.mean_acf), np.asarray(result.mean_acf)
    )
    np.testing.assert_allclose(
        np.asarray(repeated.acf_lower, dtype=float),
        np.asarray(result.acf_lower, dtype=float),
    )


def test_constants_and_censoring_remain_explicit() -> None:
    config = policy(comparison_family_size=1)
    constants = dependence_summary({"one": (np.ones(40),)}, 10, config)
    assert constants.groups == 0 and constants.constant_trajectories == 1
    assert constants.mean_acf is None and constants.censored_fraction is None
    slow = dependence_summary({"one": (np.arange(100, dtype=np.float64),)}, 2, config)
    assert slow.restricted_decay_mean_frames == 2
    assert slow.censored_fraction == 1
    assert slow.restricted_decay_interval == (None, None)


@pytest.mark.parametrize(
    "groups,lag",
    [
        ({}, 2),
        ({"a": ()}, 2),
        ({"a": (np.ones(4),)}, 0),
        ({"a": (np.ones(4),)}, 4),
        ({"a": (np.array([0.0, np.nan, 1.0]),)}, 2),
        ({"a": (np.ones((4, 2)),)}, 2),
    ],
)
def test_invalid_dependence_inputs(
    groups: dict[str, tuple[np.ndarray, ...]], lag: int
) -> None:
    with pytest.raises(DataContractError):
        dependence_summary(groups, lag, policy())


def test_dependence_overflow_is_actionable() -> None:
    with pytest.raises(ForecastError, match="overflow"):
        dependence_summary({"a": (np.array([1e308, -1e308, 1e308]),)}, 1, policy())
