"""Known unequal replica weights, paired identity guards and failed calibration."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from md_forecast.core.exceptions import DataContractError
from md_forecast.evaluation.confirmation import (
    calibrated,
    confirmation_decision,
    paired_effect,
    read_confirmation_plan,
    system_metrics,
    upper_below,
)
from md_forecast.evaluation.metrics import BenchmarkConfig


def test_system_metrics_and_paired_decision() -> None:
    """Two dependent replicas count as one group; rooted RMSE stays trajectory-local."""
    table = pa.table(
        {
            "group_id": ["a", "a", "a", "b"],
            "trajectory_id": ["a1", "a1", "a2", "b1"],
            "lead_absolute_error": [0.0, 2.0, 5.0, 9.0],
            "lead_squared_error": [0.0, 4.0, 25.0, 81.0],
            "lead_bias": [0.0, 2.0, -5.0, 9.0],
            "pinball-0.1-0.1": [0.0, 2.0, 5.0, 9.0],
            "pinball-0.5-0.5": [0.0, 2.0, 5.0, 9.0],
            "pinball-0.9-0.9": [0.0, 2.0, 5.0, 9.0],
            "coverage-0.1-0.9": [1.0, 1.0, 0.0, 1.0],
            "width-0.1-0.9": [2.0, 2.0, 4.0, 6.0],
        }
    )
    groups, values = system_metrics(table)
    assert groups == ("a", "b")
    np.testing.assert_allclose(values["mae"], [3.0, 9.0])
    np.testing.assert_allclose(values["rmse"], [(np.sqrt(2) + 5) / 2, 9.0])
    np.testing.assert_allclose(values["coverage-0.1-0.9"], [0.5, 1.0])
    np.testing.assert_array_equal(values["mean-pinball"], values["mae"])
    config = BenchmarkConfig(
        seed=42,
        bootstrap_samples=2000,
        min_groups=5,
        comparison_family_size=126,
        max_prediction_rows=100,
        max_bootstrap_bytes=1024**2,
    )
    effect = paired_effect((groups, values), (groups, values), "mae", config)
    assert effect["difference"] == 0
    assert effect["ci"] == (None, None)
    assert effect["interval_status"] == "insufficient-groups"
    assert not upper_below(effect)
    with pytest.raises(DataContractError, match="groups differ"):
        paired_effect((groups, values), (("b", "a"), values), "mae", config)
    good = {"ci": (-2.0, -1.0)}
    effects = {
        "mae": dict.fromkeys(("persistence", "selected-statistic", "compact"), good),
        "worthwhile": dict.fromkeys(
            ("persistence", "selected-statistic", "compact"), good
        ),
        "pinball": dict.fromkeys(("persistence", "selected-statistic"), good),
        "coverage": {"ci": (0.76, 0.84)},
        "width": {"ci": (-1.0, 0.0)},
    }
    assert confirmation_decision(effects, 0.8, 0.05)["useful_stronger"]
    effects["coverage"] = {"ci": (0.70, 0.74)}
    decision = confirmation_decision(effects, 0.8, 0.05)
    assert decision["worthwhile_point_gain"]
    assert not decision["useful_minimum"]
    assert not calibrated({"ci": (None, None)}, 0.8, 0.05)
    effects["coverage"] = {"ci": (0.76, 0.84)}
    effects["worthwhile"]["compact"] = {"ci": (-1.0, 0.0)}
    assert confirmation_decision(effects, 0.8, 0.05)["useful_minimum"]
    assert not confirmation_decision(effects, 0.8, 0.05)["useful_stronger"]


def test_confirmation_pin_rejects_missing_plan(tmp_path: Path) -> None:
    """A reserved read cannot proceed using an unavailable frozen plan."""
    with pytest.raises(DataContractError):
        read_confirmation_plan(tmp_path / "missing.json")
