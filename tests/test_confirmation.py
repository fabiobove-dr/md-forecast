"""Known unequal replica weights, paired identity guards and failed calibration."""

from pathlib import Path
from typing import Any

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
    effects: dict[str, Any] = {
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


def test_frozen_sources_and_confirmation_render(tmp_path: Path) -> None:
    """Tampered frozen inputs block reserved access; synthesis pins block rendering."""
    import json

    from test_baselines import grid, source
    from test_chronos import settings

    from md_forecast.data.artifacts import metadata_hash, write_metadata
    from md_forecast.evaluation.confirmation import ConfirmationPlan
    from md_forecast.evaluation.confirmation_view import render_confirmation
    from md_forecast.evaluation.overview import (
        OverviewConfig,
        OverviewSource,
        OverviewSynthesis,
        file_hash,
    )

    file = tmp_path / "fixed.json"
    file.write_text("fixed training choices")
    digest = "sha256:" + "a" * 64
    bootstrap = BenchmarkConfig(
        seed=42,
        bootstrap_samples=60000,
        min_groups=20,
        comparison_family_size=126,
        max_prediction_rows=1000,
        max_bootstrap_bytes=128 * 1024**2,
    )
    plan = ConfirmationPlan(
        grid=grid(),
        primary=grid(),
        inference=bootstrap,
        descriptive=bootstrap,
        settings=settings(),
        training_dataset=source().registry.dataset,
        native_cohort_hash=digest,
        external_reserve_hash=digest,
        seen_reserve_hash=digest,
        probability_config_hash=digest,
        input_conditions={"position": "joint", "constant": "target-only"},
        fine_checkpoints={"position": "model1", "constant": "model2"},
        compact_states={"c6-h2": "compact.json"},
        residual_states={"c6-h2": {digest: "residual.json"}},
        statistical_choices={"c6-h2": {"position": digest, "constant": digest}},
        source_files={"fixed.json": file_hash(file)},
        expected_groups={"native": 100, "external": 80, "replica": 6},
        worthwhile_reduction=0.1,
        coverage_tolerance=0.05,
        calibration_scope="TRAIN-residuals-only; no confirmation refitting",
        confirmation_state="no reserved outcomes decoded",
        code_commit="a" * 40,
    )
    path = tmp_path / "plan.json"
    write_metadata(path, plan)
    assert read_confirmation_plan(path, tmp_path) == plan
    file.write_text("changed choices")
    with pytest.raises(DataContractError, match="frozen confirmation source changed"):
        read_confirmation_plan(path, tmp_path)
    methods = {
        name: {"metrics": {"mae": 1.0}, "marginal_ci": {"mae": [0.9, 1.1]}}
        for name in (
            "persistence",
            "selected-statistic",
            "compact",
            "zero-shot",
            "fine-tuned",
        )
    }
    effect = {
        "decision": dict.fromkeys(
            (
                "measurable_point_gain",
                "worthwhile_point_gain",
                "calibrated",
                "pinball_gain",
                "width_not_worse",
                "useful_minimum",
                "useful_stronger",
            ),
            False,
        ),
        "coverage": {"mean": 0.7, "ci": [None, None]},
    }
    cell = {
        "position": {
            "methods": methods,
            "corrected_effects": {"zero-shot": effect, "fine-tuned": effect},
        }
    }
    summaries = {
        task: {
            "primary_cell": "c6-h2",
            "cells": {"c6-h2": cell},
            "independent_groups": count,
            "deviations": [],
            "decision": "useful skill not established",
            "corrected_interval_status": "insufficient-groups",
        }
        for task, count in plan.expected_groups.items()
    }
    synthesis = tmp_path / "synthesis.json"
    bundle = tmp_path / "saved"
    bundle.mkdir()
    summary = bundle / "summary.json"
    summary.write_text("{}")
    payload = {
        "frozen_plan": plan.model_dump(mode="json"),
        "config_hash": metadata_hash(plan),
        "primary_family_size": 126,
        "tasks": summaries,
        "source_summaries": {str(summary): file_hash(summary)},
    }
    synthesis.write_text(json.dumps(payload))
    config = OverviewConfig(
        conclusion="synthetic",
        sources=(
            OverviewSource(
                title="synthetic",
                role="regression",
                note="no scientific evidence",
                kind="probability",
                root=bundle,
                expected_hash=file_hash(summary),
                dataset=plan.training_dataset,
            ),
        ),
        confirmation_summary=OverviewSynthesis(
            path=synthesis, expected_hash=file_hash(synthesis)
        ),
    )
    html = render_confirmation(config)
    assert "Not established" in html
    assert "unresolved" in html
    assert "<svg" in html
    summary.write_text("tampered")
    with pytest.raises(DataContractError, match="synthesis source differs"):
        render_confirmation(config)
    synthesis.write_text("tampered")
    with pytest.raises(DataContractError, match="synthesis checksum differs"):
        render_confirmation(config)
