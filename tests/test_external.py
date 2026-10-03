"""Validation-only baseline selection and paired complex-level external effects."""

import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError
from test_baselines import series
from test_benchmark import setup

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.evaluation.benchmark import CellManifest, evaluate_cell
from md_forecast.evaluation.external import (
    ExternalConfig,
    select_statistics,
    selected_effects,
)


def test_frozen_configuration() -> None:
    config = ExternalConfig.model_validate(
        tomllib.loads(Path("configs/benchmarks/mdbind-common.toml").read_text())
    )
    assert config.replica_split.replica_partitions is not None
    assert config.replica_split.replica_partitions["10"] == Split.TEST
    assert config.calibration_tolerance == 0.1
    with pytest.raises(ValidationError, match="70/30"):
        ExternalConfig.model_validate(
            config.model_dump()
            | {"misato_split": {"mode": "grouped", "seed": 42, "ratios": (0.8, 0.2, 0)}}
        )


def test_selection_and_paired_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models, manifest, _ = setup(monkeypatch, tmp_path)
    manifest = CellManifest.model_validate(
        manifest.model_dump() | {"partition": Split.VALIDATION}
    )
    tables = series(manifest.split)

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    result = evaluate_cell(
        models,
        iter_forecasts(
            manifest.split,
            manifest.grid,
            Split.VALIDATION,
            manifest.spec.feature_ids,
            loader,
            batch_size=2,
            with_targets=True,
        ),
        manifest,
    )
    selection = select_statistics(result, "sha256:" + "a" * 64)
    # Linear synthetic channel is exactly extrapolated by the linear baseline.
    chosen = [c for c in selection.choices if c.feature_id == "position"]
    linear = next(m for m in models if m.config.model_id == ModelId.LINEAR)
    assert {c.model_hash for c in chosen} == {metadata_hash(linear.config)}
    effects = selected_effects(result, selection, metadata_hash(linear.config))
    position = [r for r in effects.to_pylist() if r["feature_id"] == "position"]
    assert all(r["independent_groups"] == 2 for r in position)
    np.testing.assert_allclose([r["mae_difference"] for r in position], 0, atol=1e-12)
    with pytest.raises(ForecastError, match="not paired"):
        selected_effects(result, selection, "sha256:" + "b" * 64)
    with pytest.raises(ForecastError, match="grid"):
        selected_effects(
            result,
            selection.model_copy(update={"choices": selection.choices[:-1]}),
            metadata_hash(linear.config),
        )
    with pytest.raises(ForecastError, match="VALIDATION"):
        select_statistics(
            result.__class__(
                **(
                    result.__dict__
                    | {
                        "manifest": manifest.model_copy(
                            update={"partition": Split.TEST}
                        )
                    }
                )
            ),
            "sha256:" + "a" * 64,
        )
