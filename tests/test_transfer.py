"""Explicit transfer preserves fitted provenance and validates equivalent domains."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from test_baselines import config, grid, series, source, training
from test_chronos import settings
from test_finetuning import backend, checkpoint, manifest
from test_residuals import fitting

from md_forecast.core.constants import DatasetId, ModelId, Split
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import bind_evaluation_spec, iter_forecasts
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.finetuning import FineTunedChronos2Adapter
from md_forecast.models.learned import NLinearModel, fit_nlinear
from md_forecast.models.residuals import ResidualBaseline, fit_residual_quantiles


def test_explicit_equivalent_domain_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    split = source()
    tables = series(split)

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    features = ("position", "constant")
    batch, _ = next(
        iter_forecasts(split, grid(), Split.VALIDATION, features, loader, batch_size=1)
    )
    target = batch.spec.model_copy(
        update={
            "split_hash": "sha256:" + "f" * 64,
            "dataset": batch.spec.dataset.model_copy(
                update={
                    "dataset_id": DatasetId.MDBIND,
                    "dataset_version": "external-reviewed",
                }
            ),
        }
    )
    altered = replace(
        batch,
        spec=target,
        indices=tuple(replace(i, split_hash=target.split_hash) for i in batch.indices),
    )
    fitted = fit_nlinear(
        split, grid(), features, loader, config(ModelId.NLINEAR), training()
    )
    residual = ResidualBaseline(
        fit_residual_quantiles(
            split,
            grid(),
            features,
            loader,
            StatisticalBaseline(config(ModelId.PERSISTENCE)),
            fitting(),
        )
    )
    for model in (fitted, residual):
        with pytest.raises(ForecastError):
            model.predict(altered)
        adapted = (
            NLinearModel(model.state, evaluation_spec=target)
            if isinstance(model, NLinearModel)
            else ResidualBaseline(model.state, evaluation_spec=target)
        )
        assert adapted.artifact_hash == model.artifact_hash
        assert metadata_hash(adapted.state) == metadata_hash(model.state)
        np.testing.assert_array_equal(adapted.predict(altered), model.predict(batch))
        with pytest.raises(ForecastError):
            adapted.predict(batch)
    backend(monkeypatch)
    frozen = manifest()
    path = tmp_path / "checkpoint"
    checkpoint(path, frozen)
    original = FineTunedChronos2Adapter(settings(), checkpoint_dir=path)
    bound = FineTunedChronos2Adapter(
        settings(), checkpoint_dir=path, evaluation_spec=target
    )
    assert bound.artifact_hash == original.artifact_hash
    np.testing.assert_array_equal(
        bound.forecast(altered).values, original.forecast(batch).values
    )
    with pytest.raises(ForecastError, match="evaluation binding"):
        bound.forecast(batch)
    for changed in (
        target.model_copy(update={"feature_ids": tuple(reversed(features))}),
        target.model_copy(
            update={
                "dataset": target.dataset.model_copy(
                    update={"feature_set_version": "unreviewed"}
                )
            }
        ),
        target.model_copy(
            update={
                "dataset": target.dataset.model_copy(
                    update={
                        "features": (
                            target.dataset.features[0].model_copy(
                                update={"definition": "changed convention"}
                            ),
                            target.dataset.features[1],
                        )
                    }
                )
            }
        ),
        target.model_copy(update={"context_frames": target.context_frames + 1}),
    ):
        with pytest.raises(DataContractError):
            NLinearModel(fitted.state, evaluation_spec=changed)
    flexible = target.model_copy(update={"context_frames": target.context_frames + 1})
    assert bind_evaluation_spec(frozen.spec, flexible, fixed_grid=False) == flexible
