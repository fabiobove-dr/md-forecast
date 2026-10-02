"""Exact predictions, six-model integration, and train/validation leakage guards."""

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError
from test_splits_windows import dataset, record

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import ForecastBatch, ForecastSpec, iter_forecasts
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import FloatArray, to_arrow
from md_forecast.data.splits import SplitConfig, SplitManifest, build_split
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.baselines import BaselineEvaluation, evaluate_baselines
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline, ridge_fit
from md_forecast.models.learned import (
    NLinearModel,
    NLinearSelection,
    NLinearState,
    TrainingConfig,
    fit_nlinear,
    select_nlinear,
)


def source() -> SplitManifest:
    registry = Registry(
        dataset=dataset(),
        trajectories=tuple(record(index, frame_count=12) for index in range(6)),
    )
    return build_split(registry, SplitConfig(mode="official", seed=42))


def test_versioned_baseline_configuration() -> None:
    settings = tomllib.loads(
        Path("configs/models/baselines.toml").read_text(encoding="utf-8")
    )
    models = tuple(ModelConfig.model_validate(item) for item in settings["models"])
    assert {model.model_id for model in models} == set(ModelId) - {ModelId.CHRONOS2}
    assert TrainingConfig.model_validate(settings["training"]).max_windows == 10000


def grid() -> WindowConfig:
    return WindowConfig(contexts=(6.0,), horizons=(2.0,), stride_frames=1)


def series(split: SplitManifest) -> dict[str, pa.Table]:
    time = np.arange(12, dtype=np.float64)
    result = {}
    for index, entry in enumerate(split.registry.trajectories):
        values = np.column_stack(
            [index * 100 + time * (1 + index / 4), np.full(12, 7.0)]
        )
        result[entry.trajectory_id] = to_arrow(entry, dataset(), time, values)
    return result


def config(model: ModelId, **changes: object) -> ModelConfig:
    return ModelConfig.model_validate({"model_id": model, "seed": 42} | changes)


def training(**changes: object) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "seed": 42,
            "batch_size": 2,
            "max_windows": 100,
            "max_design_bytes": 1024 * 1024,
        }
        | changes
    )


def pairs(
    partition: Split = Split.TRAIN,
) -> list[tuple[ForecastBatch, FloatArray | None]]:
    split = source()
    arrays = series(split)
    return list(
        iter_forecasts(
            split,
            grid(),
            partition,
            ("position", "constant"),
            lambda entry: arrays[entry.trajectory_id],
            batch_size=2,
            with_targets=True,
        )
    )


@pytest.mark.parametrize(
    "model,expected",
    [
        (ModelId.PERSISTENCE, [5.0, 5.0]),
        (ModelId.CONTEXT_MEAN, [2.5, 2.5]),
        (ModelId.LINEAR, [6.0, 7.0]),
        (ModelId.AR, [6.0, 7.0]),
        (ModelId.VAR, [6.0, 7.0]),
    ],
)
def test_exact_statistical_predictions(model: ModelId, expected: list[float]) -> None:
    batch, _ = pairs()[0]
    settings = (
        config(model, lags=1) if model in (ModelId.AR, ModelId.VAR) else config(model)
    )
    prediction = StatisticalBaseline(settings).predict(batch)
    np.testing.assert_allclose(prediction[0, :, 0], expected, atol=1e-10)
    np.testing.assert_allclose(prediction[:, :, 1], 7.0, atol=1e-10)
    assert prediction.shape == (2, 2, 2)
    assert not batch.context.flags.writeable and not hasattr(batch, "targets")
    np.testing.assert_array_equal(batch.context[0, :, 0], np.arange(6))


def test_all_adapters_common_evaluator_and_selection(tmp_path: Path) -> None:
    split = source()
    arrays = series(split)
    opened: list[Split | None] = []

    def loader(entry: TrajectoryManifest) -> pa.Table:
        assert entry.split != Split.TEST
        opened.append(entry.split)
        return arrays[entry.trajectory_id]

    candidates = (
        config(ModelId.NLINEAR, ridge=100.0),
        config(ModelId.NLINEAR, ridge=0.0),
    )
    learned, selection = select_nlinear(
        split, grid(), ("position", "constant"), loader, candidates, training()
    )
    assert learned.config.ridge == 0
    assert selection.selected_config_hash == metadata_hash(candidates[1])
    assert (
        selection.normalized_validation_mae[1] < selection.normalized_validation_mae[0]
    )
    assert (
        Split.TEST not in opened
        and Split.TRAIN in opened
        and Split.VALIDATION in opened
    )
    assert learned.state.training_trajectory_ids == (
        "trajectory-0",
        "trajectory-1",
        "trajectory-2",
    )
    assert learned.state.training_window_count == 15
    assert learned.state.scaler.sample_count == 30
    models: tuple[ForecastModel, ...] = tuple(
        StatisticalBaseline(config(model, lags=1))
        if model in (ModelId.AR, ModelId.VAR)
        else StatisticalBaseline(config(model))
        for model in ModelId
        if model not in (ModelId.NLINEAR, ModelId.CHRONOS2)
    ) + (learned,)
    result = evaluate_baselines(models, pairs(Split.VALIDATION))
    assert len(result.scores) == 6 and result.partition == Split.VALIDATION
    assert result.reference_config_hash == metadata_hash(config(ModelId.PERSISTENCE))
    for score in result.scores:
        assert len(score.groups) == 2
        assert all(
            group.trajectory_count == 1 and group.window_count == 5
            for group in score.groups
        )
        if score.config.model_id not in (ModelId.PERSISTENCE, ModelId.CONTEXT_MEAN):
            assert all(max(group.mae) < 1e-9 for group in score.groups)
    state_path = tmp_path / "model.json"
    write_metadata(state_path, learned.state)
    restored = NLinearModel(read_metadata(state_path, NLinearState))
    batch, labels = pairs(Split.VALIDATION)[0]
    assert labels is not None
    np.testing.assert_allclose(restored.predict(batch), labels, atol=1e-9)
    shifted = ForecastBatch(batch.spec, batch.indices, batch.context + 10000.0)
    np.testing.assert_allclose(
        restored.predict(shifted), restored.predict(batch) + 10000.0, atol=1e-9
    )


def test_learned_fit_does_not_load_or_fit_heldout() -> None:
    split = source()
    arrays = series(split)

    def loader(entry: TrajectoryManifest) -> pa.Table:
        assert entry.split == Split.TRAIN
        return arrays[entry.trajectory_id]

    fitted = fit_nlinear(
        split,
        grid(),
        ("position", "constant"),
        loader,
        config(ModelId.NLINEAR),
        training(),
    )
    for entry in split.registry.trajectories:
        if entry.split != Split.TRAIN:
            arrays[entry.trajectory_id] = arrays[entry.trajectory_id].set_column(
                1, "position", pa.array(np.full(12, 1e20))
            )
    assert (
        fit_nlinear(
            split,
            grid(),
            ("position", "constant"),
            loader,
            config(ModelId.NLINEAR),
            training(),
        ).state
        == fitted.state
    )
    entry = split.registry.trajectories[0]
    values = arrays[entry.trajectory_id]["position"].to_numpy().copy()
    values[-2:] = 1e6
    arrays[entry.trajectory_id] = arrays[entry.trajectory_id].set_column(
        1, "position", pa.array(values)
    )
    changed = fit_nlinear(
        split,
        grid(),
        ("position", "constant"),
        loader,
        config(ModelId.NLINEAR),
        training(),
    )
    assert changed.state.scaler == fitted.state.scaler
    assert changed.state.weights != fitted.state.weights


def test_context_only_never_inspects_targets() -> None:
    split = source()
    arrays = series(split)
    exact = WindowConfig(contexts=(10.0,), horizons=(2.0,), stride_frames=1)
    for entry in split.registry.trajectories:
        values = arrays[entry.trajectory_id]["position"].to_numpy().copy()
        values[-2:] = np.nan
        arrays[entry.trajectory_id] = arrays[entry.trajectory_id].set_column(
            1, "position", pa.array(values)
        )
    contexts = list(
        iter_forecasts(
            split,
            exact,
            Split.TRAIN,
            ("position",),
            lambda entry: arrays[entry.trajectory_id],
            batch_size=2,
        )
    )
    assert len(contexts) == 3 and all(labels is None for _, labels in contexts)
    for batch, _ in contexts:
        assert np.isfinite(
            StatisticalBaseline(config(ModelId.PERSISTENCE)).predict(batch)
        ).all()
    with pytest.raises(DataContractError, match="finite"):
        list(
            iter_forecasts(
                split,
                exact,
                Split.TRAIN,
                ("position",),
                lambda entry: arrays[entry.trajectory_id],
                batch_size=2,
                with_targets=True,
            )
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": ModelId.AR},
        {"model_id": ModelId.VAR},
        {"model_id": ModelId.LINEAR, "lags": 1},
        {"model_id": ModelId.PERSISTENCE, "ridge": 1},
        {"ridge": -1},
        {"seed": True},
    ],
)
def test_invalid_configs(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"model_id": ModelId.NLINEAR, "seed": 42} | changes)


def test_failure_modes_and_resource_preflight() -> None:
    split = source()

    def never(_: TrajectoryManifest) -> pa.Table:
        pytest.fail("must reject before numerical reads")

    for budget in (training(max_windows=1), training(max_design_bytes=1)):
        with pytest.raises(ForecastError, match="budget"):
            fit_nlinear(
                split,
                grid(),
                ("position", "constant"),
                never,
                config(ModelId.NLINEAR),
                budget,
            )
    with pytest.raises(ForecastError, match="one explicit"):
        fit_nlinear(
            split,
            WindowConfig(contexts=(1.0, 2.0), horizons=(1.0,), stride_frames=1),
            ("position",),
            never,
            config(ModelId.NLINEAR),
            training(),
        )
    no_validation = build_split(
        Registry(
            dataset=dataset(),
            trajectories=(record(0, frame_count=12), record(5, frame_count=12)),
        ),
        SplitConfig(mode="official", seed=42),
    )
    with pytest.raises(ForecastError, match="validation partition is empty"):
        select_nlinear(
            no_validation,
            grid(),
            ("position",),
            never,
            (config(ModelId.NLINEAR),),
            training(),
        )
    for candidates in (
        (),
        (config(ModelId.NLINEAR),) * 2,
        (config(ModelId.PERSISTENCE),),
    ):
        with pytest.raises(ForecastError):
            select_nlinear(split, grid(), ("position",), never, candidates, training())
    batch, labels = pairs()[0]
    with pytest.raises(ForecastError, match="trained weights"):
        StatisticalBaseline(config(ModelId.NLINEAR))
    with pytest.raises(ForecastError, match="lagged observations"):
        StatisticalBaseline(config(ModelId.VAR, lags=3)).predict(batch)
    persistence = StatisticalBaseline(config(ModelId.PERSISTENCE))
    with pytest.raises(ForecastError, match="persistence reference"):
        evaluate_baselines(
            (StatisticalBaseline(config(ModelId.CONTEXT_MEAN)),), [(batch, labels)]
        )
    with pytest.raises(ForecastError, match="duplicate"):
        evaluate_baselines((persistence, persistence), [(batch, labels)])
    with pytest.raises(ForecastError, match="nonempty"):
        evaluate_baselines((persistence,), [])
    with pytest.raises(ForecastError, match="separate target"):
        evaluate_baselines((persistence,), [(batch, None)])
    with pytest.raises(ForecastError, match="mix"):
        evaluate_baselines(
            (persistence,), [(batch, labels), pairs(Split.VALIDATION)[0]]
        )
    with pytest.raises(DataContractError, match="protocol"):
        ForecastBatch(
            batch.spec,
            (replace(batch.indices[0], horizon_frames=99),),
            batch.context[:1],
        )
    with pytest.raises(DataContractError, match="windows"):
        ForecastBatch(batch.spec, (), batch.context)
    with pytest.raises(DataContractError, match="float64"):
        ForecastBatch(batch.spec, batch.indices, batch.context.astype(np.float32))
    with pytest.raises(ValidationError, match="unknown"):
        ForecastSpec.model_validate(
            batch.spec.model_dump() | {"feature_ids": ["unknown"]}
        )
    with pytest.raises(ValidationError, match="duplicate"):
        ForecastSpec.model_validate(
            batch.spec.model_dump() | {"feature_ids": ["position", "position"]}
        )


def test_solver_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("failed SVD")

    monkeypatch.setattr(np.linalg, "lstsq", fail)
    with pytest.raises(ForecastError, match="did not converge") as error:
        ridge_fit(np.ones((3, 2)), np.ones((3, 1)), 1.0)
    assert isinstance(error.value.__cause__, np.linalg.LinAlgError)


def fit_model() -> NLinearModel:
    split = source()
    arrays = series(split)
    return fit_nlinear(
        split,
        grid(),
        ("position", "constant"),
        lambda entry: arrays[entry.trajectory_id],
        config(ModelId.NLINEAR),
        training(),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"weights": []},
        {"training_trajectory_ids": ["trajectory-5"]},
        {"training_window_count": 1000},
        {"preprocessing_hash": "sha256:" + "f" * 64},
    ],
)
def test_invalid_fitted_state(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NLinearState.model_validate(fit_model().state.model_dump() | changes)


def test_source_order_and_prediction_protocol_guards() -> None:
    model = fit_model()
    payload = model.state.model_dump(mode="json")
    payload["spec"]["feature_ids"].reverse()
    with pytest.raises(ValidationError, match="scaler"):
        NLinearState.model_validate(payload)
    batch, _ = pairs()[0]
    changed_spec = ForecastSpec.model_validate(
        batch.spec.model_dump() | {"feature_ids": ["constant", "position"]}
    )
    with pytest.raises(ForecastError, match="differs from fitted"):
        model.predict(ForecastBatch(changed_spec, batch.indices, batch.context))
    split = source()
    arrays = series(split)
    with pytest.raises(DataContractError, match="batch size"):
        next(
            iter_forecasts(
                split,
                grid(),
                Split.TRAIN,
                ("position",),
                lambda _: pytest.fail("must not load"),
                batch_size=0,
            )
        )
    with pytest.raises(DataContractError, match="canonical source"):
        next(
            iter_forecasts(
                split,
                grid(),
                Split.TRAIN,
                ("position",),
                lambda _: arrays["trajectory-5"],
                batch_size=2,
            )
        )
    poisoned = batch.context.copy()
    poisoned[0, 0, 0] = np.nan
    with pytest.raises(DataContractError, match="finite"):
        ForecastBatch(batch.spec, batch.indices, poisoned)
    tiny_grid = WindowConfig(contexts=(1.0,), horizons=(1.0,), stride_frames=1)
    tiny, _ = next(
        iter_forecasts(
            split,
            tiny_grid,
            Split.TRAIN,
            ("position",),
            lambda entry: arrays[entry.trajectory_id],
            batch_size=1,
        )
    )
    with pytest.raises(ForecastError, match="two observed"):
        StatisticalBaseline(config(ModelId.LINEAR)).predict(tiny)


def test_var_really_uses_cross_channel_history() -> None:
    split = source()
    arrays = series(split)
    values = np.zeros((12, 2), dtype=np.float64)
    values[0] = [2.0, -1.0]
    transition = np.array([[0.6, 0.3], [-0.2, 0.5]])
    for step in range(1, 12):
        values[step] = transition @ values[step - 1] + [1.0, -0.5]
    entry = split.registry.trajectories[0]
    arrays[entry.trajectory_id] = to_arrow(
        entry, dataset(), np.arange(12, dtype=np.float64), values
    )
    batch, targets = next(
        iter_forecasts(
            split,
            grid(),
            Split.TRAIN,
            ("position", "constant"),
            lambda entry: arrays[entry.trajectory_id],
            batch_size=1,
            with_targets=True,
        )
    )
    assert targets is not None
    prediction = StatisticalBaseline(config(ModelId.VAR, lags=1)).predict(batch)
    np.testing.assert_allclose(prediction, targets, atol=1e-9)
    assert not np.allclose(
        StatisticalBaseline(config(ModelId.AR, lags=1)).predict(batch), targets
    )


@pytest.mark.parametrize(
    "field", ["config_hash", "mae", "group_id", "reference", "window_count"]
)
def test_evaluation_serialization_guards(field: str) -> None:
    models = (
        StatisticalBaseline(config(ModelId.PERSISTENCE)),
        StatisticalBaseline(config(ModelId.CONTEXT_MEAN)),
    )
    payload = evaluate_baselines(models, pairs()).model_dump(mode="json")
    if field == "config_hash":
        payload["scores"][0][field] = "sha256:" + "f" * 64
    elif field == "reference":
        payload["reference_config_hash"] = "sha256:" + "f" * 64
    elif field == "mae":
        payload["scores"][0]["groups"][0][field] = [1.0]
    elif field == "group_id":
        payload["scores"][0]["groups"][0][field] = "zzzzz"
    else:
        payload["scores"][0]["groups"][0][field] += 1
    with pytest.raises(ValidationError):
        BaselineEvaluation.model_validate(payload)


def test_numerical_overflow_is_not_silently_scored() -> None:
    batch, labels = pairs()[0]
    assert labels is not None
    huge = ForecastBatch(batch.spec, batch.indices, np.full(batch.context.shape, 1e308))
    with pytest.raises(ForecastError, match="overflowed"):
        StatisticalBaseline(config(ModelId.CONTEXT_MEAN)).predict(huge)
    with pytest.raises(ForecastError, match="errors overflowed"):
        evaluate_baselines(
            (StatisticalBaseline(config(ModelId.PERSISTENCE)),),
            [(huge, np.full(labels.shape, -1e308))],
        )


def test_selection_and_scaler_scope_integrity() -> None:
    from md_forecast.data.preprocessing import ScalerMetadata

    split = source()
    arrays = series(split)
    model, selection = select_nlinear(
        split,
        grid(),
        ("position", "constant"),
        lambda entry: arrays[entry.trajectory_id],
        (config(ModelId.NLINEAR, ridge=10.0), config(ModelId.NLINEAR)),
        training(),
    )
    assert (
        NLinearSelection.model_validate_json(selection.model_dump_json()) == selection
    )
    assert selection.preprocessing_hash == model.state.preprocessing_hash
    payload = selection.model_dump(mode="json")
    payload["evaluation"]["partition"] = "test"
    with pytest.raises(ValidationError, match="validation"):
        NLinearSelection.model_validate(payload)
    for change in (
        {"normalized_validation_mae": [1.0]},
        {"selected_config_hash": "sha256:" + "a" * 64},
    ):
        with pytest.raises(ValidationError):
            NLinearSelection.model_validate(selection.model_dump() | change)
    state = model.state.model_dump(mode="json")
    state["scaler"]["config"]["scope"] = "context-local"
    state["scaler"]["fit_regions"] = state["scaler"]["fit_regions"][:1]
    state["scaler"]["sample_count"] = 10
    state["preprocessing_hash"] = metadata_hash(
        ScalerMetadata.model_validate(state["scaler"])
    )
    with pytest.raises(ValidationError, match="training-contexts"):
        NLinearState.model_validate(state)
    evaluation = selection.evaluation.model_dump(mode="json")
    evaluation["scores"].append(evaluation["scores"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        BaselineEvaluation.model_validate(evaluation)


def test_backend_overflow_and_nonfinite_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md_forecast.models import learned

    model = fit_model()
    batch, _ = pairs()[0]

    def overflow(*args: object, **kwargs: object) -> FloatArray:
        raise FloatingPointError("simulated backend overflow")

    monkeypatch.setattr(np, "einsum", overflow)
    with pytest.raises(ForecastError, match="prediction overflowed"):
        model.predict(batch)
    monkeypatch.setattr(learned, "ridge_fit", overflow)
    with pytest.raises(ForecastError, match="fitting overflowed"):
        fit_model()

    def invalid_coefficients(*args: object, **kwargs: object) -> tuple[FloatArray]:
        return (np.full((2, 1), np.nan),)

    monkeypatch.setattr(np.linalg, "lstsq", invalid_coefficients)
    with pytest.raises(ForecastError, match="nonfinite"):
        ridge_fit(np.ones((3, 2)), np.ones((3, 1)), 0.0)


def test_variable_grid_batches() -> None:
    split = source()
    arrays = series(split)
    variable = WindowConfig(contexts=(2.0, 6.0), horizons=(1.0, 2.0), stride_frames=1)
    batches = list(
        iter_forecasts(
            split,
            variable,
            Split.TRAIN,
            ("position", "constant"),
            lambda entry: arrays[entry.trajectory_id],
            batch_size=3,
            with_targets=True,
        )
    )
    assert sum(len(batch.indices) for batch, _ in batches) == 90
    assert {
        (batch.spec.context_frames, batch.spec.horizon_frames) for batch, _ in batches
    } == {(2, 1), (2, 2), (6, 1), (6, 2)}
    for batch, labels in batches:
        assert labels is not None
        assert (
            StatisticalBaseline(config(ModelId.PERSISTENCE)).predict(batch).shape
            == labels.shape
        )
