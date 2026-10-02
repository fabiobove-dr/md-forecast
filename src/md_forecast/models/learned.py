"""Per-channel NLinear with train-only scaling and validation selection."""

from dataclasses import dataclass
from typing import Annotated, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    ModelId,
    Split,
    TimeUnit,
)
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import (
    ForecastBatch,
    ForecastSpec,
    iter_forecasts,
    validate_array,
)
from md_forecast.data.preprocessing import (
    ScalerMetadata,
    ScalingConfig,
    SeriesLoader,
    fit_training_scaler,
    transform,
)
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    Identifier,
    SchemaVersion,
)
from md_forecast.data.series import FloatArray
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, iter_windows
from md_forecast.evaluation.baselines import BaselineEvaluation, evaluate_baselines
from md_forecast.models.base import ModelConfig
from md_forecast.models.baselines import StatisticalBaseline, ridge_fit


class TrainingConfig(BoundaryModel):
    """Bounded CPU fitting budget; never silently subsample training windows."""

    seed: Annotated[int, Field(strict=True, ge=0)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    max_windows: Annotated[int, Field(strict=True, gt=0)]
    max_design_bytes: Annotated[int, Field(strict=True, gt=0)]


class NLinearState(BoundaryModel):
    """Portable fitted coefficients (F,C+1,H) and complete preprocessing provenance."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    config: ModelConfig
    spec: ForecastSpec
    training: TrainingConfig
    scaler: ScalerMetadata
    preprocessing_hash: ArtifactHash
    training_trajectory_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    training_window_count: Annotated[int, Field(strict=True, gt=0)]
    weights: tuple[tuple[tuple[float, ...], ...], ...]

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Reject incompatible dimensions, feature order, scopes or artifact linkage."""
        _check_nlinear(self.config)
        _check_state_scaler(self)
        _check_training_ids(self)
        shape = (
            len(self.spec.feature_ids),
            self.spec.context_frames + 1,
            self.spec.horizon_frames,
        )
        if np.asarray(self.weights).shape != shape:
            raise ValueError(
                "NLinear weight dimensions differ from forecast specification"
            )
        return self


def _check_training_ids(state: NLinearState) -> None:
    expected = tuple(
        sorted({region.trajectory_id for region in state.scaler.fit_regions})
    )
    if state.training_trajectory_ids != expected:
        raise ValueError("training identities differ from fitted scaler regions")
    if state.training_window_count > state.training.max_windows:
        raise ValueError("fitted training windows exceed the declared budget")


def _check_nlinear(config: ModelConfig) -> None:
    if config.model_id != ModelId.NLINEAR:
        raise ForecastError("learned fitting requires an NLinear configuration")


def _check_state_scaler(state: NLinearState) -> None:
    scaler = state.scaler
    if scaler.config.scope != "training-contexts":
        raise ValueError("NLinear requires training-contexts preprocessing")
    expected = (
        state.spec.dataset,
        state.spec.feature_ids,
        state.spec.split_hash,
        state.spec.window_config_hash,
    )
    actual = (
        scaler.dataset,
        scaler.config.feature_ids,
        scaler.split_hash,
        scaler.window_config_hash,
    )
    if actual != expected or state.preprocessing_hash != metadata_hash(scaler):
        raise ValueError("NLinear scaler differs from feature/source/hash protocol")


@dataclass(frozen=True)
class NLinearModel:
    """Independent single-layer temporal mappings fitted with deterministic ridge LS."""

    state: NLinearState

    def __post_init__(self) -> None:
        """Revalidate loaded or copied states at the adapter boundary."""
        object.__setattr__(
            self,
            "state",
            NLinearState.model_validate_json(self.state.model_dump_json()),
        )

    @property
    def config(self) -> ModelConfig:
        """Recorded seed and NLinear regularization."""
        return self.state.config

    def predict(self, batch: ForecastBatch) -> FloatArray:
        """Map last-level-centered contexts to H steps and restore native units."""
        if batch.spec != self.state.spec:
            raise ForecastError(
                "NLinear batch differs from fitted source/features/grid cell"
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                context = _scaled(batch.context, self.state.scaler)
                last = context[:, -1:, :]
                design = _design(context - last)
                weights = np.asarray(self.state.weights, dtype=np.float64)
                prediction = np.einsum("bcf,fch->bhf", design, weights) + last
                result = _scaled(prediction, self.state.scaler, inverse=True)
        except FloatingPointError as error:
            raise ForecastError("NLinear prediction overflowed") from error
        validate_array(
            result,
            (
                len(batch.indices),
                batch.spec.horizon_frames,
                len(batch.spec.feature_ids),
            ),
        )
        return result


def _scaled(
    values: FloatArray, scaler: ScalerMetadata, *, inverse: bool = False
) -> FloatArray:
    shape = values.shape
    return transform(values.reshape(-1, shape[-1]), scaler, inverse=inverse).reshape(
        shape
    )


def _design(centered: FloatArray) -> FloatArray:
    return np.concatenate(
        (centered, np.ones((len(centered), 1, centered.shape[2]))), axis=1
    )


def _check_fit_budget(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    training: TrainingConfig,
) -> int:
    if (grid.unit, len(grid.contexts), len(grid.horizons)) != (TimeUnit.FRAME, 1, 1):
        raise ForecastError("fit one explicit frame-grid cell per NLinear model")
    count = sum(1 for _ in iter_windows(split, grid, Split.TRAIN))
    if not 0 < count <= training.max_windows:
        raise ForecastError(
            "training windows exceed budget or no training windows are available"
        )
    context, horizon = int(grid.contexts[0]), int(grid.horizons[0])
    # Estimate simultaneous arrays and ridge penalty, not total process RSS.
    estimated = np.dtype(np.float64).itemsize * (
        4 * count * (context + horizon + 1) * len(features) + (context + 1) ** 2
    )
    if estimated > training.max_design_bytes:
        raise ForecastError(
            f"training arrays need an estimated {estimated} bytes, exceeding budget"
        )
    return count


def fit_nlinear(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    loader: SeriesLoader,
    config: ModelConfig,
    training: TrainingConfig,
) -> NLinearModel:
    """Fit every training window; never load held-out series for fitting."""
    _check_nlinear(config)
    _check_fit_budget(split, grid, features, training)
    scaler = fit_training_scaler(
        split,
        grid,
        ScalingConfig(scope="training-contexts", feature_ids=features),
        loader,
    )
    pairs = list(
        iter_forecasts(
            split,
            grid,
            Split.TRAIN,
            features,
            loader,
            batch_size=training.batch_size,
            with_targets=True,
        )
    )
    contexts, targets, spec, ids = _training_arrays(pairs, scaler)
    try:
        with np.errstate(over="raise", invalid="raise"):
            last = contexts[:, -1:, :]
            design, labels = _design(contexts - last), targets - last
            weights = np.stack(
                [
                    ridge_fit(design[:, :, index], labels[:, :, index], config.ridge)
                    for index in range(len(features))
                ]
            )
    except FloatingPointError as error:
        raise ForecastError("NLinear fitting overflowed") from error
    state = NLinearState(
        config=config,
        spec=spec,
        training=training,
        scaler=scaler,
        preprocessing_hash=metadata_hash(scaler),
        training_trajectory_ids=ids,
        training_window_count=len(contexts),
        weights=tuple(tuple(tuple(row) for row in channel) for channel in weights),
    )
    return NLinearModel(state)


def _training_arrays(
    pairs: list[tuple[ForecastBatch, FloatArray | None]],
    scaler: ScalerMetadata,
) -> tuple[FloatArray, FloatArray, ForecastSpec, tuple[str, ...]]:
    contexts: list[FloatArray] = []
    targets: list[FloatArray] = []
    identities: set[str] = set()
    for batch, labels in pairs:
        assert labels is not None
        contexts.append(batch.context)
        targets.append(labels)
        identities.update(index.trajectory_id for index in batch.indices)
    return (
        _scaled(np.concatenate(contexts), scaler),
        _scaled(np.concatenate(targets), scaler),
        pairs[0][0].spec,
        tuple(sorted(identities)),
    )


class NLinearSelection(BoundaryModel):
    """Validation-only selection using equal-group MAE normalized by train std."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    selected_config_hash: ArtifactHash
    preprocessing_hash: ArtifactHash
    normalized_validation_mae: Annotated[
        tuple[Annotated[float, Field(ge=0)], ...], Field(min_length=1)
    ]
    evaluation: BaselineEvaluation

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Selection must describe validation-only candidates and its actual argmin."""
        actual = (self.evaluation.partition, self.evaluation.scores[0].config.model_id)
        if actual != (Split.VALIDATION, ModelId.PERSISTENCE):
            raise ValueError(
                "NLinear selection requires validation and persistence first"
            )
        _check_selection_scores(self)
        return self


def _check_selection_scores(selection: NLinearSelection) -> None:
    candidates = tuple(score.config for score in selection.evaluation.scores[1:])
    _check_candidates(candidates)
    if len(candidates) != len(selection.normalized_validation_mae):
        raise ValueError("validation score dimensions differ from candidates")
    index = int(np.argmin(selection.normalized_validation_mae))
    if selection.selected_config_hash != metadata_hash(candidates[index]):
        raise ValueError("selected configuration differs from validation argmin")


def select_nlinear(
    split: SplitManifest,
    grid: WindowConfig,
    features: tuple[str, ...],
    loader: SeriesLoader,
    candidates: tuple[ModelConfig, ...],
    training: TrainingConfig,
) -> tuple[NLinearModel, NLinearSelection]:
    """Train candidates on TRAIN and select on VALIDATION; never access TEST values."""
    _check_candidates(candidates)
    _require_validation(split)
    models = tuple(
        fit_nlinear(split, grid, features, loader, config, training)
        for config in candidates
    )
    reference = StatisticalBaseline(
        ModelConfig(model_id=ModelId.PERSISTENCE, seed=training.seed)
    )
    pairs = iter_forecasts(
        split,
        grid,
        Split.VALIDATION,
        features,
        loader,
        batch_size=training.batch_size,
        with_targets=True,
    )
    result = evaluate_baselines((reference, *models), pairs)
    scores = _selection_scores(result, models[0].state.scaler)
    selected = int(np.argmin(scores))
    return models[selected], NLinearSelection(
        selected_config_hash=metadata_hash(models[selected].config),
        preprocessing_hash=models[selected].state.preprocessing_hash,
        normalized_validation_mae=tuple(scores),
        evaluation=result,
    )


def _check_candidates(candidates: tuple[ModelConfig, ...]) -> None:
    if not candidates or len({metadata_hash(config) for config in candidates}) != len(
        candidates
    ):
        raise ForecastError("selection requires nonempty, unique NLinear candidates")
    for config in candidates:
        _check_nlinear(config)


def _require_validation(split: SplitManifest) -> None:
    if not any(entry.split == Split.VALIDATION for entry in split.assignments):
        raise ForecastError(
            "validation partition is empty; never substitute training or test"
        )


def _selection_scores(result: BaselineEvaluation, scaler: ScalerMetadata) -> FloatArray:
    scale = np.asarray(scaler.scale, dtype=np.float64)
    return np.array(
        [
            np.mean(np.asarray([group.mae for group in score.groups]) / scale)
            for score in result.scores[1:]
        ],
        dtype=np.float64,
    )
