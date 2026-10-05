"""Native-unit MAE with trajectory-then-group aggregation and persistence reference."""

from collections.abc import Iterable
from typing import Annotated, Self, cast

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, ModelId, Split
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch, ForecastSpec, validate_array
from md_forecast.data.schemas import (
    ArtifactHash,
    BoundaryModel,
    Identifier,
    SchemaVersion,
)
from md_forecast.data.series import FloatArray
from md_forecast.models.base import ForecastModel, ModelConfig

type Totals = dict[str, dict[str, tuple[FloatArray, int]]]


class GroupMAE(BoundaryModel):
    """Equal trajectory weight within one dependent system group; no window CI."""

    group_id: Identifier
    mae: tuple[Annotated[float, Field(ge=0)], ...]
    trajectory_count: Annotated[int, Field(strict=True, gt=0)]
    window_count: Annotated[int, Field(strict=True, gt=0)]


class ModelScore(BoundaryModel):
    """Frozen model configuration and per-group errors in ordered native units."""

    config: ModelConfig
    config_hash: ArtifactHash
    groups: Annotated[tuple[GroupMAE, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_score(self) -> Self:
        """Reject mislabeled configurations and repeated or unordered groups."""
        if self.config_hash != metadata_hash(self.config):
            raise ValueError("baseline score configuration hash differs")
        ids = tuple(group.group_id for group in self.groups)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("baseline groups must be sorted and unique")
        return self


class BaselineEvaluation(BoundaryModel):
    """One comparable grid cell and partition, always retaining persistence."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    spec: ForecastSpec
    partition: Split
    reference_config_hash: ArtifactHash
    scores: Annotated[tuple[ModelScore, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        """Require paired groups, matching feature dimensions and persistence."""
        _validate_evaluation_reference(self)
        _validate_score_groups(self.scores, len(self.spec.feature_ids))
        return self


def normalized_group_mae(
    evaluation: BaselineEvaluation, training_scale: FloatArray
) -> FloatArray:
    """Score each candidate with equal-system MAE divided by TRAIN feature scales.

    The caller supplies verified training-only scale provenance. This shared
    score averages native-unit ratios across features, never mixes raw units
    or treats dependent windows as independent selection observations.
    """
    validate_array(training_scale, (len(evaluation.spec.feature_ids),))
    if np.any(training_scale <= 0):
        raise ForecastError("selection scales must be positive and fitted on TRAIN")
    return np.asarray(
        [
            np.mean(np.asarray([group.mae for group in score.groups]) / training_scale)
            for score in evaluation.scores
        ],
        dtype=np.float64,
    )


def _validate_evaluation_reference(result: BaselineEvaluation) -> None:
    configs = tuple(score.config for score in result.scores)
    keys = tuple(score.config_hash for score in result.scores)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate evaluation configurations")
    if result.reference_config_hash != _reference_key(configs, keys):
        raise ValueError("evaluation reference differs from persistence")


def _group_signature(score: ModelScore) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (group.group_id, group.trajectory_count, group.window_count)
        for group in score.groups
    )


def _validate_score_groups(scores: tuple[ModelScore, ...], features: int) -> None:
    signature = _group_signature(scores[0])
    for score in scores:
        if _group_signature(score) != signature:
            raise ValueError("baseline errors do not describe paired evaluation groups")
        for group in score.groups:
            if len(group.mae) != features:
                raise ValueError("baseline error dimensions differ from features")


def _model_keys(models: tuple[ForecastModel, ...]) -> tuple[tuple[str, ...], str]:
    keys = tuple(metadata_hash(model.config) for model in models)
    if len(set(keys)) != len(keys):
        raise ForecastError("duplicate baseline configurations")
    return keys, _reference_key(tuple(model.config for model in models), keys)


def _reference_key(configs: tuple[ModelConfig, ...], keys: tuple[str, ...]) -> str:
    reference = [
        key
        for config, key in zip(configs, keys, strict=True)
        if config.model_id == ModelId.PERSISTENCE
    ]
    if len(reference) != 1:
        raise ForecastError("evaluation requires exactly one persistence reference")
    return reference[0]


def evaluate_baselines(
    models: tuple[ForecastModel, ...],
    pairs: Iterable[tuple[ForecastBatch, FloatArray | None]],
) -> BaselineEvaluation:
    """Run adapters on identical contexts and score future labels separately."""
    keys, reference = _model_keys(models)
    totals: dict[str, Totals] = {key: {} for key in keys}
    spec: ForecastSpec | None = None
    partition: Split | None = None
    for batch, targets in pairs:
        spec, partition, labels = _check_evaluation_batch(
            batch, targets, spec, partition
        )
        _evaluate_batch(models, keys, totals, batch, labels)
    if spec is None:
        raise ForecastError("baseline evaluation requires nonempty windows")
    return BaselineEvaluation(
        spec=spec,
        partition=cast(Split, partition),
        reference_config_hash=reference,
        scores=tuple(
            ModelScore(
                config=model.config, config_hash=key, groups=_group_scores(totals[key])
            )
            for model, key in zip(models, keys, strict=True)
        ),
    )


def _check_evaluation_batch(
    batch: ForecastBatch,
    targets: FloatArray | None,
    spec: ForecastSpec | None,
    partition: Split | None,
) -> tuple[ForecastSpec, Split, FloatArray]:
    if targets is None:
        raise ForecastError("evaluation requires separate target labels")
    if spec is not None and (batch.spec != spec or batch.indices[0].split != partition):
        raise ForecastError(
            "evaluation cannot mix grid cells, source protocols or partitions"
        )
    validate_array(
        targets,
        (len(batch.indices), batch.spec.horizon_frames, len(batch.spec.feature_ids)),
    )
    return batch.spec, batch.indices[0].split, targets


def _evaluate_batch(
    models: tuple[ForecastModel, ...],
    keys: tuple[str, ...],
    totals: dict[str, Totals],
    batch: ForecastBatch,
    targets: FloatArray,
) -> None:
    try:
        with np.errstate(over="raise", invalid="raise"):
            for model, key in zip(models, keys, strict=True):
                predictions = model.predict(batch)
                validate_array(predictions, targets.shape)
                errors = np.abs(predictions - targets).mean(axis=1)
                _accumulate(totals[key], batch, errors)
    except FloatingPointError as error:
        raise ForecastError("baseline evaluation errors overflowed") from error


def _accumulate(totals: Totals, batch: ForecastBatch, errors: FloatArray) -> None:
    for index, error in zip(batch.indices, errors, strict=True):
        group = totals.setdefault(index.group_id, {})
        previous, count = group.get(index.trajectory_id, (np.zeros_like(error), 0))
        group[index.trajectory_id] = (previous + error, count + 1)


def _group_scores(totals: Totals) -> tuple[GroupMAE, ...]:
    return tuple(
        _group_score(group, trajectories)
        for group, trajectories in sorted(totals.items())
    )


def _group_score(
    group: str, trajectories: dict[str, tuple[FloatArray, int]]
) -> GroupMAE:
    mean = np.stack([total / count for total, count in trajectories.values()]).mean(
        axis=0
    )
    return GroupMAE(
        group_id=group,
        mae=tuple(mean),
        trajectory_count=len(trajectories),
        window_count=sum(count for _, count in trajectories.values()),
    )
