"""Persistence, context mean, linear trend and observed-context AR/VAR."""

from dataclasses import dataclass

import numpy as np

from md_forecast.core.constants import ModelId
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import ForecastBatch, validate_array
from md_forecast.data.series import FloatArray
from md_forecast.models.base import ModelConfig


def ridge_fit(design: FloatArray, targets: FloatArray, ridge: float) -> FloatArray:
    """Solve least squares with an unpenalized final intercept, never invert X'X."""
    augmented, labels = design, targets
    if ridge > 0:
        penalty = np.sqrt(ridge) * np.eye(design.shape[1])
        penalty[-1, -1] = 0
        augmented = np.vstack((design, penalty))
        labels = np.vstack((targets, np.zeros((len(penalty), targets.shape[1]))))
    try:
        coefficients = np.asarray(
            np.linalg.lstsq(augmented, labels, rcond=None)[0], dtype=np.float64
        )
    except np.linalg.LinAlgError as error:
        raise ForecastError("baseline least-squares solver did not converge") from error
    if not np.isfinite(coefficients).all():
        raise ForecastError("baseline least-squares coefficients are nonfinite")
    return coefficients


@dataclass(frozen=True)
class StatisticalBaseline:
    """Context-local deterministic adapter; no dataset-level fitting or fallback."""

    config: ModelConfig

    @property
    def artifact_hash(self) -> str:
        """Stateless adapter identity equals its complete configuration hash."""
        return metadata_hash(self.config)

    def __post_init__(self) -> None:
        """Reject learned-model configurations at the statistical boundary."""
        object.__setattr__(
            self,
            "config",
            ModelConfig.model_validate_json(self.config.model_dump_json()),
        )
        if self.config.model_id == ModelId.NLINEAR:
            raise ForecastError(
                "NLinear requires trained weights, not a statistical adapter"
            )
        if self.config.model_id == ModelId.CHRONOS2:
            raise ForecastError("Chronos-2 requires its pretrained adapter")

    def predict(self, batch: ForecastBatch) -> FloatArray:
        """Fit only observed values and recursively forecast AR/VAR if requested."""
        horizon = batch.spec.horizon_frames
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = self._predict(batch.context, horizon)
        except FloatingPointError as error:
            raise ForecastError("baseline prediction overflowed") from error
        validate_array(
            values, (len(batch.indices), horizon, len(batch.spec.feature_ids))
        )
        return values

    def _predict(self, context: FloatArray, horizon: int) -> FloatArray:
        if self.config.model_id == ModelId.PERSISTENCE:
            return np.repeat(context[:, -1:, :], horizon, axis=1)
        if self.config.model_id == ModelId.CONTEXT_MEAN:
            return np.repeat(context.mean(axis=1, keepdims=True), horizon, axis=1)
        if self.config.model_id == ModelId.LINEAR:
            return _linear(context, horizon)
        return np.stack(
            [_autoregression(values, horizon, self.config) for values in context]
        )


def _linear(context: FloatArray, horizon: int) -> FloatArray:
    frames = context.shape[1]
    if frames < 2:
        raise ForecastError(
            "linear extrapolation requires at least two observed samples"
        )
    time = np.arange(frames, dtype=np.float64)
    time -= time.mean()
    mean = context.mean(axis=1, keepdims=True)
    slope = np.sum((context - mean) * time[None, :, None], axis=1) / (time @ time)
    future = np.arange(frames, frames + horizon, dtype=np.float64) - (frames - 1) / 2
    return np.asarray(
        mean + slope[:, None, :] * future[None, :, None], dtype=np.float64
    )


def _autoregression(
    context: FloatArray, horizon: int, config: ModelConfig
) -> FloatArray:
    assert config.lags is not None
    if config.model_id == ModelId.AR:
        columns = [
            _recursive(
                context[:, index : index + 1], horizon, config.lags, config.ridge
            )
            for index in range(context.shape[1])
        ]
        return np.column_stack(columns)
    return _recursive(context, horizon, config.lags, config.ridge)


def _recursive(
    context: FloatArray, horizon: int, lags: int, ridge: float
) -> FloatArray:
    rows = len(context) - lags
    width = lags * context.shape[1] + 1
    if rows < width:
        raise ForecastError(
            f"AR/VAR needs at least {width} lagged observations, got {rows}"
        )
    design = np.array(
        [
            np.append(context[index - lags : index].ravel(), 1.0)
            for index in range(lags, len(context))
        ],
        dtype=np.float64,
    )
    coefficients = ridge_fit(design, context[lags:], ridge)
    history = np.concatenate((context, np.zeros((horizon, context.shape[1]))))
    for step in range(len(context), len(history)):
        history[step] = (
            np.append(history[step - lags : step].ravel(), 1.0) @ coefficients
        )
    return history[len(context) :]
