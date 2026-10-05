"""Held-out predictive gain and observed-context conventional lag diagnostics."""

from typing import Annotated, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.data.forecast import ForecastBatch
from md_forecast.data.schemas import ArtifactHash
from md_forecast.data.series import FloatArray
from md_forecast.evaluation.ablations import AblationConfig
from md_forecast.models.baselines import ridge_fit


class CouplingConfig(AblationConfig):
    """Freeze one target and regional input variants with a replica holdout."""

    regions_hash: ArtifactHash
    lag_order: Annotated[int, Field(strict=True, gt=0)] = 1
    ridge: Annotated[float, Field(ge=0)] = 0.000001
    max_correlation_lag: Annotated[int, Field(strict=True, ge=1)] = 5

    @model_validator(mode="after")
    def validate_coupling(self) -> Self:
        """Keep region direction and replica generalization explicit."""
        if self.target != "distal_rg" or tuple(self.variants.values()) != (
            ("distal_rg",),
            ("distal_rg", "pocket_rg"),
        ):
            raise ValueError(
                "coupling requires distal B alone then distal B + pocket A"
            )
        if self.split.mode != "unseen-replica":
            raise ValueError("coupling experiment requires held-out replicas")
        return self


def lag_diagnostics(
    batch: ForecastBatch, config: CouplingConfig
) -> list[dict[str, object]]:
    """Calculate A-leading-B correlations and lagged fit gain from context alone.

    No future labels enter the API. Positive lag means A(t) is correlated with
    B(t+lag). In-context SSE improvement is descriptive, not a Granger p-value.
    """
    if batch.spec.feature_ids != ("distal_rg", "pocket_rg"):
        raise ValueError("diagnostics require ordered B,A context")
    if config.max_correlation_lag >= batch.spec.context_frames - 1:
        raise ValueError("lagged correlation needs at least two paired samples")
    rows = []
    for index, values in zip(batch.indices, batch.context, strict=True):
        gain = _lagged_fit_gain(values, config.lag_order, config.ridge)
        for lag in range(-config.max_correlation_lag, config.max_correlation_lag + 1):
            rows.append(
                dict(
                    trajectory_id=index.trajectory_id,
                    group_id=index.group_id,
                    start=index.start,
                    lag_frames=lag,
                    a_leads_b_correlation=_correlation(values, lag),
                    context_sse_gain_fraction=gain,
                )
            )
    return rows


def _correlation(values: FloatArray, lag: int) -> float | None:
    a, b = values[:, 1], values[:, 0]
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    return _pearson(a, b)


def _pearson(a: FloatArray, b: FloatArray) -> float | None:
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return None
    return float(np.clip(a @ b / denominator, -1, 1))


def _lagged_fit_gain(values: FloatArray, lags: int, ridge: float) -> float | None:
    rows = len(values) - lags
    if rows < lags * values.shape[1] + 1:
        raise ValueError("lagged regression has too few context observations")
    scales = values.std(axis=0)
    scales[scales == 0] = 1
    standardized = (values - values.mean(axis=0)) / scales
    target = standardized[lags:, :1]
    restricted = _lag_design(standardized[:, :1], lags)
    joint = _lag_design(standardized, lags)
    reference = _sse(restricted, target, ridge)
    if reference <= np.finfo(np.float64).eps:
        return None
    return 1 - _sse(joint, target, ridge) / reference


def _lag_design(values: FloatArray, lags: int) -> FloatArray:
    return np.asarray(
        [np.append(values[i - lags : i].ravel(), 1) for i in range(lags, len(values))],
        dtype=np.float64,
    )


def _sse(design: FloatArray, target: FloatArray, ridge: float) -> float:
    coefficients = ridge_fit(design, target, ridge)
    return float(np.sum((design @ coefficients - target) ** 2))
