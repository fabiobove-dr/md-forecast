"""Descriptive forecast spread and whole-system development dependence estimates."""

from collections.abc import Mapping
from typing import Annotated

import numpy as np
from pydantic import Field

from md_forecast.analysis.qc import DECAY_LEVEL, autocorrelation
from md_forecast.core.exceptions import DataContractError, ForecastError
from md_forecast.data.forecast import ForecastBatch, validate_array
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.series import FloatArray
from md_forecast.evaluation.metrics import BenchmarkConfig, group_interval

# Population spread describes the finite observed/predicted horizon, not a
# sample-variance estimator. A one-sample horizon therefore has zero spread.
SPREAD_DDOF = 0


def forecast_diagnostics(
    batch: ForecastBatch, points: FloatArray, targets: FloatArray
) -> dict[str, FloatArray]:
    """Return (B,F) horizon summaries and (B,H,F) lead errors in native units.

    Constant truth has an undefined spread ratio (NaN in memory, null on export).
    Change errors use the same last observed origin on both sides, so their MAE
    equals ordinary MAE; they are not additional independent accuracy evidence.
    Targets are used only for diagnostics after context-only prediction.
    """
    shape = (len(batch.indices), batch.spec.horizon_frames, len(batch.spec.feature_ids))
    validate_array(points, shape)
    validate_array(targets, shape)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _forecast_summaries(batch.context[:, -1, :], points, targets)
    except FloatingPointError as error:
        raise ForecastError("forecast diagnostic arithmetic overflowed") from error


def _forecast_summaries(
    origin: FloatArray, points: FloatArray, targets: FloatArray
) -> dict[str, FloatArray]:
    error = points - targets
    predicted_std = points.std(axis=1, ddof=SPREAD_DDOF)
    actual_std = targets.std(axis=1, ddof=SPREAD_DDOF)
    ratio = np.divide(
        predicted_std,
        actual_std,
        out=np.full(actual_std.shape, np.nan),
        where=actual_std != 0,
    )
    return {
        "predicted_std": predicted_std,
        "actual_std": actual_std,
        "spread_ratio": ratio,
        "mean_bias": error.mean(axis=1),
        "mae": np.abs(error).mean(axis=1),
        "rmse": np.sqrt(np.square(error).mean(axis=1)),
        "predicted_change_from_origin": points.mean(axis=1) - origin,
        "actual_change_from_origin": targets.mean(axis=1) - origin,
        "change_from_origin_mae": np.abs(error).mean(axis=1),
        "lead_bias": error,
        "lead_absolute_error": np.abs(error),
        "lead_squared_error": np.square(error),
    }


class DependenceSummary(BoundaryModel):
    """Equal-system averages; replicas are averaged before system resampling."""

    max_lag_frames: Annotated[int, Field(strict=True, gt=0)]
    groups: Annotated[int, Field(strict=True, ge=0)]
    trajectories: Annotated[int, Field(strict=True, gt=0)]
    constant_trajectories: Annotated[int, Field(strict=True, ge=0)]
    mean_acf: tuple[float, ...] | None
    acf_lower: tuple[float | None, ...] | None
    acf_upper: tuple[float | None, ...] | None
    restricted_decay_mean_frames: float | None
    restricted_decay_interval: tuple[float | None, float | None]
    censored_fraction: float | None
    positive_integrated_acf_mean_frames: float | None
    positive_integrated_acf_interval: tuple[float | None, float | None]


def dependence_summary(
    groups: Mapping[str, tuple[FloatArray, ...]],
    max_lag: int,
    uncertainty: BenchmarkConfig,
) -> DependenceSummary:
    """Estimate descriptive ACF and restricted correlation times on admitted data.

    First 1/e crossing is capped at max_lag; noncrossers are right-censored.
    Positive integrated ACF is 0.5 + sum before the first nonpositive lag,
    capped by the measured lag limit. Neither estimate establishes intrinsic
    relaxation or irreducible noise in short, sparsely sampled sequences.
    Constant replicas have undefined correlation and are counted, not imputed.
    The caller must use a development-only loader before calling this routine.
    """
    _validate_groups(groups, max_lag)
    curves, decay, integrated, censored, constants = _group_dependence(groups, max_lag)
    n = sum(len(rows) for rows in groups.values())
    if not curves:
        return _constant_dependence(max_lag, n, constants)
    return _dependence_estimates(
        curves, decay, integrated, censored, max_lag, n, constants, uncertainty
    )


def _validate_groups(
    groups: Mapping[str, tuple[FloatArray, ...]], max_lag: int
) -> None:
    if max_lag < 1 or not groups or any(not rows for rows in groups.values()):
        raise DataContractError("dependence requires nonempty groups and positive lag")


def _constant_dependence(max_lag: int, n: int, constants: int) -> DependenceSummary:
    return DependenceSummary(
        max_lag_frames=max_lag,
        groups=0,
        trajectories=n,
        constant_trajectories=constants,
        mean_acf=None,
        acf_lower=None,
        acf_upper=None,
        restricted_decay_mean_frames=None,
        restricted_decay_interval=(None, None),
        censored_fraction=None,
        positive_integrated_acf_mean_frames=None,
        positive_integrated_acf_interval=(None, None),
    )


def _dependence_estimates(
    curves: list[FloatArray],
    decay: list[float],
    integrated: list[float],
    censored: list[float],
    max_lag: int,
    n: int,
    constants: int,
    uncertainty: BenchmarkConfig,
) -> DependenceSummary:
    matrix = np.stack(curves)
    intervals = [
        group_interval(matrix[:, lag], uncertainty) for lag in range(max_lag + 1)
    ]
    return DependenceSummary(
        max_lag_frames=max_lag,
        groups=len(curves),
        trajectories=n,
        constant_trajectories=constants,
        mean_acf=tuple(matrix.mean(axis=0)),
        acf_lower=tuple(lo for lo, _ in intervals),
        acf_upper=tuple(hi for _, hi in intervals),
        restricted_decay_mean_frames=float(np.mean(decay)),
        restricted_decay_interval=group_interval(np.asarray(decay), uncertainty),
        censored_fraction=float(np.mean(censored)),
        positive_integrated_acf_mean_frames=float(np.mean(integrated)),
        positive_integrated_acf_interval=group_interval(
            np.asarray(integrated), uncertainty
        ),
    )


def _group_dependence(
    groups: Mapping[str, tuple[FloatArray, ...]], max_lag: int
) -> tuple[list[FloatArray], list[float], list[float], list[float], int]:
    curves, decay, integrated, censored = [], [], [], []
    constants = 0
    for name in sorted(groups):
        curve, times, count = _replica_dependence(groups[name], max_lag)
        constants += count
        if curve is not None:
            curves.append(curve)
            decay.append(float(times[0]))
            integrated.append(float(times[1]))
            censored.append(float(times[2]))
    return curves, decay, integrated, censored, constants


def _replica_dependence(
    rows: tuple[FloatArray, ...], max_lag: int
) -> tuple[FloatArray | None, FloatArray, int]:
    replicas = [_checked_acf(values, max_lag) for values in rows]
    valid = [curve for curve in replicas if curve is not None]
    constants = len(replicas) - len(valid)
    if not valid:
        return None, np.zeros(3), constants
    curve, times = _average_replicas(valid, max_lag)
    return curve, times, constants


def _average_replicas(
    valid: list[FloatArray], max_lag: int
) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(
        [_correlation_times(curve, max_lag) for curve in valid], dtype=np.float64
    )
    return np.stack(valid).mean(axis=0), times.mean(axis=0)


def _checked_acf(values: FloatArray, max_lag: int) -> FloatArray | None:
    if values.ndim != 1 or len(values) <= max_lag or not np.isfinite(values).all():
        raise DataContractError(
            "dependence requires finite 1D series longer than lag limit"
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return autocorrelation(values, max_lag)
    except FloatingPointError as error:
        raise ForecastError("dependence arithmetic overflowed") from error


def _correlation_times(curve: FloatArray, max_lag: int) -> tuple[int, float, bool]:
    crossings = np.flatnonzero(curve[1:] <= DECAY_LEVEL)
    decay = int(crossings[0]) + 1 if len(crossings) else max_lag
    zero = np.flatnonzero(curve[1:] <= 0)
    stop = int(zero[0]) + 1 if len(zero) else max_lag + 1
    return decay, float(0.5 + curve[1:stop].sum()), not len(crossings)
