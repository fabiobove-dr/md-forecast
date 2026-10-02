"""Model-independent native quantile forecasts and optional accelerator telemetry."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from md_forecast.data.forecast import (
    ForecastSpec,
    validate_array,
    validate_forecast_indices,
)
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.series import FloatArray
from md_forecast.data.windows import WindowIndex


def validate_quantile_levels(levels: tuple[float, ...]) -> None:
    """Reject ambiguous labels; point compatibility requires a native median."""
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("quantiles must be ordered and unique")
    if not all(0 < level < 1 for level in levels) or 0.5 not in levels:
        raise ValueError("quantiles must be in (0,1) and include 0.5")


class RuntimeStats(BoundaryModel):
    """Synchronized elapsed time and process CUDA allocator peaks, not free VRAM."""

    seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    peak_allocated_bytes: Annotated[int, Field(strict=True, ge=0)]
    peak_reserved_bytes: Annotated[int, Field(strict=True, ge=0)]


@dataclass(frozen=True)
class QuantileForecast:
    """Native (B,H,F,Q) values with unchanged source windows and quantile labels."""

    spec: ForecastSpec
    indices: tuple[WindowIndex, ...]
    quantile_levels: tuple[float, ...]
    values: FloatArray
    runtime: RuntimeStats

    def __post_init__(self) -> None:
        """Validate output axes before detaching immutable native arrays."""
        validate_quantile_levels(self.quantile_levels)
        validate_forecast_indices(self.indices, self.spec)
        validate_array(
            self.values,
            (
                len(self.indices),
                self.spec.horizon_frames,
                len(self.spec.feature_ids),
                len(self.quantile_levels),
            ),
        )
        values = self.values.copy()
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    @property
    def median(self) -> FloatArray:
        """Explicit point view, not a predictive mean."""
        return self.values[..., self.quantile_levels.index(0.5)]
