"""Synchronous, revision-pinned Chronos-2 boundary; no training or CPU fallback."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from md_forecast.core.constants import ModelId
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.forecast import (
    ForecastBatch,
    ForecastSpec,
    validate_array,
    validate_forecast_indices,
)
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.series import FloatArray
from md_forecast.data.windows import WindowIndex
from md_forecast.models.base import ModelConfig


class ChronosConfig(BoundaryModel):
    """Explicit zero-shot settings, hashed into the common model identity."""

    model_name: Literal["amazon/chronos-2"]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    device: Annotated[str, Field(pattern=r"^cuda:[0-9]+$")]
    dtype: Literal["float32"]
    seed: Annotated[int, Field(strict=True, ge=0)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    quantile_levels: Annotated[tuple[float, ...], Field(min_length=1)]
    max_vram_bytes: Annotated[int, Field(strict=True, gt=0, le=24 * 1024**3)]

    @model_validator(mode="after")
    def validate_quantiles(self) -> Self:
        """Reject ambiguous labels; the common point view is the native median."""
        _validate_quantiles(self.quantile_levels)
        return self


def _validate_quantiles(levels: tuple[float, ...]) -> None:
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
        _validate_quantiles(self.quantile_levels)
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
        """Explicit point view, never a claim that upstream's median is a mean."""
        return self.values[..., self.quantile_levels.index(0.5)]


class Chronos2Adapter:
    """Optional upstream objects stay private; callers receive canonical arrays."""

    def __init__(self, settings: ChronosConfig, *, cache_dir: Path) -> None:
        """Load only an exact local snapshot and verify the requested GPU placement."""
        self.settings = ChronosConfig.model_validate_json(settings.model_dump_json())
        self.config = ModelConfig(
            model_id=ModelId.CHRONOS2,
            seed=settings.seed,
            adapter_config_hash=metadata_hash(self.settings),
        )
        try:
            self._torch = importlib.import_module("torch")
            backend = importlib.import_module("chronos")
            hub = importlib.import_module("huggingface_hub")
            self._validate_device()
            self._torch.cuda.synchronize(settings.device)
            self._torch.cuda.reset_peak_memory_stats(settings.device)
            start = perf_counter()
            snapshot = hub.snapshot_download(
                repo_id=settings.model_name,
                revision=settings.revision,
                cache_dir=str(cache_dir),
                allow_patterns=["config.json", "model.safetensors"],
            )
            self._pipeline: Any = backend.Chronos2Pipeline.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                device_map=settings.device,
                dtype=getattr(self._torch, settings.dtype),
                use_safetensors=True,
            )
            self._pipeline.model.eval()
            self._validate_pipeline()
            self.load_runtime = self._runtime(start)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise ForecastError(f"Chronos-2 load failed: {error}") from error

    def _validate_device(self) -> None:
        index = int(self.settings.device.split(":")[1])
        if (
            not self._torch.cuda.is_available()
            or index >= self._torch.cuda.device_count()
        ):
            raise ForecastError("requested CUDA device is unavailable; no CPU fallback")

    def _validate_pipeline(self) -> None:
        model = self._pipeline.model
        if str(model.device) != self.settings.device or model.dtype != getattr(
            self._torch, self.settings.dtype
        ):
            raise ForecastError("upstream model differs from requested device/dtype")
        if not set(self.settings.quantile_levels) <= set(self._pipeline.quantiles):
            raise ForecastError("only native Chronos-2 quantiles are supported")

    def _runtime(self, start: float) -> RuntimeStats:
        cuda, device = self._torch.cuda, self.settings.device
        cuda.synchronize(device)
        result = RuntimeStats(
            seconds=perf_counter() - start,
            peak_allocated_bytes=cuda.max_memory_allocated(device),
            peak_reserved_bytes=cuda.max_memory_reserved(device),
        )
        if result.peak_reserved_bytes > self.settings.max_vram_bytes:
            raise ForecastError("Chronos-2 exceeded the configured VRAM budget")
        return result

    def forecast(self, batch: ForecastBatch) -> QuantileForecast:
        """Jointly predict channels within each window, never across trajectories."""
        self._validate_request(batch)
        torch, settings = self._torch, self.settings
        device_index = int(settings.device.split(":")[1])
        try:
            torch.cuda.synchronize(settings.device)
            torch.cuda.reset_peak_memory_stats(settings.device)
            start = perf_counter()
            with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
                torch.random.default_generator.manual_seed(settings.seed)
                with torch.cuda.device(device_index):
                    torch.cuda.manual_seed(settings.seed)
                quantiles, _ = self._pipeline.predict_quantiles(
                    np.transpose(batch.context, (0, 2, 1)).copy(),
                    prediction_length=batch.spec.horizon_frames,
                    quantile_levels=list(settings.quantile_levels),
                    batch_size=settings.batch_size,
                    context_length=batch.spec.context_frames,
                    cross_learning=False,
                    limit_prediction_length=True,
                )
            values = np.stack([item.detach().cpu().numpy() for item in quantiles])
            return QuantileForecast(
                batch.spec,
                batch.indices,
                settings.quantile_levels,
                np.asarray(values.transpose(0, 2, 1, 3), dtype=np.float64),
                self._runtime(start),
            )
        except (ValueError, RuntimeError) as error:
            raise ForecastError(f"Chronos-2 inference failed: {error}") from error

    def _validate_request(self, batch: ForecastBatch) -> None:
        if batch.spec.context_frames > self._pipeline.model_context_length:
            raise ForecastError("context exceeds native Chronos-2 limit; no truncation")
        if batch.spec.horizon_frames > self._pipeline.model_prediction_length:
            raise ForecastError("horizon exceeds native Chronos-2 limit; no unrolling")
        if len(batch.spec.feature_ids) > self.settings.batch_size:
            raise ForecastError(
                "scalar batch budget cannot fit one multivariate window"
            )

    def predict(self, batch: ForecastBatch) -> FloatArray:
        """Return the explicit median view required by the baseline point protocol."""
        return self.forecast(batch).median
