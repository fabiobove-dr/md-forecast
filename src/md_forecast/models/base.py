"""Explicit model configuration and the single context-only prediction protocol."""

from typing import Annotated, Protocol, Self

from pydantic import Field, model_validator

from md_forecast.core.constants import CANONICAL_SCHEMA_VERSION, ModelId
from md_forecast.data.forecast import ForecastBatch
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, SchemaVersion
from md_forecast.data.series import FloatArray


class ModelConfig(BoundaryModel):
    """Record deterministic baseline identity, seed and applicable hyperparameters."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    model_id: ModelId
    seed: Annotated[int, Field(strict=True, ge=0)]
    lags: Annotated[int, Field(strict=True, gt=0)] | None = None
    ridge: Annotated[float, Field(ge=0)] = 0.0
    adapter_config_hash: ArtifactHash | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_hyperparameters(self) -> Self:
        """Reject silently ignored options rather than mislabeling model variants."""
        _validate_lags(self)
        _validate_adapter_identity(self)
        if (
            self.model_id in (ModelId.PERSISTENCE, ModelId.CONTEXT_MEAN, ModelId.LINEAR)
            and self.ridge != 0
        ):
            raise ValueError("this baseline does not accept ridge regularization")
        return self


def _validate_adapter_identity(config: ModelConfig) -> None:
    if config.model_id == ModelId.CHRONOS2:
        if config.adapter_config_hash is None or config.ridge != 0:
            raise ValueError("Chronos-2 requires adapter settings hash and no ridge")
    elif config.adapter_config_hash is not None:
        raise ValueError("adapter settings hash is not applicable to this baseline")


def _validate_lags(config: ModelConfig) -> None:
    if config.model_id in (ModelId.AR, ModelId.VAR):
        if config.lags is None:
            raise ValueError("AR/VAR require an explicit lag order")
    elif config.lags is not None:
        raise ValueError("lag order only applies to AR/VAR")


class ForecastModel(Protocol):
    """Every model receives contexts only and returns native-unit (B,H,F) points."""

    @property
    def config(self) -> ModelConfig:
        """Recorded adapter configuration."""
        ...

    @property
    def artifact_hash(self) -> str:
        """Identity of fitted weights or the complete pretrained settings."""
        ...

    def predict(self, batch: ForecastBatch) -> FloatArray:
        """Predict the entire requested horizon without access to target labels."""
        ...
