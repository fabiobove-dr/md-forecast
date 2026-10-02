"""Versioned, model-independent metadata validated once at I/O boundaries."""

from math import isclose
from typing import Annotated, Self, cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    CHECKSUM_PATTERN,
    PS_PER_NS,
    TIME_ATOL,
    TIME_COLUMN,
    TIME_RTOL,
    DatasetId,
    FeatureUnit,
    SamplingStatus,
    Split,
    TimeUnit,
)

type Identifier = Annotated[str, Field(min_length=1)]
type SchemaVersion = Annotated[
    int, Field(strict=True, ge=CANONICAL_SCHEMA_VERSION, le=CANONICAL_SCHEMA_VERSION)
]


class BoundaryModel(BaseModel):
    """Immutable boundary metadata; reject unknown fields and nonfinite numbers."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class Provenance(BoundaryModel):
    """Identity, integrity, and license of the actual input artifact."""

    dataset_id: DatasetId
    dataset_version: Identifier
    source_record: Identifier
    source_checksum: Annotated[str, Field(pattern=f"^{CHECKSUM_PATTERN}$")]
    license_id: Identifier
    provenance_uri: HttpUrl

    @model_validator(mode="after")
    def reject_credentials(self) -> Self:
        """Serialized provenance must not leak URL credentials."""
        if self.provenance_uri.username or self.provenance_uri.password:
            raise ValueError("provenance_uri cannot contain credentials")
        return self


class SystemIdentity(BoundaryModel):
    """Biological identity independent of trajectory replica or forecast model."""

    system_id: Identifier
    pdb_id: Identifier | None
    protein_id: Identifier | None
    ligand_id: Identifier | None


class TrajectoryIdentity(BoundaryModel):
    """One source trajectory and its indivisible split group."""

    trajectory_id: Identifier
    replicate_id: Identifier
    split_group_id: Identifier
    split: Split | None = None


class TimeAxis(BoundaryModel):
    """Relative time from the first retained frame, never a guessed MD origin."""

    frame_count: Annotated[int, Field(strict=True, gt=0)]
    time_unit: TimeUnit = TimeUnit.FRAME
    sampling_status: SamplingStatus = SamplingStatus.UNAVAILABLE
    frame_interval_ps: Annotated[float, Field(gt=0)] | None = None
    duration_ns: Annotated[float, Field(ge=0)] | None = None
    sampling_note: Identifier | None = None

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        """Keep missing physical time distinct from assumed/verified sampling."""
        if self.time_unit == TimeUnit.FRAME:
            _validate_frame_sampling(self)
        else:
            _validate_physical_sampling(self)
        return self


def _validate_frame_sampling(axis: TimeAxis) -> None:
    if (
        axis.sampling_status != SamplingStatus.UNAVAILABLE
        or axis.frame_interval_ps is not None
        or axis.duration_ns is not None
    ):
        raise ValueError("frame axes cannot claim physical sampling or duration")


def _validate_physical_sampling(axis: TimeAxis) -> None:
    if axis.sampling_status == SamplingStatus.UNAVAILABLE or axis.duration_ns is None:
        raise ValueError("physical axes require sampling status and duration_ns")
    if axis.sampling_status == SamplingStatus.ASSUMED and axis.sampling_note is None:
        raise ValueError(
            "assumed sampling requires a sampling_note explaining the assumption"
        )
    _validate_uniform_span(axis)


def _validate_uniform_span(axis: TimeAxis) -> None:
    if axis.frame_interval_ps is not None:
        span = (axis.frame_count - 1) * axis.frame_interval_ps / PS_PER_NS
        if not isclose(
            span, cast(float, axis.duration_ns), rel_tol=TIME_RTOL, abs_tol=TIME_ATOL
        ):
            raise ValueError("duration_ns differs from the retained frame span")


class TrajectoryManifest(Provenance, SystemIdentity, TrajectoryIdentity, TimeAxis):
    """Flat canonical record; numerical frame values live in Arrow, not here."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    feature_set_version: Identifier


class FeatureDefinition(BoundaryModel):
    """A semantically specified observable, not merely a column label."""

    feature_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    unit: FeatureUnit
    description: Identifier
    definition: Identifier

    @model_validator(mode="after")
    def reserve_time(self) -> Self:
        """Do not let a feature shadow the canonical time column."""
        if self.feature_id == TIME_COLUMN:
            raise ValueError("time is reserved for the time axis")
        return self


class DatasetConfig(BoundaryModel):
    """A versioned feature set shared by every trajectory in a registry."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    dataset_id: DatasetId
    dataset_version: Identifier
    feature_set_version: Identifier
    features: Annotated[tuple[FeatureDefinition, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_features(self) -> Self:
        """Keep column identities unambiguous."""
        names = [feature.feature_id for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature IDs")
        return self


class ExperimentConfig(BoundaryModel):
    """Minimal dataset selection and reproducibility boundary, without models."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    dataset: DatasetConfig
    seed: Annotated[int, Field(strict=True, ge=0)]
    trajectory_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_selection(self) -> Self:
        """Reject ambiguous repeated selections."""
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("duplicate selected trajectory IDs")
        return self


def stable_system_id(dataset_id: DatasetId, source_system_id: str) -> str:
    """Namespace exact source identity; adapters own case/alias normalization."""
    source = _identity_component(source_system_id)
    return f"{dataset_id.value}:system:{quote(source, safe='')}"


def stable_trajectory_id(
    system_id: str, source_trajectory_id: str, replicate_id: str
) -> str:
    """Encode every component to avoid delimiter collisions and process hashes."""
    components = (system_id, source_trajectory_id, replicate_id)
    return "trajectory:" + ":".join(
        quote(_identity_component(item), safe="") for item in components
    )


def _identity_component(value: str) -> str:
    """Validate source identifiers without changing their identity."""
    if not value or value != value.strip():
        raise ValueError(
            "identity components must be nonempty, without edge whitespace"
        )
    return value
