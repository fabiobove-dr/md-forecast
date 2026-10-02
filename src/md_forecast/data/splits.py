"""Deterministic official/grouped partitions established before window generation."""

import math
import random
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from md_forecast.core.constants import (
    CANONICAL_SCHEMA_VERSION,
    DEFAULT_GROUP_FIELD,
    RATIO_ATOL,
    Split,
)
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.registry import Registry
from md_forecast.data.schemas import (
    BoundaryModel,
    Identifier,
    SchemaVersion,
    TrajectoryManifest,
)

GROUP_FIELDS = frozenset(
    {"system_id", "split_group_id", "pdb_id", "protein_id", "ligand_id"}
)
type GroupLabels = dict[str, str]


class SplitConfig(BoundaryModel):
    """Explicit seed and ratios; custom group fields use supplied identity labels."""

    mode: Literal["official", "grouped"]
    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    seed: Annotated[int, Field(strict=True, ge=0)]
    group_field: Identifier = DEFAULT_GROUP_FIELD
    ratios: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        """Official assignments must not be silently resampled with ratios."""
        if self.mode == "official":
            if self.ratios is not None:
                raise ValueError("official splits do not accept ratios")
        else:
            _validate_ratios(self.ratios)
        return self


def _validate_ratios(ratios: tuple[float, float, float] | None) -> None:
    if ratios is None:
        raise ValueError("grouped splits require train/validation/test ratios")
    _validate_ratio_values(ratios)


def _validate_ratio_values(ratios: tuple[float, float, float]) -> None:
    if any(value < 0 or value > 1 for value in ratios):
        raise ValueError("split ratios must be between zero and one")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=RATIO_ATOL):
        raise ValueError("split ratios must sum to one")


class SplitAssignment(BoundaryModel):
    """One trajectory belongs to exactly one indivisible dependency component."""

    trajectory_id: Identifier
    group_id: Identifier
    split: Split


class SplitManifest(BoundaryModel):
    """Self-contained source registry, group labels, and verified assignments."""

    schema_version: SchemaVersion = CANONICAL_SCHEMA_VERSION
    registry: Registry
    config: SplitConfig
    group_labels: GroupLabels
    assignments: Annotated[tuple[SplitAssignment, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_assignments(self) -> Self:
        """Recompute the declared protocol; do not trust serialized split labels."""
        _validate_labels(self.registry, self.config, self.group_labels)
        expected = _assignments(self.registry, self.config, self.group_labels)
        if self.assignments != expected:
            raise ValueError(
                "split assignments violate the declared grouping/seed/protocol"
            )
        return self


def _validate_labels(
    registry: Registry, config: SplitConfig, labels: GroupLabels
) -> None:
    _check_label_coverage(registry, labels)
    if config.group_field in GROUP_FIELDS:
        if labels != _source_labels(registry, config.group_field):
            raise ValueError("group labels differ from canonical source identity")


def _check_label_coverage(registry: Registry, labels: GroupLabels) -> None:
    ids = {record.trajectory_id for record in registry.trajectories}
    if set(labels) != ids:
        raise ValueError("group labels must exactly cover all trajectories")
    for value in labels.values():
        _check_group_label(value)


def _check_group_label(value: str) -> None:
    if not value or value != value.strip():
        raise ValueError("group labels must contain nonempty exact identities")


def _source_labels(registry: Registry, field: str) -> GroupLabels:
    labels: GroupLabels = {}
    for record in registry.trajectories:
        label = getattr(record, field)
        if label is None:
            raise ValueError(f"missing {field} for trajectory {record.trajectory_id}")
        labels[record.trajectory_id] = label
    return labels


def _components(
    registry: Registry, labels: GroupLabels
) -> dict[str, list[TrajectoryManifest]]:
    # Union the three real dependency relations; no dense pairwise comparisons.
    parents = {
        record.trajectory_id: record.trajectory_id for record in registry.trajectories
    }
    first: dict[tuple[str, str], str] = {}
    for record in sorted(registry.trajectories, key=lambda item: item.trajectory_id):
        relations = (
            ("system", record.system_id),
            ("source", record.split_group_id),
            ("configured", labels[record.trajectory_id]),
        )
        for relation in relations:
            previous = first.setdefault(relation, record.trajectory_id)
            left, right = _root(parents, previous), _root(parents, record.trajectory_id)
            parents[max(left, right)] = min(left, right)
    groups: dict[str, list[TrajectoryManifest]] = {}
    for record in sorted(registry.trajectories, key=lambda item: item.trajectory_id):
        groups.setdefault(_root(parents, record.trajectory_id), []).append(record)
    return groups


def _root(parents: dict[str, str], node: str) -> str:
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def _partition_counts(
    count: int, ratios: tuple[float, float, float]
) -> tuple[int, ...]:
    quotas = [count * value for value in ratios]
    counts = [math.floor(value) for value in quotas]
    remainder = count - sum(counts)
    order = sorted(
        range(len(ratios)), key=lambda index: (-(quotas[index] - counts[index]), index)
    )
    for index in order[:remainder]:
        counts[index] += 1
    _check_partition_counts(ratios, counts)
    return tuple(counts)


def _check_partition_counts(
    ratios: tuple[float, float, float], counts: list[int]
) -> None:
    for ratio, size in zip(ratios, counts, strict=True):
        if ratio > 0 and size == 0:
            raise ValueError(
                "too few dependency groups for nonempty requested partitions"
            )


def _grouped_partitions(
    groups: dict[str, list[TrajectoryManifest]], config: SplitConfig
) -> dict[str, Split]:
    assert config.ratios is not None
    counts = _partition_counts(len(groups), config.ratios)
    keys = sorted(groups)
    random.Random(config.seed).shuffle(keys)
    partitions: dict[str, Split] = {}
    offset = 0
    for split, count in zip(Split, counts, strict=True):
        partitions.update(dict.fromkeys(keys[offset : offset + count], split))
        offset += count
    return partitions


def _official_partition(records: list[TrajectoryManifest]) -> Split:
    partitions = {record.split for record in records}
    if None in partitions or len(partitions) != 1:
        raise ValueError("official assignments are missing or cross a dependency group")
    partition = records[0].split
    assert partition is not None
    return partition


def _assignments(
    registry: Registry, config: SplitConfig, labels: GroupLabels
) -> tuple[SplitAssignment, ...]:
    groups = _components(registry, labels)
    if config.mode == "official":
        partitions = {
            key: _official_partition(records) for key, records in groups.items()
        }
    else:
        partitions = _grouped_partitions(groups, config)
    assignments = [
        SplitAssignment(
            trajectory_id=record.trajectory_id, group_id=key, split=partitions[key]
        )
        for key, records in groups.items()
        for record in records
    ]
    return tuple(sorted(assignments, key=lambda item: item.trajectory_id))


def build_split(
    registry: Registry,
    config: SplitConfig,
    group_labels: Mapping[str, str] | None = None,
) -> SplitManifest:
    """Bind official or seeded grouped partitions to the entire canonical registry.

    A custom group field requires an explicit trajectory-to-family mapping.
    System and source split-group dependencies remain enforced under every field.
    """
    ordered = Registry(
        dataset=registry.dataset,
        trajectories=tuple(
            sorted(registry.trajectories, key=lambda item: item.trajectory_id)
        ),
    )
    try:
        labels = _resolve_labels(ordered, config, group_labels)
        _validate_labels(ordered, config, labels)
        return SplitManifest(
            registry=ordered,
            config=config,
            group_labels=labels,
            assignments=_assignments(ordered, config, labels),
        )
    except ValueError as error:
        raise DataContractError(f"cannot build split: {error}") from error


def _resolve_labels(
    registry: Registry, config: SplitConfig, labels: Mapping[str, str] | None
) -> GroupLabels:
    if config.group_field in GROUP_FIELDS:
        if labels is not None:
            raise ValueError(
                "canonical identity fields cannot be overridden with external labels"
            )
        return _source_labels(registry, config.group_field)
    if labels is None:
        raise ValueError("custom group_field requires explicit group labels")
    return dict(labels)
