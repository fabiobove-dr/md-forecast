"""MISATO-only baseline selection and paired external complex-level effects."""

from typing import Annotated, Any, Self

import numpy as np
import pyarrow as pa
from pydantic import Field, model_validator

from md_forecast.core.constants import DatasetId, ModelId, Split
from md_forecast.core.exceptions import ForecastError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, Identifier
from md_forecast.data.splits import SplitConfig
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.benchmark import CellResult
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    group_interval,
    interval_status,
)
from md_forecast.models.chronos import ChronosConfig
from md_forecast.models.learned import TrainingConfig

STATISTICAL_MODELS = frozenset(
    {ModelId.PERSISTENCE, ModelId.CONTEXT_MEAN, ModelId.LINEAR, ModelId.AR, ModelId.VAR}
)


class ExternalConfig(BoundaryModel):
    """Freeze both tasks, calibration and budgets before any external forecast."""

    misato_split: SplitConfig
    replica_split: SplitConfig
    grid: WindowConfig
    evaluation: BenchmarkConfig
    chronos: ChronosConfig
    training: TrainingConfig
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    calibration_tolerance: Annotated[float, Field(gt=0, lt=1)]

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Keep validation selection and replica adaptation explicitly separate."""
        if self.misato_split.mode != "grouped" or self.misato_split.ratios != (
            0.7,
            0.3,
            0.0,
        ):
            raise ValueError(
                "common development requires MISATO TRAIN-only 70/30 holdout"
            )
        if self.replica_split.mode != "unseen-replica":
            raise ValueError("external replica task requires explicit replica protocol")
        return self


class BaselineChoice(BoundaryModel):
    """One strongest statistical configuration for one observable and lead."""

    feature_id: Identifier
    step: Annotated[int, Field(strict=True, ge=0)]
    model_hash: ArtifactHash


class BaselineSelection(BoundaryModel):
    """Frozen choices linked to the scored MISATO development report."""

    validation_report_hash: ArtifactHash
    choices: Annotated[tuple[BaselineChoice, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_cells(self) -> Self:
        """Do not allow ambiguous favorable baseline choices."""
        keys = [(c.feature_id, c.step) for c in self.choices]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate baseline selection cell")
        return self


def select_statistics(result: CellResult, report_hash: str) -> BaselineSelection:
    """Select per feature/lead MAE exclusively on MISATO validation complexes."""
    _require_misato_validation(result)
    allowed = _statistical_hashes(result)
    rows = [r for r in _mae_rows(result, "aggregate") if r["model_hash"] in allowed]
    choices = tuple(
        _baseline_choice(rows, feature, step)
        for feature, step in sorted(_cells(result))
    )
    return BaselineSelection(validation_report_hash=report_hash, choices=tuple(choices))


def selected_effects(
    result: CellResult, selection: BaselineSelection, model_hash: str
) -> pa.Table:
    """Average replicas within complexes and bootstrap paired complex differences."""
    rows = _mae_rows(result, "group")
    expected = _cells(result)
    if {(c.feature_id, c.step) for c in selection.choices} != expected:
        raise ForecastError("selection differs from the external feature/lead grid")
    output = [
        _selected_effect(rows, choice, model_hash, result.manifest.config)
        for choice in selection.choices
    ]
    return pa.Table.from_pylist(output)


def _selected_effect(
    rows: list[dict[str, Any]],
    choice: BaselineChoice,
    model_hash: str,
    config: BenchmarkConfig,
) -> dict[str, object]:
    cell = _cell_rows(rows, choice.feature_id, choice.step)
    model = _group_values(cell, model_hash)
    base = _group_values(cell, choice.model_hash)
    if not model or set(model) != set(base):
        raise ForecastError("selected baseline and model groups are not paired")
    groups = sorted(model)
    values = np.asarray([model[g] - base[g] for g in groups], dtype=np.float64)
    marginal = group_interval(values, config)
    corrected = group_interval(values, config, corrected=True)
    baseline = float(np.mean(list(base.values())))
    return dict(
        feature_id=choice.feature_id,
        step=choice.step,
        model_hash=model_hash,
        reference_hash=choice.model_hash,
        mae_difference=float(values.mean()),
        ratio_to_selected=None
        if baseline == 0
        else float(np.mean(list(model.values()))) / baseline,
        marginal_ci_lower=marginal[0],
        marginal_ci_upper=marginal[1],
        corrected_ci_lower=corrected[0],
        corrected_ci_upper=corrected[1],
        independent_groups=len(groups),
        uncertainty_status=interval_status(len(groups), config, corrected=True),
    )


def _cells(result: CellResult) -> set[tuple[str, int]]:
    return {
        (f, s)
        for f in result.manifest.spec.feature_ids
        for s in range(result.manifest.spec.horizon_frames + 1)
    }


def _mae_rows(result: CellResult, level: str) -> list[dict[str, Any]]:
    return [
        r
        for r in result.metrics.to_pylist()
        if (r["level"], r["metric"]) == (level, "mae")
    ]


def _statistical_hashes(result: CellResult) -> set[str]:
    return {
        metadata_hash(c)
        for c in result.manifest.models
        if c.model_id in STATISTICAL_MODELS
    }


def _cell_rows(
    rows: list[dict[str, Any]], feature: str, step: int
) -> list[dict[str, Any]]:
    return [r for r in rows if (r["feature_id"], r["step"]) == (feature, step)]


def _baseline_choice(
    rows: list[dict[str, Any]], feature: str, step: int
) -> BaselineChoice:
    candidates = _cell_rows(rows, feature, step)
    if not candidates:
        raise ForecastError("missing validation baseline cell")
    winner = min(candidates, key=lambda r: (r["value"], r["model_hash"]))
    return BaselineChoice(
        feature_id=feature, step=step, model_hash=winner["model_hash"]
    )


def _group_values(rows: list[dict[str, Any]], model: str) -> dict[str, float]:
    return {r["identity"]: r["value"] for r in rows if r["model_hash"] == model}


def _require_misato_validation(result: CellResult) -> None:
    manifest = result.manifest
    if (
        manifest.partition != Split.VALIDATION
        or manifest.spec.dataset.dataset_id != DatasetId.MISATO
    ):
        raise ForecastError("statistical selection requires MISATO VALIDATION")
