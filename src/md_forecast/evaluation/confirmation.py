"""Frozen confirmation provenance and corrected, independent-system decisions."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import read_metadata
from md_forecast.data.schemas import ArtifactHash, BoundaryModel, DatasetConfig
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.metrics import BenchmarkConfig
from md_forecast.evaluation.overview import confined, file_hash
from md_forecast.models.chronos import ChronosConfig


class ConfirmationPlan(BoundaryModel):
    """Every choice fixed before confirmation outcomes, with portable source pins."""

    grid: WindowConfig
    primary: WindowConfig
    inference: BenchmarkConfig
    descriptive: BenchmarkConfig
    settings: ChronosConfig
    training_dataset: DatasetConfig
    native_cohort_hash: ArtifactHash
    external_reserve_hash: ArtifactHash
    seen_reserve_hash: ArtifactHash
    probability_config_hash: ArtifactHash
    input_conditions: dict[str, Literal["joint", "target-only"]]
    fine_checkpoints: dict[str, str]
    compact_states: dict[str, str]
    residual_states: dict[str, dict[str, str]]
    statistical_choices: dict[str, dict[str, ArtifactHash]]
    source_files: dict[str, ArtifactHash]
    expected_groups: dict[str, Annotated[int, Field(gt=0, strict=True)]]
    worthwhile_reduction: Annotated[float, Field(gt=0, lt=1)]
    nominal_coverage: Annotated[float, Field(ge=0.8, le=0.8)] = 0.8
    coverage_tolerance: Annotated[float, Field(gt=0, lt=0.2)]
    calibration_scope: Literal["TRAIN-residuals-only; no confirmation refitting"]
    confirmation_state: Literal["no reserved outcomes decoded"]
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]

    @model_validator(mode="after")
    def complete_choices(self) -> Self:
        """Reject missing feature/cell bindings or a changed primary policy."""
        features = set(self.input_conditions)
        if features != set(self.fine_checkpoints):
            raise ValueError(
                "fine-tuned checkpoints differ from declared input features"
            )
        _validate_reference_cells(self)
        self._validate_policy()
        return self

    def _validate_policy(self) -> None:
        if len(self.primary.contexts) != 1 or len(self.primary.horizons) != 1:
            raise ValueError("confirmation requires one predeclared primary cell")
        if self.primary.horizons[0] < 2:
            raise ValueError("primary confirmation horizon must be nontrivial")
        if self.inference.comparison_family_size != 126:
            raise ValueError(
                "confirmation must preserve the 126-interval primary family"
            )


def _validate_reference_cells(plan: ConfirmationPlan) -> None:
    cells = {
        f"c{int(c)}-h{int(h)}" for c in plan.grid.contexts for h in plan.grid.horizons
    }
    for reference in (
        plan.compact_states,
        plan.residual_states,
        plan.statistical_choices,
    ):
        if cells != set(reference):
            raise ValueError(
                "confirmation references must cover every frozen grid cell"
            )


def read_confirmation_plan(path: Path, root: Path = Path(".")) -> ConfirmationPlan:
    """Verify the immutable plan and every frozen input before reserved access."""
    plan = read_metadata(path, ConfirmationPlan)
    for relative, expected in plan.source_files.items():
        if file_hash(confined(root, relative)) != expected:
            raise DataContractError("frozen confirmation source changed: " + relative)
    return plan
