"""Plan independent-group precision from old development reports, never fresh tests."""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Annotated

import pyarrow.parquet as pq
from pydantic import Field

from md_forecast.core.constants import ModelId
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.cohort import CohortConfig, ExternalReserve
from md_forecast.data.schemas import BoundaryModel
from md_forecast.evaluation.report import read_benchmark


class PrecisionPlan(BoundaryModel):
    """Approximate power assumptions, separately frozen from final inference."""

    confidence: Annotated[float, Field(gt=0, lt=1)]
    power: Annotated[float, Field(gt=0.5, lt=1)]
    comparisons: Annotated[int, Field(strict=True, gt=0)]
    worthwhile_mae_reduction_fraction: Annotated[float, Field(gt=0, lt=1)]
    paired_sd_inflation: Annotated[float, Field(ge=1)]


def planning(root: Path, groups: int, config: PrecisionPlan) -> dict[str, object]:
    """Use equal-weight group MAEs for the historical C40/H10 contact-count pilot."""
    report = read_benchmark(root)
    cell = report.manifest.cells[0]
    if len(report.manifest.cells) != 1 or (
        cell.spec.context_frames,
        cell.spec.horizon_frames,
    ) != (40, 10):
        raise DataContractError("precision planning requires the frozen C40/H10 pilot")
    hashes = {m.model_id: metadata_hash(m) for m in cell.models}
    metrics = pq.read_table(root / "cell-0/metrics.parquet").to_pylist()
    samples = {
        model: {
            row["identity"]: float(row["value"])
            for row in metrics
            if (
                row["model_hash"],
                row["level"],
                row["feature_id"],
                row["metric"],
                row["step"],
            )
            == (hashes[model], "group", "protein_ligand_contact_count", "mae", 0)
        }
        for model in (ModelId.AR, ModelId.CHRONOS2)
    }
    base, candidate = samples[ModelId.AR], samples[ModelId.CHRONOS2]
    if base.keys() != candidate.keys() or len(base) < 2:
        raise DataContractError(
            "planning requires matched independent development groups"
        )
    baseline = mean(base.values())
    sd = stdev(candidate[key] - base[key] for key in sorted(base))
    difference = baseline * config.worthwhile_mae_reduction_fraction
    z = NormalDist().inv_cdf(1 - (1 - config.confidence) / (2 * config.comparisons))
    z_power = NormalDist().inv_cdf(config.power)
    inflated = sd * config.paired_sd_inflation
    return {
        "report_hash": report.artifact_id,
        "pilot_groups": len(base),
        "baseline_mean_mae": baseline,
        "paired_sd": sd,
        "normal_approximation_groups": math.ceil(
            ((z + z_power) * sd / difference) ** 2
        ),
        "inflated_normal_approximation_groups": math.ceil(
            ((z + z_power) * inflated / difference) ** 2
        ),
        "planned_independent_groups": groups,
        "inflated_ci_halfwidth_fraction_of_baseline": z
        * inflated
        / math.sqrt(groups)
        / baseline,
        "limitations": "Pilot variance and normal approximation are uncertain; "
        "these are planning estimates, not measured intervals or guaranteed power. "
        "Freeze the final comparison family and resolvable resampling separately.",
    }


def main() -> None:
    """Read integrity-checked historical reports and write ignored planning output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/followup-precision.json"),
    )
    parser.add_argument(
        "--native-cohort",
        type=Path,
        default=Path("configs/datasets/followup-cohort.json"),
    )
    parser.add_argument(
        "--external-reserve",
        type=Path,
        default=Path("configs/datasets/followup-external-reserve.json"),
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=Path("data/processed/mdbind-common-benchmark/misato-validation"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/processed/mdbind-common-benchmark/mdbind-unseen-complex"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PrecisionPlan.model_validate_json(args.config.read_text())
    native = CohortConfig.model_validate_json(args.native_cohort.read_text())
    external = ExternalReserve.model_validate_json(args.external_reserve.read_text())
    output = {
        "planning_config": config.model_dump(mode="json"),
        "native": planning(args.native, native.counts["confirmation"], config),
        "external": planning(args.external, len(external.confirmation), config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
