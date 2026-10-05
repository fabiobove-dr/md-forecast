"""Apply the frozen 126-comparison primary family; retain every grid cell."""

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.series import FloatArray
from md_forecast.evaluation.confirmation import (
    ConfirmationPlan,
    confirmation_decision,
    paired_effect,
    read_confirmation_plan,
    system_metrics,
)
from md_forecast.evaluation.metrics import group_interval, interval_status
from md_forecast.evaluation.overview import confined, file_hash
from md_forecast.features.structural import BASE_FEATURE_IDS

METHODS = ("persistence", "selected-statistic", "compact", "zero-shot", "fine-tuned")
FOUNDATIONS = ("zero-shot", "fine-tuned")


def primary_effects(
    groups: dict[str, tuple[tuple[str, ...], dict[str, FloatArray]]],
    method: str,
    plan: ConfirmationPlan,
) -> dict[str, Any]:
    """Ten fixed intervals per foundation/feature/task; no selective omission."""
    effect = {"mae": {}, "worthwhile": {}, "pinball": {}}
    for reference in METHODS[:3]:
        effect["mae"][reference] = paired_effect(
            groups[method], groups[reference], "mae", plan.inference
        )
        effect["worthwhile"][reference] = paired_effect(
            groups[method],
            groups[reference],
            "mae",
            plan.inference,
            factor=1 - plan.worthwhile_reduction,
        )
    for reference in METHODS[:2]:
        effect["pinball"][reference] = paired_effect(
            groups[method], groups[reference], "mean-pinball", plan.inference
        )
    effect["width"] = paired_effect(
        groups[method], groups["selected-statistic"], "width-0.1-0.9", plan.inference
    )
    coverage = groups[method][1]["coverage-0.1-0.9"]
    effect["coverage"] = {
        "mean": float(coverage.mean()),
        "ci": group_interval(coverage, plan.inference, corrected=True),
        "interval_status": interval_status(
            len(coverage), plan.inference, corrected=True
        ),
        "independent_groups": len(coverage),
        "corrected": True,
    }
    effect["decision"] = confirmation_decision(
        effect, plan.nominal_coverage, plan.coverage_tolerance
    )
    return effect


def summarize_task(root: Path, plan: ConfirmationPlan) -> dict[str, Any]:
    """Verify every saved table, then average replicas before system resampling."""
    summary = json.loads((root / "summary.json").read_text())
    if summary["config_hash"] != metadata_hash(plan):
        raise DataContractError("confirmation bundle uses another frozen plan")
    for name, expected in summary["file_sha256"].items():
        if file_hash(confined(root, name)) != "sha256:" + expected:
            raise DataContractError("confirmation bundle integrity failed: " + name)
    primary = f"c{int(plan.primary.contexts[0])}-h{int(plan.primary.horizons[0])}"
    cells = {}
    for cell, features in summary["results"].items():
        cells[cell] = {}
        for feature in BASE_FEATURE_IDS:
            groups = {
                method: system_metrics(
                    pq.read_table(root / cell / f"{feature}-{method}-leads.parquet")
                )
                for method in METHODS
            }
            identities = {group[0] for group in groups.values()}
            if len(identities) != 1:
                raise DataContractError("confirmation methods have different groups")
            descriptive = {}
            for method in METHODS:
                metric = groups[method][1]
                descriptive[method] = {
                    "metrics": {
                        name: float(value.mean()) for name, value in metric.items()
                    },
                    "marginal_ci": {
                        name: group_interval(value, plan.descriptive)
                        for name, value in metric.items()
                    },
                    "independent_groups": len(groups[method][0]),
                    "mae_ratios": {
                        reference: float(
                            metric["mae"].mean() / groups[reference][1]["mae"].mean()
                        )
                        if groups[reference][1]["mae"].mean() > 0
                        else None
                        for reference in METHODS[:3]
                    },
                    "spread": features[feature][method]["spread"][0],
                }
            cells[cell][feature] = {"methods": descriptive, "primary": cell == primary}
            if cell == primary:
                cells[cell][feature]["corrected_effects"] = {
                    method: primary_effects(groups, method, plan)
                    for method in FOUNDATIONS
                }
                cells[cell][feature]["fine_minus_zero_mae"] = paired_effect(
                    groups["fine-tuned"], groups["zero-shot"], "mae", plan.inference
                )
            else:
                cells[cell][feature]["descriptive_mae_effects"] = {
                    method: {
                        reference: paired_effect(
                            groups[method],
                            groups[reference],
                            "mae",
                            plan.descriptive,
                            corrected=False,
                        )
                        for reference in METHODS[:3]
                    }
                    for method in FOUNDATIONS
                }
    primary_features = cells[primary]
    useful = any(
        data["corrected_effects"][method]["decision"]["useful_minimum"]
        for data in primary_features.values()
        for method in FOUNDATIONS
    )
    decision = "useful skill confirmed" if useful else "useful skill not established"
    return {
        "task": summary["task"],
        "independent_groups": summary["independent_groups"],
        "source_summary_hash": file_hash(root / "summary.json"),
        "split_hash": summary["split_hash"],
        "deviations": summary["deviations"],
        "primary_cell": primary,
        "decision": decision,
        "cells": cells,
        "corrected_interval_status": interval_status(
            summary["independent_groups"], plan.inference, corrected=True
        ),
    }


def main() -> None:
    """Produce a separate hash-pinned synthesis; never alter saved forecast bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/manifests/followup-confirmation45.json")
    )
    parser.add_argument(
        "--native",
        type=Path,
        default=Path("data/reports/followup-confirmation45-native"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/reports/followup-confirmation45-external"),
    )
    parser.add_argument(
        "--replica",
        type=Path,
        default=Path("data/reports/followup-confirmation45-replica"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise DataContractError("inference output exists; preserve previous synthesis")
    plan = read_confirmation_plan(args.plan)
    sources = {task: getattr(args, task) for task in ("native", "external", "replica")}
    tasks = {task: summarize_task(root, plan) for task, root in sources.items()}
    payload = {
        "config_hash": metadata_hash(plan),
        "frozen_plan": plan.model_dump(mode="json"),
        "source_summaries": {
            str(root / "summary.json"): file_hash(root / "summary.json")
            for root in sources.values()
        },
        "primary_family_size": 126,
        "family": (
            "native/external: 36 MAE + 36 worthwhile + 24 pinball "
            "+ 12 width + 12 coverage + 6 fine-minus-zero; "
            "replica descriptive/insufficient groups"
        ),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print({task: data["decision"] for task, data in tasks.items()})


if __name__ == "__main__":
    main()
