"""Fit TRAIN residual references, compare native VAL, freeze confirmation inference."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finetune_followup_geometry import AdaptationConfig
from pydantic import Field, model_validator
from run_followup_diagnostics import diagnostic_tables, load_native
from select_followup_baselines import BaselineSelectionConfig
from summarize_followup_diagnostics import error_summary, spread_summary

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.schemas import ArtifactHash, BoundaryModel
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.ablations import observed_input
from md_forecast.evaluation.metrics import (
    BenchmarkConfig,
    group_interval,
    interval_status,
    quantile_losses,
)
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.base import ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.finetuning import (
    FineTunedChronos2Adapter,
    FineTuneResult,
    read_checkpoint,
)
from md_forecast.models.residuals import (
    ResidualBaseline,
    ResidualConfig,
    fit_residual_quantiles,
)


class ProbabilityConfig(BoundaryModel):
    """Separate development summaries from a completely prospective primary family."""

    grid: WindowConfig
    primary: WindowConfig
    residuals: ResidualConfig
    development: BenchmarkConfig
    confirmation: BenchmarkConfig
    comparison_counts: dict[str, Annotated[int, Field(strict=True, gt=0)]]
    expected_groups: dict[str, Annotated[int, Field(strict=True, gt=0)]]
    coverage_tolerance: Annotated[float, Field(gt=0, lt=1)]
    worthwhile_reduction: Annotated[float, Field(gt=0, lt=1)]
    input_selection_sha256: ArtifactHash
    baseline_provenance_sha256: ArtifactHash

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        """Reject unresolvable tails or a primary cell outside the saved full grid."""
        if (
            len(self.primary.contexts) != 1
            or len(self.primary.horizons) != 1
            or self.primary.horizons[0] < 2
        ):
            raise ValueError("primary inference requires one nontrivial horizon cell")
        if not set(self.primary.contexts) <= set(self.grid.contexts) or not set(
            self.primary.horizons
        ) <= set(self.grid.horizons):
            raise ValueError("primary cell must belong to the full descriptive grid")
        if (
            sum(self.comparison_counts.values())
            != self.confirmation.comparison_family_size
        ):
            raise ValueError("prospective family count differs from all primary claims")
        for task in ("native", "external"):
            groups = self.expected_groups[task]
            if (
                interval_status(groups, self.confirmation, corrected=True)
                != "descriptive-bootstrap"
            ):
                raise ValueError("prospective corrected tails/groups are unresolved")
            if (
                self.confirmation.bootstrap_samples * groups * 16
                > self.confirmation.max_bootstrap_bytes
            ):
                raise ValueError("prospective group bootstrap exceeds memory budget")
        return self


def check_file(path: Path, expected: str) -> None:
    """Verify the frozen development choice before reading its results."""
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise DataContractError("frozen selection/provenance file changed")


def probabilistic_summary(
    table: pa.Table, config: BenchmarkConfig, *, leads: bool
) -> list[dict[str, object]]:
    """Average trajectories within systems; windows/points are never bootstrap units."""
    keys = ["lead"] if leads else []
    columns = [
        c
        for c in table.column_names
        if c.startswith(("pinball-", "coverage-", "width-"))
    ]
    trajectory = table.group_by(keys + ["group_id", "trajectory_id"]).aggregate(
        [(c, "mean") for c in columns]
    )
    group = trajectory.group_by(keys + ["group_id"]).aggregate(
        [(c + "_mean", "mean") for c in columns]
    )
    rows = group.to_pylist()
    result = []
    for lead in sorted({r.get("lead", 0) for r in rows}):
        selected = [r for r in rows if r.get("lead", 0) == lead]
        means = {
            c: float(np.mean([r[c + "_mean_mean"] for r in selected])) for c in columns
        }
        bounds = {
            c: group_interval(
                np.asarray([r[c + "_mean_mean"] for r in selected]), config
            )
            for c in columns
        }
        result.append(
            {
                "lead": lead,
                "independent_groups": len(selected),
                "metrics": means,
                "marginal_ci": bounds,
                "interval_status": interval_status(len(selected), config),
            }
        )
    return result


def foundation(
    args: argparse.Namespace,
    target: str,
    method: str,
    selection: dict[str, Any],
    settings: AdaptationConfig,
) -> Chronos2Adapter:
    """Use the fixed representative seed/checkpoint/input choice from #42."""
    if method == "zero-shot":
        return Chronos2Adapter(settings.settings, cache_dir=Path("data/cache/chronos2"))
    choice = selection["selected_inputs"][target]
    condition = choice["condition"]
    seed = choice["representative_seed"]
    trial = (
        f"target{BASE_FEATURE_IDS.index(target)}-s{seed}"
        if condition == "target-only"
        else f"lr1e-05-s{seed}"
    )
    root = args.inputs if condition == "target-only" else args.adaptation
    result = read_metadata(root / "runs" / trial / "result.json", FineTuneResult)
    path = root / "runs" / trial / f"checkpoint-{result.selected_step}"
    expected = selection["results"][target][f"fine-{seed}"][condition][
        "training_identity"
    ]["checkpoint_hash"]
    if metadata_hash(read_checkpoint(path)) != expected:
        raise DataContractError("selected input checkpoint changed")
    return FineTunedChronos2Adapter(settings.settings, checkpoint_dir=path)


def run(args: argparse.Namespace) -> None:
    """Publish real validation comparisons without opening calibration/confirmation."""
    config = ProbabilityConfig.model_validate_json(args.config.read_text())
    if args.output.exists():
        raise DataContractError("output exists; preserve the previous bundle")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
        text=True,
    ).strip():
        raise DataContractError("commit experiment source before residual fitting")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    selection_file = args.validation / "selection.json"
    baseline_file = args.baselines / "provenance.json"
    check_file(selection_file, config.input_selection_sha256)
    check_file(baseline_file, config.baseline_provenance_sha256)
    selection = json.loads(selection_file.read_text())
    baseline = json.loads(baseline_file.read_text())
    candidates = read_metadata(args.baselines / "config.json", BaselineSelectionConfig)
    statistics = {metadata_hash(c): c for c in candidates.statistics}
    settings = read_metadata(args.adaptation / "plan.json", AdaptationConfig)
    split, loader = load_native(args.native)
    if metadata_hash(split) != baseline["split_hash"]:
        raise DataContractError("probability source differs from baseline selection")
    # Verify saved aligned predictions before trusting the published input choice.
    for rel, expected in selection["files_sha256"].items():
        path = (args.validation / rel).resolve()
        if (
            not path.is_relative_to(args.validation.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise DataContractError("saved input comparison file differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".probability-"
    ) as temporary:
        root = Path(temporary)
        write_metadata(root / "config.json", config)
        results: dict[str, Any] = {}
        states: dict[str, object] = {}
        for context in config.grid.contexts:
            for horizon in config.grid.horizons:
                grid = config.grid.model_copy(
                    update={"contexts": (context,), "horizons": (horizon,)}
                )
                cell = f"c{int(context)}-h{int(horizon)}"
                directory = root / cell
                directory.mkdir()
                selected = baseline["cells"][cell]["best_statistical_per_feature"]
                persistence = ModelConfig(model_id=ModelId.PERSISTENCE, seed=42)
                unique = {metadata_hash(persistence): persistence} | {
                    key: statistics[key] for key in set(selected.values())
                }
                residual_models = {}
                for key, point_config in unique.items():
                    state = fit_residual_quantiles(
                        split,
                        grid,
                        BASE_FEATURE_IDS,
                        loader,
                        StatisticalBaseline(point_config),
                        config.residuals,
                    )
                    filename = directory / (key.split(":")[1] + ".json")
                    write_metadata(filename, state)
                    residual_models[key] = ResidualBaseline(state)
                states[cell] = {
                    key: model.artifact_hash for key, model in residual_models.items()
                }
                results[cell] = {}
                for target in BASE_FEATURE_IDS:
                    condition = selection["selected_inputs"][target]["condition"]
                    results[cell][target] = {}
                    reference = None
                    for method in (
                        "persistence",
                        "selected-statistic",
                        "zero-shot",
                        "fine-tuned",
                    ):
                        model: ResidualBaseline | Chronos2Adapter
                        if method == "persistence":
                            model = residual_models[metadata_hash(persistence)]
                        elif method == "selected-statistic":
                            model = residual_models[selected[target]]
                        else:
                            model = foundation(
                                args, target, method, selection, settings
                            )
                        windows = []
                        leads = []
                        runtimes = []
                        for full, labels in iter_forecasts(
                            split,
                            grid,
                            Split.VALIDATION,
                            BASE_FEATURE_IDS,
                            loader,
                            batch_size=32,
                            with_targets=True,
                        ):
                            assert labels is not None
                            batch = (
                                full
                                if isinstance(model, ResidualBaseline)
                                else observed_input(
                                    full, target, condition, shift_frames=0
                                )
                            )
                            truth = (
                                labels
                                if len(batch.spec.feature_ids) == 3
                                else labels[
                                    :,
                                    :,
                                    BASE_FEATURE_IDS.index(
                                        target
                                    ) : BASE_FEATURE_IDS.index(target) + 1,
                                ]
                            )
                            forecast = model.forecast(batch)
                            runtimes.append(forecast.runtime.model_dump(mode="json"))
                            window, lead = diagnostic_tables(
                                batch, truth, forecast.median, model
                            )
                            for i, level in enumerate(forecast.quantile_levels):
                                lead = lead.append_column(
                                    f"quantile-{level}",
                                    pa.array(forecast.values[..., i].ravel()),
                                )
                            for (metric, lower, upper), values in quantile_losses(
                                forecast.values,
                                truth,
                                forecast.quantile_levels,
                                ((0.1, 0.9),),
                            ).items():
                                lead = lead.append_column(
                                    f"{metric}-{lower}-{upper}",
                                    pa.array(values.ravel()),
                                )
                            window = window.filter(
                                pc.equal(window["feature_id"], target)
                            )
                            window = window.set_column(
                                window.schema.get_field_index("spread_ratio"),
                                "spread_ratio",
                                pa.array(
                                    window["spread_ratio"].to_numpy(), from_pandas=True
                                ),
                            )
                            windows.append(window)
                            leads.append(
                                lead.filter(pc.equal(lead["feature_id"], target))
                            )
                        window, lead = (
                            pa.concat_tables(windows),
                            pa.concat_tables(leads),
                        )
                        aligned = lead.select(
                            [
                                "trajectory_id",
                                "group_id",
                                "start",
                                "lead",
                                "origin",
                                "target",
                            ]
                        )
                        if reference is None:
                            reference = aligned
                        elif not reference.equals(aligned):
                            raise DataContractError(
                                "probabilistic windows or labels differ"
                            )
                        name = f"{target}-{method}"
                        pq.write_table(window, directory / (name + "-windows.parquet"))
                        pq.write_table(lead, directory / (name + "-leads.parquet"))
                        results[cell][target][method] = {
                            "artifact_hash": model.artifact_hash,
                            "point_config": model.config.model_dump(mode="json"),
                            "input_features": batch.spec.feature_ids,
                            "errors": error_summary(lead, leads=False),
                            "per_lead_error": error_summary(lead, leads=True),
                            "spread": spread_summary(window),
                            "probability": probabilistic_summary(
                                lead, config.development, leads=False
                            ),
                            "per_lead_probability": probabilistic_summary(
                                lead, config.development, leads=True
                            ),
                            "runtime": runtimes,
                        }
                        print(
                            cell,
                            target,
                            method,
                            results[cell][target][method]["probability"][0]["metrics"],
                            flush=True,
                        )
                        del model
                        gc.collect()
        files = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "config_hash": metadata_hash(config),
                    "code_commit": code,
                    "lockfile_sha256": hashlib.sha256(
                        Path("uv.lock").read_bytes()
                    ).hexdigest(),
                    "split_hash": metadata_hash(split),
                    "selected_inputs": selection["selected_inputs"],
                    "residual_state_hashes": states,
                    "results": results,
                    "prospective_resolution": {
                        task: {
                            "groups": n,
                            "status": interval_status(
                                n, config.confirmation, corrected=True
                            ),
                            "estimated_bootstrap_bytes": n
                            * config.confirmation.bootstrap_samples
                            * 16,
                        }
                        for task, n in config.expected_groups.items()
                    },
                    "file_sha256": files,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        root.rename(args.output)


def main() -> None:
    """Run bounded development comparisons; prospective decisions remain untouched."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/followup-probability.json"),
    )
    p.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    p.add_argument(
        "--inputs", type=Path, default=Path("data/processed/followup-inputs42")
    )
    p.add_argument(
        "--validation",
        type=Path,
        default=Path("data/reports/followup-inputs42-validation"),
    )
    p.add_argument(
        "--baselines",
        type=Path,
        default=Path("data/processed/followup-baselines40-expanded"),
    )
    p.add_argument(
        "--adaptation",
        type=Path,
        default=Path("data/processed/followup-finetuning41-verified"),
    )
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
