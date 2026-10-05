"""Score frozen models on three separate reserved tasks; never fit or select."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from run_followup_diagnostics import diagnostic_tables
from run_followup_probability import probabilistic_summary
from summarize_followup_diagnostics import error_summary, spread_summary

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.cohort import (
    CohortLoader,
    CohortManifest,
    ExternalReserve,
    permitted_external_sources,
)
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.public.mdbind import MDBindSubset
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.series import read_series
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.evaluation.ablations import observed_input
from md_forecast.evaluation.confirmation import read_confirmation_plan
from md_forecast.evaluation.metrics import quantile_losses
from md_forecast.evaluation.overview import confined, file_hash
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.base import ModelConfig
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.finetuning import FineTunedChronos2Adapter
from md_forecast.models.learned import NLinearModel, NLinearState
from md_forecast.models.residuals import ResidualBaseline, ResidualState


def load_task(root, task, plan):
    """Validate exact reserved identities before reading any canonical target table."""
    registry = read_registry(root / "registry.json")
    exported = json.loads((root / "qc.json").read_text())["exported"]
    if task == "native":
        cohort = read_metadata(root / "cohort.json", CohortManifest)
        if metadata_hash(cohort) != plan.native_cohort_hash:
            raise DataContractError("native confirmation cohort differs")
        if {record.pdb_id for record in registry.trajectories} != set(
            cohort.roles["confirmation"]
        ):
            raise DataContractError(
                "native registry is not exactly the reserved cohort"
            )
        loader = CohortLoader(root, registry, cohort, exported, purpose="confirmation")
        split = build_split(registry, SplitConfig(mode="official", seed=42))
    else:
        reserve = read_metadata(root / "reserve.json", ExternalReserve)
        expected = (
            plan.external_reserve_hash if task == "external" else plan.seen_reserve_hash
        )
        if metadata_hash(reserve) != expected:
            raise DataContractError("external confirmation reserve differs")
        subset = read_metadata(root / "source.json", MDBindSubset)
        sources = {
            source.accession: source
            for source in permitted_external_sources(subset, reserve, "confirmation")
        }
        if {record.source_record for record in registry.trajectories} != set(sources):
            raise DataContractError(
                "confirmation registry differs from permitted sources"
            )
        for record in registry.trajectories:
            source = sources[record.source_record]
            if (record.pdb_id, record.replicate_id, record.source_checksum) != (
                source.pdb_id,
                str(source.replica),
                source.files["trajectory.xtc"].checksum,
            ):
                raise DataContractError(
                    "confirmation record differs from frozen raw identity"
                )

        def loader(record):
            return read_series(
                confined(root, exported[record.trajectory_id]),
                Registry(dataset=registry.dataset, trajectories=(record,)),
            )

        split = build_split(
            registry, SplitConfig(mode="grouped", seed=42, ratios=(0.0, 0.0, 1.0))
        )
    count = len({assignment.group_id for assignment in split.assignments})
    if count != plan.expected_groups[task]:
        raise DataContractError("confirmation group count differs from the frozen plan")
    return split, loader


def run(args):
    """Publish all four cells and five fixed methods with aligned saved forecasts."""
    plan = read_confirmation_plan(args.plan)
    if args.output.exists():
        raise DataContractError("output exists; preserve the confirmation result")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
        text=True,
    ).strip():
        raise DataContractError("commit experiment source before confirmation scoring")
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    split, loader = load_task(args.input, args.task, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".confirmation-"
    ) as temporary:
        root = Path(temporary)
        write_metadata(root / "plan.json", plan)
        write_metadata(root / "split.json", split)
        results = {}
        for context in plan.grid.contexts:
            for horizon in plan.grid.horizons:
                grid = plan.grid.model_copy(
                    update={"contexts": (context,), "horizons": (horizon,)}
                )
                cell = f"c{int(context)}-h{int(horizon)}"
                directory = root / cell
                directory.mkdir()
                batches = list(
                    iter_forecasts(
                        split,
                        grid,
                        Split.TEST,
                        BASE_FEATURE_IDS,
                        loader,
                        batch_size=32,
                        with_targets=True,
                    )
                )
                spec = batches[0][0].spec
                results[cell] = {}
                for target in BASE_FEATURE_IDS:
                    results[cell][target] = {}
                    reference = None
                    for method in (
                        "persistence",
                        "selected-statistic",
                        "compact",
                        "zero-shot",
                        "fine-tuned",
                    ):
                        if method in ("persistence", "selected-statistic"):
                            key = (
                                metadata_hash(
                                    ModelConfig(model_id=ModelId.PERSISTENCE, seed=42)
                                )
                                if method == "persistence"
                                else plan.statistical_choices[cell][target]
                            )
                            state = read_metadata(
                                Path(plan.residual_states[cell][key]), ResidualState
                            )
                            model = ResidualBaseline(state, evaluation_spec=spec)
                        elif method == "compact":
                            model = NLinearModel(
                                read_metadata(
                                    Path(plan.compact_states[cell]), NLinearState
                                ),
                                evaluation_spec=spec,
                            )
                        elif method == "zero-shot":
                            model = Chronos2Adapter(
                                plan.settings, cache_dir=Path("data/cache/chronos2")
                            )
                        else:
                            evaluation = observed_input(
                                batches[0][0],
                                target,
                                plan.input_conditions[target],
                                shift_frames=0,
                            ).spec
                            model = FineTunedChronos2Adapter(
                                plan.settings,
                                checkpoint_dir=Path(plan.fine_checkpoints[target]),
                                evaluation_spec=evaluation,
                            )
                        windows, leads, runtimes = [], [], []
                        for full, labels in batches:
                            assert labels is not None
                            batch = (
                                observed_input(
                                    full,
                                    target,
                                    plan.input_conditions[target],
                                    shift_frames=0,
                                )
                                if method in ("zero-shot", "fine-tuned")
                                else full
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
                            started = perf_counter()
                            forecast = (
                                None if method == "compact" else model.forecast(batch)
                            )
                            points = (
                                model.predict(batch)
                                if forecast is None
                                else forecast.median
                            )
                            runtimes.append(
                                {"wall_seconds": perf_counter() - started}
                                if forecast is None
                                else forecast.runtime.model_dump(mode="json")
                            )
                            window, lead = diagnostic_tables(
                                batch, truth, points, model
                            )
                            if forecast is not None:
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
                                "confirmation models have unmatched windows or labels"
                            )
                        name = f"{target}-{method}"
                        pq.write_table(window, directory / (name + "-windows.parquet"))
                        pq.write_table(lead, directory / (name + "-leads.parquet"))
                        result = {
                            "artifact_hash": model.artifact_hash,
                            "point_config": model.config.model_dump(mode="json"),
                            "input_features": batch.spec.feature_ids,
                            "evaluation_spec": batch.spec.model_dump(mode="json"),
                            "errors": error_summary(lead, leads=False),
                            "per_lead_error": error_summary(lead, leads=True),
                            "spread": spread_summary(window),
                            "probability": []
                            if method == "compact"
                            else probabilistic_summary(
                                lead, plan.descriptive, leads=False
                            ),
                            "per_lead_probability": []
                            if method == "compact"
                            else probabilistic_summary(
                                lead, plan.descriptive, leads=True
                            ),
                            "runtime": runtimes,
                        }
                        results[cell][target][method] = result
                        print(
                            args.task,
                            cell,
                            target,
                            method,
                            result["errors"][0]["mae"],
                            flush=True,
                        )
                        del model
                        gc.collect()
        files = {
            str(path.relative_to(root)): file_hash(path).removeprefix("sha256:")
            for path in root.rglob("*")
            if path.is_file()
        }
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "config_hash": metadata_hash(plan),
                    "code_commit": code,
                    "lockfile_sha256": hashlib.sha256(
                        Path("uv.lock").read_bytes()
                    ).hexdigest(),
                    "split_hash": metadata_hash(split),
                    "task": args.task,
                    "training_dataset": plan.training_dataset.model_dump(mode="json"),
                    "evaluation_dataset": split.registry.dataset.model_dump(
                        mode="json"
                    ),
                    "independent_groups": plan.expected_groups[args.task],
                    "input_conditions": plan.input_conditions,
                    "frozen_residual_states": plan.residual_states,
                    "results": results,
                    "file_sha256": files,
                    "deviations": [],
                    "selection": (
                        "frozen before reserved outcomes; "
                        "no fitting or selection on TEST"
                    ),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        root.rename(args.output)


def main():
    """Keep native, untouched external and seen-system replica tasks separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/manifests/followup-confirmation45.json")
    )
    parser.add_argument(
        "--task", choices=("native", "external", "replica"), required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
