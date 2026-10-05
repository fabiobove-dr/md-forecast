"""Export matched geometric zero-shot and fine-tuned predictions and select LR."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from finetune_followup_geometry import AdaptationConfig, trial_name
from run_followup_diagnostics import diagnostic_tables, load_native
from summarize_followup_diagnostics import error_summary, spread_summary

from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.preprocessing import SeriesLoader
from md_forecast.data.splits import SplitManifest
from md_forecast.data.windows import WindowConfig, iter_windows
from md_forecast.evaluation.metrics import quantile_losses
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.finetuning import (
    FineTunedChronos2Adapter,
    FineTuneResult,
    TrainingExposure,
    read_checkpoint,
)
from md_forecast.models.learned import NLinearState


def exposure(path: Path) -> dict[str, object]:
    """Count actual consumed inputs against the verified frozen source window order."""
    checkpoint = read_checkpoint(path)
    log = read_metadata(path / "exposure.json", TrainingExposure)
    indices = list(
        iter_windows(checkpoint.manifest.split, checkpoint.manifest.grid, Split.TRAIN)
    )
    if (
        len(indices) != len(log.window_counts)
        or log.input_hash != checkpoint.manifest.input_hash
    ):
        raise DataContractError("exposure differs from checkpoint inputs")
    seen = [
        index for index, count in zip(indices, log.window_counts, strict=True) if count
    ]
    return {
        "step": checkpoint.step,
        "forward_batches": log.forward_batches,
        "window_draws": sum(log.window_counts),
        "unique_windows": len(seen),
        "unique_systems": len({i.system_id for i in seen}),
        "unique_trajectories": len({i.trajectory_id for i in seen}),
        "earlier_unknown_optimizer_steps": log.earlier_unknown_optimizer_steps,
    }


def cell(
    root: Path,
    split: SplitManifest,
    loader: SeriesLoader,
    grid: WindowConfig,
    model: Chronos2Adapter,
    scale: tuple[float, ...],
) -> dict[str, object]:
    """Keep all points, quantiles and errors on matched validation contexts."""
    digest = hashlib.sha256()
    runtimes = []
    summaries = []
    leads = []
    for batch, targets in iter_forecasts(
        split,
        grid,
        Split.VALIDATION,
        BASE_FEATURE_IDS,
        loader,
        batch_size=32,
        with_targets=True,
    ):
        assert targets is not None
        digest.update(
            json.dumps(
                [(i.trajectory_id, i.start) for i in batch.indices],
                separators=(",", ":"),
            ).encode()
        )
        digest.update(batch.context.astype("<f8").tobytes())
        digest.update(targets.astype("<f8").tobytes())
        forecast = model.forecast(batch)
        runtimes.append(forecast.runtime.model_dump(mode="json"))
        summary, lead = diagnostic_tables(batch, targets, forecast.median, model)
        column = summary.schema.get_field_index("spread_ratio")
        summary = summary.set_column(
            column,
            "spread_ratio",
            pa.array(summary["spread_ratio"].to_numpy(), from_pandas=True),
        )
        for index, level in enumerate(forecast.quantile_levels):
            lead = lead.append_column(
                f"quantile-{level}", pa.array(forecast.values[..., index].ravel())
            )
        losses = quantile_losses(
            forecast.values, targets, forecast.quantile_levels, ((0.1, 0.9),)
        )
        for (metric, lower, upper), values in losses.items():
            lead = lead.append_column(
                f"{metric}-{lower}-{upper}", pa.array(values.ravel())
            )
        summaries.append(summary)
        leads.append(lead)
    windows = pa.concat_tables(summaries)
    future = pa.concat_tables(leads)
    pq.write_table(windows, root / "windows.parquet")
    pq.write_table(future, root / "leads.parquet")
    errors = error_summary(future, leads=False)
    scales = dict(zip(BASE_FEATURE_IDS, scale, strict=True))
    score = float(
        np.mean([cast(float, r["mae"]) / scales[str(r["feature_id"])] for r in errors])
    )
    return {
        "input_and_label_hash": "sha256:" + digest.hexdigest(),
        "model_config": model.config.model_dump(mode="json"),
        "model_artifact_hash": model.artifact_hash,
        "errors": errors,
        "per_lead": error_summary(future, leads=True),
        "spread": spread_summary(windows),
        "normalized_mae": score,
        "runtime": runtimes,
    }


def evaluate(args: argparse.Namespace) -> None:
    """Finish planned candidates and choose LR across fixed repeat seeds."""
    if args.output.exists():
        raise DataContractError("output exists; choose fresh evaluation")
    config = read_metadata(args.root / "plan.json", AdaptationConfig)
    split, loader = load_native(args.native)
    trials = [
        trial_name(lr, seed) for lr in config.learning_rates for seed in config.seeds
    ]
    results = {
        name: read_metadata(args.root / "runs" / name / "result.json", FineTuneResult)
        for name in trials
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".fine-evaluation-"
    ) as temporary:
        root = Path(temporary)
        models = {}
        zero_digests = {}
        scores = {}
        for name in ["zero-shot", *trials]:
            if name == "zero-shot":
                model = Chronos2Adapter(
                    config.settings, cache_dir=Path("data/cache/chronos2")
                )
            else:
                checkpoint = (
                    args.root
                    / "runs"
                    / name
                    / f"checkpoint-{results[name].selected_step}"
                )
                saved = read_checkpoint(checkpoint)
                if metadata_hash(saved) != results[name].selected_checkpoint_hash:
                    raise DataContractError("selected checkpoint differs from result")
                model = FineTunedChronos2Adapter(
                    config.settings, checkpoint_dir=checkpoint
                )
            cells = {}
            directory = root / name
            directory.mkdir()
            for context in config.evaluation_grid.contexts:
                for horizon in config.evaluation_grid.horizons:
                    grid = config.evaluation_grid.model_copy(
                        update={"contexts": (context,), "horizons": (horizon,)}
                    )
                    key = f"c{int(context)}-h{int(horizon)}"
                    target = directory / key
                    target.mkdir()
                    compact = read_metadata(
                        args.baselines / key / "compact-state.json", NLinearState
                    )
                    if (compact.spec.split_hash, compact.spec.window_config_hash) != (
                        metadata_hash(split),
                        metadata_hash(grid),
                    ):
                        raise DataContractError(
                            "baseline scales differ from matched source/grid"
                        )
                    row = cell(target, split, loader, grid, model, compact.scaler.scale)
                    if name == "zero-shot":
                        zero_digests[key] = row["input_and_label_hash"]
                    elif zero_digests[key] != row["input_and_label_hash"]:
                        raise DataContractError("fine/zero contexts or labels differ")
                    cells[key] = row
            models[name] = {
                "cells": cells,
                "model_artifact_hash": model.artifact_hash,
                "normalized_mae": float(
                    np.mean([cast(float, r["normalized_mae"]) for r in cells.values()])
                ),
            }
            if name != "zero-shot":
                last = (
                    args.root
                    / "runs"
                    / name
                    / f"checkpoint-{results[name].completed_steps}"
                )
                models[name] |= {
                    "result": results[name].model_dump(mode="json"),
                    "selected_exposure": exposure(checkpoint),
                    "completed_exposure": exposure(last),
                    "learning_curve": json.loads(
                        (last / "trainer_state.json").read_text()
                    )["log_history"],
                }
            scores[name] = cast(float, models[name]["normalized_mae"])
            print(name, scores[name], flush=True)
            del model
            gc.collect()
        candidate_scores = {
            str(lr): {
                "mean": float(
                    np.mean([scores[trial_name(lr, seed)] for seed in config.seeds])
                ),
                "seed_std_ddof1": float(
                    np.std(
                        [scores[trial_name(lr, seed)] for seed in config.seeds], ddof=1
                    )
                ),
            }
            for lr in config.learning_rates
        }
        selected_lr = min(
            config.learning_rates,
            key=lambda lr: (candidate_scores[str(lr)]["mean"], lr),
        )
        selected = trial_name(selected_lr, config.representative_seed)
        frozen = {
            "plan_hash": metadata_hash(config),
            "native_split_hash": metadata_hash(split),
            "candidate_scores": candidate_scores,
            "selected_trial": selected,
            "selected_checkpoint_hash": results[selected].selected_checkpoint_hash,
            "selected_model_artifact_hash": models[selected]["model_artifact_hash"],
            "models": models,
            "files_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        }
        (root / "selection.json").write_text(
            json.dumps(frozen, indent=2, allow_nan=False) + "\n"
        )
        root.rename(args.output)
    print("selected", selected, results[selected].selected_checkpoint_hash)


def main() -> None:
    """Evaluate the completed native plan and publish separate immutable output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/processed/followup-finetuning41-verified"),
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("data/processed/followup-baselines40-expanded"),
    )
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
