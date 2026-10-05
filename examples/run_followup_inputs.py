"""Freeze, train and score bounded native geometric context-input comparisons."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finetune_followup_geometry import AdaptationConfig, trial_name
from pydantic import Field, model_validator
from run_followup_diagnostics import diagnostic_tables, load_native
from summarize_followup_diagnostics import error_summary, spread_summary

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.schemas import ArtifactHash, BoundaryModel
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.ablations import observed_input
from md_forecast.evaluation.metrics import BenchmarkConfig, group_interval
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.base import ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.finetuning import (
    FineTunedChronos2Adapter,
    FineTuneManifest,
    FineTuneResult,
    prepare_finetuning,
    read_checkpoint,
    train_chronos,
)


class InputConfig(BoundaryModel):
    """Development-only comparison, fixed control and conservative input choice."""

    grid: WindowConfig
    learning_rate: Annotated[float, Field(gt=0)]
    seeds: Annotated[
        tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=2)
    ]
    representative_seed: int
    shift_frames: Annotated[int, Field(strict=True, gt=0)]
    lag_orders: tuple[Annotated[int, Field(strict=True, gt=0)], ...]
    ridge: Annotated[float, Field(ge=0)]
    uncertainty: BenchmarkConfig
    selection: Literal["joint-only-if-mean-seed-group-mae-beats-target-and-control"]

    @model_validator(mode="after")
    def check_protocol(self) -> InputConfig:
        """Reject ambiguous cells, invalid controls and missing seed references."""
        if len(self.grid.contexts) != 1 or len(self.grid.horizons) != 1:
            raise ValueError("input comparison requires one frozen cell")
        if self.grid.unit != "frame" or self.shift_frames >= self.grid.contexts[0]:
            raise ValueError("control shift must remain within the frame context")
        if (
            len(set(self.seeds)) != len(self.seeds)
            or self.representative_seed not in self.seeds
        ):
            raise ValueError("input comparison requires unique planned seeds")
        if self.uncertainty.comparison_family_size != 1:
            raise ValueError("development intervals are explicitly marginal only")
        return self


class InputPlan(BoundaryModel):
    """Bind the input comparison to its existing joint adaptation and native source."""

    config: InputConfig
    adaptation_hash: ArtifactHash
    split_hash: ArtifactHash
    joint_checkpoint_hashes: dict[str, ArtifactHash]


def name(target: str, seed: int) -> str:
    """Use an admitted feature's portable index; manifests retain its full identity."""
    return f"target{BASE_FEATURE_IDS.index(target)}-s{seed}"


def prepare(args: argparse.Namespace) -> None:
    """Freeze all nine univariate manifests and joint identities before scoring."""
    if args.root.exists():
        raise DataContractError("plan exists; preserve previous experiment")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
        text=True,
    ).strip():
        raise DataContractError("commit implementation before freezing input plan")
    config = InputConfig.model_validate_json(args.config.read_text())
    adaptation = read_metadata(args.adaptation / "plan.json", AdaptationConfig)
    if (config.grid, config.seeds) != (
        adaptation.training_grid,
        adaptation.seeds,
    ) or config.learning_rate not in adaptation.learning_rates:
        raise DataContractError("input grid/seeds/rate differ from joint adaptation")
    split, loader = load_native(args.native)
    joint = {}
    for seed in config.seeds:
        trial = trial_name(config.learning_rate, seed)
        result = read_metadata(
            args.adaptation / "runs" / trial / "result.json", FineTuneResult
        )
        checkpoint = read_checkpoint(
            args.adaptation / "runs" / trial / f"checkpoint-{result.selected_step}"
        )
        if (
            metadata_hash(checkpoint) != result.selected_checkpoint_hash
            or checkpoint.manifest.spec.split_hash != metadata_hash(split)
        ):
            raise DataContractError(
                "joint checkpoint differs from frozen native source"
            )
        joint[str(seed)] = result.selected_checkpoint_hash
    provenance = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "lockfile_hash": "sha256:"
        + hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.root.parent, prefix=".input-plan-") as temporary:
        root = Path(temporary)
        write_metadata(
            root / "plan.json",
            InputPlan(
                config=config,
                adaptation_hash=metadata_hash(adaptation),
                split_hash=metadata_hash(split),
                joint_checkpoint_hashes=joint,
            ),
        )
        (root / "manifests").mkdir()
        for target in BASE_FEATURE_IDS:
            for seed in config.seeds:
                manifest = prepare_finetuning(
                    split=split,
                    grid=config.grid,
                    features=(target,),
                    settings=adaptation.settings.model_copy(update={"seed": seed}),
                    training=adaptation.training.model_copy(
                        update={
                            "learning_rate": config.learning_rate,
                            "batch_size": adaptation.training.batch_size
                            // len(BASE_FEATURE_IDS),
                        }
                    ),
                    loader=loader,
                    task="official-validation",
                    **provenance,
                )
                print(
                    name(target, seed),
                    write_metadata(
                        root / "manifests" / (name(target, seed) + ".json"), manifest
                    ),
                    flush=True,
                )
        root.rename(args.root)


def train(args: argparse.Namespace) -> None:
    """Train one declared target-only trial with four windows per optimizer update."""
    path = args.root / "manifests" / (args.trial + ".json")
    manifest = read_metadata(path, FineTuneManifest)
    _, loader = load_native(args.native)
    run = args.root / "runs" / args.trial
    run.parent.mkdir(exist_ok=True)
    result = train_chronos(
        manifest, loader, output_dir=run, cache_dir=Path("data/cache/chronos2")
    )
    assert result is not None
    print(result.model_dump_json(indent=2), flush=True)


def adapter(
    args: argparse.Namespace, plan: InputPlan, target: str, kind: str, condition: str
) -> Chronos2Adapter | StatisticalBaseline:
    """Reuse pinned adapters without relaxing feature/source identity guards."""
    settings = read_metadata(args.adaptation / "plan.json", AdaptationConfig).settings
    if kind == "zero-shot":
        return Chronos2Adapter(settings, cache_dir=Path("data/cache/chronos2"))
    if kind.startswith("lag"):
        return StatisticalBaseline(
            ModelConfig(
                model_id=ModelId.AR if condition == "target-only" else ModelId.VAR,
                seed=settings.seed,
                lags=int(kind[3:]),
                ridge=plan.config.ridge,
            )
        )
    seed = int(kind.split("-")[-1])
    trial = (
        name(target, seed)
        if condition == "target-only"
        else trial_name(plan.config.learning_rate, seed)
    )
    base = args.root if condition == "target-only" else args.adaptation
    result = read_metadata(base / "runs" / trial / "result.json", FineTuneResult)
    path = base / "runs" / trial / f"checkpoint-{result.selected_step}"
    checkpoint = read_checkpoint(path)
    if metadata_hash(checkpoint) != result.selected_checkpoint_hash:
        raise DataContractError("input model checkpoint identity differs")
    if (
        condition != "target-only"
        and result.selected_checkpoint_hash != plan.joint_checkpoint_hashes[str(seed)]
    ):
        raise DataContractError("joint checkpoint changed since input freeze")
    return FineTunedChronos2Adapter(settings, checkpoint_dir=path)


def groups(table: pa.Table) -> dict[str, float]:
    """Average windows within trajectories before equal-system aggregation."""
    trajectory = table.group_by(["group_id", "trajectory_id"]).aggregate(
        [("lead_absolute_error", "mean")]
    )
    result = trajectory.group_by(["group_id"]).aggregate(
        [("lead_absolute_error_mean", "mean")]
    )
    return {
        r["group_id"]: r["lead_absolute_error_mean_mean"] for r in result.to_pylist()
    }


def predictions(
    args: argparse.Namespace,
    plan: InputPlan,
    target: str,
    kind: str,
    condition: Literal["target-only", "joint", "shifted"],
    root: Path,
) -> tuple[pa.Table, dict[str, object]]:
    """Export all aligned target errors/quantiles; labels never enter the adapter."""
    split, loader = load_native(args.native)
    if metadata_hash(split) != plan.split_hash:
        raise DataContractError("input evaluation source changed")
    model = adapter(args, plan, target, kind, condition)
    windows, leads, runtimes = [], [], []
    for full, labels in iter_forecasts(
        split,
        plan.config.grid,
        Split.VALIDATION,
        BASE_FEATURE_IDS,
        loader,
        batch_size=32,
        with_targets=True,
    ):
        assert labels is not None
        batch = observed_input(
            full, target, condition, shift_frames=plan.config.shift_frames
        )
        truth = (
            labels[
                :,
                :,
                BASE_FEATURE_IDS.index(target) : BASE_FEATURE_IDS.index(target) + 1,
            ]
            if condition == "target-only"
            else labels
        )
        if isinstance(model, Chronos2Adapter):
            forecast = model.forecast(batch)
            point = forecast.median
            runtimes.append(forecast.runtime.model_dump(mode="json"))
        else:
            point = model.predict(batch)
        window, lead = diagnostic_tables(batch, truth, point, model)
        if isinstance(model, Chronos2Adapter):
            for i, level in enumerate(forecast.quantile_levels):
                lead = lead.append_column(
                    f"quantile-{level}", pa.array(forecast.values[..., i].ravel())
                )
        window = window.filter(pc.equal(window["feature_id"], target))
        window = window.set_column(
            window.schema.get_field_index("spread_ratio"),
            "spread_ratio",
            pa.array(window["spread_ratio"].to_numpy(), from_pandas=True),
        )
        windows.append(window)
        leads.append(lead.filter(pc.equal(lead["feature_id"], target)))
    window, lead = pa.concat_tables(windows), pa.concat_tables(leads)
    pq.write_table(window, root / "windows.parquet")
    pq.write_table(lead, root / "leads.parquet")
    result: dict[str, object] = {
        "artifact_hash": model.artifact_hash,
        "feature_ids": batch.spec.feature_ids,
        "source_split_hash": batch.spec.split_hash,
        "window_config_hash": batch.spec.window_config_hash,
        "preprocessing": (
            "Chronos observed-context local normalization; "
            "statistical native-unit context fit"
        ),
        "config": model.config.model_dump(mode="json"),
        "errors": error_summary(lead, leads=False),
        "per_lead": error_summary(lead, leads=True),
        "spread": spread_summary(window),
        "groups": groups(lead),
        "runtime": runtimes,
    }
    if isinstance(model, FineTunedChronos2Adapter):
        result["training_identity"] = {
            "manifest_hash": metadata_hash(model.checkpoint.manifest),
            "feature_set_hash": model.checkpoint.feature_set_hash,
            "input_hash": model.checkpoint.manifest.input_hash,
            "checkpoint_hash": metadata_hash(model.checkpoint),
            "selected_step": model.checkpoint.step,
            "budget": model.checkpoint.manifest.training.model_dump(mode="json"),
        }
    del model
    gc.collect()
    return lead.select(
        ["trajectory_id", "group_id", "start", "lead", "target", "origin"]
    ), result


def score(args: argparse.Namespace) -> None:
    """Keep every negative result; choose target inputs using development only."""
    if args.output is None or args.output.exists():
        raise DataContractError("score requires a fresh output")
    plan = read_metadata(args.root / "plan.json", InputPlan)
    if (
        metadata_hash(read_metadata(args.adaptation / "plan.json", AdaptationConfig))
        != plan.adaptation_hash
    ):
        raise DataContractError("parent adaptation changed")
    for target in BASE_FEATURE_IDS:
        for seed in plan.config.seeds:
            read_metadata(
                args.root / "runs" / name(target, seed) / "result.json", FineTuneResult
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    choices: dict[str, object] = {}
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".input-score-"
    ) as temporary:
        root = Path(temporary)
        for target in BASE_FEATURE_IDS:
            results[target], effects[target] = {}, {}
            reference = None
            kinds = [
                "zero-shot",
                *[f"fine-{seed}" for seed in plan.config.seeds],
                *[f"lag{lag}" for lag in plan.config.lag_orders],
            ]
            for kind in kinds:
                results[target][kind] = {}
                for condition in ("target-only", "joint", "shifted"):
                    path = root / target / kind / condition
                    path.mkdir(parents=True)
                    labels, row = predictions(args, plan, target, kind, condition, path)
                    if reference is None:
                        reference = labels
                    elif not reference.equals(labels):
                        raise DataContractError(
                            "input conditions differ in labels, origin or windows"
                        )
                    results[target][kind][condition] = row
                    print(target, kind, condition, row["errors"], flush=True)
                base = cast(
                    dict[str, float], results[target][kind]["target-only"]["groups"]
                )
                effects[target][kind] = {}
                for condition in ("joint", "shifted"):
                    candidate = cast(
                        dict[str, float], results[target][kind][condition]["groups"]
                    )
                    if base.keys() != candidate.keys():
                        raise DataContractError("paired system identities differ")
                    delta = np.asarray(
                        [candidate[key] - base[key] for key in sorted(base)],
                        dtype=np.float64,
                    )
                    ci = group_interval(delta, plan.config.uncertainty)
                    effects[target][kind][condition] = {
                        "mae_difference": float(delta.mean()),
                        "marginal_ci": ci,
                        "groups": len(delta),
                        "independent_unit": "system",
                        "corrected_claim": False,
                    }
            means = {
                condition: float(
                    np.mean(
                        [
                            cast(
                                float,
                                results[target][f"fine-{seed}"][condition]["errors"][0][
                                    "mae"
                                ],
                            )
                            for seed in plan.config.seeds
                        ]
                    )
                )
                for condition in ("target-only", "joint", "shifted")
            }
            seed_groups = {
                condition: {
                    group: float(
                        np.mean(
                            [
                                results[target][f"fine-{seed}"][condition]["groups"][
                                    group
                                ]
                                for seed in plan.config.seeds
                            ]
                        )
                    )
                    for group in results[target]["zero-shot"]["target-only"]["groups"]
                }
                for condition in ("target-only", "joint", "shifted")
            }
            effects[target]["fine-seed-mean"] = {}
            for condition in ("joint", "shifted"):
                delta = np.asarray(
                    [
                        seed_groups[condition][group]
                        - seed_groups["target-only"][group]
                        for group in sorted(seed_groups["target-only"])
                    ],
                    dtype=np.float64,
                )
                effects[target]["fine-seed-mean"][condition] = {
                    "mae_difference": float(delta.mean()),
                    "marginal_ci": group_interval(delta, plan.config.uncertainty),
                    "groups": len(delta),
                    "independent_unit": "system",
                    "corrected_claim": False,
                }
            choices[target] = {
                "condition": "joint"
                if means["joint"] < min(means["target-only"], means["shifted"])
                else "target-only",
                "mean_seed_mae": means,
                "representative_seed": plan.config.representative_seed,
            }
        files = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.parquet")
        }
        (root / "selection.json").write_text(
            json.dumps(
                {
                    "plan_hash": metadata_hash(plan),
                    "evaluation_code_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "lockfile_sha256": hashlib.sha256(
                        Path("uv.lock").read_bytes()
                    ).hexdigest(),
                    "results": results,
                    "paired_effects": effects,
                    "selected_inputs": choices,
                    "files_sha256": files,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        root.rename(args.output)
    print("selected inputs", choices, flush=True)


def main() -> None:
    """Execute an explicit plan phase; generated scientific artifacts stay local."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "score"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/followup-inputs.json")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/processed/followup-inputs42")
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument(
        "--adaptation",
        type=Path,
        default=Path("data/processed/followup-finetuning41-verified"),
    )
    parser.add_argument("--trial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "train" and (not args.trial or Path(args.trial).name != args.trial):
        parser.error("train requires a portable trial name")
    {"prepare": prepare, "train": train, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
