"""Freeze and run geometric Chronos adaptation on native development."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from run_followup_diagnostics import load_native

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import read_metadata, write_metadata
from md_forecast.data.schemas import BoundaryModel
from md_forecast.data.windows import WindowConfig
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.chronos import ChronosConfig
from md_forecast.models.finetuning import (
    FineTuneConfig,
    FineTuneManifest,
    prepare_finetuning,
    train_chronos,
)


class AdaptationConfig(BoundaryModel):
    """Prospective LR/seed policy with a fixed representative seed."""

    training_grid: WindowConfig
    evaluation_grid: WindowConfig
    settings: ChronosConfig
    training: FineTuneConfig
    learning_rates: Annotated[
        tuple[Annotated[float, Field(gt=0)], ...], Field(min_length=1)
    ]
    seeds: Annotated[
        tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=2)
    ]
    representative_seed: Annotated[int, Field(strict=True, ge=0)]
    lr_selection: Literal["mean-normalized-group-mae-all-cells-features-seeds"]
    checkpoint_selection: Literal["minimum-upstream-validation-loss"]
    stopping: Literal["complete-fixed-updates"]
    tie_break: Literal["lower-learning-rate"]

    @model_validator(mode="after")
    def unique_candidates(self) -> Self:
        """Reject repeated trials and an undeclared representative seed."""
        if len(set(self.learning_rates)) != len(self.learning_rates) or len(
            set(self.seeds)
        ) != len(self.seeds):
            raise ValueError("adaptation candidates must be unique")
        if self.representative_seed not in self.seeds:
            raise ValueError("representative seed must be planned")
        return self


def trial_name(lr: float, seed: int) -> str:
    """Stable portable name; full scientific identity remains in the manifest hash."""
    return f"lr{lr:g}-s{seed}"


def prepare(args: argparse.Namespace) -> None:
    """Freeze every planned manifest/input hash before any optimizer updates."""
    if args.root.exists():
        raise DataContractError("plan output exists; preserve the frozen experiment")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
        text=True,
    ).strip():
        raise DataContractError("commit source/config before freezing adaptation")
    config = AdaptationConfig.model_validate_json(args.config.read_text())
    split, loader = load_native(args.native)
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
    args.root.mkdir(parents=True)
    write_metadata(args.root / "plan.json", config)
    for lr in config.learning_rates:
        for seed in config.seeds:
            manifest = prepare_finetuning(
                split=split,
                grid=config.training_grid,
                features=BASE_FEATURE_IDS,
                settings=config.settings.model_copy(update={"seed": seed}),
                training=config.training.model_copy(update={"learning_rate": lr}),
                loader=loader,
                task="official-validation",
                **provenance,
            )
            path = args.root / "manifests" / (trial_name(lr, seed) + ".json")
            path.parent.mkdir(exist_ok=True)
            print(path, write_metadata(path, manifest), flush=True)


def train(args: argparse.Namespace) -> None:
    """Run one frozen trial; explicit trusted local resume preserves its provenance."""
    path = args.root / "manifests" / (args.trial + ".json")
    manifest = read_metadata(path, FineTuneManifest)
    _, loader = load_native(args.native)
    run = args.root / "runs" / args.trial
    run.parent.mkdir(exist_ok=True)
    resume = run / f"checkpoint-{args.resume}" if args.resume is not None else None
    result = train_chronos(
        manifest,
        loader,
        output_dir=run,
        cache_dir=Path("data/cache/chronos2"),
        resume_from_checkpoint=resume,
        stop_after_checkpoint=args.pause_at,
    )
    print(result.model_dump_json(indent=2) if result else "paused; no completed result")


def main() -> None:
    """Prepare all candidates once, then execute/resume one explicitly named trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/followup-finetuning.json"),
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/processed/followup-finetuning41")
    )
    parser.add_argument("--trial")
    parser.add_argument("--resume", type=int)
    parser.add_argument("--pause-at", type=int)
    args = parser.parse_args()
    if args.mode == "train" and (not args.trial or Path(args.trial).name != args.trial):
        parser.error("train requires a local trial name")
    prepare(args) if args.mode == "prepare" else train(args)


if __name__ == "__main__":
    main()
