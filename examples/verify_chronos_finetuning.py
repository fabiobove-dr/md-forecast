"""Reproduce the issue-12 native-sample GPU integration, not a training platform."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import pyarrow as pa

from md_forecast.core.constants import Split
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import read_series
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.data.windows import WindowConfig
from md_forecast.models.chronos import ChronosConfig
from md_forecast.models.finetuning import (
    FineTuneConfig,
    FineTuneManifest,
    prepare_finetuning,
    train_chronos,
)


def main() -> None:
    """Run from a clean checkout with the audited native sample already extracted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "pause", "resume"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path("data/processed/misato-native-sample")
    registry = read_registry(root / "registry.json")
    exported = json.loads((root / "qc.json").read_text())["exported"]
    source = Registry(
        dataset=registry.dataset,
        trajectories=tuple(r for r in registry.trajectories if r.split == Split.TRAIN),
    )

    def loader(record: TrajectoryManifest) -> pa.Table:
        assert record.split == Split.TRAIN and record.pdb_id != "16PK"
        return read_series(
            root / exported[record.trajectory_id],
            expected=Registry(dataset=registry.dataset, trajectories=(record,)),
        )

    if args.mode == "prepare":
        if args.manifest.exists():
            parser.error("manifest already exists; choose a fresh path")
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            parser.error("freeze provenance only from a clean checkout")
        budget = tomllib.loads(
            Path("configs/benchmarks/finetuning-development.toml").read_text()
        )
        split = build_split(source, SplitConfig.model_validate(budget["split"]))
        assert len(source.trajectories) == 19
        assert sum(a.split == Split.TRAIN for a in split.assignments) == 13
        assert sum(a.split == Split.VALIDATION for a in split.assignments) == 6
        settings = ChronosConfig.model_validate(
            tomllib.loads(Path("configs/models/chronos2.toml").read_text())
            | budget["chronos"]
        )
        training = FineTuneConfig.model_validate(
            tomllib.loads(Path("configs/models/chronos2-finetuning.toml").read_text())
        )
        hardware = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        frozen = prepare_finetuning(
            split=split,
            grid=WindowConfig.model_validate(budget["grid"]),
            features=tuple(f.feature_id for f in registry.dataset.features),
            settings=settings,
            training=training,
            loader=loader,
            task="development-train-holdout",
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            lockfile_hash="sha256:"
            + hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            hardware=hardware,
        )
        write_metadata(args.manifest, frozen)
        print(metadata_hash(frozen))
        return
    frozen = read_metadata(args.manifest, FineTuneManifest)
    pause = frozen.training.checkpoint_steps
    resume = args.output / f"checkpoint-{pause}" if args.mode == "resume" else None
    result = train_chronos(
        frozen,
        loader,
        output_dir=args.output,
        cache_dir=Path("data/cache/chronos2"),
        resume_from_checkpoint=resume,
        stop_after_checkpoint=pause if args.mode == "pause" else None,
    )
    print(
        result.model_dump_json(indent=2)
        if result
        else "paused; no completed result published"
    )


if __name__ == "__main__":
    main()
