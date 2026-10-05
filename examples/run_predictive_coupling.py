"""Freeze B-only versus B+A inputs, score held-out replicas and save lag controls."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from md_forecast.core.constants import ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.preprocessing import ScalingConfig, fit_training_scaler
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import read_series
from md_forecast.data.splits import build_split
from md_forecast.evaluation.ablations import paired_effects, target_rows
from md_forecast.evaluation.benchmark import CellManifest, GridManifest, evaluate_cell
from md_forecast.evaluation.coupling import CouplingConfig, lag_diagnostics
from md_forecast.evaluation.report import read_benchmark, write_benchmark
from md_forecast.features.coupling import RegionManifest
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter


def provenance() -> dict[str, str]:
    """Reject uncommitted implementation and record exact experiment environment."""
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], text=True
    ).strip()
    untracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "examples",
            "configs",
            "tests",
        ],
        text=True,
    ).strip()
    if dirty or untracked:
        raise DataContractError(
            "commit implementation/config before freezing provenance"
        )
    return dict(
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        lockfile_hash="sha256:"
        + hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        hardware=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    )


def run(config: CouplingConfig, prepared: Path, output: Path) -> None:
    """Publish two complete paired benchmarks and diagnostics without tuning."""
    if output.exists() or output.is_symlink():
        raise DataContractError("output exists; choose a new directory")
    origin = provenance()
    regions = read_metadata(prepared / "regions.json", RegionManifest)
    if metadata_hash(regions) != config.regions_hash:
        raise DataContractError("region selection differs from frozen experiment")
    registry = read_registry(prepared / "registry.json")
    if registry.dataset.features != regions.feature_definitions():
        raise DataContractError("prepared regional feature definitions differ")
    split = build_split(registry, config.split)
    partitions = {a.split for a in split.assignments}
    if partitions != {Split.TRAIN, Split.TEST}:
        raise DataContractError(
            "coupling requires TRAIN and held-out replica TEST only"
        )
    exports = json.loads((prepared / "qc.json").read_text())["exported"]
    tables = {
        r.trajectory_id: read_series(
            prepared / exports[r.trajectory_id],
            Registry(dataset=registry.dataset, trajectories=(r,)),
        )
        for r in registry.trajectories
    }

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    chronos = Chronos2Adapter(config.chronos, cache_dir=Path("data/cache/chronos2"))
    statistics = tuple(
        StatisticalBaseline(
            ModelConfig(
                model_id=model,
                seed=config.chronos.seed,
                lags=config.lag_order if model in (ModelId.AR, ModelId.VAR) else None,
                ridge=config.ridge if model in (ModelId.AR, ModelId.VAR) else 0,
            )
        )
        for model in (
            ModelId.PERSISTENCE,
            ModelId.CONTEXT_MEAN,
            ModelId.LINEAR,
            ModelId.AR,
            ModelId.VAR,
        )
    )
    models: tuple[ForecastModel, ...] = (*statistics, chronos)
    manifests = {}
    for name, features in config.variants.items():
        first = next(
            iter_forecasts(
                split,
                config.grid,
                Split.TEST,
                features,
                loader,
                batch_size=config.batch_size,
            )
        )[0]
        scaler = fit_training_scaler(
            split,
            config.grid,
            ScalingConfig(scope="training-contexts", feature_ids=features),
            loader,
        )
        manifests[name] = CellManifest(
            spec=first.spec,
            split=split,
            grid=config.grid,
            partition=Split.TEST,
            config=config.evaluation,
            scaler=scaler,
            models=tuple(m.config for m in models),
            model_artifacts=tuple(m.artifact_hash for m in models),
            batch_size=config.batch_size,
            **origin,
        )
    family = sum(
        (len(models) - 1) * len(m.spec.feature_ids) * (m.spec.horizon_frames + 1)
        for m in manifests.values()
    ) + 2 * (first.spec.horizon_frames + 1)
    if family > config.evaluation.comparison_family_size:
        raise DataContractError("correction family omits regional comparisons")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".coupling-") as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        write_metadata(staging / "config.json", config)
        write_metadata(staging / "regions.json", regions)
        for name, manifest in manifests.items():
            write_metadata(staging / f"{name}-manifest.json", manifest)
        hashes = {
            identity: "sha256:"
            + hashlib.sha256(
                json.dumps(t.to_pydict(), sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for identity, t in tables.items()
        }
        (staging / "input-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        results = {}
        for name, manifest in manifests.items():
            result = evaluate_cell(
                models,
                iter_forecasts(
                    split,
                    config.grid,
                    Split.TEST,
                    manifest.spec.feature_ids,
                    loader,
                    batch_size=config.batch_size,
                    with_targets=True,
                ),
                manifest,
            )
            report = write_benchmark(
                staging / name, GridManifest(cells=(manifest,)), (result,)
            )
            assert read_benchmark(staging / name) == report
            results[name] = result
            print(name, report.artifact_id, flush=True)
        reference, joint = tuple(results.values())
        effects = []
        for model in (chronos, statistics[-1]):
            key = metadata_hash(model.config)
            if not target_rows(reference.predictions, config.target, key).equals(
                target_rows(joint.predictions, config.target, key)
            ):
                raise DataContractError(
                    "regional variants scored different labels/windows"
                )
            effects.extend(
                dict(model=model.config.model_id.value, **row)
                for row in paired_effects(
                    reference.metrics,
                    joint.metrics,
                    config.target,
                    key,
                    config.evaluation,
                )
            )
        pq.write_table(pa.Table.from_pylist(effects), staging / "effects.parquet")
        diagnostics = []
        for batch, _ in iter_forecasts(
            split,
            config.grid,
            Split.TEST,
            ("distal_rg", "pocket_rg"),
            loader,
            batch_size=config.batch_size,
            with_targets=False,
        ):
            diagnostics.extend(lag_diagnostics(batch, config))
        pq.write_table(
            pa.Table.from_pylist(diagnostics),
            staging / "context-lag-diagnostics.parquet",
        )
        (staging / "COMPLETE").write_text(
            "Paired replica experiment and context-only diagnostics complete.\n"
        )
        staging.rename(output)


def main() -> None:
    """Run the fixed regional experiment with already-extracted local inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/predictive-coupling.toml"),
    )
    parser.add_argument(
        "--prepared", type=Path, default=Path("data/processed/mdbind-coupling-regions")
    )
    args = parser.parse_args()
    run(
        CouplingConfig.model_validate(tomllib.loads(args.config.read_text())),
        args.prepared,
        args.output,
    )


if __name__ == "__main__":
    main()
