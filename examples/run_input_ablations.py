"""Run frozen zero-shot input ablations on the audited MISATO development holdout."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from md_forecast.core.constants import TIME_COLUMN, ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.preprocessing import ScalingConfig, fit_training_scaler
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import read_series, to_arrow
from md_forecast.data.splits import build_split
from md_forecast.evaluation.ablations import AblationConfig, paired_effects, target_rows
from md_forecast.evaluation.benchmark import CellManifest, GridManifest, evaluate_cell
from md_forecast.evaluation.report import read_benchmark, write_benchmark
from md_forecast.models.base import ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter


def load_inputs(native: Path, structural: Path) -> tuple[Registry, dict[str, pa.Table]]:
    """Join verified source/frame identities; never load the official TEST record."""
    left, right = (read_registry(p / "registry.json") for p in (native, structural))
    exports = [
        json.loads((p / "qc.json").read_text())["exported"]
        for p in (native, structural)
    ]
    version = "misato-ablation-v1-" + metadata_hash(right.dataset).split(":")[1]
    dataset = left.dataset.model_copy(
        update={
            "feature_set_version": version,
            "features": left.dataset.features + right.dataset.features,
        }
    )
    records, tables = [], {}
    right_records = {r.trajectory_id: r for r in right.trajectories}
    for record in left.trajectories:
        if record.split != Split.TRAIN:
            continue
        other = right_records[record.trajectory_id]
        if record.model_dump(exclude={"feature_set_version"}) != other.model_dump(
            exclude={"feature_set_version"}
        ):
            raise DataContractError("native/structural source identities differ")
        a = read_series(
            native / exports[0][record.trajectory_id],
            Registry(dataset=left.dataset, trajectories=(record,)),
        )
        b = read_series(
            structural / exports[1][record.trajectory_id],
            Registry(dataset=right.dataset, trajectories=(other,)),
        )
        if not a[TIME_COLUMN].equals(b[TIME_COLUMN]):
            raise DataContractError("native/structural frame axes differ")
        combined = record.model_copy(update={"feature_set_version": version})
        values = np.column_stack(
            [
                t[f.feature_id].to_numpy()
                for t, ds in ((a, left.dataset), (b, right.dataset))
                for f in ds.features
            ]
        )
        tables[record.trajectory_id] = to_arrow(
            combined, dataset, a[TIME_COLUMN].to_numpy(), values
        )
        records.append(combined)
    return Registry(dataset=dataset, trajectories=tuple(records)), tables


def run(config: AblationConfig, native: Path, structural: Path, output: Path) -> None:
    """Freeze inputs before inference; save benchmarks and paired effects."""
    if output.exists():
        raise DataContractError("output exists; choose a fresh directory")
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
            "examples",
            "configs",
            "src",
        ],
        text=True,
    ).strip()
    if dirty or untracked:
        raise DataContractError(
            "commit experiment implementation before freezing provenance"
        )
    registry, tables = load_inputs(native, structural)
    split = build_split(registry, config.split)
    if not any(a.split == Split.VALIDATION for a in split.assignments) or any(
        a.split == Split.TEST for a in split.assignments
    ):
        raise DataContractError("requires a development TRAIN/VAL holdout without TEST")

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    chronos = Chronos2Adapter(config.chronos, cache_dir=Path("data/cache/chronos2"))
    persistence = StatisticalBaseline(
        ModelConfig(model_id=ModelId.PERSISTENCE, seed=config.chronos.seed)
    )
    models = (persistence, chronos)
    provenance = dict(
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
    manifests = {}
    for name, features in config.variants.items():
        pairs = iter_forecasts(
            split,
            config.grid,
            Split.VALIDATION,
            features,
            loader,
            batch_size=config.batch_size,
            with_targets=True,
        )
        first = next(pairs)[0]
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
            partition=Split.VALIDATION,
            config=config.evaluation,
            scaler=scaler,
            models=tuple(m.config for m in models),
            model_artifacts=tuple(m.artifact_hash for m in models),
            batch_size=config.batch_size,
            **provenance,
        )
    family = sum(
        len(m.spec.feature_ids) * (m.spec.horizon_frames + 1)
        for m in manifests.values()
    ) + (len(manifests) - 1) * (first.spec.horizon_frames + 1)
    if config.evaluation.comparison_family_size < family:
        raise DataContractError("correction family must cover every scored comparison")
    output.mkdir(parents=True)
    write_metadata(output / "config.json", config)
    for name, manifest in manifests.items():
        write_metadata(output / f"{name}-manifest.json", manifest)
    # Freeze input contents too, before any forecast is scored.
    input_hashes = {
        identity: "sha256:"
        + hashlib.sha256(
            json.dumps(table.to_pydict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        for identity, table in tables.items()
    }
    (output / "input-hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    reference = None
    effects = []
    for name, manifest in manifests.items():
        result = evaluate_cell(
            models,
            iter_forecasts(
                split,
                config.grid,
                Split.VALIDATION,
                manifest.spec.feature_ids,
                loader,
                batch_size=config.batch_size,
                with_targets=True,
            ),
            manifest,
        )
        labels = target_rows(
            result.predictions, config.target, metadata_hash(chronos.config)
        )
        if reference is None:
            reference = result
            reference_labels = labels
            reference_name = name
        elif not labels.equals(reference_labels):
            raise DataContractError("ablation future labels/windows differ")
        else:
            for row in paired_effects(
                reference.metrics,
                result.metrics,
                config.target,
                metadata_hash(chronos.config),
                config.evaluation,
            ):
                effects.append(dict(variant=name, reference=reference_name, **row))
        report = write_benchmark(
            output / name, GridManifest(cells=(manifest,)), (result,)
        )
        assert read_benchmark(output / name) == report
        print(name, report.artifact_id, flush=True)
    pq.write_table(pa.Table.from_pylist(effects), output / "effects.parquet")
    (output / "COMPLETE").write_text(
        "All frozen variants and paired effects completed.\n"
    )


def main() -> None:
    """Run the documented development experiment with already-extracted inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmarks/input-ablations.toml")
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/misato-native-sample")
    )
    parser.add_argument(
        "--structural",
        type=Path,
        default=Path("data/processed/misato-structural-verified"),
    )
    args = parser.parse_args()
    run(
        AblationConfig.model_validate(tomllib.loads(args.config.read_text())),
        args.native,
        args.structural,
        args.output,
    )


if __name__ == "__main__":
    main()
