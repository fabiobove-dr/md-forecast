"""Freeze on MISATO, score untouched MDbind, then fit the seen-system baseline."""

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from md_forecast.core.constants import DatasetId, ModelId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, write_metadata
from md_forecast.data.forecast import iter_forecasts
from md_forecast.data.preprocessing import (
    ScalerMetadata,
    ScalingConfig,
    fit_training_scaler,
)
from md_forecast.data.public.mdbind import common_dataset
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import TrajectoryManifest
from md_forecast.data.series import read_series, to_arrow
from md_forecast.data.splits import SplitConfig, SplitManifest, build_split
from md_forecast.evaluation.benchmark import (
    CellManifest,
    CellResult,
    GridManifest,
    evaluate_cell,
)
from md_forecast.evaluation.external import (
    ExternalConfig,
    select_statistics,
    selected_effects,
)
from md_forecast.evaluation.report import (
    BenchmarkReport,
    read_benchmark,
    write_benchmark,
)
from md_forecast.features.structural import BASE_FEATURE_IDS, load_structural_config
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter
from md_forecast.models.learned import fit_nlinear


def load_common(
    root: Path, geometry: Path, misato: bool
) -> tuple[Registry, dict[str, pa.Table]]:
    """Rewrap verified MISATO geometry without loading official TEST data."""
    registry = read_registry(root / "registry.json")
    exports = json.loads((root / "qc.json").read_text())["exported"]
    geometry_config = load_structural_config(geometry)
    dataset = common_dataset(geometry_config, registry.dataset.dataset_id)
    if (
        misato
        and registry.dataset.features[:3] != geometry_config.feature_definitions()[:3]
    ):
        raise DataContractError(
            "MISATO source geometry differs from the common formulas"
        )
    if misato:
        dataset = dataset.model_copy(
            update={"dataset_version": registry.dataset.dataset_version}
        )
    if not misato and dataset != registry.dataset:
        raise DataContractError("MDbind feature contract differs from frozen geometry")
    records, tables = [], {}
    for record in registry.trajectories:
        if misato and record.split != Split.TRAIN:
            continue
        original = read_series(
            root / exports[record.trajectory_id],
            Registry(dataset=registry.dataset, trajectories=(record,)),
        )
        common = record.model_copy(
            update={
                "dataset_version": dataset.dataset_version,
                "feature_set_version": dataset.feature_set_version,
            }
        )
        values = np.column_stack([original[f].to_numpy() for f in BASE_FEATURE_IDS])
        tables[common.trajectory_id] = to_arrow(
            common, dataset, original["time"].to_numpy(), values
        )
        records.append(common)
    return Registry(dataset=dataset, trajectories=tuple(records)), tables


def run(
    config: ExternalConfig, misato: Path, mdbind: Path, geometry: Path, output: Path
) -> None:
    """Execute the frozen order; no MDbind training precedes the first result."""
    if output.exists():
        raise DataContractError("output exists; choose a fresh directory")
    if (
        subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only"], text=True
        ).strip()
        or subprocess.check_output(
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
    ):
        raise DataContractError(
            "commit implementation/config before freezing provenance"
        )
    native, native_tables = load_common(misato, geometry, True)
    native_split = build_split(native, config.misato_split)
    chronos = Chronos2Adapter(config.chronos, cache_dir=Path("data/cache/chronos2"))
    statistics = tuple(
        StatisticalBaseline(
            ModelConfig(
                model_id=model,
                seed=42,
                lags=1 if model in (ModelId.AR, ModelId.VAR) else None,
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
    output.mkdir(parents=True)
    write_metadata(output / "config.json", config)

    def score(
        task: str,
        split: SplitManifest,
        tables: dict[str, pa.Table],
        partition: Split,
        adapters: tuple[ForecastModel, ...],
        scaler_source: SplitManifest | None = None,
        scaler: ScalerMetadata | None = None,
    ) -> tuple[CellResult, BenchmarkReport]:
        def loader(record: TrajectoryManifest) -> pa.Table:
            return tables[record.trajectory_id]

        if scaler is None:
            scaler = fit_training_scaler(
                split,
                config.grid,
                ScalingConfig(scope="training-contexts", feature_ids=BASE_FEATURE_IDS),
                loader,
            )
        first = next(
            iter_forecasts(
                split,
                config.grid,
                partition,
                BASE_FEATURE_IDS,
                loader,
                batch_size=config.batch_size,
                with_targets=True,
            )
        )[0]
        manifest = CellManifest(
            spec=first.spec,
            split=split,
            grid=config.grid,
            partition=partition,
            config=config.evaluation,
            scaler=scaler,
            scaler_source_split=scaler_source,
            models=tuple(m.config for m in adapters),
            model_artifacts=tuple(m.artifact_hash for m in adapters),
            batch_size=config.batch_size,
            **provenance,
        )
        write_metadata(output / (task + "-manifest.json"), manifest)
        hashes = {
            identity: "sha256:"
            + hashlib.sha256(
                json.dumps(table.to_pydict(), sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for identity, table in tables.items()
        }
        (output / (task + "-input-hashes.json")).write_text(
            json.dumps(hashes, indent=2) + "\n"
        )
        result = evaluate_cell(
            adapters,
            iter_forecasts(
                split,
                config.grid,
                partition,
                BASE_FEATURE_IDS,
                loader,
                batch_size=config.batch_size,
                with_targets=True,
            ),
            manifest,
        )
        report = write_benchmark(
            output / task, GridManifest(cells=(manifest,)), (result,)
        )
        assert read_benchmark(output / task) == report
        print(
            task,
            report.artifact_id,
            result.window_count,
            result.group_count,
            flush=True,
        )
        return result, report

    development, report = score(
        "misato-validation", native_split, native_tables, Split.VALIDATION, models
    )
    selection = select_statistics(development, report.artifact_id)
    write_metadata(output / "baseline-selection.json", selection)
    external, external_tables = load_common(mdbind, geometry, False)
    if external.dataset.dataset_id != DatasetId.MDBIND:
        raise DataContractError("external source must be audited MDbind")
    misato_ids = set()
    for name in ("train_MD.txt", "val_MD.txt", "test_MD.txt"):
        misato_ids.update(
            Path("data/external/misato", name).read_text().upper().split()
        )
    if misato_ids & {r.pdb_id for r in external.trajectories}:
        raise DataContractError("external cohort overlaps an official MISATO split")
    external_split = build_split(
        external, SplitConfig(mode="grouped", seed=42, ratios=(0.0, 0.0, 1.0))
    )
    result, _ = score(
        "mdbind-unseen-complex",
        external_split,
        external_tables,
        Split.TEST,
        models,
        native_split,
        development.manifest.scaler,
    )
    pq.write_table(
        selected_effects(result, selection, metadata_hash(chronos.config)),
        output / "unseen-complex-selected-effects.parquet",
    )
    # First external report is already published and verified before adaptation.
    replica_split = build_split(external, config.replica_split)

    def loader(record: TrajectoryManifest) -> pa.Table:
        return external_tables[record.trajectory_id]

    learned = fit_nlinear(
        replica_split,
        config.grid,
        BASE_FEATURE_IDS,
        loader,
        ModelConfig(model_id=ModelId.NLINEAR, seed=42, ridge=0),
        config.training,
    )
    write_metadata(output / "replica-nlinear.json", learned.state)
    result, _ = score(
        "mdbind-unseen-replica",
        replica_split,
        external_tables,
        Split.TEST,
        (*models, learned),
    )
    pq.write_table(
        selected_effects(result, selection, metadata_hash(chronos.config)),
        output / "unseen-replica-selected-effects.parquet",
    )


def main() -> None:
    """Run the committed specification and source/config identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmarks/mdbind-common.toml")
    )
    parser.add_argument(
        "--geometry", type=Path, default=Path("configs/features/common-geometry.yaml")
    )
    parser.add_argument(
        "--misato", type=Path, default=Path("data/processed/misato-structural-verified")
    )
    parser.add_argument(
        "--mdbind", type=Path, default=Path("data/processed/mdbind-common-verified")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        ExternalConfig.model_validate(tomllib.loads(args.config.read_text())),
        args.misato,
        args.mdbind,
        args.geometry,
        args.output,
    )


if __name__ == "__main__":
    main()
