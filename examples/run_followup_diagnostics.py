"""Run development-only dependence and matched forecast diagnostics."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Annotated

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from md_forecast.analysis.diagnostics import (
    SPREAD_DDOF,
    dependence_summary,
    forecast_diagnostics,
)
from md_forecast.core.constants import Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.cohort import (
    CohortLoader,
    CohortManifest,
    ExternalReserve,
    permitted_external_sources,
)
from md_forecast.data.forecast import ForecastBatch, iter_forecasts
from md_forecast.data.preprocessing import (
    ScalingConfig,
    SeriesLoader,
    fit_training_scaler,
)
from md_forecast.data.public.mdbind import MDBindSubset
from md_forecast.data.registry import Registry, read_registry
from md_forecast.data.schemas import BoundaryModel, TrajectoryManifest
from md_forecast.data.series import FloatArray, read_series
from md_forecast.data.splits import SplitConfig, SplitManifest, build_split
from md_forecast.data.windows import WindowConfig
from md_forecast.evaluation.metrics import BenchmarkConfig
from md_forecast.features.structural import BASE_FEATURE_IDS
from md_forecast.models.base import ForecastModel, ModelConfig
from md_forecast.models.baselines import StatisticalBaseline
from md_forecast.models.chronos import Chronos2Adapter, ChronosConfig


class DiagnosticConfig(BoundaryModel):
    """Small candidate grid and descriptive uncertainty/resource budgets."""

    grid: WindowConfig
    max_lag_frames: Annotated[int, Field(strict=True, gt=0)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    candidates: tuple[ModelConfig, ...]
    uncertainty: BenchmarkConfig
    chronos: ChronosConfig


def load_native(root: Path) -> tuple[SplitManifest, SeriesLoader]:
    """Bind the frozen development-only role policy before any canonical read."""
    registry = read_registry(root / "registry.json")
    manifest = read_metadata(root / "cohort.json", CohortManifest)
    exports = json.loads((root / "qc.json").read_text())["exported"]
    loader = CohortLoader(root, registry, manifest, exports, purpose="tuning")
    return build_split(
        registry, SplitConfig(mode="official", seed=manifest.config.seed)
    ), loader


def load_seen(root: Path) -> tuple[SplitManifest, SeriesLoader, Registry]:
    """Read only reserve-bound development replicas; build a VAL-only system split."""
    registry = read_registry(root / "registry.json")
    subset = read_metadata(root / "source.json", MDBindSubset)
    reserve = read_metadata(root / "reserve.json", ExternalReserve)
    sources = {
        s.accession: s for s in permitted_external_sources(subset, reserve, "tuning")
    }
    exports = json.loads((root / "qc.json").read_text())["exported"]
    if {r.source_record for r in registry.trajectories} != set(sources):
        raise DataContractError(
            "seen development registry differs from permitted sources"
        )
    tables, validation = {}, []
    for record in registry.trajectories:
        source = sources[record.source_record]
        if (record.pdb_id, record.replicate_id, record.source_checksum) != (
            source.pdb_id,
            str(source.replica),
            source.files["trajectory.xtc"].checksum,
        ):
            raise DataContractError(
                "seen development record differs from frozen source"
            )
        path = (root / exports[record.trajectory_id]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise DataContractError("seen table escapes development directory")
        tables[record.trajectory_id] = read_series(
            path, Registry(dataset=registry.dataset, trajectories=(record,))
        )
        if (
            reserve.role(source.pdb_id, source.accession, source.replica)
            == "validation"
        ):
            validation.append(record)
    split = build_split(
        Registry(dataset=registry.dataset, trajectories=tuple(validation)),
        SplitConfig(mode="grouped", seed=reserve.seed, ratios=(0.0, 1.0, 0.0)),
    )

    def loader(record: TrajectoryManifest) -> pa.Table:
        return tables[record.trajectory_id]

    return split, loader, registry


def dependence(
    registry: Registry, loader: SeriesLoader, config: DiagnosticConfig
) -> dict[str, object]:
    """Average replicas within each system before resampling system means."""
    groups: dict[str, list[FloatArray]] = defaultdict(list)
    for record in registry.trajectories:
        table = loader(record)
        groups[record.system_id].append(
            np.column_stack([table[f].to_numpy() for f in BASE_FEATURE_IDS])
        )
    return {
        feature: dependence_summary(
            {
                group: tuple(row[:, index] for row in rows)
                for group, rows in groups.items()
            },
            config.max_lag_frames,
            config.uncertainty,
        ).model_dump(mode="json")
        for index, feature in enumerate(BASE_FEATURE_IDS)
    }


def diagnostic_tables(
    batch: ForecastBatch, targets: FloatArray, points: FloatArray, model: ForecastModel
) -> tuple[pa.Table, pa.Table]:
    """Export all windows/features/leads; undefined ratios become Arrow null."""
    values = forecast_diagnostics(batch, points, targets)
    count, horizon, features = points.shape
    identity = {
        "trajectory_id": np.asarray([index.trajectory_id for index in batch.indices]),
        "group_id": np.asarray([index.group_id for index in batch.indices]),
        "start": np.asarray([index.start for index in batch.indices]),
    }
    common = {
        "model_hash": metadata_hash(model.config),
        "context_frames": batch.spec.context_frames,
        "horizon_frames": horizon,
    }
    summary = {key: np.repeat(column, features) for key, column in identity.items()}
    summary |= {
        key: np.repeat(value, count * features) for key, value in common.items()
    }
    summary["feature_id"] = np.tile(batch.spec.feature_ids, count)
    summary |= {
        key: array.ravel()
        for key, array in values.items()
        if not key.startswith("lead_")
    }
    leads = {
        key: np.repeat(column, horizon * features) for key, column in identity.items()
    }
    leads |= {
        key: np.repeat(value, count * horizon * features)
        for key, value in common.items()
    }
    leads |= {
        "feature_id": np.tile(batch.spec.feature_ids, count * horizon),
        "lead": np.tile(np.repeat(np.arange(1, horizon + 1), features), count),
        "point": points.ravel(),
        "target": targets.ravel(),
        "origin": np.repeat(batch.context[:, -1:, :], horizon, axis=1).ravel(),
    }
    leads |= {
        key: array.ravel() for key, array in values.items() if key.startswith("lead_")
    }
    return pa.Table.from_pydict(summary).replace_schema_metadata(
        {"spread_ddof": str(SPREAD_DDOF)}
    ), pa.Table.from_pydict(leads)


def score(
    root: Path,
    split: SplitManifest,
    loader: SeriesLoader,
    config: DiagnosticConfig,
    chronos: Chronos2Adapter,
) -> dict[str, object]:
    """Compare all selection windows; spread never selects models."""
    models: tuple[ForecastModel, ...] = (
        *[StatisticalBaseline(c) for c in config.candidates],
        chronos,
    )
    prediction_rows = 0
    files: list[str] = []
    scales = {}
    runtime = []
    for context in sorted(config.grid.contexts):
        for horizon in sorted(config.grid.horizons):
            grid = config.grid.model_copy(
                update={"contexts": (context,), "horizons": (horizon,)}
            )
            prefix = f"c{int(context)}-h{int(horizon)}"
            if split.config.mode == "official":
                scaler = fit_training_scaler(
                    split,
                    grid,
                    ScalingConfig(
                        scope="training-contexts", feature_ids=BASE_FEATURE_IDS
                    ),
                    loader,
                )
                scales[prefix] = scaler.model_dump(mode="json")
            with ExitStack() as stack:
                writers: dict[str, pq.ParquetWriter] = {}
                for batch, targets in iter_forecasts(
                    split,
                    grid,
                    Split.VALIDATION,
                    BASE_FEATURE_IDS,
                    loader,
                    batch_size=config.batch_size,
                    with_targets=True,
                ):
                    assert targets is not None
                    for model in models:
                        if model is chronos:
                            forecast = chronos.forecast(batch)
                            points = forecast.median
                            runtime.append(forecast.runtime.model_dump(mode="json"))
                        else:
                            points = model.predict(batch)
                        prediction_rows += points.size
                        if prediction_rows > config.uncertainty.max_prediction_rows:
                            raise DataContractError(
                                "diagnostic prediction rows exceed budget"
                            )
                        summary, leads = diagnostic_tables(
                            batch, targets, points, model
                        )
                        for kind, table in (("windows", summary), ("leads", leads)):
                            # Undefined spread ratios are exported as null.
                            if kind == "windows":
                                index = table.schema.get_field_index("spread_ratio")
                                column = pa.array(
                                    table["spread_ratio"].to_numpy(), from_pandas=True
                                )
                                table = table.set_column(index, "spread_ratio", column)
                            name = f"{prefix}-{kind}.parquet"
                            if name not in writers:
                                writers[name] = stack.enter_context(
                                    pq.ParquetWriter(root / name, table.schema)
                                )
                                files.append(name)
                            writers[name].write_table(table)
    return {
        "files": files,
        "training_scalers": scales,
        "chronos_calls": runtime,
        "models": [model.config.model_dump(mode="json") for model in models],
    }


def run(args: argparse.Namespace) -> None:
    """Publish atomically; leave old evidence untouched and forbid output reuse."""
    if args.output.exists():
        raise DataContractError("output exists; choose a fresh directory")
    if (
        not args.acf_only
        and subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only", "--", "src", "examples", "configs"],
            text=True,
        ).strip()
    ):
        raise DataContractError(
            "commit implementation/config before forecast provenance"
        )
    config = DiagnosticConfig.model_validate_json(args.config.read_text())
    native, native_loader = load_native(args.native)
    seen, seen_loader, seen_registry = load_seen(args.seen)
    provenance = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "lockfile_hash": "sha256:"
        + hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "config_hash": metadata_hash(config),
        "native_split_hash": metadata_hash(native),
        "seen_split_hash": metadata_hash(seen),
        "spread_ddof": SPREAD_DDOF,
        "native_cohort_hash": metadata_hash(
            read_metadata(args.native / "cohort.json", CohortManifest)
        ),
        "seen_reserve_hash": metadata_hash(
            read_metadata(args.seen / "reserve.json", ExternalReserve)
        ),
        "seen_source_hash": metadata_hash(
            read_metadata(args.seen / "source.json", MDBindSubset)
        ),
        "implementation_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "examples/run_followup_diagnostics.py",
                "src/md_forecast/analysis/diagnostics.py",
            )
        },
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    start = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=args.output.parent, prefix=".diagnostics-"
    ) as temporary:
        root = Path(temporary)
        acf = {}
        for partition in (Split.TRAIN, Split.VALIDATION):
            subset = native.registry.model_copy(
                update={
                    "trajectories": tuple(
                        r for r in native.registry.trajectories if r.split == partition
                    )
                }
            )
            acf[f"native-{partition}"] = dependence(subset, native_loader, config)
        reserve = read_metadata(args.seen / "reserve.json", ExternalReserve)
        for role in ("train", "validation"):
            records = tuple(
                r
                for r in seen_registry.trajectories
                if reserve.role(str(r.pdb_id), r.source_record, int(r.replicate_id))
                == role
            )
            acf[f"seen-{role}"] = dependence(
                seen_registry.model_copy(update={"trajectories": records}),
                seen_loader,
                config,
            )
        write_metadata(root / "config.json", config)
        (root / "dependence.json").write_text(json.dumps(acf, indent=2) + "\n")
        if not args.acf_only:
            chronos = Chronos2Adapter(
                config.chronos, cache_dir=Path("data/cache/chronos2")
            )
            for task, split, loader in (
                ("native", native, native_loader),
                ("seen", seen, seen_loader),
            ):
                directory = root / task
                directory.mkdir()
                result = score(directory, split, loader, config, chronos)
                (directory / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        provenance["file_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }
        provenance["seconds"] = perf_counter() - start
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(args.output)
    print(f"published {args.output}; seconds={provenance['seconds']:.2f}")


def main() -> None:
    """Expose the fixed development cohort; no confirmation option exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/followup-diagnostics.json"),
    )
    parser.add_argument(
        "--native", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument(
        "--seen", type=Path, default=Path("data/processed/followup-seen-development")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acf-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
