"""Acquire frozen raw MDbind sources and publish common geometry atomically."""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from md_forecast.core.config import DownloadSettings
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import AcquisitionSource, acquire
from md_forecast.data.artifacts import write_metadata
from md_forecast.data.public.mdbind import MDBindSubset, ReplicaSource, extract_replica
from md_forecast.data.registry import Registry, write_registry
from md_forecast.data.series import series_metadata, write_series
from md_forecast.features.structural import load_structural_config


async def download(subset: MDBindSubset, raw: Path) -> None:
    """Reuse the integrity-checked downloader; bound concurrent replica jobs."""
    semaphore = asyncio.Semaphore(3)

    async def replica(source: ReplicaSource) -> None:
        async with semaphore:
            acquisition = AcquisitionSource(
                dataset_id="mdbind",
                dataset_version="audited-2026-10-03",
                source_record=source.accession,
                artifacts=source.files,
            )
            await acquire(
                acquisition,
                DownloadSettings(destination=raw / source.accession, concurrency=2),
            )

    await asyncio.gather(*(replica(source) for source in subset.replicas))


def prepare(subset: MDBindSubset, raw: Path, geometry: Path, output: Path) -> None:
    """Reject partial cohorts; commit the complete registry and exports together."""
    if output.exists():
        raise DataContractError("output exists; choose a fresh directory")
    config = load_structural_config(geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".mdbind-") as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        records, exports = [], {}
        for source in subset.replicas:
            table = extract_replica(raw / source.accession, source, config)
            registry = series_metadata(table)
            record = registry.trajectories[0]
            name = source.accession + ".parquet"
            write_series(staging / name, table)
            records.append(record)
            exports[record.trajectory_id] = name
        write_registry(
            staging / "registry.json",
            Registry(dataset=registry.dataset, trajectories=tuple(records)),
        )
        write_metadata(staging / "source.json", subset)
        write_metadata(staging / "geometry.json", config)
        (staging / "qc.json").write_text(
            json.dumps({"exported": exports}, indent=2) + "\n"
        )
        staging.rename(output)


def main() -> None:
    """Prepare only the reviewed common geometry; native channels remain blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("configs/datasets/mdbind-common.json")
    )
    parser.add_argument(
        "--geometry", type=Path, default=Path("configs/features/common-geometry.yaml")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/external/mdbind-common-raw")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mdbind-common-geometry")
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    subset = MDBindSubset.model_validate_json(args.source.read_text())
    if args.download:
        asyncio.run(download(subset, args.raw))
    prepare(subset, args.raw, args.geometry, args.output)


if __name__ == "__main__":
    main()
