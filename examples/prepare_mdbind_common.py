"""Acquire frozen raw MDbind sources and publish common geometry atomically."""

import argparse
import asyncio
import json
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

from md_forecast.core.config import DownloadSettings
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import AcquisitionSource, acquire
from md_forecast.data.artifacts import metadata_hash, write_metadata
from md_forecast.data.cohort import (
    PURPOSE_ROLES,
    CohortPurpose,
    ExternalReserve,
    permitted_external_sources,
)
from md_forecast.data.public.mdbind import MDBindSubset, ReplicaSource, extract_replica
from md_forecast.data.registry import Registry, write_registry
from md_forecast.data.series import series_metadata, write_series
from md_forecast.evaluation.confirmation import read_confirmation_plan
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


def prepare(
    subset: MDBindSubset,
    raw: Path,
    geometry: Path,
    output: Path,
    reserve: ExternalReserve | None = None,
    purpose: CohortPurpose = "tuning",
    source_qc: dict | None = None,
) -> None:
    """Reject partial cohorts; commit the complete registry and exports together."""
    if output.exists():
        raise DataContractError("output exists; choose a fresh directory")
    config = load_structural_config(geometry)
    sources = permitted_external_sources(subset, reserve, purpose)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".mdbind-") as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        records, exports, issues = [], {}, []
        if source_qc is not None:
            issues.extend(
                {
                    "source_system_id": pdb,
                    "status": "dropped",
                    "reason": "static selection contract failed",
                    "phase": "static",
                }
                for pdb in source_qc["excluded_pdb_ids"]
            )
        ordered = sorted(sources, key=lambda source: (source.pdb_id, source.replica))
        for pdb, group in groupby(ordered, key=lambda source: source.pdb_id):
            replicas = tuple(group)
            try:
                tables = [
                    (source, extract_replica(raw / source.accession, source, config))
                    for source in replicas
                ]
            except DataContractError as error:
                if purpose != "confirmation":
                    raise
                issues.append(
                    {
                        "source_system_id": pdb,
                        "status": "dropped",
                        "reason": str(error),
                        "phase": "geometry",
                    }
                )
                print(pdb, "GEOMETRY QC REJECTED", error, flush=True)
                continue
            for source, table in tables:
                registry = series_metadata(table)
                record = registry.trajectories[0]
                name = source.accession + ".parquet"
                write_series(staging / name, table)
                records.append(record)
                exports[record.trajectory_id] = name
        if not records:
            raise DataContractError("confirmation has no admissible complex groups")
        write_registry(
            staging / "registry.json",
            Registry(dataset=registry.dataset, trajectories=tuple(records)),
        )
        write_metadata(staging / "source.json", subset)
        write_metadata(staging / "geometry.json", config)
        if reserve is not None:
            write_metadata(staging / "reserve.json", reserve)
        (staging / "qc.json").write_text(
            json.dumps(
                {"exported": exports, "issues": issues, "static_qc": source_qc},
                indent=2,
            )
            + "\n"
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
    parser.add_argument("--reserve", type=Path)
    parser.add_argument("--purpose", choices=tuple(PURPOSE_ROLES), default="tuning")
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--source-qc", type=Path)
    args = parser.parse_args()
    subset = MDBindSubset.model_validate_json(args.source.read_text())
    if args.download:
        asyncio.run(download(subset, args.raw))
    reserve = (
        ExternalReserve.model_validate_json(args.reserve.read_text())
        if args.reserve is not None
        else None
    )
    if args.purpose == "confirmation":
        if args.confirmation_plan is None or reserve is None:
            raise DataContractError("confirmation requires its frozen plan and reserve")
        plan = read_confirmation_plan(args.confirmation_plan)
        if metadata_hash(reserve) not in (
            plan.external_reserve_hash,
            plan.seen_reserve_hash,
        ):
            raise DataContractError("confirmation reserve differs from the frozen plan")
    source_qc = json.loads(args.source_qc.read_text()) if args.source_qc else None
    if source_qc is not None:
        if args.purpose != "confirmation" or source_qc["reserve_hash"] != metadata_hash(
            reserve
        ):
            raise DataContractError(
                "source QC requires its frozen confirmation reserve"
            )
        excluded = set(source_qc["excluded_pdb_ids"])
        if set(dict(subset.selection.selected)) | excluded != set(
            dict(reserve.confirmation)
        ):
            raise DataContractError(
                "source QC does not cover exactly the selected cohort"
            )
        if set(dict(subset.selection.selected)) & excluded:
            raise DataContractError("source QC excludes an admitted complex")
    prepare(
        subset, args.raw, args.geometry, args.output, reserve, args.purpose, source_qc
    )


if __name__ == "__main__":
    main()
