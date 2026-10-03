"""Freeze static regional selections, then extract verified radius series."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.public.mdbind import MDBindSubset
from md_forecast.data.registry import Registry, read_registry, write_registry
from md_forecast.data.series import to_arrow, write_series
from md_forecast.features.coupling import (
    RegionManifest,
    RegionRules,
    extract_region_series,
    freeze_regions,
)


def extract(
    subset: MDBindSubset,
    regions: RegionManifest,
    raw: Path,
    canonical: Path,
    output: Path,
) -> None:
    """Reuse admitted trajectory identities; atomically publish the whole cohort."""
    if output.exists() or output.is_symlink():
        raise DataContractError("output exists; choose a new directory")
    if regions.source_hash != metadata_hash(subset):
        raise DataContractError("region cohort differs from reviewed sources")
    original = read_registry(canonical / "registry.json")
    dataset = original.dataset.model_copy(
        update={
            "feature_set_version": "regional-radii-v1-"
            + metadata_hash(regions).split(":")[1],
            "features": regions.feature_definitions(),
        }
    )
    originals = {r.source_record: r for r in original.trajectories}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".coupling-") as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        write_metadata(staging / "regions.json", regions)
        write_metadata(staging / "source.json", subset)
        records, exports = [], {}
        for source in subset.replicas:
            record = originals[source.accession]
            if record.source_checksum != source.files["trajectory.xtc"].checksum:
                raise DataContractError("canonical source checksum differs")
            time, values = extract_region_series(
                raw / source.accession, source, regions
            )
            record = record.model_copy(
                update={"feature_set_version": dataset.feature_set_version}
            )
            name = source.accession + ".parquet"
            write_series(staging / name, to_arrow(record, dataset, time, values))
            records.append(record)
            exports[record.trajectory_id] = name
        write_registry(
            staging / "registry.json",
            Registry(dataset=dataset, trajectories=tuple(records)),
        )
        (staging / "qc.json").write_text(
            json.dumps({"exported": exports}, indent=2) + "\n"
        )
        staging.rename(output)


def main() -> None:
    """Keep static selection freezing separate from reading trajectory values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "extract"))
    parser.add_argument(
        "--source", type=Path, default=Path("configs/datasets/mdbind-common.json")
    )
    parser.add_argument(
        "--rules", type=Path, default=Path("configs/features/coupling-regions.json")
    )
    parser.add_argument(
        "--regions", type=Path, default=Path("configs/features/coupling-selection.json")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/external/mdbind-common-raw")
    )
    parser.add_argument(
        "--canonical", type=Path, default=Path("data/processed/mdbind-common-verified")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    subset = MDBindSubset.model_validate_json(args.source.read_text())
    if args.mode == "freeze":
        if args.regions.exists():
            raise DataContractError("frozen region selection already exists")
        regions = freeze_regions(
            subset, args.raw, RegionRules.model_validate_json(args.rules.read_text())
        )
        write_metadata(args.regions, regions)
        print(metadata_hash(regions))
    else:
        if args.output is None:
            parser.error("extract requires --output")
        extract(
            subset,
            read_metadata(args.regions, RegionManifest),
            args.raw,
            args.canonical,
            args.output,
        )


if __name__ == "__main__":
    main()
