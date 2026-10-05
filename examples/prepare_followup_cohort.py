"""Freeze MISATO identities before extraction; publish only development roles."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from md_forecast.core.constants import DatasetId, Split
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import validate_artifact
from md_forecast.data.artifacts import metadata_hash, read_metadata, write_metadata
from md_forecast.data.cohort import (
    PURPOSE_ROLES,
    CohortConfig,
    CohortManifest,
    select_systems,
)
from md_forecast.data.public.mdbind import common_dataset
from md_forecast.data.public.misato import (
    SPLIT_FILES,
    ExtractionConfig,
    extract_misato,
    load_misato_source,
    official_splits,
)
from md_forecast.data.registry import Registry, read_registry, write_registry
from md_forecast.data.series import read_series, to_arrow, write_series
from md_forecast.features.structural import BASE_FEATURE_IDS, load_structural_config


def freeze(args: argparse.Namespace) -> None:
    """Check complete source integrity, then read only group identity metadata."""
    if args.manifest.exists():
        raise DataContractError("manifest exists; preserve the original selection")
    source = load_misato_source(args.source)
    geometry = load_structural_config(args.geometry)
    dataset = common_dataset(geometry, DatasetId.MISATO).model_copy(
        update={"dataset_version": source.dataset_version}
    )
    validate_artifact(args.input, source.artifacts["MD.hdf5"])
    official = official_splits(source, args.splits)
    config = CohortConfig.model_validate_json(args.config.read_text())
    excluded = tuple(json.loads(args.exclusions.read_text())["excluded_ids"])
    with h5py.File(args.input, "r") as file:
        available = tuple(file.keys())
    roles = select_systems(config, official, available, excluded)
    eligible = set(available) - set(excluded)
    manifest = CohortManifest(
        config=config,
        source_hash=metadata_hash(source),
        source_checksum=source.artifacts["MD.hdf5"].checksum,
        geometry_hash=metadata_hash(geometry),
        dataset_hash=metadata_hash(dataset),
        official_list_hashes={
            split: "sha256:"
            + hashlib.sha256((args.splits / name).read_bytes()).hexdigest()
            for split, name in SPLIT_FILES.items()
        },
        roles=roles,
        original_splits={pdb: official[pdb] for ids in roles.values() for pdb in ids},
        excluded_ids=excluded,
        eligible_counts={
            split: sum(official.get(pdb) == split for pdb in eligible)
            for split in Split
        },
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    print(write_metadata(args.manifest, manifest))


def prepare(args: argparse.Namespace) -> None:
    """Decode permitted roles only, retaining official VAL labels for calibration."""
    manifest = read_metadata(args.manifest, CohortManifest)
    source = load_misato_source(args.source)
    geometry = load_structural_config(args.geometry)
    if (metadata_hash(source), metadata_hash(geometry)) != (
        manifest.source_hash,
        manifest.geometry_hash,
    ):
        raise DataContractError("source/geometry differs from the frozen cohort")
    official = official_splits(source, args.splits)
    if any(
        official.get(pdb) != label for pdb, label in manifest.original_splits.items()
    ):
        raise DataContractError("official cohort labels changed")
    if args.output.exists():
        raise DataContractError("output exists; preserve the previous preparation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".followup-") as temporary:
        staging = Path(temporary) / "complete"
        prepare_staging(args, manifest, staging)
        if args.output.exists():
            raise DataContractError("output appeared during preparation")
        staging.rename(args.output)


def prepare_staging(
    args: argparse.Namespace, manifest: CohortManifest, staging: Path
) -> None:
    """Complete source extraction and canonical rewrapping before publication."""
    source = load_misato_source(args.source)
    geometry = load_structural_config(args.geometry)
    ids = tuple(
        pdb for role in PURPOSE_ROLES[args.purpose] for pdb in manifest.roles[role]
    )
    report = extract_misato(
        source,
        ExtractionConfig(
            input_path=args.input,
            output_dir=staging,
            splits_dir=args.splits,
            system_ids=tuple(sorted(ids)),
        ),
        structural=geometry,
    )
    registry = read_registry(staging / "registry.json")
    dataset = common_dataset(geometry, DatasetId.MISATO).model_copy(
        update={"dataset_version": source.dataset_version}
    )
    records = []
    for record in registry.trajectories:
        name = report.exported[record.trajectory_id]
        original = read_series(
            staging / name,
            Registry(dataset=registry.dataset, trajectories=(record,)),
        )
        common = record.model_copy(
            update={"feature_set_version": dataset.feature_set_version}
        )
        values = np.column_stack([original[f].to_numpy() for f in BASE_FEATURE_IDS])
        write_series(
            staging / name,
            to_arrow(common, dataset, original["time"].to_numpy(), values),
        )
        records.append(common)
    write_registry(
        staging / "registry.json",
        Registry(dataset=dataset, trajectories=tuple(records)),
    )
    write_metadata(staging / "cohort.json", manifest)
    write_metadata(staging / "geometry.json", geometry)
    (staging / "PREPARED").write_text(metadata_hash(manifest) + "\n")
    print(
        f"exported={len(records)}; exclusions={len(report.issues)}; "
        "confirmation untouched"
    )


def main() -> None:
    """Require explicit freeze or preparation; confirmation decoding is excluded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare"))
    parser.add_argument(
        "--source", type=Path, default=Path("configs/datasets/misato.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/datasets/followup-cohort.json")
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("configs/datasets/followup-exclusions.json"),
    )
    parser.add_argument(
        "--geometry", type=Path, default=Path("configs/features/common-geometry.yaml")
    )
    parser.add_argument("--splits", type=Path, default=Path("data/external/misato"))
    parser.add_argument(
        "--input", type=Path, default=Path("data/external/misato/MD.hdf5")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/followup-cohort.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/followup-development")
    )
    parser.add_argument(
        "--purpose", choices=("tuning", "calibration"), default="tuning"
    )
    args = parser.parse_args()
    (freeze if args.command == "freeze" else prepare)(args)


if __name__ == "__main__":
    main()
