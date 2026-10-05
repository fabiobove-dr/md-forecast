"""Review reserved static MDbind metadata without requesting any trajectory."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from pydantic import Field

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import validate_artifact
from md_forecast.data.cohort import ExternalReserve
from md_forecast.data.public.mdbind import MDBindSubset, RawMetadata, RawRecord
from md_forecast.data.public.misato import load_misato_source, official_splits


class ReviewMetadata(RawMetadata):
    """Additional upstream identity fields; reported timing is not verified XTC time."""

    PROTSEQ: tuple[str, ...] = ()
    INCHIKEYS: tuple[str, ...] = ()
    mdFrames: Annotated[int, Field(strict=True, gt=0)] | None = None
    FRAMESTEP: Annotated[float, Field(gt=0)] | None = None


class ReviewRecord(RawRecord):
    """Source metadata schema, never a decoder of future trajectory values."""

    metadata: ReviewMetadata


class SourceReview(TypedDict):
    """Identity-only review output; all source fields were validated above."""

    pdb_id: str
    accession: str
    role: str
    url: str
    record_sha256: str
    record_bytes: int
    atoms: int
    declared_frames: int | None
    declared_frame_step_ns: float | None
    protein_sequence_hashes: list[str]
    inchikeys: list[str]
    exact_sequence_overlap_previous_raw_pilot: bool
    exact_inchikey_overlap_previous_raw_pilot: bool
    interactions: tuple[dict[str, str], ...]
    xtc_status: str


def verify_selection(args: argparse.Namespace, reserve: ExternalReserve) -> None:
    """Reconstruct the reserved identities from the frozen catalog and official IDs."""
    data = args.catalog.read_bytes()
    if hashlib.sha256(data).hexdigest() != reserve.catalog_sha256:
        raise DataContractError("external catalog differs from frozen selection")
    official = official_splits(load_misato_source(args.source), args.splits)
    excluded = set(official) | set(reserve.excluded_previous_ids)
    candidates: dict[str, str] = {}
    for project in json.loads(data)["projects"]:
        ids = project["metadata"].get("PDBIDS", [])
        if len(ids) != 1 or project.get("mdcount") != 10:
            continue
        pdb = ids[0].upper()
        if pdb in excluded:
            continue
        if "Creative Commons Attribution 4.0" not in project["metadata"].get(
            "LICENSE", ""
        ):
            continue
        accession = project["accession"]
        candidates[pdb] = min(candidates.get(pdb, accession), accession)
    ranked = sorted(
        candidates,
        key=lambda pdb: hashlib.sha256(
            f"{reserve.seed}|mdbind-unseen|{pdb}".encode()
        ).digest(),
    )
    selected = reserve.confirmation + reserve.development_seen_replica
    expected = tuple((pdb, candidates[pdb]) for pdb in ranked[: len(selected)])
    if (
        selected != expected
        or len(candidates) != reserve.eligible_unique_complexes
        or len(official) != reserve.all_official_misato_identity_excluded
    ):
        raise DataContractError(
            "external selection differs from its metadata-only rule"
        )


async def review(args: argparse.Namespace) -> None:
    """Reuse cached record bytes and compare exact identities with the old pilot."""
    reserve = ExternalReserve.model_validate_json(args.reserve.read_text())
    verify_selection(args, reserve)
    old = MDBindSubset.model_validate_json(args.old.read_text())
    pilot = []
    for source in old.replicas:
        if source.replica != 1:
            continue
        path = args.old_raw / source.accession / "record.json"
        await asyncio.to_thread(validate_artifact, path, source.files["record.json"])
        pilot.append(ReviewRecord.model_validate_json(path.read_bytes()).metadata)
    sequences = {s for m in pilot for s in m.PROTSEQ if s}
    keys = {s for m in pilot for s in m.INCHIKEYS if s}
    args.raw.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(4)
    results: list[SourceReview] = []

    async def one(
        client: httpx.AsyncClient, role: str, pdb: str, accession: str
    ) -> None:
        async with semaphore:
            path = args.raw / f"{accession}.json"
            url = "https://mdposit.mddbr.eu/api/rest/current/projects/" + accession
            if path.exists():
                data = await asyncio.to_thread(path.read_bytes)
            else:
                response = await client.get(url)
                response.raise_for_status()
                data = response.content
                await asyncio.to_thread(path.write_bytes, data)
            record = ReviewRecord.model_validate_json(data)
            if record.metadata.PDBIDS != (pdb,) or record.mdNumber != 1:
                raise DataContractError("reserved MDbind record identity changed")
            metadata = record.metadata
            source_sequences = {s for s in metadata.PROTSEQ if s}
            source_keys = {s for s in metadata.INCHIKEYS if s}
            results.append(
                {
                    "pdb_id": pdb,
                    "accession": accession,
                    "role": role,
                    "url": url,
                    "record_sha256": hashlib.sha256(data).hexdigest(),
                    "record_bytes": len(data),
                    "atoms": metadata.SYSTATS,
                    "declared_frames": metadata.mdFrames,
                    "declared_frame_step_ns": metadata.FRAMESTEP,
                    "protein_sequence_hashes": [
                        "sha256:" + hashlib.sha256(s.encode()).hexdigest()
                        for s in sorted(source_sequences)
                    ],
                    "inchikeys": sorted(source_keys),
                    "exact_sequence_overlap_previous_raw_pilot": bool(
                        source_sequences & sequences
                    ),
                    "exact_inchikey_overlap_previous_raw_pilot": bool(
                        source_keys & keys
                    ),
                    "interactions": metadata.INTERACTIONS,
                    "xtc_status": "not downloaded or decoded",
                }
            )

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        await asyncio.gather(
            *(
                one(client, role, pdb, accession)
                for role, pairs in (
                    ("confirmation", reserve.confirmation),
                    ("development_seen_replica", reserve.development_seen_replica),
                )
                for pdb, accession in pairs
            )
        )
    results.sort(key=lambda item: (item["role"], item["pdb_id"]))
    seen = [row for row in results if row["role"] == "development_seen_replica"]
    seen_sequences = {value for row in seen for value in row["protein_sequence_hashes"]}
    seen_keys = {value for row in seen for value in row["inchikeys"]}
    confirmation = [row for row in results if row["role"] == "confirmation"]
    output = {
        "sources": results,
        "exact_sequence_overlap_previous_raw_pilot": sum(
            bool(row["exact_sequence_overlap_previous_raw_pilot"]) for row in results
        ),
        "exact_inchikey_overlap_previous_raw_pilot": sum(
            bool(row["exact_inchikey_overlap_previous_raw_pilot"]) for row in results
        ),
        "confirmation_exact_sequence_overlap_new_seen_development": [
            row["pdb_id"]
            for row in confirmation
            if set(row["protein_sequence_hashes"]) & seen_sequences
        ],
        "confirmation_exact_inchikey_overlap_new_seen_development": [
            row["pdb_id"] for row in confirmation if set(row["inchikeys"]) & seen_keys
        ],
        "unknown": "MISATO target/ligand metadata, sequence-similarity and chemotype "
        "thresholds, upstream pretraining overlap; exact identity disjointness "
        "is not target/chemotype independence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"reviewed {len(results)} static records; no XTC bytes requested")


def main() -> None:
    """Write a portable metadata-only identity/overlap review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/external/mdbind-audit/catalog-geometric-complete.json"),
    )
    parser.add_argument(
        "--source", type=Path, default=Path("configs/datasets/misato.yaml")
    )
    parser.add_argument("--splits", type=Path, default=Path("data/external/misato"))
    parser.add_argument(
        "--reserve",
        type=Path,
        default=Path("configs/datasets/followup-external-reserve.json"),
    )
    parser.add_argument(
        "--old", type=Path, default=Path("configs/datasets/mdbind-common.json")
    )
    parser.add_argument(
        "--old-raw", type=Path, default=Path("data/external/mdbind-common-raw")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/external/followup-mdbind-metadata")
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(review(parser.parse_args()))


if __name__ == "__main__":
    main()
