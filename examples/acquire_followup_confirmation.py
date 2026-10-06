"""Acquire the frozen external confirmation cohort; keep XTC bytes opaque."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import mdtraj

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.acquisition import SourceArtifact
from md_forecast.data.artifacts import metadata_hash
from md_forecast.data.cohort import ExternalReserve
from md_forecast.data.public.mdbind import (
    MDBindSubset,
    RawRecord,
    ReplicaSource,
    verified_replica_topology,
)
from md_forecast.evaluation.confirmation import read_confirmation_plan
from md_forecast.evaluation.overview import file_hash

MAX_FILE_BYTES = 100 * 1024**2


async def fetch(client: httpx.AsyncClient, url: str, path: Path) -> SourceArtifact:
    """Stream bounded files atomically; cache completed bytes, never partial writes."""
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        for attempt in range(3):
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    size = 0
                    with temporary.open("wb") as file:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_FILE_BYTES:
                                raise DataContractError(
                                    "external artifact exceeds budget"
                                )
                            file.write(chunk)
                temporary.replace(path)
                break
            except httpx.HTTPError:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise DataContractError("cached external artifact exceeds budget")
    digest = await asyncio.to_thread(file_hash, path)
    return SourceArtifact(mode="metadata", url=url, size_bytes=size, checksum=digest)


async def run(args: argparse.Namespace) -> None:
    """Verify the frozen plan before requesting any reserved replica bytes."""
    plan = read_confirmation_plan(args.plan)
    reserve = ExternalReserve.model_validate_json(args.reserve.read_text())
    if metadata_hash(reserve) != plan.external_reserve_hash:
        raise DataContractError("external reserve differs from frozen confirmation")
    if args.output.exists():
        raise DataContractError(
            "source manifest exists; preserve its frozen identities"
        )
    semaphore = asyncio.Semaphore(3)
    sources = []
    failures = []

    async def replica(
        client: httpx.AsyncClient, pdb: str, base: str, number: int
    ) -> None:
        async with semaphore:
            accession = base if number == 1 else f"{base}.{number}"
            directory = args.raw / accession
            directory.mkdir(parents=True, exist_ok=True)
            url = "https://mdposit.mddbr.eu/api/rest/current/projects/" + accession
            cached = args.metadata / (base + ".json")
            if (
                number == 1
                and cached.exists()
                and not (directory / "record.json").exists()
            ):
                (directory / "record.json").write_bytes(cached.read_bytes())
            files = {}
            for name in (
                "record.json",
                "structure.pdb",
                "topology.prmtop",
                "trajectory.xtc",
            ):
                uri = url if name == "record.json" else url + "/files/" + name
                files[name] = await fetch(client, uri, directory / name)
            try:
                record = RawRecord.model_validate_json(
                    (directory / "record.json").read_bytes()
                )
                if record.metadata.PDBIDS != (pdb,) or record.mdNumber != number:
                    raise DataContractError(
                        "external replica metadata differs from selection"
                    )
                topology = await asyncio.to_thread(
                    mdtraj.load_topology, str(directory / "structure.pdb")
                )
                ligand = topology.residue(topology.n_residues - 1)
                interactions = [
                    item
                    for item in record.metadata.INTERACTIONS
                    if item["type"] == "protein-ligand"
                ]
                if len(interactions) != 1:
                    raise DataContractError(
                        "external selection requires one protein-ligand interaction"
                    )
                source = ReplicaSource(
                    accession=accession,
                    pdb_id=pdb,
                    replica=number,
                    atoms=record.metadata.SYSTATS,
                    ligand_residue_index=ligand.index,
                    ligand_name=ligand.name,
                    interaction_selection=interactions[0]["selection_2"],
                    files=files,
                )
                await asyncio.to_thread(verified_replica_topology, directory, source)
            except (DataContractError, ValueError, IndexError) as error:
                failures.append(
                    {"pdb_id": pdb, "accession": accession, "reason": str(error)}
                )
                print(accession, pdb, "STATIC QC REJECTED", error, flush=True)
                return
            sources.append(source)
            print(accession, pdb, "static topology verified; XTC opaque", flush=True)

    async with httpx.AsyncClient(
        timeout=90, follow_redirects=True, limits=httpx.Limits(max_connections=3)
    ) as client:
        await asyncio.gather(
            *(
                replica(client, pdb, base, number)
                for pdb, base in reserve.confirmation
                for number in range(1, 11)
            )
        )
    excluded = {failure["pdb_id"] for failure in failures}
    selected = tuple(pair for pair in reserve.confirmation if pair[0] not in excluded)
    sources = [source for source in sources if source.pdb_id not in excluded]
    qc = {
        "plan_hash": metadata_hash(plan),
        "reserve_hash": metadata_hash(reserve),
        "selected_groups": len(reserve.confirmation),
        "admitted_groups": len(selected),
        "excluded_pdb_ids": sorted(excluded),
        "failures": failures,
        "rule": (
            "existing static selection contract; reject entire complex "
            "if any replica fails; no replacement"
        ),
    }
    args.output.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    subset = MDBindSubset(
        selection={
            "selection_rule": reserve.selection_rule,
            "catalog_sha256": reserve.catalog_sha256,
            "selected": selected,
        },
        replicas=tuple(sorted(sources, key=lambda source: source.accession)),
        reserve_hash=metadata_hash(reserve),
    )
    args.output.write_text(subset.model_dump_json(indent=2) + "\n")
    print(
        "complete",
        len(sources),
        "replicas; bytes",
        sum(file.size_bytes for source in sources for file in source.files.values()),
        flush=True,
    )


def main() -> None:
    """Use explicit frozen reserve, bounded local storage and immutable manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("data/manifests/followup-confirmation45.json")
    )
    parser.add_argument(
        "--reserve",
        type=Path,
        default=Path("configs/datasets/followup-external-reserve80.json"),
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/external/followup-mdbind-metadata")
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/external/followup-external80-raw")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/datasets/followup-external80-source.json"),
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
