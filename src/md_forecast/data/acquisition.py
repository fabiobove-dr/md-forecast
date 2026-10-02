"""Bounded MISATO acquisition with integrity-checked atomic publication."""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, BinaryIO

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from md_forecast.core import constants
from md_forecast.core.config import DownloadSettings
from md_forecast.core.exceptions import AcquisitionError

logger = logging.getLogger(__name__)


class SourceArtifact(BaseModel):
    """Authoritative filename-independent artifact size, URL, and checksum."""

    model_config = ConfigDict(extra="ignore")
    mode: str
    url: HttpUrl
    size_bytes: int = Field(gt=0)
    checksum: str

    @model_validator(mode="after")
    def validate_integrity_contract(self) -> SourceArtifact:
        """Reject unsupported digests and credentials in portable source URLs."""
        if not re.fullmatch(constants.CHECKSUM_PATTERN, self.checksum):
            raise ValueError("checksum must use md5 or sha256 and lowercase hex")
        if self.url.username is not None or self.url.password is not None:
            raise ValueError("source URLs must not contain credentials")
        return self


class AcquisitionSource(BaseModel):
    """Acquisition subset of audited metadata; feature schemas are separate."""

    model_config = ConfigDict(extra="ignore")
    schema_version: int = constants.SOURCE_CONFIG_VERSION
    dataset_id: str
    dataset_version: str
    source_record: str
    artifacts: dict[str, SourceArtifact]

    @model_validator(mode="after")
    def validate_filenames(self) -> AcquisitionSource:
        """Require a supported source version and portable, flat filenames."""
        if self.schema_version != constants.SOURCE_CONFIG_VERSION:
            raise ValueError("unsupported acquisition source schema version")
        if not self.artifacts:
            raise ValueError("source must list at least one artifact")
        _validate_names(self.artifacts)
        return self


def _validate_names(artifacts: dict[str, SourceArtifact]) -> None:
    for name in artifacts:
        if not re.fullmatch(constants.ARTIFACT_NAME_PATTERN, name):
            raise ValueError(f"unsafe artifact filename: {name!r}")
        if name.endswith((constants.PARTIAL_SUFFIX, constants.RECEIPT_SUFFIX)):
            raise ValueError(f"reserved artifact suffix: {name!r}")


def load_source(path: Path = constants.MISATO_CONFIG) -> AcquisitionSource:
    """Read and validate audited YAML acquisition metadata.

    Args:
        path: Source configuration path, relative to the working directory.

    Returns:
        Validated artifact metadata.

    Raises:
        AcquisitionError: The file or its acquisition metadata is invalid.
    """
    try:
        return AcquisitionSource.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise AcquisitionError(f"Cannot load source config {path}: {error}") from error


async def _file_job[T](
    function: Callable[..., T],
    *args: Any,
    cleanup: Callable[[T], object] | None = None,
) -> T:
    # Await an in-flight file operation before cancellation closes its handles.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        result = await task
        if cleanup is not None:
            cleanup(result)
        raise


def _validate_file(path: Path, artifact: SourceArtifact) -> None:
    if path.is_symlink() or not path.is_file():
        raise AcquisitionError(f"Expected a regular artifact file at {path}")
    if path.stat().st_size != artifact.size_bytes:
        raise AcquisitionError(
            f"Incomplete artifact {path}; expected {artifact.size_bytes} bytes"
        )
    algorithm, _, expected = artifact.checksum.partition(":")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise AcquisitionError(
            f"Checksum mismatch for {path}; move the invalid file aside and retry"
        )


def _partial_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _reject_symlinks(*paths: Path) -> None:
    if any(path.is_symlink() for path in paths):
        raise AcquisitionError(f"Refusing symlink artifact paths: {paths}")


def _remaining_bytes(final: Path, artifact: SourceArtifact) -> int:
    partial = final.with_name(final.name + constants.PARTIAL_SUFFIX)
    _reject_symlinks(final, partial)
    if final.exists():
        return 0
    offset = _partial_size(partial)
    if offset > artifact.size_bytes:
        raise AcquisitionError(
            f"Oversized partial file {partial}; move it aside and retry"
        )
    return artifact.size_bytes - offset


def _preflight(
    destination: Path, artifacts: dict[str, SourceArtifact], reserve: int
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    remaining = sum(
        _remaining_bytes(destination / name, item) for name, item in artifacts.items()
    )
    required = remaining + reserve
    available = shutil.disk_usage(destination).free
    if available < required:
        raise AcquisitionError(
            f"Not enough disk space at {destination}: "
            f"need {required}, available {available} bytes"
        )


def _response_offset(response: httpx.Response, offset: int, expected_size: int) -> int:
    response.raise_for_status()
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise AcquisitionError(
            "Source returned encoded bytes despite Accept-Encoding: identity"
        )
    if response.status_code == HTTPStatus.OK:
        return 0  # Range ignored: restart rather than append a complete response.
    if response.status_code != HTTPStatus.PARTIAL_CONTENT:
        raise AcquisitionError(
            f"Unexpected artifact HTTP status {response.status_code}"
        )
    _validate_range(response, offset, expected_size)
    return offset


def _validate_range(response: httpx.Response, offset: int, expected_size: int) -> None:
    match = re.fullmatch(
        constants.CONTENT_RANGE_PATTERN, response.headers.get("Content-Range", "")
    )
    if match is None or tuple(map(int, match.groups())) != (
        offset,
        expected_size - 1,
        expected_size,
    ):
        raise AcquisitionError(
            "Source returned an inconsistent Content-Range; "
            "partial bytes were preserved"
        )


async def _transfer(
    client: httpx.AsyncClient,
    artifact: SourceArtifact,
    partial: Path,
    settings: DownloadSettings,
) -> None:
    offset = await _file_job(_partial_size, partial)
    if offset == artifact.size_bytes:
        return
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    async with client.stream("GET", str(artifact.url), headers=headers) as response:
        start = _response_offset(response, offset, artifact.size_bytes)
        stream = await _file_job(
            _open_partial, partial, start, cleanup=lambda file: file.close()
        )
        try:
            await _write_stream(response, stream, start, artifact, settings)
        finally:
            await _file_job(stream.close)


def _open_partial(path: Path, offset: int) -> BinaryIO:
    if offset:
        return path.open("ab")
    return path.open("wb")


async def _write_stream(
    response: httpx.Response,
    stream: BinaryIO,
    received: int,
    artifact: SourceArtifact,
    settings: DownloadSettings,
) -> None:
    async for chunk in response.aiter_raw(chunk_size=settings.chunk_bytes):
        received += len(chunk)
        if received > artifact.size_bytes:
            raise AcquisitionError(f"Source sent too many bytes for {artifact.url}")
        await _file_job(stream.write, chunk)
    await _file_job(stream.flush)
    await _file_job(os.fsync, stream.fileno())


def _write_receipt(
    final: Path, artifact: SourceArtifact, source: AcquisitionSource
) -> None:
    receipt = {
        "dataset_id": source.dataset_id,
        "dataset_version": source.dataset_version,
        "source_record": source.source_record,
        "filename": final.name,
        "source_url": str(artifact.url),
        "size_bytes": artifact.size_bytes,
        "checksum": artifact.checksum,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=final.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(receipt, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(final.with_name(final.name + constants.RECEIPT_SUFFIX))
        finally:
            temporary.unlink(missing_ok=True)


async def _acquire_one(
    client: httpx.AsyncClient,
    name: str,
    artifact: SourceArtifact,
    source: AcquisitionSource,
    settings: DownloadSettings,
    destination: Path,
) -> Path:
    final = destination / name
    if await _file_job(final.exists):
        await _file_job(_validate_file, final, artifact)
        await _file_job(_write_receipt, final, artifact, source)
        return final
    partial = final.with_name(final.name + constants.PARTIAL_SUFFIX)
    for attempt in range(settings.retries + 1):
        try:
            await _transfer(client, artifact, partial, settings)
            break
        except httpx.HTTPError as error:
            if attempt == settings.retries:
                raise AcquisitionError(
                    f"Download failed for {name}; "
                    f"retained partial bytes for restart: {error}"
                ) from error
            await asyncio.sleep(settings.retry_delay * (attempt + 1))
    await _file_job(_validate_file, partial, artifact)
    await _file_job(partial.replace, final)
    await _file_job(_write_receipt, final, artifact, source)
    logger.info("Verified artifact %s", name)
    return final


async def acquire(
    source: AcquisitionSource,
    settings: DownloadSettings,
    mode: constants.DownloadMode = "metadata",
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Path]:
    """Acquire official splits, optionally including MD, with bounded I/O.

    Args:
        source: Validated acquisition metadata.
        settings: Destination, concurrency, storage reserve, and timeouts.
        mode: Metadata-only, or metadata plus the large MD artifact.
        transport: Optional local/mock HTTP transport for tests.

    Returns:
        Verified final paths in configuration order.

    Raises:
        AcquisitionError: Source selection, HTTP, filesystem, or integrity fails.
    """
    artifacts = _select_artifacts(source, mode)
    destination = (
        settings.destination or settings.data_dir / constants.MISATO_DATA_SUBDIR
    )
    timeout = httpx.Timeout(
        connect=settings.connect_timeout,
        read=settings.read_timeout,
        write=settings.write_timeout,
        pool=settings.pool_timeout,
    )
    try:
        lock = await _file_job(_lock_destination, destination, cleanup=os.close)
        try:
            await _file_job(
                _preflight, destination, artifacts, settings.disk_reserve_bytes
            )
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=transport,
                follow_redirects=True,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                return await _run_tasks(
                    client, source, artifacts, settings, destination
                )
        finally:
            await _file_job(os.close, lock)
    except OSError as error:
        raise AcquisitionError(
            f"Storage operation failed in {destination}: {error}"
        ) from error


def _select_artifacts(
    source: AcquisitionSource, mode: constants.DownloadMode
) -> dict[str, SourceArtifact]:
    if mode not in ("metadata", "md"):
        raise AcquisitionError(f"Unsupported acquisition mode {mode!r}")
    artifacts = {
        name: item
        for name, item in source.artifacts.items()
        if item.mode in ("metadata", mode)
    }
    if not artifacts:
        raise AcquisitionError(f"No acquisition artifacts for mode {mode!r}")
    return artifacts


def _lock_destination(destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / constants.ACQUISITION_LOCK
    _reject_symlinks(lock_path)
    lock = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock)
        raise
    return lock


async def _run_tasks(
    client: httpx.AsyncClient,
    source: AcquisitionSource,
    artifacts: dict[str, SourceArtifact],
    settings: DownloadSettings,
    destination: Path,
) -> list[Path]:
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def bounded(name: str, artifact: SourceArtifact) -> Path:
        async with semaphore:
            return await _acquire_one(
                client, name, artifact, source, settings, destination
            )

    tasks = [
        asyncio.create_task(bounded(name, item)) for name, item in artifacts.items()
    ]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
