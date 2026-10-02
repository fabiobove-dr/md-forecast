"""Deterministically hashed, atomically persisted experiment metadata."""

import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from md_forecast.core.constants import SHA256_PATTERN
from md_forecast.core.exceptions import DataContractError
from md_forecast.data.registry import atomic_output
from md_forecast.data.schemas import BoundaryModel


class MetadataArtifact(BoundaryModel):
    """Integrity envelope; the payload's own model validates its schema version."""

    checksum: Annotated[str, Field(pattern=SHA256_PATTERN)]
    payload: dict[str, object]


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def metadata_hash(model: BaseModel) -> str:
    """Hash canonical JSON values, not local paths or Python process hashes."""
    return _digest(model.model_dump(mode="json"))


def write_metadata(path: Path, model: BaseModel) -> str:
    """Persist validated metadata and its full stable SHA-256 atomically."""
    payload = model.model_dump(mode="json")
    envelope = MetadataArtifact(checksum=_digest(payload), payload=payload)
    try:
        with atomic_output(path) as temporary:
            temporary.write_text(
                envelope.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
    except OSError as error:
        raise DataContractError(f"cannot write metadata {path}: {error}") from error
    return envelope.checksum


def read_metadata[T: BaseModel](path: Path, schema: type[T]) -> T:
    """Verify content integrity and then validate its versioned contract."""
    try:
        envelope = MetadataArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if _digest(envelope.payload) != envelope.checksum:
            raise ValueError("metadata checksum mismatch")
        return schema.model_validate(envelope.payload)
    except (OSError, ValueError) as error:
        raise DataContractError(f"cannot read metadata {path}: {error}") from error
