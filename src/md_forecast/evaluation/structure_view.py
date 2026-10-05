"""Optional exact atom selections in a pinned offline molecular reference viewer."""

import json
import re
from html import escape
from pathlib import Path
from typing import Annotated, Self

import numpy as np
from pydantic import Field, HttpUrl, model_validator

from md_forecast.core.exceptions import DataContractError
from md_forecast.data.schemas import ArtifactHash, BoundaryModel

RENDERER_VERSION = "2.5.5"
RENDERER_HASH = (
    "sha256:f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016"
)
LICENSE_HASH = "sha256:4c6eaaed856f3f28a3b1a98e74f4a8a71618de7d51ea4155c29f6f793bcef861"
type AtomSerial = Annotated[int, Field(gt=0, strict=True)]


class StructureConfig(BoundaryModel):
    """User-reviewed embedding permission and exact PDB atom serial membership."""

    pdb: Path
    pdb_hash: ArtifactHash
    library: Path
    license: Path
    source_uri: HttpUrl
    license_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    embedding_permitted: bool = Field(strict=True)
    protein: Annotated[tuple[AtomSerial, ...], Field(min_length=1)]
    ligand: Annotated[tuple[AtomSerial, ...], Field(min_length=1)]
    regions: dict[str, tuple[AtomSerial, ...]] = Field(default_factory=dict)
    max_bytes: int = Field(default=20_000_000, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_selections(self) -> Self:
        """Embedding requires explicit permission and unambiguous memberships."""
        if not self.embedding_permitted:
            raise ValueError("structure redistribution must permit embedding")
        if set(self.protein) & set(self.ligand):
            raise ValueError("protein and ligand atom selections overlap")
        _validate_members(self)
        return self


def _validate_members(config: StructureConfig) -> None:
    for selection in (config.protein, config.ligand, *config.regions.values()):
        if not selection or len(set(selection)) != len(selection):
            raise ValueError("exact atom selections must be nonempty and unique")
    _validate_regions(config)


def _validate_regions(config: StructureConfig) -> None:
    if any(not set(s) <= set(config.protein) for s in config.regions.values()):
        raise ValueError("declared regions must be within protein selection")


def _verified(path: Path, digest: str, maximum: int) -> str:
    from md_forecast.evaluation.overview import file_hash

    if path.stat().st_size > maximum or file_hash(path) != digest:
        raise DataContractError(
            f"structure asset exceeds budget or checksum differs: {path.name}"
        )
    return path.read_text(encoding="utf-8")


def _atoms(pdb: str) -> dict[int, tuple[float, float, float]]:
    atoms = {}
    for line in pdb.splitlines():
        if line[:6].strip() in {"ATOM", "HETATM"}:
            serial = int(line[6:11])
            if serial in atoms:
                raise DataContractError(
                    "molecular reference requires a single unique-serial PDB model"
                )
            atoms[serial] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    _validate_atoms(atoms)
    return atoms


def _validate_atoms(atoms: dict[int, tuple[float, float, float]]) -> None:
    if not atoms or not np.isfinite(list(atoms.values())).all():
        raise DataContractError("molecular reference contains no finite atoms")


def _colors(config: StructureConfig) -> dict[int, str]:
    colors = dict.fromkeys(config.protein, "#b9cbd5")
    colors.update(dict.fromkeys(config.ligand, "#e7b54b"))
    for i, selection in enumerate(config.regions.values()):
        colors.update(dict.fromkeys(selection, ("#65d099", "#b39de8")[i % 2]))
    return colors


def _static(
    atoms: dict[int, tuple[float, float, float]], config: StructureConfig
) -> str:
    xyz = np.array(list(atoms.values()))
    low, high = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    scale = 460 / max(float(np.max(high - low)), 1.0)
    colors = _colors(config)
    circles = []
    for serial, color in colors.items():
        x, y, _ = atoms[serial]
        circles.append(
            f'<circle cx="{30 + (x - low[0]) * scale:.2f}" '
            f'cy="{490 - (y - low[1]) * scale:.2f}" r="2.3" fill="{color}">'
            f"<title>Atom serial {serial}</title></circle>"
        )
    return (
        '<svg viewBox="0 0 540 520" role="img" '
        'aria-label="Observed reference projected into the xy plane">'
        + "".join(circles)
        + "</svg>"
    )


def render_structure(path: Path) -> str:
    """Embed reviewed local PDB, license and renderer; no network calls are made."""
    try:
        config = StructureConfig.model_validate_json(path.read_text())
        pdb = _verified(config.pdb, config.pdb_hash, config.max_bytes)
        library = _verified(config.library, RENDERER_HASH, config.max_bytes)
        license_text = _verified(config.license, LICENSE_HASH, config.max_bytes)
        atoms = _atoms(pdb)
        selected = set(config.protein) | set(config.ligand)
        if not selected <= atoms.keys():
            raise DataContractError(
                "declared selection includes absent PDB atom serials"
            )
        template = (
            Path(__file__).parent / "overview_assets" / "structure.html"
        ).read_text()
        data = config.model_dump(mode="json")
        data["pdb"] = pdb
        replacements = {
            "__DATA__": json.dumps(data).replace("<", r"\u003c"),
            "__LIBRARY__": library.replace("</script", r"<\/script"),
            "__LICENSE__": escape(license_text),
            "__STATIC__": _static(atoms, config),
            "__DESCRIPTION__": escape(config.description),
            "__PROVENANCE__": escape(
                f"{config.source_uri}\n{config.license_id}\n{config.pdb_hash}\n"
                f"3Dmol.js {RENDERER_VERSION}\n{RENDERER_HASH}\n{LICENSE_HASH}"
            ),
        }
        template = re.sub(r"__[A-Z]+__", lambda match: replacements[match[0]], template)
        return (
            '<iframe title="Observed molecular reference and exact atom selections" '
            'style="width:100%;height:900px;border:0" srcdoc="'
            + escape(template, quote=True)
            + '"></iframe>'
        )
    except (OSError, ValueError, KeyError) as error:
        raise DataContractError(
            f"cannot embed reviewed molecular reference: {error}"
        ) from error
