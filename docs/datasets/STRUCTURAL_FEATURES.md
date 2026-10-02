# MISATO coordinate-derived geometry

Issue #15 adds the feasible coordinate-derived channels below, not atomistic
forecasting. Audited 2026-10-02 using the same immutable MISATO 1.0.0 record,
DOI, sample SHA-256, official split checksums and upstream revision recorded
in the [source audit](MISATO.md) and `configs/datasets/misato.yaml`.
No full MD download, topology archive download or source redistribution is needed.
Existing dataset/code license distinctions and unresolved legacy terms remain.

## Admission evidence and selections

The real pinned sample has float64 `trajectory_coordinates` `(100,N,3)`,
int64 `atoms_number` `(N,)`, and int64 `molecules_begin_atom_index` `(M,)`.
Coordinate Å units were verified in the earlier audit. The pinned upstream
preprocessor treats the final segment as ligand, including peptide ligands;
this adapter uses exactly that boundary, not guessed residue numbering.
[Pinned preprocessor](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/preprocessing_db.py),
[coordinate exporter](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/h5_to_pdb.py).

Here “protein/receptor” means **all pre-final source segments**, not a new
chemical classification. The final segment includes all ligand atoms; atomic
number >1 defines heavy atoms. The inspected sample has no zero-residue-code
atoms before that boundary. `1A3E` has nonzero ligand residue codes: filtering
only mapped residue code zero would incorrectly exclude its peptide ligand.
No downloaded pickle maps are executed. Other artifacts must pass shape,
dtype, boundary, atomic-number, source checksum and finite-coordinate checks.

| Channel | Definition | Unit |
| --- | --- | --- |
| `protein_ligand_contact_count` | Number of receptor–ligand **heavy atom pairs**, not residues, with Euclidean distance **≤ cutoff**. | count |
| `fraction_reference_contacts` | Fraction of the exact frame-0 contact pairs still within the same cutoff; new pairs do not count. A system with zero initial contacts is excluded because the fraction is undefined. | dimensionless |
| `minimum_protein_ligand_heavy_distance` | Minimum distance over all receptor–ligand heavy atom pairs. | Å |
| Configured region channel | Minimum distance between two explicit, disjoint heavy atom index sets supplied per system; never inferred from mapped residue types. | Å |
| `isolated_ligand_sasa` (optional) | Sum of all final-segment atom SASAs, **including hydrogen**, on the isolated ligand. Receptor atoms are not accessibility blockers. | Å² |

Distances use raw Cartesian coordinates without minimum-image/PBC correction.
Source imaging artifacts are therefore retained, not silently repaired; distance
or contact jumps are not automatically biological events. No periodic box is
available in inspected groups.
[Upstream imaging issue](https://github.com/t7morgen/misato-dataset/issues/24).

Frame 0 alone defines reference contacts. Later coordinates cannot change that
reference or atom selection. Physical timestamps remain unavailable: output
time is the source frame index, with no invented 80 ps interval.

## Optional SASA backend

SASA uses **MDTraj 1.11.1.post2** Shrake–Rupley with the backend's pinned default
element radii, an explicit probe and sphere point count. Coordinates are shifted
to the first ligand atom per frame, converted Å→nm and cast to the backend's
float32 format; areas are converted nm²→Å². This is a new, explicitly isolated
ligand definition, not native buried SASA or solvent accessibility in the complex.
[MDTraj API](https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.shrake_rupley.html),
[algorithm and radius table](https://mdtraj.readthedocs.io/en/latest/_modules/mdtraj/geometry/sasa.html).

In local analytic tests, a multi-frame backend call produced unequal areas for
identical isolated spheres. The adapter consequently invokes the kernel on
**one frame at a time**, even when HDF5 I/O is chunked. Regression tests retain
the analytic area `4π(1.7+1.4)² Å²` for a carbon atom and exact chunk invariance;
no tolerance was widened to hide the discrepancy. Backend batching can only
return after these checks hold. Missing/wrong backend versions fail before
extraction; unsupported element/kernel failures become explicit exclusion reasons.

The `structural` extra is optional; native extraction and geometric channels
without SASA do not require MDTraj. Python 3.14 Linux wheels were verified for
the pinned release. Its lock constraints resolve NumPy 2.4.6; this changes the
lockfile from the prior baseline environment, so historical numerical artifacts
must retain their original code/lock hashes. CI exercises the extra and wheel.

## Configuration and reproducibility

Use `configs/features/misato-geometry.yaml`. All resource limits are explicit:
4-frame chunks, 128-atom pair tiles, 100,000 source atoms/frames, a 64 MiB array
estimate ceiling, and at most 512 SASA ligand atoms. The illustrative cutoff is
4.5 Å; SASA uses a 1.4 Å probe and 960 sphere points. These are declared geometry
settings, not validation-selected benchmark parameters.

The feature-set identifier is `misato-geometry-v1-<full-config-sha256>`.
Every canonical trajectory and registry carries it; `qc.json` persists the
entire validated structural configuration and actual source/split provenance.
Cutoff, probe, selections, backend or resource-setting changes produce a new
identity. Algorithm semantics changes require a new algorithm version.
Hashes attest content integrity, not an honest scientific interpretation.

For a reviewed region selection, add entries like this (index values are an
illustration, **not** a biological selection for a real system):

```yaml
regions:
  - feature_id: selected_region_distance
    description: Reviewed source atom sets; do not claim residue identity without topology
    selections:
      1ABC:
        left: [0, 1]
        right: [20, 21]
```

Indices must be sorted, unique, in bounds, heavy, and disjoint; each requested
system needs every region's selection or is excluded with a reason. Record
reviewed mapping provenance in the description when naming actual residues.
Set `ligand_sasa: null` for geometry-only extraction without the extra.

## Extraction and checks

```sh
uv sync --locked --extra structural
OMP_NUM_THREADS=1 uv run --locked --extra structural md-forecast extract-misato \
  --input data/external/misato-audit/tiny_md.hdf5 --artifact sample \
  --splits data/external/misato \
  --output data/processed/misato-structural-verified \
  --structural-config configs/features/misato-geometry.yaml \
  --systems 10GS 11GS 13GS 184L 185L 186L 187L 188L \
    1A07 1A08 1A09 1A0Q 1A1B 1A1C 1A1E 1A28 1A2C 1A30 1A3E
```

This selects the 19 official TRAIN systems in the audited sample; TEST `16PK`
is not numerically read. It uses the existing verified-input adapter and staged
directory publication: existing output is refused, storage/integrity failures
never publish a partial bundle, scientific invalidity is recorded per system.
Publication retains the existing single-writer requirement.

Coordinate working memory scales with chunk×atoms and chunk×tile², not full
trajectory×atom-pairs. The output series is bounded by `max_frames`; metadata
and conservative coordinate/pair/output/SASA workspace estimates are checked
before reading coordinates. The byte limit is **not** a guarantee of total
process RSS, including Python, Arrow and compiled-library overhead. No source
coordinate trajectory or full atom-pair matrix is materialized across all frames.

Run the complete numerical suite with:

```sh
OMP_NUM_THREADS=1 uv run --locked --extra structural pytest
```

Tiny independent fixtures check inclusive contacts, initial versus new pairs,
exact minima/region distances, hydrogen exclusion, rigid translation, chunk/tile
invariance, analytic SASA, invalid topology fields, linked datasets, overflow,
nonfinite coordinates, budgets and CLI/canonical publication. Optional SASA tests
skip without the extra; CI installs it and executes them.

## Channels not admitted

Pocket **backbone** RMSD, hydrogen-bond count and chemically hydrophobic contacts
are not synthesized from geometric proximity or generic atomic numbers. Verified
backbone atom names/residue indices, donor–hydrogen–acceptor bonding/typing, and
chemical hydrophobic selections are absent from the inspected HDF5 fields.
Mapped residue types are not residue sequence indices, and raw AMBER type codes
are not self-describing chemical labels. These channels need a reviewed topology
mapping and explicit alignment/angle/typing definitions first. No carbon-contact
or generic N/O-contact surrogate is labeled as those observables. This is the
issue's “where feasible” boundary, not a claim that such features are impossible.
Native features remain unchanged and the two feature sets are not silently mixed.
