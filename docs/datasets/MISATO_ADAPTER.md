# MISATO native observable adapter

Issue #6 implements the [audited source contract](MISATO.md), not new physics
or a full-dataset QC result. The adapter lives in
`md_forecast.data.public.misato` and uses the
[canonical schema](CANONICAL_SCHEMA.md). Its feature set is `misato-native-v1`.

## Run a selected subset

Acquire the official split lists with `md-forecast download`. Supply an
already acquired, complete HDF5 artifact and **explicit** uppercase PDB IDs:

```sh
uv run --locked md-forecast extract-misato \
  --input data/external/misato/MD.hdf5 \
  --splits data/external/misato \
  --output data/processed/misato-native-subset \
  --systems 10GS 11GS 1A3E
```

This command does not download the 132.84 GB full MD artifact. For the pinned
upstream sample, use `--artifact sample`; it selects the **sample's own**
audited size, SHA-256, and URL, never the full MD checksum:

```sh
uv run --locked md-forecast extract-misato \
  --artifact sample \
  --input data/external/misato-audit/tiny_md.hdf5 \
  --splits data/external/misato \
  --output data/processed/misato-native-sample \
  --systems 10GS 11GS 13GS 16PK 184L 185L 186L 187L 188L \
            1A07 1A08 1A09 1A0Q 1A1B 1A1C 1A1E 1A28 1A2C 1A30 1A3E
```

The sample's pinned URL/checksum are in `configs/datasets/misato.yaml`.
Source bytes and generated outputs remain ignored local data. Both commands
use the repository source config by default; an installed wheel or another
working directory needs `--source /path/to/misato.yaml`.

Python callers use `load_misato_source(path)`,
`ExtractionConfig(input_path=..., splits_dir=..., output_dir=...,
system_ids=(...), artifact="md" | "sample")`, and
`extract_misato(source, config)`. Runtime paths are not persisted.
Selection cannot be empty, duplicated, lowercase, or contain path syntax.
The output directory must not already exist, including dangling symlinks.
Choose a new directory for a rerun; the adapter does not delete existing data.

## Validation and bounded I/O

Before reading HDF5, the adapter verifies the complete input size/checksum and
all three official split files, using the same streaming integrity validator
as acquisition. Hashing the full MD artifact reads its entire byte stream once;
it is memory-bounded but may dominate the runtime for a tiny selected subset.
Receipts alone are not treated as proof that current bytes are intact.

Official split IDs must be valid, unique, and disjoint. Split assignments and
list checksums are retained. Unknown requested IDs are excluded with a reason,
not assigned to an invented partition. Each system has a namespaced stable ID;
its split group is that system ID. The opaque replica label `native` identifies
the one source export per PDB group, not a verified simulation seed or physical
replica number. Protein/ligand identifiers stay null because the export does
not provide verified canonical identities for them.

HDF5 is opened read-only. Each selected group is accessed lazily; only four
100-frame native datasets are read. Their shapes and float64 dtypes must match
the audit. Coordinates, atom tables, and unselected groups are never loaded.
External/soft links are rejected rather than followed. Dataset slices are
HDF5 hyperslab reads, not full-file materialization.
[h5py dataset I/O](https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data).

The fixed channel order and source keys are centralized in `NATIVE_BINDINGS`.
The YAML supplies scientific descriptions and definitions. Source key/unit/
shape changes or newly asserted timestamp metadata fail closed and require a
new audit; the adapter does not guess conversions. Native RMSD is not relabeled
protein-aligned RMSD, and interaction energy is not called binding free energy.

## Time and exclusions

The current verified source has no timestamp vector or confirmed spacing.
Outputs therefore retain consecutive frame indices `0..99`,
`time_unit="frame"`, `sampling_status="unavailable"`, and null physical interval/
duration. **The adapter does not use the nominal 80 ps interval.** No original
physical timestamps are invented or discarded. Physical-time benchmark claims
remain subject to the [audit's timing limitation](MISATO.md#timing-verified-facts-and-explicit-uncertainty).

Native observable units are preserved in the embedded feature definitions:
RMSD/distance in angstrom, buried SASA in angstrom squared, interaction energy
in kcal/mol. Numerical series must be finite, aligned, and monotonic under the
canonical validator. A missing channel, wrong shape/dtype, linked dataset,
nonfinite value, or invalid group excludes the **whole trajectory**, with an
explicit QC reason. There is no interpolation, zero filling, partial-feature
publication, coordinate-derived fallback, or periodic-imaging repair.

## Outputs and failure safety

A successful output directory contains:

- `<PDB_ID>.parquet` for every accepted system, with exact source-native values
  and canonical provenance/units/feature version in Arrow metadata.
- `registry.json`: accepted trajectory manifests and ordered feature definitions.
- `qc.json`: source provenance, official split checksums, requested IDs,
  trajectory-ID-to-relative-filename mapping, and missing/dropped IDs with reasons.

Counts follow directly from the lengths of `requested`, `exported`, and `issues`;
each requested system has exactly one accepted or excluded outcome. No machine-
absolute paths are stored. Every manifest records the actual input artifact's
checksum and public URI, including the distinction between full and sample data.

The whole output is built in a temporary sibling directory. JSON and Parquet
files use their atomic writers; only a completed bundle is renamed into place.
Input corruption, malformed split metadata, HDF5 read errors, or storage errors
abort publication and clean temporary outputs. They are not mislabeled as
scientific trajectory exclusions. Existing destinations are preserved.
Use one writer per output destination and do not create/change that destination
while extraction runs; concurrent publication is not part of this adapter.

## Verification evidence

On 2026-10-02 the command above processed all 20 systems in the pinned
96,598,231-byte sample against the real verified official split lists:

| Check | Result |
| --- | --- |
| Accepted / excluded | 20 / 0 |
| Frames per system / channels | 100 / 4 |
| Native values compared exactly against HDF5 | 8,000; all equal |
| Official assignments | 19 train, 0 validation, 1 test |
| Output bundle size | 222,939 bytes |
| Elapsed wall time / peak RSS | 0.86 s / 113,276 KiB |
| Actual source checksum | `sha256:554d20ea0822949e1a5b0dc1826f50b87e71de5a7ed83ac1fb5a2e68c64547de` |

Timing includes CLI startup, checksum verification, and export, measured with
`/usr/bin/time -v` on this workstation with warm source bytes. It is not a
throughput claim for the full MD artifact. Every output was independently read
back through `read_series`; provenance, units, frame indices, and official
assignments were checked. This is adapter-fidelity verification, not analysis
or model selection on the final test system.

Recheck source fidelity without reading coordinates:

```python
from pathlib import Path
import h5py
import numpy as np
from md_forecast.data.public.misato import ExtractionReport, NATIVE_BINDINGS
from md_forecast.data.series import read_series

root = Path("data/processed/misato-native-sample")
qc = ExtractionReport.model_validate_json((root / "qc.json").read_text())
with h5py.File("data/external/misato-audit/tiny_md.hdf5", "r") as source:
    for filename in qc.exported.values():
        table = read_series(root / filename)
        system = Path(filename).stem
        for feature, (key, _) in NATIVE_BINDINGS.items():
            np.testing.assert_array_equal(
                table[feature].to_numpy(), source[system][key][:]
            )
```

Default CI uses only synthetic fixtures. Tests cover exclusions, actual checksum
failure, split overlap, source-contract changes, existing destinations, invalid
HDF5/storage failures, and CLI behavior. A sparse fixture declares a 24 GB
coordinate payload; instrumented reads prove neither coordinates nor an
unselected system are accessed. Full-dataset membership and missingness still
need their own full-data QC; sample success is not evidence of zero full-data
exclusions. Derived contacts/H-bonds and model integration remain out of scope.
