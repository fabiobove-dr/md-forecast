# Canonical data schema

Schema **1** separates immutable Pydantic v2 metadata from NumPy/Arrow frame
values. It supports the `misato` and `mdbind` namespaces without dataset- or
model-specific branches. No model dependency is installed. Source adapters
remain responsible for verifying source identity, units, and feature semantics.

## Boundaries

`md_forecast.data.schemas` defines `Provenance`, `SystemIdentity`,
`TrajectoryIdentity`, `TimeAxis`, `FeatureDefinition`, `DatasetConfig`, and
`ExperimentConfig`. `TrajectoryManifest` combines the identity/provenance/time
fields into one flat record. Unknown fields, empty identifiers, nonfinite
metadata numbers, and unsupported enum values fail validation. Identity fields
must be exact canonical values, not machine paths.

The manifest includes:

| Fields | Meaning |
| --- | --- |
| `dataset_id`, `dataset_version`, `source_record` | Dataset namespace, release, and upstream record |
| `trajectory_id`, `system_id`, `replicate_id` | Stable trajectory, biological system, and replica identities |
| `pdb_id`, `protein_id`, `ligand_id` | Explicit identities, or `null` when unavailable; never fabricated |
| `split_group_id`, `split` | Indivisible group and `train`/`validation`/`test` assignment; `null` before splitting |
| `frame_count`, `time_unit`, `sampling_status` | Retained frame count and explicit time interpretation |
| `frame_interval_ps`, `duration_ns`, `sampling_note` | Known interval/span, or `null`; assumption rationale when applicable |
| `feature_set_version` | Version of the observable definitions |
| `source_checksum`, `license_id`, `provenance_uri` | Actual input artifact checksum, license identifier, and public HTTP(S) source URL |
| `schema_version` | Canonical contract version, currently `1` |

Checksums are lowercase `md5:<32 hex>` or `sha256:<64 hex>`. A sample artifact
must retain its own checksum, not the checksum of a full dataset it resembles.
HTTP(S) provenance excludes local filesystem URLs and embedded credentials.
There are no filesystem path fields in persisted metadata. A license identifier
records provenance; it is not a declaration that all upstream usage restrictions
have been legally cleared.

`stable_system_id(dataset_id, source_system_id)` namespaces exact source IDs.
`stable_trajectory_id(system_id, source_trajectory_id, replicate_id)` percent-
encodes each component separately, avoiding delimiter collisions, process hash
randomization, and machine-dependent paths. Adapters normalize upstream aliases
and casing **before** generating IDs. Replica changes do not change system IDs.

Feature IDs are extensible lowercase column names, not an enum of MISATO keys.
Definitions carry a description, scientific definition, and unit. Version 1
admits `angstrom`, `angstrom_squared`, `kcal_per_mol`, `dimensionless`, and
`count`. `time` is reserved. Matching names/units alone do not establish
cross-dataset semantic equivalence.

`DatasetConfig` binds a dataset release to ordered, uniquely named features.
`ExperimentConfig` binds that configuration to a nonnegative seed and an
optional unique trajectory selection. Window/model/evaluation settings are
outside this contract and will arrive in their own issues.

## Time and numerical storage

All axes start at zero **relative to the first retained frame**; this says
nothing about the simulation's absolute production origin.

- Unknown physical sampling: `time_unit="frame"`,
  `sampling_status="unavailable"`, consecutive frame indices, and both
  `frame_interval_ps` and `duration_ns` null.
- Physical sampling: `time_unit="ps"` or `"ns"`, status `verified` or
  `assumed`, and a declared retained span `duration_ns`. An assumption requires
  a nonempty `sampling_note`; it must never be presented as verified timing.
- Uniform physical sampling declares a positive `frame_interval_ps` and a
  duration of `(frame_count - 1) * frame_interval_ps / 1000`. Nonuniform physical
  timestamps may omit the interval but must still match the declared span.

The MISATO audit has **not** established a verified frame interval. Its default
representation is therefore frame indices. The reported retained simulation
length is not interchangeable with the span between saved frame endpoints.
Any explicit nominal-interval policy must retain `assumed` status and rationale.

`md_forecast.data.series.to_arrow(manifest, dataset, time, values)` accepts
float64 NumPy arrays of shapes `(n_frames,)` and `(n_frames, n_features)`.
The Arrow table stores `time` followed by feature columns in configuration
order, all non-null float64. One serialized registry is embedded under schema
metadata key `md_forecast.series`, preserving feature units/definitions and
provenance through Parquet round trips. Missing/invalid observations must be
handled explicitly upstream before publication; there is no silent filling.

`validate_series` checks shape, names, types, finite values, strictly increasing
time, span, and any declared uniform interval using vectorized NumPy operations.
Numerical comparisons use central tolerances `TIME_RTOL=1e-9`, `TIME_ATOL=1e-10`.
There are no Pydantic objects per frame. `write_series(path, table)` and
`read_series(path)` operate on **one trajectory** at a time, not the full dataset.
Callers iterate trajectory files to bound dataset memory. Streaming a single
trajectory larger than memory is not implemented in this initial contract.

## Registry and publication

`md_forecast.data.registry.Registry` stores one dataset configuration plus a
tuple of trajectory manifests. It rejects duplicate trajectory IDs, dataset/
feature-version mismatches, conflicting biological identities for one system,
and split groups assigned across partitions. Multiple replicas of a consistent
system are valid. This is a trajectory-level index, not a frame-level table.

`write_registry(path, registry)` sorts trajectories by ID and writes JSON;
`read_registry(path)` validates it at the boundary. Parents must already exist.
JSON and Parquet writers validate before publication and use a temporary file
in the destination directory, fsync, and atomic replacement. Failed writes
preserve previous output and remove their temporary file. I/O failures carry
actionable `DataContractError` messages with chained causes.

## Versioning and migration

Version 1 is the first canonical format; there is no earlier canonical format
to migrate. JSON registry, dataset config, experiment config, and manifest
versions are independently checked. Parquet embeds the same versioned metadata.
Readers reject unsupported versions rather than guessing or silently dropping
fields. Changes to units, timing, identity semantics, or required fields need a
new schema version and a deliberately implemented migration with round-trip
and scientific-invariant tests. Future migrations must preserve originals and
never infer missing physical metadata without explicit provenance.

Changing a feature's scientific meaning also requires a new
`feature_set_version`; changing source bytes requires recording the new checksum
and source version. An extensible feature name is not permission to relabel an
old feature definition in place.

## Verification

`tests/test_canonical_data.py` uses synthetic MISATO/MDbind metadata, verifies
JSON and Parquet round trips, deterministic identities/registry ordering,
version/unit rejection, sampling invariants, malformed arrays/tables, and
preservation of existing files on interrupted/invalid publication. These tests
prove the contract, not the correctness of an unimplemented source adapter.

```sh
uv run --locked pytest tests/test_canonical_data.py --no-cov
```
