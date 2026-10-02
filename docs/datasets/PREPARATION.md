# Splits, windows, and preprocessing

Issue #7 provides model-independent, versioned preparation contracts. No new
dependency, model training, or Chronos integration is required. These APIs use
the [canonical registry/series](CANONICAL_SCHEMA.md) and can consume the
[MISATO adapter](MISATO_ADAPTER.md)'s output.

## Split before windowing

`build_split(registry, SplitConfig(...))` produces a self-contained
`SplitManifest`: source registry, protocol configuration, group labels, and
one assignment per trajectory. Its constructor/read boundary recomputes the
declared protocol rather than trusting serialized assignments.

- `mode="official"`: preserve existing train/validation/test labels exactly;
  reject missing labels and conflicting dependent trajectories. Ratios are
  prohibited; an official subset may legitimately have an empty partition.
- `mode="grouped"`: require explicit train/validation/test ratios and a seed.
  Sort dependency components, shuffle with a local seeded standard-library RNG,
  and apportion **component counts**, not window or trajectory counts, using
  largest remainders (ties in train/validation/test order). Reject an empty
  requested positive-ratio partition rather than silently dropping it.
- `group_field`: default `system_id`; canonical `split_group_id`, `pdb_id`,
  `protein_id`, or `ligand_id` are also supported. Missing identities fail.
  A custom target/family field requires an explicit, complete
  `trajectory_id -> group_label` mapping. It cannot override canonical identities.

Grouping is the transitive union of shared system identity, original split
group, and configured group label. Therefore replicas or dependent complexes
cannot cross partitions even when the chosen field is finer than another
dependency. Component IDs are stable minimum trajectory IDs, not assertions
about protein/chemotype identity. The builder sorts source trajectories and
assignments, making results and hashes independent of input record order.

Changing the split protocol defines a different benchmark. Do not regroup an
already examined final test set into training and claim it remained untouched.
Target/family claims require verified external identity labels; merely naming
a custom group field supplies no biological evidence.

## Lazy windows and physical time

`iter_windows(split_manifest, WindowConfig(...), partition)` requires a
validated pre-existing split. It preflights every selected trajectory/grid
cell before yielding the first index. Windows are immutable index objects,
not copied numerical arrays:

```text
context = X[start : start + C]
target  = X[start + C : start + C + H]
```

Contexts/horizons and stride are explicit configuration, with no implementation
defaults for their sizes or seeds. Grid cells and trajectories are sorted;
starts advance by `stride_frames`. Every index carries source system/group,
partition, split/config hashes, and sampling status. `validate_window` rejects
forged or stale labels/hashes, unknown IDs, wrong stride/grid, and invalid bounds.
All windows of one trajectory inherit the same partition; overlap cannot
create a different split.

Frame grids require positive integer `C` and `H`. Zero/negative/fractional
frame lengths, duplicates, null/short trajectories, or `C + H > frame_count`
fail explicitly. Empty partitions yield no windows.

Physical grids (`unit="ps"` or `"ns"`) require a declared uniform interval and
verified sampling. `allow_assumed_time=True` can explicitly permit an already
documented assumed interval; it never supplies an interval or marks it verified.
Unknown spacing and nonuniform sampling cannot be silently converted.

Physical context values mean the first-to-last observed span `(C - 1) * dt`;
physical horizons mean the last observed-to-last forecast span `H * dt`. Thus
160 ps context at 80 ps spacing contains **3** observed frames; 160 ps horizon
contains **2** future frames. Zero physical context span represents one sample.
Durations must align to the sampling grid; rounding a misaligned request is
prohibited. Uniform frame grids also report these spans and sampling status;
unknown spacing leaves spans null. MISATO's default remains frame-only.

## Explicit scaler fitting scopes

`ScalingConfig` requires ordered feature IDs and a scope:

- `training-contexts`: `fit_training_scaler` derives the union of observed
  training-context intervals from the declared window grid. Overlapping frames
  count once. Validation/test series are never loaded, and forecast-only frames
  are excluded. Context intervals are merged with a bounded k-way stream,
  avoiding a list of every window. A frame observed in a later training context
  may also be an earlier training window's target: this is dataset-level
  training preprocessing, not online per-origin normalization.
- `context-local`: `fit_context_scaler` validates a particular index and fits
  **only** its observed slice. It can process a validation/test context without
  inspecting its target values or fitting shared held-out statistics. Use this
  scope when per-origin causal normalization is required.

Both APIs require a loader returning canonical Arrow series; source manifests
and feature definitions must exactly match the split registry. Numerical work
uses NumPy arrays and merged sufficient statistics, with population standard
deviation (`ddof=0`). Constant channels use scale 1, not a guessed epsilon.
Missing/nonfinite observed inputs and numerical overflow fail; there is no
imputation. Fitting checks metadata/layout globally but inspects numerical
values and timestamps only inside fitting intervals. It materializes one
observed interval at a time. Canonical publication and `read_series` separately
validate the complete series; a caller choosing that loader performs whole-file
quality control before fitting, not future-dependent normalization.

`ScalerMetadata` stores scope, features/definitions/units, split and window
hashes, unique half-open fit regions, sample count, means, and scales.
`transform(values, metadata)` never fits statistics; `inverse=True` restores
original units. Array columns must follow the metadata's declared feature order.
Use `validate_scaler` when reusing persisted statistics with another protocol;
it rejects mismatched data/hashes and undeclared fitting regions.

## Persistence and experiment linkage

`write_metadata(path, model)` writes an atomic integrity envelope containing
the payload and full SHA-256 of canonical JSON (sorted keys, finite numbers).
`read_metadata(path, Model)` verifies the checksum and validates the payload's
schema. Split/window/scaling/preprocessing contracts are version 1 and reject
unsupported versions. Parents must exist. Failed publication preserves old
files; machine-local runtime paths are not part of preparation metadata.

`bind_experiment(experiment, split, windows, scaler)` validates linkage and
returns an `ExperimentConfig` carrying `split_hash`, `window_config_hash`, and
`preprocessing_hash`. Existing experiment records remain readable with these
optional fields unset; older installations must upgrade to read enriched
records. Integrity hashes identify artifacts, not scientific validity or
protection against an author deliberately rehashing fabricated statistics.

```python
from pathlib import Path
from md_forecast.data.registry import read_registry
from md_forecast.data.splits import SplitConfig, build_split
from md_forecast.data.windows import WindowConfig, iter_windows
from md_forecast.core.constants import Split

registry = read_registry(Path("data/processed/misato-native-sample/registry.json"))
split = build_split(registry, SplitConfig(mode="official", seed=42))
# Illustration, not a scientifically selected or approved benchmark grid.
windows = WindowConfig(contexts=(20, 40), horizons=(5, 10), stride_frames=1)
for index in iter_windows(split, windows, Split.TRAIN):
    # Read one canonical series and use index.context_slice / index.target_slice.
    pass
```

## Verification evidence and limits

Synthetic tests prove transitive dependencies, official/grouped reproducibility,
reordered-input hash stability, variable grids, exact/short/null boundaries,
physical-time semantics/rejection, forged-index rejection, constant channels,
inverse scaling, serialization, and metadata linkage. Leakage mutation tests
deliberately fit full trajectories or context-plus-target and verify that the
guard assertions fail. Perturbing held-out series or forecast-only values
cannot change training statistics; perturbing a target cannot change its
context-local statistics.

A real preparation smoke test on the published 20-system sample used the
illustrative grid above and official assignments. It generated 4,826 training
indices and 254 test **indices only**, reading 19 training series and **no test
values** (`16PK` stayed untouched). Its training-context union contains 1,805
frames (95 per training system). Split, window, scaler, and linked experiment
artifacts round-trip under ignored `data/processed/misato-preparation-smoke`.
There are zero validation systems in this sample's official subset; this smoke
test cannot select models or establish benchmark performance. Physical durations
remain unknown. Issue #8 must justify a development grid and isolate final test
values; no forecast skill or scientific acceptance is claimed here.
