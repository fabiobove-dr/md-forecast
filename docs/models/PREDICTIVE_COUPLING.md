# Predictive coupling between geometric regions

Issue #16 tests whether pocket history A improves forecasting distal-region
history B: `B_past -> B_future` versus `[B_past, A_past] -> B_future`. The result
is **predictive coupling**, never a claim of physical signal transfer, causality
or established allostery. This is an exploratory extension on seen complexes,
separate from the immutable [core MVP report](MVP_REPORT.md).

## Admitted region observables and independent selection

The source cohort is the six audited MDbind complexes/60 replicas in
`configs/datasets/mdbind-common.json`. Source terms, SHA-256 identities,
PDB/AMBER atom-order checks and the actual XTC grid are documented in the
[external-validation audit](EXTERNAL_VALIDATION.md) and
[MDbind dataset audit](../datasets/MDBIND.md). No native SASA, masses, energy
or alignment convention is reused or inferred.

`configs/features/coupling-regions.json` fixes the rules. The resulting explicit
atom lists and reference PDB/topology checksums are frozen in the reviewed
`configs/features/coupling-selection.json` **before reading regional XTC
values**. Only each complex's replica-one source PDB supplies selection
coordinates; it is a source simulation structure, not asserted to be a crystal
structure. It belongs to the training portion of this experiment.

For each protein residue before the reviewed final ligand segment, compute its
minimum heavy-atom distance to the ligand in that static PDB. Heavy means atomic
number >1. Protein membership is the pinned MDTraj `is_protein` predicate;
exactly one atom named `CA` must exist per admitted residue.

- Region A: C-alpha atoms of residues with distance <=6 Å.
- Region B: C-alpha atoms of residues with distance >=12 Å.
- Residues between the thresholds are omitted. Each region needs >=3 C-alpha
  atoms; selections must be distinct, disjoint and nonnegative.
- Replica topology SHA-256 must equal the frozen reference topology. All
  per-replica source bytes/metadata and PDB/AMBER order are independently checked.

| Complex | A pocket C-alpha atoms | B distal C-alpha atoms |
| --- | ---: | ---: |
| 1A0T | 20 | 1143 |
| 1AVP | 34 | 101 |
| 1BXR | 37 | 940 |
| 1CSH | 36 | 726 |
| 1D4W | 34 | 26 |
| 1D4Y | 33 | 96 |

Each region observable is unweighted C-alpha radius of gyration:
`sqrt(mean_i ||x_i - mean_j x_j||²)` in Å. MDTraj XTC nm values are multiplied
by ten. This definition is translation/rotation invariant and requires no
alignment or assumed atomic mass. Raw Cartesian coordinates are used without
PBC repair. A region may span multiple protein chains or disconnected residue
sets: its radius is a coarse geometric proxy, not a validated functional domain.
Atom membership stays fixed across frames/replicas. All 50 retained timestamps
must be present with exact 200 ps spacing; the relative axis subtracts the
stored origin, retaining the actual 9.8 ns span. B is `distal_rg`, A is
`pocket_rg`, both native Å, with formula/selection hashes in the feature version.

Frozen region identity:
`sha256:5fc231b7559c0dad9b42a67db03e0ebf3cc524ed145ea64b7f1aca3c896b3f21`.

## Frozen experiment and controls

`configs/benchmarks/predictive-coupling.toml` fixes:

- TRAIN replicas 1–8, TEST replicas 9–10 for every complex, split before windows.
  Biological complex identities remain the uncertainty groups: 48 TRAIN
  replicas, 12 TEST replicas, six complex groups.
- C40/H10, stride 10, one window per replica. Observed sample span is 7.8 ns;
  lead 1 is 0.2 ns and lead 10 is 2 ns after the final context sample.
- Target B alone, then the same B plus A input. Identical future B labels,
  windows and group membership are checked before paired effects.
- Persistence, context mean, linear extrapolation, AR(1), VAR(1), and pinned
  Chronos-2 zero-shot. AR/VAR use fixed ridge `1e-6`; no search or fine-tuning.
  B-only VAR is the restricted one-variable autoregression; B+A VAR is the
  unrestricted lagged conventional comparator. Fits use each observed context
  only and recursively forecast the same ten leads.
- TRAIN-context scalers supply scaled-error denominators only; adapters receive
  native-unit contexts, as in the existing benchmark. No test future fitting.
- Paired B+A-minus-B MAE effects per lead and whole horizon for Chronos and VAR,
  resampling six complexes with seed 42, 100,000 bootstrap draws, family 200,
  95% confidence and >=10 tail samples. The complete planned comparison count
  is 187 (165 within-cell comparisons plus 22 paired input effects).

Conventional diagnostics additionally use **only** each held-out window's
observed context, never target labels. Correlation is Pearson at lags -5..5;
positive lag correlates `A(t)` with `B(t+lag)`. Constant-channel correlation is
unavailable. Restricted B and joint B+A lagged regressions provide descriptive
context SSE gain after context-local standardization. Perfect/zero restricted
residual SSE gives an unavailable ratio. This descriptive fit statistic is not
an inferential Granger test: short contexts, autocorrelation, slow sampling and
possible nonstationarity prevent defensible asymptotic p-values here. Forecast
incremental gain is independently measured on future labels.

No lag/region/threshold/horizon or favorable complex is selected from test
results. These replicas appeared in earlier, different-observable experiments;
this extension is not a newly untouched external-confirmation cohort. A gain
would remain limited to these geometric proxies and seen-system replicas.
Negative effects and null intervals are retained. No predictive advantage over
all statistical references or probabilistic baselines is presumed.

## Reproduce

With the audited source bytes already acquired and the reviewed selections
committed, run from the repository root:

```sh
OMP_NUM_THREADS=1 uv run --locked --extra structural --extra chronos \
  python examples/prepare_coupling.py extract \
  --output data/processed/mdbind-coupling-regions
OMP_NUM_THREADS=1 uv run --locked --extra structural --extra chronos \
  python examples/run_predictive_coupling.py \
  data/processed/predictive-coupling-v1
```

The extraction publishes the complete cohort atomically, preserving source
identities/terms/units/timestamps and embedding frozen regions/source metadata.
The runner requires committed implementation/configuration, freezes both
benchmark manifests and input hashes before scoring, then publishes complete
paired benchmark reports, `effects.parquet`, `context-lag-diagnostics.parquet`
and a completion marker together. Existing outputs are rejected; choose new
versioned directories. These generated datasets/reports remain ignored.

To independently verify static selection reproduction in a **new** path:

```sh
uv run --locked --extra structural python examples/prepare_coupling.py freeze \
  --regions data/manifests/coupling-selection-recheck.json
```

The freeze operation reads source PDB coordinates and topology/byte identities,
with no XTC value decoding. Compare its envelope identity with the reviewed
region hash. No downloads or GPU are needed for this operation.

## Verified real-data results

Two complete runs (`predictive-coupling-v1` and `predictive-coupling-repeat`)
produced identical source/input/config identities, benchmark reports, prediction
labels/values, metrics, comparisons, paired effects and context diagnostics.
Every scored cell has 12 held-out windows/replicas and six complex groups.

**No corrected interval for added-A gain excludes zero at any lead 1–10,
for either Chronos or the conventional restricted/unrestricted VAR control.**
This does not confirm predictive coupling for these regions. At lead 6, a
Chronos marginal interval is negative, but its corrected upper bound remains
positive; it is not corrected evidence of gain.

Whole-H10 target B MAE in Å, from the aggregate `distal_rg`, `mae`, `step=0`
rows in each saved `cell-0/metrics.parquet`:

| Model | B-only | B+A |
| --- | ---: | ---: |
| persistence | 0.083592438 | 0.083592438 |
| context-mean | 0.070958494 | 0.070958494 |
| linear-extrapolation | 0.089039253 | 0.089039253 |
| autoregression | 0.068956921 | 0.068956921 |
| var | 0.068956921 | 0.068950470 |
| chronos-2 | 0.075643994 | 0.074495776 |

Paired whole-H10 effects from `effects.parquet`; negative differences favor
B+A, but all reported corrected intervals include zero:

| Model | B+A minus B MAE (Å) | Corrected 95% CI (Å) |
| --- | ---: | --- |
| chronos-2 | -0.001148218 | [-0.005484923, 0.000913366] |
| var | -0.000006451 | [-0.001063604, 0.000939969] |

Complete leadwise added-A effects (Å), with corrected intervals:

| Lead (ps) | Chronos delta / CI | VAR delta / CI |
| --- | --- | --- |
| 200 | 0.0023069 [-0.0024341, 0.0078842] | -0.0018294 [-0.0136643, 0.0056059] |
| 400 | 0.0010184 [-0.0054092, 0.0109868] | 0.0006002 [-0.0050553, 0.0062403] |
| 600 | -0.0021441 [-0.0065369, 0.0016047] | 0.0008623 [-0.0018916, 0.0042906] |
| 800 | 0.0023355 [-0.0058139, 0.0084943] | 0.0000572 [-0.0011341, 0.0012928] |
| 1000 | -0.0008434 [-0.0086579, 0.0052145] | 0.0000349 [-0.0009415, 0.0010340] |
| 1200 | -0.0040728 [-0.0120961, 0.0000014] | 0.0002122 [-0.0006871, 0.0013490] |
| 1400 | -0.0035895 [-0.0118925, 0.0020866] | 0.0001655 [-0.0007608, 0.0012831] |
| 1600 | -0.0021484 [-0.0062589, 0.0031657] | 0.0001043 [-0.0007451, 0.0011476] |
| 1800 | -0.0017566 [-0.0121543, 0.0048173] | -0.0001283 [-0.0008058, 0.0009322] |
| 2000 | -0.0025881 [-0.0122300, 0.0030642] | -0.0001435 [-0.0008245, 0.0008533] |

Observed-context lag diagnostics are descriptive. For each lag, correlations
are averaged over the two replicas within each complex, then the median over
six complex means is displayed. No correlation confidence/p-value or preferred
lag is inferred. Complete per-replica values remain in the saved table.

| A-leading-B lag (ps) | Median complex-mean Pearson correlation |
| --- | ---: |
| -1000 | -0.0345065 |
| -800 | 0.0122209 |
| -600 | 0.0381302 |
| -400 | -0.0151363 |
| -200 | 0.0614961 |
| 0 | 0.0203539 |
| 200 | 0.0159695 |
| 400 | -0.0445981 |
| 600 | 0.0117385 |
| 800 | 0.0379952 |
| 1000 | -0.0517109 |

Median complex-mean **in-context** SSE gain fraction is 0.0162817.
This training-fit description does not replace the held-out error comparison.

Frozen provenance and artifact identities:

- Experiment commit: `41e45f17e74a75b4f3968c8352c142ff316a3d85`.
- Experiment configuration: `sha256:a03e14eed0db843ec51974a12ef2d4c9a52963e0d3b5b9179fb7133d43839401`.
- b_only benchmark: `sha256:d2bee5da81fc5a4ee3653a9406fdca687106aae32a8f7eb2a6baf23db4f8498f`.
- a_b benchmark: `sha256:2d763653430bebe591897b092e5b3bdb0099cd3ce68b8065b7e640c2526b6717`.
- effects.parquet: `sha256:f7e242995c9138d9989b636855e0da5e9fea78d549db748d3eeb10bdc1958914`.
- context-lag-diagnostics.parquet: `sha256:dfee846c8ea4a2f5cba8317745bfe66ed90cc3e346d432fab027035e9309653a`.

Regenerated horizon/calibration figures are saved with each benchmark.
The original runtime tables record adapter inference/host conversion and peak
GPU allocation rather than full extraction/experiment runtime.
For `b_only`, Chronos adapter total is 0.6744 s,
peak allocated 486658048 bytes and
peak reserved 503316480 bytes.
For `a_b`, Chronos adapter total is 0.1303 s,
peak allocated 486864384 bytes and
peak reserved 503316480 bytes.
The run used Python 3.14.8, MDTraj 1.11.1.post2 and the locked Torch/Chronos
stack on the RTX 4070 Laptop GPU (8 GB); exact code/lock/model/hardware
identities reside in the saved cell manifests.

These measurements cover a small reused six-complex cohort and coarse regional
radius proxies. Null results do not establish lack of molecular communication.
Software tests are synthetic integrity/geometry/leakage checks and do not
constitute biological evidence.
