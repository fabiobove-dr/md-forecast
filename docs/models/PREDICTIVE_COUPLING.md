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
OMP_NUM_THREADS=1 uv run --locked --extra structural --extra ml \
  python examples/prepare_coupling.py extract \
  --output data/processed/mdbind-coupling-regions
OMP_NUM_THREADS=1 uv run --locked --extra structural --extra ml \
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
