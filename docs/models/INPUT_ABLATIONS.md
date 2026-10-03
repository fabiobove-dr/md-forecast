# Controlled input ablations

Issue #13 compares **one common target, ligand RMSD**, across four frozen
past-input sets: RMSD alone; RMSD + ligand/receptor COM distance; all four
verified native channels; native channels + contact count, fraction of reference
contacts and minimum heavy-atom distance. The pinned zero-shot Chronos-2 weights,
seed, C40/H10, stride 10, grouped split, quantiles and evaluation policy stay fixed.
Persistence is scored alongside every variant.

The added channels are **jointly forecast auxiliary observables**, not static or
future-known covariates. Only the shared RMSD target is used to compare input
sets; averaging errors across unrelated units would change the question. The
optional static/dynamic-covariate experiment is not performed: this adapter
accepts context-only joint time series, and no new metadata semantics are assumed.
Fine-tuning comparisons remain in [the existing experiment](FINETUNING.md).

## Reproduce

Acquire/extract the audited native sample as described in the
[MISATO adapter](../datasets/MISATO_ADAPTER.md), then extract all 19 official
TRAIN systems with the versioned [structural configuration](../datasets/STRUCTURAL_FEATURES.md).
The default input directories are `data/processed/misato-native-sample` and
`data/processed/misato-structural-verified`; override them with `--native` and
`--structural`. Native and structural records must have identical source, split,
frame and time identities, apart from their explicitly combined feature version.
The official TEST record 16PK is filtered before reading any payload.

```sh
uv sync --locked --extra chronos --extra structural
uv run --locked python examples/run_input_ablations.py data/processed/input-ablations
```

Use a committed implementation and fresh output directory. Local editor files
are allowed; tracked modifications and untracked implementation files are refused
when freezing code provenance. `configs/benchmarks/input-ablations.toml` is the
validated, versioned experiment configuration. No dataset acquisition or training
occurs in the command.

The seed-42 70/30/0 split holds out six of the 19 official TRAIN systems for
**development validation**; 13 systems supply train-only scaling. These same six
systems were previously used for checkpoint selection in #12. They are neither
the official VAL partition nor an untouched final TEST set. This experiment
compares a fixed zero-shot model, with no parameter selection using its results.
Each variant scores every legal window, with exactly matching trajectory,
start, lead, future frame and native-unit target labels. A mismatch fails.

Before scoring, the command saves the validated configuration, source-value
hashes and all cell manifests (split, feature definitions/units, train scaler,
model revision/settings/artifact IDs, code commit, lockfile and GPU provenance).
Each variant is published and checksum-verified with the unchanged
[benchmark API](BENCHMARK.md), retaining predictions, point/probabilistic metrics,
per-horizon plots and runtime/VRAM telemetry. An existing output is refused.
`COMPLETE` is written only after all variants and paired effects succeed; an
interrupted directory without it is incomplete and must not be used as a result.
Source bytes, predictions and generated reports stay ignored.

`effects.parquet` reports variant-minus-RMSD-only MAE for each lead 1–10 and
whole-horizon step 0. Each complex supplies one paired mean after averaging its
windows. The existing bootstrap resamples complexes; windows are not independent
samples. Negative differences favor the added inputs. The predeclared
187-comparison family covers 154 model-vs-persistence comparisons across all
scored channels and 33 common-target input-set comparisons. The 1,000-resample
budget resolves marginal descriptive intervals but **cannot resolve corrected
tails**; corrected bounds must be null with an explicit status. Marginal bounds
are also retained and must not be interpreted as corrected superiority evidence.

The tiny audited sample lacks verified physical timestamps; horizons remain in
frames. No calibration acceptance threshold, validation-selected strongest
baseline or full-data test is supplied by this development experiment. Null,
negative and mixed results remain visible; #14/#17 must independently evaluate
external generalization and final scientific claims.

## Actual development evidence — 2026-10-03

Implementation `966f0f1749cf20cfa9807b312d182f9d6bbae1f1`
ran all four variants on the same 36 windows from six VAL complexes. All four
complete bundles passed scientific-table checksum verification. A second
independent execution produced identical predictions, metrics, comparisons,
paired effects and report IDs. Official TEST 16PK was never loaded.

The mean RMSD MAE (Å) was 0.301950 for RMSD-only, 0.300665 for RMSD+COM,
0.292750 for all native channels and 0.287099 for native+geometry. Persistence
was 0.404688 in every variant. Added inputs give a small mean reduction here,
but the paired whole-horizon **marginal descriptive** 95% intervals all include
zero: COM [-0.005707, +0.002288], native [-0.028526, +0.004388],
geometry [-0.040743, +0.001605] Å. All corrected bounds are unavailable with
`insufficient-bootstrap-resolution`. This is inconclusive evidence of input-set
benefit; no input set is declared scientifically superior or selected for final
external evaluation from this pilot.

Variant-minus-RMSD MAE differences (Å), retaining worsening as well as improving
lead steps; step 0 is the whole-horizon summary:

| Lead | COM − RMSD | Native − RMSD | Geometry − RMSD |
| --- | ---: | ---: | ---: |
| 0 | -0.001286 | -0.009200 | -0.014852 |
| 1 | +0.001245 | -0.003439 | -0.014944 |
| 2 | -0.004178 | -0.007090 | -0.025820 |
| 3 | +0.009478 | +0.008692 | +0.007836 |
| 4 | +0.002442 | +0.009555 | +0.013584 |
| 5 | +0.007709 | +0.000844 | -0.013150 |
| 6 | -0.007634 | -0.021883 | -0.029441 |
| 7 | -0.005919 | -0.015813 | -0.015535 |
| 8 | -0.010813 | -0.023167 | -0.036285 |
| 9 | +0.000050 | -0.022698 | -0.023155 |
| 10 | -0.005236 | -0.017002 | -0.011607 |

Full per-lead effect bounds, point metrics, 0.1/0.5/0.9 pinball, 80% coverage,
width and predictions remain in the saved Parquet artifacts. For illustration,
RMSD-only coverage/width were 0.808333 / 1.041223 Å; native+geometry were
0.777778 / 0.979125 Å. These values do not establish calibration acceptance.

The first complete command took 9.03 seconds, peak RSS 1,675,088 KiB and maximum
process-reserved VRAM 507,510,784 bytes. Hardware was RTX 4070 Laptop 8 GB,
driver 580.173.02, Python 3.14.8; the existing lockfile was preserved. These are
small-sample execution measurements, not full-corpus throughput estimates.

Ignored retained outputs are `data/processed/input-ablations-issue13` and
`data/processed/input-ablations-issue13-repeat`. Report identities:

| Variant | SHA-256 report ID |
| --- | --- |
| rmsd | `4a212ae5c94d06e98b196d44845c126259fddbc3596a8dd94c3debe080343774` |
| rmsd_com | `a60f18f12268cf30f2fdd7ada544671b2ce1ea21bb4a38448a397cf156784543` |
| native | `e8c3b2f466f29451e5e939614808d179b2ef51988333b1b298fff63d6e43926d` |
| native_geometry | `3f9804ba5b27b4e757efff996083aab73ef9d3b13ff9d9d5b8f79753e3d88f9b` |
