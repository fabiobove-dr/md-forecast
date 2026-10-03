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
