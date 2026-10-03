# Reproducible MVP report

Issue #17 delivers a CPU-only synthesis of **persisted** experiments. It never
loads model weights, downloads data, chooses a checkpoint, or reruns inference.
The scientific hypothesis remains unconfirmed. Engineering completion and
negative/inconclusive results are legitimate outputs of this frozen cohort.

## Regenerate from saved inputs

From the repository root with the original ignored artifacts available:

```sh
uv sync --locked
uv run --locked md-forecast report-mvp \
  --config configs/benchmarks/mvp-report.json \
  --output data/reports/mvp-v1 \
  --code-commit "$(git rev-parse HEAD)" \
  --lockfile uv.lock
```

Choose a fresh versioned output directory for every publication. Existing
files/directories and dangling output symlinks are rejected. Every source must
match the expected identity in the reviewed configuration. Missing inputs,
changed metric tables, unknown model labels, mismatched TRAIN/VAL/TEST roles,
missing pinned Chronos/checkpoint provenance and exceeded resource budgets fail
before publication. Inputs are read relative to the repository root. Dataset
bytes, checkpoints and report outputs remain ignored; this command requires
the saved local artifacts described by the experiment documentation below.
A fresh checkout alone cannot reconstruct experiments without those inputs.

The command publishes a complete directory atomically, containing:

- `report.md`: limits, dataset/evaluation roles, MAE headlines, saved effects,
  TRAIN-only autocorrelation diagnostics and training telemetry;
- per-cell `details.md`: complete whole-horizon point/quantile metrics,
  persistence skill/effects with uncertainty status, calibration and runtime;
- regenerated horizon-error and calibration SVGs, aggregate metric and
  comparison Parquet tables, and complete saved ablation/external effect tables;
- `headline-evidence.parquet`: exact original table hash and row selector for
  every displayed metric, including units;
- `experiment-manifest.json`: an integrity envelope embedding the input
  specification, complete benchmark manifests, typed provenance, generator
  code commit, lockfile digest, Python/package versions, runtime-table hashes,
  output hashes and metric evidence.

`read_mvp(Path("data/reports/mvp-v1"))` verifies the manifest and generated file
checksums. Repeating synthesis with the same inputs, commit, lock and environment
in another directory produces the same artifact identity and report contents.
Inference timings are taken from saved telemetry, never measured again. Missing
optional software versions and CPU memory telemetry remain explicitly unknown.
Original run manifests retain experiment hardware; package versions in the
outer manifest identify the **report generation** environment.

## Frozen experiment inventory

The reviewed `configs/benchmarks/mvp-report.json` pins nine benchmark reports,
20 cells, eight typed metadata inputs and three complete paired-effect tables.
No official MISATO TEST result is included.

| Evidence | Scope and interpretation | Documentation |
| --- | --- | --- |
| Context × horizon grid | 12 cells on 19 TRAIN systems; integration evidence | [Benchmark](BENCHMARK.md) |
| Native zero-shot/fine-tuning/NLinear | One development VAL cell; VAL also selected the checkpoint | [Fine-tuning](FINETUNING.md) |
| Four input ablations | Same development holdout; RMSD headlines and full other-channel appendices | [Input ablations](INPUT_ABLATIONS.md) |
| Common geometric MISATO | Development VAL selected statistical references before external scoring | [External validation](EXTERNAL_VALIDATION.md) |
| MDbind unseen complexes | Six exact PDB-disjoint complexes, 60 replicas; frozen MISATO preprocessing | [External validation](EXTERNAL_VALIDATION.md) |
| MDbind held-out replicas | 12 replicas of six seen complexes; separate local NLinear fit | [External validation](EXTERNAL_VALIDATION.md) |

Source manifests preserve dataset versions, checksums, terms, feature definitions
and units, split/preprocessing hashes, config/model identities, seeds, windowing,
bootstrap policy and experiment code/lock/hardware. Typed settings supply exact
Chronos revisions and fine-tuning checkpoint identities. The reporter verifies
that each Chronos artifact has linked settings rather than relying on a label.
Error and calibration figures are rebuilt from verified metric tables; edited
source SVGs cannot alter scientific results. Evidence selectors disambiguate
model, feature, lead, metric and quantile boundaries.

## Scientific limits

The native fine-tuning and ablation corrected confidence tails are unavailable
at the frozen resampling budget. Marginal intervals do not establish corrected
significance. Fine-tuning used eight updates and reused VAL for selection.
Native physical cadence, alignment, mass/SASA and energy semantics remain
unresolved, so native MISATO/MDbind transfer is blocked.

The common geometric external experiment does not confirm a lead >=2 that
beats both persistence and the MISATO-selected statistical reference under the
frozen corrected policy. Six complexes do not establish target/chemotype or
pretraining independence. Held-out replicas address a different task from
unseen complexes. Their supervised adaptation is disclosed separately.

The saved TRAIN QC supplies median lag-1 ACF, median first 1/e crossing in
**frames**, and censored counts. These finite-sample diagnostics are not
physical molecular timescales or independently estimated external ACFs.
Runtime tables cover adapter inference and host conversion; training telemetry
is separate. None of the outputs establish atomistic validity, free energies,
mechanistic causality or replacement of molecular dynamics simulation.

## Verified real-artifact synthesis

Two complete runs (`data/reports/mvp-v1-frozen` and `mvp-v1-repeat`) from the
persisted real-data experiments produced identical manifests and generated
contents. All Markdown links resolved and all 215 output checksums verified.
The manifest contains 2,138 metric evidence rows across nine reports/20 cells.
These counts describe report completeness, not independent sample size.

- Generator commit: `b5d52c6405b61d72296a45cbe987c10f0e89a5cc`.
- Frozen input specification: `sha256:4a3366a1cd654ef19470f1a8234424779593961fcd40ad263fe727fbdc70553e`.
- Final snapshot: `sha256:8084e9a846ecae45808e907014b7a494b9a337e9a7a6771007ba888621eaafc9`.

The subsequent documentation-only commit does not change the frozen generator.
Recreate this exact snapshot with `--code-commit` set to the generator commit
above and the original environment/lock. Other reporting environments receive
new manifest identities; scientific source report identities remain pinned.
Synthetic regression tests separately verify exact source-row linkage,
deterministic repetition, output immutability, corruption rejection, missing
model provenance, role/config errors, resource budgets and the CLI boundary.
