# Common geometric external validation

This issue #14 experiment evaluates three explicitly equivalent dry-system
geometric quantities, after the [MDbind audit](../datasets/MDBIND.md). It is a new
zero-shot experiment; the native RMSD fine-tuned checkpoint is not transferred.
The full specification is `configs/benchmarks/mdbind-common.toml` and the
feature configuration is `configs/features/common-geometry.yaml`.

## Frozen protocol

- Features: contact pair count, fraction of first-frame contacts, minimum heavy
  distance. See the audit for matching selections, reference, units and exclusions.
- MISATO development: 19 local official TRAIN systems, grouped 70/30/0 with seed
  42, giving 13 fitting and six validation complexes. Official TEST is not loaded.
- Models: revision-pinned Chronos-2 zero-shot, persistence, context mean, linear
  extrapolation, AR(1) and VAR(1); no MDbind fine-tuning. Statistical settings match
  the existing baseline specification. Per feature and lead (including the full
  horizon average), choose lowest validation MAE; ties use configuration hash.
- Grid: C40/H10, stride 10 frames. MISATO physical cadence is unresolved; no equal
  physical horizon across datasets is claimed. MDbind has 200-ps samples: context
  first-to-last span **7.8 ns**, and last-context-to-final-target horizon **2 ns**.
- Untouched external task: all ten replicas of each of six nonoverlapping PDB
  complexes, one complete window per replica. Scaled metrics use explicitly
  linked, frozen MISATO TRAIN-context moments; physical MAE/RMSE remain primary.
  Transfer rejects different feature definitions/units and overlapping PDB IDs.
- Seen-complex replica task: same true system IDs; replicas 1–8 TRAIN, 9–10 TEST.
  After the first external report has been published and verified, fit a fixed
  NLinear baseline with ridge 0 on these TRAIN replicas. Chronos remains frozen;
  this task includes local baseline adaptation and is reported separately.
- Averaging: windows within replica, replicas within complex, equal complex
  weighting. Bootstrap whole complexes; 300,000 draws, seed 42, 95% confidence,
  at least five groups, Bonferroni family 600 (594 planned point comparisons),
  at least ten samples per corrected tail. Six complexes limit inference.
- Quantiles 0.1/0.5/0.9, nominal 80% coverage, frozen acceptable deviation 0.1.
  Report coverage, width and pinball loss at every lead. Deterministic baselines
  do not supply a probabilistic comparator; calibration alone cannot establish
  useful uncertainty or the stronger scientific hypothesis.

The acquisition cohort and all model/grid/uncertainty settings precede external
scoring. Geometry QC reads source payloads but performs no model selection.
Checkpoint/model pretraining overlap and unseen-target/chemotype grouping have
not been audited; exact PDB exclusion supports only the named complex task.

## Reproduce

```sh
uv sync --locked --extra structural --extra chronos
uv run --locked python examples/prepare_mdbind_common.py --download --output data/processed/mdbind-common-verified
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run --locked python examples/run_mdbind_common.py --output data/processed/mdbind-common-benchmark
```

The MISATO structural sample and pinned Chronos snapshot must already exist,
as described in [structural extraction](../datasets/STRUCTURAL_FEATURES.md) and
[Chronos integration](CHRONOS2.md). Use fresh output directories. Report bundles,
source bytes and the fitted local baseline stay ignored under `data/`.
Implementation/configuration must be committed before the run records provenance.

## Acceptance evidence

Raw acquisition, per-replica topology checks and timestamp verification are
complete. Real forecasts and their repeatability checks are pending; this protocol
alone does not close #14 or establish measured skill. Scientific outcomes and
artifact identities will be recorded here after the frozen experiment.
