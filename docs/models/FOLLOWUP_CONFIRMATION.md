# Independent geometric confirmation

Issue [#45](https://github.com/fabiobove-dr/md-forecast/issues/45) tests the
choices developed in #37–44 on reserved outcomes. Engineering completion is
separate from scientific success. No choice is changed after confirmation.

## Frozen choices

The portable plan is
[`data/manifests/followup-confirmation45.json`](https://github.com/fabiobove-dr/md-forecast/blob/main/data/manifests/followup-confirmation45.json),
metadata hash
`sha256:f08d48b26b0994069ddcc1a5fb96a5b8977918f75f0b455b2b132767ad0694bc`.
It was committed as `4554843` before any reserved XTC decoding or native
confirmation coordinate extraction. The source code selecting these choices
was `8e1f578`; experiment runners record their own later committed code.
The original frozen plan remains unchanged when QC exclusions reduce counts.

Every source/checkpoint/configuration in `source_files` is SHA-256 verified
before reserved preparation or scoring. The plan records:

- three equivalent raw dry-system geometric observables: heavy atom-pair
  contacts within 4.5 Å, fraction of first-frame contacts retained, and minimum
  heavy receptor–ligand distance in Å;
- the complete C20/40 × H5/10 frame grid, stride 10, with **C40/H10 primary**;
- fixed representative seed 42, the validation-selected checkpoints and
  statistical/compact models, joint observed inputs for contacts and target-only
  inputs for fraction/distance;
- native TRAIN-only preprocessing, compact coefficients and residual quantiles;
  no new fitting, recalibration or checkpoint/input selection on confirmation;
- 10% worthwhile MAE reduction, nominal 80% intervals, coverage tolerance
  75–85%, quantile-loss and width comparators;
- 60,000 bootstrap draws, seed 42, 95% confidence, independent complex groups,
  Bonferroni family 126, minimum 20 groups and 10 expected corrected-tail draws.

The primary family comprises 36 point-MAE contrasts, 36 worthwhile-reduction
contrasts, 24 mean-three-quantile pinball contrasts, 12 width contrasts,
12 absolute coverage intervals and 6 fine-minus-zero MAE contrasts, across
native and untouched external tasks. The six-system replica task cannot meet
the independent-group requirement and cannot establish corrected superiority.
Other cells and individual leads remain descriptive. Resampling windows or
counting replicas as independent complexes is prohibited.

A useful minimum result requires **all** frozen conditions: corrected MAE and
10%-reduction bounds below zero against persistence and the selected statistic,
corrected quantile-loss gains against both probabilistic references, corrected
coverage wholly in 75–85%, and corrected width upper bound at most zero versus
the statistic. A stronger result additionally requires the 10% gain against the
compact supervised model. Missing intervals never pass a criterion.

## Separate tasks and QC

Native confirmation reserved 100 unseen MISATO complexes. Existing geometric
QC excludes `5SZC`: there are no frame-0 reference contacts, so the admitted
fraction is undefined. The frozen selection is retained; **99** groups and
9,900 frames are scored. No substitute ID is selected. The 132,841,014,019-byte
source was reverified against its official MD5 before extraction; the complete
preparation took 8:14.76 and its peak RSS is recorded in the cost table below.

The original external reserve of 50 complexes was prospectively extended to
80 using the next 30 eligible metadata-ranked IDs, preserving the six
seen-system IDs and original 50. The extension occurred before requesting any
of these confirmation trajectories; the reserve hash is
`sha256:28a5b784ba62ec0ed7786fe4543cffebb0670245c59a8d1f8278c930aece8e9a`.
Selection is independent of QC, geometry and model errors. Frozen static
selection rules may reject a complex; failures remain recorded without ID
replacement. Any replica failure excludes its whole complex from the matched
three-observable comparison. Acquisition and preparation retain static and
geometric rejection reasons separately.

The seen-system task reads only replica 10 of each of six previously examined
MDbind systems. It is **seen by development analysis**, while the models,
compact scaler and residual distributions remain fitted on MISATO TRAIN.
There is no undisclosed fitting on the six external systems.

Both MDbind tasks use actual 200 ps spacing: the 20/40 observed samples span
3.8/7.8 ns from first to last context sample; H5/H10 extend 1/2 ns beyond
the last observed value. MISATO's physical sample cadence is unresolved; its axes
remain frames. Exact PDB disjointness and protein-sequence overlap checks do
not establish unseen-target/chemotype or pretraining independence.

Explicit `evaluation_spec` bindings validate ordered feature definitions,
units and feature-set version while preserving the fitted state/checkpoint and
its original provenance hash. Target dataset/version/split identities remain
true to the target task. Compact and residual bindings additionally require
the original C/H; Chronos permits the predeclared grid. Default fitted adapters
still reject foreign domains. No batch identity or fitted scaler is relabelled.

## Regeneration

Acquire MISATO and reviewed frozen development artifacts using the earlier
[cohort](../datasets/FOLLOWUP_COHORT.md),
[baseline](FOLLOWUP_BASELINES.md), [fine-tuning](FOLLOWUP_FINETUNING.md),
[input](FOLLOWUP_INPUTS.md), and [probability](FOLLOWUP_PROBABILITY.md) workflows.
Install the existing locked optional model/structural dependencies. All
commands below run from the repository root; outputs must be fresh directories.

```sh
uv run --locked python examples/freeze_followup_confirmation.py
# The committed plan already exists; this command refuses to overwrite it.
uv run --locked python examples/prepare_followup_cohort.py prepare \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --output data/processed/followup-native-confirmation45
uv run --locked python examples/acquire_followup_confirmation.py
uv run --locked python examples/prepare_mdbind_common.py \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --reserve configs/datasets/followup-external-reserve80.json \
  --source configs/datasets/followup-external80-source.json \
  --source-qc configs/datasets/followup-external80-source.qc.json \
  --raw data/external/followup-external80-raw \
  --output data/processed/followup-external-confirmation45
uv run --locked python examples/prepare_mdbind_common.py \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --reserve configs/datasets/followup-external-reserve.json \
  --source configs/datasets/followup-seen-source.json \
  --raw data/external/followup-seen-raw \
  --output data/processed/followup-seen-confirmation45
uv run --locked python examples/run_followup_confirmation.py --task native \
  --input data/processed/followup-native-confirmation45 \
  --output data/reports/followup-confirmation45-native
uv run --locked python examples/run_followup_confirmation.py --task external \
  --input data/processed/followup-external-confirmation45 \
  --output data/reports/followup-confirmation45-external
uv run --locked python examples/run_followup_confirmation.py --task replica \
  --input data/processed/followup-seen-confirmation45 \
  --output data/reports/followup-confirmation45-replica
uv run --locked python examples/summarize_followup_confirmation.py \
  --output data/reports/followup-confirmation45-inference.json
uv run --locked md-forecast report-forecast \
  --config configs/experiments/performance-overview.json \
  --output data/reports/offline-overview45/index.html
```

Saved bundles include every matched forecast window, lead, truth, point error,
median spread, source group and quantile. Compact point forecasts have no
invented probabilistic bands. Each summary pins every file, the full frozen
plan and target split, model/checkpoint identities, genuine evaluation spec,
code commit, lockfile hash, QC deviations and per-batch runtime/VRAM. The
separate inference JSON verifies all bundle hashes and leaves bundles intact.

The report embeds the actual curves, residuals, baseline ratios, all cells,
corrected decisions, native-unit comparison plots and the observed molecular
reference. Its per-window tolerance ratings remain user-controlled
retrospective error checks, independent of the frozen scientific decision.
An unset tolerance remains UNRATED; scalar predictions do not imply valid
atomistic trajectories.

## Measured results

Results and costs are recorded below after all three task bundles complete.
