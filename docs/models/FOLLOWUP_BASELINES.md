# Follow-up regularized baselines

Issue [#40](https://github.com/fabiobove-dr/md-forecast/issues/40) replaces the
old ridge-zero compact pilot with a validation-selected comparator on the
[fresh cohort](../datasets/FOLLOWUP_COHORT.md). A better comparator does not
require pretraining or fine-tuning to win; all results remain development
selection evidence, with confirmation/calibration data inaccessible.

## Fitting and selection

Reuse `fit_nlinear`, `select_nlinear`, `ridge_fit`, `evaluate_baselines` and the
existing state/scaler contracts. `normalized_group_mae` is now the shared
unit-aware score for learned and statistical candidates: average native MAE
inside each trajectory/system, divide feature errors by TRAIN-context scales,
then average features and systems equally. A system with more replicas or
windows does not receive more selection weight. Scales are positive and finite.

Freeze the four 20/40 × 5/10 frame cells and stride 10 from #39. Statistics
include persistence, context mean, linear extrapolation, AR/VAR lags 1/3 with
ridge 0/0.1/1/10/100 (23 settings). Their coefficients are context-local, using
observed history only; ridge acts on native-unit designs. NLinear uses separate
channel mappings fitted on every TRAIN window, TRAIN-context scalers, and ridge
0/0.1/1/10/100/1,000/10,000 (seven settings). Only VAL chooses candidates.
No architecture or dependency is added.

The initial five-value compact grid selected its upper boundary, ridge 100,
in every cell. Retain that run, then extend the development grid to 1,000 and
10,000 before repeating selection. This is a documented development decision,
not untouched test inference. The final choices are interior: 100 in three
cells and 1,000 in C40/H10. All original and extended candidate scores remain
in local selection artifacts; unfavorable settings are retained.

Training uses 200 native systems; validation uses 60. Maximum 5,000 training
windows and 128 MiB estimated design working memory are explicit. Exceeding a
budget fails; no silent subsampling occurs. The actual fit uses 1,600 windows
for C20 and 1,200 for C40. Window overlap does not create independent systems.
The selected regularized AR/VAR and compact models are evaluated on identical
features/windows, alongside persistence, mean, linear and unregularized NLinear.

## Fitted dimensions and state identities

NLinear has C temporal columns plus one intercept per channel, with H outputs.
Last-level centering makes the final temporal column exactly zero; rank C
rather than C+1 is expected. Unregularized condition numbers are undefined
for this deficient design, not evidence of corrupted data. The existing
least-squares solver handles it; ridge adds temporal penalties while preserving
the unpenalized intercept. Record actual singular-value rank and the selected
augmented-design condition number for every channel.

| Cell | TRAIN windows | Ridge | Parameters | Design rows × columns per channel | Selected condition-number range |
| --- | ---: | ---: | ---: | --- | --- |
| c20-h5 | 1600 | 100 | 315 | 1600 × 21 | 5.114–10.863 |
| c20-h10 | 1600 | 100 | 630 | 1600 × 21 | 5.114–10.863 |
| c40-h5 | 1200 | 100 | 615 | 1200 × 41 | 6.634–13.524 |
| c40-h10 | 1200 | 1000 | 1230 | 1200 × 41 | 2.302–4.381 |

Selected fitted-state hashes (include coefficients, complete configuration and TRAIN scaler provenance):

- c20-h5: `sha256:b313e9ba9df666840920777028da1fb5a884121a17f04be7eb8dffe21defe312`
- c20-h10: `sha256:4750d376b4960c63b0b21390352d9de5beeb7235ebc3ba7f7126757a9287b06d`
- c40-h5: `sha256:639d4f98c845551707cf2c5319bc24bdf8a1f6dd49b9a5eaceb7b47e192d8a51`
- c40-h10: `sha256:1387823da8060f758a6bfd95b62ed01874745a0923a907f5285ecf2be72d9820`

AR designs have C−lags rows and lags+1 columns; VAR designs have C−lags rows
and 3×lags+1 columns. Selected dimensions/configurations, all native-unit group
validation scores, normalized selection scores and per-cell elapsed selection
costs are retained in the bundle. Per-feature best statistical references are
also recorded on VAL across the full declared statistical grid; those differ
from choosing one AR/VAR by the joint normalized feature score.

## Matched validation outcomes

Each entry below is equal-system native-unit MAE. Compact refers to the selected
ridge setting, and zero refers to the preserved unregularized fit. The full
23-statistical and seven-compact candidate outcomes stay available, including
per-group errors. Do not interpret individual table differences as independent
superiority: the same VAL chose configurations.

| Cell | Observable | Persistence | Mean | Selected AR | Selected VAR | Compact | Zero |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c20-h5 | Contacts (pairs) | 13.20167 | 11.50840 | 11.20490 | 11.17867 | 10.95558 | 10.96328 |
| c20-h5 | Reference fraction (0–1) | 0.06378 | 0.06080 | 0.05607 | 0.05736 | 0.05425 | 0.05419 |
| c20-h5 | Minimum distance (Å) | 0.09916 | 0.07130 | 0.07140 | 0.07205 | 0.07313 | 0.07347 |
| c20-h10 | Contacts (pairs) | 13.53646 | 11.73267 | 11.52625 | 11.49685 | 11.25632 | 11.24634 |
| c20-h10 | Reference fraction (0–1) | 0.06626 | 0.06345 | 0.05956 | 0.06036 | 0.05685 | 0.05683 |
| c20-h10 | Minimum distance (Å) | 0.09948 | 0.07354 | 0.07367 | 0.07402 | 0.07502 | 0.07544 |
| c40-h5 | Contacts (pairs) | 13.12778 | 11.95635 | 11.36162 | 11.39870 | 10.96692 | 11.10570 |
| c40-h5 | Reference fraction (0–1) | 0.06152 | 0.06743 | 0.05590 | 0.05736 | 0.05292 | 0.05342 |
| c40-h5 | Minimum distance (Å) | 0.10076 | 0.07283 | 0.07406 | 0.07395 | 0.07527 | 0.07635 |
| c40-h10 | Contacts (pairs) | 13.25028 | 12.05228 | 11.50995 | 11.66106 | 11.28471 | 11.26574 |
| c40-h10 | Reference fraction (0–1) | 0.06319 | 0.06864 | 0.05744 | 0.05937 | 0.05576 | 0.05536 |
| c40-h10 | Minimum distance (Å) | 0.10072 | 0.07496 | 0.07592 | 0.07612 | 0.07569 | 0.07921 |

Compact NLinear is competitive with or better than zero-shot Chronos on contact
counts in some cells of the matched #39 development grid. C40/H10 selected
compact contact MAE is 11.28471 pairs versus zero-shot Chronos 11.23182; the
unregularized compact value is 11.26574. Regularization was chosen by the joint
normalized score, not this one favorable feature. Minimum-distance MAE improves
from 0.07921 Å unregularized to 0.07569 Å selected, while context mean remains
better (0.07496 Å). Pretraining superiority is therefore not established by
these descriptive comparisons, and more training is not assumed to guarantee
improvement. Compare meaningful matched fine-tuning in #41 and frozen
independent outcomes in #45.

## Repeatability and reproduction

```sh
uv run --locked python examples/select_followup_baselines.py \
  --output data/processed/followup-baselines
uv run --locked python examples/select_followup_baselines.py \
  --output data/processed/followup-baselines-repeat
```

Commit implementation/config before fitting; use fresh output directories.
The native purpose-scoped loader rejects calibration/confirmation reads before
opening a table. Each cell publishes selected and ridge-zero states, compact
selection, all statistical validation outcomes and matched validation outcomes
as integrity-checked metadata envelopes. The provenance records source/cohort,
split/config/lockfile/commit identities, complete state hashes, selection costs,
training counts, design dimensions/conditioning and all artifact checksums.
Publication is atomic. No dataset bytes, states or generated reports are committed.

Every selected state is reloaded and checked for exact artifact identity and
identical first-batch predictions. Independent real repeated execution verifies
all **21 artifact checksums** and all selected state/config/dimension/conditioning
results are identical; only timings differ. Original five-candidate selection:
40.41 s / 178,012 KiB peak RSS. Expanded selection: 46.46 s / 175,756 KiB;
repeat: 49.40 s / 175,520 KiB. These are local CPU measurements, not GPU costs.
Existing leakage/state/budget tests are retained; numerical tests additionally
verify equal-group native-unit selection, invalid scales and regularized
rank-deficient designs with an unpenalized constant intercept.

A compact linear model suffices for this comparator. No TCN/nonlinear
integration is justified here. Finite development selection, native-unit
statistical penalties, limited features and unknown MISATO physical cadence
remain limitations. Completion establishes a fair recorded comparison, not a
positive scientific result.
