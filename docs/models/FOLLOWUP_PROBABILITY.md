# Follow-up probabilistic references and prospective inference

Issue [#43](https://github.com/fabiobove-dr/md-forecast/issues/43) adds an applicable
reference for Chronos intervals and freezes numerically resolvable confirmation
inference. Coverage alone, flat medians and successful training do not establish
useful predictions.

## Residual reference

`fit_residual_quantiles` fits empirical signed future-minus-point residuals only
on native TRAIN windows. The API has no held-out partition option. Residuals are
pooled over training windows separately for every future lead and observable;
on this native cohort each system supplies the same number of windows. Window
pooling defines a training estimator, not independent uncertainty samples.

Reuse persistence and each observable's strongest statistical configuration
selected on VAL in #40. Their coefficients remain context-local. Store all
lead-specific offsets, source/grid/feature/config hashes, complete TRAIN group
and trajectory IDs, input-value hash and window counts in `ResidualState`.
Fit budgets are 5,000 windows and 128 MiB estimated residual-array memory;
exceeding either fails before loading a series. Each cell actually has
1,600 (C20) or 1,200 (C40) TRAIN windows from 200 independent systems.

The levels are 0.1/0.5/0.9, with linear empirical quantiles. Bound lower offsets
by zero from above and upper offsets by zero from below; the median offset is
exactly zero. Thus intervals can be asymmetric while the original point stays
unchanged. This protocol is not split conformal and does not promise nominal
coverage. No support clipping changes points: residual and Chronos intervals
remain explicitly unconstrained, including possible endpoints outside observed
physical support. Monotonic/finiteness/state/source/grid guards remain explicit.

No VAL calibration is fitted. The 40 reserved calibration systems remain
unopened. VAL selects point-model configurations/checkpoints/inputs, so these
probability comparisons are development evidence; independent calibration
assessment occurs only on confirmation. Do not relabel selection VAL as an
independent calibrated test or replace historical unresolved intervals.

## Real comparison protocol

Use identical native VAL windows in the four C20/40 × H5/10 frame cells. Reuse
#42's frozen input choice for both zero-shot and fine-tuned Chronos: joint
contacts, target-only fraction/distance, with representative seed 42. All
checkpoint identities are checked. Reference statistics can use their full
admitted context. Export every target/origin, point residual, quantile, native
pinball loss, 80% coverage and width, with exact label/origin pairing.

Average points/windows within each trajectory, trajectories within systems,
and then systems equally. Report each lead and the complete horizon separately.
Development confidence intervals use 2,000 system bootstrap resamples, seed 42,
marginal 95% confidence and no corrected superiority claim. Application error
thresholds remain undeclared: an observation inside an interval is not a
point forecast within an application tolerance.

## Frozen confirmation policy

`configs/experiments/followup-probability.json` declares C40/H10 as the primary
cell, all three admitted observables, both foundation versions, the same
preprocessing/input/checkpoint choices, 10% worthwhile point-error reduction,
and 80% nominal coverage with ±5 percentage-point tolerance. Other cells and
individual leads remain descriptive; no favorable cell is chosen after testing.
These are scientific comparison criteria, not application error tolerances.

The complete primary family across native and untouched external tasks is 126:

| Claim/interval type | Count |
| --- | ---: |
| Foundation-minus-persistence/statistic/compact MAE | 36 |
| Foundation-minus-0.9 × each reference MAE | 36 |
| Mean native pinball difference against persistence/statistic | 24 |
| Interval-width difference against selected statistic | 12 |
| Foundation empirical coverage interval | 12 |
| Fine-tuned-minus-zero-shot MAE | 6 |

Pinball means equally over 0.1/0.5/0.9 levels. Coverage is horizon-averaged;
width comparisons remain in the observable's units. Useful minimum skill
requires corrected MAE and worthwhile-reduction bounds below zero against both
persistence and selected statistics, coverage bounds entirely within 0.75–0.85,
corrected pinball gains against both probability references and intervals no
wider than the selected statistical reference. The stronger criterion additionally
requires the worthwhile compact-model improvement. Publish weaker measured
point gains separately from a useful-skill decision, including inconclusive cases.

Use 60,000 system-resampling draws, seed 42, 95% Bonferroni intervals, minimum
20 independent systems, at least ten expected draws in each corrected tail and
128 MiB estimated bootstrap memory. The corrected tail has approximately
11.90 draws: unlike the historical native 616-comparison / 1,000-resample pilot,
it is numerically resolvable. Numerical resolution is not statistical power.
100 native groups need 96,000,000 estimated bytes; 80 external groups need
76,800,000. Six seen systems receive `insufficient-groups`; additional replicas
or windows do not turn them into more independent complexes.

Re-run the historical six-group planning calculation using the final family,
10% target and threefold paired-SD inflation. Approximate required groups are
16 native and 80 external, versus the original four-comparison plan's smaller
requirements. Target 100 native and **80 external** systems. The original 50
external identities remain untouched; #45 must freeze 30 additional eligible
metadata-only identities before opening any confirmation outcome. This is a
prospective count, not a claim that all 80 are already prepared. Pilot variance
and normal approximations are uncertain; these figures do not guarantee power
for every observable or the compound usefulness criterion.

## Reproduction

Complete [baseline selection](FOLLOWUP_BASELINES.md),
[geometric adaptation](FOLLOWUP_FINETUNING.md) and
[observed-input comparison](FOLLOWUP_INPUTS.md) first. Frozen parent-file hashes
reject changed selections. Commit code/config and choose a fresh output:

```sh
uv run --locked python examples/run_followup_probability.py \
  --output data/reports/followup-probability-validation-fresh
uv run --locked python examples/plan_followup_precision.py \
  --config configs/experiments/followup-confirmation-precision.json \
  --output data/reports/followup-confirmation-precision
```

The precision command reads the original reserve's 50 identities; #45's expanded
80-system manifest must be passed with `--external-reserve` for the final count.
Local planning evidence also records the prospective 80-group calculation.
Source data/checkpoints/residual states/generated reports remain ignored.
Native physical cadence remains unknown; MDbind's verified 200-ps cadence makes
its physical task distinct. Full confirmation outcomes and all failures follow
in #45, without retuning after observing them.


## Complete real development results

The real run publishes 96 prediction/error Parquets, 16 portable fitted residual
states and its frozen configuration: 113 integrity-checked files. All models
score 1,680 windows across the four cells, from the same 60 native VAL systems.
Runtime is 137.99 seconds with peak process RSS 2,176,528 KiB. Every file checksum
is verified; saved median/label agreement is separately audited. Fitted references
retain their original statistical points. The frozen implementation/config commit
is `902c1d7`; synthetic tests are distinct from these real outcomes.

Coverage below is the fraction inside the nominal 80% interval; width, MAE and
mean pinball use each observable's native units. Full lead-specific losses and
system-level marginal intervals are saved, including unfavorable cells.

| Cell | Observable | Method | MAE | Coverage | Width | Mean pinball |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| c20-h5 | protein_ligand_contact_count | persistence | 13.201667 | 0.8479 | 47.600000 | 4.344583 |
| c20-h5 | protein_ligand_contact_count | selected-statistic | 11.178671 | 0.8254 | 38.034962 | 3.699761 |
| c20-h5 | protein_ligand_contact_count | zero-shot | 11.046122 | 0.8329 | 39.466011 | 3.554548 |
| c20-h5 | protein_ligand_contact_count | fine-tuned | 10.882934 | 0.7300 | 30.074570 | 3.491858 |
| c20-h5 | fraction_reference_contacts | persistence | 0.063784 | 0.8200 | 0.212212 | 0.020891 |
| c20-h5 | fraction_reference_contacts | selected-statistic | 0.056069 | 0.8054 | 0.180113 | 0.018260 |
| c20-h5 | fraction_reference_contacts | zero-shot | 0.053806 | 0.8517 | 0.207335 | 0.017877 |
| c20-h5 | fraction_reference_contacts | fine-tuned | 0.052966 | 0.7304 | 0.145095 | 0.017161 |
| c20-h5 | minimum_protein_ligand_heavy_distance | persistence | 0.099164 | 0.8483 | 0.347739 | 0.032418 |
| c20-h5 | minimum_protein_ligand_heavy_distance | selected-statistic | 0.071158 | 0.8708 | 0.265680 | 0.023609 |
| c20-h5 | minimum_protein_ligand_heavy_distance | zero-shot | 0.074544 | 0.8687 | 0.297450 | 0.024198 |
| c20-h5 | minimum_protein_ligand_heavy_distance | fine-tuned | 0.071208 | 0.8287 | 0.250590 | 0.022751 |
| c20-h10 | protein_ligand_contact_count | persistence | 13.536458 | 0.8485 | 49.120000 | 4.457257 |
| c20-h10 | protein_ligand_contact_count | selected-statistic | 11.496851 | 0.8175 | 38.729947 | 3.811526 |
| c20-h10 | protein_ligand_contact_count | zero-shot | 11.317995 | 0.8490 | 43.083495 | 3.680636 |
| c20-h10 | protein_ligand_contact_count | fine-tuned | 11.199989 | 0.7354 | 31.683444 | 3.597629 |
| c20-h10 | fraction_reference_contacts | persistence | 0.066257 | 0.8235 | 0.222415 | 0.021851 |
| c20-h10 | fraction_reference_contacts | selected-statistic | 0.059253 | 0.8098 | 0.187119 | 0.019381 |
| c20-h10 | fraction_reference_contacts | zero-shot | 0.057141 | 0.8629 | 0.232029 | 0.019258 |
| c20-h10 | fraction_reference_contacts | fine-tuned | 0.055825 | 0.7356 | 0.154877 | 0.018160 |
| c20-h10 | minimum_protein_ligand_heavy_distance | persistence | 0.099476 | 0.8523 | 0.355071 | 0.032606 |
| c20-h10 | minimum_protein_ligand_heavy_distance | selected-statistic | 0.073544 | 0.8640 | 0.271459 | 0.024462 |
| c20-h10 | minimum_protein_ligand_heavy_distance | zero-shot | 0.075764 | 0.8850 | 0.320684 | 0.025089 |
| c20-h10 | minimum_protein_ligand_heavy_distance | fine-tuned | 0.073612 | 0.8225 | 0.255275 | 0.023602 |
| c40-h5 | protein_ligand_contact_count | persistence | 13.127778 | 0.8606 | 48.800000 | 4.317407 |
| c40-h5 | protein_ligand_contact_count | selected-statistic | 11.348039 | 0.8217 | 38.329111 | 3.748721 |
| c40-h5 | protein_ligand_contact_count | zero-shot | 11.161488 | 0.7911 | 34.613238 | 3.515474 |
| c40-h5 | protein_ligand_contact_count | fine-tuned | 10.984325 | 0.7383 | 30.896059 | 3.491523 |
| c40-h5 | fraction_reference_contacts | persistence | 0.061524 | 0.8250 | 0.209112 | 0.020147 |
| c40-h5 | fraction_reference_contacts | selected-statistic | 0.055384 | 0.8072 | 0.174556 | 0.018146 |
| c40-h5 | fraction_reference_contacts | zero-shot | 0.052052 | 0.8094 | 0.176830 | 0.016915 |
| c40-h5 | fraction_reference_contacts | fine-tuned | 0.051512 | 0.7367 | 0.141469 | 0.016601 |
| c40-h5 | minimum_protein_ligand_heavy_distance | persistence | 0.100759 | 0.8422 | 0.345992 | 0.033003 |
| c40-h5 | minimum_protein_ligand_heavy_distance | selected-statistic | 0.072743 | 0.8678 | 0.269132 | 0.024069 |
| c40-h5 | minimum_protein_ligand_heavy_distance | zero-shot | 0.074494 | 0.8206 | 0.256323 | 0.023672 |
| c40-h5 | minimum_protein_ligand_heavy_distance | fine-tuned | 0.072064 | 0.8100 | 0.241400 | 0.022921 |
| c40-h10 | protein_ligand_contact_count | persistence | 13.250278 | 0.8642 | 49.730000 | 4.355750 |
| c40-h10 | protein_ligand_contact_count | selected-statistic | 11.503589 | 0.8233 | 38.992987 | 3.809788 |
| c40-h10 | protein_ligand_contact_count | zero-shot | 11.231818 | 0.7978 | 36.316623 | 3.569166 |
| c40-h10 | protein_ligand_contact_count | fine-tuned | 11.122296 | 0.7464 | 31.964135 | 3.547953 |
| c40-h10 | fraction_reference_contacts | persistence | 0.063193 | 0.8319 | 0.219430 | 0.020844 |
| c40-h10 | fraction_reference_contacts | selected-statistic | 0.057439 | 0.8186 | 0.182771 | 0.018749 |
| c40-h10 | fraction_reference_contacts | zero-shot | 0.054712 | 0.8217 | 0.192028 | 0.017840 |
| c40-h10 | fraction_reference_contacts | fine-tuned | 0.053509 | 0.7389 | 0.147977 | 0.017212 |
| c40-h10 | minimum_protein_ligand_heavy_distance | persistence | 0.100719 | 0.8458 | 0.351882 | 0.032953 |
| c40-h10 | minimum_protein_ligand_heavy_distance | selected-statistic | 0.074925 | 0.8592 | 0.273250 | 0.024869 |
| c40-h10 | minimum_protein_ligand_heavy_distance | zero-shot | 0.076149 | 0.8236 | 0.265195 | 0.024420 |
| c40-h10 | minimum_protein_ligand_heavy_distance | fine-tuned | 0.074288 | 0.7981 | 0.242720 | 0.023730 |

For primary C40/H10, zero-shot coverage is about 79.8% contacts, 82.2% reference
fraction and 82.4% distance. Fine-tuned coverage is 74.6%, 73.9% and 79.8%,
respectively. Development marginal coverage intervals for the first two
fine-tuned targets are about [72.8%, 76.5%] and [71.8%, 75.8%]: tighter intervals
have sacrificed nominal coverage despite lower point MAE. Fine-tuning therefore
does not uniformly improve uncertainty. Its distance interval is narrower than
the selected statistical reference, with better development pinball loss; these
selection-set results still need independent evaluation.

C40/H10 persistence coverage is 83–86%, with wider intervals. The selected
statistical references have about 82–86% coverage. Neither coverage alone nor
a width reduction establishes useful uncertainty: point skill, calibrated
coverage, width and pinball must be assessed together under the frozen policy.
Raw intervals are retained; no post-scoring adjustment makes them look calibrated.

Local evidence is `data/reports/followup-probability43-validation` and
`data/reports/followup-probability43-precision.json`. Historical MVP reports and
the original four-comparison precision plan remain unchanged. A standalone HTML
presentation follows in #44, with independent decisions in #45.
