# Scientific contract

This document defines the claims and acceptance criteria for the MVP. The
[implementation plan](IMPLEMENTATION_PLAN.md) describes execution; the
[engineering standards](ENGINEERING_STANDARDS.md) govern implementation.
No dataset fields, units, timestamps, or licenses are assumed verified until
the dataset audit admits them.

## Hypotheses and scope

**Primary hypothesis:** on MISATO alone, a pretrained time-series foundation
model forecasts at least one verified, physically meaningful MD observable
from its past better than persistence and statistical baselines over a
predeclared context/horizon grid, with calibrated predictive uncertainty.

**Null hypothesis:** there is no reproducible improvement over the strongest
validation-selected persistence/statistical baseline on that grid. Failure to
reject this null is a valid result, not proof that MD observables are inherently
unpredictable.

The stronger hypothesis asks whether gains on unseen complexes also exceed a
compact model trained from scratch on the same training data. External
validation and predictive coupling are separate extensions, not prerequisites
for testing the primary hypothesis with MISATO.

## Definitions

| Term | Meaning |
| --- | --- |
| System/complex | A protein–ligand system; all its dependent trajectories and windows share a split group for unseen-complex evaluation. |
| Trajectory | One ordered sequence from one simulation replica, with source identity, timestamps, physical units, and provenance. Forecasted features do not constitute a new atomistic trajectory. |
| Observable | A versioned scalar or vector feature derived from MD, with a verified physical definition, unit, atom selection where relevant, and frame alignment. Dataset-level summaries are not time series. |
| Context | The observed samples available at the forecast origin. For origin `s + C`, inputs are `X[s:s+C]`; future-dependent inputs are prohibited. |
| Horizon | The next `H` samples, `X[s+C:s+C+H]`. Record frame count and physical duration from verified timestamps; reject windows that do not fit. |
| Split | A fixed train/validation/test assignment of trajectories or system groups made before window generation. Validation selects settings; test data evaluate the frozen choice. |
| External validation | Evaluation on an untouched dataset produced by a different pipeline, using equivalent feature definitions and explicit overlap checks. Tuning on that dataset changes the claim to external adaptation. |

Held-out windows from a seen trajectory measure temporal continuation only.
Held-out trajectories/replicas of a seen complex measure replica generalization.
Held-out complexes measure system generalization; they do not establish
unseen-target or unseen-chemotype generalization without the corresponding
grouped splits. Report these evaluations separately.

## Evaluation protocol

Before examining test results, freeze the admitted features, units,
context/horizon grid, split manifest, seeds, model choices, baseline selection,
metrics, calibration tolerances, and uncertainty procedure in the experiment
configuration. Do not choose thresholds or favorable horizons after testing.
Physical context/horizon values must await verified dataset timestamps.

- Split before windowing. Keep overlapping windows and dependent replicas in
  the correct group. Fit preprocessing on training data only, or use explicitly
  context-local transformations; never inspect a window's future.
- Evaluate every model on the same windows and physical units. Include
  persistence, context mean, linear extrapolation, autoregression, and VAR
  where applicable. Select the strongest applicable statistical baseline on
  validation data, then freeze it. Stronger claims additionally require a
  compact supervised baseline trained on the same data and splits.
- Use trajectory-aggregated MAE as the primary point metric for each observable
  and horizon; also report RMSE and error ratios against persistence and the
  selected statistical baseline. Aggregate trajectories within a complex for
  unseen-complex inference. Report absolute errors when a baseline error is
  zero rather than inventing a ratio.
- Report paired 95% confidence intervals for model-minus-baseline error by
  resampling independent evaluation groups (complexes for unseen-complex
  tests). Overlapping windows are never independent statistical samples.
  Declare the resampling seed/count and any multiple-comparison correction
  before testing. If too few independent groups exist, report the limitation
  and do not claim reproducible superiority.
- Report error versus horizon, autocorrelation/correlation-time estimates,
  and skill beyond trivial one-step or slowly varying continuation. A
  non-trivial horizon spans at least two future samples; state its physical
  duration and relation to intrinsic correlation time. Gains confined to slow
  dynamics support only that narrower claim.
- For probabilistic forecasts, report pinball loss, nominal versus empirical
  interval coverage, and interval width by horizon, with group-level
  uncertainty. Freeze nominal levels and acceptable coverage deviations on
  validation data. Coverage alone with uninformatively wide intervals does
  not establish useful uncertainty; compare width and pinball loss to an
  applicable probabilistic baseline.

## Acceptance criteria

| Level | Required evidence |
| --- | --- |
| Minimum | On held-out MISATO trajectories, at least one predeclared observable/grid cell at a non-trivial horizon beats both persistence and the validation-selected statistical baseline: paired corrected confidence intervals for MAE differences lie below zero. Its predictive intervals meet the frozen calibration tolerance, with widths and pinball loss reported. State whether complexes were seen. |
| Stronger | Minimum criteria hold on completely unseen complexes and the model also beats the compact supervised baseline under the same paired evaluation. Report the entire grid, not only winning cells. |
| External | A model frozen after MISATO development retains measurable skill against persistence and statistical baselines on an untouched, provenance-checked MDbind subset with equivalent observables. Report unseen-replica and unseen-complex results separately; disclose cross-dataset overlap. |
| Predictive coupling extension | Adding region A's past improves prediction of region B's future across independent trajectories/replicas against B-only and simpler lagged baselines. Declare regions, features, controls, and splits in advance. This supports predictive coupling only. |

Successful infrastructure, plausible curves, training loss reduction, or a
single favorable window do not satisfy scientific acceptance. A completed,
reproducible benchmark that fails these criteria is a valid negative MVP
outcome; engineering completion and scientific success are different statuses.

## Claims and limits

Permitted claims describe forecast skill for the tested observable definitions,
datasets, splits, and physical horizons, including uncertainty and limitations.
Describe pretraining's benefit only after comparison with the supervised
baseline. Describe fine-tuning's benefit only relative to the frozen zero-shot
model on the same evaluation.

Do not claim atomistic reconstruction, physically valid simulated trajectories,
replacement of an MD engine, validated free-energy calculations from forecasts,
causality from predictive coupling, or generalization beyond the tested split
and dataset. Observational forecast gains do not establish mechanistic signal
transfer or long-timescale behavior absent from the source trajectories.

## Experiment names and reporting

Use a lowercase, hyphen-separated experiment name:

```text
<dataset>-<split-task>-<feature-set>-<model>-c<C>-h<H>-s<seed>-<config-hash>
misato-unseen-complex-native-v1-persistence-c40-h10-s42-a1b2c3d4
```

`C` and `H` are frame counts; the example is a naming illustration, not an
approved benchmark grid. Split tasks include `seen-trajectory-window`,
`unseen-replica`, `unseen-complex`, and `grouped-target`. Use `cgrid-hgrid` for
a run spanning multiple cells, preserving each cell's dimensions in results.
Use the first eight characters of the full configuration checksum in the name
and retain the full checksum in the manifest; names are labels, not unique IDs.

Every report must include:

- [ ] Hypothesis, claim level, frozen acceptance thresholds, and deviations.
- [ ] Dataset version, source URI/checksum, provenance/license audit, admitted
  feature definitions/units, missingness, exclusions, and QC results.
- [ ] Split task/hash, group identities/counts, overlap checks, and preprocessing
  fit scope; distinguish windows, trajectories, replicas, and complexes.
- [ ] Context/horizon grid in frames and physical time, baseline settings,
  validation-selection procedure, model revision, and training data/budget.
- [ ] Full per-observable/horizon results, absolute errors, baseline ratios,
  paired group-level uncertainty, calibration, width, and autocorrelation.
- [ ] Code commit, lockfile/config checksums, full experiment identifier,
  seeds, hardware, runtime, peak memory/VRAM where applicable, and commands
  sufficient to reproduce artifacts.
- [ ] Failures, unsupported cells, negative results, limitations, and any
  exploratory analyses clearly separated from confirmatory results.

For a negative result, retain the same grid and baselines, report effect sizes
and confidence intervals, distinguish implementation/data failures from lack
of measured skill, and document inconclusive calibration or inadequate sample
size. Do not hide failed observables or redefine success after seeing test
results. Follow-up hypotheses require a new frozen experiment specification.
