# Project Status

This page is the repository-level status summary for `md-forecast`.

For current work, see [prediction-quality roadmap #46](https://github.com/fabiobove-dr/md-forecast/issues/46).
The [original MVP roadmap #32](https://github.com/fabiobove-dr/md-forecast/issues/32)
remains the completed historical record.

## Progress

**17 / 17 planned implementation issues complete (100%).**

The foundational engineering and ML work is complete. All nine prediction-quality follow-up deliverables are also complete, including independent real-data confirmation. Engineering completion and practical scientific usefulness remain separate outcomes.

## Completed

### Foundations
- #1 Scientific contract and MVP acceptance criteria
- #2 Python 3.14 package, tooling, CI, MkDocs, quality gates

### MISATO data pipeline
- #3 MISATO schema/provenance/license audit
- #4 Resumable verified MISATO acquisition
- #5 Canonical Pydantic/Arrow/Parquet data contracts
- #6 MISATO native observable adapter
- #7 Leak-free splits, variable windows, scoped preprocessing
- #8 MISATO QC and autocorrelation analysis
- #15 Structural geometry feature extraction

### Forecasting and evaluation
- #9 Persistence/statistical/supervised baselines
- #10 Chronos-2 zero-shot integration
- #11 Reproducible grouped context × horizon benchmark
- #12 Chronos-2 fine-tuning within the <=24 GB GPU target
- #13 Controlled input ablations on the development holdout; repeated results agree, with inconclusive paired input-set benefit
- #14 Audited common geometric MDbind subset, equivalent-feature adapter, untouched external and seen-system replica benchmarks; repeated results agree, without confirmed external superiority
- #17 Reproducible immutable MVP synthesis from saved experiments; evidence-linked tables/figures and explicit null results ([workflow](models/MVP_REPORT.md))
- #16 Frozen pocket/distal predictive-coupling experiment on held-out MDbind replicas; repeated results agree, without corrected added-region benefit ([experiment](models/PREDICTIVE_COUPLING.md))

## What has been achieved technically

The repository now supports:

- typed Pydantic configuration and scientific data contracts;
- reproducible public-dataset acquisition and provenance;
- versioned canonical trajectory storage;
- memory-bounded MISATO extraction;
- leak-free forecasting datasets;
- train-only/context-local preprocessing;
- autocorrelation-aware exploratory QC;
- persistence, context mean, linear extrapolation, AR, VAR, and NLinear;
- pinned Chronos-2 zero-shot inference;
- grouped probabilistic benchmarking with trajectory/system-level uncertainty;
- reproducible Chronos-2 fine-tuning and checkpoint resume;
- structural interaction feature extraction;
- strict CI, typing, tests, coverage, complexity checks, and MkDocs documentation.

## Current scientific status

Independent confirmation is complete on **99 unseen native complexes**, **66
untouched external complexes / 660 replicas**, and **six development-seen
complexes with untouched replica 10**. Every model uses matched windows and
frozen development choices. No fitting or recalibration occurred on TEST.
See [the complete confirmation results](models/FOLLOWUP_CONFIRMATION.md) for
all cells, corrected effects, failure evidence, provenance and measured costs.

**No model satisfies all frozen practical-usefulness criteria:** a corrected
10% MAE gain over both references plus useful uncertainty. Smaller measurable
point gains do exist for contact count and retained fraction. Native zero-shot
contacts meet the original contract's less demanding minimum (corrected gains
and calibrated intervals with width/pinball reported), but the stronger
compact-model claim and the follow-up practical-usefulness threshold do not
pass. This is limited scalar-forecast skill, not an MD replacement.

Fine-tuning is now genuinely measured against zero-shot. Native primary
fine-tuned MAE is 11.47 atom-pair contacts, 0.0562 retained fraction (5.62
percentage points) and 0.0816 Å. Native contact/fraction intervals under-cover;
external fine-tuning worsens minimum-distance error. Seven crossed fine-tuned
contact quantiles invalidate probability metrics for two external cells;
all raw values and point errors remain included. Median curves still capture
little future fluctuation amplitude. These findings do not prove intrinsic
unpredictability.

Native physical cadence remains unresolved; external horizons cover only
1–2 ns. QC reduces the external cohort below its planned 80-system precision
target; six seen systems cannot support corrected replica superiority. Target,
chemotype and pretraining independence remain unknown. The older unresolved
native observables stay separate from the newly equivalent raw geometric
features. Historical MVP and predictive-coupling results remain unchanged;
added-region benefit remains inconclusive.

## Prediction-quality follow-up

All nine follow-up issues have complete implementation and real-data evidence.
Their acceptance criteria are tracked in [roadmap #46](https://github.com/fabiobove-dr/md-forecast/issues/46).
They do not change the completed MVP count. Prediction improvements are limited
to the measured effects above; practical usefulness remains unconfirmed.

1. [#37](https://github.com/fabiobove-dr/md-forecast/issues/37), complete: admit
   follow-up feature semantics, timing and structural preprocessing.
2. [#38](https://github.com/fabiobove-dr/md-forecast/issues/38), complete: prepare
   260 native development systems and 48 seen-system external trajectories;
   reserve 100 native and 50 unseen external confirmation systems before
   decoding outcomes ([protocol and real QC](datasets/FOLLOWUP_COHORT.md)).
3. [#39](https://github.com/fabiobove-dr/md-forecast/issues/39), complete: measure
   forecast spread separately from error, estimate system-level development
   dependence, and freeze the feasible 20/40 × 5/10 frame grid
   ([matched results](models/FORECAST_DIAGNOSTICS.md)).
4. [#40](https://github.com/fabiobove-dr/md-forecast/issues/40), complete:
   validation-select regularized compact/statistical baselines using existing
   implementations; repeated real selection gives identical fitted states
   and outcomes ([results](models/FOLLOWUP_BASELINES.md)).
5. [#41](https://github.com/fabiobove-dr/md-forecast/issues/41), complete: run
   six geometric fine-tuning trials and repeat matched zero-shot comparisons;
   representative normalized development MAE improves 1.29%, while compact
   references remain competitive ([evidence](models/FOLLOWUP_FINETUNING.md)).
6. [#42](https://github.com/fabiobove-dr/md-forecast/issues/42), complete: test
   zero-shot and matched geometric fine-tuning under target-only/joint/shifted
   histories with AR/VAR controls; added-input benefit remains inconclusive
   ([all results](models/FOLLOWUP_INPUTS.md)).
7. [#43](https://github.com/fabiobove-dr/md-forecast/issues/43), complete: add
   TRAIN-only residual probability references and freeze resolvable primary
   inference; fine-tuned contact/fraction intervals under-cover development
   ([full probability evidence](models/FOLLOWUP_PROBABILITY.md)).
8. [#44](https://github.com/fabiobove-dr/md-forecast/issues/44), complete: package
   the checksum-verified offline HTML overview with per-point errors, explicit
   tolerance rules, matched curves and inline 3Dmol.js ([workflow](models/OFFLINE_OVERVIEW.md)).
9. [#45](https://github.com/fabiobove-dr/md-forecast/issues/45), complete: score
   all three frozen independent tasks and publish a negative/inconclusive
   practical-usefulness decision with smaller measurable effects and preserved
   failures ([full evidence](models/FOLLOWUP_CONFIRMATION.md)).

The standalone offline HTML overview includes the complete confirmation grid,
matched fine-tuned/zero-shot/reference curves, errors, probability-invalid
flags and the inline observed molecular reference. An application tolerance
remains user-controlled and unset by default; it is separate from scientific
acceptance. Smoother or noisier median curves alone do not determine accuracy.
