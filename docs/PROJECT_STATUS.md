# Project Status

This page is the repository-level status summary for `md-forecast`.

For current work, see [prediction-quality roadmap #46](https://github.com/fabiobove-dr/md-forecast/issues/46).
The [original MVP roadmap #32](https://github.com/fabiobove-dr/md-forecast/issues/32)
remains the completed historical record.

## Progress

**17 / 17 planned implementation issues complete (100%).**

The foundational engineering and ML work is complete. The planned software and experiment deliverables are complete. Independent confirmation of the scientific hypothesis remains open.

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

The software stack is mature enough to run the intended experiments, but the final scientific claim is **not yet established**.

The common geometric MDbind benchmark is complete and reproducible on six exact
PDB-disjoint complexes and 60 replicas. No lead beats both persistence and the
MISATO-selected statistical baseline under the frozen corrected comparison
policy. The replica task is separate, with its local baseline fit disclosed.
See [external validation](models/EXTERNAL_VALIDATION.md) for the full negative /
inconclusive result and sampling limits. The input ablation findings remain
small-development evidence; larger independent confirmation is still needed.

The predictive-coupling extension is also complete on independently frozen
geometric regions and held-out replicas; added-region gain remains
inconclusive for Chronos and its conventional autoregressive comparator.

The final MVP report consolidates measured outcomes and unsupported claims
without assuming that successful infrastructure proves the scientific
hypothesis. Native MISATO/MDbind equivalence remains blocked.

## Prediction-quality follow-up

The first two follow-up issues are complete; seven implementation/experiment
issues remain open. Their acceptance criteria
are tracked in [roadmap #46](https://github.com/fabiobove-dr/md-forecast/issues/46).
They do not change the completed MVP count or establish better predictions yet.

1. [#37](https://github.com/fabiobove-dr/md-forecast/issues/37), complete: admit
   follow-up feature semantics, timing and structural preprocessing.
2. [#38](https://github.com/fabiobove-dr/md-forecast/issues/38), complete: prepare
   260 native development systems and 48 seen-system external trajectories;
   reserve 100 native and 50 unseen external confirmation systems before
   decoding outcomes ([protocol and real QC](datasets/FOLLOWUP_COHORT.md)).
3. [#39](https://github.com/fabiobove-dr/md-forecast/issues/39): diagnose forecast
   flattening and choose development context/horizon grids.
4. [#40](https://github.com/fabiobove-dr/md-forecast/issues/40): validation-select
   regularized compact/statistical baselines using existing implementations.
5. [#41](https://github.com/fabiobove-dr/md-forecast/issues/41): run meaningful
   geometric fine-tuning with matched zero-shot comparisons.
6. [#42](https://github.com/fabiobove-dr/md-forecast/issues/42): test added observed
   inputs with lagged comparators and negative-input controls.
7. [#43](https://github.com/fabiobove-dr/md-forecast/issues/43): add probabilistic
   references and prospectively resolvable grouped uncertainty.
8. [#44](https://github.com/fabiobove-dr/md-forecast/issues/44): package the offline
   HTML report workflow, including inline 3Dmol.js and per-point errors.
9. [#45](https://github.com/fabiobove-dr/md-forecast/issues/45): run the frozen
   independent benchmark and publish a useful-skill decision.

Issue #44 can proceed independently. Model selection uses development only;
confirmation follows the dependencies recorded in each issue. Smoother or
noisier median curves alone do not determine accuracy. The existing eight-update
native fine-tuning pilot is an integration result, not adequate evidence about
the benefit of geometric domain adaptation.
