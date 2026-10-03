# Project Status

This page is the repository-level status summary for `md-forecast`.

For live task tracking, see [GitHub issue #32 — Project Roadmap](https://github.com/fabiobove-dr/md-forecast/issues/32).

## Progress

**15 / 17 planned implementation issues complete (~88%).**

The foundational engineering and ML work is complete. The remaining work is primarily scientific validation, external generalization, and final reporting.

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

## Next

### MVP scientific validation
1. **#17 — Reproducible MVP report**
   - consolidate held-out results, uncertainty, calibration, ablations, external validation, hardware/runtime, and limitations into a paper-ready artifact.

## Research extension

- **#16 — Predictive coupling / allosteric-region forecasting**
  - compare `B_past -> B_future` with `[A_past, B_past] -> B_future`;
  - treat improved prediction as predictive coupling, not causality.

This is valuable research work but is not required to establish the core MVP.

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

Issue #17 can now consolidate engineering completion, measured outcomes and
unsupported claims without assuming that successful infrastructure proves the
scientific hypothesis. Native MISATO/MDbind equivalence remains blocked.
