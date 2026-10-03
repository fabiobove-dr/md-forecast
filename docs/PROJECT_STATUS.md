# Project Status

This page is the repository-level status summary for `md-forecast`.

For live task tracking, see [GitHub issue #32 — Project Roadmap](https://github.com/fabiobove-dr/md-forecast/issues/32).

## Progress

**14 / 17 planned implementation issues complete (~82%).**

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

## Next

### MVP scientific validation
1. **#14 — MDbind external validation**
   - build the external adapter and semantic-equivalence checks;
   - evaluate transfer without MDbind fine-tuning first;
   - report unseen-replica and unseen-complex results separately;
   - the [source audit](datasets/MDBIND.md) is prepared, but native feature equivalence remains unresolved; no external result is available.

2. **#17 — Reproducible MVP report**
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

The current experiments demonstrate implementation correctness and development/validation behavior. The decisive evidence still needs to come from:

- an independent confirmation of the small development ablation findings (#13), and
- external generalization on MDbind (#14).

Only after those are complete should the final MVP conclusion be frozen in #17.
