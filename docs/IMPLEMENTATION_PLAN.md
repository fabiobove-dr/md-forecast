# md-forecast — Public-Data-First Implementation Plan

## 1. Objective

The MVP tests whether a time-series foundation model can forecast **future molecular-dynamics-derived observables from a partial protein–ligand MD trajectory** with useful accuracy and calibrated uncertainty.

The project does **not** claim to reconstruct future atomistic coordinates or replace a molecular dynamics engine.

Primary hypothesis:

> Given the first part of an MD trajectory, can a pretrained time-series foundation model forecast selected future physical observables better than persistence, statistical baselines, and a compact model trained from scratch?

A stronger secondary hypothesis is:

> Does information from region A improve prediction of future dynamics in region B?

For that extension we use the term **predictive coupling**, not causality or proven signal transfer.

---

## 2. Core strategy: public-data-first

The original plan included generating a new GROMACS training corpus. That is no longer the default.

For the MVP we use large public protein–ligand MD datasets and defer new MD generation until one of these becomes true:

1. public datasets lack a required observable;
2. the forecasting result is strong enough to justify prospective validation;
3. we need longer trajectories or a specific allosteric system absent from public data;
4. an independent controlled simulation set is required for publication/review.

This keeps the first implementation focused on the forecasting hypothesis rather than simulation infrastructure.

---

## 3. Dataset plan

### 3.1 Primary dataset — MISATO

MISATO is the primary MVP dataset.

Relevant properties reported by the dataset authors:

- 16,972 protein–ligand complexes;
- explicit-solvent MD simulations;
- 10 ns simulation length;
- first 2 ns discarded;
- 8 ns retained for analysis;
- 100 stored snapshots per complex;
- downloadable train/validation/test lists;
- MD HDF5 on the order of ~133 GB;
- coordinates and several MD-derived properties are available.

Primary sources:

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/
- Dataset: https://zenodo.org/records/7711953
- Code: https://github.com/t7morgen/misato-dataset

Why MISATO first:

- large enough for domain fine-tuning;
- operationally manageable compared with multi-terabyte raw MD datasets;
- many distinct complexes;
- native MD-derived observables can bootstrap the first benchmark;
- coordinates remain available for later structural feature extraction.

### 3.2 External benchmark — MDbind

MDbind is the preferred external/replicate-aware benchmark after the MISATO pipeline works.

Reported properties:

- 6,300 protein–ligand complexes;
- 10 independent 10-ns simulations per complex;
- 63,000 trajectories total;
- 50 frames per simulation;
- 200 ps frame interval;
- processed datasets are available separately from the much larger raw trajectory collection.

Primary sources:

- Paper: https://academic.oup.com/bioinformatics/article/41/8/btaf429/8238154
- Dataset: https://zenodo.org/records/10390550

Why MDbind matters:

- the replicate structure directly supports stochastic-generalization tests;
- it was produced by a different pipeline, so it is useful as external validation;
- it allows us to distinguish **unseen replica** from **unseen complex** generalization.

The MVP should **not** download the entire raw multi-terabyte collection unless required. Prefer processed data or a documented subset.

### 3.3 License and provenance

Public availability is not equivalent to unrestricted redistribution.

Before using any dataset in published/company-backed experiments:

- record authoritative dataset license/terms;
- record relevant upstream restrictions;
- do not commit source dataset bytes to this repository;
- do not redistribute structures/trajectories unless terms explicitly permit it;
- commit only adapters, manifests, checksums, metadata, derived statistics, and reproducible acquisition instructions.

---

## 4. Forecasting task

For a multivariate observable sequence:

```text
X = [x_1, x_2, ..., x_T]
```

the model receives a prefix of length `C` and predicts the next `H` samples:

```text
X[0:C] -> X[C:C+H]
```

The framework must support:

- variable context length;
- variable forecast horizon;
- univariate forecasting;
- multivariate forecasting;
- probabilistic forecasts where supported;
- optional static/dynamic covariates;
- evaluation in both frame units and physical time (ps/ns).

No model-specific assumptions should leak into the canonical dataset API.

---

## 5. MVP observables

### 5.1 First stage — native MISATO observables

The first benchmark should preferentially use already-computed time-resolved observables rather than recomputing descriptors from coordinates.

Candidate channels to verify during dataset audit:

1. ligand RMSD after protein alignment;
2. complex/protein RMSD;
3. ligand–receptor center-of-mass distance;
4. buried solvent-accessible surface area;
5. MM/GBSA-related quantities when truly available as time series.

The dataset audit must verify exact keys, shapes, units, frame alignment, and missingness before any channel is admitted to the benchmark.

### 5.2 Second stage — derived structural observables

After the native-feature benchmark works, derive additional channels such as:

- pocket backbone RMSD;
- protein–ligand contact count;
- fraction of reference contacts;
- hydrogen-bond count;
- hydrophobic contact count;
- ligand SASA;
- minimum ligand–protein heavy-atom distance;
- selected residue–residue or residue–ligand distances.

Feature definitions must be versioned and include exact atom selections, cutoffs, and units.

---

## 6. Canonical data model

Every source trajectory is normalized into one internal representation.

### 6.1 Trajectory manifest

Required fields:

```text
dataset_id
dataset_version
source_record
trajectory_id
system_id
pdb_id
protein_id
ligand_id
replicate_id
split_group_id
frame_count
frame_interval_ps
duration_ns
feature_set_version
source_checksum
license_id
provenance_uri
```

Unavailable fields may be null, but schema and semantics must remain stable.

### 6.2 Storage

Suggested local layout:

```text
data/
├── external/       # ignored; downloaded source datasets
├── cache/          # ignored
├── processed/      # ignored by default; normalized series
├── manifests/      # small/versioned metadata may be committed
└── README.md
```

Normalized tabular time series should use Parquet/Arrow-oriented storage.

Committed manifests must not contain absolute local paths.

---

## 7. Leakage prevention and split strategy

The most important data rule is:

> **Split trajectories/systems before generating overlapping forecasting windows.**

Never generate overlapping windows first and randomly assign them across train/validation/test.

Required split modes:

### A. Official dataset split
Reproduce the dataset authors' train/validation/test split where available.

### B. Unseen complex
All windows from the same protein–ligand complex belong to one split.

### C. Grouped structural/target split
Where metadata permits, group related systems so highly similar complexes do not leak across train and test.

### D. MDbind unseen-replica benchmark
Hold out one or more independent replicas while the complex identity remains seen.

This is a separate task and must never be reported as unseen-complex generalization.

---

## 8. Context × horizon benchmark

The benchmark is defined in physical time and converted to frame counts for each dataset.

The exact MISATO grid must be chosen after verifying actual timestamps, but the framework should support combinations such as:

- short context -> short horizon;
- medium context -> short/medium horizon;
- long context -> multiple horizons up to the remaining trajectory length.

Impossible pairs must fail explicitly rather than silently truncate.

Each result record should include:

```text
dataset
split
trajectory_id
feature
context_frames
context_ns
horizon_frames
horizon_ns
model
model_version
seed
metrics
runtime
hardware
```

---

## 9. Baselines

A foundation model must not be evaluated in isolation.

Required baselines:

### Naive/statistical
- last-value persistence;
- context mean;
- linear extrapolation;
- autoregressive model;
- VAR for multivariate series.

### Learned-from-scratch
At least one compact forecasting model trained on exactly the same training data, such as NLinear/DLinear or a compact TCN.

This separates the value of pretraining from the value of simply fitting a nonlinear predictor to a large MD dataset.

---

## 10. Primary foundation model — Chronos-2

Chronos-2 is the primary TSFM for the MVP.

The implementation should expose it through a generic `ModelAdapter` so the project remains model-agnostic.

Experiment order:

1. persistence/statistical baselines;
2. supervised-from-scratch baseline;
3. Chronos-2 zero-shot;
4. Chronos-2 domain fine-tuning on MISATO;
5. held-out MISATO complex evaluation;
6. untouched MDbind external evaluation.

Do not fine-tune before zero-shot/evaluation infrastructure and leakage checks are complete.

---

## 11. 24 GB GPU target

All ML workflows should be designed for one 24 GB VRAM GPU.

Requirements:

- mixed precision where appropriate;
- configurable batch size;
- gradient accumulation;
- resumable checkpoints;
- deterministic seeds where feasible;
- peak VRAM logging;
- throughput logging;
- CPU/RAM/I/O profiling.

Dataset I/O may become the practical bottleneck before model memory.

---

## 12. Evaluation

### Point forecast metrics
- MAE;
- RMSE;
- normalized/scaled error;
- MASE or explicit error ratio versus persistence.

### Probabilistic metrics
Where supported:
- pinball loss;
- prediction interval coverage;
- interval width;
- CRPS if implemented robustly.

### Horizon analysis
The central result is **forecast error versus horizon**, not one aggregate score.

For each observable report:

- baseline error vs horizon;
- TSFM error vs horizon;
- relative improvement vs persistence;
- uncertainty calibration vs horizon.

### Autocorrelation control
For every forecasted observable:

- estimate autocorrelation / characteristic correlation time;
- compare useful forecast horizon with intrinsic autocorrelation;
- explicitly detect cases where apparent skill is explained only by slow feature dynamics.

---

## 13. Experimental stages

### Stage 0 — dataset feasibility
Load a small MISATO subset, verify exact observables, units, timestamps, splits, and license/provenance.

### Stage 1 — forecasting harness
Build canonical storage, leak-free windowing, and baseline evaluation.

### Stage 2 — Chronos-2 zero-shot
Determine whether an off-the-shelf TSFM already contains useful priors for MD observables.

### Stage 3 — MISATO domain fine-tuning
Test whether MD-specific fine-tuning materially improves held-out forecasting.

### Stage 4 — multivariate/covariate ablation
Compare RMSD-only versus increasingly richer physical state representations.

### Stage 5 — MDbind external validation
Evaluate an untouched subset without MDbind fine-tuning first. Report unseen-replica and unseen-complex results separately.

### Stage 6 — predictive coupling
For regions A and B compare:

```text
B_past -> B_future
```

against:

```text
[A_past, B_past] -> B_future
```

A reproducible gain is evidence of predictive coupling, not automatically causality.

---

## 14. Repository architecture

```text
md-forecast/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── AGENTS.md
├── configs/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── src/md_forecast/
│   ├── data/
│   │   ├── schemas.py
│   │   ├── registry.py
│   │   ├── splits.py
│   │   ├── windows.py
│   │   └── public/
│   │       ├── misato.py
│   │       └── mdbind.py
│   ├── features/
│   │   ├── native.py
│   │   └── structural.py
│   ├── models/
│   │   ├── base.py
│   │   ├── persistence.py
│   │   ├── statistical.py
│   │   ├── supervised.py
│   │   └── chronos2.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   ├── calibration.py
│   │   └── leakage.py
│   └── analysis/
│       ├── autocorrelation.py
│       └── predictive_coupling.py
├── tests/
└── data/
    └── README.md
```

---

## 15. MVP success criteria

The project is **not** successful merely because a forecast looks plausible.

### Minimum result
> On held-out MISATO trajectories, Chronos-2 produces calibrated forecasts for at least one physically meaningful MD observable and beats persistence plus statistical baselines over a non-trivial horizon.

### Stronger result
> Improvement remains for completely held-out protein–ligand complexes and exceeds a compact supervised model trained from scratch on the same MD series.

### External result
> A model trained/fine-tuned on MISATO retains measurable forecast skill on an untouched MDbind subset.

### Research-extension result
> Dynamics from region A reproducibly improve prediction of future dynamics in region B across independent trajectories/replicas after controlling against simpler lagged baselines.

Negative results are valid and should be reported.

---

## 16. Explicitly deferred work

Out of scope for the MVP:

- generating a new multi-microsecond MD training corpus;
- forecasting atom-by-atom coordinates;
- claiming forecasted observables form a physically valid MD trajectory;
- replacing MD for free-energy calculations;
- claiming causality from predictive coupling;
- production deployment/UI/API.

Controlled prospective MD generation can be reintroduced later as an independent validation phase if the public-data results justify it.
