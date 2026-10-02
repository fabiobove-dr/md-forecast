# MISATO development QC

Issue #8 adds descriptive QC and sample-index dependence diagnostics, not a
forecasting experiment. It consumes the [canonical extraction](MISATO_ADAPTER.md)
and [official pre-window split](PREPARATION.md). No Chronos, GPU, new dependency,
full MD download, or source redistribution is involved.

## Frozen development selection

`DevelopmentManifest` embeds the original official `SplitManifest`, selection
seed/group limit, and sorted trajectory identities. Selection shuffles sorted
**training dependency groups** with a local seeded RNG, takes up to the explicit
limit, and includes every trajectory in a selected group. It never reads values
to select systems. Grouped/reassigned source splits, no training groups, partial
groups, and forged/held-out selections fail. The QC loader additionally checks
Parquet source metadata **before reading numerical columns**, so even a test
file accidentally mapped under a training filename is rejected before its
payload is read. Source paths must be direct local extraction files.

The pinned 20-system source sample has 19 official training systems, no
validation systems, and one test system (`16PK`). This analysis uses only:

```text
10GS 11GS 13GS 184L 185L 186L 187L 188L 1A07 1A08
1A09 1A0Q 1A1B 1A1C 1A1E 1A28 1A2C 1A30 1A3E
```

Source SHA-256:
`554d20ea0822949e1a5b0dc1826f50b87e71de5a7ed83ac1fb5a2e68c64547de`.
This is the upstream **sample** hash, not the full MISATO MD checksum.
The development manifest hash is
`sha256:b8b9653f54bc516453d933f3e542df415e07e6de497a1ca7ebb0176443fd99ed`.
The analysis configuration hash is
`sha256:3714b44b74a1fae74aa293314c663177a1026cdc225452b6061fdc392567b2b8`.
Full source feature definitions, units, provenance and original held-out
**metadata only** remain in the manifest; test numerical values are absent.

## Diagnostic definitions

The explicit protocol is [misato-sample-qc.yaml](https://github.com/fabiobove-dr/md-forecast/blob/main/configs/analysis/misato-sample-qc.yaml)
in the repository (the documentation site does not bundle configuration files).
It records group limit/seed, each channel's near-constant standard-deviation
cutoff in native units, event threshold, lag limit, and window grid.

- Each trajectory gets min/Q1/median/Q3/max, mean, population variance/std,
  missingness, exact-constant and near-constant flags. Published canonical series
  cannot contain missing/nonfinite values: malformed inputs abort rather than
  being silently removed or interpolated. The bundle preserves extraction QC,
  including requested/missing/dropped source systems; zero post-extraction
  missingness is not evidence that the full dataset has no missing data.
- ACF uses global demeaning and a fixed lag-zero sum-of-squares denominator,
  following the [NIST sample ACF definition](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm).
  Each curve includes lags 0 through `min(max_lag_frames, T-1)`. Constant channels
  have null curves, not fabricated perfect correlation.
- The characteristic decay diagnostic is the **first** positive lag with
  `ACF <= exp(-1)`. No crossing within the reported lag range is right-censored;
  constants have no defined decay. This is not an integrated autocorrelation
  time, an equilibrium relaxation estimate, or a model-selection optimum.
- Linear drift per frame and second-half minus first-half mean divided by whole
  trajectory std flag exploratory location changes. They are not formal tests
  of stationarity. No detrending is applied to the raw ACF.
- Jump-event prevalence counts increments whose deviation from the mean
  increment exceeds `event_sigma * std(increments)`. Counts and denominators
  are stored per trajectory; this heuristic is not a biological event label
  or a formal change-point algorithm.
- Cross-correlations are zero-lag, within-trajectory Pearson coefficients.
  Constant rows/columns are null. No causal or predictive-coupling claim follows.
- Feasibility uses the existing window resolver and the exact stride-dependent
  count per cell. Impossible/unsupported cells stay visible with zero counts
  and reasons; no silent truncation.

Sample-index ACF is descriptive: physical equispacing cannot be verified from
the inspected MISATO export. **No ps/ns correlation times or physical window
durations are assigned.** Nonuniform physical sampling must not be interpreted
as a uniformly sampled physical ACF. All statistics below summarize trajectories
equally, never treat their overlapping windows as independent observations.

## Real development-sample results

Nineteen trajectories × 100 frames × four features were analyzed. There were
zero missing/nonfinite canonical values; the original extraction reported zero
missing/dropped requested systems. There are **no exact or near-constant
channels** under the conservative numerical QC thresholds. Those thresholds
are engineering flags, not biological significance thresholds or feature
exclusion rules.

| Observable (native unit) | Per-trajectory std min / median / max | First 1/e decay min / median / max (frames) | Median ACF at lag 1 / 5 / 10 / 20 |
| --- | --- | --- | --- |
| Native ligand RMSD (Å) | 0.134 / 0.342 / 1.526 | 1 / 2 / 20 | 0.464 / 0.130 / 0.067 / 0.033 |
| Ligand–receptor COM distance (Å) | 0.227 / 0.358 / 1.691 | 1 / 2 / 18 | 0.429 / 0.190 / 0.114 / 0.003 |
| Buried SASA (Å²) | 12.916 / 33.375 / 59.765 | 1 / 1 / 9 | 0.187 / 0.040 / 0.071 / −0.010 |
| Native interaction energy (kcal/mol) | 1.459 / 4.095 / 19.596 | 1 / 1 / 10 | 0.147 / 0.020 / 0.037 / −0.039 |

No decay was censored within 40 lags. All **76 complete per-observable,
per-trajectory curves** are available in the generated `acf.csv` and `report.json`;
the table is only a descriptive summary, not a replacement for those curves.
Plot `lag_frames` against `acf`, grouping by `feature_id` and `trajectory_id`.
Curves are not pooled across unrelated trajectory endpoints.

Median absolute half-mean shifts were 0.398, 0.470, 0.330 and 0.403 whole-series
standard deviations respectively. This motivates retaining drift diagnostics
and does not establish stationarity. At the declared 3σ increment threshold,
the four channels had 12, 4, 10 and 5 flagged increments out of **1,881 each**.
These sparse flags must not be overinterpreted as molecular transitions.

Median within-trajectory cross-correlations:

| Pair | Median Pearson r |
| --- | --- |
| RMSD / COM distance | 0.232 |
| RMSD / buried SASA | 0.054 |
| RMSD / interaction energy | 0.073 |
| COM distance / buried SASA | −0.133 |
| COM distance / interaction energy | 0.131 |
| Buried SASA / interaction energy | −0.036 |

## Development grid and justification

All selected trajectories have 100 ordered samples. The exploratory frame grid
uses C={20,40,60}, H={1,5,10,20}, stride=1:

| Context frames | H=1 | H=5 | H=10 | H=20 |
| --- | --- | --- | --- | --- |
| 20 | 1,520 | 1,444 | 1,349 | 1,159 |
| 40 | 1,140 | 1,064 | 969 | 779 |
| 60 | 760 | 684 | 589 | 399 |

Counts are overlapping **development windows**, each cell involving 19 systems,
not independent sample sizes. `C+H <= 80` leaves at least 21 start positions per
trajectory in every cell. C=20 spans roughly the longest observed first-decay
lag; C=40/60 test longer histories within the actual 100-frame budget. H=1 is a
trivial-continuation control; H=5/10/20 probe beyond median decay and up to the
slowest first crossing. H=20 is not beyond every trajectory's dependence scale;
the full curves and short-series/drift limitations remain relevant. None of
these choices was based on final-test data or model performance.

This grid is suitable for rapid **development** iteration on the sample. It is
not an approved physical-time grid or frozen confirmatory benchmark. The sample
is not a random or representative draw of all MISATO complexes; no validation
systems are available here. Before model selection/reporting, acquire a suitable
official validation subset and freeze the benchmark protocol. Do not repurpose
`16PK` as validation or infer full-dataset forecast skill from this QC.

## Reproduction and artifacts

After reproducing the pinned sample extraction described in the adapter docs,
run from a clean checkout containing this implementation:

```sh
uv run --locked md-forecast qc-misato \
  --input data/processed/misato-native-sample \
  --output data/processed/misato-development-qc \
  --config configs/analysis/misato-sample-qc.yaml \
  --code-commit "$(git rev-parse HEAD)" --lockfile uv.lock
```

Output contains atomic hash-verified `report.json`, `development.json`,
`config.json`, `extraction.json`, and complete `acf.csv`. The report includes
source metadata, split/development/config hashes, caller-declared code revision,
and actual lockfile hash. Code revision is an explicit provenance declaration,
not an attestation: run from a clean tree and record any deviations. Identical
data/config/code/lockfile reproduce payloads and CSV bytes. Existing output is
refused; a failed run publishes no partial bundle. Publication assumes one
writer, as does extraction. Analyze one trajectory at a time; only summaries
and bounded ACF curves accumulate. ACF work is O(T × requested lag count),
sufficient for 100-frame exports; longer admitted series may require FFT.

Generated reports/source bytes stay ignored and local. This page is the reviewed
project methodology and curated result summary, not a committed dataset or raw
generated report. Passing implementation tests proves boundary/numerical behavior,
not stationarity, molecular event validity, calibrated uncertainty, or forecast
skill.
