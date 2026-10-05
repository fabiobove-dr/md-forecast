# Development forecast diagnostics

Issue [#39](https://github.com/fabiobove-dr/md-forecast/issues/39) separates
forecast smoothness from accuracy on the fresh development cohort. These
results select development settings; they do not confirm superiority on
untouched systems. Calibration and confirmation outcomes remain unopened.

## What is measured

`forecast_diagnostics` uses population standard deviation (`ddof=0`) across
each window's future horizon. The ratio is predicted spread / actual spread:
zero is a constant prediction, one means equal spread, and neither value
measures accuracy. An exactly constant actual future has an undefined ratio,
exported as Arrow null, with its count retained. A one-step horizon also has
undefined spread ratio. Subtract a retained value before computing spread to
preserve exact constants despite floating-point rounding.

Mean bias is prediction minus observation; MAE is mean absolute error in the
observable's native unit. RMSE takes the root after averaging squared errors
within each trajectory. Per-lead errors cover every future step. Average
windows within a trajectory, then replicas within a system, then systems with
equal weight. Overlapping windows do not become independent samples.

Changes subtract the last observed origin on both sides. Predicted/actual
mean changes help show reversion or continuation; change-from-origin MAE is
algebraically the same as ordinary MAE, not a second accuracy claim. Every
candidate, feature, window and lead is retained, including losing settings.
No percentage "accuracy" is invented for continuous values.

## Development dependence and grid

Reuse the existing globally demeaned, fixed-denominator sample ACF. Average
replicas inside their source system before resampling system means (seed 42,
2,000 resamples, descriptive 95% intervals). Whole-system uncertainty measures
between-system variation conditional on these finite sequences; it does not
remove short-sequence bias or estimate a precise intrinsic relaxation time.
All lag intervals are exploratory and pointwise, without superiority claims.

Restricted decay is the first 1/e crossing, capped at lag 20. Noncrossing
sequences are explicitly right-censored. Positive integrated ACF is 0.5 plus
the sum before the first nonpositive lag, limited to observed lags. Constant
replicas have undefined correlation and are counted rather than imputed.
No constant trajectory occurred in these admitted data. Native TRAIN and VAL
are separate; seen-system TRAIN replicas 1–6 and VAL replicas 7–8 are separate
but remain dependent within each of their six source complexes.

| Task | Observable | Systems | Restricted decay, frames (95% interval) | Censored fraction |
| --- | --- | ---: | --- | ---: |
| native-train | Contacts (pairs) | 200 | 2.210 (1.835–2.635) | 0.000 |
| native-train | Reference fraction (0–1) | 200 | 4.420 (3.735–5.180) | 0.015 |
| native-train | Minimum distance (Å) | 200 | 1.220 (1.090–1.375) | 0.000 |
| native-validation | Contacts (pairs) | 60 | 3.117 (2.250–4.150) | 0.000 |
| native-validation | Reference fraction (0–1) | 60 | 4.967 (3.783–6.335) | 0.017 |
| native-validation | Minimum distance (Å) | 60 | 1.017 (1.000–1.050) | 0.000 |
| seen-train | Contacts (pairs) | 6 | 1.861 (1.167–2.667) | 0.000 |
| seen-train | Reference fraction (0–1) | 6 | 2.667 (1.722–3.778) | 0.000 |
| seen-train | Minimum distance (Å) | 6 | 1.250 (1.028–1.528) | 0.000 |
| seen-validation | Contacts (pairs) | 6 | 1.083 (1.000–1.250) | 0.000 |
| seen-validation | Reference fraction (0–1) | 6 | 1.917 (1.000–3.417) | 0.000 |
| seen-validation | Minimum distance (Å) | 6 | 1.000 (1.000–1.000) | 0.000 |

Freeze contexts **20 / 40 frames**, horizons **5 / 10 frames**, stride 10.
These contexts exceed the mean training decay estimates; horizon 5 approaches
or exceeds them, and horizon 10 extends beyond them. The longer tail of
individual systems and censored fraction remain limitations; no claim that
all systems decorrelate within these means is supported. Every cell fits the
100-frame native and 50-frame external exports. Long-context external cells
have only one window per validation replica, an explicit sampling constraint.

Native time stays in **frames**: no physical cadence has been verified. External
200 ps spacing gives 3.8 / 7.8 ns first-to-last context spans and 1 / 2 ns
origin-to-last-future horizons. Distinct source cadences do not establish the
same physical forecast task across datasets. Sparse 200 ps sampling can hide
faster dynamics; smooth conditional medians do not establish irreducible noise.

## Matched descriptive errors

Candidates include persistence, context mean and existing context-local
regularized AR/VAR: lags 1/3, ridge 0.1/1/10. Select an AR and VAR configuration
per cell on native VAL equal-system MAE divided by native TRAIN-context
standard deviations, averaged across the three channels. Keep every candidate
score. Transfer these choices unchanged to seen-system VAL. Ridge acts on
native-unit designs here; more complete baseline/scaler selection belongs to
#40. The same VAL outcomes used for selection make these comparisons optimistic.

### Native selection data

| Cell | Observable | Persistence MAE | Mean MAE | Selected AR MAE | Selected VAR MAE | Chronos MAE | Chronos spread ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c20-h5 | Contacts (pairs) | 13.20167 | 11.50840 | 11.20490 | 11.17867 | 11.04612 | 0.101 |
| c20-h5 | Reference fraction (0–1) | 0.06378 | 0.06080 | 0.05758 | 0.05736 | 0.05391 | 0.123 |
| c20-h5 | Minimum distance (Å) | 0.09916 | 0.07130 | 0.07116 | 0.07205 | 0.07394 | 0.114 |
| c20-h10 | Contacts (pairs) | 13.53646 | 11.73267 | 11.75508 | 11.49685 | 11.31800 | 0.103 |
| c20-h10 | Reference fraction (0–1) | 0.06626 | 0.06345 | 0.05925 | 0.06036 | 0.05728 | 0.127 |
| c20-h10 | Minimum distance (Å) | 0.09948 | 0.07354 | 0.07379 | 0.07402 | 0.07531 | 0.098 |
| c40-h5 | Contacts (pairs) | 13.12778 | 11.95635 | 11.36162 | 11.41004 | 11.16149 | 0.109 |
| c40-h5 | Reference fraction (0–1) | 0.06152 | 0.06743 | 0.05590 | 0.05969 | 0.05241 | 0.121 |
| c40-h5 | Minimum distance (Å) | 0.10076 | 0.07283 | 0.07406 | 0.07351 | 0.07426 | 0.110 |
| c40-h10 | Contacts (pairs) | 13.25028 | 12.05228 | 11.50994 | 11.66106 | 11.23182 | 0.096 |
| c40-h10 | Reference fraction (0–1) | 0.06319 | 0.06864 | 0.05843 | 0.05937 | 0.05461 | 0.116 |
| c40-h10 | Minimum distance (Å) | 0.10072 | 0.07496 | 0.07564 | 0.07612 | 0.07590 | 0.085 |

### Seen selection data

| Cell | Observable | Persistence MAE | Mean MAE | Selected AR MAE | Selected VAR MAE | Chronos MAE | Chronos spread ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c20-h5 | Contacts (pairs) | 12.23333 | 9.65472 | 9.63332 | 9.60644 | 9.56727 | 0.095 |
| c20-h5 | Reference fraction (0–1) | 0.06799 | 0.06894 | 0.06367 | 0.06432 | 0.05851 | 0.174 |
| c20-h5 | Minimum distance (Å) | 0.10349 | 0.08692 | 0.08751 | 0.08625 | 0.08788 | 0.101 |
| c20-h10 | Contacts (pairs) | 11.93056 | 9.70472 | 9.69717 | 9.67272 | 9.48311 | 0.085 |
| c20-h10 | Reference fraction (0–1) | 0.06624 | 0.06793 | 0.05562 | 0.06329 | 0.05843 | 0.155 |
| c20-h10 | Minimum distance (Å) | 0.10426 | 0.08609 | 0.08659 | 0.08526 | 0.08472 | 0.129 |
| c40-h5 | Contacts (pairs) | 10.93333 | 9.03792 | 8.86425 | 8.97910 | 9.21996 | 0.113 |
| c40-h5 | Reference fraction (0–1) | 0.05156 | 0.07065 | 0.04961 | 0.06353 | 0.04737 | 0.081 |
| c40-h5 | Minimum distance (Å) | 0.11680 | 0.08601 | 0.08992 | 0.08723 | 0.09054 | 0.148 |
| c40-h10 | Contacts (pairs) | 10.57500 | 8.80083 | 8.67942 | 8.99299 | 8.65632 | 0.122 |
| c40-h10 | Reference fraction (0–1) | 0.05174 | 0.07406 | 0.05220 | 0.05457 | 0.04984 | 0.081 |
| c40-h10 | Minimum distance (Å) | 0.11610 | 0.08250 | 0.08343 | 0.08714 | 0.08420 | 0.136 |

Native Chronos medians have about **8–13%** of actual future spread, while
selected AR/VAR are often comparably smooth. On native C40/H10, Chronos contact
MAE is 11.23182 pairs versus persistence 13.25028 and selected AR 11.50994;
reference-fraction MAE is 0.05461 versus 0.06319 and 0.05843. Minimum-distance
MAE is 0.07590 Å versus context mean 0.07496 Å: simpler averaging still wins.

Contact-count advantages are not confined to one-step continuation: at native
C40/H10 lead 10, Chronos / selected AR / persistence MAEs are 10.99409 /
11.38480 / 12.83611 pairs. Chronos loses slightly to AR at lead 5 (11.49821
versus 11.46782). These are mixed descriptive validation differences, without
corrected superiority inference. Noisy curves would not be a valid remedy.
Seen-system contact differences vary across cells and its sample is only six
complexes; this is a replica-development task, not unseen-complex confirmation.

All models see 1,680 native validation windows across 60 systems and 96 external
validation windows across 12 replicas/six complexes, on identical cells within
each task. Counts by cell are native 480/480/360/360 and seen 36/36/12/12.
Each native five-step cell has one constant-future fraction window; its spread
ratio is null for every model. Ten-step and external cells have no constant
futures. The all-window/lead exports retain this information.

## Reproduction and artifact contract

```sh
uv run --locked python examples/run_followup_diagnostics.py \
  --acf-only --output data/reports/followup-diagnostics-acf
uv run --locked python examples/run_followup_diagnostics.py \
  --output data/reports/followup-diagnostics
uv run --locked python examples/summarize_followup_diagnostics.py \
  data/reports/followup-diagnostics \
  --output data/reports/followup-diagnostics-summary.json
```

Use fresh output directories; publication is atomic. Commit source/config
before forecast provenance. Only frozen development loaders are exposed;
there is no confirmation option. `config.json` contains a verified metadata
envelope; `provenance.json` records configuration, native split/cohort,
external split/source/reserve, implementation SHA-256s, commit, lockfile,
hardware and every generated file checksum. The summarizer verifies hashes
before reading results. Per-task `run.json` keeps all candidate configurations,
TRAIN-context scaler provenance and synchronized Chronos runtime/VRAM.
`*-windows.parquet` holds descriptive spread/bias/change metrics for each
window/feature; `*-leads.parquet` holds points, labels, origin and errors for
every lead. No scalar summary substitutes for these full results.

Verified real run: RTX 4070 Laptop 8 GiB, code commit
`54716e0` (full hash in local provenance), 19.33 s wall time,
1,663,272 KiB peak RSS; 240 native plus 48 seen Chronos calls,
505,413,632 bytes peak reserved VRAM. No fine-tuned model is evaluated here;
meaningful matched geometric fine-tuning follows in #41. Probabilistic
references and frozen independent inference follow in #43/#45. Old MVP
artifacts remain unchanged. Generated data/reports stay local and ignored.
