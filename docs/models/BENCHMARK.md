# Reproducible context × horizon benchmark

The model-independent Python API evaluates point and probabilistic adapters
on identical canonical context-only batches, with target labels supplied
separately. No new dependency, model factory or fine-tuning is introduced.
There is no benchmark CLI yet.

## Freeze before scoring

For every cell construct a `CellManifest` containing its `ForecastSpec`, full
split manifest, single-cell `WindowConfig`, partition, TRAIN-context scaler,
ordered model configurations and matching `model.artifact_hash` identities,
code commit, lockfile checksum, hardware, batch size and statistical policy.
Statistical artifacts identify their configurations; NLinear identifies its
complete weights/scaler/state; pretrained models identify revision/settings.

Construct a `GridManifest(cells=(...))` before evaluating any cell. It rejects
duplicate cells, mixed partitions/sources/features/model choices/policies and
a Bonferroni comparison family smaller than the complete grid. Scaling
metadata must match the cell and use only TRAIN trajectories. One cell may
have different trained weights/scaler, but not different unfrozen model choices.

```python
from pathlib import Path
from md_forecast.evaluation.benchmark import GridManifest, evaluate_grid
from md_forecast.evaluation.report import read_benchmark, write_benchmark

# cells, models_by_cell, pairs_by_cell are prepared with frozen settings.
# Every pairs iterable is iter_forecasts(..., with_targets=True).
frozen = GridManifest(cells=cells)
results = evaluate_grid(models_by_cell, pairs_by_cell, frozen)
report = write_benchmark(Path("data/processed/benchmark-run"), frozen, results)
assert read_benchmark(Path("data/processed/benchmark-run")) == report
print(report.artifact_id)
```

Every cell requires all frozen windows exactly once. The default selector is
`all`; the explicit development smoke selector `first-per-trajectory` chooses
only each trajectory's first legal window. Budget/empty partition failures
occur before inference. Missing, duplicate, forged, wrong-source or wrong-split
windows fail rather than silently changing the evaluation subset. Adapters
receive no targets. The evaluator detects only the canonical `forecast`
capability; it imports neither Chronos nor Torch. Quantile outputs live in
`data.predictions`, with backwards-compatible re-exports from the Chronos module.

## Metrics and aggregation

- MAE: absolute native-unit error.
- RMSE: root mean squared error within each trajectory (not mean absolute error).
- Scaled MAE: absolute error divided by frozen TRAIN-context standard deviation;
  constant-channel scale uses the existing scaler policy. Not MASE.
- Persistence ratio: macro MAE / macro persistence MAE; improvement is `1-ratio`.
  Zero persistence error yields null ratio/improvement, never an invented epsilon.
- Pinball: `max(q*(y-fq), (q-1)*(y-fq))`, without the optional factor of two used
  by some [quantile-score conventions](https://otexts.com/fpp3/distaccuracy.html).
- Coverage: inclusive `lower <= y <= upper`; width: native-unit `upper-lower`.
  Each interval retains both endpoint labels. Quantile crossing, missing
  endpoints, invalid shapes and nonfinite/overflowing values fail explicitly.

Point-only models have no fabricated probabilistic metrics. CRPS is deliberately
not implemented without a robust full-distribution representation/reference.

Metrics retain each lead step `1..H`; `step=0` averages the full requested
horizon. Reduce windows equally within trajectories, take RMSE's square root
there, then average trajectories equally within systems, systems equally within
dependent split groups, and independent groups equally in the aggregate. Macro
RMSE is consequently the mean of trajectory RMSEs, not a pooled-window RMSE.
No native units or unrelated features are averaged together. The report states
window/trajectory/system/group counts, with every identity in the prediction table.

## Frozen uncertainty/effect-size policy

The independent bootstrap unit is the split manifest's dependency group:
it may contain one trajectory or multiple dependent replicas/systems. Entire
group values are sampled together. The seed, resample count, confidence level,
minimum group count, minimum resolved tail samples and declared comparison
family are typed in `BenchmarkConfig`. No overlapping-window bootstrap exists.
Point/probabilistic aggregate metrics receive marginal percentile intervals.
Model-minus-persistence MAE differences receive paired group intervals with
Bonferroni tail probabilities over the entire predeclared model × observable ×
lead-step/cell family, including whole-horizon summaries. The absolute paired
MAE difference is the effect size; no p-values or post-hoc model selection.
These are approximate descriptive percentile-bootstrap intervals, not an
automatic significance/forecast-superiority decision. The procedure follows
the [paired percentile bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
using NumPy, without adding SciPy as a runtime dependency.

If independent groups are below `min_groups`, confidence bounds are null.
If `bootstrap_samples * tail_probability < min_tail_samples`, bounds are also
null: reporting empirical extremes as a well-resolved corrected interval would
be misleading. Comparisons explicitly distinguish `insufficient-groups`,
`insufficient-bootstrap-resolution` and `descriptive-bootstrap`. More groups
and/or an adequately budgeted bootstrap are required before inferential claims.
TRAIN reports are development diagnostics, never held-out success evidence.
Calibration acceptance tolerances and strongest-baseline selection must still
be frozen using an actual validation partition before scientific TEST evaluation.

## Persisted bundle and identity

`report.json` is a Pydantic/hash-verified complete-grid boundary. Each `cell-N`
directory contains `predictions.parquet`, `metrics.parquet`, `comparisons.parquet`,
`runtime.parquet` and native SVG horizon/calibration curves. Predictions preserve
source start, future target frame, feature ID, native target/point values and
every quantile with its labels. Specifications retain feature definitions/units
and sampling provenance. Horizon figures always include persistence and use
physical ps only when all relevant timestamps are verified/uniform; otherwise
the axis is explicitly frames. Calibration plots compare nominal interval
levels with empirical coverage; all per-lead values/bounds remain tabular.

`artifact_id` hashes the frozen manifests and actual scientific Parquet file
checksums. Repeat predictions/metrics produce the same ID. Timing and process
allocated/reserved VRAM are operational telemetry, excluded from scientific
identity; CPU VRAM is null (unmeasured), not a fabricated zero. Runtime covers
adapter inference/host conversion, not training or data acquisition. Peak GPU
metadata remains process-wide as documented by the adapter.

Publication stages the complete grid beside the destination, then renames
once. Existing output is refused; failures leave no promoted partial bundle.
Single-writer publication is the current ceiling. `read_benchmark` checks all
scientific file hashes. Data, predictions and generated figures stay ignored.

## Development verification specification

`configs/benchmarks/development.toml` freezes the development-only first-window
selector, C={20,40,60}, H={1,5,10,20}, 32-window batch cap, all four native
channels, seven quantile levels and 50/80/90% intervals. The models are the six
versioned [CPU baselines](BASELINES.md) plus pinned [Chronos-2](CHRONOS2.md).
NLinear fits only official TRAIN contexts/labels; scaling uses TRAIN contexts.
The real audited sample has 19 TRAIN groups, no VAL, and TEST 16PK, which must
not be loaded for this integration proof. The 1,000-resample budget resolves
marginal 95% intervals but not the 2,880-comparison corrected tails: corrected
intervals must be flagged unavailable. No physical-time, held-out-skill or
calibration-success claim is authorized by this smoke specification.

## Actual development evidence — 2026-10-02

Clean implementation revision `3ee469fabd29b4e2a6ad028ea6556259077b15bf` ran the
above 12 cells on the 19 official TRAIN systems. Each cell has 19 first windows,
19 trajectories/systems/independent groups: 228 cell-windows and seven models
(1,596 window/model forecasts), 57,456 scalar prediction rows and 946,560 metric
rows. Quantile/calibration outputs retain 50/80/90% intervals. Every horizon
figure includes persistence; unknown physical sampling stays labeled frames.

A guarded loader asserted TRAIN membership and rejected TEST 16PK. NLinear and
scalers fit only TRAIN data; no validation selection was possible or claimed.
All native point/quantile predictions were finite/noncrossing. Marginal group
intervals were produced; all multiplicity-corrected comparisons were explicitly
`insufficient-bootstrap-resolution`, as required by the predeclared pilot budget.
No conclusions about held-out accuracy or calibration success were drawn.

Two independent calls of `evaluate_grid` with the same frozen manifest/adapters
produced equal prediction, metric and comparison tables and identical report
IDs. Both persisted bundles passed full scientific-table checksum verification.
The input manifest, baseline configs, pretrained settings and all 12 fitted
NLinear states were saved as separate hash-addressed metadata artifacts under
ignored `data/processed/benchmark-inputs`. Retain these model inputs beside the
report; the generic evaluator intentionally does not reconstruct model objects.
Verified outputs are ignored `misato-benchmark-verified` and `misato-benchmark-repeat`.

- Manifest: `sha256:7d9153240d45df86c87bf4bc83e17d72512b6bfb09a83a9c62e2555c5aa9f358`.
- Report: `sha256:b315b1906b5d393a3be91ebb38d55e36f148906d39e4b0dadff57071ec82e78a`.
- Lockfile: `sha256:6114c9b8d23e5e8b2bf890dcedacfca4d5569bb9d1e6958a498d29d10d40681b`.

Workstation: i7-13700HX, RTX 4070 Laptop 8 GB, Python 3.14.8, Torch 2.14.1+cu130.
The complete fit/evaluate/publish/repeat/check run took 20.49 s, peak RSS
2,251,704 KiB and peak reserved VRAM 511,705,088 bytes (~488 MiB). These are
small-subset integration measurements, not full-data throughput estimates.
The local verification helper used the documented API/configs:

```sh
/usr/bin/time -v uv run --locked python /tmp/md-forecast-issue11-smoke.py \
  data/processed/misato-benchmark-verified data/processed/misato-benchmark-repeat
```

That helper is workstation-local, not a bundled CLI; use fresh output paths
on reruns. Synthetic numerical/reference and every-adapter tests provide the
portable CI reproduction without external data or a GPU. Full experiment
automation remains separately scoped.
