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
