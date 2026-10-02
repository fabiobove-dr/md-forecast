# CPU baselines

All six adapters implement `ForecastModel.predict(ForecastBatch)`. Contexts
are detached, read-only float64 arrays `(batch, context, features)`; predictions
are finite float64 `(batch, horizon, features)` in the same ordered native units.
`ForecastSpec` carries dataset/feature definitions and split/window hashes.
Future labels are separate: `iter_forecasts(..., with_targets=False)` never
reads them. A caller's canonical loader may independently perform whole-file
publication QC; that is not fitting or future-dependent preprocessing.

## Algorithms and configuration

`ModelConfig` requires an explicit nonnegative seed and centralized `ModelId`.
`lags` is required only for AR/VAR; `ridge` defaults to zero and must be
nonnegative. Persistence, mean, and linear trend prohibit ridge regularization.
The algorithms are deterministic; seeds are recorded, not used to introduce
randomness. Numerical reproducibility across BLAS/platform revisions is not
promised.

| Model | Behavior |
| --- | --- |
| Persistence | Repeat the last observed value. |
| Context mean | Repeat each channel's entire observed-context mean. |
| Linear extrapolation | Fit an intercept and trend to every context sample, independently per channel; requires at least two samples. |
| AR | Fit lagged least squares independently per channel in each context; recurse using predictions only. |
| VAR | Fit all channels jointly in each context; recurse using predictions only. |
| NLinear | Fit a separate context-to-horizon affine map per channel on all training windows, after subtracting the last context level; add that level back at prediction. |

AR/VAR require `C - p >= p * F + 1` lagged observations for `p` lags
(`F=1` for independent AR). Unsupported cells fail explicitly, without silent
persistence fallback. Intercepts are not ridge-penalized. Singular designs use
the [NumPy least-squares minimum-norm solution](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
There is no stationarity constraint or clipping; numerical overflow fails closed.

NLinear follows the last-level normalization of
[LTSF-Linear](https://arxiv.org/abs/2205.13504), using its
[individual-channel architecture](https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/models/NLinear.py).
This implementation solves a convex ridge least-squares problem with NumPy,
not the original optimizer/training recipe. It does not reproduce paper results.
No Torch, GPU dependency, model zoo, or training CLI is added.

## Fitting and validation

Load the versioned illustrative configuration with the standard library:

```python
import tomllib
from pathlib import Path

from md_forecast.models.base import ModelConfig
from md_forecast.models.learned import TrainingConfig

settings = tomllib.loads(Path("configs/models/baselines.toml").read_text())
configs = tuple(ModelConfig.model_validate(item) for item in settings["models"])
training = TrainingConfig.model_validate(settings["training"])
```

These settings are examples, not statistically selected parameters. Wheels do
not bundle repository configuration. Supply an audited `SplitManifest`, a
single explicit frame-grid `WindowConfig`, ordered feature IDs, and the same
`SeriesLoader` used by [data preparation](../datasets/PREPARATION.md):

```python
from md_forecast.models.learned import fit_nlinear, select_nlinear

# A predeclared setting; this does not select using training error.
model = fit_nlinear(split, grid, features, loader, configs[-1], training)

# For an actual selection, supply predeclared, unique NLinear candidates.
model, selection = select_nlinear(split, grid, features, loader, candidates, training)
```

Fitting loads TRAIN only. The existing training-context scaler fits the union
of observed training context regions, not target-only frames; training future
labels are used for supervised weights only. The scaler is frozen for prediction
and validation. Last-level subtraction is explicitly context-local.
Selection loads VALIDATION only after fitting and never TEST. Empty validation
fails before any source load; training or test is never substituted.

Validation candidates are ranked by per-feature MAE divided by the TRAIN
standard deviation, averaged equally over features and independent groups.
This avoids adding incompatible physical units. Constant training channels
use the scaler's documented unit scale. Ties retain candidate order.
Native-unit errors and the preprocessing hash remain in the selection artifact.

`TrainingConfig` requires seed, batch size, maximum windows, and an estimated
design-array byte budget. Fitting includes every training window or fails;
it never silently subsamples. The byte estimate is not a total-process RSS
limit: source tables and LAPACK workspaces also consume memory. Check resources
before larger runs. Each learned model handles one frame-grid cell.

## Shared evaluation and persistence

```python
from md_forecast.core.constants import Split
from md_forecast.data.forecast import iter_forecasts
from md_forecast.evaluation.baselines import evaluate_baselines
from md_forecast.models.baselines import StatisticalBaseline

models = (*tuple(StatisticalBaseline(config) for config in configs[:-1]), model)
pairs = iter_forecasts(
    split,
    grid,
    Split.VALIDATION,
    features,
    loader,
    batch_size=training.batch_size,
    with_targets=True,
)
evaluation = evaluate_baselines(models, pairs)
```

The evaluator requires exactly one persistence reference, identical windows,
one grid cell/partition, and unique model configurations. It averages absolute
error over horizon samples, then windows within each trajectory, then equally
over trajectories within each dependent group. Each feature retains its native
unit. Windows are not treated as independent replicates.

This is the small adapter integration evaluator required by issue #9, **not**
the complete scientific benchmark harness: per-horizon metrics, baseline
selection across statistical models, ratios, paired confidence intervals and
probabilistic calibration belong to the later evaluation issue.

Persist `model.state`, `selection`, and `evaluation` with
`write_metadata(path, object)` and load them with `read_metadata` using
`NLinearState`, `NLinearSelection`, or `BaselineEvaluation`. Construct
`NLinearModel(loaded_state)` for inference. Shape, scope, identity and content
hashes are checked, but hashes attest integrity, not honest provenance.
Record code commit, lockfile checksum, hardware and runtime separately for
published experiments as required by the scientific contract.

## Evidence

Synthetic regression tests assert exact deterministic forecasts, genuine VAR
cross-channel behavior, train-only fitting, validation-only selection, target
isolation, serialization, resource limits and overflow handling. All six
adapters run through the common evaluator on the same tiny validation fixture.
This demonstrates implementation correctness, not molecular forecast skill.

A local smoke test on the previously audited native MISATO sample also runs
all six adapters on TRAIN only: 19 trajectories, four native features,
`C=20, H=5, stride=1`, 1,444 windows and 1,805 scaler frames. Repeated fitting
and saved/reloaded predictions agree. The sample has no official validation
trajectories, so selection correctly fails before loading values. The official
test trajectory is never loaded. This checks API/data compatibility only;
no held-out result, physical-time horizon, or superiority claim is made.
