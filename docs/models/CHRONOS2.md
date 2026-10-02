# Chronos-2 zero-shot adapter

Install with `uv sync --locked --extra chronos`. The default CPU installation
and tests do not import Torch, download weights, or require CUDA. Inference
is synchronous; fine-tuning, covariates and a forecasting CLI are out of scope.

## Configuration and loading

`configs/models/chronos2.toml` explicitly selects `amazon/chronos-2`, revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`, `cuda:0`, `float32`, seed 42,
scalar-series batch budget 32, quantiles `(0.1, 0.5, 0.9)` and a 24 GiB allocator
ceiling. Package versions are `chronos-forecasting==2.3.2` and `torch==2.14.1`;
transitive versions and distribution hashes are in `uv.lock`.

The pinned [checkpoint configuration](https://huggingface.co/amazon/chronos-2/resolve/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json)
defines a native context limit of 8192 samples and prediction limit of 1024.
API behavior was inspected at upstream source revision
[`10afa9ebe016e514f9d7dc1aa873f66af57e116b`](https://github.com/amazon-science/chronos-forecasting/blob/10afa9ebe016e514f9d7dc1aa873f66af57e116b/src/chronos/chronos2/pipeline.py)
and confirmed against the installed 2.3.2 distribution. This inspection SHA
does not claim that the PyPI release was built from that source revision.
The [checkpoint](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)
has Apache-2.0 terms and a 477,930,472-byte safetensors file.

Loading fetches only `config.json` and `model.safetensors` into the caller's
explicit cache directory at the exact revision, then loads that local snapshot
with `local_files_only=True`, `trust_remote_code=False`, `use_safetensors=True`.
This prevents upstream adapter discovery from consulting a moving Hub branch.
Device/dtype and native quantile membership are checked after loading. Only
float32 and CUDA are admitted initially; no unavailable GPU falls back to CPU.

```python
import tomllib
from pathlib import Path
from md_forecast.models.chronos import Chronos2Adapter, ChronosConfig

settings = ChronosConfig.model_validate(
    tomllib.loads(Path("configs/models/chronos2.toml").read_text())
)
adapter = Chronos2Adapter(settings, cache_dir=Path("data/cache/chronos2"))
# batch: canonical ForecastBatch from iter_forecasts(..., with_targets=False)
result = adapter.forecast(batch)
quantiles = result.values  # read-only float64 (B,H,F,Q), native feature units
levels = result.quantile_levels
points = result.median  # q=0.5, not the predictive mean
```

## Canonical contract and measurements

`forecast` preserves ordered, unique quantile labels; unsupported levels are
rejected, not interpolated. Context-only `(B,F,C)` input jointly models channels
within each window. `cross_learning=False` keeps different trajectories/windows
independent. The scalar batch budget must fit one complete multivariate window.
No context is silently truncated and no horizon is heuristically unrolled.

Output `(B,H,F,Q)` retains exact `ForecastSpec`/`WindowIndex` identities:
`target_slice` identifies the future frame axis, with original sampling status
and available physical spans. No unpublished timestamps are invented.
`predict` is the explicit median-only compatibility view for `ForecastModel`
and the point evaluator. Probabilistic consumers must use `forecast`; full
probabilistic scoring is #11. No upstream objects enter schemas/evaluation.

`adapter.config` is a common `ModelConfig` with Chronos identity, seed and
`adapter_config_hash=metadata_hash(adapter.settings)`. Persist those settings
alongside the identity. Changes to revision/device/dtype/batching/quantiles
change the model hash without leaking backend fields into evaluation schemas.
Baseline configurations omit this inapplicable field, preserving existing
serialization. Statistical adapters explicitly reject Chronos identities.

`load_runtime` measures snapshot acquisition/loading. Result `runtime` measures
synchronized inference and host transfer. CUDA allocator peaks are reset before
each operation. Allocated/reserved peaks include weights and other allocations
in this process, not total board usage or incremental model-only memory. The
reserved peak is checked against the ceiling; OOM/backend failures raise
`ForecastError`, malformed/nonfinite output a canonical `DataContractError`.
Use a dedicated inference process: concurrent calls/unrelated Torch work would
share global peak statistics. Seeded calls restore CPU/selected-device RNG state.

## Real development acceptance — 2026-10-02

The audited native MISATO sample has 20 trajectories, 100 frames each: 19
official TRAIN, zero VAL, and TEST system 16PK. Select the first start-zero
window of each TRAIN trajectory in canonical order at `(C,H)=(20,5),(40,10)`.
For each cell run `ligand_rmsd` alone and all four native channels in registry
order. This predeclared smoke selector yields 76 window/mode forecasts, each
repeated once. No TEST data, target labels, preprocessing fitting or tuning.

Hardware: RTX 4070 Laptop GPU, 8,184,725,504 bytes visible VRAM, NVIDIA driver
580.173.02; Python 3.14.8, Torch `2.14.1+cu130`. Cached loading: 0.1421 s,
480,247,808 reserved bytes; initial download/load: 5.5869 s.

| C / H | F | Shape `(B,H,F,Q)` | First / repeat seconds | Peak allocated / reserved bytes |
| --- | --- | --- | --- | --- |
| 20 / 5 | 1 | `(19,5,1,3)` | 1.2601 / 0.0133 | 489,623,552 / 505,413,632 |
| 20 / 5 | 4 | `(19,5,4,3)` | 0.0738 / 0.0335 | 492,389,888 / 507,510,784 |
| 40 / 10 | 1 | `(19,10,1,3)` | 0.0589 / 0.0129 | 491,384,320 / 507,510,784 |
| 40 / 10 | 4 | `(19,10,4,3)` | 0.0363 / 0.0385 | 493,288,448 / 507,510,784 |

All outputs were finite, correctly aligned and bitwise identical on repeat on
this device/stack. This proves the single-device budget on a smaller 8 GB GPU,
not a physical 24 GB card. First-call kernel warm-up is included. These are
smoke measurements, not stable performance or accuracy/calibration claims.

Settings hash: `sha256:096f1b6638d1f7d72f89fd536b5e0d25be6cc947af295a0c0ed0904317879e08`.
Split hash: `sha256:aeae4a9c0d630463cfffc5eeca9ca5868c96bc4fa4fb9a9f64a10c3add537e64`.
Window hashes (20/5, 40/10):
`sha256:3a82025b1c51bee1fa809d878bd6efde0813d03ae69ae42e41dc444851dd18c7`,
`sha256:1fbf0b8bed7fd6d4c26f0181b869964afdc40a1688832a473ade6eb079d5f90b`.
Source checksum/units/feature definitions and timing limitations remain those
of the [MISATO adapter](../datasets/MISATO_ADAPTER.md) and
[development QC](../datasets/MISATO_QC.md). Data/checkpoints stay ignored.

Synthetic tests cover axes, variable uni/multivariate requests, quantile labels,
immutability, common identity/protocol, native limits, missing/misplaced GPU,
wrong dtype, missing dependencies, OOM, malformed output and budget failures.
Default CI requires neither GPU nor external data/network.
