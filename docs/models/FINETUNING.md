# Chronos-2 full fine-tuning

The synchronous API uses the installed `chronos-forecasting==2.3.2`
`Chronos2Dataset` and `Chronos2Trainer`, with locked Transformers 5.18.0.
No custom training loop, LoRA dependency, distributed platform or training CLI.
Full fine-tuning fits the measured single-device budget, so PEFT is unnecessary.
The [upstream fit implementation](https://github.com/amazon-science/chronos-forecasting/blob/10afa9ebe016e514f9d7dc1aa873f66af57e116b/src/chronos/chronos2/pipeline.py)
uses these same components, but `fit()` does not forward a resume checkpoint
to `Trainer.train`. This adapter instantiates those components directly.

## Frozen inputs and configuration

`FineTuneConfig` exposes steps, scalar-series batch size, learning rate,
Adam betas/epsilon, weight decay, clipping, warmup, gradient accumulation,
precision, determinism, checkpoint interval/retention and CPU input budgets.
Versioned settings are in `configs/models/chronos2-finetuning.toml`.
The optimizer is upstream AdamW with a linear learning-rate schedule;
all numerical hyperparameters are explicit. Validation loss (upstream quantile
training objective) selects the minimum-loss checkpoint, not scientific MAE
success or TEST performance. Equal evaluation/save intervals divide final steps.

`prepare_finetuning` receives a validated pre-window split, one frame-grid cell,
ordered features, canonical series loader, base `ChronosConfig`, training config,
task label, code/lock/hardware provenance. It checks nonempty TRAIN/VAL and
array/window budgets **before payload reads**, loads only those partitions and
hashes their actual float64 window values. Training rechecks that input hash.
The complete dataset registry retains source checksums, units and exclusions;
the split/spec preserve source identities and exact window configuration.

Every upstream input is one canonical `(F,C+H)` window. `min_past=C` leaves
exactly one legal origin, so upstream random training sampling chooses declared
windows with replacement, not shorter or off-grid contexts. VAL enumerates all
declared windows. Native Chronos scaling remains context-local. There are no
future-known covariates. A scalar batch must fit an integer number of complete
multivariate windows, avoiding upstream batch overshoot. Gradient accumulation
is configurable but unnecessary in the measured pilot (one optimizer update
per batch). Inputs are bounded eager arrays, not a full-corpus streaming claim.

Two task labels are supported: `official-validation` requires the official
partition; `development-train-holdout` requires grouped TRAIN/VAL assignments
over **originally official TRAIN records only**, with no TEST assignment.
Original source split labels are never rewritten. This explicit holdout was
authorized for development because the audited sample has no official VAL.
It is not the official scientific split or an untouched TEST result.

## Checkpoints and resume

`train_chronos(manifest, loader, output_dir=..., cache_dir=...)` returns
`FineTuneResult` only after all frozen steps complete. Optional
`stop_after_checkpoint` pauses at an intermediate save and returns `None`,
without a completed result. Resume explicitly passes
`resume_from_checkpoint=output_dir / "checkpoint-N"` from that same run.

Each checkpoint contains safetensors, config, optimizer, scheduler, RNG and
Trainer state plus atomic `provenance.json`. Provenance includes the full frozen
manifest, dataset/ordered feature-set/training-config hashes and checksums of
all saved state. `read_checkpoint` verifies file integrity and the declared step.
Resume rejects changed input values/config/split/code, foreign run directories,
terminal checkpoints and rewinding a completed run. Retention uses the upstream
policy with at least two slots (best and latest); a pruned checkpoint cannot be
resumed. Existing fresh-run outputs are never overwritten. Resume files must
be trusted, locally generated artifacts, not arbitrary downloaded optimizer/RNG
pickles. Hashes detect accidental corruption, not maliciously rewritten envelopes.

`ignore_data_skip=True` is intentional: the upstream dataset is an infinite,
random NumPy iterator, not a finite epoch sampler. HF restores its checkpoint
RNG before creating the resumed iterator instead of replaying consumed batches.
No dataloader workers or distributed execution are admitted; exactly one CUDA
device must be visible and match the requested device. No CPU fallback.
The [HF Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
describes checkpoint optimizer/scheduler restoration and best-model selection.

Use a dedicated worker process. HF full determinism sets global seeds,
deterministic algorithms and CUDA environment/backend flags; TF32 is disabled.
Bfloat16 autocast uses float32 model weights; unsupported bf16 fails explicitly.
Bitwise repeatability is verified on the recorded stack, not promised across
drivers/devices/library upgrades. CUDA allocator peaks cover this process,
not total board memory. Synchronized runtime includes training, VAL and checkpoint
I/O/hashing; throughput is completed optimizer steps per measured second
(remaining steps only on resume). OOM/budget/backend failures do not publish
a completed result, but valid prior checkpoints remain recoverable.

## Unchanged benchmark integration

`FineTunedChronos2Adapter(settings, checkpoint_dir=selected_checkpoint)` verifies
local state, then reuses the zero-shot forecast implementation and canonical
`QuantileForecast`/`ForecastModel` contracts. It rejects mismatched source,
features or split, supports native inference C/H limits and never exposes
upstream types to evaluation. Its common Chronos identity includes the fitted
model/config file hashes, selected step, frozen training manifest and inference
settings. Optimizer state, path-dependent Trainer state and wall time are not
the scientific forecast identity. Zero-shot behavior remains unchanged.

Pass persistence, all statistical baselines, TRAIN-fitted NLinear, zero-shot and
fine-tuned adapters together to the **unchanged** [#11 benchmark](BENCHMARK.md).
It identifies zero-shot and fine-tuned models by different common config/artifact
hashes, not a new evaluator special case. Persist the fitted NLinear state too.
For paired fine-minus-zero (or other baseline) effects, join existing `level=group`,
`metric=mae` rows on group/feature/step, subtract paired values and use existing
`group_interval(..., corrected=True)` with the frozen comparison family.
Do not bootstrap overlapping windows as independent observations.

## Reproduce the development training proof

Install `uv sync --locked --extra chronos --extra structural` and first acquire/
extract the audited sample as documented in the [MISATO adapter](../datasets/MISATO_ADAPTER.md).
No new dataset acquisition is performed by the verification example. From a
clean checkout, use fresh ignored paths:

```sh
uv run --locked python examples/verify_chronos_finetuning.py prepare data/processed/finetuning-manifest.json data/processed/ft-full
uv run --locked python examples/verify_chronos_finetuning.py train data/processed/finetuning-manifest.json data/processed/ft-full
uv run --locked python examples/verify_chronos_finetuning.py train data/processed/finetuning-manifest.json data/processed/ft-repeat
uv run --locked python examples/verify_chronos_finetuning.py pause data/processed/finetuning-manifest.json data/processed/ft-resume
uv run --locked python examples/verify_chronos_finetuning.py resume data/processed/finetuning-manifest.json data/processed/ft-resume
sha256sum data/processed/ft-full/checkpoint-8/model.safetensors data/processed/ft-repeat/checkpoint-8/model.safetensors data/processed/ft-resume/checkpoint-8/model.safetensors
```

`configs/benchmarks/finetuning-development.toml` freezes system-grouped seed-42
70/30/0 holdout, C40/H10, stride 10, all four native features and the uncertainty
policy before scoring. This gives 13 TRAIN and 6 VAL systems, 78 and 36 windows.
Eight optimizer steps, batch four scalar series (one multivariate window), bf16,
LR 1e-6, accumulation one, saves at steps 4/8, selection by VAL loss.
The inference allocator ceiling is 7 GiB on the available 8 GB GPU.

## Development effects and limitations

The pilot evaluates all eight models on the same 36 VAL windows from six
independent complexes, with 11,520 scalar prediction rows and 28,424 metric rows.
Both Chronos variants retain native 0.1/0.5/0.9 quantiles, 80% coverage, width
and pinball loss per lead. Repeat benchmark tables are identical. Scalers and
NLinear fit only the 13 development TRAIN systems; TEST 16PK is never loaded.
Source checksum remains `554d20ea0822949e1a5b0dc1826f50b87e71de5a7ed83ac1fb5a2e68c64547de`.
Physical timestamps remain unverified: all horizons are frames, not invented ps.

Whole-H10 macro MAE, in native units, from the development pilot:

| Model | Ligand RMSD (Å) | COM distance (Å) | Buried SASA (Å²) | Energy (kcal/mol) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.404688 | 0.344686 | 32.119130 | 3.398453 |
| Context mean | 0.352399 | 0.356607 | 27.356993 | 2.482257 |
| Linear extrapolation | 0.387351 | 0.389655 | 29.138762 | 2.843939 |
| AR | 0.342493 | 0.341736 | 26.642132 | 2.496239 |
| VAR | 0.334627 | 0.341041 | 27.005542 | 2.518393 |
| NLinear | 0.592586 | 0.501410 | 42.974209 | 4.243166 |
| Chronos zero-shot | 0.292750 | 0.323365 | 26.852469 | 2.528117 |
| Chronos fine-tuned | 0.292563 | 0.323643 | 26.838950 | 2.517107 |

Paired fine-minus-zero MAE effects with **uncorrected descriptive** 95% group
bootstrap intervals (1,000 resamples, seed 42):

| Feature | MAE difference | Marginal 95% interval |
| --- | ---: | --- |
| Ligand RMSD | -0.000187 Å | [-0.000903, 0.000596] Å |
| COM distance | +0.000278 Å | [-0.000957, 0.001784] Å |
| Buried SASA | -0.013518 Å² | [-0.064983, 0.055394] Å² |
| Energy | -0.011010 kcal/mol | [-0.023649, -0.001292] kcal/mol |

The frozen 616-comparison family covers persistence and fine-minus-alternative
comparisons across features/leads, including whole-horizon summaries. Its
corrected tails cannot be resolved by 1,000 resamples: **every corrected interval
is null, `insufficient-bootstrap-resolution`**. Six groups and VAL reuse for
checkpoint selection further limit interpretation. The apparently favorable
energy marginal interval is not confirmatory evidence. Eight updates prove the
training path, not sufficient domain adaptation. No reproducible superiority,
official TEST success, calibrated-success or full-corpus throughput claim.

## Verified artifacts — 2026-10-02

Clean implementation `e82856cab951092fb06240d797c261fb0624bb8a` ran the documented
example in separate processes: two uninterrupted runs and an intentional pause
at step 4 followed by resume to step 8. All three selected step 8 with identical
VAL loss `6.184868335723877` and bitwise-identical final weights. Every checkpoint
file passed checksum verification. HF's aggregate resumed `train_loss` is not
comparable to the uninterrupted aggregate because it accounts only for updates
in that invocation; resumed VAL loss and final weights agree exactly.

The first run measured 5.6192 s training/VAL/checkpoint time, 1.4237 optimizer
steps/s, 2,399,809,536 allocated and 2,642,411,520 reserved CUDA bytes (2.46 GiB).
Whole process wall time was 9.78 s, peak RSS 2,859,148 KiB, `/usr/bin/time -v`.
Repeat: 5.5838 s; resumed remaining four updates: 3.8270 s, 2,621,440,000
reserved bytes. Hardware: RTX 4070 Laptop 8 GB, i7-13700HX, driver 580.173.02,
Python 3.14.8, Torch 2.14.1+cu130. This proves a budget smaller than 24 GB,
not access to a physical 24 GB card. CUDA emitted a workspace-capacity warning;
it did not invalidate determinism or cause a budget failure on this stack.

- Training manifest: `sha256:671facac5f8227a792bc0c69ce60944bf1505f9addd7dc16be7ffe1c73879d64`.
- Dataset: `sha256:dc6a7c2e4c13687b79b87af40697e53fbdc1f7859153e3327898a09887edb098`.
- Split: `sha256:816e3422e0fa5719d6c4336dc1abe30ae1fb4127cb21981148f78a628caede05`.
- Ordered feature set: `sha256:1c53ed57aa5981ad134050fd7556402d0ff50cb414700eb70c3700e96c04597c`.
- Training config: `sha256:a53e0c3133d15241a6e11dd62d3b1c8938a0bfe348e5fa11334b881009d51c45`.
- Weights: `sha256:a5bbd648ce6abcbe4171043abef29aceb4203363d0adb4412b9fd30907d91355`.
- Lockfile: `sha256:6114c9b8d23e5e8b2bf890dcedacfca4d5569bb9d1e6958a498d29d10d40681b`.
- Benchmark grid: `sha256:ac45d9f82f074ac9a2882008de2bca262775a7d9e5914480587e7c7bed8e00c0`.
- Benchmark report: `sha256:46102dd3012feb8426308cf4322c0e142dd1ead4a8e263e61779c67272b68498`.

Ignored outputs: `data/processed/chronos12-final-manifest.json`, training runs
`chronos12-final-full`, `-repeat`, `-resume`, and equal verified reports
`chronos12-final-benchmark`, `chronos12-final-benchmark-repeat`. Selected model
provenance and fitted NLinear state remain beside those reports/runs; checkpoints
and data are not committed or redistributed. All scientific tables were bitwise
equal on repeated evaluation with the same frozen model inputs. The local helper
used the unchanged evaluator and group-metric API described above:

```sh
uv run --locked python /tmp/md-forecast-issue12-evaluate.py \
  data/processed/chronos12-final-manifest.json data/processed/chronos12-final-full \
  data/processed/chronos12-final-benchmark data/processed/chronos12-final-benchmark-repeat
```

The evaluation helper is workstation-local, not a bundled CLI; training
reproduction is the committed example. General experiment automation remains
separately scoped. A fresh CPU-only wheel installed outside the checkout
verified the real checkpoint and both report bundles without importing Torch
or Chronos. Synthetic tests cover typed configs, source/split leakage, canonical
inputs, budgets before reads, invalid/corrupt state, checkpoint callback/pause/
resume/selection, wrong revision/features, GPU/precision/OOM failures and identity.
