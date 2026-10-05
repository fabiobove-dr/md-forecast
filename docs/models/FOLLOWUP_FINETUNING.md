# Follow-up geometric fine-tuning

Issue [#41](https://github.com/fabiobove-dr/md-forecast/issues/41) replaces the
historical eight-update native integration pilot with matched adaptation on the
three [admitted common geometric channels](../datasets/FOLLOWUP_ADMISSION.md).
This is development selection evidence; it establishes neither independent
superiority nor application usefulness.

## Frozen budget and selection

`configs/experiments/followup-finetuning.json` fixes pinned Chronos-2, bfloat16
full training, 512 optimizer updates, four three-channel windows per update
(upstream scalar batch size 12), 16 warmup updates, checkpoints every 128 updates
and retention of two checkpoints. The pool is 200 TRAIN systems / 1,200 windows,
C40/H10 and stride 10; the same source supplies 60 VAL systems / 360 windows.
The two learning rates (1e-6, 1e-5), three seeds (42, 43, 44), stopping rule,
checkpoint rule and representative seed 42 were frozen in all six manifests
before updates. All trials complete 512 updates; there is no early stopping.

A roughly 19-second 128-update pilot established measured local feasibility.
The bounded 512-update budget supplies 2,048 random window draws, about 1.71
per available window. Sampling is with replacement: this is not a guaranteed
full epoch. A training-forward hook records actually consumed indices, excluding
prefetch, validation and unsuccessful forward calls. Hashed checkpoint exposure
survives resume. Older checkpoints lacking telemetry retain explicit unknown
prior optimizer steps rather than invented exposure counts.

Within each trial, minimum upstream validation quantile loss chooses the saved
checkpoint. That normalized upstream loss is not MAE in pairs/fraction/Å.
Across learning rates, selection averages equal-system native MAE divided by
TRAIN-context feature scales, equally over three features, four C20/40 × H5/10
cells and three seeds; ties choose the lower learning rate. Seed 42 is fixed as
the representative, so the luckiest seed is not published as the chosen model.
Inference settings are identical for zero-shot and all checkpoints.

Native calibration/confirmation and unseen external outcomes remain unopened.
Purpose-scoped cohort loaders reject those reads before opening files. All
predictions hash the exact trajectory/window IDs, observed contexts and future
labels and require identical hashes to zero-shot. Source data, checkpoints,
per-point errors/quantiles, learning curves and reports remain ignored locally.

## All candidates and resources

Every corrected trial completed 512 updates and consumed 2,048 draws from all
200 systems. Selected checkpoints can have substantially less exposure.
Selected window/system counts and completed unique windows are distinct below.
Runtime/RSS include process startup; VRAM is peak reserved CUDA memory.

| Trial | Selected step | Selected unique windows / systems | Completed unique windows | Normalized VAL MAE | Wall time | RSS KiB | VRAM GiB |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| lr1e-06-s42 | 256 | 693 / 198 | 960 | 0.298946880 | 1:04.71 | 3039880 | 2.473 |
| lr1e-06-s43 | 256 | 688 / 195 | 1006 | 0.298530034 | 1:22.47 | 2813816 | 2.492 |
| lr1e-06-s44 | 256 | 690 / 198 | 991 | 0.299047517 | 1:25.02 | 2848832 | 2.492 |
| lr1e-05-s42 | 128 | 417 / 185 | 960 | 0.298783881 | 1:17.33 | 2906844 | 2.492 |
| lr1e-05-s43 | 256 | 688 / 195 | 1006 | 0.297581385 | 1:28.14 | 2913408 | 2.492 |
| lr1e-05-s44 | 512 | 991 / 200 | 991 | 0.298531118 | 1:24.73 | 2888280 | 2.492 |

The first 1e-6/42 attempt paused at 128, resumed and completed training, but
summary publication failed after checkpoint retention pruned the initial resume
directory. Preserve that failed attempt under `followup-finetuning41`; do not
count it as a completed published run. The fix captures the trusted initial step
before training. A fresh six-manifest plan under `followup-finetuning41-verified`
repeated the pause/resume successfully; terminal model weights are byte-identical
to the preserved failed attempt. The corrected first trial's resumed runtime
excludes its separate 27.08-second / 2,661,332-KiB pause process. Its checkpoint
telemetry and learning history include the initial 128 updates. No other planned
candidate failed. Synthetic regression tests cover this retention/resume case.

The learning-rate means (seed SD, ddof=1) are 0.298841477 (0.000274371) for 1e-6
and 0.298298795 (0.000634019) for 1e-5. Zero-shot is 0.302696578. The selected
rate's mean improves this descriptive score by 1.45%; its representative seed
improves by 1.29%. These seeds measure training variability, not independent
population uncertainty. All checkpoint losses at 128/256/384/512 and interval
training losses are retained in `selection.json` learning curves.

The selected representative is **1e-5 / seed 42 / checkpoint 128**, with 512
consumed draws, **417 unique windows / 185 systems**. The full trial ran 512
updates and saw all 200 systems; do not assign that terminal exposure to the
selected 128-update model.

Checkpoint identity: `sha256:bb083e2998f4df03a4be1dee7d6ec6713f09fead59af29484cbeb236cfe776e6`.
Model artifact identity: `sha256:cb20a525d9674ba5e39962f1ebd8f6360825c47fe34e8e3e0efe903c1aae794e`.
Each checkpoint verifies its complete model/trainer/state/exposure file hashes;
training manifests record code commit, lockfile, hardware, input/scaler hashes,
optimizer configuration, source split and feature identities. No native RMSD
checkpoint is reused as a geometric model.

## Matched real validation results

MAE and RMSE retain native units: contact pairs, reference fraction 0–1 and
minimum distance Å. RMSE takes the root of each trajectory’s mean-squared error
before averaging trajectories within systems and then systems equally. Spread is forecast temporal standard deviation divided
by future temporal standard deviation, averaged over defined window ratios.
Undefined constant-truth ratios are exported as null and excluded explicitly.
Each model has 1,680 windows across the four cells and 60 independent systems.
Window overlap does not increase the independent system count.

| Cell | Observable | Zero MAE | Fine MAE | Zero RMSE | Fine RMSE | Zero spread ratio | Fine spread ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c20-h5 | protein_ligand_contact_count | 11.046122 | 10.882934 | 13.859332 | 13.642184 | 0.1006 | 0.0401 |
| c20-h5 | fraction_reference_contacts | 0.053909 | 0.053958 | 0.069406 | 0.069046 | 0.1235 | 0.0524 |
| c20-h5 | minimum_protein_ligand_heavy_distance | 0.073938 | 0.072073 | 0.093074 | 0.090705 | 0.1138 | 0.0375 |
| c20-h10 | protein_ligand_contact_count | 11.317995 | 11.199989 | 14.231968 | 14.077365 | 0.1033 | 0.0501 |
| c20-h10 | fraction_reference_contacts | 0.057281 | 0.056658 | 0.074235 | 0.073228 | 0.1269 | 0.0611 |
| c20-h10 | minimum_protein_ligand_heavy_distance | 0.075314 | 0.074112 | 0.095676 | 0.094225 | 0.0979 | 0.0449 |
| c40-h5 | protein_ligand_contact_count | 11.161488 | 10.984325 | 13.842552 | 13.648943 | 0.1094 | 0.0396 |
| c40-h5 | fraction_reference_contacts | 0.052405 | 0.052302 | 0.066408 | 0.066119 | 0.1215 | 0.0601 |
| c40-h5 | minimum_protein_ligand_heavy_distance | 0.074264 | 0.072850 | 0.092979 | 0.091447 | 0.1102 | 0.0338 |
| c40-h10 | protein_ligand_contact_count | 11.231818 | 11.122296 | 14.057884 | 13.923444 | 0.0962 | 0.0507 |
| c40-h10 | fraction_reference_contacts | 0.054614 | 0.053923 | 0.069401 | 0.068253 | 0.1164 | 0.0693 |
| c40-h10 | minimum_protein_ligand_heavy_distance | 0.075896 | 0.074969 | 0.095885 | 0.095089 | 0.0847 | 0.0399 |

Fine-tuning makes median curves even flatter (roughly 3–7% of future variability)
while slightly reducing aggregate error. Reference-fraction C20/H5 worsens
slightly; retain it. The [selected compact comparator](FOLLOWUP_BASELINES.md)
remains competitive: its C40/H5 contacts MAE is 10.96692 versus fine-tuned
10.98433. C40/H10 minimum-distance context-mean MAE is 0.074955 versus fine-tuned
0.074969. A visually wiggly curve is not an accuracy objective, and these
mixed selection-set differences are not a confirmed fine-tuning advantage.

The per-point Parquet exports include prediction, truth, signed/absolute/squared
error, 0.1/0.5/0.9 quantiles, native pinball losses and 80% interval coverage/width.
Grouped uncertainty and probabilistic-reference comparisons follow in #43;
application validity remains unrated until meaningful unit-specific tolerances
are declared. Unknown MISATO physical cadence prevents physical-time claims.

## Reproduction and repeat audit

Prepare [the native development cohort](../datasets/FOLLOWUP_COHORT.md) and
[baseline selection](FOLLOWUP_BASELINES.md) first. Commit source/config before
freezing. Use fresh output directories; every candidate must finish before
hyperparameter selection can publish.

```sh
uv run --locked python examples/finetune_followup_geometry.py prepare \
  --root data/processed/followup-finetuning-fresh
uv run --locked python examples/finetune_followup_geometry.py train \
  --root data/processed/followup-finetuning-fresh --trial lr1e-06-s42 --pause-at 128
uv run --locked python examples/finetune_followup_geometry.py train \
  --root data/processed/followup-finetuning-fresh --trial lr1e-06-s42 --resume 128
```

Run the same `train` command without pause/resume for `lr1e-06-s43`,
`lr1e-06-s44`, `lr1e-05-s42`, `lr1e-05-s43` and `lr1e-05-s44`, then:

```sh
uv run --locked python examples/evaluate_followup_finetuning.py \
  --root data/processed/followup-finetuning-fresh \
  --baselines data/processed/followup-baselines40-expanded \
  --output data/reports/followup-finetuning-validation-fresh
```

Real evidence is under `data/processed/followup-finetuning41-verified`,
`data/reports/followup-finetuning41-training` and
`data/reports/followup-finetuning41-validation`. Matched evaluation took
100.32 seconds / 2,104,300 KiB peak RSS. A fresh repeated evaluation selected the
same checkpoint and produced **56 byte-identical prediction/error Parquet files**.
Selection means are identical; seed SD differs by less than 1e-14 from floating
reduction order. The audit records this tolerance instead of claiming exact
JSON equality. This confirms deterministic saved predictions, not independent
scientific replication or a positive result.
