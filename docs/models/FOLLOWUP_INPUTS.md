# Follow-up observed-input comparison

Issue [#42](https://github.com/fabiobove-dr/md-forecast/issues/42) tests target-only
versus joint admitted geometric histories on the native development cohort.
No new descriptor, feature extractor or model integration is introduced.
The three channels measure different summaries of the same interaction:
contact count tracks all close heavy-atom pairs, reference fraction tracks
retention of initial pairs, and minimum distance tracks the closest pair.
Their dependence makes incremental information plausible but far from assured.

## Frozen comparison

`configs/experiments/followup-inputs.json` declares C40/H10, stride 10, all three
scored targets, native 200 TRAIN / 60 VAL systems, LR 1e-5 from #41, seeds
42/43/44 and representative seed 42. All nine univariate manifests and all three
joint checkpoint identities are frozen before any new optimizer update or
input-condition scoring. Source feature definitions/order, initial contact
reference, raw Cartesian units and split identities are preserved.

Reuse the pinned zero-shot Chronos adapter, the existing geometric fine-tuning
adapter and existing AR/VAR implementations. The conditions are:

- Target-only: exactly the target history and its matching one-channel model.
- Joint: all three admitted histories in their original feature order.
- Shifted: circularly shift both auxiliary histories by 20 frames inside each
  same observed context, keeping the target history and window/group identities
  unchanged. Both auxiliaries share the shift; their mutual alignment remains.

The negative transformation uses only values already available at the forecast
origin. It preserves each auxiliary's value distribution and system identity,
but changes alignment with the target and introduces a circular boundary.
It is a sensitivity control, not a physically generated trajectory or a formal
randomization null. Fine-tuned shifted inference uses the existing naturally
joint-trained model; it does not train a separate model on artificial histories.
Labels are carried separately and never enter `observed_input` or an adapter.

## Comparable adaptation and conventional references

Train nine target-only trials with the same 512-update / 16-warmup / 128-checkpoint
schedule, precision, optimizer, deterministic seeds and four-window-per-update
budget as #41. Scalar upstream batch size is four for one channel and twelve
for three channels. Models have the same pretrained weights/parameter count.
All full trials see 2,048 random window draws; selected checkpoints can have
less exposure. Joint adaptation supervises three channels instead of one, so
fine-tuned differences combine observed-input and multitask-adaptation effects;
they do not isolate a mechanism of information transfer. Checkpoint selection
uses each trial's own minimum upstream VAL loss, not incomparable raw loss
numbers across feature sets. Chronos normalization is context-local; no held-out
scaler is fitted. Manifest input/feature-set/checkpoint hashes bind the complete
training provenance.

Simple lagged comparators are AR (target-only) and VAR (joint/shifted), each at
lags 1 and 3 with ridge 0.1 fixed before scoring. They fit observed histories
only and recursively forecast ten samples. AR and VAR have the same context
rows but different design widths; VAR has more coefficients. Ridge acts on
native-unit designs as in the existing baseline path. These controls supplement
the independently validation-selected compact/statistical references in #40.

## Pairing, selection and interpretation

All 54 target/model/input combinations export their complete future predictions,
truth, origin, signed/absolute/squared errors, window spread and per-lead errors.
Chronos exports 0.1/0.5/0.9 quantiles. Exact window/group/lead/truth/origin columns
must match across every condition; corrupt checkpoint/source identities fail
before evaluation. Counts are 360 windows per target/condition and 60 independent
native systems, not 3,600 independent future points.

Average errors over windows within a trajectory, then trajectories within a
system, then systems equally. Pair joint-minus-target and shifted-minus-target
MAE by the same systems for every model/seed and for the fine-tuned seed mean.
The prospective development bootstrap uses 2,000 resamples, seed 42 and marginal
95% intervals only. All contrasts remain exploratory on selection VAL; no
multiple-comparison superiority claim follows from these intervals.

Choose inputs separately for each target using mean equal-system MAE over
three fine-tuning seeds: joint is retained only if its mean is strictly below
both target-only and shifted; otherwise choose target-only, including ties.
The negative condition is never selected for deployment. Representative seed 42
is fixed rather than selected by favorable error. Calibration/confirmation and
unseen external outcomes remain unopened. Corrected independent comparisons
follow the protocol in #43/#45. Improvement would support these tested summaries
only, not causal allostery, atomistic reconstruction or replacement of MD.

## Reproduction

Prepare the [fresh native cohort](../datasets/FOLLOWUP_COHORT.md) and complete
[geometric adaptation](FOLLOWUP_FINETUNING.md) first. Commit code/config and use
fresh directories:

```sh
uv run --locked python examples/run_followup_inputs.py prepare \
  --root data/processed/followup-inputs-fresh
uv run --locked python examples/run_followup_inputs.py train \
  --root data/processed/followup-inputs-fresh --trial target0-s42
```

Repeat `train` for `target0-s43`, `target0-s44`, `target1-s42`, `target1-s43`,
`target1-s44`, `target2-s42`, `target2-s43`, `target2-s44`. Target indices retain
the admitted feature order: count, reference fraction, minimum distance.

```sh
uv run --locked python examples/run_followup_inputs.py score \
  --root data/processed/followup-inputs-fresh \
  --output data/reports/followup-inputs-validation-fresh
```

Existing model guards remain strict: a three-channel checkpoint cannot masquerade
as a one-channel checkpoint. Native cadence is unknown, so C40/H10 and the
control offset denote frames, not invented physical durations. Application
point-error tolerances remain undeclared and forecasts are unrated.


## Real development outcomes

All nine univariate trials completed 512 updates, with no failed candidate.
Every terminal window-count vector is exactly identical to the corresponding
joint trial of the same seed: 2,048 draws and 960/1,006/991 unique windows for
seeds 42/43/44, respectively. All 200 TRAIN systems are consumed at completion.
Selected steps are 256/128/128 for contacts, 128/256/256 for reference fraction,
and 128/128/256 for minimum distance. Selected unique-window counts range
413–693; selected checkpoints therefore have less exposure than terminal trials.
All checkpoint hashes/exposure snapshots and learning histories remain available.

Local full-trial wall times span 74.87–91.29 seconds; peak process RSS spans
2,812,712–2,936,616 KiB, and peak reserved CUDA memory is 2.461 GiB. Actual training
runtime excludes startup and spans 66.91–80.60 seconds. Matching all 54 evaluation
conditions takes 169.31 seconds / 2,121,032 KiB peak RSS. There are 108 prediction
Parquet files and 194,400 target future rows; independent evaluation size remains
60 systems. Generated artifacts are under `data/processed/followup-inputs42`,
`data/reports/followup-inputs42-training` and `data/reports/followup-inputs42-validation`.

All native-unit MAEs and both marginal paired contrasts follow. Difference signs
are candidate-minus-target-only; negative is lower error. Brackets are exploratory
system-bootstrap 95% intervals, with no corrected superiority interpretation.

| Observable | Model | Target-only MAE | Joint MAE | Shifted MAE | Joint difference [CI] | Shifted difference [CI] |
| --- | --- | ---: | ---: | ---: | --- | --- |
| protein_ligand_contact_count | zero-shot | 11.363331 | 11.231818 | 11.267338 | -0.131513 [-0.278617, 0.028693] | -0.095993 [-0.259049, 0.060321] |
| protein_ligand_contact_count | fine-42 | 11.106800 | 11.122296 | 11.118184 | 0.015496 [-0.061452, 0.099788] | 0.011383 [-0.057446, 0.080623] |
| protein_ligand_contact_count | fine-43 | 11.082037 | 10.983243 | 11.024705 | -0.098794 [-0.190436, -0.012508] | -0.057333 [-0.167453, 0.038751] |
| protein_ligand_contact_count | fine-44 | 11.069378 | 11.084151 | 11.187859 | 0.014773 [-0.117975, 0.157138] | 0.118481 [-0.025715, 0.266713] |
| protein_ligand_contact_count | lag1 | 11.737660 | 11.702501 | 11.751317 | -0.035160 [-0.104191, 0.028501] | 0.013656 [-0.014068, 0.043063] |
| protein_ligand_contact_count | lag3 | 11.509942 | 11.661056 | 11.612108 | 0.151114 [-0.039463, 0.354006] | 0.102166 [0.005440, 0.201500] |
| fraction_reference_contacts | zero-shot | 0.054712 | 0.054614 | 0.055631 | -0.000098 [-0.000859, 0.000740] | 0.000920 [0.000025, 0.001785] |
| fraction_reference_contacts | fine-42 | 0.053509 | 0.053923 | 0.054114 | 0.000415 [-0.000097, 0.000924] | 0.000605 [0.000032, 0.001176] |
| fraction_reference_contacts | fine-43 | 0.053473 | 0.053388 | 0.053564 | -0.000085 [-0.000568, 0.000401] | 0.000092 [-0.000368, 0.000532] |
| fraction_reference_contacts | fine-44 | 0.053759 | 0.054229 | 0.054482 | 0.000469 [-0.000167, 0.001113] | 0.000723 [-0.000012, 0.001437] |
| fraction_reference_contacts | lag1 | 0.063231 | 0.062828 | 0.063442 | -0.000404 [-0.001258, 0.000660] | 0.000210 [-0.000122, 0.000552] |
| fraction_reference_contacts | lag3 | 0.058435 | 0.059373 | 0.060086 | 0.000938 [-0.000733, 0.003471] | 0.001651 [0.000854, 0.002562] |
| minimum_protein_ligand_heavy_distance | zero-shot | 0.076149 | 0.075896 | 0.075925 | -0.000253 [-0.001102, 0.000612] | -0.000225 [-0.000853, 0.000456] |
| minimum_protein_ligand_heavy_distance | fine-42 | 0.074288 | 0.074969 | 0.074775 | 0.000682 [-0.000151, 0.001860] | 0.000487 [-0.000172, 0.001349] |
| minimum_protein_ligand_heavy_distance | fine-43 | 0.075247 | 0.075408 | 0.074990 | 0.000161 [-0.000540, 0.000884] | -0.000257 [-0.001063, 0.000415] |
| minimum_protein_ligand_heavy_distance | fine-44 | 0.075607 | 0.074958 | 0.075124 | -0.000649 [-0.001269, -0.000071] | -0.000484 [-0.001442, 0.000255] |
| minimum_protein_ligand_heavy_distance | lag1 | 0.075091 | 0.075368 | 0.075320 | 0.000277 [-0.000314, 0.001318] | 0.000230 [-0.000075, 0.000630] |
| minimum_protein_ligand_heavy_distance | lag3 | 0.075645 | 0.076116 | 0.076587 | 0.000471 [-0.000197, 0.001141] | 0.000943 [0.000398, 0.001506] |

The predeclared mean-seed rule selects **joint for contact count**, **target-only
for reference fraction** and **target-only for minimum distance**. Joint contact
mean MAE is 11.063230 versus target-only 11.086072, a tiny 0.206% difference;
its paired interval includes zero (−0.090483 to +0.050553 pairs). The fixed
representative seed 42 actually favors target-only contacts by 0.015496 pairs;
retain that unfavorable result instead of substituting seed 43. All three
fine-tuned mean-seed joint-minus-target intervals include zero. The shifted
minimum-distance mean is lower than joint, which also argues against a strong
added-input claim. The negative condition is never a deployable input choice.

Selection mean MAEs for target-only/joint/shifted are respectively:
contacts 11.086072 / 11.063230 / 11.110249 pairs;
reference fraction 0.053580 / 0.053847 / 0.054053;
minimum distance 0.075047 / 0.075112 / 0.074963 Å.
These are development input choices, not confirmed predictive coupling.
Quantiles, signed residuals, RMSE/bias, all ten future leads and spread diagnostics
are retained for report overlays rather than discarded after selection.

Frozen selection bundle SHA-256: `a979fad17edbaba7b4223f775727dddc3651536741e6528c1b241276a43f3b5f`.
It records the preparation plan hash, inference code/lockfile identities,
ordered input features and each selected training manifest, feature-set, input,
checkpoint and model-artifact hash. The native reference definition is bound
through feature-set/input hashes; context-local Chronos normalization has no
standalone fitted scaler artifact. No confirmation outcome selected these choices.
