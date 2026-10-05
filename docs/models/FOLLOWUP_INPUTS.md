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
