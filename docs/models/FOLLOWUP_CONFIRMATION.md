# Independent geometric confirmation

Issue [#45](https://github.com/fabiobove-dr/md-forecast/issues/45) tests the
choices developed in #37–44 on reserved outcomes. Engineering completion is
separate from scientific success. No choice is changed after confirmation.

## Frozen choices

The portable plan is
[`data/manifests/followup-confirmation45.json`](https://github.com/fabiobove-dr/md-forecast/blob/main/data/manifests/followup-confirmation45.json),
metadata hash
`sha256:f08d48b26b0994069ddcc1a5fb96a5b8977918f75f0b455b2b132767ad0694bc`.
It was committed as `4554843` before any reserved XTC decoding or native
confirmation coordinate extraction. The source code selecting these choices
was `8e1f578`; experiment runners record their own later committed code.
The original frozen plan remains unchanged when QC exclusions reduce counts.

Every source/checkpoint/configuration in `source_files` is SHA-256 verified
before reserved preparation or scoring. The plan records:

- three equivalent raw dry-system geometric observables: heavy atom-pair
  contacts within 4.5 Å, fraction of first-frame contacts retained, and minimum
  heavy receptor–ligand distance in Å;
- the complete C20/40 × H5/10 frame grid, stride 10, with **C40/H10 primary**;
- fixed representative seed 42, the validation-selected checkpoints and
  statistical/compact models, joint observed inputs for contacts and target-only
  inputs for fraction/distance;
- native TRAIN-only preprocessing, compact coefficients and residual quantiles;
  no new fitting, recalibration or checkpoint/input selection on confirmation;
- 10% worthwhile MAE reduction, nominal 80% intervals, coverage tolerance
  75–85%, quantile-loss and width comparators;
- 60,000 bootstrap draws, seed 42, 95% confidence, independent complex groups,
  Bonferroni family 126, minimum 20 groups and 10 expected corrected-tail draws.

The primary family comprises 36 point-MAE contrasts, 36 worthwhile-reduction
contrasts, 24 mean-three-quantile pinball contrasts, 12 width contrasts,
12 absolute coverage intervals and 6 fine-minus-zero MAE contrasts, across
native and untouched external tasks. The six-system replica task cannot meet
the independent-group requirement and cannot establish corrected superiority.
Other cells and individual leads remain descriptive. Resampling windows or
counting replicas as independent complexes is prohibited.

A useful minimum result requires **all** frozen conditions: corrected MAE and
10%-reduction bounds below zero against persistence and the selected statistic,
corrected quantile-loss gains against both probabilistic references, corrected
coverage wholly in 75–85%, and corrected width upper bound at most zero versus
the statistic. A stronger result additionally requires the 10% gain against the
compact supervised model. Missing intervals never pass a criterion.

## Separate tasks and QC

Native confirmation reserved 100 unseen MISATO complexes. Existing geometric
QC excludes `5SZC`: there are no frame-0 reference contacts, so the admitted
fraction is undefined. The frozen selection is retained; **99** groups and
9,900 frames are scored. No substitute ID is selected. The 132,841,014,019-byte
source was reverified against its official MD5 before extraction; the complete
preparation took 8:14.76 and its peak RSS is recorded in the cost table below.

The original external reserve of 50 complexes was prospectively extended to
80 using the next 30 eligible metadata-ranked IDs, preserving the six
seen-system IDs and original 50. The extension occurred before requesting any
of these confirmation trajectories; the reserve hash is
`sha256:28a5b784ba62ec0ed7786fe4543cffebb0670245c59a8d1f8278c930aece8e9a`.
Selection is independent of QC, geometry and model errors. Frozen static
selection rules may reject a complex; failures remain recorded without ID
replacement. Any replica failure excludes its whole complex from the matched
three-observable comparison. Acquisition and preparation retain static and
geometric rejection reasons separately.

Static admission rejects 12/80 selected external complexes: 1EX8, 1GUI,
1SQQ, 2BZ8, 2JG8, 2VPG, 2WIJ, 2WOQ, 4ORY, 5MB1, 6EE3 and 6G46.
Their reviewed selections are ambiguous or contain no single protein–ligand
interaction matching the admitted final segment. All raw failure artifacts
are checksum-pinned in `followup-external80-rejections.json`; no XTC outcomes
were used for this selection. Subsequent frozen geometric QC rejects 1TJP
and 3DXJ for incompatible XTC frame counts. **66 complexes / 660 replicas /
33,000 frames** remain, below the prospective 80-system precision target.
The original 80 IDs are retained and exclusions are disclosed, without
replacement or changing the corrected family.

The seen-system task reads only replica 10 of each of six previously examined
MDbind systems. It is **seen by development analysis**, while the models,
compact scaler and residual distributions remain fitted on MISATO TRAIN.
There is no undisclosed fitting on the six external systems.

Both MDbind tasks use actual 200 ps spacing: the 20/40 observed samples span
3.8/7.8 ns from first to last context sample; H5/H10 extend 1/2 ns beyond
the last observed value. MISATO's physical sample cadence is unresolved; its axes
remain frames. Exact PDB disjointness and protein-sequence overlap checks do
not establish unseen-target/chemotype or pretraining independence.

Explicit `evaluation_spec` bindings validate ordered feature definitions,
units and feature-set version while preserving the fitted state/checkpoint and
its original provenance hash. Target dataset/version/split identities remain
true to the target task. Compact and residual bindings additionally require
the original C/H; Chronos permits the predeclared grid. Default fitted adapters
still reject foreign domains. No batch identity or fitted scaler is relabelled.

## Regeneration

Acquire MISATO and reviewed frozen development artifacts using the earlier
[cohort](../datasets/FOLLOWUP_COHORT.md),
[baseline](FOLLOWUP_BASELINES.md), [fine-tuning](FOLLOWUP_FINETUNING.md),
[input](FOLLOWUP_INPUTS.md), and [probability](FOLLOWUP_PROBABILITY.md) workflows.
Install the existing locked optional model/structural dependencies. All
commands below run from the repository root; outputs must be fresh directories.

```sh
uv run --locked python examples/freeze_followup_confirmation.py
# The committed plan already exists; this command refuses to overwrite it.
uv run --locked python examples/prepare_followup_cohort.py prepare \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --output data/processed/followup-native-confirmation45
uv run --locked python examples/acquire_followup_confirmation.py
uv run --locked python examples/prepare_mdbind_common.py \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --reserve configs/datasets/followup-external-reserve80.json \
  --source configs/datasets/followup-external80-source.json \
  --source-qc configs/datasets/followup-external80-source.qc.json \
  --raw data/external/followup-external80-raw \
  --output data/processed/followup-external-confirmation45
uv run --locked python examples/prepare_mdbind_common.py \
  --purpose confirmation \
  --confirmation-plan data/manifests/followup-confirmation45.json \
  --reserve configs/datasets/followup-external-reserve.json \
  --source configs/datasets/followup-seen-source.json \
  --raw data/external/followup-seen-raw \
  --output data/processed/followup-seen-confirmation45
uv run --locked python examples/run_followup_confirmation.py --task native \
  --input data/processed/followup-native-confirmation45 \
  --output data/reports/followup-confirmation45-native
uv run --locked python examples/run_followup_confirmation.py --task external \
  --input data/processed/followup-external-confirmation45 \
  --output data/reports/followup-confirmation45-external
uv run --locked python examples/run_followup_confirmation.py --task replica \
  --input data/processed/followup-seen-confirmation45 \
  --output data/reports/followup-confirmation45-replica
uv run --locked python examples/summarize_followup_confirmation.py \
  --output data/reports/followup-confirmation45-inference.json
uv run --locked md-forecast report-forecast \
  --config configs/experiments/performance-overview.json \
  --output data/reports/offline-overview45/index.html
```

Saved bundles include every matched forecast window, lead, truth, point error,
median spread, source group and quantile. Compact point forecasts have no
invented probabilistic bands. Each summary pins every file, the full frozen
plan and target split, model/checkpoint identities, genuine evaluation spec,
code commit, lockfile hash, QC deviations and per-batch runtime/VRAM. The
separate inference JSON verifies all bundle hashes and leaves bundles intact.

The report embeds the actual curves, residuals, baseline ratios, all cells,
corrected decisions, native-unit comparison plots and the observed molecular
reference. Its per-window tolerance ratings remain user-controlled
retrospective error checks, independent of the frozen scientific decision.
An unset tolerance remains UNRATED; scalar predictions do not imply valid
atomistic trajectories.

## Failed probability forecasts

The initial external scorer aborted on crossed fine-tuned quantiles. A target-
only check reproduced five contact-count crossings in 4NNI at C20/H10,
plus five auxiliary-channel crossings; these are actual model outputs, not
assumed valid intervals. The scorer now evaluates only the declared target
and preserves **every median and raw quantile**, tagging exact crossing points.
No sorting, clipping, checkpoint change, data replacement or point-error
exclusion occurs. An affected target/cell has unavailable probabilistic
metrics rather than metrics averaged over the remaining valid points.

All primary comparisons remain in the original family, including unavailable
intervals. Such a cell cannot pass useful-uncertainty criteria. The report
shows the failed quantile values, suppresses affected-window shaded bands,
and exports probability-invalid flags separately from application tolerance
ratings. Initial failure logs and the reproduced crossing audit remain local
ignored report artifacts. This handling is an experiment reporting correction;
it does not repair or retune the model after confirmation.

## Measured results

**Practical useful skill is not established in any of the three tasks.** All
primary useful-minimum/stronger decisions fail at least one frozen criterion.
This is a completed negative/inconclusive benchmark, not evidence that MD
forecasting is intrinsically impossible.

There is smaller measurable point skill: native contact count improves for
zero-shot and fine-tuned Chronos against both references; fine-tuned retained
fraction improves in native and external tasks. External fine-tuned contacts
also improve point error, but their probability forecasts are invalid. These
corrected positive effects must not be conflated with the stronger frozen 10%
practical-usefulness threshold. The original contract's less demanding native
minimum is supported by zero-shot contacts: corrected point gains and calibrated
80% intervals, with width/pinball reported. The follow-up useful criterion is
stricter and does not pass; the stronger unseen-complex/compact claim remains
unsupported.

Fine-tuning improves native retained-fraction MAE over zero-shot: corrected
fine-minus-zero difference −0.00173202, CI [−0.00285515, −0.000629245]. Contact
and minimum-distance fine-minus-zero intervals include zero. Fine-tuned native
retained-fraction MAE 0.0561941 means **5.62 percentage points of absolute error**;
it is not 94.38% accuracy. Count MAE measures atom-pair contacts, not residues.
External minimum-distance MAE worsens from zero-shot 0.195011 Å to fine-tuned
0.252547 Å. Do not select a different checkpoint after this result.

Native fine-tuned future median spread is only 5.02%, 6.92% and 4.49% of actual
future spread (contacts/fraction/distance). Near-flat medians may estimate the
conditional level without resolving stochastic fluctuation timing. Spread is
a separate diagnostic, not an accuracy rate or proof of impossibility. Added
input benefit was inconclusive on development; physical cadence for native MD,
limited target history, short external horizons and model fit all limit claims.
The external 66 groups fall below the planned 80-group precision target; six
seen systems support descriptive replica evidence only.

### Primary point errors: C40/H10

MAE is averaged within trajectory, then within complex, then equally across
complexes. RMSE is rooted within each trajectory before this aggregation.
All models use identical windows; overlapping windows are never independent
bootstrap units.

| Task | Observable (unit) | Persistence | Statistic | Compact | Zero-shot | Fine-tuned |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| native | contacts (count) | 14.5165 | 11.7562 | 11.7371 | 11.4757 | 11.4681 |
| native | retained fraction (fraction) | 0.0686483 | 0.0594785 | 0.0587526 | 0.0579261 | 0.0561941 |
| native | minimum distance (Å) | 0.103074 | 0.0822401 | 0.0822135 | 0.0822814 | 0.0815883 |
| external | contacts (count) | 14.6827 | 12.7082 | 12.3151 | 12.0084 | 11.7946 |
| external | retained fraction (fraction) | 0.0581775 | 0.054514 | 0.0517339 | 0.052511 | 0.049807 |
| external | minimum distance (Å) | 0.213001 | 0.227628 | 0.357173 | 0.195011 | 0.252547 |
| replica | contacts (count) | 14.2 | 12.5073 | 13.0202 | 14.2465 | 13.6152 |
| replica | retained fraction (fraction) | 0.107037 | 0.0966872 | 0.0948149 | 0.0991933 | 0.0923707 |
| replica | minimum distance (Å) | 0.105386 | 0.0796485 | 0.0828165 | 0.0764754 | 0.0787626 |

### Corrected primary effects and uncertainty

All intervals below use the frozen 126-comparison family, 60,000 complex-level
bootstrap draws, 95% family confidence and seed 42. Negative MAE differences
favor the foundation model; zero inside an interval means superiority remains
unresolved. Coverage must have its *entire* corrected interval in 75–85%.
The bootstrap implementation labels available intervals `descriptive-bootstrap`;
`corrected: true` and family 126 distinguish these primary adjusted intervals
from the unadjusted descriptive intervals used elsewhere. Six-group corrected
intervals are deliberately unavailable. INVALID probability contrasts remain
in the original family and cannot pass.

| Task | Observable | Model | MAE minus statistic (CI) | ≥10% vs both | 80% coverage (CI) | Pinball vs both | Width vs statistic | Useful / stronger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | contacts | zero-shot | -0.280492 [-0.570558, -0.0390275] | False | 80.22% [77.9798, 82.4074] | True | False | False / False |
| native | contacts | fine-tuned | -0.28804 [-0.57946, -0.0185805] | False | 75.19% [72.9125, 77.3585] | True | True | False / False |
| native | retained fraction | zero-shot | -0.00155238 [-0.00423462, 0.000597644] | False | 82.41% [80.0152, 84.7844] | False | False | False / False |
| native | retained fraction | fine-tuned | -0.0032844 [-0.00582966, -0.00129416] | False | 74.70% [72.3537, 77.0891] | True | True | False / False |
| native | minimum distance | zero-shot | 4.12819e-05 [-0.00214058, 0.00197483] | False | 82.34% [80.4345, 84.0588] | True | False | False / False |
| native | minimum distance | fine-tuned | -0.000651767 [-0.0022684, 0.000455326] | False | 79.73% [77.9782, 81.4983] | True | False | False / False |
| external | contacts | zero-shot | -0.699811 [-3.15204, 0.112805] | False | 81.53% [79.1032, 83.6696] | True | False | False / False |
| external | contacts | fine-tuned | -0.913652 [-3.32729, -0.118365] | False | INVALID crossing quantiles | False | False | False / False |
| external | retained fraction | zero-shot | -0.00200299 [-0.00663809, 0.00160062] | False | 85.09% [82.4242, 87.4091] | False | False | False / False |
| external | retained fraction | fine-tuned | -0.00470703 [-0.0114648, -0.000827605] | False | 76.27% [73.3607, 79.2576] | True | True | False / False |
| external | minimum distance | zero-shot | -0.0326164 [-0.145217, 0.00177749] | False | 83.12% [81.0289, 85.0924] | True | False | False / False |
| external | minimum distance | fine-tuned | 0.0249189 [-0.0190585, 0.173296] | False | 80.71% [78.4394, 82.8802] | False | False | False / False |
| replica | contacts | zero-shot | 1.73919 unresolved | False | 66.67% unresolved | False | False | False / False |
| replica | contacts | fine-tuned | 1.10791 unresolved | False | 66.67% unresolved | False | False | False / False |
| replica | retained fraction | zero-shot | 0.00250615 unresolved | False | 63.33% unresolved | False | False | False / False |
| replica | retained fraction | fine-tuned | -0.00431645 unresolved | False | 58.33% unresolved | False | False | False / False |
| replica | minimum distance | zero-shot | -0.00317305 unresolved | False | 83.33% unresolved | False | False | False / False |
| replica | minimum distance | fine-tuned | -0.00088589 unresolved | False | 85.00% unresolved | False | False | False / False |

### Full descriptive grid

Every cell/method is retained, including failed probability cells. The ratios
are MAE divided by the respective reference (below 1 is better). Pinball is
the mean over quantiles 0.1/0.5/0.9; widths and pinball have observable units.
Coverage is the empirical fraction inside the nominal 80% interval. Compact
has no probability forecast; its uncertainty columns are unavailable.
The saved synthesis and HTML contain all marginal intervals, paired effects,
per-lead errors, spread diagnostics and raw window values.

| Task | C/H | Observable | Method | MAE | RMSE | Ratio pers/stat/compact | Coverage | Width | Mean pinball | Median spread ratio |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| native | c20-h5 | contacts | persistence | 13.927 | 17.3144 | 1/1.22888/1.24864 | 0.826263 | 47.6 | 4.52905 | 0 |
| native | c20-h5 | contacts | selected-statistic | 11.3331 | 14.1741 | 0.813747/1/1.01608 | 0.821465 | 38.035 | 3.70023 | 0.140276 |
| native | c20-h5 | contacts | compact | 11.1537 | 13.9464 | 0.800868/0.984174/1 | unavailable | unavailable | unavailable | 0.107027 |
| native | c20-h5 | contacts | zero-shot | 11.2984 | 14.1305 | 0.811257/0.99694/1.01297 | 0.843687 | 41.7848 | 3.64452 | 0.105591 |
| native | c20-h5 | contacts | fine-tuned | 11.106 | 13.8815 | 0.797444/0.979966/0.995724 | 0.735606 | 31.7101 | 3.56108 | 0.0403469 |
| native | c20-h5 | retained fraction | persistence | 0.0657711 | 0.0855579 | 1/1.14136/1.17966 | 0.815152 | 0.212212 | 0.0219473 | 0 |
| native | c20-h5 | retained fraction | selected-statistic | 0.0576251 | 0.0763622 | 0.876146/1/1.03355 | 0.815404 | 0.180113 | 0.0193691 | 0.132381 |
| native | c20-h5 | retained fraction | compact | 0.0557544 | 0.0737293 | 0.847703/0.967537/1 | unavailable | unavailable | unavailable | 0.138624 |
| native | c20-h5 | retained fraction | zero-shot | 0.0566639 | 0.0748814 | 0.861531/0.98332/1.01631 | 0.852273 | 0.218296 | 0.0190558 | 0.145173 |
| native | c20-h5 | retained fraction | fine-tuned | 0.0550728 | 0.0730009 | 0.83734/0.955709/0.987775 | 0.733081 | 0.152053 | 0.0182383 | 0.069414 |
| native | c20-h5 | minimum distance | persistence | 0.102052 | 0.127389 | 1/1.2832/1.27293 | 0.836616 | 0.347739 | 0.0339485 | 0 |
| native | c20-h5 | minimum distance | selected-statistic | 0.0795296 | 0.100137 | 0.779301/1/0.991998 | 0.834343 | 0.26568 | 0.0269978 | 0.0501832 |
| native | c20-h5 | minimum distance | compact | 0.0801711 | 0.101002 | 0.785588/1.00807/1 | unavailable | unavailable | unavailable | 0.145388 |
| native | c20-h5 | minimum distance | zero-shot | 0.0807824 | 0.101896 | 0.791578/1.01575/1.00762 | 0.866667 | 0.311711 | 0.0262896 | 0.0952472 |
| native | c20-h5 | minimum distance | fine-tuned | 0.0785861 | 0.099576 | 0.770056/0.988136/0.980229 | 0.815657 | 0.264266 | 0.0252039 | 0.0364401 |
| native | c20-h10 | contacts | persistence | 14.4289 | 17.9788 | 1/1.23536/1.25148 | 0.827273 | 49.12 | 4.68974 | 0 |
| native | c20-h10 | contacts | selected-statistic | 11.6799 | 14.674 | 0.809479/1/1.01305 | 0.815404 | 38.7299 | 3.81675 | 0.0930374 |
| native | c20-h10 | contacts | compact | 11.5294 | 14.5034 | 0.799051/0.987118/1 | unavailable | unavailable | unavailable | 0.0902705 |
| native | c20-h10 | contacts | zero-shot | 11.6303 | 14.632 | 0.806039/0.99575/1.00874 | 0.857449 | 45.4403 | 3.80653 | 0.106668 |
| native | c20-h10 | contacts | fine-tuned | 11.4899 | 14.4557 | 0.796313/0.983735/0.996573 | 0.743939 | 33.3635 | 3.68697 | 0.0511607 |
| native | c20-h10 | retained fraction | persistence | 0.0697918 | 0.0906747 | 1/1.13674/1.17498 | 0.81553 | 0.222415 | 0.023113 | 0 |
| native | c20-h10 | retained fraction | selected-statistic | 0.0613967 | 0.0817785 | 0.879712/1/1.03365 | 0.809975 | 0.187119 | 0.0206922 | 0.0822918 |
| native | c20-h10 | retained fraction | compact | 0.0593982 | 0.0784637 | 0.851077/0.96745/1 | unavailable | unavailable | unavailable | 0.141527 |
| native | c20-h10 | retained fraction | zero-shot | 0.060454 | 0.0801517 | 0.866204/0.984645/1.01777 | 0.861237 | 0.244154 | 0.020679 | 0.165046 |
| native | c20-h10 | retained fraction | fine-tuned | 0.0584475 | 0.0779831 | 0.837455/0.951964/0.983994 | 0.739899 | 0.162615 | 0.0193868 | 0.0700458 |
| native | c20-h10 | minimum distance | persistence | 0.102909 | 0.128802 | 1/1.27128/1.26238 | 0.840152 | 0.355071 | 0.0342433 | 0 |
| native | c20-h10 | minimum distance | selected-statistic | 0.0809489 | 0.101578 | 0.786607/1/0.992994 | 0.834596 | 0.271459 | 0.0273859 | 0 |
| native | c20-h10 | minimum distance | compact | 0.0815201 | 0.102559 | 0.792157/1.00706/1 | unavailable | unavailable | unavailable | 0.148682 |
| native | c20-h10 | minimum distance | zero-shot | 0.0815706 | 0.102882 | 0.792648/1.00768/1.00062 | 0.885985 | 0.335788 | 0.0268376 | 0.0936475 |
| native | c20-h10 | minimum distance | fine-tuned | 0.0801584 | 0.101185 | 0.778925/0.990234/0.983296 | 0.818434 | 0.269351 | 0.025602 | 0.0486234 |
| native | c40-h5 | contacts | persistence | 13.9414 | 17.2506 | 1/1.23309/1.26017 | 0.831987 | 48.8 | 4.53654 | 0 |
| native | c40-h5 | contacts | selected-statistic | 11.3061 | 14.1572 | 0.810972/1/1.02196 | 0.823906 | 38.3291 | 3.70554 | 0.20923 |
| native | c40-h5 | contacts | compact | 11.0632 | 13.8195 | 0.793546/0.978512/1 | unavailable | unavailable | unavailable | 0.143528 |
| native | c40-h5 | contacts | zero-shot | 11.0095 | 13.8167 | 0.789695/0.973763/0.995147 | 0.803367 | 36.2422 | 3.51777 | 0.111328 |
| native | c40-h5 | contacts | fine-tuned | 10.9994 | 13.7549 | 0.788975/0.972876/0.99424 | 0.754882 | 32.3444 | 3.51055 | 0.0387883 |
| native | c40-h5 | retained fraction | persistence | 0.0648504 | 0.0831334 | 1/1.14884/1.18892 | 0.813131 | 0.209112 | 0.0216282 | 0 |
| native | c40-h5 | retained fraction | selected-statistic | 0.0564483 | 0.0739792 | 0.87044/1/1.03488 | 0.817172 | 0.174556 | 0.0189757 | 0.225169 |
| native | c40-h5 | retained fraction | compact | 0.0545456 | 0.0710535 | 0.8411/0.966293/1 | unavailable | unavailable | unavailable | 0.177858 |
| native | c40-h5 | retained fraction | zero-shot | 0.0544673 | 0.0711659 | 0.839892/0.964905/0.998564 | 0.819529 | 0.185388 | 0.0179499 | 0.13474 |
| native | c40-h5 | retained fraction | fine-tuned | 0.05345 | 0.06986 | 0.824205/0.946883/0.979913 | 0.745791 | 0.147669 | 0.0174954 | 0.073707 |
| native | c40-h5 | minimum distance | persistence | 0.102961 | 0.127753 | 1/1.26736/1.25907 | 0.826599 | 0.345992 | 0.0341817 | 0 |
| native | c40-h5 | minimum distance | selected-statistic | 0.0812404 | 0.101491 | 0.789043/1/0.99346 | 0.827946 | 0.269132 | 0.0276416 | 0.000374099 |
| native | c40-h5 | minimum distance | compact | 0.0817752 | 0.102628 | 0.794238/1.00658/1 | unavailable | unavailable | unavailable | 0.236452 |
| native | c40-h5 | minimum distance | zero-shot | 0.0817394 | 0.103461 | 0.79389/1.00614/0.999562 | 0.816162 | 0.267398 | 0.0259536 | 0.0925478 |
| native | c40-h5 | minimum distance | fine-tuned | 0.0801502 | 0.10107 | 0.778455/0.986581/0.980129 | 0.801347 | 0.253981 | 0.0254112 | 0.0299438 |
| native | c40-h10 | contacts | persistence | 14.5165 | 17.9773 | 1/1.2348/1.23681 | 0.828956 | 49.73 | 4.70929 | 0 |
| native | c40-h10 | contacts | selected-statistic | 11.7562 | 14.7441 | 0.809849/1/1.00163 | 0.815993 | 38.993 | 3.846 | 0.142484 |
| native | c40-h10 | contacts | compact | 11.7371 | 14.7127 | 0.808532/0.998373/1 | unavailable | unavailable | unavailable | 0.0603032 |
| native | c40-h10 | contacts | zero-shot | 11.4757 | 14.4532 | 0.790527/0.976141/0.977731 | 0.802189 | 37.9196 | 3.65818 | 0.0948769 |
| native | c40-h10 | contacts | fine-tuned | 11.4681 | 14.3904 | 0.790007/0.975499/0.977088 | 0.751852 | 33.4268 | 3.65127 | 0.0502031 |
| native | c40-h10 | retained fraction | persistence | 0.0686483 | 0.0877138 | 1/1.15417/1.16843 | 0.8133 | 0.21943 | 0.022693 | 0 |
| native | c40-h10 | retained fraction | selected-statistic | 0.0594785 | 0.0777426 | 0.866423/1/1.01236 | 0.814983 | 0.182771 | 0.0198994 | 0.174497 |
| native | c40-h10 | retained fraction | compact | 0.0587526 | 0.0758103 | 0.855849/0.987796/1 | unavailable | unavailable | unavailable | 0.0785953 |
| native | c40-h10 | retained fraction | zero-shot | 0.0579261 | 0.0755751 | 0.84381/0.9739/0.985933 | 0.824074 | 0.20211 | 0.019284 | 0.14629 |
| native | c40-h10 | retained fraction | fine-tuned | 0.0561941 | 0.0734564 | 0.818579/0.94478/0.956453 | 0.74697 | 0.15498 | 0.0184438 | 0.0692422 |
| native | c40-h10 | minimum distance | persistence | 0.103074 | 0.128629 | 1/1.25333/1.25373 | 0.833165 | 0.351882 | 0.0342017 | 0 |
| native | c40-h10 | minimum distance | selected-statistic | 0.0822401 | 0.102513 | 0.797875/1/1.00032 | 0.830808 | 0.27325 | 0.0278509 | 0.000226096 |
| native | c40-h10 | minimum distance | compact | 0.0822135 | 0.102719 | 0.797617/0.999677/1 | unavailable | unavailable | unavailable | 0.115462 |
| native | c40-h10 | minimum distance | zero-shot | 0.0822814 | 0.103884 | 0.798275/1.0005/1.00083 | 0.823401 | 0.277695 | 0.0260963 | 0.0862645 |
| native | c40-h10 | minimum distance | fine-tuned | 0.0815883 | 0.10249 | 0.791551/0.992075/0.992396 | 0.797306 | 0.25557 | 0.0257263 | 0.0449127 |
| external | c20-h5 | contacts | persistence | 14.1238 | 17.2315 | 1/1.19617/1.21821 | 0.822525 | 47.6 | 4.66074 | 0 |
| external | c20-h5 | contacts | selected-statistic | 11.8076 | 14.6106 | 0.836003/1/1.01843 | 0.805455 | 38.035 | 3.9225 | 0.172608 |
| external | c20-h5 | contacts | compact | 11.5939 | 14.3618 | 0.820878/0.981908/1 | INVALID | unavailable | unavailable | 0.115417 |
| external | c20-h5 | contacts | zero-shot | 11.6784 | 14.4578 | 0.826856/0.989058/1.00728 | 0.847273 | 43.1533 | 3.76963 | 0.116091 |
| external | c20-h5 | contacts | fine-tuned | 11.4964 | 14.269 | 0.813971/0.973646/0.991586 | 0.736667 | 32.4958 | 3.70346 | 0.0450699 |
| external | c20-h5 | retained fraction | persistence | 0.0579595 | 0.0721966 | 1/1.07652/1.13751 | 0.859192 | 0.212212 | 0.0196277 | 0 |
| external | c20-h5 | retained fraction | selected-statistic | 0.0538398 | 0.0676964 | 0.928921/1/1.05666 | 0.841818 | 0.180113 | 0.0180554 | 0.154195 |
| external | c20-h5 | retained fraction | compact | 0.0509528 | 0.0638298 | 0.87911/0.946377/1 | INVALID | unavailable | unavailable | 0.144908 |
| external | c20-h5 | retained fraction | zero-shot | 0.0513642 | 0.0645462 | 0.886209/0.954019/1.00807 | 0.860909 | 0.207261 | 0.0172807 | 0.144184 |
| external | c20-h5 | retained fraction | fine-tuned | 0.0497408 | 0.0625393 | 0.858199/0.923866/0.976213 | 0.748384 | 0.141619 | 0.0161421 | 0.0778165 |
| external | c20-h5 | minimum distance | persistence | 0.191928 | 0.262523 | 1/1.01024/0.765019 | 0.86697 | 0.347739 | 0.0820493 | 0 |
| external | c20-h5 | minimum distance | selected-statistic | 0.189982 | 0.251443 | 0.989864/1/0.757264 | 0.870606 | 0.26568 | 0.0845903 | 0.0522658 |
| external | c20-h5 | minimum distance | compact | 0.25088 | 0.323982 | 1.30716/1.32054/1 | INVALID | unavailable | unavailable | 0.158937 |
| external | c20-h5 | minimum distance | zero-shot | 0.198879 | 0.274949 | 1.03622/1.04683/0.792727 | 0.857273 | 0.620726 | 0.0699914 | 0.101525 |
| external | c20-h5 | minimum distance | fine-tuned | 0.216486 | 0.290573 | 1.12795/1.1395/0.862905 | 0.811515 | 0.583006 | 0.0756007 | 0.0424468 |
| external | c20-h10 | contacts | persistence | 14.6942 | 18.0064 | 1/1.19316/1.21249 | 0.822576 | 49.12 | 4.86372 | 0 |
| external | c20-h10 | contacts | selected-statistic | 12.3154 | 15.325 | 0.838113/1/1.0162 | 0.801263 | 38.7299 | 4.10047 | 0.124303 |
| external | c20-h10 | contacts | compact | 12.1191 | 15.0945 | 0.824752/0.984058/1 | INVALID | unavailable | unavailable | 0.0971745 |
| external | c20-h10 | contacts | zero-shot | 12.1733 | 15.1724 | 0.828439/0.988458/1.00447 | 0.859192 | 47.4156 | 3.98472 | 0.115924 |
| external | c20-h10 | contacts | fine-tuned | 12.0153 | 15.0086 | 0.817691/0.975633/0.991438 | INVALID | unavailable | unavailable | 0.0548087 |
| external | c20-h10 | retained fraction | persistence | 0.0618912 | 0.0773696 | 1/1.07993/1.13397 | 0.850455 | 0.222415 | 0.0209957 | 0 |
| external | c20-h10 | retained fraction | selected-statistic | 0.0573104 | 0.0724822 | 0.925985/1/1.05004 | 0.834798 | 0.187119 | 0.0193187 | 0.0918746 |
| external | c20-h10 | retained fraction | compact | 0.0545795 | 0.0687472 | 0.881861/0.952349/1 | INVALID | unavailable | unavailable | 0.151965 |
| external | c20-h10 | retained fraction | zero-shot | 0.0557911 | 0.0704524 | 0.901438/0.973491/1.0222 | 0.865202 | 0.236482 | 0.019172 | 0.175032 |
| external | c20-h10 | retained fraction | fine-tuned | 0.0536054 | 0.0680315 | 0.866122/0.935353/0.982153 | 0.746566 | 0.152475 | 0.0175823 | 0.0816167 |
| external | c20-h10 | minimum distance | persistence | 0.233023 | 0.326241 | 1/0.655863/0.775663 | 0.873182 | 0.355071 | 0.1027 | 0 |
| external | c20-h10 | minimum distance | selected-statistic | 0.355292 | 0.44038 | 1.52471/1/1.18266 | 0.87399 | 0.271459 | 0.167329 | 0 |
| external | c20-h10 | minimum distance | compact | 0.300418 | 0.385215 | 1.28922/0.845552/1 | INVALID | unavailable | unavailable | 0.160035 |
| external | c20-h10 | minimum distance | zero-shot | 0.259631 | 0.360594 | 1.11419/0.730754/0.864233 | 0.879343 | 0.724342 | 0.0926346 | 0.100065 |
| external | c20-h10 | minimum distance | fine-tuned | 0.280898 | 0.373266 | 1.20545/0.790613/0.935026 | 0.81697 | 0.634649 | 0.0981104 | 0.0534219 |
| external | c40-h5 | contacts | persistence | 14.1421 | 16.2286 | 1/1.17598/1.2156 | 0.831818 | 48.8 | 4.64955 | 0 |
| external | c40-h5 | contacts | selected-statistic | 12.0258 | 14.0718 | 0.850355/1/1.03369 | 0.812121 | 38.3291 | 4.0448 | 0.223667 |
| external | c40-h5 | contacts | compact | 11.6339 | 13.6383 | 0.822639/0.967407/1 | INVALID | unavailable | unavailable | 0.152093 |
| external | c40-h5 | contacts | zero-shot | 11.5571 | 13.5293 | 0.817211/0.961023/0.993401 | 0.812727 | 37.5721 | 3.66277 | 0.118193 |
| external | c40-h5 | contacts | fine-tuned | 11.3229 | 13.2886 | 0.800651/0.941549/0.973271 | 0.753333 | 32.7867 | 3.63308 | 0.0476079 |
| external | c40-h5 | retained fraction | persistence | 0.0542951 | 0.0630011 | 1/1.10385/1.14529 | 0.868182 | 0.209112 | 0.0184556 | 0 |
| external | c40-h5 | retained fraction | selected-statistic | 0.0491872 | 0.0575778 | 0.905923/1/1.03754 | 0.854848 | 0.174556 | 0.0167301 | 0.239179 |
| external | c40-h5 | retained fraction | compact | 0.0474074 | 0.055364 | 0.873143/0.963816/1 | INVALID | unavailable | unavailable | 0.205667 |
| external | c40-h5 | retained fraction | zero-shot | 0.0478316 | 0.0559442 | 0.880957/0.972441/1.00895 | 0.84303 | 0.17982 | 0.0156765 | 0.153078 |
| external | c40-h5 | retained fraction | fine-tuned | 0.0461187 | 0.0539607 | 0.849409/0.937617/0.972818 | 0.766667 | 0.135984 | 0.0148593 | 0.0974709 |
| external | c40-h5 | minimum distance | persistence | 0.186971 | 0.215924 | 1/0.992092/0.679858 | 0.867576 | 0.345992 | 0.0795162 | 0 |
| external | c40-h5 | minimum distance | selected-statistic | 0.188462 | 0.216013 | 1.00797/1/0.685278 | 0.874242 | 0.269132 | 0.0839716 | 0.0178562 |
| external | c40-h5 | minimum distance | compact | 0.275015 | 0.302535 | 1.47089/1.45926/1 | INVALID | unavailable | unavailable | 0.268348 |
| external | c40-h5 | minimum distance | zero-shot | 0.157343 | 0.185317 | 0.841538/0.834883/0.572127 | 0.819697 | 0.550211 | 0.0581838 | 0.0933534 |
| external | c40-h5 | minimum distance | fine-tuned | 0.185533 | 0.213718 | 0.992307/0.98446/0.674628 | 0.806667 | 0.632311 | 0.0678501 | 0.036729 |
| external | c40-h10 | contacts | persistence | 14.6827 | 17.1911 | 1/1.15537/1.19225 | 0.826364 | 49.73 | 4.85291 | 0 |
| external | c40-h10 | contacts | selected-statistic | 12.7082 | 15.1848 | 0.865522/1/1.03192 | 0.807121 | 38.993 | 4.31746 | 0.179518 |
| external | c40-h10 | contacts | compact | 12.3151 | 14.7175 | 0.838747/0.969065/1 | INVALID | unavailable | unavailable | 0.0640978 |
| external | c40-h10 | contacts | zero-shot | 12.0084 | 14.4082 | 0.81786/0.944932/0.975098 | 0.815303 | 39.9364 | 3.82989 | 0.103151 |
| external | c40-h10 | contacts | fine-tuned | 11.7946 | 14.1803 | 0.803296/0.928105/0.957733 | INVALID | unavailable | unavailable | 0.0551992 |
| external | c40-h10 | retained fraction | persistence | 0.0581775 | 0.0687043 | 1/1.0672/1.12455 | 0.861364 | 0.21943 | 0.0198118 | 0 |
| external | c40-h10 | retained fraction | selected-statistic | 0.054514 | 0.064847 | 0.937028/1/1.05374 | 0.841212 | 0.182771 | 0.0185541 | 0.198708 |
| external | c40-h10 | retained fraction | compact | 0.0517339 | 0.0614842 | 0.889241/0.949001/1 | INVALID | unavailable | unavailable | 0.0877097 |
| external | c40-h10 | retained fraction | zero-shot | 0.052511 | 0.062617 | 0.9026/0.963257/1.01502 | 0.850909 | 0.203882 | 0.01738 | 0.185336 |
| external | c40-h10 | retained fraction | fine-tuned | 0.049807 | 0.0594808 | 0.85612/0.913655/0.962754 | 0.762727 | 0.144269 | 0.0160793 | 0.0896706 |
| external | c40-h10 | minimum distance | persistence | 0.213001 | 0.26206 | 1/0.935742/0.596352 | 0.870606 | 0.351882 | 0.0925884 | 0 |
| external | c40-h10 | minimum distance | selected-statistic | 0.227628 | 0.27652 | 1.06867/1/0.637303 | 0.87803 | 0.27325 | 0.103676 | 0.00938933 |
| external | c40-h10 | minimum distance | compact | 0.357173 | 0.407402 | 1.67686/1.56911/1 | INVALID | unavailable | unavailable | 0.137308 |
| external | c40-h10 | minimum distance | zero-shot | 0.195011 | 0.248294 | 0.915542/0.856712/0.545985 | 0.831212 | 0.6853 | 0.0697335 | 0.0847449 |
| external | c40-h10 | minimum distance | fine-tuned | 0.252547 | 0.308929 | 1.18566/1.10947/0.70707 | 0.807121 | 0.735893 | 0.0890721 | 0.0486343 |
| replica | c20-h5 | contacts | persistence | 12.4111 | 15.1311 | 1/1.08016/1.09741 | 0.855556 | 47.6 | 4.01444 | 0 |
| replica | c20-h5 | contacts | selected-statistic | 11.49 | 13.875 | 0.925786/1/1.01597 | 0.811111 | 38.035 | 3.68088 | 0.0860389 |
| replica | c20-h5 | contacts | compact | 11.3095 | 13.8911 | 0.911238/0.984286/1 | unavailable | unavailable | unavailable | 0.0771444 |
| replica | c20-h5 | contacts | zero-shot | 11.0086 | 13.7012 | 0.886992/0.958096/0.973392 | 0.822222 | 38.3651 | 3.56833 | 0.0744452 |
| replica | c20-h5 | contacts | fine-tuned | 11.3982 | 13.9207 | 0.918388/0.992009/1.00785 | 0.7 | 29.5263 | 3.64309 | 0.0271021 |
| replica | c20-h5 | retained fraction | persistence | 0.0786297 | 0.0947685 | 1/1.24127/1.22116 | 0.744444 | 0.212212 | 0.0257187 | 0 |
| replica | c20-h5 | retained fraction | selected-statistic | 0.0633463 | 0.0776779 | 0.805628/1/0.9838 | 0.744444 | 0.180113 | 0.0214084 | 0.0991937 |
| replica | c20-h5 | retained fraction | compact | 0.0643894 | 0.0788014 | 0.818894/1.01647/1 | unavailable | unavailable | unavailable | 0.123672 |
| replica | c20-h5 | retained fraction | zero-shot | 0.0613606 | 0.0762251 | 0.780374/0.968653/0.952961 | 0.833333 | 0.249359 | 0.0204539 | 0.173752 |
| replica | c20-h5 | retained fraction | fine-tuned | 0.064516 | 0.0778691 | 0.820504/1.01846/1.00197 | 0.711111 | 0.174972 | 0.0209284 | 0.0650735 |
| replica | c20-h5 | minimum distance | persistence | 0.104853 | 0.128826 | 1/1.27677/1.29857 | 0.811111 | 0.347739 | 0.0331057 | 0 |
| replica | c20-h5 | minimum distance | selected-statistic | 0.0821237 | 0.0998973 | 0.783226/1/1.01708 | 0.855556 | 0.26568 | 0.0253885 | 0.0391404 |
| replica | c20-h5 | minimum distance | compact | 0.0807449 | 0.0996913 | 0.770076/0.983211/1 | unavailable | unavailable | unavailable | 0.160329 |
| replica | c20-h5 | minimum distance | zero-shot | 0.084107 | 0.103533 | 0.802141/1.02415/1.04164 | 0.833333 | 0.330916 | 0.0269821 | 0.108578 |
| replica | c20-h5 | minimum distance | fine-tuned | 0.0822587 | 0.0997993 | 0.784513/1.00164/1.01875 | 0.844444 | 0.281233 | 0.0260183 | 0.0350987 |
| replica | c20-h10 | contacts | persistence | 12.8278 | 15.7926 | 1/1.08904/1.09556 | 0.827778 | 49.12 | 4.23215 | 0 |
| replica | c20-h10 | contacts | selected-statistic | 11.779 | 14.5932 | 0.91824/1/1.00599 | 0.8 | 38.7299 | 3.92181 | 0.0637534 |
| replica | c20-h10 | contacts | compact | 11.7088 | 14.4974 | 0.912772/0.994045/1 | unavailable | unavailable | unavailable | 0.0822575 |
| replica | c20-h10 | contacts | zero-shot | 11.4758 | 14.3402 | 0.894606/0.974262/0.980098 | 0.833333 | 42.3928 | 3.68924 | 0.103165 |
| replica | c20-h10 | contacts | fine-tuned | 11.7333 | 14.5582 | 0.91468/0.996123/1.00209 | 0.727778 | 31.35 | 3.76624 | 0.0473662 |
| replica | c20-h10 | retained fraction | persistence | 0.082453 | 0.097616 | 1/1.20954/1.22521 | 0.733333 | 0.222415 | 0.0259823 | 0 |
| replica | c20-h10 | retained fraction | selected-statistic | 0.0681691 | 0.0856281 | 0.826763/1/1.01296 | 0.733333 | 0.187119 | 0.0228063 | 0.0992345 |
| replica | c20-h10 | retained fraction | compact | 0.0672972 | 0.0828966 | 0.816189/0.98721/1 | unavailable | unavailable | unavailable | 0.130407 |
| replica | c20-h10 | retained fraction | zero-shot | 0.0653273 | 0.0832714 | 0.792297/0.958313/0.970728 | 0.811111 | 0.278779 | 0.0226488 | 0.1247 |
| replica | c20-h10 | retained fraction | fine-tuned | 0.0664702 | 0.0825984 | 0.806159/0.975079/0.987712 | 0.711111 | 0.185751 | 0.022045 | 0.0505996 |
| replica | c20-h10 | minimum distance | persistence | 0.0981666 | 0.121852 | 1/1.24263/1.25002 | 0.827778 | 0.355071 | 0.0316161 | 0 |
| replica | c20-h10 | minimum distance | selected-statistic | 0.0789991 | 0.0969863 | 0.804745/1/1.00595 | 0.844444 | 0.271459 | 0.0245426 | 0 |
| replica | c20-h10 | minimum distance | compact | 0.078532 | 0.0960765 | 0.799987/0.994087/1 | unavailable | unavailable | unavailable | 0.171904 |
| replica | c20-h10 | minimum distance | zero-shot | 0.081397 | 0.0990733 | 0.829172/1.03035/1.03648 | 0.894444 | 0.359323 | 0.0267288 | 0.111675 |
| replica | c20-h10 | minimum distance | fine-tuned | 0.0797688 | 0.0964612 | 0.812586/1.00974/1.01575 | 0.872222 | 0.287683 | 0.0250199 | 0.0503272 |
| replica | c40-h5 | contacts | persistence | 12.6 | 14.4879 | 1/1.17523/1.06894 | 0.866667 | 48.8 | 3.94889 | 0 |
| replica | c40-h5 | contacts | selected-statistic | 10.7213 | 12.7051 | 0.850898/1/0.909563 | 0.833333 | 38.3291 | 3.32323 | 0.0990924 |
| replica | c40-h5 | contacts | compact | 11.7873 | 13.6874 | 0.935502/1.09943/1 | unavailable | unavailable | unavailable | 0.148786 |
| replica | c40-h5 | contacts | zero-shot | 11.8694 | 13.5921 | 0.942012/1.10708/1.00696 | 0.733333 | 33.6463 | 3.72955 | 0.139872 |
| replica | c40-h5 | contacts | fine-tuned | 11.8632 | 13.4825 | 0.941522/1.1065/1.00643 | 0.7 | 30.0545 | 3.71432 | 0.0410258 |
| replica | c40-h5 | retained fraction | persistence | 0.0937413 | 0.104737 | 1/1.13139/1.17534 | 0.6 | 0.209112 | 0.0305321 | 0 |
| replica | c40-h5 | retained fraction | selected-statistic | 0.0828553 | 0.0923788 | 0.883871/1/1.03885 | 0.633333 | 0.174556 | 0.0277024 | 0.232636 |
| replica | c40-h5 | retained fraction | compact | 0.0797565 | 0.0889252 | 0.850815/0.962601/1 | unavailable | unavailable | unavailable | 0.106885 |
| replica | c40-h5 | retained fraction | zero-shot | 0.080091 | 0.0920022 | 0.854383/0.966637/1.00419 | 0.666667 | 0.210475 | 0.0287124 | 0.178594 |
| replica | c40-h5 | retained fraction | fine-tuned | 0.0780139 | 0.0873398 | 0.832225/0.941568/0.97815 | 0.666667 | 0.173225 | 0.026666 | 0.0733542 |
| replica | c40-h5 | minimum distance | persistence | 0.121371 | 0.140537 | 1/1.39308/1.35753 | 0.7 | 0.345992 | 0.0400583 | 0 |
| replica | c40-h5 | minimum distance | selected-statistic | 0.0871243 | 0.102785 | 0.717834/1/0.97448 | 0.766667 | 0.269132 | 0.0273536 | 0.000314051 |
| replica | c40-h5 | minimum distance | compact | 0.089406 | 0.102881 | 0.736633/1.02619/1 | unavailable | unavailable | unavailable | 0.260887 |
| replica | c40-h5 | minimum distance | zero-shot | 0.0796743 | 0.0957344 | 0.656452/0.91449/0.891152 | 0.766667 | 0.276385 | 0.0246616 | 0.136451 |
| replica | c40-h5 | minimum distance | fine-tuned | 0.0853307 | 0.100022 | 0.703056/0.979413/0.954418 | 0.8 | 0.275567 | 0.0264144 | 0.03365 |
| replica | c40-h10 | contacts | persistence | 14.2 | 16.8164 | 1/1.13534/1.09061 | 0.766667 | 49.73 | 4.506 | 0 |
| replica | c40-h10 | contacts | selected-statistic | 12.5073 | 15.2711 | 0.880793/1/0.960601 | 0.733333 | 38.993 | 4.14855 | 0.0777503 |
| replica | c40-h10 | contacts | compact | 13.0202 | 15.6433 | 0.916919/1.04102/1 | unavailable | unavailable | unavailable | 0.0526148 |
| replica | c40-h10 | contacts | zero-shot | 14.2465 | 17.0666 | 1.00327/1.13905/1.09418 | 0.666667 | 35.1541 | 4.74296 | 0.175311 |
| replica | c40-h10 | contacts | fine-tuned | 13.6152 | 16.288 | 0.958815/1.08858/1.04569 | 0.666667 | 31.0113 | 4.44155 | 0.0626186 |
| replica | c40-h10 | retained fraction | persistence | 0.107037 | 0.117094 | 1/1.10705/1.12891 | 0.55 | 0.21943 | 0.0338392 | 0 |
| replica | c40-h10 | retained fraction | selected-statistic | 0.0966872 | 0.107789 | 0.903303/1/1.01975 | 0.633333 | 0.182771 | 0.0337618 | 0.202895 |
| replica | c40-h10 | retained fraction | compact | 0.0948149 | 0.105257 | 0.885812/0.980636/1 | unavailable | unavailable | unavailable | 0.0631171 |
| replica | c40-h10 | retained fraction | zero-shot | 0.0991933 | 0.111776 | 0.926717/1.02592/1.04618 | 0.633333 | 0.228283 | 0.0355401 | 0.20979 |
| replica | c40-h10 | retained fraction | fine-tuned | 0.0923707 | 0.103697 | 0.862977/0.955357/0.974222 | 0.583333 | 0.180096 | 0.0317402 | 0.0841365 |
| replica | c40-h10 | minimum distance | persistence | 0.105386 | 0.126361 | 1/1.32313/1.27252 | 0.75 | 0.351882 | 0.0359862 | 0 |
| replica | c40-h10 | minimum distance | selected-statistic | 0.0796485 | 0.0956401 | 0.755782/1/0.961746 | 0.816667 | 0.27325 | 0.0254581 | 0.000194073 |
| replica | c40-h10 | minimum distance | compact | 0.0828165 | 0.10063 | 0.785843/1.03978/1 | unavailable | unavailable | unavailable | 0.142025 |
| replica | c40-h10 | minimum distance | zero-shot | 0.0764754 | 0.0915843 | 0.725673/0.960162/0.923432 | 0.833333 | 0.287375 | 0.0239474 | 0.106913 |
| replica | c40-h10 | minimum distance | fine-tuned | 0.0787626 | 0.0941974 | 0.747376/0.988877/0.951049 | 0.85 | 0.278462 | 0.0248657 | 0.053472 |

### Preserved failed cells

External fine-tuned contacts fail quantile ordering at **5 / 19,800 points
in C20/H10 (five windows)** and **2 / 6,600 points in C40/H10 (one window)**.
All seven point predictions remain included. No other evaluated target/cell
fails ordering. One reproduced raw 4NNI forecast has q0.1=0.133046,
q0.5=−0.220100 and q0.9=17.190212: the lower quantile exceeds the median,
and the count median is outside the physical nonnegative support. No clipping
or distribution repair is applied. These are genuine limitations of the frozen
model. Ordered quantiles alone do not validate physical support or MD dynamics.

### Provenance and measured costs

Portable raw-source provenance is committed; datasets, checkpoints, saved
forecasts and generated HTML remain local ignored files. Each task summary
pins 122 files and its genuine target split/evaluation dataset. All fitted
artifact hashes match the frozen development choices; cross-dataset bindings
do not relabel training provenance. The independently saved synthesis checksum is
`sha256:63f9be248faac769ec89b52d369fe01d9fbff1c2740c8972a9179415a480d1e8`.

| Task | Summary SHA-256 | Scoring code commit | Groups / trajectories / frames | Scoring wall | Peak RSS KiB | Peak allocated VRAM bytes |
| --- | --- | --- | --- | --- | ---: | ---: |
| native | `sha256:489be74cd1c3d679fcad56bfbd02b0e928c2edc3186d2397e7181a7a6529ba8e` | `63deeaafcb63bd651f88fb50af1645f8255b2ea8` | 99 / 99 / 9,900 | 1:08.36 | 2217636 | 491442176 |
| external | `sha256:de57cc8a06f2b286c38c0ca98ad06e4395f3f6062eecb3216acd6852ca47bbc6` | `855102ec1709409228c9aafe50591b7893665b70` | 66 / 660 / 33,000 | 4:19.64 | 2483084 | 488607232 |
| replica | `sha256:fa51684dec70dafdbc1f4e5f315a9bcf2f4baa33b02fe3073bf8fe2b791100c8` | `0e5a797f62035ce4b81a8436e436507b09c0b40b` | 6 / 6 / 300 | 0:30.38 | 2034228 | 488607232 |

Scoring used the NVIDIA RTX 4070 Laptop GPU (8 GiB), Python 3.14 and the pinned
Chronos-2 revision/checkpoints. Per-batch timings and allocated/reserved VRAM are
stored in every scored method result; wall times above include model loading.
Native preparation: 8:14.76 / 211,164 KiB RSS. External acquisition of admitted
7,525,583,944 bytes: 14:51.57 / 1,069,168 KiB RSS; unsuccessful acquisition
attempts additionally took 0:37.33 and 4:49.16. External geometric preparation:
6:32.89 / 442,536 KiB RSS. Seen-replica preparation: 0:06.61 / 275,732 KiB RSS.
The first external scoring failure took 1:33.93 / 2,284,476 KiB RSS; the second
failed attempt is retained in its log and is not counted as a successful run.
Final synthesis wall/RSS were not recovered after the session restart;
no estimate is substituted. Account/model usage and monetary cost are unknown.

The three new bundles contribute 24,300 observable/cell windows to the overview;
with historical/development panels the HTML contains 64 panels and 29,844 such
windows. Counts refer to repeated observable/cell views, not independent data.

The final report is generated by the command above, then checked with optional
`playwright==1.63.0` and local Chrome, entirely offline:

```sh
/tmp/md-overview-python/bin/python examples/check_overview_browser.py \
  data/reports/offline-overview45/index.html \
  --output data/reports/offline-overview45/browser-check
```

Interpretation remains restricted to these dry-source scalar geometries and
horizons. PDB-disjoint complexes are not guaranteed unseen targets/chemotypes;
exact protein sequence overlap with the earlier pilot is zero, but ligand
InChIKey overlaps occur (including 4OCK with the seen cohort), and pretraining
membership is unknown. No free-energy, mechanistic causality, atomistic
trajectory or long-time MD replacement claim follows from these forecasts.
