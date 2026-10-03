# Common geometric external validation

This issue #14 experiment evaluates three explicitly equivalent dry-system
geometric quantities, after the [MDbind audit](../datasets/MDBIND.md). It is a new
zero-shot experiment; the native RMSD fine-tuned checkpoint is not transferred.
The full specification is `configs/benchmarks/mdbind-common.toml` and the
feature configuration is `configs/features/common-geometry.yaml`.

## Frozen protocol

- Features: contact pair count, fraction of first-frame contacts, minimum heavy
  distance. See the audit for matching selections, reference, units and exclusions.
- MISATO development: 19 local official TRAIN systems, grouped 70/30/0 with seed
  42, giving 13 fitting and six validation complexes. Official TEST is not loaded.
- Models: revision-pinned Chronos-2 zero-shot, persistence, context mean, linear
  extrapolation, AR(1) and VAR(1); no MDbind fine-tuning. Statistical settings match
  the existing baseline specification. Per feature and lead (including the full
  horizon average), choose lowest validation MAE; ties use configuration hash.
- Grid: C40/H10, stride 10 frames. MISATO physical cadence is unresolved; no equal
  physical horizon across datasets is claimed. MDbind has 200-ps samples: context
  first-to-last span **7.8 ns**, and last-context-to-final-target horizon **2 ns**.
- Untouched external task: all ten replicas of each of six nonoverlapping PDB
  complexes, one complete window per replica. Scaled metrics use explicitly
  linked, frozen MISATO TRAIN-context moments; physical MAE/RMSE remain primary.
  Transfer rejects different feature definitions/units and overlapping PDB IDs.
- Seen-complex replica task: same true system IDs; replicas 1–8 TRAIN, 9–10 TEST.
  After the first external report has been published and verified, fit a fixed
  NLinear baseline with ridge 0 on these TRAIN replicas. Chronos remains frozen;
  this task includes local baseline adaptation and is reported separately.
- Averaging: windows within replica, replicas within complex, equal complex
  weighting. Bootstrap whole complexes; 300,000 draws, seed 42, 95% confidence,
  at least five groups, Bonferroni family 600 (594 planned point comparisons),
  at least ten samples per corrected tail. Six complexes limit inference.
- Quantiles 0.1/0.5/0.9, nominal 80% coverage, frozen acceptable deviation 0.1.
  Report coverage, width and pinball loss at every lead. Deterministic baselines
  do not supply a probabilistic comparator; calibration alone cannot establish
  useful uncertainty or the stronger scientific hypothesis.

The acquisition cohort and all model/grid/uncertainty settings precede external
scoring. Geometry QC reads source payloads but performs no model selection.
Checkpoint/model pretraining overlap and unseen-target/chemotype grouping have
not been audited; exact PDB exclusion supports only the named complex task.

## Reproduce

```sh
uv sync --locked --extra structural --extra chronos
uv run --locked python examples/prepare_mdbind_common.py --download --output data/processed/mdbind-common-verified
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 uv run --locked python examples/run_mdbind_common.py --output data/processed/mdbind-common-benchmark
```

The MISATO structural sample and pinned Chronos snapshot must already exist,
as described in [structural extraction](../datasets/STRUCTURAL_FEATURES.md) and
[Chronos integration](CHRONOS2.md). Use fresh output directories. Report bundles,
source bytes and the fitted local baseline stay ignored under `data/`.
Implementation/configuration must be committed before the run records provenance.

## Measured outcome and repeatability

The complete experiment ran twice at commit
`5317f5b19b391f68b09972fad6ed459a33c52b15`. All three scientific report identities,
prediction/metric/comparison tables, the frozen baseline selection, selected
baseline effects and fitted replica NLinear state were identical across runs.
Operational timings are excluded from scientific identities. There was no
MDbind fine-tuning before or after the first external result; only the explicitly
separate replica baseline was fitted after the untouched report was verified.

**External scientific success is not established.** At the full-horizon average,
contact count and minimum distance beat persistence with corrected paired
intervals below zero. The reference-contact fraction beats the MISATO-selected
AR baseline, but its persistence difference includes zero. Contact count's
selected-baseline corrected interval includes zero, and minimum distance matches
its selected VAR baseline. No lead >=2 in either external task has both paired
corrected MAE intervals below zero against persistence and the selected baseline.
The entire grid is retained; this is a reproducible negative/inconclusive external
outcome, not proof that MD observables cannot be forecast.

| Task | Feature (native unit) | Persistence MAE | Selected MAE | Chronos MAE | Chronos − selected, corrected 95% CI | 80% coverage |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| unseen-complex | Contact pairs (count) | 17.6983 | 15.6966 | 14.3238 | [-2.49024, 0.0856971] | 0.8167 |
| unseen-complex | Reference fraction (1) | 0.0445343 | 0.0561873 | 0.0436973 | [-0.0353443, -0.00160637] | 0.8367 |
| unseen-complex | Minimum distance (Å) | 0.0680422 | 0.0538491 | 0.0538716 | [-0.00135699, 0.00115017] | 0.8517 |
| unseen-replica | Contact pairs (count) | 16.2167 | 15.7098 | 14.2528 | [-5.17197, 0.208102] | 0.8167 |
| unseen-replica | Reference fraction (1) | 0.0378514 | 0.0485842 | 0.0400814 | [-0.0236697, 0.00668482] | 0.9083 |
| unseen-replica | Minimum distance (Å) | 0.0699596 | 0.0612112 | 0.0619328 | [-0.00218698, 0.00309642] | 0.7917 |

The full-horizon choices frozen on MISATO are AR(1) for contact count/fraction and
VAR(1) for minimum distance. MDbind outcomes do not reselect these baselines.
On MISATO validation Chronos MAEs were 12.025695, 0.096848 and 0.086539, versus
selected 11.753042, 0.094776 and 0.085057 respectively; native units follow the
table. The new geometric development run itself does not establish superiority.

For unseen-replica evaluation, the fixed NLinear baseline was fitted on 48 TRAIN
replicas/windows and evaluated on 12 TEST replicas/windows of six seen complexes.
Its MAEs were 56.720444 count, 0.136057 fraction and 0.118717 Å. This small,
unregularized baseline performs poorly; it was neither retuned nor hidden.
Chronos' reference fraction coverage is 0.908333 here, outside the frozen
[0.7,0.9] band. With six complexes, coverage intervals also leave some calibration
comparisons inconclusive. No calibrated-superiority claim is made, and there is
no probabilistic baseline comparison establishing informative interval width.

## Artifact and resource identity

Configuration: `sha256:40229376afd537bf44cbdda3889d23ff1918b96ea5c848a393e1ffdca91a829b`.
Lockfile: `sha256:6114c9b8d23e5e8b2bf890dcedacfca4d5569bb9d1e6958a498d29d10d40681b`.
Baseline-selection payload:
`sha256:067797de387f1d584545ade6660573271bf6229cde6baf8139bb6786da13aa60`.

| Task | Windows / trajectories / complexes | Scientific report SHA-256 |
| --- | --- | --- |
| misato-validation | 36 / 6 / 6 | `sha256:e338844399e9d481ae36548227bc27f8500b3ea510e50f36ae7e5290b6ca21ec` |
| mdbind-unseen-complex | 60 / 60 / 6 | `sha256:a4eeaa3a156a0138c6ba11abdffd871b8d081cfa82df2519f50d98469b6b1ecb` |
| mdbind-unseen-replica | 12 / 12 / 6 | `sha256:4e0a82b4da786c62cb7a4661bf07e9723e6b642766658c7270c6dff2bd8a82fc` |

Full split, source, feature, preprocessing, checkpoint and table hashes are in
each verified report manifest. The external scaler is the unchanged MISATO
TRAIN-context scaler; the replica scaler is fitted exclusively on MDbind TRAIN
contexts. Replicas 9/10 never enter that fit. Code/config provenance is captured
before inference; reproducing exact report identity requires the frozen commit,
whereas later documentation changes alter only the recorded code identity.

Hardware: NVIDIA RTX 4070 Laptop, 8188 MiB, driver 580.173.02; Python 3.14.8.
First-run Chronos adapter times (including its initial weight load) were 0.599 s
MISATO, 0.617 s external and 0.123 s replica; peak allocated VRAM <489 MB, peak
reserved 505,413,632 bytes. Acquisition/extraction, table aggregation and bootstrap
are separate costs. Timing varies between repeated runs.

Local ignored bundles are `data/processed/mdbind-common-benchmark` and its
`-repeat` sibling. Each contains complete native MAE/RMSE/scaled-MAE, probabilistic
metrics, per-replica/system/group/aggregate rows, horizon curves, prediction
quantiles, paired persistence comparisons, runtime telemetry and input hashes.
The selected-baseline effect tables retain marginal and corrected complex CIs.
Source and generated report bundles are not committed.

## Every external lead

Lead 0 is the average over all ten future samples; leads 1–10 occur at
0.2–2.0 ns from the last observed sample. The selected statistical baseline is
frozen separately for each lead on MISATO. CI is Chronos-minus-selected MAE,
Bonferroni corrected. Coverage/width refer to the 80% interval; pinball is the
mean over the frozen three quantiles. Width and error follow the feature's native
unit. Full group CIs, RMSE and persistence ratios remain in the report tables.

### unseen-complex: Contact pairs (count)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 17.6983 | 15.6966 | 14.3238 | [-2.49024, 0.0856971] | 0.8167 | 48.2438 | 3.17817 |
| 1 | 16.2 | 13.3651 | 12.7405 | [-3.88137, 1.51393] | 0.8500 | 42.1531 | 2.68502 |
| 2 | 18.0333 | 13.5037 | 13.4603 | [-4.15576, 3.40634] | 0.7833 | 44.7815 | 3.14869 |
| 3 | 19.5 | 16.699 | 16.0678 | [-2.46707, 0.708853] | 0.7833 | 46.285 | 3.40165 |
| 4 | 16.85 | 14.7483 | 13.5036 | [-3.94531, 1.20742] | 0.8500 | 47.5171 | 3.08278 |
| 5 | 19.8167 | 16.4132 | 16.0255 | [-2.36416, 1.13437] | 0.7000 | 48.2542 | 3.27767 |
| 6 | 17.9667 | 15.2308 | 14.2476 | [-4.32114, 1.36647] | 0.8333 | 49.2236 | 2.94303 |
| 7 | 19.0167 | 16.6367 | 15.1287 | [-3.06295, 0.108765] | 0.8000 | 49.956 | 3.55157 |
| 8 | 15.2833 | 13.9077 | 12.9062 | [-6.09193, 4.5303] | 0.8500 | 50.7088 | 2.96826 |
| 9 | 17.85 | 16.6843 | 14.9996 | [-5.37952, 1.65483] | 0.8500 | 51.5149 | 3.0666 |
| 10 | 16.4667 | 15.3575 | 14.1586 | [-4.2272, 3.28326] | 0.8667 | 52.0434 | 3.65646 |

### unseen-complex: Reference fraction (1)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.0445343 | 0.0561873 | 0.0436973 | [-0.0353443, -0.00160637] | 0.8367 | 0.168008 | 0.0106057 |
| 1 | 0.0388806 | 0.0400342 | 0.0337543 | [-0.0235944, 0.00398081] | 0.8500 | 0.131509 | 0.00678962 |
| 2 | 0.043536 | 0.0436841 | 0.0352435 | [-0.0240861, 0.00606094] | 0.8667 | 0.143353 | 0.0100495 |
| 3 | 0.0471309 | 0.0599087 | 0.0445139 | [-0.0448503, -0.00330929] | 0.8167 | 0.151997 | 0.010092 |
| 4 | 0.0415953 | 0.0495549 | 0.039578 | [-0.0323177, 0.00521527] | 0.8167 | 0.160055 | 0.0098269 |
| 5 | 0.0454196 | 0.0551584 | 0.0421027 | [-0.0352353, -0.00316462] | 0.8833 | 0.167636 | 0.00948262 |
| 6 | 0.0490419 | 0.0598788 | 0.0463012 | [-0.038226, -0.00175331] | 0.7500 | 0.172885 | 0.011841 |
| 7 | 0.0407145 | 0.0591102 | 0.0433196 | [-0.0453949, 0.000765021] | 0.8833 | 0.178612 | 0.0102629 |
| 8 | 0.0452385 | 0.0613917 | 0.050357 | [-0.0313048, -0.0011716] | 0.8167 | 0.185862 | 0.0120844 |
| 9 | 0.0464929 | 0.0682325 | 0.0520379 | [-0.0396128, 0.00124611] | 0.8000 | 0.192712 | 0.0135962 |
| 10 | 0.0472931 | 0.0801486 | 0.0497648 | [-0.0739168, -0.00256995] | 0.8833 | 0.195461 | 0.0120317 |

### unseen-complex: Minimum distance (Å)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.0680422 | 0.0538491 | 0.0538716 | [-0.00135699, 0.00115017] | 0.8517 | 0.193178 | 0.0122602 |
| 1 | 0.0747093 | 0.0563101 | 0.0597712 | [-0.00212013, 0.00993047] | 0.7833 | 0.174636 | 0.0164579 |
| 2 | 0.0600365 | 0.0440568 | 0.0481256 | [-0.00290696, 0.0109743] | 0.9333 | 0.182781 | 0.00952388 |
| 3 | 0.0700138 | 0.0508218 | 0.0482503 | [-0.0113425, 0.0092874] | 0.8833 | 0.187256 | 0.0105601 |
| 4 | 0.0752213 | 0.0692996 | 0.068106 | [-0.00403331, 0.00210564] | 0.7833 | 0.191184 | 0.013261 |
| 5 | 0.0624321 | 0.0426969 | 0.0445435 | [-0.00259217, 0.00853445] | 0.8833 | 0.192987 | 0.0100788 |
| 6 | 0.0816856 | 0.0635219 | 0.0626342 | [-0.00564043, 0.00427468] | 0.8333 | 0.195694 | 0.0139602 |
| 7 | 0.0670234 | 0.0492474 | 0.0506404 | [-0.00461686, 0.0144649] | 0.8500 | 0.198195 | 0.0122511 |
| 8 | 0.0665972 | 0.0521922 | 0.0521304 | [-0.00353155, 0.00363702] | 0.8167 | 0.200805 | 0.0117625 |
| 9 | 0.0699024 | 0.0601089 | 0.0587796 | [-0.01006, 0.00347428] | 0.8333 | 0.203429 | 0.0133391 |
| 10 | 0.0527999 | 0.048881 | 0.0457345 | [-0.00878215, 0.00248757] | 0.9167 | 0.204812 | 0.0114077 |

### unseen-replica: Contact pairs (count)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 16.2167 | 15.7098 | 14.2528 | [-5.17197, 0.208102] | 0.8167 | 49.9128 | 3.49781 |
| 1 | 11.9167 | 11.3698 | 12.232 | [-2.00163, 4.32408] | 0.8333 | 44.2549 | 3.03157 |
| 2 | 16 | 15.1268 | 12.9161 | [-8.02852, 2.42035] | 0.7500 | 46.6771 | 3.50899 |
| 3 | 11.5 | 11.7949 | 10.0303 | [-8.38398, 1.81703] | 1.0000 | 48.2306 | 2.26815 |
| 4 | 18.0833 | 16.65 | 14.3948 | [-7.80035, 0.976459] | 0.9167 | 49.298 | 2.21487 |
| 5 | 22.5 | 20.0913 | 19.3237 | [-4.28285, 1.53215] | 0.5000 | 50.0071 | 3.47086 |
| 6 | 13.1667 | 9.47319 | 10.0696 | [-4.17803, 8.43881] | 0.9167 | 50.8125 | 2.97205 |
| 7 | 16 | 19.4917 | 17.2461 | [-7.31939, -0.00393384] | 0.7500 | 51.3817 | 3.64258 |
| 8 | 14.9167 | 7.16989 | 10.393 | [-6.65838, 16.8142] | 0.9167 | 52.2072 | 2.94192 |
| 9 | 21.3333 | 20.0573 | 18.8701 | [-8.9806, 6.67035] | 0.7500 | 52.8255 | 4.41676 |
| 10 | 16.75 | 17.6253 | 17.0521 | [-13.4642, 11.7478] | 0.8333 | 53.4332 | 6.5103 |

### unseen-replica: Reference fraction (1)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.0378514 | 0.0485842 | 0.0400814 | [-0.0236697, 0.00668482] | 0.9083 | 0.165748 | 0.00932935 |
| 1 | 0.0317484 | 0.0364262 | 0.0359486 | [-0.0121479, 0.0130512] | 0.8333 | 0.134832 | 0.00803662 |
| 2 | 0.046278 | 0.0457545 | 0.0460382 | [-0.0195583, 0.0212701] | 0.8333 | 0.145611 | 0.0137596 |
| 3 | 0.0408844 | 0.0597488 | 0.044881 | [-0.0206234, -0.00699215] | 0.9167 | 0.152911 | 0.00987289 |
| 4 | 0.0443252 | 0.0511304 | 0.0475 | [-0.0281205, 0.0302593] | 0.9167 | 0.159788 | 0.00807539 |
| 5 | 0.0305532 | 0.0392727 | 0.0274215 | [-0.0278642, 0.0123327] | 1.0000 | 0.165406 | 0.00792039 |
| 6 | 0.0311803 | 0.0312564 | 0.0299805 | [-0.0215427, 0.028463] | 0.9167 | 0.16957 | 0.00851338 |
| 7 | 0.0467109 | 0.060864 | 0.049576 | [-0.0287613, 0.0228864] | 0.9167 | 0.17463 | 0.00951146 |
| 8 | 0.0398424 | 0.0437011 | 0.0425882 | [-0.0277496, 0.031449] | 0.9167 | 0.180618 | 0.0106995 |
| 9 | 0.0372798 | 0.0601473 | 0.0399535 | [-0.0292842, -0.0079781] | 0.9167 | 0.18546 | 0.00840977 |
| 10 | 0.0297115 | 0.0713386 | 0.0369268 | [-0.0539751, 0.00136207] | 0.9167 | 0.18865 | 0.00849458 |

### unseen-replica: Minimum distance (Å)

| Lead | Persistence MAE | Selected MAE | Chronos MAE | Corrected Δ CI | Coverage | Width | Pinball |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.0699596 | 0.0612112 | 0.0619328 | [-0.00218698, 0.00309642] | 0.7917 | 0.190573 | 0.0135614 |
| 1 | 0.0980644 | 0.083332 | 0.0868283 | [-0.0102333, 0.0133174] | 0.5000 | 0.173048 | 0.0200715 |
| 2 | 0.0667997 | 0.045364 | 0.0476485 | [-0.00730057, 0.0130674] | 1.0000 | 0.181214 | 0.0107511 |
| 3 | 0.0647095 | 0.0577258 | 0.0587156 | [-0.00876753, 0.0192892] | 0.8333 | 0.184931 | 0.00912052 |
| 4 | 0.0844801 | 0.0641259 | 0.0640435 | [-0.0111697, 0.00789955] | 0.7500 | 0.189448 | 0.0112648 |
| 5 | 0.0333725 | 0.0309538 | 0.0306172 | [-0.013493, 0.0107918] | 1.0000 | 0.191712 | 0.00821153 |
| 6 | 0.0788299 | 0.0726338 | 0.0763333 | [-0.00199696, 0.00709277] | 0.8333 | 0.193511 | 0.0162842 |
| 7 | 0.0692973 | 0.0572982 | 0.0598326 | [-0.0055559, 0.00852234] | 0.7500 | 0.194545 | 0.0158062 |
| 8 | 0.063961 | 0.0661951 | 0.0661646 | [-0.00687569, 0.00850577] | 0.6667 | 0.197866 | 0.0155667 |
| 9 | 0.0740865 | 0.0738384 | 0.0743312 | [-0.00838328, 0.00552166] | 0.7500 | 0.199303 | 0.0161616 |
| 10 | 0.0659946 | 0.0571856 | 0.0548137 | [-0.0161064, 0.00848383] | 0.8333 | 0.200153 | 0.0123755 |

## Context autocorrelation diagnostic

This **exploratory reporting diagnostic** does not select models or horizons.
Reuse `analysis.qc.autocorrelation` on each scored window's 40 observed samples,
with maximum lag 10, global demeaning and fixed denominator. Never use future
samples for this diagnostic. Median lag-1 ACFs for count/fraction/distance are
0.1915/0.2304/−0.0035 on 36 MISATO validation contexts,
0.1592/0.3117/0.0247 on 60 external contexts, and
0.1645/0.3408/0.0257 on the 12 held-out replica contexts.
Median first 1/e crossings are one sample in all cases; no curve is censored at
lag 10. For MDbind this median is 0.2 ns, versus a 2-ns horizon. These are noisy
short-context dependence estimates, not established molecular correlation times
or evidence for long-timescale physical generalization.

The following reproduces the diagnostic from verified bundles:

```sh
uv run --locked python - <<'PY'
import json
from pathlib import Path
import numpy as np
from md_forecast.analysis.qc import autocorrelation
from md_forecast.data.artifacts import read_metadata
from md_forecast.evaluation.benchmark import CellManifest
from md_forecast.data.series import read_series
from md_forecast.data.windows import iter_windows
root=Path('data/processed/mdbind-common-benchmark')
for task, source in [('misato-validation','misato-structural-verified'),('mdbind-unseen-complex','mdbind-common-verified'),('mdbind-unseen-replica','mdbind-common-verified')]:
 m=read_metadata(root/f'{task}-manifest.json',CellManifest)
 directory=Path('data/processed')/source
 exports=json.loads((directory/'qc.json').read_text())['exported']
 tables={};rows={f:[] for f in m.spec.feature_ids}
 for window in iter_windows(m.split,m.grid,m.partition):
  table=tables.setdefault(window.trajectory_id,read_series(directory/exports[window.trajectory_id]))
  for feature in rows:
   values=table[feature].to_numpy()[window.context_slice]
   acf=autocorrelation(values,10)
   crossings=[] if acf is None else np.flatnonzero(acf[1:]<=np.exp(-1)).tolist()
   rows[feature].append({'rho1':None if acf is None else float(acf[1]),'decay_frames':None if not crossings else crossings[0]+1,'censored':acf is not None and not crossings})
 print(task,{f:{'contexts':len(v),'median_rho1':float(np.median([r['rho1'] for r in v if r['rho1'] is not None])),'median_decay_frames':float(np.median([r['decay_frames'] for r in v if r['decay_frames'] is not None])),'censored':sum(r['censored'] for r in v)} for f,v in rows.items()})
PY
```
