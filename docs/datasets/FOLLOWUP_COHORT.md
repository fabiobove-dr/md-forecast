# Fresh development and confirmation cohorts

The old MVP and performance-report scores are **development evidence**. They
cannot validate a newly chosen configuration independently. Issue
[#38](https://github.com/fabiobove-dr/md-forecast/issues/38) enlarges development
and reserves new identities before any forecast window or model selection.

## Native protocol

`configs/datasets/followup-cohort.json` declares seed 42,
200 official TRAIN systems, 60 official VAL systems for selection, another 40
VAL systems for calibration, and 100 official TEST systems for confirmation.
Calibration retains its original VAL label; its separate role prevents its use
for checkpoint selection. No official TEST system becomes training data.

The `configs/datasets/followup-exclusions.json` excludes
110 distinct PDB identities: every system in the upstream tiny MISATO sample,
every complex in the MDbind tracking pilot, the six raw MDbind pilot complexes,
and the separately inspected 4DPY. The categories overlap. These exclusions
are conservative: inspecting source/labels during the earlier analysis is
enough to exclude a system from fresh confirmation.

Only HDF5 root group names and verified official split lists enter selection.
Within each original split, sort by the raw SHA-256 digest of
`42|<original split>|<uppercase PDB>`. Consume VAL selection identities first,
then calibration identities without replacement. The eligible population is
source groups intersecting official labels, minus every excluded identity.
Missing populations fail; do not add favorable systems after scoring.

`CohortManifest` binds exact assignments, declared counts, original labels,
eligible counts, excluded identities, source-configuration hash, original
source-byte checksum, common dataset/geometry hashes and official-list SHA-256
hashes. The portable envelope itself has a verified canonical JSON SHA-256.
No trajectories need to be decoded to create this manifest.

The complete source passed its official byte-count/MD5 check on 2026-10-05.
The frozen native cohort envelope is
`sha256:8a3d790129a1e6cb2538d7bfdeefbf7754e107949655aba18479c33fd8d0f9af`.
Eligible populations observed from root identities are 13,667 TRAIN,
1,595 VAL and 1,610 TEST. All 400 requested identities were assigned before
coordinate decoding. The 110 exclusions contain 100 official MISATO IDs;
the others belong to previously inspected external sources.

## External protocol and overlap

`configs/datasets/followup-external-reserve.json`
uses the earlier frozen public MDbind catalog. Require ten replicas and catalog
CC-BY-4.0 evidence. Exclude exact overlap with **all 16,972 official MISATO
identities** and the 110 previously inspected identities. Collapse duplicate
PDB identities to their lexicographically first accession. There are **312
eligible distinct complexes** under these rules. Rank by the SHA-256 digest of
`42|mdbind-unseen|<PDB>`; reserve the first 50 for unseen-complex confirmation.

The next six complexes form a separate **seen-system replica** task. Their
replicas 1–6 are development training, 7–8 selection, 9 calibration, and 10
confirmation. These six complexes cannot also contribute to the 50-complex
unseen-system task. Six independent groups are still a small sample; do not
claim that their many overlapping windows establish confirmatory power.

`configs/datasets/followup-external-metadata.json`
checks the selected replica-1 source records, source identity, dry atom counts,
CC-BY-4.0 URL and ligand interaction selection. It records source hashes,
protein-sequence hashes and available InChIKeys. **No XTC payload was requested
or decoded for this review.** Reported cadence/frame counts are metadata;
actual XTC times/boxes and topology order must pass the admitted extractor
before scoring. Missing or inadmissible sources stay documented exclusions;
never silently select replacements after outcomes are seen.

Across the 56 reviewed source records, none shares an exact protein sequence
with the six old raw MDbind pilot complexes; seven share an exact InChIKey.
Between the 50 reserved unseen complexes and the six new seen-system
development complexes, exact protein-sequence hashes are disjoint; 4OCK
shares an exact InChIKey with the new seen-system development cohort.
This is not an audit of sequence similarity or chemical scaffold similarity.
MISATO target/ligand mappings and model pretraining overlap remain unresolved.
Exact PDB disjointness supports an unseen-complex claim, not an unseen-target
or unseen-chemotype claim. New chemical independence must not be assumed.

## Precision planning and resource budget

Planning uses the six-group historical common-geometric development results,
not the new reserved outcomes. For contact-count MAE, paired Chronos-minus-AR
standard deviations were 0.354532 on MISATO and 1.061831 on MDbind; corresponding
AR means were 11.753042 and 15.696593 contacts. The planning target is a **10%
relative MAE reduction**. This is a model-development target, not a physical
per-prediction tolerance or an application utility threshold.

A normal approximation with 80% power and Bonferroni-adjusted 95% confidence
for four planned primary comparisons gives implausibly small raw sample-size
estimates (2 native / 6 external groups). Six pilot groups cannot establish
those variances reliably. Inflate the paired standard deviation by three,
and choose **100 native / 50 external independent confirmation complexes**.
Approximate confidence half-widths are then 2.26% / 7.17% of the pilot baseline
MAE. These are planning approximations, not measured confidence intervals or
guaranteed power. Final comparison-family size, resamples, grid and reference
selection are frozen under #43/#45; a larger family needs revised precision
and a numerically resolvable tail. Window counts are never sample-size units.

The verified complete source is 132,841,014,019 bytes (123.72 GiB), MD5
`9bc6446922cd80e0f2f3f69349bf88ed`, from immutable
[Zenodo record 7711953](https://zenodo.org/records/7711953). The local disk had
536 GiB free before acquisition. The initial single-stream rate was about
6 MiB/s (roughly six hours); parallel HTTP ranges preserve transferred bytes
and still require the original complete checksum before publication.
Only coordinate-derived three-channel common geometry is admitted. Extraction
is one system at a time with the existing 64 MiB numerical working budget;
MISATO stays in frame units because physical sampling remains unverified.
Source bytes, canonical tables and generated QC stay ignored locally.

## Preparation and access policy

Run from the repository root with the locked environment:

```sh
uv run --locked md-forecast download --mode md --destination data/external/misato
uv run --locked python examples/prepare_followup_cohort.py freeze
uv run --locked python examples/prepare_followup_cohort.py prepare
uv run --locked python examples/prepare_followup_cohort.py prepare \
  --purpose calibration --output data/processed/followup-calibration
uv run --locked python examples/review_followup_external.py \
  --output data/reports/followup-cohort/external-metadata.json
uv run --locked python examples/plan_followup_precision.py \
  --output data/reports/followup-cohort/precision-planning.json
uv run --locked python examples/prepare_mdbind_common.py \
  --source configs/datasets/followup-seen-source.json \
  --reserve configs/datasets/followup-external-reserve.json --purpose tuning \
  --raw data/external/followup-seen-raw \
  --output data/processed/followup-seen-development --download
uv run --locked python examples/audit_periodic_geometry.py \
  --source configs/datasets/followup-seen-source.json \
  --reserve configs/datasets/followup-external-reserve.json --purpose tuning \
  --raw data/external/followup-seen-raw --max-pairs 2000000 \
  data/reports/followup-cohort/seen-development-periodic.json
```

The first command verifies the full source and official lists. `freeze` checks
those bytes, reads identity metadata only and refuses to overwrite a manifest.
`prepare` checks frozen source/geometry/official labels, decodes only its
permitted role and atomically publishes complete canonical tables, the
registry, QC, geometry and cohort envelopes. Scientific invalidity is recorded
per system by the existing extractor; no interpolation or replacement occurs.
The preparation CLI deliberately exposes no confirmation purpose.

Use `CohortLoader(..., purpose="tuning")` for every tuning experiment: only
TRAIN and selection VAL records are accessible. Explicit calibration processes
use `purpose="calibration"`. The final frozen evaluator alone uses
`purpose="confirmation"`. Source checksum, dataset identity, exact canonical
record, preserved official split and role scope are checked **before payload
read**. Missing exports and paths/symlinks escaping the declared root fail.
Caller mutation of the original manifest/registry cannot expand access.
This is an accidental-leakage guard, not an OS security boundary against code
that deliberately opens source HDF5 itself. Reserved external trajectories
are absent from the development loader and cannot be admitted to development
by the native manifest.

The existing MDbind preparer accepts `--reserve
configs/datasets/followup-external-reserve.json --purpose tuning` and filters
whole roles before the first topology/trajectory extraction. Foreign PDBs,
altered source accessions and sources with no permitted roles fail. On the
seen-system task, tuning admits replicas 1–8; calibration admits only 9;
confirmation admits only 10. All replicas of the 50 unseen complexes require
the explicit confirmation purpose. A source manifest covering all ten
replicas must still be integrity-checked before extraction; the static
metadata review does not substitute for those raw payload checks.

The seen-system source manifest binds the full reserve SHA-256. Omitting the
reserve, changing its roles or using a foreign source fails before any XTC
decode, including in the periodic-geometry audit. Downloading and checking
opaque reserved bytes is distinct from decoding reserved outcomes.

Real preparation of the seen-system development task verified 538,607,156
source bytes across 60 replicas and published only replicas 1–8: **48
trajectories / 2,400 frames / six complexes**. All decoded values are finite;
contact counts are nonnegative integers, reference fractions lie in [0, 1],
and minimum distances are nonnegative. Actual XTC sampling is uniformly
200 ps with a 9.8 ns retained span. On these 2,400 admitted development frames,
raw versus periodic diagnostics found zero differences for contact counts,
reference fractions and minimum distances. Replicas 9 and 10 remain opaque.
Preparation took 49.28 s, peak RSS 271,880 KiB; the separate periodic diagnostic
took 103.31 s, peak RSS 323,744 KiB. These measurements do not validate the
as-yet-unopened confirmation frames or future cohorts.

Real native development preparation exported **200 TRAIN / 60 selection VAL
systems, 26,000 frames**, with zero exclusions. All three common geometric
channels are finite and satisfy their ranges; each trajectory has 100 ordered
frame samples. Its registry hash is
`sha256:517fa47cb013e34cd649333ad86c9e0ff754c214cc238865fb326caebce48366`.
A decode-access audit recorded exactly these 260 identities and no calibration
or confirmation identity. Full-source verification plus extraction took
12 min 1.85 s, peak RSS 206,240 KiB; metadata-only cohort freezing, including
another full integrity check, took 5 min 22.61 s, peak RSS 112,788 KiB.
The 40 calibration and 100 confirmation native systems remain unopened.
These are observed preparation/QC results, not forecast scores or verified
physical-time measurements. Generated access/QC logs remain local under
`data/reports/followup-admission-audit/`; the reviewed portable identity
manifest is committed under `data/manifests/`.
