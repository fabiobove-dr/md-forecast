# Follow-up feature admission

Issue #37 rechecks the assumptions needed by the prediction-quality follow-up.
Retrieved **2026-10-05**. The decision record is
`configs/datasets/followup-admission.json`: it contains source URL/checksum
evidence, admitted definitions, existing configuration identities and explicit
exclusions. It is audit metadata, not permission to assume missing upstream
information or a new feature implementation.

## Sources and measured evidence

The MISATO [immutable record API](https://zenodo.org/api/records/7711953)
still reports version 1.0.0, CC-BY-4.0 and `MD.hdf5` of 132,841,014,019 bytes,
MD5 `9bc6446922cd80e0f2f3f69349bf88ed`. The
[pinned preprocessor](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/preprocessing_db.py)
preserves the native fields and coordinate arrays; its coordinate alignment
helper does not establish how the native RMSD series was originally calculated.
The [pinned trajectory exporter](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/h5_to_traj.py)
does not supply a verified physical timestamp vector or periodic cell for the
HDF5 samples. All 20 groups of the checksum-verified published sample were
re-inspected: no time/box/cell field or group attribute was present.
This is sample evidence, not a full-corpus schema assertion.

The author continues to acknowledge periodic-image reconstruction errors in
[upstream issue #24](https://github.com/t7morgen/misato-dataset/issues/24).
No source-backed corrected timestamp/box field was found in these pinned
representations. Do not infer an 80-ps cadence, unwrap coordinates using an
assumed box, or relabel native RMSD as a specified protein-aligned quantity.
See the original [MISATO audit](MISATO.md) for units and constituent-term gaps.

The MDbind [processed record API](https://zenodo.org/api/records/10390550)
still declares `etalab-2.0`, independently of the raw source records' CC-BY-4.0.
The [pinned tracking generator](https://github.com/ICOA-SBC/MD_DL_BA/blob/227f8d7ef6f719695ecd23a7d787be06249bb464/datasets/spatio-temporal_learning_dataset.py)
centers ligand coordinates; that representation cannot supply receptor geometry,
absolute timestamps or native MISATO feature equivalence. No additional source
representation is admitted here. Existing source-specific terms and constituent
redistribution uncertainty remain in the [MDbind audit](MDBIND.md).

## Admission matrix

All contact/distance rows refer to **all dry-source heavy receptor atoms outside
the final ligand segment**, including retained cofactors/ions. They are not a
protein-only mask. Heavy means atomic number >1; distances use raw Cartesian
Å coordinates, with an inclusive 4.5-Å cutoff. MISATO segment boundaries and
MDbind's reviewed final residue select equivalent receptor extents; individual
topologies and atom identities remain source-specific.

| Observable | Reference / selection | Unit | Decision and scope |
| --- | --- | --- | --- |
| Heavy-pair contact count | Reviewed receptor/ligand pairs; inclusive 4.5 Å | count | Admit existing common raw dry geometry; no claim of corrected physical contacts |
| Fraction of reference contacts | First retained frame of each trajectory; same pair identities/cutoff; reject zero initial contacts | fraction | Admit same definition across sources; each replica has its own initial reference, not a universal crystal or TRAIN reference |
| Minimum heavy-pair distance | Same reviewed heavy atom sets | Å | Admit existing raw geometric definition |
| Pocket/distal radius of gyration | Unweighted Cα; fixed TRAIN replica-one PDB selection, residue minimum ligand-heavy distance ≤6 / ≥12 Å | Å | Admit existing MDbind proxy only; no functional-domain/allostery claim |
| Protein-aligned ligand RMSD | Native source fit mask and reference unresolved | Å | Exclude this interpreted label and native cross-source transfer; retain historical native-series results |
| Native COM / buried SASA / energy equivalence | Mass/selection/probe/decomposition conventions unresolved | native units | Exclude new cross-source equivalence claims |
| MISATO-derived regional/aligned observables | Protein residue sequence/Cα identity not admitted by numeric residue-type codes | — | Exclude until an identity-backed adapter/reference is separately audited |

The first retained contact reference is observed before every scored future;
it is fixed during feature extraction and never selected using prediction error.
References are **not** reset at each forecast origin. Pre-context observations
can define a causal feature, but reports must disclose this trajectory-specific
reference. Native model checkpoints cannot become geometric checkpoints by
renaming channels.

The formulas and version/hash identities are validated through existing
`StructuralConfig`, `MDBindSubset` and `RegionManifest` models. No adapter or
feature formula changed; historical experiments retain their identities.
See [structural features](STRUCTURAL_FEATURES.md) and
[regional selection](../models/PREDICTIVE_COUPLING.md).

## Periodic-image diagnostic on the actual old cohort

All 60 frozen MDbind sources were checksum-verified again, including metadata,
PDB, topology and XTC. The committed reproduction below additionally rechecks
PDB/AMBER atom identities and source interaction selection.

Each XTC has 50 frames, finite periodic boxes, measured ps origin 210 and last
timestamp 10,010, with exact 200-ps spacing (9.8-ns retained span). Comparing
MDTraj 1.11.1.post2 raw versus minimum-image distance calculations for every
admitted receptor/ligand heavy pair gave:

| Diagnostic | Result |
| --- | ---: |
| Replicas / measured frames | 60 / 3,000 |
| Frames with different contact count | 0 |
| Frames with different reference-contact fraction | 0 |
| Maximum minimum-distance difference | 0 Å |

Both sides of this diagnostic use the same float32 MDTraj kernel. It tests the
**imaging policy**; it is not a bitwise comparison to canonical float64 geometry.
Canonical extraction is unchanged. Agreement on these sources does not prove
that new complexes or every protein region are unwrapped. It does not validate
MISATO's unknown boxes or erase force-field/cadence/source differences.

```sh
uv run --locked --extra structural --extra chronos python \
  examples/audit_periodic_geometry.py \
  data/reports/followup-admission-audit/periodic-recheck.json
```

The source cohort is `configs/datasets/mdbind-common.json`, raw inputs are under
ignored `data/external/mdbind-common-raw`, and the derived output stays ignored.
The example bounds the all-pair diagnostic to 400,000 pairs per replica and
rejects changed bytes, invalid boxes, nonfinite coordinates and undefined
reference fractions. It loads no model, fits no preprocessing and downloads
nothing. Optional paths and the pair budget are explicit CLI arguments.

## Conditions for the follow-up

- MISATO remains a **frame-index** task. MDbind may report its measured ps axis.
  Use separate source grids; an identical frame count is not matched physical
  duration. A new cadence assumption must be declared, not inferred from
  nominal simulation length divided by sample count.
- New MDbind sources require the same checksum, identity, interaction,
  timestamp/box and periodic-policy audit before extraction. Any imaging repair
  changes the observable version and requires a new frozen experiment.
- New MISATO coordinate cohorts require finite/boundary QC and explicit imaging
  limitations. A suspicious jump is a diagnostic, not automatic evidence of
  unbinding or authority to invent an imaging correction.
- Regional radii use fixed, source-backed atom membership and raw Cartesian
  coordinates. Periodic agreement of ligand contacts does not validate every
  regional radius; no cross-MISATO regional transfer is admitted.
- Acquisition/training must preserve original source/split provenance. Reserve
  new confirmation systems before tuning; the old scored cohorts are development
  evidence for new hypotheses.
- Keep dataset/structure bytes and acquired metadata ignored. Record attribution
  and source-specific terms; no redistribution right is inferred from access.

These decisions unblock explicitly limited common-geometric development and
new-cohort preparation. Missing native definitions, verified MISATO physical
cadence and broader mechanistic claims remain unsupported.
