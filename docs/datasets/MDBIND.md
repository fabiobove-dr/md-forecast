# MDbind feasibility and semantic audit

Retrieved **2026-10-03** for issue #14. This is an audit and handoff, not a
completed adapter or external benchmark. No MDbind forecasts, model selection,
fine-tuning or scientific test result have been produced. Cross-dataset scoring
of the existing native MISATO checkpoint remains blocked by the equivalence
requirements below. `configs/datasets/mdbind-audit.json` records portable evidence;
it is audit metadata, not an executable acquisition/adapter configuration.

## Sources, access and terms

The paper describes 6,300 complexes, ten 10-ns replicas per complex and a
200-ps frame interval. Its processed datasets are on Zenodo and raw trajectories
are served by MDposit. These are different representations and license records.
[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12371333/),
[Zenodo record](https://zenodo.org/records/10390550),
[MDposit collection](https://mdposit.mddbr.eu/#/browse?search=MDBind).

The immutable Zenodo record DOI is `10.5281/zenodo.10390550`, published
2024-06-06; the page labels it v1 but the REST metadata has no explicit `version`.
The authoritative API reports open access and license `etalab-2.0`.
Open Licence 2.0 requires acknowledgement of the information's source and update
and covers reuse, including commercial reuse, subject to its terms. The upstream
code is separately BSD-3-Clause at commit
`227f8d7ef6f719695ecd23a7d787be06249bb464`.
[Record API](https://zenodo.org/api/records/10390550),
[license text](https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf),
[code license](https://github.com/ICOA-SBC/MD_DL_BA/blob/227f8d7ef6f719695ecd23a7d787be06249bb464/LICENSE).

The inspected raw record `MD-A003OP` (PDB 4DPY, node `cin`, local ID `A046U`)
separately declares CC-BY-4.0 and ten named replicas, Amber ff14SB/GAFF and TIP3P.
This raw-record license must not be substituted for the Zenodo license, or
extrapolated to every raw record. Underlying complexes come from PDBbind; no
conclusion about constituent redistribution rights is inferred from download
access. Source bytes and acquired manifests remain ignored locally.
[Raw record](https://mdposit.mddbr.eu/api/rest/current/projects/MD-A003OP),
[raw-record license](https://creativecommons.org/licenses/by/4.0/).

## Verified processed artifact

`4D_only_ligand_tracking_test_set.zip` is **15,815,806 bytes**, checksum
`md5:4f72c4306748f79dfe7616fb9d399d10`; complete downloaded bytes matched both.
The complex archive is 398,590,692 bytes, MD5
`8cc578dfc32e5bb8557077bc19de2195`, and was not downloaded. Training complex
archives are multi-GB, while the raw corpus is described as approximately 4.5 TB.
The local filesystem had 533 GiB available; inspection used the small tracking
archive and one approximately 1-MB XTC, not the complete corpus.
[Artifact metadata](https://zenodo.org/api/records/10390550),
[tracking download](https://zenodo.org/api/records/10390550/files/4D_only_ligand_tracking_test_set.zip/content).

All 830 `.npy` arrays in the tracking archive were read with
`allow_pickle=False`, one member at a time. They cover 83 complexes with exactly
replicas 1–10; every array is finite float64 `(50, N_ligand_atoms, 22)`.
For example `test_set/1p1n/replicate_1.npy` is `(50,15,22)`. The first three
columns contain coordinates according to the upstream generator; the remaining
19 are featurizer channels. **They are not 19 MD observable series.** The arrays
carry no timestamps, atom-ID metadata, protein coordinates, masses or native
MISATO observable keys.
[Generator](https://github.com/ICOA-SBC/MD_DL_BA/blob/227f8d7ef6f719695ecd23a7d787be06249bb464/datasets/spatio-temporal_learning_dataset.py).

The tracking generator subtracts each frame's ligand geometric centroid. The
untracked representation subtracts the pocket centroid from ligand and pocket
coordinates. Frames are numerically sorted, and simulations without the expected
50 pocket files are omitted. The selected tracking sample's centroid norms were
approximately 2–7 × 10^-6 in its stored coordinate units. Centering cannot restore
protein-aligned ligand RMSD or a receptor COM distance when protein/mass data are
absent. Atom order across exported frames and feature units must be verified
before deriving new observables; shape alone is not that verification.
[Generator, frame processing](https://github.com/ICOA-SBC/MD_DL_BA/blob/227f8d7ef6f719695ecd23a7d787be06249bb464/datasets/spatio-temporal_learning_dataset.py#L116).

## Raw sample, timestamps and replica access

The raw API provides `structure.pdb`, `trajectory.xtc`, `topology.prmtop`,
per-replica metadata and analysis outputs. File descriptors for the first raw
replica report PDB 416,132 bytes, XTC 979,772 bytes and topology 2,236,693 bytes.
The PDB and XTC were downloaded; topology was not. Raw replica 2 was identified
through `MD-A003OP.2`: its response has `mdIndex=1`, `mdNumber=2`, `mdName=replica 2`
and an XTC descriptor of 979,716 bytes. Its trajectory payload was not inspected.
The UI constructs the `.N` accession suffix from a one-based replica number.
[File descriptors](https://mdposit.mddbr.eu/api/rest/current/projects/MD-A003OP/filenotes),
[replica 2](https://mdposit.mddbr.eu/api/rest/current/projects/MD-A003OP.2),
[API documentation](https://mdposit.mddbr.eu/api/rest/docs/).

Pinned local MDTraj 1.11.1.post2 read the first XTC as **50 frames, 5,137 atoms**.
Its actual timestamps are **210, 410, …, 10,010 ps**: the interval is 200 ps,
the retained span is **9,800 ps**, and the absolute stored origin is 210 ps.
A reported 10-ns simulation length must not be used to invent a zero origin or
10-ns retained span. Canonical relative time would subtract 210 ps, while keeping
the stored origin in provenance. The API's `FRAMESTEP=0.2` is insufficient on its
own to establish a unit; the XTC read supplies the measured ps axis here.
[Sample XTC](https://mdposit.mddbr.eu/api/rest/current/projects/MD-A003OP/files/trajectory.xtc).

MDTraj exposes coordinates in nm, requiring explicit multiplication by 10 for
Angstrom features. The PDB has 330 protein residues and final ligand residue 331
named `2P0` with 29 atoms, matching the record's protein/ligand interaction
selection and protein atom count 5,108. This checks one sample, not all catalog
records. A raw adapter must verify topology/coordinate ordering, reviewed ligand
selection, elements, receptor extent and periodic-image policy for every selected
complex before admitting structural features.
[Raw metadata](https://mdposit.mddbr.eu/api/rest/current/projects/MD-A003OP),
[MDTraj trajectory contract](https://mdtraj.readthedocs.io/en/latest/api/generated/mdtraj.Trajectory.html).

GET download URLs work and redirect to the federated node. HEAD requests returned
HTTP 500 in this inspection; that was not evidence of unavailable payloads.
The catalog query `search=MDBind&limit=1` reported 4,960 matching project records
on retrieval. This is not verified full coverage of the paper's 6,300 complexes.
Freeze a catalog snapshot/explicit accession list rather than relying on changing
server order or the first returned record.
[Catalog query](https://mdposit.mddbr.eu/api/rest/current/projects?limit=1&search=MDBind).

## Splits, overlap and uncertainty

Pinned upstream `by_complex` lists contain 4,753 training, 1,179 validation and
83 test complex IDs, each unique; their pairwise intersections are empty. The
tracking test ZIP agrees with the 83-ID test list and has all ten replicas.
The inspected `by_sim` file encodes paths such as `./1p1n/replicate_8.npy`.
Never treat ten neural-network model seeds as ten simulation replicas.
[Pinned split lists](https://github.com/ICOA-SBC/MD_DL_BA/tree/227f8d7ef6f719695ecd23a7d787be06249bb464/datasets/version4D/inputs).

Cross-dataset IDs matter: of these 83 MDbind test complexes, **79 occur in the
locally checksum-verified official MISATO TRAIN list**, zero in its VAL/TEST
lists. None occur among the 20 local MISATO sample complexes (including the
19 official TRAIN systems used for development). Thus the absence of overlap
with the small fitted checkpoint does not establish absence from full MISATO
training. Sequence/chemotype relationships and model-pretraining overlap have
not been audited. A full-MISATO unseen-complex claim must exclude matching
training IDs and report the remaining independent-group count, not call a
different dataset automatically unseen.
[Verified MISATO split source](MISATO.md),
[MDbind test list](https://github.com/ICOA-SBC/MD_DL_BA/blob/227f8d7ef6f719695ecd23a7d787be06249bb464/datasets/version4D/inputs/by_complex/test_samples.txt).

For unseen-complex evaluation, split complexes before windows and average windows
within each replica, replicas within each complex, and bootstrap complexes.
For unseen-replica evaluation, retain original system identity and hold out named
replicas while the system remains seen. Report that task separately. The current
split implementation unions all records sharing `system_id`; a dedicated,
validated replica protocol is required. Do not bypass its checks by inventing new
system IDs. Nested complex/replica dependence still governs uncertainty.
[Scientific contract](../SCIENTIFIC_CONTRACT.md),
[preparation contract](PREPARATION.md).

## Feature semantic equivalence gate

All rows require explicit native units and matching preprocessing. The existing
MISATO native model does not have a transferable feature definition merely
because another quantity has the same name.

| Quantity | MISATO definition | Inspected MDbind support | Decision |
| --- | --- | --- | --- |
| Native ligand RMSD (Å) | Reference, fit mask and alignment procedure unpublished | Tracking removes translation; raw coordinates permit a newly defined RMSD | Block existing native equivalence until masks/reference are sourced |
| Native ligand/receptor COM distance (Å) | Exact atom/mass selection unpublished | Tracking lacks receptor/masses; raw topology is available but uninspected | Block existing native equivalence |
| Native buried SASA (Å²) | Probe, atom selections and normalization unpublished | Raw analyses named `sasa` do not establish matching buried SASA | Block; isolated ligand SASA is a different quantity |
| Native interaction energy (kcal/mol) | Exact energy decomposition/settings unresolved | An `energies` analysis name is not an MM/GBSA equivalence table | Block existing native equivalence |
| Minimum receptor/ligand heavy distance (Å) | All source receptor/ligand heavy atoms, raw Cartesian, no PBC correction | Raw full-system coordinates and ligand metadata offer a candidate | Candidate for a new common structural benchmark; verify every selected topology and image policy first |
| Contact count / reference-contact fraction | Inclusive 4.5-Å heavy pair contacts, first stored frame reference | Same candidate raw source | Candidate; preserve receptor extent, cutoff, reference and zero-reference exclusions |

MISATO's unresolved native definitions are recorded in its existing audit;
structural definitions are explicit and versioned. AMBER simulation protocols,
retained origins, coordinate precision and imaging may still differ even with a
common geometric quantity; these must be stated as irreducible source differences.
[Native audit](MISATO.md#observable-admission-and-units),
[structural definitions](STRUCTURAL_FEATURES.md).

## Remaining acceptance and handoff

Issue #14 stays open. No adapter schema with guessed channels or times has been
created, and no test result or issue closure is implied by this audit.

To transfer the existing native checkpoint, authoritative native definitions
and a matching MDbind computation are required. Alternatively, freeze a new
common geometric feature set and run its MISATO benchmark/model selection before
untouched MDbind scoring; that changes the evaluated feature/model experiment.
The latter route also requires per-system topology/selection checks, deterministic
accession/replica selection with overlap exclusions, content-verified acquisition,
the explicit replica split protocol, complex-level uncertainty and separate
unseen-complex/unseen-replica reports. No MDbind fine-tuning may precede the first
external result.

Local ignored evidence is under `data/external/mdbind-audit/`. It includes the
Zenodo API snapshot, pinned generator/split/license files, complete tracking ZIP
and shape inspection, raw metadata/file descriptors, one PDB/XTC, and the paper
BioC text. Portable hashes, observed dimensions and admission status are in the
audit manifest. This is schema and feasibility evidence, not proof of forecast
skill, scientific null results or global data quality.
