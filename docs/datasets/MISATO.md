# MISATO source audit

Audited 2026-10-02 for issue #3. This is an audit of source contracts, not a
full-dataset QC result. Machine-readable acquisition metadata and feature
definitions live in `configs/datasets/misato.yaml` at the repository root.

## Pinned provenance and access

Use record **7711953**, DOI **10.5281/zenodo.7711953**, version **1.0.0**,
published 2023-05-24. The record API gives exact byte sizes and MD5 checksums;
the YAML records all seven artifacts and their content URLs. MD5 is the
upstream integrity checksum, not a security guarantee. A later downloader
must validate complete bytes before promotion. The full MD file has not been
downloaded or locally checksummed during this audit.
[Zenodo record](https://zenodo.org/records/7711953),
[authoritative API](https://zenodo.org/api/records/7711953).

The MD artifact is **132,841,014,019 bytes** (132.84 decimal GB, about 123.71
GiB), rather than 133 GiB. Official split files are small; QM, density, and
restart artifacts are optional and should not be fetched for the first native
observable benchmark. HTTP range support was verified on the MD content
endpoint with status 206 and an exact `Content-Range` total.
[Zenodo API](https://zenodo.org/api/records/7711953).

The paper is [Siebenmorgen et al., DOI 10.1038/s43588-024-00627-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/).
Upstream code is pinned to
[`7b06d532e2ed0719411fcc1b3ac39743db4ca10d`](https://github.com/t7morgen/misato-dataset/tree/7b06d532e2ed0719411fcc1b3ac39743db4ca10d).
Do not silently replace the dataset record, sample, split lists, or code revision
with a mutable latest version.

## License and upstream terms

The dataset API declares **CC-BY-4.0**. Record attribution, DOI, license link,
and changes in outputs. The upstream implementation declares **LGPL-2.1 or
later**; that code license is separate from the dataset license. This project
uses independently written adapters rather than copying the upstream library.
[Dataset metadata](https://zenodo.org/api/records/7711953),
[code license](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/LICENSE),
[upstream license notice](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/h5_to_traj.py).

MISATO starts from PDBbind structures. The current PDBbind+ service has its
own restrictive terms and distinguishes commercial releases starting in 2021;
these current service terms do not establish the exact legacy PDBbind v2020
terms applicable to this artifact. The legacy enrollment page could not be
retrieved in this audit. Do not claim that the record's license resolves all
upstream redistribution rights. Record that gap before company-backed use or
redistribution; keep source structures/trajectories out of Git and distribute
code, portable manifests, acquisition instructions, and aggregate results only
within verified terms.
[Paper provenance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/),
[PDBbind+](https://pdbbind-plus.org.cn/),
[current service terms](https://www.pdbbind-plus.org.cn/termofuse),
[legacy enrollment page](http://www.pdbbind.org.cn/enroll.php).

## Directly inspected HDF5 layout

The pinned upstream `tiny_md.hdf5` contains 20 PDB-ID groups and is 96,598,231
bytes; SHA-256 is recorded in YAML. All 20 groups were inspected, including
all numeric values for finite-value counts. Independently, groups `10GS`,
`11GS`, and `1A3E` were read from the real full Zenodo HDF5 through 11 bounded
range requests totaling 5,309,187 bytes. Their keys, shapes, dtypes, and first
five native feature values agree with the corresponding sample groups.
[Published sample](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/data/MD/h5_files/tiny_md.hdf5),
[full artifact](https://zenodo.org/records/7711953/files/MD.hdf5).

Observed schema for each inspected PDB-ID group (`N` atoms, `M` molecule/chain
segments):

| Key | Shape | Stored dtype | Meaning |
| --- | --- | --- | --- |
| `atoms_element` | `(N,)` | int64 | Upstream mapped element codes, not atomic numbers |
| `atoms_number` | `(N,)` | int64 | Atomic numbers |
| `atoms_residue` | `(N,)` | int64 | Mapped residue types, not sequence positions |
| `atoms_type` | `(N,)` | int64 | Mapped AMBER atom types |
| `molecules_begin_atom_index` | `(M,)` | int64 | Zero-based beginnings of molecule/chain segments |
| `trajectory_coordinates` | `(100, N, 3)` | float64 | Ordered Cartesian snapshots |
| `frames_rmsd_ligand` | `(100,)` | float64 | Native ligand RMSD samples |
| `frames_distance` | `(100,)` | float64 | Native ligand–receptor COM distance samples |
| `frames_bSASA` | `(100,)` | float64 | Native buried SASA samples |
| `frames_interaction_energy` | `(100,)` | float64 | Native MM/GBSA-related interaction-energy samples |

For `10GS`, `N=6593`, `M=3`, and segment starts are `[0, 3267, 6534]`.
The sample has 18 distinct atom counts (1767–6606) and segment counts 2, 3,
or 5. All ten fields are present in all 20 sample groups, with no NaN or
infinite numeric values. This says nothing about missingness in the other
full-dataset groups; global missingness/QC belongs to issue #8. Root, inspected
groups, and datasets have no HDF5 attributes. No periodic box, time, or
velocity field is present in the inspected groups.

Mappings and molecule boundaries must follow the author definitions, not
guessed sequence/chain numbering. Do not execute downloaded pickle maps as
trusted code in an adapter.
[Author explanation](https://github.com/t7morgen/misato-dataset/issues/1#issuecomment-1628932208),
[upstream maps](https://github.com/t7morgen/misato-dataset/tree/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/Maps),
[preprocessor](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/preprocessing_db.py).

## Observable admission and units

The upstream supplement's page 5 heatmap labels verify units independently
of numerical magnitude: RMSD and distance in **Å**, buried SASA in **Å²**,
and interaction energy in **kcal/mol**. The plot labels are embedded in an
image and are missed by plain PDF text extraction. The supplement checksum
is recorded in YAML.
[Author supplement, page 5](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/Supplementary_Information_MISATO.pdf).

| Candidate | Audit decision |
| --- | --- |
| Ligand RMSD after protein alignment | `frames_rmsd_ligand` is time-resolved and measured in Å, but the exact reference, fit mask, and protein-alignment procedure are unpublished in the inspected code. Admit only as **native ligand RMSD**, never rename it “protein-aligned ligand RMSD.” |
| Complex/protein RMSD | Mentioned in the supplement, but absent from all inspected HDF5 groups and the preprocessor field list. Unavailable for the initial native benchmark; do not synthesize a key. |
| Ligand–receptor COM distance | `frames_distance` is a 100-sample series in Å; retain the author's native definition. Exact atom/mass selections are not available for cross-dataset equivalence claims. |
| Buried SASA | `frames_bSASA` is a 100-sample series in Å². Do not claim a particular probe radius, atom selection, or normalization without further source evidence. |
| MM/GBSA-related quantity | `frames_interaction_energy` is a 100-sample interaction-energy series in kcal/mol, not an aggregate, measured binding affinity, or validated binding free energy. |
| Coordinates | `trajectory_coordinates` is time-resolved in Å. The upstream PDB exporter transfers values without unit conversion and the pocket-selection implementation uses an Å cutoff. |
| Timestamps | No stored key or attributes found; see the timing limitation below. |

Native observables use one common frame index. All four were read as full
100-entry arrays in the published sample. Alignment to exact physical
timestamps is not established by array length alone.
[Paper methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/),
[field preservation](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/preprocessing_db.py),
[coordinate exporter](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/h5_to_pdb.py).

## Timing: verified facts and explicit uncertainty

The paper reports 10 ns production, discarding the first 2 ns, with 8 ns
retained in 100 snapshots. The supplement plot uses a retained simulation-time
axis starting at zero. **80 ps is a nominal interval inferred from 8 ns / 100,
not a verified timestamp vector.** The exact export sampling command, whether
both endpoints are included, and the absolute first sample time are unavailable
in the inspected publication/code. Do not silently choose between 80 ps and
8000/99 ps or assert an absolute 2 ns origin.
[Paper methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/),
[supplement time axis](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/Supplementary_Information_MISATO.pdf).

Accordingly YAML stores `verified_frame_interval_ps: null`, an explicitly
inferred nominal value, and an unverified endpoint convention. Adapters can
extract ordered frame-index series. Physical context/horizon conversions
must require verified source timestamps/sampling instructions or a deliberate
documented configuration assumption, preserved in the manifest and report.
Synthetic fixtures cannot resolve this source-data ambiguity.

## Official splits

Downloaded the three official files directly from their immutable Zenodo
record URLs. Each byte count and MD5 matches metadata; additional locally
computed SHA-256 values are in YAML.

| Split file | Bytes | Rows / unique PDB IDs |
| --- | --- | --- |
| `train_MD.txt` | 68,825 | 13,765 |
| `val_MD.txt` | 7,975 | 1,595 |
| `test_MD.txt` | 8,060 | 1,612 |

All IDs are uppercase, there are no duplicates within a split, all pairwise
intersections are empty, and the union has 16,972 systems. Full-HDF5 membership
coverage remains a later ingestion/QC check. The paper describes sequence
clustering for the official MD task; these lists alone do not expose cluster
IDs or independently prove a target/sequence similarity threshold. Group all
windows of each system before windowing and report the split as the official
dataset split, with its verified list hashes.
[Official split metadata](https://zenodo.org/api/records/7711953),
[paper split procedure](https://pmc.ncbi.nlm.nih.gov/articles/PMC11136668/).

## Known limitations and follow-up evidence

Author replies confirm periodic-boundary imaging errors in some trajectories
and splitting chains at unresolved residues. These affect coordinate-derived
features and jump/QC checks; do not “repair” them by assuming unknown boxes.
[Imaging acknowledgment](https://github.com/t7morgen/misato-dataset/issues/24),
[missing-residue clarification](https://github.com/t7morgen/misato-dataset/issues/34).

Before physical-time benchmark claims, obtain sampling instructions/timestamps
or record a user-approved spacing assumption. Before protein-aligned RMSD or
cross-dataset feature equivalence claims, obtain exact masks/references/probes
or derive new versioned definitions from coordinates. Before redistribution
or company-backed use, resolve applicable legacy upstream terms. These are
explicit source limitations, not values to fill in by guesswork.

Re-inspection can use a temporary environment without adding scientific
dependencies to the core scaffold:

```sh
uv run --locked --with h5py python -c 'import h5py; f=h5py.File("data/external/misato-audit/tiny_md.hdf5", "r"); print([(k, v.shape, str(v.dtype)) for k, v in f["10GS"].items()]); f.close()'
```

Acquire that sample only from the pinned URL/checksum in YAML and keep its
bytes under ignored local data. No full MD download, model training, upstream
code execution, or external author message was performed for this audit.
