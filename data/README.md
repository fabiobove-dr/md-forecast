# Local data

Dataset bytes are external inputs, never repository fixtures. The whole `data/`
tree is ignored except this file and reviewed `manifests/*.json` or
`manifests/*.toml` metadata. `git add -f` bypasses this protection; do not use it
for external data.

Suggested local directories (created only when needed):

- `external/`: downloaded upstream files.
- `cache/`: disposable intermediate files.
- `processed/`: normalized series.
- `manifests/`: small, versioned provenance metadata; no source dataset bytes,
  secrets, or absolute machine paths.

Keep checkpoints and generated experiment reports in ignored `checkpoints/`
and `reports/` at the repository root. Record checksums, versions, and
acquisition instructions after the dataset audit; public access does not imply
permission to redistribute.
