# Offline forecast overview

`md-forecast report-forecast` turns **saved results** into one standalone HTML
file. It imports no model backend and makes no network calls. Opening the file
requires neither a server nor an internet connection. Use this for inspecting
errors; scientific model selection and confirmation remain separate workflows.

## Reproduce the saved development overview

From the checkout root, with the reviewed local experiment artifacts available:

```sh
uv run --locked md-forecast report-forecast \
  --config configs/experiments/performance-overview.json \
  --output data/reports/offline-overview44/index.html
```

The example pins the larger geometric development bundle from
[the probabilistic follow-up](FOLLOWUP_PROBABILITY.md) and six historical
benchmark bundles. It verifies the frozen MVP snapshot with `read_mvp` and every
benchmark with `read_benchmark`. Follow-up bundles pin `summary.json` by its
file SHA-256 and verify **every** `file_sha256` entry before parsing predictions.
Residual-state feature definitions must match the configured dataset.
No checkpoint is loaded, no model is rerun and no selection is changed.
The full native development grid is retained; C40/H10 is the initial view
because it is the predeclared primary cell, not because of its measured score.

The reviewed render has **28 observable/cell panels** and **5,544 selectable
observable windows**. These include dependent windows and replicas; they are
not 5,544 independent experimental units. The new validation cohort contains
60 independent complexes. Historical pilots retain their original, narrower
roles and sample counts. The native physical cadence remains unresolved;
verified MDbind timing is displayed in ps from the frozen trajectory manifests.

## Input configuration and guarantees

`OverviewConfig` is a Pydantic v2 boundary. Paths in the portable example are
relative to the working directory; installed wheels require explicit local
artifact paths. Each source has a title, evaluation role and limitation note,
a root, a `kind` (`benchmark` or `probability`) and an `expected_hash`:

- A benchmark hash is its `BenchmarkReport.artifact_id`.
- A follow-up hash is `sha256:<file digest>` of `summary.json`. Its `dataset`
  supplies the admitted, versioned feature definitions, checked against the
  saved fitted residual states when those are present.
- Optional `model_labels` bind human-readable roles to exact model hashes.
  Legacy model configurations alone do not identify a fine-tuned checkpoint;
  the report does not guess that role from model ordering.
- Optional `statistical_references` bind observable IDs to an already selected
  model hash. A report never chooses a statistical reference on test results.
  The follow-up bundle explicitly identifies `selected-statistic`.
- `mvp_root` optionally verifies the historical frozen synthesis.
- `default_panel` chooses only the initial view; all panels stay available.
- `max_rows` defaults to 2,000,000 per table; `max_input_bytes` defaults to
  1 GiB across source files. Oversized inputs fail before table loading.

Models must share the same trajectory IDs, group IDs, starts, leads, measured
future values and saved origins. Duplicate/missing leads, crossing/nonfinite
quantiles and median inconsistencies are rejected. Benchmark windows,
features, model identities and target-frame alignment are checked against the
frozen manifest. Hashes alone do not authorize an incorrect alignment.
Output uses atomic replacement: failed integrity/render checks preserve the
previous HTML. Data and text are escaped separately; source strings are never
inserted as executable markup or substituted as template instructions.

## Reading the results

All aggregate and lead-level scores remain available, including MAE, RMSE,
bias, quantile pinball, empirical coverage, width and their saved intervals.
Saved correction policies and paired comparisons remain in the expandable
source evidence and JSON download. The renderer does not invent new confidence
intervals or treat dependent windows as independent trials.

All windows are sorted by trajectory ID/start. Point tables show signed and
absolute errors, separately from membership in the 80% prediction interval.
Plots show every future lead, paired Chronos variants only when present,
persistence and an explicitly declared statistical reference. Origin-centered
plots require a saved observed origin; legacy bundles lacking it offer absolute
values only. Curves display saved future samples; context length is declared,
but legacy past series are not reconstructed from predictions.

The per-window future spread ratio uses population standard deviations
(`ddof=0`); it is undefined for a constant measured future. A smooth median can
have modest MAE while reproducing little fluctuation. Neither a good spread
ratio nor an attractive molecular picture establishes forecast skill.

The application tolerance starts empty: **UNRATED**. Enter a nonnegative value
in the observable's native unit and choose whole-window mean absolute error or
all-points absolute error. Zero is allowed. PASS/FAIL is retrospective
satisfaction of that chosen rule, **not** physical validity, a guarantee about
an unseen future or the frozen scientific acceptance decision. A fraction
error of 0.04 means four percentage points; it is not “96% accurate.”
Thresholds stay separate per panel and persist only during the current page
session. CSV downloads include actual/predicted values, every signed/absolute
error, source identity and the chosen rule. Metrics and evidence also download;
forecast/error plots export as standalone SVG.

## Optional inline molecular reference

`structure_config` points to a `StructureConfig` JSON. The reviewed example is
`configs/experiments/overview-structure.json`: actual **1D4W**, MDbind
MD-A005YY replica-one observed simulation reference. It preserves exact PDB
atom serials for the reviewed protein, 85 ligand heavy atoms, 34 pocket and
26 distal C-alpha atoms. The simulation source merges a peptide ligand into
its final residue; the report describes this instead of inventing a ligand
chemical identity. It is separate from the README's experimental 1HVR image.

Embedding requires `embedding_permitted: true`, a source URI, redistribution
license, description, PDB SHA-256, and exact nonempty atom selections. Protein
and ligand are disjoint; regions must be subsets of protein. Absent or repeated
atom serials and multi-model duplicate serials fail. These are reviewed
memberships, not runtime distance guesses.

The renderer is pinned to **3Dmol.js 2.5.5** and its complete upstream license;
both hashes are enforced in `structure_view.py`. Supply cached files at the
configured paths. If the renderer cache is missing, acquire the versioned
public files once before report generation:

```sh
mkdir -p data/external/overview-viewer44
curl --fail --location https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js \
  --output data/external/overview-viewer44/3Dmol-min.js
curl --fail --location https://raw.githubusercontent.com/3dmol/3Dmol.js/2.5.5/LICENSE \
  --output data/external/overview-viewer44/LICENSE
```

The report verifies these bytes before embedding them. There is no CDN or
coordinate fetch at viewing time. Drag/zoom, reset, region toggles and PNG
export are supported. WebGL failure exposes a static xy projection generated
from the same observed coordinates and exact selections. With no structure
supplied, an explicit scalar-only notice appears. The report never animates
scalar forecasts as invented atomistic motion. Source bytes and generated
HTML/CSV/SVG/PNG stay ignored; only portable metadata and renderer templates
belong in Git.

## Validation

Synthetic pytest cases exercise saved benchmark/follow-up formats, hand-known
values, matching failures, checksum failures, finite/quantile invariants,
escaping, preservation of previous output and exact structure selections.
They provide software evidence, not scientific outcome evidence.

A separate browser check uses optional tooling; it is not a runtime dependency:

```sh
uv run --no-project --python 3.14 --with playwright==1.63.0 \
  python examples/check_overview_browser.py \
  data/reports/offline-overview44/index.html \
  --output data/reports/offline-overview44/browser
```

Use `--chrome` for another installed Chromium executable. The check disables
network access, visits all panels, checks displayed values against embedded
saved values, exercises mean/all-points tolerance rules and CSV/SVG downloads,
exports the molecular PNG, checks a 390 px mobile viewport, then tests a
WebGL-disabled static fallback. The real review used Chrome with software
WebGL: zero page errors and zero remote requests. Installed-wheel generation
outside the source checkout also verifies that HTML/JS templates are packaged.

## Independent confirmation synthesis

The optional `confirmation_summary` object pins `path` and `expected_hash`
(SHA-256). It supplies the three-task frozen inference produced by
[the confirmation workflow](FOLLOWUP_CONFIRMATION.md). The renderer checks
the synthesis checksum, frozen family and source-summary hashes against
configured probability sources before adding criterion decisions, native-unit
MAE comparisons, descriptive uncertainty whiskers and all-grid inference
evidence. The saved prediction bundles retain their original identities.

Three-observable QC failures exclude an entire complex without ID replacement;
actual group counts and reasons remain visible. Development and historical
results remain separate from the independent confirmation panels.
