# Development

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from
a fresh clone run:

```sh
uv sync --locked
```

uv provisions Python **3.14** from `.python-version` and installs the package
plus the default `dev` and `docs` dependency groups from the committed lockfile.
The project deliberately supports the 3.14 minor series only at this stage.
CI pins the uv release used to validate the lockfile; use that release or newer
locally. No system Python upgrade is required.

Core dependencies are Pydantic v2, pydantic-settings, HTTPX, PyYAML, NumPy,
PyArrow, and h5py. NumPy/Arrow are runtime dependencies of canonical series storage;
h5py is used by the MISATO adapter. Their Python 3.14 wheels are verified by
CI and the outside-checkout wheel
smoke test. `dev` contains lint,
typing, testing, coverage, complexity, and pre-commit tooling; `docs` contains
MkDocs. Additional analysis and ML/GPU groups are deferred until their implementing
issues establish dependencies and Python compatibility. No Chronos or GPU
package is installed by the scaffold.

For a runtime-only environment, use `uv sync --locked --no-default-groups`.
Use `uv add` / `uv add --group <group>` for deliberate dependency updates and
commit both `pyproject.toml` and `uv.lock`. Normal checks use `--locked` to
prevent silent dependency changes.

## Package and configuration

The layout is intentionally shallow:

```text
src/md_forecast/
├── __init__.py       # authoritative version, also used by Hatchling
├── cli/__init__.py   # argparse help/version and acquisition
├── data/
│   ├── acquisition.py # bounded, validated artifact acquisition
│   ├── schemas.py     # versioned metadata boundaries
│   ├── registry.py    # trajectory-level JSON index
│   ├── series.py      # vector-validated Arrow/Parquet trajectories
│   ├── splits.py      # official/grouped pre-window partitions
│   ├── windows.py     # lazy indices and physical-grid validation
│   ├── preprocessing.py # scoped, provenance-bearing scaler statistics
│   ├── artifacts.py   # atomic, hashed experiment metadata
│   └── public/misato.py # lazy native-observable extraction
└── core/
    ├── config.py     # Pydantic settings
    ├── constants.py  # stable typed defaults
    ├── exceptions.py # shared domain exception base
    └── logging.py    # explicit standard logging setup
```

Construct `md_forecast.core.config.Settings()` at an application boundary.
Settings do not create directories or load `.env` automatically. Python
keyword arguments override environment variables, which override defaults.
An application may explicitly pass `_env_file` when dotenv loading is wanted.

| Setting | Environment variable | Default |
| --- | --- | --- |
| `data_dir` | `MD_FORECAST_DATA_DIR` | `data`, relative to the working directory |
| `log_level` | `MD_FORECAST_LOG_LEVEL` | `INFO` |

Log levels are validated against `DEBUG`, `INFO`, `WARNING`, `ERROR`, and
`CRITICAL`. Unknown explicit setting fields raise validation errors. Call
`configure_logging(settings.log_level)` when starting an application; it keeps
existing root handlers. Imports do not instantiate settings or configure logs.
Dataset/model-specific settings arrive with their respective implementations.

```sh
uv run md-forecast --help
uv run md-forecast --version
```

Download the small official MISATO split lists, or explicitly include the
132.8 GB MD artifact:

```sh
uv run md-forecast download
uv run md-forecast download --mode md
uv run md-forecast download --destination data/external/misato
```

Run from the checkout root, or pass `--source /path/to/misato.yaml`; the default
audited source config is `configs/datasets/misato.yaml`. Installed wheels do
not bundle this versioned scientific configuration. `metadata` mode downloads
split lists only; `md` downloads the same lists plus `MD.hdf5`. QM, densities,
and restart/topology archives are excluded from both modes.

`DownloadSettings` extends the environment-backed settings. The following
values are configurable through `MD_FORECAST_` variables; explicit CLI
destination takes precedence:

| Setting | Default |
| --- | --- |
| `destination` | `data_dir / external/misato` |
| `concurrency` | 2 complete artifact operations |
| `chunk_bytes` | 1 MiB |
| `disk_reserve_bytes` | 64 MiB beyond remaining payload bytes |
| `connect_timeout` | 10 seconds |
| `read_timeout` | 60 seconds per received chunk |
| `write_timeout` | 30 seconds |
| `pool_timeout` | 10 seconds |
| `retries` | 2 after the first HTTP failure |
| `retry_delay` | 1 second, multiplied by the retry number |

Timeouts must be positive, concurrency/chunks must be positive integers, and
retry counts/reserves cannot be negative. URL, filename, size, and supported
MD5/SHA-256 checksum contracts are validated before acquisition. A POSIX file
lock prevents concurrent acquisition processes from sharing a destination;
this implementation targets Linux/POSIX, as does the current CI.

Acquisition checks available disk for all selected remaining payloads plus
the reserve. Downloads use `<filename>.part`; HTTP 206 resumes must report
the exact requested byte range and full source size. A 200 response to a
range request restarts the partial file. Invalid ranges or encoded responses
fail before writing. Transport failures retry within the configured limit.
Oversized, truncated, or checksum-invalid data are never published as final
artifacts. An existing final file is revalidated and reused without HTTP;
invalid existing files are preserved with an actionable error.

File writes, fsync, hashing, finalization, and receipt publication run in
worker threads. Cancellation waits for an in-flight file operation before
closing its handle, closes HTTP streams, cancels sibling jobs, and keeps only
restartable partial bytes. The OS releases the destination lock; its empty
lock file may remain. No failed partial file receives a success receipt.

After full size/checksum validation, atomic rename publishes the final path.
Each verified artifact gets `<filename>.receipt.json` containing source record,
version, URL, filename, checksum, byte count, and verification time, with no
absolute machine paths. The receipt is also atomically written. Re-running
updates its verification time. If receipt publication fails after a valid
artifact is finalized, retrying validates the file and repairs the receipt.
Move an invalid `.part`/existing file aside before retrying; the downloader
does not delete potentially valuable local data.

No training or forecasting commands are implemented yet. See the
[MISATO audit](datasets/MISATO.md) for scientific timing and license limitations,
and [canonical schema](datasets/CANONICAL_SCHEMA.md) for registry/storage APIs.
The [MISATO adapter](datasets/MISATO_ADAPTER.md) documents selected-system
extraction, real-sample verification, QC outputs, and failure safety. Run
`uv run --locked md-forecast extract-misato --help` for required local paths and
selection arguments. It does not download HDF5, infer missing physical time,
or install a model dependency.
The [data preparation contract](datasets/PREPARATION.md) describes split/window
configuration, scoped scaling, stable metadata hashes, and experiment linkage.

## Quality gates

Run the same checks as CI from the repository root:

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest
uv run --locked xenon --max-absolute A --max-modules A --max-average A src
uv build --no-sources
uv run --locked mkdocs build --strict
```

`uv run pytest` also works. Pytest emits branch coverage and enforces a 90%
floor for the initial package. Raise or adjust that guardrail only with a
documented rationale; meaningful critical-path coverage matters more than
the number. Tests use synthetic configuration inputs and no external MD data.
Xenon rejects source blocks or module averages above complexity grade A.

Verify wheel installation outside the checkout, so editable imports cannot
hide missing package files:

```sh
uv venv /tmp/md-forecast-wheel --python 3.14
uv pip install --python /tmp/md-forecast-wheel/bin/python dist/*.whl
cd /tmp
/tmp/md-forecast-wheel/bin/python -c 'from md_forecast import __version__; from md_forecast.core.config import Settings; assert __version__; Settings()'
/tmp/md-forecast-wheel/bin/md-forecast --version
/tmp/md-forecast-wheel/bin/md-forecast --help
```

Use a new temporary environment path if that path already contains unrelated
work. Build outputs under `dist/` and the MkDocs site under `site/` are ignored.

Install optional fast local hooks:

```sh
uv run pre-commit install
uv run pre-commit run --all-files
```

Hooks run the locked Ruff and mypy tools. The full test/build/docs gates run
in CI and must pass before merge. A single CPU job avoids duplicate checks;
no datasets or GPU jobs are needed. Preview documentation with
`uv run mkdocs serve`.

The ignored `data/` tree holds local source/cache/processed data; only its
README and reviewed small JSON/TOML files under `data/manifests/` are eligible
for commits. Common trajectory and checkpoint formats are also ignored
globally. Secrets, generated reports, and local environments stay untracked.
