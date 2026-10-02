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

Core dependencies are Pydantic v2 and pydantic-settings. `dev` contains lint,
typing, testing, coverage, complexity, and pre-commit tooling; `docs` contains
MkDocs. Data/analysis and ML/GPU groups are deferred until their implementing
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
├── cli/__init__.py   # argparse help/version only
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

There are no download, training, or forecasting commands yet.

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
