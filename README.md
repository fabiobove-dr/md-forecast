# md-forecast
Forecasting Molecular Dynamics Observables from Partial Trajectories with Time-Series Foundation Models

Start with the [scientific contract](docs/SCIENTIFIC_CONTRACT.md),
[implementation plan](docs/IMPLEMENTATION_PLAN.md), and
[engineering standards](docs/ENGINEERING_STANDARDS.md).

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv sync --locked
uv run md-forecast --version
uv run pytest
```

Python 3.14, the package, and development/documentation tools are provisioned
by the first command. See [development guidance](docs/DEVELOPMENT.md) for
configuration, all quality checks, and the documentation site.

Dataset ingestion and forecasting models are not implemented yet.
