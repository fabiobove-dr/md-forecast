# Engineering Standards

These rules are normative for `md-forecast`. Prefer the simplest implementation that satisfies them.

## Principles

- **KISS:** prefer direct, readable implementations over clever abstractions.
- **YAGNI:** do not add extensibility until a real use case requires it.
- **Boy Scout Rule:** leave touched code cleaner than it was.
- Keep modules small, single-purpose, typed, and easy to test.
- Avoid hidden import side effects, mutable globals, circular dependencies, and framework-specific coupling in the domain/data contracts.
- Preserve scientific reproducibility over convenience.

## Python

- Target **Python 3.14** unless a required scientific dependency proves incompatible; any downgrade must be documented and justified.
- Follow current PEP conventions and idiomatic Python.
- Use modern type hints everywhere on public and internal interfaces.
- Use **Google-style docstrings** for public modules, classes, methods, and functions where behavior is not self-evident.
- Prefer explicit names over abbreviations.

## Packaging and dependency management

- Use `pyproject.toml` with PEP 621 metadata.
- Use **uv** and commit `uv.lock`.
- Prefer **Hatchling** as the build backend.
- Separate core, dev/test, docs, data/analysis, and ML/GPU dependencies where practical.
- Do not make Conda the primary development workflow.
- Pin experiment-critical package/model revisions through the lockfile/config/manifest.

## Pydantic and data contracts

- Use **Pydantic v2** for configuration, external inputs, manifests, structured results, and serialization boundaries.
- Use `pydantic-settings` for environment-backed settings.
- Do not instantiate Pydantic models for every frame/row in million-row numerical hot paths. Validate the boundary once, then use typed NumPy/PyArrow/Pandas structures internally.
- Use frozen typed dataclasses only for small immutable internal value objects when Pydantic validation/serialization is unnecessary.
- Enums should represent closed vocabularies such as split names, dataset IDs, model IDs, and units where appropriate.

## Configuration, constants, and magic values

There must be no unexplained magic numbers or strings inside business/scientific logic.

- **User-tunable values** belong in typed Pydantic settings/config models and versioned YAML/TOML configuration.
- **Semantic invariants and stable defaults** belong in `core/constants.py` or typed enums.
- Dataset field names/keys that are part of an adapter contract should be centralized in that adapter or a constants module.
- Never duplicate thresholds, model IDs, paths, time units, feature names, or metric names across modules.
- Secrets and machine-specific paths must come from environment/config, never source code.

## Async policy

Use async where it provides real concurrency:

- HTTP/network access;
- downloads;
- subprocess orchestration;
- filesystem orchestration when beneficial;
- long-running workflow coordination.

Prefer `httpx.AsyncClient`, explicit timeouts, bounded concurrency, and cancellation-safe code for network operations.

Do **not** make CPU/GPU-bound numerical code async merely for stylistic consistency. NumPy, PyArrow, h5py, statsmodels, PyTorch inference/training, and MDAnalysis work should remain synchronous in their computational core. When blocking I/O must be invoked from an async orchestration layer, isolate it explicitly (for example with `asyncio.to_thread`) rather than hiding blocking work in an `async def`.

This rule prevents "fake async" and event-loop blocking.

## Architecture

Keep the architecture shallow. Initial target:

```text
src/md_forecast/
├── core/
│   ├── config.py
│   ├── constants.py
│   ├── exceptions.py
│   └── logging.py
├── data/
│   ├── schemas.py
│   ├── registry.py
│   ├── splits.py
│   ├── windows.py
│   └── public/
│       ├── misato.py
│       └── mdbind.py
├── features/
├── models/
├── evaluation/
├── analysis/
└── cli/
```

Add `services/`, repositories, ports, factories, or dependency-injection layers only when concrete duplication or replacement requirements justify them.

External systems/datasets/models should be behind small adapters. Domain/data schemas must not depend on Chronos internals.

## Errors and logging

- Define a small custom exception hierarchy in `core/exceptions.py`.
- Raise domain-specific errors with actionable messages.
- Preserve underlying exceptions with exception chaining.
- User-facing errors should be concise; logs may contain technical detail.
- Use standard Python logging with structured/contextual fields where useful.
- Never use `print()` for library/runtime diagnostics.

## Scientific correctness

- Units are explicit and validated.
- Time axes are monotonic and carry physical units.
- No future information may enter preprocessing of a forecast context.
- Split before windowing.
- Fit normalization/scalers on training data only unless the transformation is explicitly context-local.
- Treat overlapping windows from one trajectory as correlated observations.
- Report uncertainty at the **trajectory/system level**, not as if windows were independent.
- Any cross-dataset comparison must use semantically equivalent feature definitions.
- Record dataset version, source checksum, split hash, feature-set version, model revision, config hash, code commit, seeds, and hardware for published experiments.

## Testing

Use `pytest`.

Required test layers:

- unit tests for pure logic;
- schema/config validation tests;
- adapter tests using tiny fixtures;
- leakage/split invariant tests;
- deterministic numerical regression tests where appropriate;
- install/import smoke test;
- CLI smoke tests once a CLI exists.

Do not use full public datasets in CI.

Coverage is a guardrail, not a target to game. Critical scientific/data-contract paths must be covered.

## Quality gates

Before merge:

- Ruff format/check;
- mypy;
- pytest;
- coverage report;
- Xenon complexity check;
- build/install smoke test;
- MkDocs strict build.

Use pre-commit to run fast local checks.

## Documentation

Use **MkDocs** with a minimal documentation stack.

Documentation should include:

- project/scientific contract;
- architecture;
- engineering standards;
- dataset provenance and schemas;
- configuration reference;
- reproducible experiment workflow;
- model/evaluation contracts;
- development commands.

Prefer Mermaid for simple architecture/data-flow diagrams. Avoid manually maintained duplicate documentation.

## Performance

- Stream/chunk large HDF5/Parquet data; do not load entire datasets into RAM without an explicit reason.
- Measure before optimizing.
- Keep data preprocessing deterministic and cacheable.
- Avoid per-row Python/Pydantic overhead in large numerical tables.
- GPU code must record peak VRAM and throughput.
- Optimize I/O only after profiling confirms it is a bottleneck.

## Pull requests

Each implementation PR should:

- be scoped to one issue or one coherent change;
- avoid opportunistic rewrites unrelated to the issue;
- update tests and docs with behavior changes;
- preserve backward-compatible contracts unless a deliberate migration is documented;
- include exact commands used to validate the change.
