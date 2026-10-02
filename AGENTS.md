# Repository guidance

Read `docs/SCIENTIFIC_CONTRACT.md` and `docs/ENGINEERING_STANDARDS.md` before
implementation. Use `docs/IMPLEMENTATION_PLAN.md` for issue order and scope.

- Keep each PR scoped to one issue; describe the problem, resulting behavior,
  and exact validation commands. Complete its acceptance criteria before merge.
- Prefer existing code and the standard library. Do not create empty future
  modules, services, factories, or model integrations.
- Target Python 3.14, use uv, preserve `uv.lock`, and use modern type hints and
  Google-style docstrings. Validate external configuration with Pydantic v2.
- Instantiate settings and configure logging explicitly; imports have no
  environment, filesystem, or logging side effects.
- Never commit dataset bytes, checkpoints, local settings, or generated reports.
  Use tiny synthetic fixtures and reviewed portable provenance manifests.
- Follow split-before-windowing, train-only/context-local preprocessing,
  explicit units, and system-level uncertainty rules in the scientific contract.
- Run the commands in `docs/DEVELOPMENT.md` before merge. CI uses the same
  checks, including a clean wheel install outside the source checkout.

Project documentation belongs here. Personal plans, analyses, and reports for
Fabio belong in `/home/fabio/Documenti/Obsidian Vault/Fabio/Codex/` with filenames
`YYYY-MM-DD - Short Title.md`.
