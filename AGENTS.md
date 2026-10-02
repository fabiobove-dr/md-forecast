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

## Repetitive workflows

Repository skills live in `.agents/skills/`:

- `md-issue-delivery`: dependency ordering, acceptance evidence, quality gates,
  scoped PRs, and squash-merge handoff.
- `md-dataset-audit`: upstream metadata/schema/unit/license verification before
  admitting features or writing dataset adapters.

Read the relevant `SKILL.md` before using its workflow. Scoped guidance in
`src/AGENTS.md` and `tests/AGENTS.md` adds implementation and test invariants.
Use `.github/PULL_REQUEST_TEMPLATE.md` to make intent and validation reviewable.

Check GitHub API quota before starting an issue and after substantial remote
work. Treat unavailable billing/model usage as unknown. Finish the current
issue and verify its merge before starting another; if a quota or external
dependency interrupts delivery, record the branch/PR and remaining acceptance
criteria in the handoff. Never close an issue on partial or synthetic evidence
when its acceptance criteria require real-data results.
