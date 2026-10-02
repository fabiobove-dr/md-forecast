---
name: md-issue-delivery
description: Deliver md-forecast GitHub issues in dependency order with acceptance evidence, scoped PRs, quality checks, and an authorized squash merge. Use for implementing repository issues, not standalone explanations or reviews.
---

# Issue delivery

1. Inspect the worktree, current branch, remote main, open PRs, and the issue
   body. Preserve existing edits. Read its dependencies and the repository
   [standards](../../../docs/ENGINEERING_STANDARDS.md) and
   [scientific contract](../../../docs/SCIENTIFIC_CONTRACT.md). Choose the next
   issue whose dependencies are actually satisfied; issue number alone is not
   sufficient. Carry forward user restrictions and authorized merge scope.
2. Check `gh api rate_limit`. Unavailable account/assistant usage is unknown,
   not an assumed balance. Before expensive acquisition/experiments inspect
   artifact size, available disk, hardware, and required runtime. If resources
   cannot support acceptance, complete independent work and report the gap.
3. Create one scoped branch from synchronized main. Derive required files,
   behavior, tests, docs, and real-data evidence from the issue. Trace existing
   paths before adding code; use existing helpers or the standard library.
   New dependencies must have a concrete use and a Python compatibility check.
4. Implement and verify each acceptance criterion. Run the quality gates and
   wheel-install smoke test in [DEVELOPMENT.md](../../../docs/DEVELOPMENT.md).
   For documentation/skill changes, validate local links and skill frontmatter;
   rely on CI for the existing runtime gates. Separate actual experiment
   evidence from synthetic tests and planned functionality.
5. Review the diff, stage only scoped files, commit, push, and create the PR
   using [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md).
   Include exact validation commands/results and any remaining limitations.
   Use `Closes #N` only when all issue acceptance criteria are met. Supply PR
   bodies through a file or structured argument to preserve literal newlines.
6. Wait for checks on the current PR head; resolve failures and recheck the
   changed head. When the user has authorized merging, squash with
   `gh pr merge NUMBER --squash --match-head-commit FULL_SHA`. Verify merged
   status and issue closure, fast-forward local main, check post-merge CI and
   a clean worktree, then proceed to the next satisfied dependency.

If interrupted, inspect the existing branch, PR head/checks, and process handles
before repeating a mutation or expensive job. Record completed evidence and
missing criteria in the handoff; do not call incomplete work complete or start
another issue simply to bypass a blocker. Permission, unavailable compute,
dataset restrictions, and unresolved scientific assumptions need concrete
evidence and a concise user decision when independent progress is exhausted.
