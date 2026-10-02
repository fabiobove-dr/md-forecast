# Test guidance

Use small synthetic fixtures and local/mock HTTP sources. External dataset
bytes and network/GPU access do not belong in the default pytest/CI suite.
Temporary files belong under pytest's `tmp_path`.

Test the issue's failure modes as well as success: invalid metadata/units,
future leakage, dependent split groups, impossible windows, interrupted or
corrupt acquisition, and deterministic numerical outputs where applicable.
Prefer assertions on behavior over implementation details or duplicated
algorithms. Distinguish synthetic regression tests from real-data experiments;
passing the former does not prove a scientific acceptance criterion.

Keep the coverage guardrail meaningful. Do not remove critical-path checks or
lower thresholds merely to obtain a passing PR. Installed-package smoke tests
must run outside the source checkout, as documented in DEVELOPMENT.md.
