# Implementation guidance

Apply the root guidance and normative engineering/scientific documents.
Before changing shared logic, trace all callers and fix the common boundary.
Keep dataset/model adapters independent of canonical contracts.

For data operations, validate manifests and configuration at entry; retain
source identities, units, timestamps, checksums, and feature versions through
output. Numerical hot paths use arrays, not per-frame Pydantic models.
I/O must remain bounded; large hashing and file work must not block an async
event loop. Cancellation or failed validation must never promote partial data.

Add only the modules and CLI commands needed by the current issue. Public
errors use the project exception hierarchy with actionable messages and
chained causes. Expose configuration defaults in the development docs.
