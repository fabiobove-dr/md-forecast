---
name: md-dataset-audit
description: Audit MISATO or MDbind source metadata, scientific schema, units, timestamps, splits, and usage terms before admitting features or implementing adapters. Use for dataset audits and verifying changed dataset assumptions, not routine model training.
---

# Dataset audit

Read the target issue and [scientific contract](../../../docs/SCIENTIFIC_CONTRACT.md).
Use primary evidence: immutable dataset record/API, paper, and pinned upstream
code. Record retrieval date, DOI/version, artifact URLs, exact sizes/checksums,
and upstream revision. Distinguish record license, code license, and restrictions
on constituent structures/ligands; public download access alone proves no
redistribution right. Do not invent legal conclusions from a missing notice.

Inspect a real representative artifact, bounded subset, or remote schema when
possible. Avoid downloading a full large artifact merely to inspect metadata.
For each candidate channel record exact key, shape, dtype, units, frame
alignment, time-resolved versus aggregate status, missingness, and evidence.
Separate directly observed values from author descriptions and inferences.
Unknown or conflicting definitions block feature admission; leave them explicit.

Check that stored sample timestamps/spacing match the actual export code and
paper. Duration, interval, sample count, and endpoint conventions are distinct:
do not divide a reported duration by frame count and claim verified timestamps.
Verify official split files, identity grouping, disjointness, and coverage.
Do not conflate a held-out replica with an unseen complex.

Write the requested project dataset documentation and portable configuration
under `docs/datasets/` and `configs/datasets/`. Put citations next to claims;
configuration must carry exact provenance and only verified feature definitions.
Source bytes stay in ignored local data. Personal research notes belong in
Fabio's Obsidian vault according to root AGENTS.md.

Validate machine-readable configuration, local documentation links, and strict
MkDocs rendering. Mark missing source/schema evidence as unresolved rather
than treating a mock fixture, upstream sample, or guessed unit as exhaustive
dataset verification. List the evidence required to resolve each gap before
dependent implementations consume it.
