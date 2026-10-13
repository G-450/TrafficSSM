# Superseded condition runs

Runs kept for the record but excluded from the Phase 12 aggregate. The
aggregator globs `experiments/missingness/*_manifest.json` at the top level
only, so nothing in this directory enters a locked table.

Nothing here is deleted: these are real runs, and the reason each one was
replaced is part of the study's history.

## `baseline-persistence-test-node20-seed2026-20261013-110858`

The Phase 11 mechanism-check run, reused as the 20%/seed-2026 persistence
condition when the Phase 12 matrix was first executed. It was produced from an
uncommitted working tree (`git_dirty: true`) at commit `17df01d`, which predates
the Phase 12 code, so it is not traceable to a committed revision and the
aggregator's provenance gate now rejects it.

It was replaced by `baseline-persistence-test-node20-seed2026-20261013-201920`,
run from a committed tree at `16895ef`. The two agree on all 117 metrics to
within 1e-12 and carry the same mask checksum, so the published figure never
changed — only its provenance did.
