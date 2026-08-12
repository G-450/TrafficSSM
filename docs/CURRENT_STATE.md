# Current project state

**Status date:** 2026-08-13
**Delivery state:** clean implementation restart; Phase 0 governance is complete and Phase 1 data provenance is next.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an empty open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- **Phase 1 Data provenance:** Script to download and verify pinned PEMS-BAY dataset, with strict checksum and structural integrity checks.

## Deliberately removed

The prior Phase 1 loader, graph code, runner, configuration, and recorded preprocessing artefacts were discarded on 2026-08-13 at the project owner's request. They must not be reused as evidence or treated as a baseline. There is currently no implemented data pipeline, model, experiment, test suite, or result.

## Next authorized work

Phase 1 implementation is complete and ready for human review. Once approved, begin [Phase 2 — Data-contract hardening](IMPLEMENTATION_PLAN.md): implement the accepted `0.0`/`NaN` native-missingness policy, strict five-minute cadence validation, graph/data alignment checks, and tests. Do not build models yet.
