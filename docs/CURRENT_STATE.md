# Current project state

**Status date:** 2026-08-13
**Delivery state:** clean implementation restart; Phase 0 governance is complete and Phase 1 data provenance is next.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an empty open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- Empty package/config/test directory scaffolding ready for implementation.

## Deliberately removed

The prior Phase 1 loader, graph code, runner, configuration, and recorded preprocessing artefacts were discarded on 2026-08-13 at the project owner's request. They must not be reused as evidence or treated as a baseline. There is currently no implemented data pipeline, model, experiment, test suite, or result.

## Next authorized work

Begin [Phase 1 — Data provenance](IMPLEMENTATION_PLAN.md): obtain the ADR-0006 dataset files, verify their checksums, and create the dataset manifest. Then progress through Phase 2 data-contract hardening and Phase 3 reproducible preprocessing. Do not start model development before those gates pass.
