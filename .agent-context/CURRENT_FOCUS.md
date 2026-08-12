# Current focus

Begin the clean implementation with Phase 1 — Data provenance:

1. Download and checksum-verify the ADR-0006 dataset files; write the local dataset manifest.
2. Create only the minimum configuration and manifest tooling required to validate the data source.
3. Open a focused PR with the provenance evidence and tests.

After Phase 1 is accepted, move to Phase 2 data-contract hardening: implement the accepted `0.0`/`NaN` native-missingness policy, strict five-minute cadence validation, graph/data alignment checks, and tests. Do not build models yet.

See `docs/IMPLEMENTATION_PLAN.md` for gates and sequencing, and consult the accepted decision records before making a material design choice. Record any newly discovered ambiguity in `docs/OPEN_QUESTIONS.md`.
