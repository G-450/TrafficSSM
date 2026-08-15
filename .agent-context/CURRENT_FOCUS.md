# Current focus

Phase 6 spatial-temporal encoder is complete (canonical 2-block ST-GCN encoder, causal gated temporal convolutions, Chebyshev graph convolutions, strict causality verification, parameter capacity report, and CLI verification runner) and verified across 155 tests.

The next focus is Phase 7 — Probabilistic forecast head. Do not build transition state-space models yet.

See `docs/IMPLEMENTATION_PLAN.md` for gates and sequencing, and consult the accepted decision records before making a material design choice. Record any newly discovered ambiguity in `docs/OPEN_QUESTIONS.md`.

