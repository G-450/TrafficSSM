# Current project state

**Status date:** 2026-08-13
**Delivery state:** Phase 4 evaluation foundation is implemented and validated. The evaluation foundation gate passes. Phase 5 deterministic baseline is next. No model or experimental result exists yet.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- **Phase 1 Data provenance:** Script to download and verify pinned PEMS-BAY dataset, with strict checksum and structural integrity checks.
- **Phase 2 Data-contract hardening:** `st-dssm-validate` command, `st_dssm.validator` for structural checks, `st_dssm.missingness` separating values and masks, and `st_dssm.imputation` for causal forward-filling. Exact missingness statistics (521 zeros, 0 NaNs) were verified against the canonical dataset.
- **Phase 3 Reproducible preprocessing:** `st-dssm-preprocess` command with chronologically disjoint splits, training-only scalar fitting, safe NPZ serialization, and exact deterministic validation.
- **Phase 4 Evaluation foundation:** Model-independent evaluation metrics (`mae`, `rmse`, `gaussian_nll`, `gaussian_crps`, `picp`, `mpiw`), `inverse_transform_predictions` utility, machine-readable JSON result and run-manifest schemas (`MetricRecord`, `RunManifest`), and plotting utilities. Validated with comprehensive synthetic test suites.

## Deliberately removed

The prior Phase 1 loader, graph code, runner, configuration, and recorded preprocessing artefacts were discarded on 2026-08-13 at the project owner's request. They must not be reused as evidence or treated as a baseline. There is currently no implemented data pipeline, model, experiment, test suite, or result.

## Next authorized work

Phase 4 implementation is complete and ready for human review. Once approved, begin [Phase 5 — Deterministic baseline](IMPLEMENTATION_PLAN.md). Baseline and model development remains unauthorized until its prerequisite gates pass.
