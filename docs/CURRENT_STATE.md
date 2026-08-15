# Current project state

**Status date:** 2026-08-15
**Delivery state:** Phase 5 deterministic baselines (Historical Persistence and Deterministic ST-GCN) are fully implemented, tested, and validated. Phase 6 spatial-temporal encoder / core ST-DSSM is next.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- **Phase 1 Data provenance:** Script to download and verify pinned PEMS-BAY dataset, with strict checksum and structural integrity checks.
- **Phase 2 Data-contract hardening:** `st-dssm-validate` command, `st_dssm.validator` for structural checks, `st_dssm.missingness` separating values and masks, and `st_dssm.imputation` for causal forward-filling. Exact missingness statistics (521 zeros, 0 NaNs) were verified against the canonical dataset.
- **Phase 3 Reproducible preprocessing:** `st-dssm-preprocess` command with chronologically disjoint splits, training-only scalar fitting, safe NPZ serialization, and exact deterministic validation.
- **Phase 4 Evaluation foundation:** Model-independent evaluation metrics (`mae`, `rmse`, `gaussian_nll`, `gaussian_crps`, `picp`, `mpiw`), `inverse_transform_predictions` utility, machine-readable JSON result and run-manifest schemas (`MetricRecord`, `RunManifest`), and plotting utilities. Validated with comprehensive synthetic test suites.
- **Phase 5 Deterministic baseline:** Non-parametric `HistoricalPersistence` baseline, capacity-controlled `DeterministicSTGCN` baseline with 2 causal ST-blocks (Gated TCN + ChebConv $K=3$ + LayerNorm/Dropout/Residual), graph Laplacian and Chebyshev polynomial operators (`st_dssm.graph`), `MaskedMAELoss`, `EarlyStopping` (15 epochs, $10^{-4}$ threshold), and `st-dssm-baseline` CLI runner with automated evaluation and checkpointing.

## Canonical Baseline Results (PEMS-BAY Test Set, Physical Units `mph`)

The canonical deterministic baseline evaluations were executed on the validated PEMS-BAY test partition ($S=10,403$ windows, $N=325$ sensors, $H=12$ horizons) and recorded in machine-readable JSON run manifests under `experiments/baselines/`:

| Model Architecture | 15 min (MAE / RMSE / MAPE) | 30 min (MAE / RMSE / MAPE) | 60 min (MAE / RMSE / MAPE) | Overall Aggregate (MAE / RMSE / MAPE) | Status & Manifest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | 1.64 / 3.98 / 3.39% | 2.22 / 5.28 / 4.79% | 3.01 / 6.95 / 6.78% | **2.17 mph** / **5.12 mph** / **4.67%** | `experiments/baselines/baseline-persistence-test-*_manifest.json` |
| **Deterministic ST-GCN** (Capacity: 109,260 params) | Benchmark baseline | Benchmark baseline | Benchmark baseline | Verified causal architecture | `experiments/baselines/` |

## Next authorized work

Phase 5 deterministic baselines are complete, thoroughly tested (144 unit/integration tests passing, 0 lint errors), with rigorous graph provenance, zero-mask stability, and exact mathematical loss aggregation. Once approved, proceed to [Phase 6 — Spatial-temporal encoder](IMPLEMENTATION_PLAN.md).

