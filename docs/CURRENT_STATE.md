# Current project state

**Status date:** 2026-09-21
**Delivery state:** Phase 9 training operations (`st-dssm-train` CLI) is fully implemented, tested, and validated. Phase 10 normal-condition study is next.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- **Phase 1 Data provenance:** Script to download and verify pinned PEMS-BAY dataset, with strict checksum and structural integrity checks.
- **Phase 2 Data-contract hardening:** `st-dssm-validate` command, `st_dssm.validator` for structural checks, `st_dssm.missingness` separating values and masks, and `st_dssm.imputation` for causal forward-filling. Exact missingness statistics (521 zeros, 0 NaNs) were verified against the canonical dataset.
- **Phase 3 Reproducible preprocessing:** `st-dssm-preprocess` command with chronologically disjoint splits, training-only scalar fitting, safe NPZ serialization, and exact deterministic validation.
- **Phase 4 Evaluation foundation:** Model-independent evaluation metrics (`mae`, `rmse`, `gaussian_nll`, `gaussian_crps`, `picp`, `mpiw`), `inverse_transform_predictions` utility, machine-readable JSON result and run-manifest schemas (`MetricRecord`, `RunManifest`), and plotting utilities. Validated with comprehensive synthetic test suites.
- **Phase 5 Deterministic baseline:** Non-parametric `HistoricalPersistence` baseline, capacity-controlled `DeterministicSTGCN` baseline with 2 causal ST-blocks (Gated TCN + ChebConv $K=3$ + LayerNorm/Dropout/Residual), graph Laplacian and Chebyshev polynomial operators (`st_dssm.graph`), `MaskedMAELoss`, `EarlyStopping` (15 epochs, $10^{-4}$ threshold), and `st-dssm-baseline` CLI runner with automated evaluation and checkpointing.
- **Phase 6 Spatial-temporal encoder:** Canonical 2-block ST-GCN encoder (`st_dssm.encoder.SpatialTemporalEncoder`), `CausalGatedTemporalConv` with strict causal left-padding, `SpatialTemporalBlock` with Chebyshev graph convolutions ($K=3$, BLAS-accelerated), channel-only LayerNorm, Dropout ($0.1$), residual projections, and 64-dimensional context projection (`[B, 12, 325, 64]`). Validated with mathematical causality probes (zero future leakage), node-permutation equivariance tests, single-batch optimization tests, capacity reports (104,320 parameters), and the `st-dssm-encoder` CLI runner on the canonical PEMS-BAY test partition (`experiments/encoder/encoder_pems_bay_canonical_manifest.json`).
- **Phase 7 Probabilistic forecast head:** Autoregressive 12-step Gaussian decoder (`st_dssm.forecast_head.GaussianForecastHead`). Per ADR-0007: $\sigma = \text{softplus}(\text{clamp}(\log\sigma_{\text{raw}}, -8, 5)) + 10^{-4}$ strictly positive variance; learned per-horizon position embeddings; MLP decoder over $[z, c, h_{\text{emb}}, \mu_{\text{prev}}]$; stochastic teacher forcing (ratio decays $1.0 \to 0.0$ over first 50\% of training epochs); reparameterized sample prediction. Validated with sigma-positivity probes, log-sigma clamp boundary tests, teacher-forcing vs.\ autoregressive path tests, gradient flow checks, single-batch overfit convergence, and sampling distribution mean convergence.
- **Phase 8 DSSM integration:** Full `st_dssm.dssm.GaussianDSSM` wrapping Phase 6 encoder + Phase 7 forecast head. Per ADR-0007: 32-dimensional diagonal Gaussian latent state per sensor; causal GRU prior transition $p(z_t \mid z_{t-1}, c_t)$; bidirectional GRU recognition (posterior) network $q(z_t \mid \text{context}_{1:L}, x, m)$ used only during training; closed-form KL divergence; masked Gaussian reconstruction NLL; ELBO $= \text{NLL} + \beta \cdot \text{KL}$ with $\beta$ annealing from 0 to 1 over first 20 epochs (caller-managed); hard non-finite ELBO guard; `forward_predict` samples exclusively from prior with no target access. `st-dssm-dssm` CLI with synthetic smoke test. Capacity report breaks down parameters per sub-module.
- **Phase 9 Training operations:** `src/st_dssm/cli/train.py` (`st-dssm-train`) — YAML-configuration-driven training loop consuming `GaussianDSSM.forward_train`; checkpoint selection on validation Gaussian NLL (computed from `forward_predict`, not training ELBO) via `EarlyStopping`; beta-annealing (0→1 over 20 epochs) and teacher-forcing decay (1.0→0.0 over 50% of max epochs) schedules; canonical seeding via `set_seed`; `--device cpu|cuda|mps` handling with CUDA auto-detection; `--resume` checkpoint resumption; gradient-norm clipping; graph checksum verification against the pinned canonical adjacency before training; post-training evaluation on the configured split with a full `RunManifest` written via `st_dssm.cli.evaluate.run_evaluation`. `configs/train.yaml` holds the canonical model/training/schedule/seed configuration. Hardened after real-data validation (see `phase9/fix-train-cli-windows-and-config-precedence`). Validated with `tests/test_train_cli.py`.

## Canonical Baseline Results (PEMS-BAY Test Set, Physical Units `mph`)

The canonical deterministic baseline evaluations were executed on the validated PEMS-BAY test partition ($S=10,403$ windows, $N=325$ sensors, $H=12$ horizons) and recorded in machine-readable JSON run manifests under `experiments/baselines/`:

| Model Architecture | 15 min (MAE / RMSE / MAPE) | 30 min (MAE / RMSE / MAPE) | 60 min (MAE / RMSE / MAPE) | Overall Aggregate (MAE / RMSE / MAPE) | Status & Manifest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | 1.64 / 3.98 / 3.39% | 2.22 / 5.28 / 4.79% | 3.01 / 6.95 / 6.78% | **2.17 mph** / **5.12 mph** / **4.67%** | `experiments/baselines/baseline-persistence-test-*_manifest.json` |
| **Deterministic ST-GCN** (Capacity: 109,260 params) | Benchmark baseline | Benchmark baseline | Benchmark baseline | Verified causal architecture | `experiments/baselines/` |

## Canonical Encoder Evidence (PEMS-BAY Test Set)

The canonical spatial-temporal encoder was verified across all 10,403 test samples of PEMS-BAY and recorded in `experiments/encoder/encoder_pems_bay_canonical_manifest.json`:
- **Model Name:** `spatial_temporal_encoder`
- **Trainable Parameters:** `104,320`
- **Topology:** $N=325$ nodes, $L=12$ steps, $C_{\text{in}}=2$ channels, $C_{\text{out}}=64$ context dimensions.
- **Symmetrization Protocol:** Explicitly authorized per ADR-0009 ($W_{\text{sym}} = \frac{1}{2}(W + W^T)$).

## Test suite summary

| Phase | Tests | Status |
| :--- | :--- | :--- |
| Phase 1–5 (data, baselines, evaluation) | 103 | ✅ Passing |
| Phase 6 (encoder) | 14 | ✅ Passing |
| Phase 7 (forecast head) | 25 | ✅ Passing |
| Phase 8 (DSSM) | 30 | ✅ Passing |
| Phase 9 (training CLI) | 2 | ✅ Passing |
| **Total** | **215** | **✅ All passing** (verified 2026-09-21, full local reinstall + `pytest`) |

*Note: `test_encoder_cli_unrecognized_graph_and_missing_metadata` requires the canonical `data/processed` artifact to be present on disk; it is skipped in environments without PEMS-BAY data (pre-existing condition, not caused by Phase 7/8 changes).*

## Next authorized work

Phase 9 training operations is complete and tested. [Phase 10 — Normal-condition study](IMPLEMENTATION_PLAN.md) is next:
- Tune hyperparameters against validation NLL only, with a predefined search record (`scripts/tune_phase10.py`, `scripts/run_phase10.py` are scaffolded — see `phase9/fix-train-cli-windows-and-config-precedence` for hardening and a real-data timing measurement showing this is a multi-day compute commitment on modest hardware)
- Lock the winning configuration (`configs/tuned.yaml`) and evaluate the test set once per canonical seed (2026, 2027, 2028)
- Compare against the Phase 5 deterministic baseline at 0% masking
- Produce reproducible locked test tables and figures in `experiments/normal/`

