# Current project state

**Status date:** 2026-10-07
**Delivery state:** Phase 9 training operations is implemented and tested (synthetic smoke + unit tests; real-data run starts in Phase 10). Phase 10 normal-condition study is next.

## Implemented foundation

- Canonical project documentation, accepted architecture/protocol decisions, and an open-question register.
- Durable AI-agent context under `.agent-context/`.
- Professional directory contract, packaging metadata, Git workflow, and GitHub review templates.
- **Phase 1 Data provenance:** Script to download and verify pinned PEMS-BAY dataset, with strict checksum and structural integrity checks.
- **Phase 2 Data-contract hardening:** `st-dssm-validate` command, `st_dssm.validator` for structural checks, `st_dssm.missingness` separating values and masks, and `st_dssm.imputation` for causal forward-filling. Exact missingness statistics (521 zeros, 0 NaNs) were verified against the canonical dataset.
- **Phase 3 Reproducible preprocessing:** `st-dssm-preprocess` command with chronologically disjoint splits, training-only scalar fitting, safe NPZ serialization, and exact deterministic validation. The canonical artifact was regenerated on 2026-10-10 in the environment used for all Phase 10+ experiments (Python 3.12.10, Windows 11); `data/processed/processed_metadata.json` records its checksums. Values differ from the 2026-08-15 artifact by at most 1.5e-5 (float32 rounding across Python/NumPy versions), but checksums are environment-specific, so regenerate with `st-dssm-preprocess generate --force` on a new machine.
- **Phase 4 Evaluation foundation:** Model-independent evaluation metrics (`mae`, `rmse`, `gaussian_nll`, `gaussian_crps`, `picp`, `mpiw`), `inverse_transform_predictions` utility, machine-readable JSON result and run-manifest schemas (`MetricRecord`, `RunManifest`), and plotting utilities. Validated with comprehensive synthetic test suites.
- **Phase 5 Deterministic baseline:** Non-parametric `HistoricalPersistence` baseline, capacity-controlled `DeterministicSTGCN` baseline with 2 causal ST-blocks (Gated TCN + ChebConv $K=3$ + LayerNorm/Dropout/Residual), graph Laplacian and Chebyshev polynomial operators (`st_dssm.graph`), `MaskedMAELoss`, `EarlyStopping` (15 epochs, $10^{-4}$ threshold), and `st-dssm-baseline` CLI runner with automated evaluation and checkpointing.
- **Phase 6 Spatial-temporal encoder:** Canonical 2-block ST-GCN encoder (`st_dssm.encoder.SpatialTemporalEncoder`), `CausalGatedTemporalConv` with strict causal left-padding, `SpatialTemporalBlock` with Chebyshev graph convolutions ($K=3$, BLAS-accelerated), channel-only LayerNorm, Dropout ($0.1$), residual projections, and 64-dimensional context projection (`[B, 12, 325, 64]`). Validated with mathematical causality probes (zero future leakage), node-permutation equivariance tests, single-batch optimization tests, capacity reports (104,320 parameters), and the `st-dssm-encoder` CLI runner on the canonical PEMS-BAY test partition (`experiments/encoder/encoder_pems_bay_canonical_manifest.json`).
- **Phase 7 Probabilistic forecast head:** Autoregressive 12-step Gaussian decoder (`st_dssm.forecast_head.GaussianForecastHead`). Per ADR-0007: $\sigma = \text{softplus}(\text{clamp}(\log\sigma_{\text{raw}}, -8, 5)) + 10^{-4}$ strictly positive variance; learned per-horizon position embeddings; MLP decoder over $[z, c, h_{\text{emb}}, \mu_{\text{prev}}]$; stochastic teacher forcing (ratio decays $1.0 \to 0.0$ over first 50\% of training epochs); reparameterized sample prediction. Validated with sigma-positivity probes, log-sigma clamp boundary tests, teacher-forcing vs.\ autoregressive path tests, gradient flow checks, single-batch overfit convergence, and sampling distribution mean convergence.
- **Phase 8 DSSM integration:** Full `st_dssm.dssm.GaussianDSSM` wrapping Phase 6 encoder + Phase 7 forecast head. Per ADR-0007: 32-dimensional diagonal Gaussian latent state per sensor; causal GRU prior transition $p(z_t \mid z_{t-1}, c_t)$; bidirectional GRU recognition (posterior) network $q(z_t \mid \text{context}_{1:L}, x, m)$ used only during training; closed-form KL divergence; masked Gaussian reconstruction NLL; ELBO $= \text{NLL} + \beta \cdot \text{KL}$ with $\beta$ annealing from 0 to 1 over first 20 epochs (caller-managed); hard non-finite ELBO guard; `forward_predict` samples exclusively from prior with no target access. `st-dssm-dssm` CLI with synthetic smoke test. Capacity report breaks down parameters per sub-module.

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
| Phase 1–5 (data, baselines, evaluation) | 137 | ✅ Passing |
| Phase 6 (encoder + graph) | 24 | ✅ Passing |
| Phase 7 (forecast head) | 25 | ✅ Passing |
| Phase 8 (DSSM) | 30 | ✅ Passing |
| Phase 9 (training CLI) | 9 | ✅ Passing |
| Phase 11 (missingness mechanism) | 15 | ✅ Passing |
| **Total** | **240** | **✅ All passing** |

*Note: `test_encoder_cli_unrecognized_graph_and_missing_metadata` requires the canonical `data/processed` artifact to be present on disk; it is skipped in environments without PEMS-BAY data (pre-existing condition, not caused by Phase 7/8 changes).*

## Next authorized work

Phase 9 Training operations is complete, with the `st-dssm-train` CLI and its tests merged. Once approved, proceed to [Phase 10 — Normal-condition study](IMPLEMENTATION_PLAN.md):
- Tune ST-DSSM hyperparameters against validation NLL only, with predefined search record
- Lock configuration, evaluate test set once per seed (2026, 2027, 2028)
- Compare with Phase 5 deterministic baseline at 0% masking
- Produce locked test tables and figures in `experiments/normal/`

*Note: `scripts/tune_phase10.py` and `scripts/run_phase10.py` are scaffolded but have not been executed end-to-end.*

*Phase 9 hardening (#12): fixed a Windows console crash in `st-dssm-train` and CLI/config
precedence, and `tune_phase10.py` now reads val NLL from the run manifest.*

*Training memory and evaluation fix (2026-10-10): at `batch_size: 64` training needs about
7 GB of GPU memory; on 6–8 GB laptop GPUs under Windows the overflow spills into system RAM
and an epoch takes ~40–100 min. `training.micro_batch_size: 32` runs each batch as two
micro-batches whose weighted gradients sum exactly to the full-batch gradient (tested), so
the optimisation protocol is unchanged and peak memory is ~3.5 GB. Measured on an RTX 4060
Laptop GPU with real PEMS-BAY data: one epoch including validation takes ~5 min, so a
100-epoch run is at most ~8 h. The same change fixes `st-dssm-train` saving predicted
sigmas under a key the evaluator did not read, which had silently dropped NLL, CRPS, PICP
and MPIW from every ST-DSSM run manifest. Run manifests now also record the training seed,
checkpoint, best epoch, epochs run and the full training config.*

*Resumable training (2026-10-10): `st-dssm-train` and `st-dssm-baseline --model st_gcn` save a
`<run_id>_last.pt` training state after every epoch (weights, optimizer, early-stopping state
and all RNG states) and delete it when the run completes. `--resume auto` continues the newest
unfinished run for the same split and seed; a resumed run ends with bit-identical weights and
metrics to an uninterrupted one (tested).*

