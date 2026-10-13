# Current project state

**Status date:** 2026-10-14
**Delivery state:** Phase 12 robustness study is complete: all nine conditions
(10/20/30% × seeds 2026/2027/2028) were executed for ST-DSSM, deterministic ST-GCN and
historical persistence, and the locked tables live in `experiments/missingness/`.
Phase 13 calibration and robustness review is next. The Phase 10 and Phase 11/12
results remain locked and verifiable from manifests alone.

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
- **Phase 9 Training operations:** `st-dssm-train` CLI with YAML configuration, validation-NLL
  checkpoint selection, beta annealing, teacher-forcing decay, micro-batched gradient
  accumulation and per-epoch resumable state (`--resume auto`).
- **Phase 11 Missingness mechanism:** nine published mask artifacts under
  `experiments/masks/` (10/20/30% x seeds 2026/2027/2028), each recording the withheld
  sensor indices, their PEMS-BAY IDs and a checksum, regenerated or verified with
  `scripts/generate_masks.py`. `st-dssm-train` and `st-dssm-baseline` accept a `masking:`
  config block or `--mask-ratio/--mask-seed/--mask-path/--mask-training` and withhold whole
  sensors at every input step with `M_obs = 0` (ADR-0007); targets are never masked. Per
  ADR-0011 a condition applies at inference, so `--predict-from <checkpoint>` scores the
  locked Phase 10 model under a condition without retraining, and run identity carries the
  condition. `st-dssm-evaluate` reports every metric over `all`, `masked` and `unmasked`
  target groups and records `mask_seed`, `mask_condition`, `mask_sensor_ids` and
  `mask_checksum`. The leakage gate is an invariance test: two artifacts differing only in
  the withheld sensors' input values produce bit-identical predictions and metrics through
  the real pipeline.
- **Phase 12 Robustness study:** all nine missingness conditions (10/20/30% × seeds 2026/
  2027/2028) executed for ST-DSSM, deterministic ST-GCN and historical persistence using
  locked Phase 10 checkpoints via `--predict-from` (forward passes only, no retraining).
  `scripts/run_phase12.py` automates the 27 runs; `scripts/aggregate_phase12.py` writes
  the locked tables and `--check` re-derives them. Key finding: all models degrade under
  missingness; ST-DSSM shows higher seed variance at 20/30% (MAE 2.6–3.5 mph at 20%)
  than ST-GCN (2.7–3.0 mph). Masked-sensor MAE is 2–3× the unmasked-sensor MAE across
  all models and conditions. The masking-representation ablation
  (`apply_to_training: true`) retrains under the mask, so it is a different model from a
  primary condition run: its run identity carries `-masktrain`, and the aggregator reports
  it in its own labelled section rather than in a primary cell (ADR-0011).
  See `experiments/missingness/phase12_results.md` for locked tables.

- **Phase 10 Normal-condition study:** hyperparameters tuned against validation NLL only
  (`scripts/tune_phase10.py`, search recorded in `experiments/tuning/`, best val NLL 2.118),
  configuration locked in `configs/tuned.yaml`, and the test set evaluated once per canonical
  seed (2026, 2027, 2028) for ST-DSSM and for the Phase 5 baselines under the same protocol.
  `scripts/aggregate_phase10.py` writes the locked deliverable from the run manifests and
  `--check` re-derives it to prove the tables are reproducible.

- **Phase 8 DSSM integration:** Full `st_dssm.dssm.GaussianDSSM` wrapping Phase 6 encoder + Phase 7 forecast head. Per ADR-0007: 32-dimensional diagonal Gaussian latent state per sensor; causal GRU prior transition $p(z_t \mid z_{t-1}, c_t)$; bidirectional GRU recognition (posterior) network $q(z_t \mid \text{context}_{1:L}, x, m)$ used only during training; closed-form KL divergence; masked Gaussian reconstruction NLL; ELBO $= \text{NLL} + \beta \cdot \text{KL}$ with $\beta$ annealing from 0 to 1 over first 20 epochs (caller-managed); hard non-finite ELBO guard; `forward_predict` samples exclusively from prior with no target access. `st-dssm-dssm` CLI with synthetic smoke test. Capacity report breaks down parameters per sub-module.

## Canonical Baseline Results (PEMS-BAY Test Set, Physical Units `mph`)

The canonical deterministic baseline evaluations were executed on the validated PEMS-BAY
test partition ($S=10,403$ windows, $N=325$ sensors, $H=12$ horizons). The Phase 10 runs
under `experiments/normal/baselines/` are the current record; the locked, manifest-derived
tables live in [`experiments/normal/phase10_results.md`](../experiments/normal/phase10_results.md).

| Model Architecture | 15 min (MAE / RMSE / MAPE) | 30 min (MAE / RMSE / MAPE) | 60 min (MAE / RMSE / MAPE) | Overall Aggregate (MAE / RMSE / MAPE) |
| :--- | :--- | :--- | :--- | :--- |
| **Historical Persistence** | 1.59 / 3.39 / 3.23% | 2.17 / 4.95 / 4.64% | 3.04 / 6.98 / 6.82% | **2.17 mph** / **5.12 mph** / **4.67%** |
| **Deterministic ST-GCN** (109,260 params, 3 seeds) | 1.39 / 2.92 / 2.92% | 1.82 / 4.05 / 4.15% | 2.33 / 5.25 / 5.64% | **1.78 mph** / **4.06 mph** / **4.06%** |

*Correction (2026-10-13): the per-horizon persistence figures previously recorded here
(1.64 / 2.22 / 3.01 mph MAE) were not reproducible from any run manifest. The values above
are generated by `scripts/aggregate_phase10.py` from the manifests and verifiable with
`--check`; do not hand-edit them.*

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
| Phase 10 (run scripts, aggregation, provenance, plots) | 18 | ✅ Passing |
| Phase 11 (mask generation) | 15 | ✅ Passing |
| Phase 11 (masking wiring, leakage gate, target groups) | 21 | ✅ Passing |
| Phase 12 (condition aggregator, ablation arm, run script) | 18 | ✅ Passing |
| **Total** | **297** | **✅ All passing** |

*Note: `test_encoder_cli_unrecognized_graph_and_missing_metadata` requires the canonical `data/processed` artifact to be present on disk; it is skipped in environments without PEMS-BAY data (pre-existing condition, not caused by Phase 7/8 changes).*

## Next authorized work

Phase 12 is complete. Proceed to
[Phase 13 — Calibration and robustness review](IMPLEMENTATION_PLAN.md):
- Calibration diagnostics: reliability diagrams, expected calibration error, temperature
  scaling analysis (secondary, per ADR-0008)
- Error analysis: per-horizon degradation curves under each condition, masked vs unmasked
  sensor breakdown
- Latency records and claims reconciled with evidence from `experiments/missingness/`
- Limitations statement (seed variance at 20/30%, masking-representation ablation deferred)

**Compute note (2026-10-13):** the installed PyTorch is `2.12.1+cpu`, so
`torch.cuda.is_available()` is `False`. Per ADR-0011 a condition is a forward pass over
the test split rather than a training run, so the Phase 12 condition matrix does not need
the GPU; the masked-training ablation does. The Phase 10 runs were trained on CUDA (an
RTX 4060 Laptop GPU), so reinstall a CUDA build of torch before any run that fits
parameters, or the ~5 min/epoch figure below no longer holds.

## Reproducing the Phase 10 results

```bash
python scripts/aggregate_phase10.py --check    # verify the locked tables
python scripts/aggregate_phase10.py            # regenerate tables and figures
```

Metrics are bit-identical when recomputed from the same predictions in the same
environment. Values recomputed in a different process can differ by ~1e-7 relative
(float32 summation order over ~4e7 targets), which is six significant figures below
the reported precision.



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

*Run-provenance fix (2026-10-13): training provenance used to be patched into a run
manifest after the evaluator had already saved it. The Phase 10 seed-2028 ST-DSSM run was
killed inside that window, while rendering its calibration curve, and its manifest reported
seed `null`, `0` epochs and `0` parameters for an otherwise valid result. `run_evaluation`
now takes the provenance and writes it in the single atomic save, and `st-dssm-train` and
`st-dssm-baseline` pass it in instead of patching. The same write now also fills
`config_hash`, `package_versions`, `dataset_manifest_checksum` and
`split_artifact_checksum`, which the schema carried but nothing had ever populated.
All seven Phase 10 manifests were regenerated from their saved predictions with
`st-dssm-evaluate --provenance-checkpoint ...`, which leaves metrics unchanged to six
significant figures, and the seed-2028 calibration plot is no longer missing.*
