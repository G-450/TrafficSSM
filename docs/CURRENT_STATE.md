# Current project state

**Status date:** 2026-10-14
**Delivery state:** **Released as v1.0.0.** All fourteen phases are complete.
Every experimental phase (1-13) is done and locked, and Phase 14 has verified a
clean-environment install, re-derived all three locked deliverables against the
release commit, and published the final report
([FINAL_REPORT.md](FINAL_REPORT.md)), which carries the study's conclusions with
every number traceable to the command that reproduces it.
The locked deliverables are `experiments/normal/phase10_results.md`,
`experiments/missingness/phase12_results.md` (30 runs, including the completed
masked-training ablation) and `experiments/calibration/phase13_results.md`
(calibration diagnostics, error analysis, measured latency), each regenerable and
verifiable with its aggregator's `--check` flag. Scientific claims and their
limits are stated in [LIMITATIONS.md](LIMITATIONS.md).

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
  missingness; in the primary arm ST-DSSM shows higher seed variance at 20/30%
  (MAE 2.945 ± 0.471 mph at 20%) than ST-GCN (2.830 ± 0.141). Masked-sensor MAE is
  2–3× the unmasked-sensor MAE across all models and conditions. The
  masking-representation ablation (`apply_to_training: true`) retrains under the mask, so
  it is a different model from a primary condition run: its run identity carries
  `-masktrain`, and the aggregator reports it in its own labelled section rather than in a
  primary cell (ADR-0011). **The ablation completed on 2026-10-14 for all three seeds and
  is locked** (30 manifests total): training under the mask gives MAE 2.108 ± 0.018 mph
  at 20%, with masked-sensor MAE falling 5.986 -> 2.841, a 26× tighter across-seed
  spread. That refutes the earlier reading of the seed variance as the latent state
  amplifying input disruption — it is a train/test representation mismatch. See
  [LIMITATIONS.md](LIMITATIONS.md) §3 for the caveats that travel with this result
  (one condition, ST-DSSM only, mask known at training time).
  See `experiments/missingness/phase12_results.md` for locked tables and
  `experiments/missingness/plots/phase12_*.png` for the per-condition comparison figures.

- **Phase 13 Calibration and robustness review:** `st_dssm.calibration` adds reliability
  tables with signed coverage gaps over seven nominal levels, an interval expected
  calibration error, PIT diagnostics with a total-variation PIT error, and the ADR-0008
  secondary temperature-scaling analysis, whose NLL-optimal scalar has the closed form
  $T^{*} = \sqrt{\text{mean}(r^2)}$ over standardized residuals (verified against a dense
  grid search). `scripts/run_phase13.py` scores each locked Phase 10 checkpoint over the
  validation split so the temperature is fitted on validation and frozen before any test
  number; `scripts/measure_latency.py` records measured inference latency on the GPU and
  the CPU fallback; `scripts/aggregate_phase13.py` locks
  `experiments/calibration/phase13_results.{md,json}` plus 12 figures, and `--check`
  re-derives them from checksummed manifests *and* prediction artifacts. Key findings:
  point accuracy degrades gracefully under sensor loss while the predictive distribution
  does not (aggregate NLL 2.117 -> 6.675 at 20%, and 24.224 at withheld sensors, where
  PICP falls to 0.665 against a nominal 0.95); the forecast is over-dispersed in the bulk
  and under-dispersed in the tail even in the normal condition, so the fitted temperature
  (1.0694 ± 0.0196) lowers NLL while raising interval ECE (0.0724 -> 0.0887) — a
  scalar cannot repair a shape mismatch. Aggregate interval ECE *falls* under missingness
  while NLL triples, and must not be quoted as evidence of robust calibration. The
  per-horizon analysis inverts the usual intuition: coverage is worst at the *shortest*
  horizon, and at a withheld sensor the 5-minute 95% interval covers the truth only 39%
  of the time (0.794 at 60 min), because a short-horizon interval is narrow precisely
  where the observation justifying it was removed. The predictive mean is also biased
  low, increasingly so as sensors are withheld (mean PIT 0.524 normal -> 0.588 at 30%).
  H0 is rejected and H1's "calibration remains reasonable" clause is false at withheld
  sensors (interval ECE 0.0724 -> 0.2775). Limits and caveats:
  [LIMITATIONS.md](LIMITATIONS.md).

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
| Phase 13 (calibration diagnostics and temperature scaling) | 28 | ✅ Passing |
| Phase 13 (calibration aggregator, gates, runner) | 32 | ✅ Passing |
| Phase 13 (reliability and PIT plots) | 6 | ✅ Passing |
| **Total** | **363** | **✅ All passing** |

*Note: `test_encoder_cli_unrecognized_graph_and_missing_metadata` requires the canonical `data/processed` artifact to be present on disk; it is skipped in environments without PEMS-BAY data (pre-existing condition, not caused by Phase 7/8 changes).*

## Release state and remaining optional scope

**Phase 14 is complete and the project is released as v1.0.0.** The release
verified a clean-environment install from `pyproject.toml` alone, re-derived all
three locked deliverables with their `--check` flags, published
[FINAL_REPORT.md](FINAL_REPORT.md), and tagged a clean history with Phases 11-13
merged to `main`.

**Gate (met):** a reviewer can rerun the documented workflow ([run.md](../run.md))
and audit every conclusion.

The two items previously parked for Phase 14 are settled as stated limitations,
not as open work:
- The study spans **three** recorded run environments, not two; the manifests
  record `torch 2.6.0+cu124` (26 runs), `2.12.1+cu130` (7) and `2.12.1+cpu` (8,
  including the regenerated Phase 10 manifests). Each aggregator lists every
  distinct environment with the runs that used it, and
  [LIMITATIONS.md](LIMITATIONS.md) §6 states the consequence and the caveat that a
  recorded environment is the one that wrote the manifest, not necessarily the one
  that trained the model.
- Masked-training ablation runs exist at 20% only, for ST-DSSM only (~4.5
  GPU-hours per additional run). Extending to 10%/30% is optional post-release
  scope, recorded in [LIMITATIONS.md](LIMITATIONS.md) §3 and
  [FINAL_REPORT.md](FINAL_REPORT.md) §11.

**Compute note (2026-10-14, supersedes the 2026-10-13 note):** the installed
PyTorch is now `2.6.0+cu124` and `torch.cuda.is_available()` is `True`, so the
~5 min/epoch figure below holds again and no reinstall is needed before a
parameter-fitting run. Per ADR-0011 a condition is a forward pass over the test
split rather than a training run, so the Phase 12 condition matrix does not need
the GPU; the masked-training ablation does. A clean environment installing from
`pyproject.toml` alone resolves `torch>=2.0.0` to a current release (2.14.1 as of
this date) — intended, since the package must install on current dependencies
while the exact study environments stay recorded per run.

## Reproducing the Phase 10 results

```bash
python scripts/aggregate_phase10.py --check    # verify the locked tables
python scripts/aggregate_phase10.py            # regenerate tables and figures
python scripts/aggregate_phase12.py --check    # Phase 12 condition matrix (30 manifests)
python scripts/aggregate_phase13.py --check    # Phase 13 calibration diagnostics
```

`aggregate_phase13.py` recomputes calibration from the prediction artifacts rather
than reading manifests, so it needs the git-ignored `*_predictions.npz` files on
disk and takes a few minutes; it fails with explicit guidance when they are absent.

Metrics are bit-identical when recomputed from the same predictions in the same
environment. Values recomputed in a different process, or under a different torch
or numpy version, can differ by ~1e-7 relative (float32 summation order over ~4e7
targets), which is six significant figures below the reported precision. The
Phase 13 gate therefore compares at `CHECK_RTOL = 1e-4` — four significant
figures, the precision at which the deliverable reports values — because at 1e-9
it passed only on the machine that produced the lock. It proves the tables derive
from those exact artifacts and that no reported number moved, not bit-identical
recomputation; see [LIMITATIONS.md](LIMITATIONS.md) §6.



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
