# Uncertainty-Aware Spatial-Temporal Forecasting on PEMS-BAY — final report

**Release:** v1.0.0 · **Report date:** 2026-10-14 · **Phases 1–14 complete**

Every number in this report is read from a locked deliverable that a reviewer can
re-derive from checksummed run manifests. No figure here is typed by hand; where
this report cites a value, the citation names the file and the command that
reproduces it. What the results do **not** support is stated in
[LIMITATIONS.md](LIMITATIONS.md), which this report does not restate but does not
contradict.

| Deliverable | Locked file | Verify with |
|---|---|---|
| Normal-condition study | [`experiments/normal/phase10_results.md`](../experiments/normal/phase10_results.md) | `python scripts/aggregate_phase10.py --check` |
| Robustness condition matrix (30 runs) | [`experiments/missingness/phase12_results.md`](../experiments/missingness/phase12_results.md) | `python scripts/aggregate_phase12.py --check` |
| Calibration review, error analysis, latency | [`experiments/calibration/phase13_results.md`](../experiments/calibration/phase13_results.md) | `python scripts/aggregate_phase13.py --check` |

---

## 1. Research question and verdict

> When traffic-sensor observations are reduced, can ST-DSSM retain useful
> forecasts while expressing uncertainty whose calibration and width are
> appropriate?

**The forecasts stay useful. The uncertainty does not stay appropriate at the
sensors whose observations were withheld.** The two halves of the question
separate cleanly, and reporting them jointly would hide the result.

| Hypothesis | Verdict | Decisive evidence |
|---|---|---|
| **H0** — 20% masking does not meaningfully change uncertainty measures | **Rejected** | Aggregate NLL `2.117 → 6.675`; `24.224` at withheld sensors |
| **H1** — the distribution changes, intervals may widen, calibration stays reasonable | **Partly supported; final clause false** | MPIW `11.070 → 13.125` mph (widening is real and unforced), but withheld-sensor interval ECE `0.0724 → 0.2775` and PICP `0.665` against nominal `0.95` |

The widening of the intervals under missingness was not enforced by any loss
term, post-processing rule, or plotting choice — it is measured. It is also far
too small, and far too uniform across sensors, to cover the errors it would need
to cover. **The model does not know *where* it has lost evidence.**

## 2. What was built

A reproducible PEMS-BAY pipeline and three forecasters evaluated under one
protocol: 325 sensors, 12 input steps (60 min) to 12 forecast horizons (60 min)
at 5-minute cadence, chronological 70/10/20 split with all data-dependent
preprocessing fitted on training data only.

- **ST-DSSM** (397,058 params) — a 2-block causal ST-GCN encoder (gated temporal
  convolution + Chebyshev graph convolution, *K*=3) feeding a deep state-space
  model with a 32-dimensional diagonal Gaussian latent state per sensor, a causal
  GRU prior transition, a bidirectional GRU recognition network used only in
  training, and an autoregressive 12-step Gaussian forecast head trained on an
  ELBO with β annealing.
- **Deterministic ST-GCN** (109,260 params) — a capacity-controlled point
  forecaster inside the 0.5×–2.0× band ADR-0008 requires.
- **Historical persistence** (0 params) — a non-parametric reference.

Design decisions are binding and recorded as eleven ADRs in
[DECISIONS/](DECISIONS/README.md); the architecture is specified in
[ARCHITECTURE.md](ARCHITECTURE.md) and the data rules in
[DATA_CONTRACT.md](DATA_CONTRACT.md).

## 3. Normal-condition results

Test set, 10,403 windows, 40,569,096 observed targets scored, mean ± sd over
seeds 2026/2027/2028, physical units (mph). Hyperparameters were tuned against
validation NLL only; the test set was scored once per configuration.

| Model | Params | MAE | RMSE | MAPE | NLL | CRPS | PICP@95% | MPIW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical persistence | — | 2.169 | 5.122 | 4.670% | — | — | — | — |
| Deterministic ST-GCN | 109,260 | **1.777 ± 0.004** | **4.056 ± 0.008** | **4.064 ± 0.040%** | — | — | — | — |
| ST-DSSM | 397,058 | 1.936 ± 0.011 | 4.095 ± 0.004 | 4.411 ± 0.108% | 2.117 ± 0.006 | 1.431 ± 0.007 | 0.9612 ± 0.0013 | 11.070 ± 0.297 |

ST-DSSM pays about 0.16 mph of MAE against the deterministic baseline for a
predictive distribution the baseline cannot provide at all. That is the trade the
project set out to measure, and it is a real cost, not a rounding difference.

## 4. Degradation under sensor loss

A condition withholds whole sensors uniformly at random across the entire input
history (`M_obs = 0`), at inference, using the locked Phase 10 checkpoints
(ADR-0011). Targets are never masked, so every condition splits into withheld and
observed sensor groups over one set of predictions. Nine published mask artifacts
(10/20/30% × 3 seeds) fix exactly which sensors are withheld; a leakage gate
proves two artifacts differing only in the withheld sensors' input values produce
bit-identical predictions.

**Point accuracy (MAE, mph), all sensors:**

| Model | normal | 10% | 20% | 30% |
|---|---:|---:|---:|---:|
| Historical persistence | 2.169 | 2.489 ± 0.047 | 2.813 ± 0.013 | 3.115 ± 0.029 |
| Deterministic ST-GCN | 1.777 | 2.267 ± 0.043 | **2.830 ± 0.141** | **3.250 ± 0.217** |
| ST-DSSM | 1.936 | 2.444 ± 0.179 | 2.945 ± 0.471 | 3.415 ± 0.704 |

All three models degrade, and by 20% all three have converged to roughly
persistence-level error. ST-DSSM does not beat the deterministic baseline under
missingness: at 20% the intervals overlap (`2.945 ± 0.471` vs `2.830 ± 0.141`)
with ST-DSSM the more variable of the two. **No robustness advantage from the
latent state is claimed, and none is measured.**

**The distribution degrades much faster than the mean.** At 20%:

| 20% condition | all sensors | withheld | observed |
|---|---:|---:|---:|
| MAE (mph) | 2.945 ± 0.471 | 5.986 ± 1.397 | 2.184 ± 0.241 |
| NLL | 6.675 ± 4.931 | **24.224 ± 24.728** | 2.287 ± 0.053 |
| PICP@95% | 0.9035 ± 0.0491 | **0.6654 ± 0.2106** | 0.9630 ± 0.0091 |
| interval ECE | 0.0500 ± 0.0100 | **0.2775 ± 0.1600** | 0.0798 ± 0.0147 |

MAE at withheld sensors is 2.5–3× the observed-sensor figure in every condition,
but NLL there is more than *ten times* the observed-sensor value. Calibration
holds up at observed sensors and collapses at withheld ones, so only the
per-group tables answer the question honestly.

> [!IMPORTANT]
> Aggregate interval ECE *appears to improve* under missingness
> (`0.0724` normal → `0.0500` at 20%) while NLL more than triples. This is an
> artifact: the normal-condition forecast is over-dispersed in the bulk, so the
> extra width cancels part of that bias at the levels the metric samples, while
> the tail failures driving NLL sit outside them. It must never be quoted as
> evidence of robust calibration.

## 5. Calibration findings

**The predictive shape is wrong, not just its scale — even with no masking.** The
signed coverage gap is strongly positive at low nominal levels (`+0.1311` at 50%)
and slightly negative at the extreme (`-0.0076` at 99%), with a hump-shaped PIT
histogram. Too wide in the bulk, not wide enough in the tail: the signature of
standardized residuals much heavier-tailed than the assumed Gaussian. The
limitation is the predictive *family*, not its fitted parameters.

**Temperature scaling cannot repair that, and makes coverage worse.** The
validation-fitted scalar (`1.0694 ± 0.0196`, one per seed, frozen before any test
number per ADR-0008, with the NLL-optimal closed form `T* = sqrt(mean(r²))`
verified against a dense grid search) lowers NLL as it must, while *raising*
normal-condition interval ECE from `0.0724` to `0.0887`. A single positive scalar
corrects a scale error, not a shape error. **No claim that the model is
"calibrated after temperature scaling" is supportable.**

**The near-term forecast is the least trustworthy one under sensor loss**, which
inverts the usual intuition. Coverage of the nominal 95% interval:

| Condition / group | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|
| normal, all | 0.957 | 0.961 | 0.961 | 0.963 |
| 20%, all | 0.845 | 0.888 | 0.908 | 0.930 |
| **20%, withheld sensors** | **0.389** | 0.583 | 0.686 | 0.794 |
| 20%, observed sensors | 0.959 | 0.964 | 0.963 | 0.964 |

At 5 minutes the predictive scale is normally very small, because the next step
is close to the last observation. Withhold that observation and a confident
interval remains with nothing left to justify it: at a withheld sensor the
5-minute 95% interval contains the truth **39%** of the time. By 60 minutes the
scale is wide regardless and the missing input matters relatively less. The
operational consequence is that interval width alone gives no warning — the
interval is narrowest exactly where it is least justified.

**The predictive mean is biased low, increasingly so as sensors are withheld**
(mean PIT `0.524` normal → `0.588` at 30%; 66.9% of targets fall above the
predicted mean at 30%). Losing evidence pulls forecasts toward lower speeds. A
scale-only calibration cannot correct this, because ADR-0008 forbids calibration
from touching the mean, and the symmetric Gaussian intervals inherit the bias.

## 6. The decisive ablation: training under the mask

Retraining with the same sensors withheld (`apply_to_training: true`, 20%, three
seeds) recovers almost all of both losses. These runs are a **different model**
from the primary condition runs and are never averaged into a primary cell
(ADR-0011).

| 20% condition, withheld sensors | Primary (inference-only mask) | Trained under the mask |
|---|---:|---:|
| MAE (mph) | 5.986 ± 1.397 | **2.841 ± 0.044** |
| NLL | 24.224 ± 24.728 | **2.991 ± 0.118** |
| interval ECE | 0.2775 ± 0.1600 | **0.0426 ± 0.0055** |

Aggregate MAE falls from `2.945 ± 0.471` to `2.108 ± 0.018` mph — a 26× tighter
across-seed spread. **This refutes an earlier reading of the Phase 12 seed
variance**, which had suggested the latent state amplifies input disruption. It
does not: the variance was a train/test representation mismatch, and the
corrected interpretation is recorded rather than quietly dropped.

Three limits travel with this result and must not be separated from it: it was
run at 20% only and for ST-DSSM only (≈4.5 GPU-hours per additional run); it
requires the outage pattern to be **known at training time**, with identical
masked sensors in training and evaluation, so it measures adaptation to a known
fixed outage rather than robustness to an unseen one; and it does **not** fix the
shape miscalibration, its observed-sensor interval ECE (`0.0688 ± 0.0027`) being
comparable to the primary arm's.

## 7. Measured inference latency

One forward pass producing 12 horizons for all 325 sensors, with warmup and CUDA
synchronization around every timed region, median over 30 timed batches of 64.

| Device | ST-DSSM | Deterministic ST-GCN | Persistence |
|---|---:|---:|---:|
| RTX 4060 Laptop (CUDA) | 1.61 ms/window | 1.12 ms/window | 0.0006 ms/window |
| CPU fallback | 25.74 ms/window | 26.54 ms/window | 0.0005 ms/window |

These are batched per-window figures, not single-request serving latency, and
come from one laptop. Note that ST-GCN is *not* faster than ST-DSSM on CPU
despite 3.6× fewer parameters — its cost is dominated by Chebyshev graph
convolutions over all 12 input steps. Quote the measured numbers, never
parameter ratios.

## 8. Requirements coverage

Every functional requirement in the [PRD](PRD.md) has acceptance evidence in the
repository.

| ID | Requirement | Evidence |
|---|---|---|
| FR-01 | Pinned PEMS-BAY distribution, 325 aligned sensors | `data/manifest/`, Phase 1 provenance checks |
| FR-02 | Chronological 70/10/20, preprocessing fitted on train only | `data/processed/processed_metadata.json`, Phase 2–3 tests |
| FR-03 | 12-step history → 12-step forecast samples | Phase 3 shape/content tests |
| FR-04 | Mean and distribution parameters per horizon and sensor | Phase 7–8 model-interface tests |
| FR-05 | Defined proper probabilistic objective | ELBO with masked Gaussian NLL, `configs/tuned.yaml` |
| FR-06 | Normal plus 10/20/30% inference masking conditions | `experiments/masks/` (9 artifacts), `phase12_results.md` |
| FR-07 | Withheld truth never enters input or imputation | Phase 11 leakage gate (bit-identical-prediction invariance test) |
| FR-08 | MAE, RMSE, NLL, CRPS, PICP, MPIW with aggregation rules | All three locked deliverables |
| FR-09 | Revision, config, manifest, seed, mask ID, metrics per run | `RunManifest` schema; 41 run manifests |

## 9. Reproducing this report

```bash
pip install -e ".[dev]"
python -m pytest                               # 363 tests
st-dssm-provenance verify                      # dataset checksums
st-dssm-preprocess generate --force            # regenerate data/processed
python scripts/aggregate_phase10.py --check    # normal-condition tables
python scripts/aggregate_phase12.py --check    # condition matrix (30 manifests)
python scripts/aggregate_phase13.py --check    # calibration (~25 min)
```

Full command sequences, including training and condition runs from scratch, are
in [run.md](../run.md).

Four caveats a reviewer will meet, all documented rather than worked around:
prediction artifacts (`*_predictions.npz`, ~160 MB each) are git-ignored and the
Phase 13 aggregator needs them on disk, failing loudly and by name when they are
absent; preprocessing checksums are environment-specific, so regenerate rather
than expect a match; the study spans three recorded run environments, each listed
in the deliverables with the runs that used it; and the Phase 13 gate verifies to
four significant figures rather than bit-exactly, because it recomputes from the
artifacts and float32 reduction order differs across torch and numpy versions.
All four are stated in [LIMITATIONS.md](LIMITATIONS.md) §6.

## 10. What this project does not claim

Stated in full in [LIMITATIONS.md](LIMITATIONS.md) §7. In short: not that
ST-DSSM's intervals are calibrated, before or after temperature scaling; not that
its uncertainty is robust to missingness; not that it beats the deterministic
baseline under missingness; not that the latent state confers robustness; and no
epistemic-uncertainty, cross-dataset, or single-request-latency claim. Results
are PEMS-BAY only, aleatoric only, diagonal-Gaussian only, capacity-controlled
rather than state-of-the-art, with three seeds and one locked test evaluation per
configuration.

## 11. Conclusion and future work

A probabilistic spatial-temporal forecaster's **point accuracy degrades
gracefully under sensor loss while its predictive distribution degrades sharply,
and specifically at the sensors where the evidence was lost.** Aggregate
calibration metrics actively conceal this; only per-group, per-horizon reporting
exposes it. Training under the missingness representation recovers nearly all of
both the accuracy and the calibration loss, but requires the outage pattern to be
known in advance.

The two findings that would most change practice, and the work each implies:

1. **A heavier-tailed predictive family.** The shape mismatch is present with no
   masking at all and no scalar can repair it; a Student-*t* or mixture head is
   the direct test.
2. **Uncertainty that localizes missing evidence.** The model widens intervals
   roughly uniformly instead of where the inputs went dark. Conditioning the
   predictive scale explicitly on the observation mask is the obvious next step,
   and the masked-training ablation is evidence that the information is learnable.

Lower-priority extensions: masked-training runs at 10% and 30%; per-condition
temperatures (outside the locked ADR-0008 protocol); correlated and intermittent
outage mechanisms in place of the synthetic uniform one; and a second sensor
network to test whether any of this transfers.
