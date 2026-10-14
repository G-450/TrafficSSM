# Limitations and claim reconciliation

**Status date:** 2026-10-14 (Phase 14, release v1.0.0)

This document states what the project's measured results do and do not support.
Every number cited here is read from a locked deliverable:

- `experiments/normal/phase10_results.md` — normal-condition accuracy
- `experiments/missingness/phase12_results.md` — robustness condition matrix
- `experiments/calibration/phase13_results.md` — calibration, error analysis, latency

Each is regenerable and verifiable with its aggregator's `--check` flag. Where a
claim earlier in the project is not supported, it is corrected here rather than
quietly dropped.

---

## 1. Verdict on the stated hypotheses

The research question of [RESEARCH_AND_EXPERIMENTS.md](RESEARCH_AND_EXPERIMENTS.md):

> When traffic-sensor observations are reduced, can ST-DSSM retain useful
> forecasts while expressing uncertainty whose calibration and width are
> appropriate?

**Answer: the forecasts stay useful; the uncertainty does not stay appropriate
at the sensors that were withheld.** The two halves of the question separate
cleanly, and only reporting them separately is honest.

### H0 — "20% masking does not meaningfully change uncertainty measures"

**Rejected.** At the primary 20% condition, aggregate NLL rises from
`2.1171 ± 0.0055` (normal) to `6.6746 ± 4.9314`, and at the withheld sensors it
reaches `24.2240 ± 24.7281`. Coverage at the withheld sensors falls to
`0.6654 ± 0.2106` against a nominal 95%. These are not marginal changes.

### H1 — "masking changes the predictive distribution, potentially increasing interval width, while calibration remains reasonable"

**Partly supported, and the final clause is false as stated.**

- *Distribution changes:* supported. MPIW rises from `11.070 ± 0.297` mph to
  `13.125 ± 3.296` mph at 20%, so intervals do widen on aggregate. The widening
  was not enforced by any loss term or post-processing.
- *Calibration remains reasonable:* **false for withheld sensors.** Interval ECE
  at the withheld sensors is `0.2775 ± 0.1600` at 20%, against `0.0724 ± 0.0037`
  in the normal condition, and PIT ECE reaches `0.4060 ± 0.0942`. At observed
  sensors calibration does hold up (`0.0798 ± 0.0147` interval ECE, PICP
  `0.9630 ± 0.0091`).

The widening is real but far too small, and far too uniform, to cover the errors
it is supposed to cover. The model does not know *where* it has lost evidence.

### The interval-ECE trap

Aggregate interval ECE appears to **improve** under missingness —
`0.0724` (normal) to `0.0500` (20%) to `0.0507` (30%) — while NLL degrades by
more than 3x. This is not a calibration gain. The normal-condition forecast is
over-dispersed in the bulk, so the extra width caused by missingness cancels
part of that bias at the levels the metric samples, while the tail failures that
drive NLL sit outside them. **Aggregate interval ECE must not be quoted as
evidence of robust calibration**; the per-group tables and NLL are the evidence.

---

## 2. Calibration limitations

### The predictive shape is wrong, not just its scale

Even in the normal condition the forecast is not calibrated. The signed coverage
gap is strongly positive at low levels (`+0.1311` at 50%) and slightly negative
at the extreme (`-0.0076` at 99%), and the PIT histogram is hump-shaped
(`phase13_pit_normal.png`). That combination — too wide in the bulk, not wide
enough at the extreme — is the signature of standardized residuals much
heavier-tailed than the Gaussian the model assumes. The predictive *family* is
the limitation, not its parameters.

### Near-term forecasts are the least trustworthy under missingness

Coverage of the nominal 95% interval, by horizon:

| Condition / group | 5 min | 15 min | 30 min | 60 min |
| :--- | ---: | ---: | ---: | ---: |
| normal, all | 0.957 | 0.961 | 0.961 | 0.963 |
| 20%, all | 0.845 | 0.888 | 0.908 | 0.930 |
| 30%, all | 0.782 | 0.846 | 0.876 | 0.910 |
| **20%, withheld sensors** | **0.389** | 0.583 | 0.686 | 0.794 |
| 20%, observed sensors | 0.959 | 0.964 | 0.963 | 0.964 |

This inverts the usual intuition that short horizons are the safe ones. At 5
minutes the predictive scale is normally very small, because the next step is
close to the last observation; withholding that observation leaves a confident
interval with nothing to justify it. **At a withheld sensor the 5-minute 95%
interval contains the truth 39% of the time.** By 60 minutes the scale is wide
regardless and the missing input matters relatively less.

The operational consequence: under sensor loss the near-term forecast is the one
that must not be trusted, and an interval width alone gives no warning, because
the interval is narrow precisely where it is least justified.

### The predictive mean is biased low, and the bias grows with missingness

The PIT histograms are not only hump-shaped, they are shifted right. Derived from
the locked PIT densities in `phase13_results.json`, the share of targets falling
*above* the predicted mean is:

| Condition | normal | 10% | 20% | 30% |
| :--- | ---: | ---: | ---: | ---: |
| mass above the predicted mean | 0.565 | 0.615 | 0.649 | 0.669 |
| mean PIT (0.5 is unbiased) | 0.524 | 0.550 | 0.571 | 0.588 |

The model under-forecasts speed, mildly in normal operation and progressively more
as sensors are withheld: losing evidence pulls predictions toward lower speeds. A
scale-only calibration cannot correct this — the bias is in the mean, which
ADR-0008 forbids calibration from touching — and the symmetric Gaussian intervals
inherit it. Any downstream use that is asymmetric in cost (for example treating
"slower than forecast" and "faster than forecast" differently) must account for it.

### Temperature scaling cannot fix it, and makes coverage worse

The validation-fitted temperature is `1.0694 ± 0.0196`. It lowers NLL, as it
must, but it **raises** normal-condition interval ECE from `0.0724 ± 0.0037` to
`0.0887 ± 0.0008`. `T*` minimizes a mean-square functional of the standardized
residuals, so with heavy tails it is dominated by rare extreme residuals and
widens intervals that were already too wide in the bulk. A single positive
scalar can correct a scale error, not a shape error.

Per ADR-0008 this analysis is secondary and the uncalibrated result is primary;
nothing in the primary results depends on it. **No claim that the model is
"calibrated after temperature scaling" is supportable.**

### The temperature is frozen under the normal condition by design

One temperature per seed is fitted on the normal-condition validation split and
applied unchanged to every condition, because ADR-0008 requires it to be frozen
before test evaluation. It is therefore not the optimal temperature for any
missingness condition, and the calibrated arm under missingness should be read
as "does a calibration fitted in normal operation survive degradation?" — it
largely does not. Per-condition temperatures were not fitted; that would require
validation runs under each condition and is outside the locked protocol.

---

## 3. Robustness limitations

### Seed variance is large under missingness

At the primary 20% condition the primary arm's across-seed MAE spread is
`±0.471` mph on a mean of `2.945`, and the NLL spread is `±4.93` on a mean of
`6.67`. With only three seeds (ADR-0008), these standard deviations are
themselves weakly estimated — three samples. Differences between conditions
smaller than roughly their spread should not be treated as resolved.

### Masked training removes most of the failure, but is a different model

The masked-training ablation (`apply_to_training: true`, 20%, three seeds) is
the strongest result in the study:

| 20% condition, withheld sensors | Primary | Masked-training |
| :--- | ---: | ---: |
| MAE (mph) | 5.986 ± 1.397 | 2.841 ± 0.044 |
| NLL | 24.2240 ± 24.7281 | 2.9906 ± 0.1180 |
| interval ECE | 0.2775 ± 0.1600 | 0.0426 ± 0.0055 |

Aggregate MAE falls from `2.945 ± 0.471` to `2.108 ± 0.018` mph, a 26x tighter
seed spread. The limitations on this result are real and must travel with it:

- It is a **different model** from the primary condition runs (ADR-0011) and is
  never averaged into a primary cell.
- It was run at **20% only**, for ST-DSSM only. 10% and 30% masked-training runs
  were not executed (≈4.5 GPU-hours each).
- It requires the missingness pattern to be **known at training time**. The
  masked sensors are identical in training and evaluation, so this measures
  adaptation to a *known, fixed* outage pattern, not robustness to an arbitrary
  unseen one. A deployment facing unpredictable failures cannot assume this.
- It does **not** fix the shape miscalibration: its interval ECE on observed
  sensors (`0.0688 ± 0.0027`) is comparable to the primary arm's.

### The missingness mechanism is synthetic

A condition withholds whole sensors uniformly at random for the complete input
history (ADR-0004, ADR-0010). Real sensor outages are correlated in space and
time, intermittent, and may coincide with the traffic events that matter most.
The study measures sensitivity to a controlled, idealized mechanism, which is
the right thing for a controlled comparison and is not a claim about field
failure modes.

---

## 4. Scope limitations

- **One dataset, one graph.** All results are PEMS-BAY with its fixed adjacency
  (ADR-0001), symmetrized per ADR-0009. Nothing here generalizes to other sensor
  networks, and no cross-dataset claim is made.
- **Aleatoric uncertainty only.** The Gaussian forecast scale is predictive
  conditional uncertainty. No ensembling or MC dropout is in scope (ADR-0008), so
  **no epistemic-uncertainty claim is made anywhere in this project.**
- **Diagonal Gaussian.** The predictive distribution is independent per sensor
  and horizon, so it cannot represent spatially or temporally correlated error.
  Joint-coverage statements are therefore unavailable, and §2 shows the marginal
  shape is already mis-specified.
- **Capacity-controlled, not state-of-the-art.** ST-DSSM has 397,058 parameters
  and the ST-GCN baseline 109,260, within the 0.5x–2.0x band ADR-0008 requires.
  These are student-scale models; the comparison is internally controlled, not a
  claim against published leaderboards.
- **One locked test evaluation per configuration.** Hyperparameters were tuned
  on validation NLL only (`experiments/tuning/`). The test set was scored once
  per configuration, which is the point — but it means no test-set confidence
  interval beyond the three-seed spread.

---

## 5. Latency claims

Measured, not estimated; see the latency tables in
`experiments/calibration/phase13_results.md`. One forward pass producing 12
horizons for all 325 sensors:

| Device | ST-DSSM | Deterministic ST-GCN | Persistence |
| :--- | ---: | ---: | ---: |
| RTX 4060 Laptop (CUDA) | 1.61 ms/window | 1.12 ms/window | 0.0006 ms/window |
| CPU fallback | 25.74 ms/window | 26.54 ms/window | 0.0005 ms/window |

Limitations on these numbers:

- **Batched throughput, not serving latency.** These are per-window figures
  derived from batches of 64. A deployment forecasting one window at a time pays
  the full per-batch overhead for a single window, so single-request latency is
  substantially higher than the per-window figure. No single-window
  serving-latency claim is made.
- **One machine.** A laptop RTX 4060 (8 GB) and its CPU. No claim about server
  accelerators.
- **ST-GCN is not faster than ST-DSSM on CPU** (26.54 vs 25.74 ms/window)
  despite having 3.6x fewer parameters. Parameter count does not predict latency
  here: the baseline's cost is dominated by Chebyshev graph convolutions over all
  12 input steps. Quote the measured numbers, not parameter ratios.
- Training cost is recorded in the Phase 10/12 run manifests (≈5 min/epoch on
  the same GPU at `micro_batch_size: 32`), not in the latency record.

---

## 6. Reproducibility limitations

- **Prediction artifacts are not in Git.** The Phase 13 calibration diagnostics
  are recomputed from `*_predictions.npz` (~160 MB each), which `.gitignore`
  excludes. `phase13_results.json` records a SHA256 for every prediction file it
  read, so the lock is verifiable when the artifacts are present and
  `--check` fails loudly when they are not. Regenerate them with
  `scripts/run_phase10.py`, `run_phase12.py` and `run_phase13.py`.
- **The study spans three recorded environments.** The locked deliverables cite
  40 run manifests (7 Phase 10 + 30 Phase 12 + 3 Phase 13 validation passes), and
  `experiments/` holds one further manifest — a superseded 20% persistence
  condition run, `baseline-persistence-test-node20-seed2026-20261013-110858` —
  which no lock cites and which was replaced by the `...-201920` re-run. Across
  the 41: 26 runs (most of the Phase 12 condition matrix) used
  `torch 2.6.0+cu124 / numpy 2.5.2`; 7 runs (the three masked-training ablation
  runs, the re-run persistence condition, and the Phase 13 validation passes)
  used `torch 2.12.1+cu130 / numpy 1.26.4`; and 8 — the seven Phase 10
  normal-condition manifests, which were regenerated from saved predictions on
  2026-10-13, plus the superseded run — recorded
  `torch 2.12.1+cpu / numpy 1.26.4`. Each
  aggregator lists every distinct environment with the runs that used it rather
  than reporting one run's versions as the study's. Two caveats on reading these
  records: the recorded versions are those of the process that *wrote the
  manifest*, which for a regenerated manifest is the evaluation environment and
  not the environment the model was trained in (the Phase 10 models were trained
  on CUDA); and metric differences across these environments are ~1e-7 relative,
  six significant figures below reported precision, but the arms are not
  bit-identical environments and the tables say so.
- **The Phase 13 gate verifies to four significant figures, not bit-exactly.**
  Phases 10 and 12 compare values stored in the run manifests, so their gates
  hold to 1e-12. Phase 13 instead *recomputes* every diagnostic from the
  prediction artifacts, and float32 reduction order differs across torch and
  numpy versions: re-deriving the lock from checksum-identical artifacts in a
  different environment lands about 1e-7 away, with a worst observed deviation
  of 1.7e-5 on a three-seed standard deviation, where cancellation amplifies the
  relative error. `aggregate_phase13.py` therefore sets `CHECK_RTOL = 1e-4` —
  four significant figures, the precision at which the deliverable reports every
  value. The consequence is explicit: the Phase 13 gate proves that the tables
  derive from those exact artifacts and that no reported number has moved; it
  does not prove bit-identical recomputation, and a drift smaller than 1e-4
  relative would not be caught. A tighter gate was measured and rejected,
  because at 1e-9 it passed only on the machine that produced the lock, which is
  not a reproducibility guarantee.
- **Preprocessing checksums are environment-specific.** `data/processed`
  checksums were produced on Python 3.12.10 / Windows 11; values differ by at
  most 1.5e-5 across Python and NumPy versions (float32 rounding). Regenerate
  with `st-dssm-preprocess generate --force` on a new machine rather than
  expecting a checksum match.

---

## 7. Claims this project does not make

For the avoidance of doubt, none of the following is supported by the recorded
evidence, and none appears in the locked deliverables:

1. That ST-DSSM's intervals are calibrated, before or after temperature scaling.
2. That ST-DSSM's uncertainty is robust to missingness. It is not, at the
   sensors where evidence was lost.
3. That ST-DSSM beats the deterministic ST-GCN on point accuracy under
   missingness. At 20% the primary arm gives `2.945 ± 0.471` vs
   `2.830 ± 0.141` mph — overlapping, with ST-DSSM the more variable.
4. That the latent state confers robustness. The Phase 12 headline previously
   suggested the opposite (that it *amplifies* disruption); the masked-training
   ablation shows the variance was a train/test mismatch artifact. Neither claim
   is now made: what is measured is that mismatch, not a property of the state
   space.
5. Any epistemic-uncertainty, cross-dataset, or single-request-latency claim.

What the project does support: a fully reproducible, manifest-verifiable
comparison on PEMS-BAY showing that a state-space probabilistic forecaster's
*point* accuracy degrades gracefully under sensor loss while its *predictive
distribution* degrades sharply at the affected sensors, and that training under
the missingness representation recovers almost all of both.
