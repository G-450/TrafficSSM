# ADR-0008: Fix baselines, seeds, calibration, and uncertainty claims

**Status:** Accepted
**Date:** 2026-08-13
**Resolves:** OQ-06, OQ-07, OQ-08, OQ-09

## Context

Comparable final results require a baseline, finite compute rules, frozen seeds, an early-stopping rule, and a policy for calibration and epistemic claims. Leaving these choices until after seeing test results would create avoidable researcher degrees of freedom.

## Decision

### Baselines and capacity reporting

Use two required deterministic baselines:

- historical persistence: repeat the last observed value for all 12 horizons, using the same input mask and zero-plus-mask representation;
- deterministic ST-GCN: use the encoder block specification and 64-channel width from ADR-0007, followed by a direct 12-horizon linear point-forecast head trained with MAE.

The deterministic ST-GCN is the capacity-controlled graph baseline. Exact equality in parameter count is not required; its trainable parameter count must be reported and must remain between `0.5x` and `2.0x` the trainable count of the ST-DSSM. If it falls outside that range, adjust only the baseline hidden width and record the chosen width before tuning. All models receive identical splits, histories, masks, inverse transforms, and point-metric code.

### Seeds, compute budget, and stopping

Canonical final training seeds are `2026`, `2027`, and `2028`. Canonical experimental-mask seeds are the same three values, paired by seed across models and masking conditions. A run may train for at most 100 epochs, with one model fit at a time on one accelerator (or CPU fallback), automatic mixed precision permitted when recorded, and no more than 30 hyperparameter trials per learned model family.

Checkpoint selection uses validation NLL for probabilistic models and validation MAE for deterministic models. Stop after 15 consecutive epochs without an improvement of at least `1e-4` in the selection metric; always retain the best checkpoint. The test partition is evaluated only after configuration and calibration are locked. Failed runs are reported and are not silently replaced with favorable seeds.

### Calibration

No post-hoc calibration is part of the primary result. Validation-fitted scalar temperature scaling of predictive standard deviation is allowed only as a separately labelled secondary analysis. It uses one positive scalar shared across sensors and horizons, selected by validation NLL, then frozen before test evaluation. Both raw and calibrated results must be reported; calibration may not alter predictive means.

### Epistemic scope

No additional epistemic method is in the canonical project scope. Ensembles and MC dropout may be future, separately versioned studies, but their outputs must not be mixed into the primary ST-DSSM result. The project describes the canonical Gaussian forecast scale as predictive conditional/aleatoric uncertainty and does not claim epistemic uncertainty.

## Consequences

The required comparison is feasible on student-scale compute, selection is validation-only, and optional calibration cannot replace the uncalibrated primary result. Claims are narrower but technically defensible.
