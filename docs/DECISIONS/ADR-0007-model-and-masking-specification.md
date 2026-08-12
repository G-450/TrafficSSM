# ADR-0007: Fix the canonical ST-DSSM and masked-input specification

**Status:** Accepted
**Date:** 2026-08-13
**Resolves:** OQ-03, OQ-04, OQ-05; accepts and supersedes the proposal in ADR-0005

## Context

The name “ST-GCN + DSSM” did not specify a reproducible model. Encoder dimensions, latent inference, the variational objective, multi-step decoding, and masked-value handling all affect correctness and comparison fairness.

## Decision

### Spatial-temporal encoder

The canonical encoder uses two causal spatial-temporal residual blocks. Each block applies:

1. a causal gated temporal convolution with kernel width 3;
2. a Chebyshev graph convolution of order `K=3` over the fixed scaled graph Laplacian;
3. a second causal gated temporal convolution with kernel width 3;
4. layer normalization over channel and node features, dropout, and a residual projection when dimensions differ.

The input is `[B, 12, 325, 2]`: one normalized value channel and one binary observation-mask channel. The two blocks use 64 channels throughout, dropout is `0.1`, and temporal convolutions preserve length by left padding only. A final linear projection produces a 64-dimensional context per node and time step. No sensor-ID embedding is used in the canonical model.

### Masked-input representation

Native missingness and experimental masking are combined only for model visibility: `M_obs = M_native AND M_experiment`, where `1` means observed. After training-only normalization, every unobserved numeric input is set to `0.0` and the separate binary `M_obs` channel is always supplied. No forward fill, backward fill, interpolation, or target-derived imputation is used at the model boundary. Native and experimental masks remain separately persisted for auditing and group metrics.

### Latent state-space model

Each sensor has a 32-dimensional Gaussian latent state. The prior transition is a diagonal Gaussian whose mean and log-scale are produced by a GRU transition from the previous latent state and current encoder context. During training, a bidirectional GRU recognition network consumes encoder context and observed target values with their masks and parameterizes a diagonal-Gaussian filtering/smoothing posterior. The posterior may use targets only inside the training objective; validation and test forecasts sample exclusively from the prior.

Forecasting is autoregressive for 12 steps. At each step the decoder consumes the previous sampled latent state, encoder summary, horizon embedding, and previous predicted mean. Teacher forcing is used only during training, with a probability that decays linearly from `1.0` to `0.0` over the first 50% of the maximum training epochs. Validation and test use no teacher forcing.

The emission is a diagonal Gaussian over normalized speed. Its mean is unconstrained; its scale is `softplus(raw_scale) + 1e-4`. Log-scale parameters are clamped to `[-8, 5]` before conversion. Reported traffic-unit predictions and metrics are inverse-transformed with the frozen training scaler.

### Objective

Optimize the negative evidence lower bound: masked Gaussian reconstruction NLL plus `beta * KL(q || p)`, averaged over observed training targets, sensors, horizons, and batch items. Do not add an MSE term. `beta` increases linearly from `0` to `1` over the first 20 training epochs and remains `1` thereafter. Report reconstruction NLL and KL separately, and reject non-finite loss, parameters, or gradients.

## Consequences

The architecture and masking interface are testable and configuration-ready before Phase 6. Zero after normalization is not treated as evidence of observation because the mask is always present. The Gaussian emission represents conditional/aleatoric uncertainty; it is not, by itself, evidence of epistemic uncertainty.
