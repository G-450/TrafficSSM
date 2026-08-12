# Architecture

## Status and design boundary

This is the **accepted target architecture**, not an implementation claim. No data pipeline, graph preprocessing, or model is currently implemented; see [current state](CURRENT_STATE.md).

## System flow

```text
raw PEMS-BAY + adjacency
  -> validation and training-only scaling
  -> historical tensor X and target tensor Y
  -> observation/missingness mask M
  -> spatial-temporal encoder
  -> probabilistic latent state-space model
  -> probabilistic forecast head
  -> mean, scale (and optional samples)
  -> metrics, calibration, and reproducible reports
```

## Interfaces

| Component | Input | Output | Required invariants |
|---|---|---|---|
| Data module | Raw `[T, N]`, graph `[N, N]` | Window tensors `[B, L, N, 1]`, `[B, H, N, 1]` | `N=325`; chronological split; scaler fit only on train |
| Masking module | Input window and mask specification | masked input and binary observation mask | ground truth is never used to fill masked values |
| ST encoder | `[B, L, N, F]`, graph, observation mask | embeddings | node ordering equals graph ordering |
| DSSM | embeddings, observation mask | latent prior/posterior parameters | stable, finite parameters; train/eval semantics documented |
| Forecast head | latent representation | `mu`, positive `scale` shaped `[B, H, N, 1]` | scale has a lower numerical bound |
| Evaluator | targets, prediction distribution | per-horizon and aggregate metrics | values inverse-transformed before traffic-unit reporting |

## Target modeling design

The spatial encoder uses the fixed PEMS-BAY adjacency matrix and the causal gated temporal/Chebyshev graph-convolution blocks fixed by [ADR-0007](DECISIONS/ADR-0007-model-and-masking-specification.md). Operators will be generated and tested during implementation; no prior generated operator is retained as evidence.

The DSSM will model a latent state over the input/forecast timeline. It must define:

- transition distribution `p(z_t | z_{t-1}, c_t)`;
- inference/posterior distribution `q(z_t | z_{t-1}, c_t, y_t, m_t)` during training where appropriate;
- emission/forecast distribution `p(y_{t+1:t+H} | z_t, c_t)`;
- the role of the observation mask `m_t`.

A Gaussian predictive distribution is the accepted output contract: `y ~ Normal(mu, sigma)`, where `sigma` is strictly positive. It captures conditional/aleatoric dispersion. [ADR-0008](DECISIONS/ADR-0008-baseline-and-final-evaluation-protocol.md) excludes an additional epistemic method from canonical scope, so the project makes no epistemic-uncertainty claim.

## Loss and inference contract

The model optimizes the negative ELBO fixed by ADR-0007: observed-target Gaussian reconstruction NLL plus an annealed diagonal-Gaussian KL term. A separate MSE term is not added. The same decision fixes teacher forcing and autoregressive multi-step decoding.

## Missingness contract

Masking occurs on the **model input at inference**, with target values retained only by the evaluator. The main mask unit is a sensor node across the complete historical input window; this represents a sensor outage rather than independent random cells. A mask seed and exact sensor IDs must be saved. Per ADR-0007, unobserved normalized inputs are set to zero and a separate binary observation-mask channel is always supplied; native and experimental masks remain separately auditable.

## Failure handling

- Reject shape or node-order mismatch.
- Reject NaN/infinite model outputs, loss, or metrics.
- Fail runs whose dataset manifest, config, or mask specification is missing.
- Mark interrupted runs as incomplete; do not aggregate their partial metrics with completed runs.
