# Evaluation foundation and metrics specification

## 1. Overview

Phase 4 establishes the model-independent evaluation foundation for the ST-DSSM project per [ADR-0007](DECISIONS/ADR-0007-model-and-masking-specification.md), [ADR-0008](DECISIONS/ADR-0008-baseline-and-final-evaluation-protocol.md), and [docs/RESEARCH_AND_EXPERIMENTS.md](RESEARCH_AND_EXPERIMENTS.md).

All metrics, inverse transformations, schema definitions, and plotting routines are implemented independently of forecasting models.

---

## 2. Mathematical definitions

### 2.1 Point-forecast metrics

#### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{|V|} \sum_{i \in V} |y_i - \hat{\mu}_i|$$
where $V$ is the set of valid (observed) evaluation targets and $\hat{\mu}_i$ is the predicted mean in physical traffic-speed units (mph).

#### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{|V|} \sum_{i \in V} (y_i - \hat{\mu}_i)^2}$$

#### Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{|V_{\ge \tau}|} \sum_{i \in V_{\ge \tau}} \frac{|y_i - \hat{\mu}_i|}{y_i}$$
where $\tau = 1.0\text{ mph}$ is the speed threshold preventing division by zero and extreme percentage distortion on near-zero traffic observations.

---

### 2.2 Distributional proper scores

#### Gaussian Negative Log-Likelihood (NLL)
$$\text{NLL}(y, \hat{\mu}, \hat{\sigma}) = \frac{1}{|V|} \sum_{i \in V} \left[ \frac{1}{2}\ln(2\pi) + \ln(\hat{\sigma}_i) + \frac{1}{2}\left(\frac{y_i - \hat{\mu}_i}{\hat{\sigma}_i}\right)^2 \right]$$
where $\hat{\sigma}_i > 0$ is the predicted standard deviation.

#### Gaussian Continuous Ranked Probability Score (CRPS)
Closed-form analytical solution:
$$\text{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma \left[ z\left(2\Phi(z) - 1\right) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]$$
where $z = \frac{y - \mu}{\sigma}$, $\Phi(z)$ is the standard normal cumulative distribution function (CDF), and $\phi(z)$ is the standard normal probability density function (PDF).

---

### 2.3 Prediction interval diagnostics

#### Interval bounds
For nominal confidence level $(1 - \alpha)$ (e.g. $\alpha = 0.05$ for $95\%$ PI):
$$\hat{L}_i = \hat{\mu}_i - z_{1 - \alpha/2} \hat{\sigma}_i, \quad \hat{U}_i = \hat{\mu}_i + z_{1 - \alpha/2} \hat{\sigma}_i$$

#### Prediction Interval Coverage Probability (PICP)
$$\text{PICP} = \frac{1}{|V|} \sum_{i \in V} \mathbf{1}_{[\hat{L}_i, \hat{U}_i]}(y_i)$$

#### Mean Prediction Interval Width (MPIW)
$$\text{MPIW} = \frac{1}{|V|} \sum_{i \in V} (\hat{U}_i - \hat{L}_i)$$

---

## 3. Inverse transformation rules

Predictions and variances must be inverse-transformed to physical units ($\text{mph}$) using frozen training-only per-sensor scaler parameters $(\mu_{\text{train}, n}, \sigma_{\text{train}, n})$:

| Parameter | Standardized | Physical transformation |
|---|---|---|
| Location / Mean | $\mu_{\text{norm}}$ | $\mu_{\text{raw}} = \mu_{\text{norm}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$ |
| Standard Deviation (Scale) | $\sigma_{\text{norm}}$ | $\sigma_{\text{raw}} = \sigma_{\text{norm}} \cdot \sigma_{\text{train}}$ |
| Variance | $\text{Var}_{\text{norm}}$ | $\text{Var}_{\text{raw}} = \text{Var}_{\text{norm}} \cdot \sigma_{\text{train}}^2$ |
| Interval Bounds $(L, U)$ | $L_{\text{norm}}, U_{\text{norm}}$ | $L_{\text{raw}} = L_{\text{norm}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$ |

> [!IMPORTANT]
> Standard deviations and interval widths are scale parameters and **must not** have the mean added during inverse transformation.

---

## 4. Result schema and reproducibility

Evaluation results are saved as versioned JSON records matching `st_dssm.result_schema.RunManifest`.

Key schema elements:
- `schema_version`: `"1.0"`
- `run_id`: Unique timestamped identifier
- `git_commit` & `git_dirty`: Code provenance
- `split_evaluated`: e.g. `"test"`
- `manifest_checksum`: Deterministic SHA256 checksum over canonical JSON
- `metrics`: Array of `MetricRecord` items specifying metric name, value, unit (`mph`/`percent`/`ratio`), horizon (1..12 or `"aggregate"`), horizon in minutes (5..60), target group (`"all"`/`"masked"`/`"unmasked"`), and valid observation count.

---

## 5. CLI usage

### Synthetic smoke run (Self-test)
```bash
st-dssm-evaluate --synthetic-smoke --output-dir artifacts/results
```

### Configured evaluation
```bash
st-dssm-evaluate --config configs/evaluation/default.yaml
```
