# Product requirements document

## 1. Product definition

**Name:** Uncertainty-Aware Time-Series Forecasting via Deep Spatial-Temporal State Space Models (ST-DSSM)
**Domain:** probabilistic spatial-temporal traffic forecasting
**Primary dataset:** PEMS-BAY only
**Primary experiment:** evaluate normal operation versus reproducible 20% sensor-node missingness at inference.

ST-DSSM will forecast future traffic speed at 325 connected sensors from historical observations and a fixed sensor graph. Unlike a point-only forecaster, it must return a predictive distribution whose uncertainty can be evaluated. The research objective is to measure—not force—whether uncertainty responds appropriately when observations are withheld.

## 2. Problem and users

Traffic monitoring systems can lose observations through failures, latency, or outages. A point forecast alone can hide reduced evidence. The direct users of this project are its student/researcher maintainers and evaluators; the intended operational analogue is a traffic operator who needs both forecasts and an honest indication of forecast reliability.

## 3. Goals

1. Build a reproducible PEMS-BAY forecasting pipeline with leakage-safe data handling.
2. Implement an end-to-end spatial-temporal probabilistic model.
3. Compare it against defined baselines under identical data, split, masks, and metrics.
4. Measure point accuracy, distributional quality, calibration, interval width, and degradation under missingness.
5. Produce evidence-backed conclusions with enough metadata for an independent rerun.

## 4. Non-goals and boundaries

- This is not a production traffic-control system.
- It does not claim a new state-space theory or state-of-the-art accuracy by default.
- It does not use METR-LA or mix PEMS-BAY distributions.
- It does not treat wider intervals alone as success; width must be assessed with coverage and CRPS/NLL.
- It does not claim epistemic uncertainty unless the chosen inference method supports and validates that interpretation.

## 5. Functional requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| FR-01 | Load one pinned PEMS-BAY time-series/graph distribution with 325 aligned sensors. | Dataset manifest and validation log |
| FR-02 | Split chronologically 70/10/20 and fit all data-dependent preprocessing on training data only. | Tests and saved scaler metadata |
| FR-03 | Generate configurable 12-step history to 12-step forecast samples by default. | Shape/content tests |
| FR-04 | Produce a forecast mean and distribution parameters for every horizon and sensor. | Model-interface test |
| FR-05 | Train using a defined proper probabilistic objective. | Training config and logs |
| FR-06 | Evaluate normal operation and 0%, 10%, 20%, and 30% inference masking conditions. | Versioned masks and result tables |
| FR-07 | Ensure withheld ground truth never enters model input or imputation at inference. | Masking tests and audit |
| FR-08 | Report MAE, RMSE, NLL, CRPS, PICP, and interval width with aggregation rules. | Evaluation artefact |
| FR-09 | Save code revision, config, data manifest, seed, mask ID, metrics, and artefact paths per run. | Run manifest |

## 6. Success criteria

A successful project has a working, tested pipeline and a reproducible experimental record. Scientific success is judged from measured results: reasonable point accuracy, calibrated intervals near their nominal target, proper-score comparisons, and transparent degradation under missingness. The hypothesis that intervals widen is not an acceptance criterion by itself.

## 7. Stakeholder decisions already made

- Dataset scope is PEMS-BAY only (ADR-0001).
- Split is chronological 70/10/20 (ADR-0002).
- Default task is 60 minutes history to 60 minutes ahead at five-minute cadence (ADR-0003).
- The 20% sensor-node masking condition is primary; 0/10/30% are planned comparison conditions (ADR-0004).

See [open questions](OPEN_QUESTIONS.md) for decisions not yet authorized by evidence.
