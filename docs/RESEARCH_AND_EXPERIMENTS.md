# Research and experiment protocol

## Research question

When traffic-sensor observations are reduced, can ST-DSSM retain useful forecasts while expressing uncertainty whose calibration and width are appropriate?

## Hypotheses

- **H0:** Controlled 20% sensor-node masking does not meaningfully change uncertainty measures relative to no masking.
- **H1:** Controlled 20% masking changes the predictive distribution, potentially increasing interval width, while calibration remains reasonable.

H1 is evaluated from results; no loss, post-processing rule, or chart may artificially enforce widening.

### Verdict (Phase 13, 2026-10-14)

Decided from the locked deliverables, not from expectation. Full reasoning and
every caveat is in [LIMITATIONS.md](LIMITATIONS.md).

| Hypothesis | Verdict | Decisive evidence |
|---|---|---|
| **H0** — masking does not meaningfully change uncertainty | **Rejected** | Aggregate NLL `2.117 -> 6.675` at 20%; `24.224` at withheld sensors |
| **H1** — distribution changes, intervals may widen, calibration stays reasonable | **Partly supported; final clause false** | MPIW `11.070 -> 13.125` mph (widening real), but withheld-sensor interval ECE `0.0724 -> 0.2775` and PICP `0.665` vs nominal 0.95 |

The research question splits: point accuracy degrades gracefully (MAE
`1.936 -> 2.945` mph at 20%), the predictive distribution does not. Calibration
holds at observed sensors and fails at withheld ones, so only the per-group
tables answer the question honestly.

> [!WARNING]
> Aggregate interval ECE *falls* under missingness (`0.0724 -> 0.0500` at 20%)
> while NLL more than triples. The normal-condition forecast is over-dispersed
> in the bulk, so condition-induced widening cancels part of that bias at the
> sampled levels while the tail failures driving NLL sit outside them. Do not
> quote aggregate interval ECE as evidence of robust calibration.

## Conditions

| Condition | Purpose | Status |
|---|---|---|
| 0% node masking | normal-condition reference | complete and locked (Phase 10, `experiments/normal/`) |
| 10% node masking | sensitivity extension | complete and locked (Phase 12, `experiments/missingness/`) |
| 20% node masking | primary research condition | complete and locked (Phase 12); masked-training ablation also complete |
| 30% node masking | sensitivity extension | complete and locked (Phase 12, `experiments/missingness/`) |

The masks for the three masked conditions are published, checksummed artifacts under
`experiments/masks/` (nine files: three levels x mask seeds 2026/2027/2028, nested per
[ADR-0010](DECISIONS/ADR-0010-nested-node-masks.md)). A condition is applied to the
model's input history at inference and never to the targets, so every condition scores
the same target set; see [ADR-0011](DECISIONS/ADR-0011-condition-applies-at-inference.md).

For each condition, mask the same sampled sensor IDs across every compared model for a given seed. Primary unit: a selected sensor is withheld over the complete input history window. Report results separately for masked and unmasked target sensors as well as the overall aggregate.

## Comparisons

The required baseline set is historical persistence plus the capacity-controlled deterministic ST-GCN fixed by [ADR-0008](DECISIONS/ADR-0008-baseline-and-final-evaluation-protocol.md). Additional epistemic or non-spatial models are outside canonical scope. Every baseline uses the same split, horizon, observation information, masks, evaluation code, and inverse transform.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| MAE | mean absolute error of predictive mean | lower |
| RMSE | root mean square error of predictive mean | lower |
| NLL | negative log likelihood under forecast distribution | lower |
| CRPS | proper score for full predictive distribution | lower |
| PICP | empirical coverage of nominal interval | close to nominal |
| MPIW | mean prediction-interval width | interpret with PICP/CRPS, not alone |

Use nominal 95% intervals unless an ADR changes it. Report each metric by horizon (5–60 minutes), target group (all/masked/unmasked), condition, seed, and an aggregate with spread across seeds. Report raw traffic units for MAE/RMSE/interval width; probability metrics must state whether they were calculated in normalized or raw units.

## Reproducibility and statistical discipline

- Use final training seeds `2026`, `2027`, and `2028`; pair experimental-mask seeds with the same values across models and report mean and standard deviation.
- Do not select a model based on test performance. Use validation data for architecture and calibration choices, then make one locked final test evaluation per chosen configuration.
- Treat uncalibrated output as primary. Validation-fitted scalar scale-temperature calibration is allowed only as the separately labelled secondary analysis defined by ADR-0008. It was executed in Phase 13 (`experiments/calibration/`): the fitted temperature is `1.0694 +/- 0.0196`, it lowers NLL and *raises* interval ECE, and the primary results do not depend on it.
- Keep an immutable run manifest containing code commit, package environment, config, dataset manifest/checksum, training seed, masking seed, and output paths.
- Report failed runs and exclusions with reason.

## Claims allowed in the final report

Only claims directly supported by recorded results are allowed. “Wider intervals” is meaningful only alongside coverage, proper scores, and point accuracy. “Epistemic uncertainty” is allowed only if the chosen inference procedure supports it and the evidence is described. Results apply to the fixed PEMS-BAY setup, not all sensor networks.
