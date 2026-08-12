# Professional implementation plan

## Delivery principles

Each phase ends with a reviewable pull request, documented evidence, and explicit exit criteria. A later phase may not treat a proposed decision as settled. The project restarted its implementation on 2026-08-13; Phase 0 is complete and Phase 1 is the next work item.

| Phase | Objective | Core deliverables | Exit criteria |
|---:|---|---|---|
| 0 | Governance and repository baseline | canonical docs, agent context, issue/PR templates, decision log | docs approved; legacy status clear |
| 1 | Data provenance | pinned PEMS-BAY manifest, checksums, source-pair evidence | dataset can be identified and reproduced |
| 2 | Data-contract hardening | cadence/alignment/missingness validation tests; zero-value policy | data gate passes automatically |
| 3 | Reproducible preprocessing | versioned processed artefacts, split/window/scaler tests | deterministic rerun matches manifest |
| 4 | Evaluation foundation | metrics library, inverse-transform rules, result schema, plots | synthetic tests validate each metric |
| 5 | Deterministic baseline | simple persistence baseline and graph baseline | baseline normal-condition metrics recorded |
| 6 | Spatial-temporal encoder | tested ST-GCN block, graph/operator interfaces | shape, gradient, and overfit-one-batch tests pass |
| 7 | Probabilistic forecast head | stable distribution parameters, NLL, interval sampling | finite-loss and calibration-sanity tests pass |
| 8 | DSSM integration | transition, inference, emission, ELBO/NLL training loop | end-to-end learning on a small fixture |
| 9 | Training operations | configs, checkpoints, early stopping, run manifests, device handling | interrupted/resumed and deterministic-smoke runs work |
| 10 | Normal-condition study | tuned ST-DSSM and baseline comparisons at 0% masking | locked test table and figures produced |
| 11 | Missingness mechanism | mask generator, masked-input representation, safety tests | no target leakage; masks reproducible/fair |
| 12 | Robustness study | 10/20/30% conditions, seed repetitions, ablations | primary 20% analysis complete |
| 13 | Calibration and robustness review | calibration diagnostics, error analysis, latency record | claims and limitations reconciled with evidence |
| 14 | Final release | report, reproducibility package, clean setup instructions | independent clean-environment rerun checklist passes |

## Detailed work packages

### Phase 0 — Governance and documentation

- Make `docs/` and `.agent-context/` the source of truth.
- Add issue templates, PR template, and a CODEOWNERS/reviewer policy only if the collaboration platform supports them.
- Create ADRs for every unresolved architecture or protocol choice.

**Gate:** A contributor can identify current state, next work, rules, and open questions without reading chat history.

### Phases 1–3 — Data readiness

- Create a `data/manifest` record with source, checksums, licensing note, sensor IDs, graph pairing, and acquisition date.
- Correct and test timestamp-interval validation; verify exact node ordering rather than only dimensions.
- Decide whether source zeros are valid speeds or missing observations; implement that one policy consistently.
- Add unit tests for loaders, split boundaries, scaler fit scope, imputation, windows, and graph recurrence.
- Add a smoke command that regenerates an artefact and compares structural metadata.

**Gate:** No undocumented source, missingness policy, or unchecked alignment can pass into training.

### Phases 4–5 — Evaluation and baselines

- Implement metrics independently of models, using known analytical/synthetic cases.
- Define machine-readable result and run-manifest schemas.
- Implement persistence and then the selected deterministic graph baseline; document its capacity and input treatment.

**Gate:** A baseline can train/evaluate end-to-end and produces a complete result record.

### Phases 6–8 — ST-DSSM core

- Freeze encoder architecture, latent state dimension, output distribution, masking interface, and objective through ADRs/configuration.
- Build each module with strict shape checks and finite-value guards.
- Test individual components, one-batch overfit, deterministic smoke training, and a small end-to-end fixture.
- Diagnose posterior collapse, exploding variance, and unstable gradients before larger training.

**Gate:** ST-DSSM produces valid distributions and improves its training objective on a controlled fixture without leakage.

### Phases 9–10 — Training and normal-condition results

- Implement configuration-driven training, checkpoint selection on validation only, logging, and rerun metadata.
- Tune only against validation metrics with a predefined search record.
- Lock the selected configuration and evaluate normal-condition test performance once per final seed.

**Gate:** Normal-condition tables and figures are reproducible from manifests, not manually assembled.

### Phases 11–13 — Missingness and scientific analysis

- Generate node-level masks with published seeds and IDs; prevent masked ground truth from reaching inputs.
- Execute 10/20/30% conditions using the same masks across models.
- Evaluate all/masked/unmasked targets; compare accuracy, NLL/CRPS, PICP, and width by horizon.
- Conduct ablations that isolate graph context, state-space inference, and masking representation.
- Write limitations first, then only claims the results support.

**Gate:** The primary 20% condition is repeated across seeds and directly traceable to code/config/data/masks.

### Phase 14 — Final release

- Verify a new environment can install dependencies and complete a smoke run.
- Freeze manifests and outputs; tag the release when Git history is clean.
- Finalize documentation and report with links to evidence.

**Gate:** A reviewer can rerun the documented workflow and audit every conclusion.

## Dependency order

`0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14`.

Phases 4 and 5 may overlap once the data contract is stable. Do not begin final missingness claims before the normal-condition study and masking-leakage tests are complete.
