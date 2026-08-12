# ADR-0005: Freeze the model specification before implementation

**Status:** Superseded by ADR-0007
**Date:** 2026-08-12

## Context

The project specifies the family “ST-GCN + DSSM,” but not the exact encoder blocks, latent-state transition, posterior parameterization, decoder, masking injection, or variational objective. Implementing from an informal label would create unreviewable assumptions and make comparisons irreproducible.

## Proposed decision

Before Phase 6 begins, approve a model-specification record that fixes:

- graph/temporal encoder block type, depth, hidden dimensions, and graph operator;
- latent-state dimension, transition, emission, and inference distributions;
- multi-step decoding and teacher-forcing policy;
- likelihood/output distribution and numerical stability bounds;
- objective terms, KL schedule, and optimizer/training settings;
- missing-value representation and the observation-mask interface;
- whether and how epistemic uncertainty is estimated.

## Consequences

Model code may prototype behind an experiment flag, but no benchmark result or scientific claim may be treated as canonical until this decision is accepted and encoded in configuration/tests.

## Resolution

[ADR-0007](ADR-0007-model-and-masking-specification.md) accepts the gate and supplies the required canonical specification.
