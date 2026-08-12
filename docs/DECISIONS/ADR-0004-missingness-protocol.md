# ADR-0004: Primary missingness experiment is reproducible node-level input masking

**Status:** Accepted
**Date:** 2026-08-12

## Decision

The primary robustness condition withholds 20% of the 325 sensor nodes (65 sensors) from the model's historical input at inference. The same selected nodes are masked for every input step in each evaluated window. Their original future targets remain available only to the evaluator. 0%, 10%, and 30% conditions are supporting sensitivity comparisons.

## Consequences

Each run must persist a mask seed and selected IDs/checksum. Comparisons use identical masks for a given seed. The numeric representation and required observation-mask channel are subsequently fixed by [ADR-0007](ADR-0007-model-and-masking-specification.md), resolving OQ-05.
