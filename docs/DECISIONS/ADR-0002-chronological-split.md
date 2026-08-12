# ADR-0002: Use a 70/10/20 chronological split

**Status:** Accepted
**Date:** 2026-08-12

## Decision

Split the complete ordered series into 70% training, 10% validation, and 20% test partitions without shuffling. Fit data-dependent preprocessing only on training observations.

## Consequences

This preserves temporal order and prevents validation/test statistics leaking into training. Any alternative split is a new experiment and must be separately named rather than replacing the canonical benchmark.
