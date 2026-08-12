# ADR-0003: Generate windows within split boundaries

**Status:** Accepted
**Date:** 2026-08-12

## Context

The current preprocessing implementation creates windows separately for train, validation, and test partitions.

## Decision

Retain this split-local window protocol for the canonical benchmark: no input/target window crosses a partition boundary.

## Consequences

This is conservative and leakage-safe, but discards up to `input_length + forecast_horizon - 1` candidate windows at each partition boundary. All models and baselines must use the same derived data. A context-carrying evaluation protocol would be a separately versioned study.
