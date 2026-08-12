# ADR-0001: Use PEMS-BAY only

**Status:** Accepted
**Date:** 2026-08-12

## Context

Earlier concept material mentioned both METR-LA and PEMS-BAY, while the Phase 1 implementation and final specification use PEMS-BAY.

## Decision

All implementation, baselines, and final claims use PEMS-BAY only: 325 traffic-speed sensors with its matching graph.

## Consequences

The project is focused and reproducible, but its conclusions are limited to this dataset/setup. METR-LA must not appear as an active dataset requirement or be mixed into preprocessing.
