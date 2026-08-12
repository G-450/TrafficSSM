# ADR-0006: Pin the PEMS-BAY release and treat zero as native missingness

**Status:** Accepted
**Date:** 2026-08-13

## Context

The prior Phase 1 run recorded 521 zero-valued observations (0.003075978% of all values) and no NaNs, while the code reported zeros as missing but filled only NaNs. This was an inconsistent and undocumented data policy.

## Decision

Use the [Zenodo PEMS-BAY release 4263971](https://zenodo.org/records/4263971) as the canonical raw-data distribution. Use only this matched file set:

| File | Required MD5 |
|---|---|
| `pems-bay.h5` | `bfa47e6ee7cc2d665e9b62c0c95c0b41` |
| `adj_mx_bay.pkl` | `55d25daceac847312b7748c59ded6f77` |
| `pems-bay-meta.h5` | `fbde3d20ac413e29f304d0cb528cfdc2` |

For this project, `0.0` and `NaN` are dataset-native missing observations. They must be represented in a native-missingness mask and must not be treated as observed traffic speeds.

## Consequences

The discarded prior Phase 1 code was not compliant because it only imputed NaNs after reporting zeros as missing. The new data-hardening implementation must correctly handle both kinds of native missingness, avoid future-information leakage, and test the policy before declaring Phase 1 complete. Experimental sensor masking remains a distinct, additional mask.
