# Project context

**Project:** ST-DSSM, uncertainty-aware spatial-temporal traffic forecasting.
**Dataset:** PEMS-BAY only; 325 sensors; five-minute traffic-speed observations.
**Target:** a spatial-temporal encoder plus probabilistic deep state-space model that produces forecast distributions.
**Primary experiment:** 20% reproducible sensor-node input masking at inference; do not force a desired uncertainty outcome.

## Facts currently true

- The implementation was restarted on 2026-08-13. Phases 1-9 (data provenance, contract hardening, preprocessing, evaluation, deterministic baseline, spatial-temporal encoder, forecast head, DSSM integration, and training operations) are fully implemented and tested.
- Canonical raw data is Zenodo PEMS-BAY release 4263971; `0.0` and `NaN` are native missingness (ADR-0006).
- Canonical detailed documentation is in `docs/`; old root reports are not authoritative.
- Repository placement rules are binding and documented in `docs/REPOSITORY_STRUCTURE.md`.
- All previously recorded design questions are resolved by ADR-0007 and ADR-0008; no open question currently blocks implementation.

## Non-negotiables

- Do not use METR-LA or combine PEMS-BAY sources.
- Never leak validation/test statistics into training, or masked target values into model inputs.
- Treat prediction-interval widening as a hypothesis, not a requirement.
- Do not label conditional output variance as epistemic uncertainty without a documented method/evidence.
- Every material assumption becomes an ADR or open question.
