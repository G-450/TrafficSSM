# Project context

**Project:** ST-DSSM, uncertainty-aware spatial-temporal traffic forecasting.
**Dataset:** PEMS-BAY only; 325 sensors; five-minute traffic-speed observations.
**Target:** a spatial-temporal encoder plus probabilistic deep state-space model that produces forecast distributions.
**Primary experiment:** 20% reproducible sensor-node input masking at inference; do not force a desired uncertainty outcome.

## Facts currently true

- The implementation was restarted on 2026-08-13. Phases 1-11 (data provenance, contract hardening, preprocessing, evaluation, deterministic baseline, spatial-temporal encoder, forecast head, DSSM integration, training operations, the normal-condition study, and the missingness mechanism) are fully implemented and tested. The Phase 10 study has been run end-to-end on real data and its results are locked in `experiments/normal/`; the masked conditions themselves are Phase 12 work.
- A missingness condition withholds sensors from the model's input at inference and is scored over `all`/`masked`/`unmasked` target groups; the normal-condition checkpoint is the model under test (ADR-0011). Masks are published, checksummed artifacts in `experiments/masks/` and are nested across levels for a given seed (ADR-0010).
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
