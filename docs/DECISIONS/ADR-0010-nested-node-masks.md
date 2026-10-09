# ADR-0010: Nested node masks across missingness levels

**Status:** Accepted
**Date:** 2026-10-09
**Refines:** ADR-0004

## Context

ADR-0004 fixes 10%, 20% and 30% node-level masking conditions with a persisted mask seed. If each level draws its sensors independently, differences between levels mix the effect of the amount of missingness with the effect of which sensors were selected.

## Decision

For a given mask seed, sensors are ordered by one seeded permutation (`numpy.random.default_rng(seed).permutation(325)`), and a missingness level of `r` masks the first `round(325 * r)` sensors in that order: 32 at 10%, 65 at 20%, 98 at 30%. Masks are therefore nested: every sensor masked at 10% is also masked at 20% and 30%. Implemented in `st_dssm.mask_generator.MaskGenerator`.

## Consequences

Trends across missingness levels within one seed reflect only the amount of withheld input. Variation between seeds still captures sensitivity to which sensors are withheld. Masks generated before this decision are not comparable and must not be used.
