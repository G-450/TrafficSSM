# ADR-0011: A missingness condition applies at inference; masked training is an ablation

**Status:** Accepted
**Date:** 2026-10-13
**Refines:** ADR-0004, ADR-0007

## Context

ADR-0004 fixes the condition as withholding sensors "from the model's historical input at inference", while the Phase 11 objective is to wire masks "through training and evaluation". That left one question open: does running a condition mean retraining a model under it, or evaluating a trained model under it?

The two answers measure different things. Evaluating the normal-condition model under masking measures robustness of a fixed model to input loss, which is the question H0/H1 ask. Retraining under a known masking rate measures how well the architecture adapts to a masking rate it was fitted to, which is a question about the masking representation. Retraining is also roughly 8 h per run on the available GPU, against one forward pass for evaluation, so conflating the two would also make the condition matrix unaffordable.

## Decision

1. A condition is applied at the model input boundary of every forward pass over the evaluated split: for each withheld sensor, all input steps of the window are set to `0.0` after normalization and the `M_obs` channel is `0` (ADR-0007). The Phase 10 normal-condition checkpoint is the model under test; `st-dssm-train --predict-from <checkpoint>` and `st-dssm-baseline --model st_gcn --predict-from <checkpoint>` evaluate it without fitting anything. Training and validation inputs stay unmasked.
2. `masking.apply_to_training: true` additionally withholds the same sensors during training and validation. It is reserved for the Phase 12 masking-representation ablation, must be labelled as such in every table, and its runs may not be compared head-to-head with primary condition runs.
3. Targets are never masked. A condition changes what the model sees, not what is scored, so the evaluated target set and its `all`-group counts are identical across conditions. Each condition is reported over three target groups: `all`, `masked` (withheld sensors) and `unmasked` (observed sensors).
4. Run identity carries the condition (`...-seed<seed>-node<level>-seed<mask seed>-...`), so conditions cannot overwrite each other's outputs or resume each other's training state.
5. Masks are published artifacts under `experiments/masks/` with a checksum over the mask vector; a run records `mask_seed`, `mask_condition`, `mask_sensor_ids` and `mask_checksum` in its manifest, and loading a mask artifact verifies both its checksum and its re-derivation from the seed.

## Consequences

Phase 12 executes each of the nine (level, seed) conditions as one forward pass per model over the test split, so the study does not depend on the GPU being reinstated. The primary claims describe fixed models losing input, which is what the research question states. Any claim about training under missingness requires the separately labelled ablation, and the mask artifacts that predate this record remain valid because the mask itself is unchanged (ADR-0010).

Reversal condition: if Phase 12 evidence shows the fixed-model result is dominated by the distribution shift of zeroed inputs rather than by information loss, the masked-training ablation becomes the primary protocol and this record is superseded rather than edited.
