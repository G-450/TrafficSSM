# Architecture decision records

Decision records are binding once marked **Accepted**. They capture context, decision, consequences, and reversal conditions. New records use the next zero-padded number and are never silently edited to rewrite history; supersede them with a new record when necessary.

| Record | Status | Decision |
|---|---|---|
| [ADR-0001](ADR-0001-pems-bay-only.md) | Accepted | Use PEMS-BAY only. |
| [ADR-0002](ADR-0002-chronological-split.md) | Accepted | Use a 70/10/20 chronological split. |
| [ADR-0003](ADR-0003-window-boundaries.md) | Accepted | Generate windows within each split. |
| [ADR-0004](ADR-0004-missingness-protocol.md) | Accepted | Primary experiment is 20% reproducible node-level input masking. |
| [ADR-0005](ADR-0005-model-specification-gate.md) | Superseded | Freeze ST encoder and DSSM inference design before implementation. |
| [ADR-0006](ADR-0006-dataset-provenance-and-zero-policy.md) | Accepted | Pin Zenodo PEMS-BAY files and treat zero/NaN as native missingness. |
| [ADR-0007](ADR-0007-model-and-masking-specification.md) | Accepted | Fix the canonical ST-DSSM architecture, objective, and masked-input representation. |
| [ADR-0008](ADR-0008-baseline-and-final-evaluation-protocol.md) | Accepted | Fix baselines, seeds, compute/stopping rules, calibration, and uncertainty claims. |
