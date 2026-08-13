# Open questions

There is currently 1 unresolved project question. New material ambiguities must be recorded here and must not be silently guessed.

| ID | Resolution | Decision record |
|---|---|---|
| OQ-03–OQ-05 | Resolved on 2026-08-13 | [ADR-0007](DECISIONS/ADR-0007-model-and-masking-specification.md) |
| OQ-06–OQ-09 | Resolved on 2026-08-13 | [ADR-0008](DECISIONS/ADR-0008-baseline-and-final-evaluation-protocol.md) |
| OQ-10 | *Open* | The canonical PEMS-BAY dataset contains a non-five-minute timestamp interval jump of 65 minutes on 2017-03-12 03:00:00 UTC, which corresponds to the US Pacific Time Daylight Saving Time transition. Phase 2 cadence validation correctly fails and flags this. Does this require a data-policy decision (e.g., reindexing with NaNs) or should the strict cadence check make an exception for this specific timestamp gap? |
