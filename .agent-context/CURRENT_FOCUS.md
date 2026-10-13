# Current focus

Phase 10 normal-condition study is complete and locked: tuning, three seeded test runs per
model, and manifest-derived tables, figures and a machine-readable aggregate in
`experiments/normal/` (`python scripts/aggregate_phase10.py --check` re-derives them).

Phase 11 missingness mechanism is complete: the nine canonical masks are published under
`experiments/masks/`, masks are wired through every predictor's input boundary
(`masking:` config block or `--mask-ratio/--mask-seed/--mask-path/--mask-training`), a
condition is run against a finished checkpoint with `--predict-from`, and every metric is
reported over `all`/`masked`/`unmasked` target groups with the mask recorded in the run
manifest. The leakage gate is `tests/test_phase11_masking.py`.

The next focus is Phase 12 — Robustness study: execute the nine conditions (10/20/30% x
mask seeds 2026/2027/2028) for ST-DSSM, deterministic ST-GCN and historical persistence,
lock the condition tables the way Phase 10 locked the normal condition, and run the
ablations isolating graph context, state-space inference and the masking representation.
Per ADR-0011 a condition is a forward pass over the test split, not a training run, so the
matrix does not need the GPU; the masked-training ablation does, and the installed torch is
currently CPU-only.

See `docs/IMPLEMENTATION_PLAN.md` for gates and sequencing, and consult the accepted
decision records before making a material design choice. Record any newly discovered
ambiguity in `docs/OPEN_QUESTIONS.md`.
