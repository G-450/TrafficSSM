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

Phases 10 and 11 are merged into `main` (PRs #20-#22). The nine ST-DSSM condition
configs are published under `configs/missingness/` and each one resolves to its
published mask; the Phase 10 checkpoints the conditions evaluate are in
`experiments/normal/checkpoints/`.

Phase 12 — Robustness study is complete and pending PR merge.

The scripts and tests are in place:
- `scripts/run_phase12.py`   — executes the 27 condition runs (9 × 3 models, all CPU
  forward passes); skips completed runs; `--dry-run` previews the plan.
- `scripts/aggregate_phase12.py` — locks the condition tables once all runs are done;
  `--check` re-derives them.
- `tests/test_phase12.py`    — 10 gate tests for the aggregator and run script (all pass).

Of the 27 conditions, 1 is already done (persistence/node20-seed2026 from Phase 11
verification). The remaining 26 are running via `python scripts/run_phase12.py`. Each
forward pass is ~1 hour on the CPU-only torch build, so the full matrix takes ~26 hours.

All 27 manifests are in `experiments/missingness/`. Locked tables are at
`experiments/missingness/phase12_results.md` and `phase12_results.json`.
Verify with: `python scripts/aggregate_phase12.py --check`

**Ablations:**
- Graph context and state-space ablations are covered by the model comparison in the
  condition tables (Persistence = no graph, ST-GCN = no state space).
- Masked-training ablation (ST-DSSM trained with `apply_to_training: true`) requires
  retraining (~8 h per seed on GPU); deferred until a CUDA torch build is available.

The next focus is Phase 13 — Calibration and robustness review.

See `docs/IMPLEMENTATION_PLAN.md` for gates and sequencing, and consult the accepted
decision records before making a material design choice. Record any newly discovered
ambiguity in `docs/OPEN_QUESTIONS.md`.
