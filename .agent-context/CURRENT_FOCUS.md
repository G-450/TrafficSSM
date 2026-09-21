# Current focus

Phase 9 training operations (`st-dssm-train` CLI, `configs/train.yaml`) is complete and verified: 215/215 tests pass, canonical PEMS-BAY data has been downloaded, verified, and preprocessed.

The next focus is Phase 10 — Normal-condition study: run the predefined hyperparameter search on validation NLL, lock the winning configuration, execute the three canonical seeds (2026, 2027, 2028) on the test split, and produce reproducible tables/figures in `experiments/normal/`. `scripts/tune_phase10.py` and `scripts/run_phase10.py` are hardened (see `docs/CURRENT_STATE.md` "Phase 9 hardening") but had not been executed end-to-end as of 2026-09-21.

**Compute budgeting note:** one training epoch on real PEMS-BAY data took ~40 minutes on an RTX 3050 Laptop GPU (6 GB). The full Phase 10 protocol (3 tuning runs + 3 canonical-seed runs, each up to 100 epochs) is a multi-day compute commitment on hardware of that class — plan for a machine with sustained GPU availability (or a longer background window) before starting, rather than assuming it fits in a single interactive session.

See `docs/IMPLEMENTATION_PLAN.md` for gates and sequencing, and consult the accepted decision records before making a material design choice. Record any newly discovered ambiguity in `docs/OPEN_QUESTIONS.md`.

