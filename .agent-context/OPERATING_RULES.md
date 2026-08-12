# Agent operating rules

1. Read `docs/README.md`, this context package, and any task-relevant ADR before changing files.
2. Inspect existing code and tests; describe current behavior accurately. Never claim planned work is implemented.
3. If a requirement is unclear, add/update an open question or propose an ADR—do not guess.
4. Keep each change small and focused. Update docs, tests, configuration, and run metadata together.
5. Use branch names `codex/<area>-<description>` and conventional descriptive commits. Open a PR for review; do not self-approve or merge scientific/design decisions.
6. Do not commit raw data, processed tensors, checkpoints, results, secrets, or machine-specific paths.
7. For an experiment, save the Git revision, resolved config, dataset manifest, seeds, masks, metrics, units, and artefact paths.
8. Before reporting a result, verify that validation—not test—data selected the configuration and that mask/target leakage tests pass.
9. Follow `docs/REPOSITORY_STRUCTURE.md`: production code belongs in `src/st_dssm/`, configs in `configs/`, tests in `tests/`, and generated artefacts in ignored locations.
