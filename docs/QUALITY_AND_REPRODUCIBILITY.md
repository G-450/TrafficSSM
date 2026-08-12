# Quality and reproducibility

## Definition of done

A feature is done only when its implementation, tests, configuration, documentation, and evidence are included in the same reviewable change. “Runs on my machine” is not sufficient.

## Required checks by layer

| Layer | Required checks |
|---|---|
| Data | source manifest, dimensions, sensor/node order, strict cadence, duplicates, missingness policy, no leakage |
| Preprocessing | split ratios/boundaries, train-only scaler, window contents, deterministic artefact structure |
| Graph | finite square adjacency, self-loop policy, Laplacian/Chebyshev recurrence, device movement |
| Model | tensor shapes, positive finite scale, finite loss/gradients, one-batch overfit, checkpoint round trip |
| Masking | deterministic IDs, correct rate, no masked targets in inputs, identical masks across comparisons |
| Metrics | known-value tests, interval coverage cases, raw/normalized unit handling, horizon/group aggregation |
| End-to-end | CPU smoke run, resume/interruption behavior, run-manifest completeness |

## Test structure to introduce

```text
tests/
  test_data_loader.py
  test_preprocessing.py
  test_graph.py
  test_metrics.py
  test_masking.py
  test_model_components.py
  test_end_to_end_smoke.py
```

Use small synthetic fixtures committed to the repository; never require the large raw dataset for ordinary unit tests. A full-data benchmark is a separately labelled integration run.

## Reproducibility record

Every training or evaluation run must write a JSON manifest with:

- UTC start/end time and run status;
- Git commit and dirty-worktree status;
- resolved configuration and package/environment versions;
- dataset manifest identifier/checksum;
- split/window/scaler artefact identifier;
- model and training seed(s);
- mask condition, seed, mask IDs/checksum;
- checkpoint selected and selection metric;
- metric outputs, units, aggregation, and output paths.

Set Python, NumPy, PyTorch CPU/CUDA seeds and deterministic settings as feasible. Record any nondeterministic operation rather than claiming full determinism. Compare numerical outputs with tolerances, not exact equality, where hardware kernels make exact equality unrealistic.

## Scientific quality gates

- Validation selects model configuration; test data is reserved for final evaluation.
- Each final comparison uses training seeds `2026`, `2027`, and `2028`, unless a deviation is formally documented.
- Present mean, spread, sample count, and exclusions.
- Pair interval width with coverage and a proper score.
- Keep result files immutable once cited; corrections create a new run rather than overwriting evidence.

## Security and data hygiene

Do not commit raw PEMS-BAY data, large processed tensors, checkpoints, secrets, or machine-specific paths. Keep source provenance/checksums and small synthetic fixtures under version control. Review `.gitignore` whenever a new artefact class is introduced.
