# ST-DSSM — Uncertainty-Aware Spatial-Temporal Forecasting

This repository develops a probabilistic traffic-forecasting research project using PEMS-BAY and a planned Spatial-Temporal Graph Convolutional Network plus Deep State Space Model (ST-DSSM).

## Documentation

The authoritative project documentation is in [docs/](docs/README.md). Begin with:

- [Product requirements](docs/PRD.md)
- [Current implementation state](docs/CURRENT_STATE.md)
- [Professional implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [AI-agent context package](.agent-context/README.md)

## Current state

The project is intentionally restarting from a clean implementation baseline. Documentation, AI-agent context, repository structure, package metadata, and collaboration rules are in place; no data pipeline, model, configuration, experiment result, or automated test is currently implemented. See [current state](docs/CURRENT_STATE.md).

## Development setup

Create a Python 3.10+ environment and install the project:

```bash
pip install -e ".[dev]"
```

The first implementation task is [Phase 1 — Data provenance](docs/IMPLEMENTATION_PLAN.md). Raw data and generated processed tensors are intentionally excluded from Git. Before a benchmark run, follow the [data contract](docs/DATA_CONTRACT.md) and record a dataset manifest.
