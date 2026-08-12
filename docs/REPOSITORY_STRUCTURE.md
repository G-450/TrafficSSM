# Repository structure

## Contract

This layout is the required home for project material. New files must be placed according to their role; do not recreate root-level one-off scripts, configurations, reports, or notebooks without a documented reason.

```text
.
├── .agent-context/          # compact, durable instructions for AI agents
├── .github/                 # future issue/PR/CI configuration
├── artifacts/               # generated checkpoints and result outputs (ignored)
│   ├── checkpoints/
│   └── results/
├── configs/                 # version-controlled, named experiment configuration
├── data/
│   ├── raw/                 # acquired source data (ignored except .gitkeep)
│   └── processed/           # derived tensors (large tensors ignored; metadata may be tracked)
├── docs/                    # canonical human/project documentation
│   └── DECISIONS/           # architecture decision records
├── experiments/             # small, versioned run manifests and summaries
├── notebooks/               # exploratory work only; no production logic
├── src/
│   └── st_dssm/             # importable application package
│       ├── cli/             # supported command-line entry points
│       ├── data.py          # loading and preprocessing interfaces
│       └── graph.py         # graph operators
├── tests/                   # automated tests mirroring source responsibilities
├── .gitignore
├── pyproject.toml           # packaging, dependencies, tools, entry points
└── README.md                # concise repository entry point
```

## Placement rules

| Material | Required location | Version-control rule |
|---|---|---|
| Reusable Python implementation | `src/st_dssm/` | tracked |
| Command entry point | `src/st_dssm/cli/` | tracked |
| Test | `tests/` | tracked |
| Experiment configuration | `configs/` | tracked; never overwrite a published config |
| Raw dataset | `data/raw/` | ignored; record provenance/checksum in a manifest |
| Derived tensors | `data/processed/` | tensors ignored; small metadata may be tracked |
| Checkpoints/plots/result arrays | `artifacts/` | ignored; cite immutable run metadata instead |
| Run manifests/small summaries | `experiments/<study>/` | tracked when they do not expose data or become unwieldy |
| Research documentation | `docs/` | tracked; canonical |
| AI-agent working context | `.agent-context/` | tracked and kept concise |
| Exploratory analysis | `notebooks/` | tracked only when reproducible and cleaned of outputs/data |

## Commands

Install for development from the repository root:

```bash
pip install -e ".[dev]"
```

There is intentionally no runnable pipeline yet. The agent creates the first Phase 1 configuration and command entry point only after the data-provenance contract is complete.

`pyproject.toml` is the dependency and packaging source of truth. Do not add a second dependency list unless an external deployment tool requires one, in which case generate it from the locked environment and document why.
