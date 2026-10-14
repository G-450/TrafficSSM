# ST-DSSM — Uncertainty-Aware Spatial-Temporal Forecasting

A probabilistic traffic-forecasting research project on PEMS-BAY: a Spatial-Temporal
Graph Convolutional encoder coupled to a Deep State Space Model with a Gaussian
forecast head (ST-DSSM), measured against a capacity-controlled deterministic
ST-GCN and historical persistence, under controlled sensor missingness.

Every reported number is produced by a committed script from a checksummed run
manifest and can be re-derived with that script's `--check` flag.

## Documentation

The authoritative project documentation is in [docs/](docs/README.md). Begin with:

- [Final report](docs/FINAL_REPORT.md) — the finished study, every number linked to its evidence
- [Current implementation state](docs/CURRENT_STATE.md)
- [Limitations and claim reconciliation](docs/LIMITATIONS.md) — what the results do **not** support
- [Product requirements](docs/PRD.md)
- [Professional implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [End-to-end commands](run.md)
- [AI-agent context package](.agent-context/README.md)

## Current state

**Released as v1.0.0.** All fourteen phases are complete, implemented, verified
and locked; 363 automated tests pass. The finished study is written up in the
[final report](docs/FINAL_REPORT.md), and [current state](docs/CURRENT_STATE.md)
has the detail.

| Deliverable | Locked results | Verify with |
|---|---|---|
| Normal-condition study | [`experiments/normal/phase10_results.md`](experiments/normal/phase10_results.md) | `python scripts/aggregate_phase10.py --check` |
| Robustness condition matrix (30 runs) | [`experiments/missingness/phase12_results.md`](experiments/missingness/phase12_results.md) | `python scripts/aggregate_phase12.py --check` |
| Calibration review, error analysis, latency | [`experiments/calibration/phase13_results.md`](experiments/calibration/phase13_results.md) | `python scripts/aggregate_phase13.py --check` |

### Headline results

Test set, PEMS-BAY, 325 sensors, 12 horizons (5–60 min), mean ± sd over seeds
2026/2027/2028, physical units (mph).

| Model | Normal MAE | 20% missing MAE | Parameters |
|---|---:|---:|---:|
| Historical persistence | 2.17 | 2.813 ± 0.013 | 0 |
| Deterministic ST-GCN | 1.78 | 2.830 ± 0.141 | 109,260 |
| **ST-DSSM** | **1.936 ± 0.011** | 2.945 ± 0.471 | 397,058 |
| ST-DSSM, trained under the mask¹ | — | **2.108 ± 0.018** | 397,058 |

¹ The masked-training ablation is a *different model* from the primary condition
runs and is never averaged into a primary cell (ADR-0011). It requires the
missingness pattern to be known at training time.

**Point accuracy degrades gracefully under sensor loss; the predictive
distribution does not.** At 20% missingness aggregate NLL rises from 2.117 to
6.675, and at the withheld sensors it reaches 24.2 with coverage falling to 0.665
against a nominal 95%. The forecast is over-dispersed in the bulk and
under-dispersed in the tail even in the normal condition, so the validation-fitted
temperature (1.0694 ± 0.0196) lowers NLL while *raising* interval calibration
error: a single scalar cannot repair a shape mismatch. Training under the mask
recovers nearly all of both the accuracy and the calibration loss.

The failure is sharpest where intuition says it should be safest: at a withheld
sensor, the 5-minute 95% interval contains the truth **39%** of the time, against
79% at 60 minutes. Short-horizon intervals are narrow precisely where the evidence
for them is missing, and width alone gives no warning.

> [!IMPORTANT]
> Aggregate interval calibration error *falls* under missingness while NLL
> triples. It must not be quoted as evidence of robust calibration. See
> [LIMITATIONS.md](docs/LIMITATIONS.md) for the full reconciliation of claims,
> including what this project explicitly does not claim.

## Development setup

Create a Python 3.10+ environment and install the project:

```bash
pip install -e ".[dev]"
pytest
```

Raw data and generated processed tensors are intentionally excluded from Git.
Acquire the pinned dataset and build the canonical artifact first:

```bash
st-dssm-provenance --download     # pinned PEMS-BAY, checksum-verified
st-dssm-validate                  # data-contract checks
st-dssm-preprocess generate       # chronologically disjoint splits
```

Then follow [run.md](run.md), which gives the exact commands for every phase from
the baselines through the Phase 13 calibration review. Before a benchmark run,
follow the [data contract](docs/DATA_CONTRACT.md) and record a dataset manifest.

Training needs a CUDA build of PyTorch; at `batch_size: 64` with
`training.micro_batch_size: 32` it fits in ~3.5 GB of GPU memory (~5 min/epoch on
an RTX 4060 Laptop). Condition runs are forward passes and need no GPU.
