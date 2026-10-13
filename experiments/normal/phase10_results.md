# Phase 10 — Normal-condition study (locked results)

**Generated:** 2026-10-13T03:39:00.671891+00:00

Locked test-set results at 0% masking. Every number is read from the run
manifests listed under [Provenance](#provenance); regenerate this file with
`python scripts/aggregate_phase10.py` and verify it with `--check`.

Values are **mean ± sample standard deviation across the three canonical seeds**
(2026, 2027, 2028). Historical Persistence is deterministic and has no seed, so it
shows a single value. Point metrics are in physical units (mph); PICP and MPIW are
for the 95% nominal prediction interval.

## Evaluation scope

- Split: `test` (10,403 windows, 325 sensors, 12 horizons)
- Observed targets scored: 40,569,096
- Masking condition: 0% (normal condition)
- Data artifact: `processed_metadata.json` MD5 `ad5a3a8254a02c2da2df261f6c2fb305`
- Split tensors digest: `3d4a62c1ed05f0b9d2555f53ef592cc6690866729f92e29d67cf404c6087490f`

## Table 1 — Overall test metrics

| Model | Params | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | -- | 2.169 | 5.122 | 4.670 | -- | -- | -- | -- |
| Deterministic ST-GCN | 109,260 | 1.777 ± 0.004 | 4.056 ± 0.008 | 4.064 ± 0.040 | -- | -- | -- | -- |
| ST-DSSM (ours) | 397,058 | 1.936 ± 0.011 | 4.095 ± 0.004 | 4.411 ± 0.108 | 2.117 ± 0.006 | 1.431 ± 0.007 | 0.9612 ± 0.0013 | 11.070 ± 0.297 |

## Table 2 — Point accuracy by horizon

| Model | 15 min MAE / RMSE | 30 min MAE / RMSE | 60 min MAE / RMSE |
| :--- | ---: | ---: | ---: |
| Historical Persistence | 1.59 / 3.39 | 2.17 / 4.95 | 3.04 / 6.98 |
| Deterministic ST-GCN | 1.39 ± 0.00 / 2.92 ± 0.01 | 1.82 ± 0.01 / 4.05 ± 0.00 | 2.33 ± 0.01 / 5.25 ± 0.02 |
| ST-DSSM (ours) | 1.46 ± 0.01 / 2.93 ± 0.01 | 1.96 ± 0.01 / 4.08 ± 0.00 | 2.63 ± 0.02 / 5.31 ± 0.02 |

## Table 3 — Uncertainty quality by horizon

Nominal interval coverage is 95%; PICP closer to that
value is better calibrated, and MPIW is the mean interval width.

**ST-DSSM (ours)**

| Horizon | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |
| ---: | ---: | ---: | ---: | ---: |
| 5 min | 1.467 ± 0.007 | 0.666 ± 0.005 | 0.9568 ± 0.0052 | 4.98 ± 0.23 |
| 10 min | 1.725 ± 0.004 | 0.901 ± 0.004 | 0.9601 ± 0.0022 | 6.90 ± 0.20 |
| 15 min | 1.884 ± 0.004 | 1.079 ± 0.005 | 0.9607 ± 0.0020 | 8.28 ± 0.21 |
| 20 min | 1.998 ± 0.004 | 1.224 ± 0.006 | 0.9609 ± 0.0017 | 9.41 ± 0.19 |
| 25 min | 2.089 ± 0.004 | 1.346 ± 0.007 | 0.9613 ± 0.0015 | 10.39 ± 0.25 |
| 30 min | 2.163 ± 0.005 | 1.452 ± 0.009 | 0.9614 ± 0.0012 | 11.21 ± 0.29 |
| 35 min | 2.226 ± 0.006 | 1.549 ± 0.010 | 0.9616 ± 0.0011 | 11.98 ± 0.34 |
| 40 min | 2.280 ± 0.007 | 1.636 ± 0.012 | 0.9617 ± 0.0010 | 12.65 ± 0.39 |
| 45 min | 2.329 ± 0.007 | 1.718 ± 0.012 | 0.9620 ± 0.0010 | 13.31 ± 0.40 |
| 50 min | 2.374 ± 0.007 | 1.795 ± 0.013 | 0.9625 ± 0.0012 | 13.96 ± 0.45 |
| 55 min | 2.416 ± 0.007 | 1.869 ± 0.014 | 0.9627 ± 0.0014 | 14.57 ± 0.49 |
| 60 min | 2.456 ± 0.007 | 1.942 ± 0.015 | 0.9630 ± 0.0017 | 15.20 ± 0.55 |

## Figures

- `plots/phase10_mae_by_horizon.png`
- `plots/phase10_rmse_by_horizon.png`
- `plots/phase10_picp_by_horizon.png`
- `plots/phase10_mpiw_by_horizon.png`
- `plots/phase10_crps_by_horizon.png`

## Per-seed values

| Model | Seed | Best epoch | Epochs run | MAE | RMSE | MAPE | NLL | CRPS | PICP | MPIW |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | -- | -- | -- | 2.1691 | 5.1223 | 4.6697 | -- | -- | -- | -- |
| Deterministic ST-GCN | 2026 | 76 | 91 | 1.7742 | 4.0592 | 4.0241 | -- | -- | -- | -- |
| Deterministic ST-GCN | 2027 | 100 | 100 | 1.7745 | 4.0467 | 4.0651 | -- | -- | -- | -- |
| Deterministic ST-GCN | 2028 | 95 | 100 | 1.7815 | 4.0627 | 4.1032 | -- | -- | -- | -- |
| ST-DSSM (ours) | 2026 | 58 | 60 | 1.9366 | 4.0935 | 4.5188 | 2.1225 | 1.4296 | 0.9615 | 10.9213 |
| ST-DSSM (ours) | 2027 | 51 | 60 | 1.9467 | 4.0927 | 4.4096 | 2.1115 | 1.4395 | 0.9624 | 11.4113 |
| ST-DSSM (ours) | 2028 | 60 | 60 | 1.9253 | 4.1001 | 4.3036 | 2.1175 | 1.4251 | 0.9598 | 10.8760 |

## Provenance

| Model | Seed | Run manifest | Manifest SHA256 | Code commit |
| :--- | ---: | :--- | :--- | :--- |
| Deterministic ST-GCN | 2026 | `experiments/normal/baselines/baseline-st_gcn-test-seed2026-20261010-014123_manifest.json` | `410f12cdac622272…` | `28d04bd71a62` |
| Deterministic ST-GCN | 2027 | `experiments/normal/baselines/baseline-st_gcn-test-seed2027-20261010-051506_manifest.json` | `4df91f31a9e6fdc1…` | `28d04bd71a62` |
| Deterministic ST-GCN | 2028 | `experiments/normal/baselines/baseline-st_gcn-test-seed2028-20261011-015301_manifest.json` | `ee5428e29ee94c8b…` | `28d04bd71a62` |
| Historical Persistence | -- | `experiments/normal/baselines/baseline-persistence-test-20261010-013721_manifest.json` | `ea1ef03db8eb1f06…` | `28d04bd71a62` |
| ST-DSSM (ours) | 2026 | `experiments/normal/st_dssm-test-seed2026-20261012-020607_manifest.json` | `a6b38879ae068184…` | `28d04bd71a62` |
| ST-DSSM (ours) | 2027 | `experiments/normal/st_dssm-test-seed2027-20261012-153815_manifest.json` | `6c93f0c8c251ee2a…` | `28d04bd71a62` |
| ST-DSSM (ours) | 2028 | `experiments/normal/st_dssm-test-seed2028-20261012-200224_manifest.json` | `491c4d5c3be35b14…` | `28d04bd71a62` |

**Environment:** matplotlib 3.10.8, numpy 1.26.4, python 3.12.10, scipy 1.17.1, torch 2.12.1+cpu
