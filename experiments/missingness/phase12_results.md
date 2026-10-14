# Phase 12 — Robustness condition study (locked results)

**Generated:** 2026-10-14T12:29:32.706625+00:00

Locked test-set results across 10/20/30% sensor-node missingness conditions.
Every number is read from the run manifests listed under [Provenance](#provenance).
Regenerate with `python scripts/aggregate_phase12.py` and verify with `--check`.

Values are **mean +/- sample standard deviation across the three canonical seeds**
(2026, 2027, 2028). Point metrics are in physical units (mph).

## Evaluation scope

- Split: `test` (10,403 windows, 325 sensors, 12 horizons)
- Data artifact: `processed_metadata.json` MD5 `ad5a3a8254a02c2da2df261f6c2fb305`
- Split tensors digest: `3d4a62c1ed05f0b9d2555f53ef592cc6690866729f92e29d67cf404c6087490f`

## Table — 10% masking condition (all sensors, primary arm)

| Model | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | 2.489 +/- 0.047 | 5.711 +/- 0.104 | 5.633 +/- 0.166 | -- | -- | -- | -- |
| Deterministic ST-GCN | 2.267 +/- 0.043 | 5.080 +/- 0.152 | 5.368 +/- 0.224 | -- | -- | -- | -- |
| ST-DSSM (ours) | 2.444 +/- 0.179 | 5.092 +/- 0.144 | 5.609 +/- 0.166 | 4.431 +/- 2.433 | 1.833 +/- 0.102 | 0.9337 +/- 0.0272 | 12.309 +/- 1.857 |

## Table — 20% masking condition (all sensors, primary arm)

| Model | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | 2.813 +/- 0.013 | 6.212 +/- 0.043 | 6.520 +/- 0.091 | -- | -- | -- | -- |
| Deterministic ST-GCN | 2.830 +/- 0.141 | 5.953 +/- 0.031 | 6.678 +/- 0.054 | -- | -- | -- | -- |
| ST-DSSM (ours) | 2.945 +/- 0.471 | 5.904 +/- 0.410 | 6.720 +/- 0.564 | 6.675 +/- 4.931 | 2.216 +/- 0.250 | 0.9035 +/- 0.0491 | 13.125 +/- 3.296 |

## Table — 30% masking condition (all sensors, primary arm)

| Model | MAE (mph) | RMSE (mph) | MAPE (%) | NLL (nats) | CRPS (mph) | PICP | MPIW (mph) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | 3.115 +/- 0.029 | 6.645 +/- 0.081 | 7.375 +/- 0.164 | -- | -- | -- | -- |
| Deterministic ST-GCN | 3.250 +/- 0.217 | 6.562 +/- 0.117 | 7.730 +/- 0.239 | -- | -- | -- | -- |
| ST-DSSM (ours) | 3.415 +/- 0.704 | 6.587 +/- 0.572 | 7.789 +/- 0.809 | 7.646 +/- 5.015 | 2.581 +/- 0.362 | 0.8696 +/- 0.0662 | 13.640 +/- 4.159 |

## Masked vs unmasked degradation (MAE, mph)

| Model | Condition | All | Masked | Unmasked |
| :--- | :--- | ---: | ---: | ---: |
| Historical Persistence | 10% | 2.489 +/- 0.047 | 5.451 +/- 0.527 | 2.166 +/- 0.008 |
| Historical Persistence | 20% | 2.813 +/- 0.013 | 5.389 +/- 0.068 | 2.169 +/- 0.003 |
| Historical Persistence | 30% | 3.115 +/- 0.029 | 5.255 +/- 0.163 | 2.192 +/- 0.030 |
| Deterministic ST-GCN | 10% | 2.267 +/- 0.043 | 5.349 +/- 0.774 | 1.930 +/- 0.039 |
| Deterministic ST-GCN | 20% | 2.830 +/- 0.141 | 5.479 +/- 0.228 | 2.168 +/- 0.188 |
| Deterministic ST-GCN | 30% | 3.250 +/- 0.217 | 5.285 +/- 0.390 | 2.371 +/- 0.311 |
| ST-DSSM (ours) | 10% | 2.444 +/- 0.179 | 5.915 +/- 0.799 | 2.065 +/- 0.129 |
| ST-DSSM (ours) | 20% | 2.945 +/- 0.471 | 5.986 +/- 1.397 | 2.184 +/- 0.241 |
| ST-DSSM (ours) | 30% | 3.415 +/- 0.704 | 5.946 +/- 1.496 | 2.322 +/- 0.363 |

## Per-seed values

| Model | Condition | Seed | All MAE | Masked MAE | Unmasked MAE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Historical Persistence | 10% | 2026 | 2.479 | 5.409 | 2.159 |
| Historical Persistence | 10% | 2027 | 2.448 | 4.946 | 2.175 |
| Historical Persistence | 10% | 2028 | 2.541 | 5.997 | 2.163 |
| Historical Persistence | 20% | 2026 | 2.806 | 5.362 | 2.166 |
| Historical Persistence | 20% | 2027 | 2.805 | 5.338 | 2.172 |
| Historical Persistence | 20% | 2028 | 2.829 | 5.467 | 2.169 |
| Historical Persistence | 30% | 2026 | 3.148 | 5.443 | 2.158 |
| Historical Persistence | 30% | 2027 | 3.092 | 5.152 | 2.203 |
| Historical Persistence | 30% | 2028 | 3.105 | 5.169 | 2.214 |
| Deterministic ST-GCN | 10% | 2026 | 2.276 | 5.343 | 1.941 |
| Deterministic ST-GCN | 10% | 2027 | 2.220 | 4.578 | 1.963 |
| Deterministic ST-GCN | 10% | 2028 | 2.305 | 6.126 | 1.887 |
| Deterministic ST-GCN | 20% | 2026 | 2.964 | 5.615 | 2.301 |
| Deterministic ST-GCN | 20% | 2027 | 2.843 | 5.216 | 2.250 |
| Deterministic ST-GCN | 20% | 2028 | 2.683 | 5.605 | 1.953 |
| Deterministic ST-GCN | 30% | 2026 | 3.411 | 5.694 | 2.426 |
| Deterministic ST-GCN | 30% | 2027 | 3.334 | 4.916 | 2.651 |
| Deterministic ST-GCN | 30% | 2028 | 3.004 | 5.246 | 2.036 |
| ST-DSSM (ours) | 10% | 2026 | 2.267 | 4.994 | 1.970 |
| ST-DSSM (ours) | 10% | 2027 | 2.626 | 6.418 | 2.211 |
| ST-DSSM (ours) | 10% | 2028 | 2.438 | 6.332 | 2.013 |
| ST-DSSM (ours) | 20% | 2026 | 2.557 | 4.761 | 2.006 |
| ST-DSSM (ours) | 20% | 2027 | 3.469 | 7.508 | 2.459 |
| ST-DSSM (ours) | 20% | 2028 | 2.809 | 5.689 | 2.089 |
| ST-DSSM (ours) | 30% | 2026 | 2.883 | 4.859 | 2.031 |
| ST-DSSM (ours) | 30% | 2027 | 4.214 | 7.653 | 2.729 |
| ST-DSSM (ours) | 30% | 2028 | 3.147 | 5.327 | 2.205 |

## Ablation — Masked-training ablation (apply_to_training: true)

Trained and validated with the same sensors withheld, so these runs are a different model from the primary condition runs and must not be compared head-to-head with other models (ADR-0011).

`Primary` repeats the inference-only cell from the tables above so the
two arms can be read side by side. `Delta` is ablation minus primary:
negative means training under the mask helped.

| Model | Condition | Group | Primary MAE | Ablation MAE | Delta |
| :--- | :--- | :--- | ---: | ---: | ---: |
| ST-DSSM (ours) | 20% | all | 2.945 +/- 0.471 | 2.108 +/- 0.018 | -0.837 |
| ST-DSSM (ours) | 20% | masked | 5.986 +/- 1.397 | 2.841 +/- 0.044 | -3.145 |
| ST-DSSM (ours) | 20% | unmasked | 2.184 +/- 0.241 | 1.924 +/- 0.017 | -0.260 |

## Provenance

| Model | Arm | Condition | Seed | Run manifest | Manifest SHA256 | Code commit |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| Deterministic ST-GCN | primary | 10% | 2026 | `experiments/missingness/baseline-st_gcn-test-seed2026-node10-seed2026-predict-20261013-190308_manifest.json` | `c99db9d349066a01...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 10% | 2027 | `experiments/missingness/baseline-st_gcn-test-seed2027-node10-seed2027-predict-20261013-191028_manifest.json` | `3919d016b8241f52...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 10% | 2028 | `experiments/missingness/baseline-st_gcn-test-seed2028-node10-seed2028-predict-20261013-191752_manifest.json` | `4d4c0e926543f7cc...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 20% | 2026 | `experiments/missingness/baseline-st_gcn-test-seed2026-node20-seed2026-predict-20261013-190519_manifest.json` | `347c154f473f5484...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 20% | 2027 | `experiments/missingness/baseline-st_gcn-test-seed2027-node20-seed2027-predict-20261013-191301_manifest.json` | `e6eee5be2d18202b...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 20% | 2028 | `experiments/missingness/baseline-st_gcn-test-seed2028-node20-seed2028-predict-20261013-192019_manifest.json` | `91144e39150d8af1...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 30% | 2026 | `experiments/missingness/baseline-st_gcn-test-seed2026-node30-seed2026-predict-20261013-190756_manifest.json` | `0db94b8603e3ac9e...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 30% | 2027 | `experiments/missingness/baseline-st_gcn-test-seed2027-node30-seed2027-predict-20261013-191534_manifest.json` | `c0922fce73b13c82...` | `6620418ba880` |
| Deterministic ST-GCN | primary | 30% | 2028 | `experiments/missingness/baseline-st_gcn-test-seed2028-node30-seed2028-predict-20261013-192248_manifest.json` | `394272ba2011c7ea...` | `6620418ba880` |
| Historical Persistence | primary | 10% | 2026 | `experiments/missingness/baseline-persistence-test-node10-seed2026-20261013-190241_manifest.json` | `91f2651ecab86d37...` | `6620418ba880` |
| Historical Persistence | primary | 10% | 2027 | `experiments/missingness/baseline-persistence-test-node10-seed2027-20261013-191003_manifest.json` | `32e21f51d1fa2225...` | `6620418ba880` |
| Historical Persistence | primary | 10% | 2028 | `experiments/missingness/baseline-persistence-test-node10-seed2028-20261013-191733_manifest.json` | `c1a13ab46522b655...` | `6620418ba880` |
| Historical Persistence | primary | 20% | 2026 | `experiments/missingness/baseline-persistence-test-node20-seed2026-20261013-201920_manifest.json` | `4331c07c07e1da21...` | `16895ef8bdc5` |
| Historical Persistence | primary | 20% | 2027 | `experiments/missingness/baseline-persistence-test-node20-seed2027-20261013-191234_manifest.json` | `1aa67ad1701d5a32...` | `6620418ba880` |
| Historical Persistence | primary | 20% | 2028 | `experiments/missingness/baseline-persistence-test-node20-seed2028-20261013-191955_manifest.json` | `e6840f740010c089...` | `6620418ba880` |
| Historical Persistence | primary | 30% | 2026 | `experiments/missingness/baseline-persistence-test-node30-seed2026-20261013-190731_manifest.json` | `b24b614f2de840c6...` | `6620418ba880` |
| Historical Persistence | primary | 30% | 2027 | `experiments/missingness/baseline-persistence-test-node30-seed2027-20261013-191508_manifest.json` | `a344b4fd18f9604b...` | `6620418ba880` |
| Historical Persistence | primary | 30% | 2028 | `experiments/missingness/baseline-persistence-test-node30-seed2028-20261013-192223_manifest.json` | `8abde10895fb76ba...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 10% | 2026 | `experiments/missingness/st_dssm-test-seed2026-node10-seed2026-predict-20261013-190353_manifest.json` | `4e6d7db88ad5f7bc...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 10% | 2027 | `experiments/missingness/st_dssm-test-seed2027-node10-seed2027-predict-20261013-191112_manifest.json` | `d98444de0e7d9c93...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 10% | 2028 | `experiments/missingness/st_dssm-test-seed2028-node10-seed2028-predict-20261013-191836_manifest.json` | `11d8edd885339d11...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 20% | 2026 | `experiments/missingness/st_dssm-test-seed2026-node20-seed2026-predict-20261013-190606_manifest.json` | `7a4cdbdbe5893d6d...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 20% | 2027 | `experiments/missingness/st_dssm-test-seed2027-node20-seed2027-predict-20261013-191345_manifest.json` | `fa184dbacd58abee...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 20% | 2028 | `experiments/missingness/st_dssm-test-seed2028-node20-seed2028-predict-20261013-192103_manifest.json` | `62216438c157df5b...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 30% | 2026 | `experiments/missingness/st_dssm-test-seed2026-node30-seed2026-predict-20261013-190840_manifest.json` | `3aaa9c35eae9a431...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 30% | 2027 | `experiments/missingness/st_dssm-test-seed2027-node30-seed2027-predict-20261013-191618_manifest.json` | `3e1cee17f1997c05...` | `6620418ba880` |
| ST-DSSM (ours) | primary | 30% | 2028 | `experiments/missingness/st_dssm-test-seed2028-node30-seed2028-predict-20261013-192331_manifest.json` | `d00904a2fe052b99...` | `6620418ba880` |
| ST-DSSM (ours) | masked-training | 20% | 2026 | `experiments/missingness/st_dssm-test-seed2026-node20-seed2026-masktrain-20261013-205242_manifest.json` | `9981d03f23e36c80...` | `4ff72868ac21` |
| ST-DSSM (ours) | masked-training | 20% | 2027 | `experiments/missingness/st_dssm-test-seed2027-node20-seed2027-masktrain-20261014-011429_manifest.json` | `6aed150ee31f218f...` | `4ff72868ac21` |
| ST-DSSM (ours) | masked-training | 20% | 2028 | `experiments/missingness/st_dssm-test-seed2028-node20-seed2028-masktrain-20261014-054209_manifest.json` | `168a60599b39a48e...` | `4ff72868ac21` |

**Environments:** the runs above were not all executed under one installed stack, so each is recorded with the runs that used it.

1. matplotlib 3.10.8, numpy 1.26.4, python 3.12.10, scipy 1.17.1, torch 2.12.1+cu130 — 4 run(s):

   - `baseline-persistence-test-node20-seed2026-20261013-201920`
   - `st_dssm-test-seed2026-node20-seed2026-masktrain-20261013-205242`
   - `st_dssm-test-seed2027-node20-seed2027-masktrain-20261014-011429`
   - `st_dssm-test-seed2028-node20-seed2028-masktrain-20261014-054209`

2. matplotlib 3.11.1, numpy 2.5.2, python 3.12.10, scipy 1.18.0, torch 2.6.0+cu124 — 26 run(s):

   - `baseline-persistence-test-node10-seed2026-20261013-190241`
   - `baseline-persistence-test-node10-seed2027-20261013-191003`
   - `baseline-persistence-test-node10-seed2028-20261013-191733`
   - `baseline-persistence-test-node20-seed2027-20261013-191234`
   - `baseline-persistence-test-node20-seed2028-20261013-191955`
   - `baseline-persistence-test-node30-seed2026-20261013-190731`
   - `baseline-persistence-test-node30-seed2027-20261013-191508`
   - `baseline-persistence-test-node30-seed2028-20261013-192223`
   - `baseline-st_gcn-test-seed2026-node10-seed2026-predict-20261013-190308`
   - `baseline-st_gcn-test-seed2026-node20-seed2026-predict-20261013-190519`
   - `baseline-st_gcn-test-seed2026-node30-seed2026-predict-20261013-190756`
   - `baseline-st_gcn-test-seed2027-node10-seed2027-predict-20261013-191028`
   - `baseline-st_gcn-test-seed2027-node20-seed2027-predict-20261013-191301`
   - `baseline-st_gcn-test-seed2027-node30-seed2027-predict-20261013-191534`
   - `baseline-st_gcn-test-seed2028-node10-seed2028-predict-20261013-191752`
   - `baseline-st_gcn-test-seed2028-node20-seed2028-predict-20261013-192019`
   - `baseline-st_gcn-test-seed2028-node30-seed2028-predict-20261013-192248`
   - `st_dssm-test-seed2026-node10-seed2026-predict-20261013-190353`
   - `st_dssm-test-seed2026-node20-seed2026-predict-20261013-190606`
   - `st_dssm-test-seed2026-node30-seed2026-predict-20261013-190840`
   - `st_dssm-test-seed2027-node10-seed2027-predict-20261013-191112`
   - `st_dssm-test-seed2027-node20-seed2027-predict-20261013-191345`
   - `st_dssm-test-seed2027-node30-seed2027-predict-20261013-191618`
   - `st_dssm-test-seed2028-node10-seed2028-predict-20261013-191836`
   - `st_dssm-test-seed2028-node20-seed2028-predict-20261013-192103`
   - `st_dssm-test-seed2028-node30-seed2028-predict-20261013-192331`
