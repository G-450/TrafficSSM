# Phase 13 — Calibration and robustness review (locked results)

**Generated:** 2026-10-14T12:53:21.442332+00:00

Calibration diagnostics, per-horizon error analysis and measured inference
latency for the locked ST-DSSM model, under the normal condition and all
three missingness conditions. Every number here is recomputed from the
prediction artifacts and run manifests listed under [Provenance](#provenance);
`python scripts/aggregate_phase13.py --check` re-derives them.

Only ST-DSSM is diagnosed: calibration is a property of a predictive
distribution and the two baselines are point forecasters (ADR-0008). They
appear in the latency record below and in the Phase 10/12 accuracy tables.

## Temperature scaling (secondary analysis, ADR-0008)

One positive scalar per seed, shared across sensors and horizons, fitted on the validation split by Gaussian NLL and frozen before test evaluation (ADR-0008). Predictive means are never altered.

`T > 1` means the raw forecast was over-confident in the NLL sense and the
fit widens the predictive scale. The fit sees the validation split only.

| Seed | Fitted T | Val NLL raw | Val NLL calibrated | Val interval ECE raw | Val interval ECE calibrated |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2026 | 1.0766 | 2.1448 | 2.1390 | 0.0653 | 0.0838 |
| 2027 | 1.0473 | 2.1429 | 2.1407 | 0.0718 | 0.0831 |
| 2028 | 1.0844 | 2.1404 | 2.1334 | 0.0639 | 0.0840 |

Across-seed temperature: **1.0694 +/- 0.0196**.

## Calibration under each condition (test set)

Values are **mean +/- sample standard deviation across the three canonical
seeds** [2026, 2027, 2028]. `interval ECE` is the mean absolute gap
between empirical and nominal coverage over levels [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]; `PIT ECE` is the
total-variation distance between the PIT histogram and the uniform. Both are
0 for a perfectly calibrated forecast. MAE is identical in both arms because
temperature scaling cannot move a predictive mean.

### Normal (0% missing)

| Group | Arm | NLL | CRPS (mph) | PICP@95% | MPIW (mph) | interval ECE | PIT ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | raw | 2.1171 +/- 0.0055 | 1.431 +/- 0.007 | 0.9612 +/- 0.0013 | 11.070 +/- 0.297 | 0.0724 +/- 0.0037 | 0.1487 +/- 0.0071 |
| all | calibrated | 2.1146 +/- 0.0043 | 1.444 +/- 0.005 | 0.9681 +/- 0.0010 | 11.834 +/- 0.103 | 0.0887 +/- 0.0008 | 0.1755 +/- 0.0046 |

### 10% missing

| Group | Arm | NLL | CRPS (mph) | PICP@95% | MPIW (mph) | interval ECE | PIT ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | raw | 4.4313 +/- 2.4326 | 1.833 +/- 0.102 | 0.9337 +/- 0.0272 | 12.309 +/- 1.857 | 0.0606 +/- 0.0153 | 0.1639 +/- 0.0410 |
| all | calibrated | 4.1084 +/- 2.0486 | 1.843 +/- 0.102 | 0.9414 +/- 0.0241 | 13.141 +/- 1.734 | 0.0734 +/- 0.0141 | 0.1820 +/- 0.0391 |
| masked | raw | 24.7865 +/- 24.5912 | 4.662 +/- 0.579 | 0.6629 +/- 0.2123 | 15.192 +/- 7.651 | 0.2637 +/- 0.1837 | 0.3822 +/- 0.1293 |
| masked | calibrated | 21.5353 +/- 20.7627 | 4.638 +/- 0.569 | 0.6818 +/- 0.2025 | 16.149 +/- 7.806 | 0.2419 +/- 0.1808 | 0.3655 +/- 0.1249 |
| unmasked | raw | 2.2082 +/- 0.0367 | 1.524 +/- 0.081 | 0.9633 +/- 0.0071 | 11.994 +/- 1.228 | 0.0794 +/- 0.0119 | 0.1816 +/- 0.0371 |
| unmasked | calibrated | 2.2051 +/- 0.0387 | 1.538 +/- 0.080 | 0.9697 +/- 0.0046 | 12.812 +/- 1.080 | 0.0953 +/- 0.0075 | 0.2064 +/- 0.0290 |

### 20% missing

| Group | Arm | NLL | CRPS (mph) | PICP@95% | MPIW (mph) | interval ECE | PIT ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | raw | 6.6746 +/- 4.9314 | 2.216 +/- 0.250 | 0.9035 +/- 0.0491 | 13.125 +/- 3.296 | 0.0500 +/- 0.0100 | 0.1858 +/- 0.0538 |
| all | calibrated | 6.0346 +/- 4.1558 | 2.222 +/- 0.250 | 0.9125 +/- 0.0452 | 13.995 +/- 3.243 | 0.0592 +/- 0.0105 | 0.1957 +/- 0.0553 |
| masked | raw | 24.2240 +/- 24.7281 | 4.652 +/- 0.673 | 0.6654 +/- 0.2106 | 15.319 +/- 9.090 | 0.2775 +/- 0.1600 | 0.4060 +/- 0.0942 |
| masked | calibrated | 21.0480 +/- 20.8726 | 4.626 +/- 0.673 | 0.6852 +/- 0.2013 | 16.267 +/- 9.327 | 0.2545 +/- 0.1586 | 0.3865 +/- 0.0931 |
| unmasked | raw | 2.2873 +/- 0.0530 | 1.607 +/- 0.150 | 0.9630 +/- 0.0091 | 12.576 +/- 1.870 | 0.0798 +/- 0.0147 | 0.2023 +/- 0.0562 |
| unmasked | calibrated | 2.2813 +/- 0.0639 | 1.621 +/- 0.149 | 0.9693 +/- 0.0064 | 13.427 +/- 1.758 | 0.0958 +/- 0.0106 | 0.2252 +/- 0.0490 |

### 30% missing

| Group | Arm | NLL | CRPS (mph) | PICP@95% | MPIW (mph) | interval ECE | PIT ECE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | raw | 7.6458 +/- 5.0152 | 2.581 +/- 0.362 | 0.8696 +/- 0.0662 | 13.640 +/- 4.159 | 0.0507 +/- 0.0271 | 0.2067 +/- 0.0533 |
| all | calibrated | 6.8791 +/- 4.2083 | 2.583 +/- 0.360 | 0.8805 +/- 0.0615 | 14.536 +/- 4.150 | 0.0504 +/- 0.0172 | 0.2107 +/- 0.0551 |
| masked | raw | 19.8660 +/- 16.6933 | 4.619 +/- 0.687 | 0.6574 +/- 0.1990 | 14.746 +/- 8.629 | 0.2936 +/- 0.1341 | 0.4215 +/- 0.0722 |
| masked | calibrated | 17.3470 +/- 14.0303 | 4.591 +/- 0.686 | 0.6787 +/- 0.1894 | 15.660 +/- 8.851 | 0.2701 +/- 0.1328 | 0.4007 +/- 0.0731 |
| unmasked | raw | 2.3701 +/- 0.0750 | 1.701 +/- 0.222 | 0.9612 +/- 0.0096 | 13.163 +/- 2.301 | 0.0766 +/- 0.0132 | 0.2134 +/- 0.0638 |
| unmasked | calibrated | 2.3600 +/- 0.0879 | 1.716 +/- 0.220 | 0.9676 +/- 0.0069 | 14.051 +/- 2.220 | 0.0927 +/- 0.0103 | 0.2360 +/- 0.0566 |

## Reliability by nominal level

Signed coverage gap (empirical minus nominal) for the `all` target group.
A positive gap means the interval is too wide at that level.

| Condition | Arm | 50% | 60% | 70% | 80% | 90% | 95% | 99% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | raw | +0.1311 | +0.1287 | +0.1116 | +0.0802 | +0.0364 | +0.0112 | -0.0076 |
| normal | calibrated | +0.1612 | +0.1568 | +0.1353 | +0.0980 | +0.0472 | +0.0181 | -0.0046 |
| 10% | raw | +0.1061 | +0.1025 | +0.0847 | +0.0528 | +0.0084 | -0.0163 | -0.0319 |
| 10% | calibrated | +0.1353 | +0.1299 | +0.1081 | +0.0704 | +0.0194 | -0.0086 | -0.0276 |
| 20% | raw | +0.0609 | +0.0570 | +0.0418 | +0.0145 | -0.0249 | -0.0465 | -0.0569 |
| 20% | calibrated | +0.0894 | +0.0844 | +0.0659 | +0.0333 | -0.0126 | -0.0375 | -0.0512 |
| 30% | raw | +0.0098 | +0.0035 | -0.0110 | -0.0329 | -0.0638 | -0.0804 | -0.0835 |
| 30% | calibrated | +0.0372 | +0.0305 | +0.0138 | -0.0123 | -0.0496 | -0.0695 | -0.0758 |

## Error analysis: horizon decay under each condition

Per-horizon values are read from the run manifests, not recomputed here.
MAE in mph for the `all` target group.

| Condition | 15 min | 30 min | 45 min | 60 min |
| :--- | ---: | ---: | ---: | ---: |
| normal | 1.457 +/- 0.008 | 1.960 +/- 0.014 | 2.325 +/- 0.018 | 2.631 +/- 0.020 |
| 10% | 1.876 +/- 0.068 | 2.438 +/- 0.159 | 2.888 +/- 0.281 | 3.274 +/- 0.385 |
| 20% | 2.298 +/- 0.159 | 2.918 +/- 0.435 | 3.442 +/- 0.706 | 3.886 +/- 0.914 |
| 30% | 2.703 +/- 0.219 | 3.374 +/- 0.658 | 3.956 +/- 1.065 | 4.442 +/- 1.367 |

### Withheld vs observed sensors

The experimental mask withholds model inputs; every target is still scored
(ADR-0004), so a condition splits into withheld (`masked`) and observed
(`unmasked`) sensor groups over one set of predictions.

| Condition | masked MAE | unmasked MAE | ratio | masked PICP@95% | unmasked PICP@95% |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 10% | 5.915 +/- 0.799 | 2.065 +/- 0.129 | 2.86x | 0.6629 +/- 0.2123 | 0.9633 +/- 0.0071 |
| 20% | 5.986 +/- 1.397 | 2.184 +/- 0.241 | 2.74x | 0.6654 +/- 0.2106 | 0.9630 +/- 0.0091 |
| 30% | 5.946 +/- 1.496 | 2.322 +/- 0.363 | 2.56x | 0.6574 +/- 0.1990 | 0.9612 +/- 0.0096 |

### Coverage decay across the horizon

Empirical coverage of the nominal 95% interval, by horizon and target group.
Coverage is **worst at the shortest horizon** under missingness, which inverts
the usual intuition: at 5 minutes the predictive scale is normally very small,
because the next step is close to the last observation, so withholding that
observation leaves a confident interval with nothing to justify it. By 60
minutes the scale is wide regardless and the missing input matters less.

| Condition | Group | 5 min | 15 min | 30 min | 60 min |
| :--- | :--- | ---: | ---: | ---: | ---: |
| normal | all | 0.957 | 0.961 | 0.961 | 0.963 |
| 10% | all | 0.905 | 0.926 | 0.935 | 0.947 |
| 10% | masked | 0.408 | 0.581 | 0.679 | 0.789 |
| 10% | unmasked | 0.959 | 0.963 | 0.963 | 0.965 |
| 20% | all | 0.845 | 0.888 | 0.908 | 0.930 |
| 20% | masked | 0.389 | 0.583 | 0.686 | 0.794 |
| 20% | unmasked | 0.959 | 0.964 | 0.963 | 0.964 |
| 30% | all | 0.782 | 0.846 | 0.876 | 0.910 |
| 30% | masked | 0.370 | 0.574 | 0.678 | 0.793 |
| 30% | unmasked | 0.960 | 0.963 | 0.961 | 0.960 |

## Masked-training ablation (apply_to_training: true)

These runs retrain with the same sensors withheld, so they are a
different model from a primary condition run and are reported separately
(ADR-0011). `Primary` repeats the inference-only cell above.

| Condition | Group | Arm | Primary NLL | Ablation NLL | Primary interval ECE | Ablation interval ECE |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| 20% | all | raw | 6.6746 +/- 4.9314 | 2.2792 +/- 0.0202 | 0.0500 +/- 0.0100 | 0.0617 +/- 0.0031 |
| 20% | all | calibrated | 6.0346 +/- 4.1558 | 2.2625 +/- 0.0192 | 0.0592 +/- 0.0105 | 0.0785 +/- 0.0043 |
| 20% | masked | raw | 24.2240 +/- 24.7281 | 2.9906 +/- 0.1180 | 0.2775 +/- 0.1600 | 0.0426 +/- 0.0055 |
| 20% | masked | calibrated | 21.0480 +/- 20.8726 | 2.9158 +/- 0.1027 | 0.2545 +/- 0.1586 | 0.0552 +/- 0.0088 |
| 20% | unmasked | raw | 2.2873 +/- 0.0530 | 2.1014 +/- 0.0105 | 0.0798 +/- 0.0147 | 0.0688 +/- 0.0027 |
| 20% | unmasked | calibrated | 2.2813 +/- 0.0639 | 2.0992 +/- 0.0101 | 0.0958 +/- 0.0106 | 0.0855 +/- 0.0028 |

## Inference latency

One forward pass producing 12 horizons for all 325 sensors from one window
of history. Measured with warmup and, on CUDA, device synchronization around
every timed region; the median and p95 are reported because latency
distributions are right-skewed. Training cost is recorded in the Phase 10/12
run manifests, not here.

### accelerator — NVIDIA GeForce RTX 4060 Laptop GPU (cuda)

torch 2.12.1+cu130, batch size 64, 10 warmup + 30 timed batches.

| Model | Parameters | median ms/window | p95 ms/window | ms/batch | windows/s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | 0 | 0.0006 | 0.0012 | 0.04 | 1,775,312.2 |
| Deterministic ST-GCN | 109,260 | 1.1159 | 1.2232 | 71.42 | 896.1 |
| ST-DSSM (ours) | 397,058 | 1.6104 | 1.7323 | 103.06 | 621.0 |

### cpu — Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (cpu)

torch 2.12.1+cu130, batch size 64, 10 warmup + 30 timed batches.

| Model | Parameters | median ms/window | p95 ms/window | ms/batch | windows/s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Historical Persistence | 0 | 0.0005 | 0.0006 | 0.03 | 1,984,495.8 |
| Deterministic ST-GCN | 109,260 | 26.5401 | 27.2900 | 1698.56 | 37.7 |
| ST-DSSM (ours) | 397,058 | 25.7403 | 26.1296 | 1647.38 | 38.8 |

## Provenance

Phase 13 recomputes calibration from the predictive distribution, so both
the manifest and the prediction artifact are checksummed. Prediction
artifacts are git-ignored; regenerate them with the Phase 10/12/13 runners.

| Kind | Condition | Arm | Seed | Run manifest | Manifest SHA256 | Predictions SHA256 | Code commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| validation | normal | primary | 2026 | `experiments/calibration/st_dssm-val-seed2026-predict-20261014-120550_manifest.json` | `77199f1e72dd2f18...` | `814397d88f36799c...` | `684c363281e1` |
| validation | normal | primary | 2027 | `experiments/calibration/st_dssm-val-seed2027-predict-20261014-120632_manifest.json` | `a0afdd82be3d01f6...` | `2a44bdaca3b189c2...` | `684c363281e1` |
| validation | normal | primary | 2028 | `experiments/calibration/st_dssm-val-seed2028-predict-20261014-120709_manifest.json` | `7e74ab863b2de719...` | `b4378c72b7c1107e...` | `684c363281e1` |
| test | 10% | primary | 2026 | `experiments/missingness/st_dssm-test-seed2026-node10-seed2026-predict-20261013-190353_manifest.json` | `4e6d7db88ad5f7bc...` | `04635873462b7cef...` | `6620418ba880` |
| test | 10% | primary | 2027 | `experiments/missingness/st_dssm-test-seed2027-node10-seed2027-predict-20261013-191112_manifest.json` | `d98444de0e7d9c93...` | `aecf0ce5288cb5af...` | `6620418ba880` |
| test | 10% | primary | 2028 | `experiments/missingness/st_dssm-test-seed2028-node10-seed2028-predict-20261013-191836_manifest.json` | `11d8edd885339d11...` | `2f321fc0648839e6...` | `6620418ba880` |
| test | 20% | masked-training | 2026 | `experiments/missingness/st_dssm-test-seed2026-node20-seed2026-masktrain-20261013-205242_manifest.json` | `9981d03f23e36c80...` | `8cda7cd05808882f...` | `4ff72868ac21` |
| test | 20% | masked-training | 2027 | `experiments/missingness/st_dssm-test-seed2027-node20-seed2027-masktrain-20261014-011429_manifest.json` | `6aed150ee31f218f...` | `5058803f0e190f99...` | `4ff72868ac21` |
| test | 20% | masked-training | 2028 | `experiments/missingness/st_dssm-test-seed2028-node20-seed2028-masktrain-20261014-054209_manifest.json` | `168a60599b39a48e...` | `3542ce786ec74ec6...` | `4ff72868ac21` |
| test | 20% | primary | 2026 | `experiments/missingness/st_dssm-test-seed2026-node20-seed2026-predict-20261013-190606_manifest.json` | `7a4cdbdbe5893d6d...` | `48fd7d4604689127...` | `6620418ba880` |
| test | 20% | primary | 2027 | `experiments/missingness/st_dssm-test-seed2027-node20-seed2027-predict-20261013-191345_manifest.json` | `fa184dbacd58abee...` | `657954eaf83c9911...` | `6620418ba880` |
| test | 20% | primary | 2028 | `experiments/missingness/st_dssm-test-seed2028-node20-seed2028-predict-20261013-192103_manifest.json` | `62216438c157df5b...` | `0d1c46cf0fc2ff97...` | `6620418ba880` |
| test | 30% | primary | 2026 | `experiments/missingness/st_dssm-test-seed2026-node30-seed2026-predict-20261013-190840_manifest.json` | `3aaa9c35eae9a431...` | `2e52edbd95afcb9d...` | `6620418ba880` |
| test | 30% | primary | 2027 | `experiments/missingness/st_dssm-test-seed2027-node30-seed2027-predict-20261013-191618_manifest.json` | `3e1cee17f1997c05...` | `6f32ed97dceb36a3...` | `6620418ba880` |
| test | 30% | primary | 2028 | `experiments/missingness/st_dssm-test-seed2028-node30-seed2028-predict-20261013-192331_manifest.json` | `d00904a2fe052b99...` | `b30962288dec7184...` | `6620418ba880` |
| test | normal | primary | 2026 | `experiments/normal/st_dssm-test-seed2026-20261012-020607_manifest.json` | `a6b38879ae068184...` | `df6684d12f0fedae...` | `28d04bd71a62` |
| test | normal | primary | 2027 | `experiments/normal/st_dssm-test-seed2027-20261012-153815_manifest.json` | `6c93f0c8c251ee2a...` | `58f37ee703b0c765...` | `28d04bd71a62` |
| test | normal | primary | 2028 | `experiments/normal/st_dssm-test-seed2028-20261012-200224_manifest.json` | `491c4d5c3be35b14...` | `2146760399fbc4ce...` | `28d04bd71a62` |

**Run environments:** the source runs were not all executed under one installed stack, so each is recorded with the runs that used it.

1. matplotlib 3.10.8, numpy 1.26.4, python 3.12.10, scipy 1.17.1, torch 2.12.1+cpu — 3 run(s):

   - `st_dssm-test-seed2026-20261012-020607`
   - `st_dssm-test-seed2027-20261012-153815`
   - `st_dssm-test-seed2028-20261012-200224`

2. matplotlib 3.10.8, numpy 1.26.4, python 3.12.10, scipy 1.17.1, torch 2.12.1+cu130 — 6 run(s):

   - `st_dssm-test-seed2026-node20-seed2026-masktrain-20261013-205242`
   - `st_dssm-test-seed2027-node20-seed2027-masktrain-20261014-011429`
   - `st_dssm-test-seed2028-node20-seed2028-masktrain-20261014-054209`
   - `st_dssm-val-seed2026-predict-20261014-120550`
   - `st_dssm-val-seed2027-predict-20261014-120632`
   - `st_dssm-val-seed2028-predict-20261014-120709`

3. matplotlib 3.11.1, numpy 2.5.2, python 3.12.10, scipy 1.18.0, torch 2.6.0+cu124 — 9 run(s):

   - `st_dssm-test-seed2026-node10-seed2026-predict-20261013-190353`
   - `st_dssm-test-seed2026-node20-seed2026-predict-20261013-190606`
   - `st_dssm-test-seed2026-node30-seed2026-predict-20261013-190840`
   - `st_dssm-test-seed2027-node10-seed2027-predict-20261013-191112`
   - `st_dssm-test-seed2027-node20-seed2027-predict-20261013-191345`
   - `st_dssm-test-seed2027-node30-seed2027-predict-20261013-191618`
   - `st_dssm-test-seed2028-node10-seed2028-predict-20261013-191836`
   - `st_dssm-test-seed2028-node20-seed2028-predict-20261013-192103`
   - `st_dssm-test-seed2028-node30-seed2028-predict-20261013-192331`

Calibration diagnostics were recomputed in the environment recorded in the
latency record above, which may differ from the environment of the original
runs; the recomputation reads saved predictions and does not re-run a model.

