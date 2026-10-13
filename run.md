### How to Run the Code

**Step 1: Download raw data**
```bash
st-dssm-provenance --download
```

**Step 2: Validate the downloaded data**
```bash
st-dssm-validate --data-dir data/raw
```

**Step 3: Preprocess the data into tensors**
```bash
st-dssm-preprocess generate
```

**Step 4: Run the Baselines**
To quickly test the end-to-end pipeline with synthetic data (without needing to wait for a full dataset load), you can run:
```bash
st-dssm-baseline --synthetic-smoke
```

To run the full `Historical Persistence` baseline on the actual test set:
```bash
st-dssm-baseline --model persistence --artifact-dir data/processed --output-dir experiments/baselines
```

To train and evaluate the `Deterministic ST-GCN` baseline:
```bash
st-dssm-baseline --model st_gcn --artifact-dir data/processed --adj-mx-path data/raw/adj_mx_bay.pkl --output-dir experiments/baselines --checkpoint-dir experiments/checkpoints --epochs 15
```

---
**Step 5: Phase 10 — Normal-condition study**

Tune against validation NLL only, then run the locked configuration once per canonical
seed. Both scripts skip finished work, so re-running them resumes an interrupted study:

```bash
python scripts/tune_phase10.py                  # writes configs/tuned.yaml
python scripts/run_phase10.py                   # seeds 2026, 2027, 2028 -> experiments/normal/
```

Rerun the Phase 5 baselines under the same protocol for the comparison:

```bash
st-dssm-baseline --model persistence --artifact-dir data/processed \
  --output-dir experiments/normal/baselines
st-dssm-baseline --model st_gcn --artifact-dir data/processed \
  --adj-mx-path data/raw/adj_mx_bay.pkl --output-dir experiments/normal/baselines \
  --checkpoint-dir experiments/normal/checkpoints --seed 2026
```

Lock the tables and figures, and verify them later:

```bash
python scripts/aggregate_phase10.py             # -> phase10_results.md / .json / plots
python scripts/aggregate_phase10.py --check     # fails if the tables drifted
```

---
**Step 6: Phase 11 — Missingness mechanism**

Publish (or verify) the canonical masks. They are a deterministic function of the sensor
count and the seed, so `--check` re-derives them rather than redrawing:

```bash
python scripts/generate_masks.py                # -> experiments/masks/node{10,20,30}-seed{2026,2027,2028}.json
python scripts/generate_masks.py --check        # fails if a published mask drifted
```

A condition withholds whole sensors from the model's input history at inference, so it
evaluates the normal-condition checkpoint rather than training a new model (ADR-0004,
ADR-0011). Pair the mask seed with the run's training seed; all nine ST-DSSM condition
configs are published under `configs/missingness/`:

```bash
# ST-DSSM under the primary 20% condition, seed 2026
st-dssm-train --config configs/missingness/node20-seed2026.yaml \
  --predict-from experiments/normal/checkpoints/<st_dssm run_id>_best.pt \
  --output-dir experiments/missingness

# Deterministic ST-GCN baseline under the same mask
st-dssm-baseline --model st_gcn --config configs/baselines/st_gcn.yaml \
  --artifact-dir data/processed --seed 2026 \
  --predict-from experiments/normal/checkpoints/<st_gcn run_id>_best.pt \
  --mask-path experiments/masks/node20-seed2026.json \
  --output-dir experiments/missingness --checkpoint-dir experiments/missingness/checkpoints

# Historical persistence under the same mask (it fits no parameters, so nothing is loaded)
st-dssm-baseline --model persistence --artifact-dir data/processed --seed 2026 \
  --mask-path experiments/masks/node20-seed2026.json \
  --output-dir experiments/missingness
```

`--mask-ratio 0.2 --mask-seed 2026` regenerates the same mask without the artifact file.
`--mask-training` additionally withholds the sensors during training, which is the Phase 12
masking-representation ablation and must be labelled as such.

Each run's manifest records `mask_seed`, `mask_condition`, `mask_sensor_ids` and
`mask_checksum`, and carries every metric three times over: `all`, `masked` (withheld
sensors) and `unmasked`. Targets are never masked, so the `all` group scores exactly the
same targets as the normal condition.

To add the target-group breakdown to predictions that are already saved:

```bash
st-dssm-evaluate --artifact-dir data/processed \
  --predictions-path experiments/missingness/<run_id>_predictions.npz \
  --split test --run-id <run_id> --model-name st_dssm \
  --mask-path experiments/masks/node20-seed2026.json \
  --output-dir experiments/missingness
```

---
**Regenerating a single run's manifest**

Evaluation is deterministic given a run's saved `*_predictions.npz`, so a manifest can be
rebuilt without retraining — useful if a run was interrupted after its predictions were
written. Pass the original run id so the manifest is replaced in place, and read the
training provenance back from the run's checkpoint:

```bash
st-dssm-evaluate \
  --artifact-dir data/processed \
  --predictions-path experiments/normal/<run_id>_predictions.npz \
  --split test --output-dir experiments/normal \
  --run-id <run_id> --model-name st_dssm \
  --provenance-checkpoint experiments/normal/checkpoints/<run_id>_best.pt \
  --selection-metric val_nll --total-epochs 60
```
