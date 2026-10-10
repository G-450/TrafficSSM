"""Hyperparameter Tuning Script for Phase 10 ST-DSSM.

Tunes the hyperparameters against Validation NLL with a predefined search record.
Outputs the best configuration to be used for the normal-condition study.
"""

import argparse
import copy
import glob
import os
import sys

import yaml

from st_dssm.cli.train import train_st_dssm
from st_dssm.result_schema import load_run_manifest

# Predefined search grid (validation-NLL selection only; never tuned against test).
GRID = [
    {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1},
    {"learning_rate": 5e-4, "weight_decay": 1e-5, "dropout": 0.2},
    {"learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1},
]

# Fixed seed for tuning only, distinct from the canonical seed schedule
# (2026, 2027, 2028) used for the locked normal-condition study, so that
# tuning never shares a seed with a reported result.
TUNING_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 Tuning Script")
    parser.add_argument("--base-config", type=str, default="configs/train.yaml")
    parser.add_argument("--output-dir", type=str, default="experiments/tuning")
    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: Base config file not found at {args.base_config}")
        sys.exit(1)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    best_nll = float("inf")
    best_cfg_path = None

    print(f"Starting Phase 10 Tuning over {len(GRID)} configurations...")

    for i, params in enumerate(GRID):
        print(f"\n--- Tuning Run {i + 1}/{len(GRID)} ---")
        print(f"Params: {params}")

        cfg = copy.deepcopy(base_config)
        cfg.setdefault("training", {})
        cfg.setdefault("model", {})
        cfg["training"]["learning_rate"] = params["learning_rate"]
        cfg["training"]["weight_decay"] = params["weight_decay"]
        cfg["model"]["dropout"] = params["dropout"]
        cfg["seed"] = TUNING_SEED
        cfg["split"] = "val"

        temp_cfg_path = os.path.join(args.output_dir, f"temp_config_{i}.yaml")
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        run_output_dir = os.path.join(args.output_dir, f"run_{i}")
        run_checkpoint_dir = os.path.join(args.output_dir, f"checkpoints_{i}")

        try:
            # Re-running the script skips finished grid points and resumes an
            # interrupted one from its last saved epoch.
            finished = sorted(glob.glob(os.path.join(run_output_dir, "*_manifest.json")))
            if finished:
                print(f"Run {i} already complete: {finished[-1]}")
                val_nll = load_run_manifest(finished[-1]).overall_metrics["NLL"]
            else:
                result = train_st_dssm(
                    config=cfg,
                    output_dir=run_output_dir,
                    checkpoint_dir=run_checkpoint_dir,
                    save_plots=False,
                    resume_path="auto",
                )
                val_nll = result["manifest"].overall_metrics["NLL"]
            print(f"Run {i} Val NLL: {val_nll}")
            if val_nll < best_nll:
                best_nll = val_nll
                best_cfg_path = temp_cfg_path
        except Exception as e:  # noqa: BLE001 - one bad config must not abort the grid
            print(f"Error during tuning run {i}: {e}")
            continue

    print("\n" + "=" * 60)
    if best_cfg_path:
        print(f"Tuning Complete. Best Val NLL: {best_nll}")
        print(f"Best configuration saved at: {best_cfg_path}")
        with open(best_cfg_path, "r", encoding="utf-8") as f:
            tuned_cfg = yaml.safe_load(f)
        # seed and split were only for tuning; the locked config must not carry them.
        tuned_cfg.pop("seed", None)
        tuned_cfg.pop("split", None)
        with open("configs/tuned.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(tuned_cfg, f)
        print("Locked configuration copied to configs/tuned.yaml")
    else:
        print("Tuning failed to find a valid configuration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
