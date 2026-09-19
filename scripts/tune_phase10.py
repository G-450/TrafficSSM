"""Hyperparameter Tuning Script for Phase 10 ST-DSSM.

Tunes the hyperparameters against Validation NLL with a predefined search record.
Outputs the best configuration to be used for the normal-condition study.
"""

import argparse
import os
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Tuning Script")
    parser.add_argument("--base-config", type=str, default="configs/train.yaml")
    parser.add_argument("--output-dir", type=str, default="experiments/tuning")
    args = parser.parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: Base config file not found at {args.base_config}")
        sys.exit(1)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Predefined search grid
    grid = [
        {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1},
        {"learning_rate": 5e-4, "weight_decay": 1e-5, "dropout": 0.2},
        {"learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    best_nll = float('inf')
    best_cfg = None
    seed = 42  # Use a fixed seed for tuning

    print(f"Starting Phase 10 Tuning over {len(grid)} configurations...")

    for i, params in enumerate(grid):
        print(f"\n--- Tuning Run {i+1}/{len(grid)} ---")
        print(f"Params: {params}")

        # Update config
        cfg = base_config.copy()
        if "training" not in cfg:
            cfg["training"] = {}
        if "model" not in cfg:
            cfg["model"] = {}
        
        cfg["training"]["learning_rate"] = params["learning_rate"]
        cfg["training"]["weight_decay"] = params["weight_decay"]
        cfg["model"]["dropout"] = params["dropout"]
        
        # Save temporary config
        temp_cfg_path = os.path.join(args.output_dir, f"temp_config_{i}.yaml")
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        # Run training
        # Assuming early stopping and we parse the final NLL from stdout or logs
        # For simplicity, we just run the training. In a real setup, we would read the metrics.json
        cmd = [
            "st-dssm-train",
            "--config", temp_cfg_path,
            "--seed", str(seed),
            "--output-dir", os.path.join(args.output_dir, f"run_{i}"),
            "--checkpoint-dir", os.path.join(args.output_dir, f"checkpoints_{i}"),
            "--split", "val" # Tune on validation
        ]

        try:
            # We don't want to block the terminal fully or we want to capture output
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Parse output for Best Val NLL
            val_nll = None
            for line in res.stdout.split('\n'):
                if "Best Val NLL:" in line:
                    val_nll = float(line.split("Best Val NLL:")[1].strip())
            
            if val_nll is not None:
                print(f"Run {i} Best Val NLL: {val_nll}")
                if val_nll < best_nll:
                    best_nll = val_nll
                    best_cfg = temp_cfg_path
            else:
                print(f"Warning: Could not parse Best Val NLL for run {i}")
        except subprocess.CalledProcessError as e:
            print(f"Error during tuning run {i}: {e.stderr}")
            continue

    print("\n" + "="*60)
    if best_cfg:
        print(f"Tuning Complete. Best Val NLL: {best_nll}")
        print(f"Best configuration saved at: {best_cfg}")
        # Copy best config to configs/tuned.yaml
        import shutil
        shutil.copy(best_cfg, "configs/tuned.yaml")
        print("Locked configuration copied to configs/tuned.yaml")
    else:
        print("Tuning failed to find a valid configuration.")
    print("="*60)

if __name__ == "__main__":
    main()
