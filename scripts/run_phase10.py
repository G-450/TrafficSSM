"""Automation script for Phase 10 Normal-condition study.

This script executes the ST-DSSM training pipeline over the canonical
seeds (2026, 2027, 2028), using the YAML configuration in configs/train.yaml.
It automatically triggers evaluations on the test split for each run and saves
the resulting manifests to experiments/normal/.
"""

import argparse
import os
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Automation Script")
    parser.add_argument("--config", type=str, default="configs/tuned.yaml", help="Path to locked tuned config file")
    parser.add_argument("--output-dir", type=str, default="experiments/normal", help="Output directory for manifests and plots")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/normal/checkpoints", help="Output directory for checkpoints")
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [2026, 2027, 2028])

    print(f"Starting Phase 10 Normal-condition study for seeds: {seeds}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Executing ST-DSSM training for seed {seed}")
        print(f"{'='*60}")

        cmd = [
            "st-dssm-train",
            "--config", args.config,
            "--seed", str(seed),
            "--output-dir", args.output_dir,
            "--checkpoint-dir", args.checkpoint_dir,
            "--split", "test"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"\nSuccessfully completed run for seed {seed}.")
        except subprocess.CalledProcessError as e:
            print(f"\nError: Training failed for seed {seed} with exit code {e.returncode}.")
            sys.exit(e.returncode)
        
    print("\nPhase 10 automation completed successfully.")
    print(f"All run manifests, predictions, and plots are saved in {args.output_dir}.")

if __name__ == "__main__":
    main()
