"""Automation script for Phase 12 Robustness condition matrix.

Executes the nine missingness conditions (10/20/30% x mask seeds 2026/2027/2028)
for all three models (ST-DSSM, deterministic ST-GCN, historical persistence),
reusing the locked Phase 10 checkpoints via ``--predict-from`` (ADR-0011).

All runs are forward passes over the test split; no training happens and no GPU
is required. Each run writes a manifest to ``experiments/missingness/``.
Re-running skips conditions that already have a manifest.

Usage:
    python scripts/run_phase12.py
    python scripts/run_phase12.py --dry-run          # show which runs would execute
    python scripts/run_phase12.py --ratio 20         # only the 20% conditions
    python scripts/run_phase12.py --seed 2026        # only conditions paired with seed 2026
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Condition matrix
# ---------------------------------------------------------------------------

RATIOS = [10, 20, 30]
SEEDS = [2026, 2027, 2028]

CHECKPOINT_DIR = "experiments/normal/checkpoints"
CONFIG_DIR = "configs/missingness"
OUTPUT_DIR = "experiments/missingness"


def _find_checkpoint(prefix_glob: str) -> str:
    """Return the unique checkpoint matching a glob pattern, or raise."""
    matches = sorted(glob.glob(prefix_glob))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found matching: {prefix_glob}\n"
            "Run scripts/run_phase10.py first to generate Phase 10 checkpoints."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoints matched {prefix_glob}: {matches}\n"
            "Disambiguate manually."
        )
    return matches[0]


def _manifest_exists(pattern: str) -> bool:
    return bool(glob.glob(os.path.join(OUTPUT_DIR, pattern)))


def build_run_list(
    ratios: list[int],
    seeds: list[int],
) -> list[dict]:
    """Build the full ordered run list for the condition matrix."""
    runs = []
    for seed in seeds:
        for ratio in ratios:
            label = f"node{ratio}-seed{seed}"
            config = os.path.join(CONFIG_DIR, f"{label}.yaml")

            # --- persistence ---
            runs.append(
                {
                    "label": f"persistence/{label}",
                    "done_pattern": f"baseline-persistence-test-{label}-*_manifest.json",
                    "cmd": [
                        sys.executable, "-m", "st_dssm.cli.baseline",
                        "--model", "persistence",
                        "--config", config,
                        "--output-dir", OUTPUT_DIR,
                        "--split", "test",
                    ],
                }
            )

            # --- st_gcn ---
            ckpt_glob = os.path.join(
                CHECKPOINT_DIR, f"baseline-st_gcn-test-seed{seed}-*_best.pt"
            )
            runs.append(
                {
                    "label": f"st_gcn/{label}",
                    "done_pattern": f"baseline-st_gcn-test-seed{seed}-{label}-predict-*_manifest.json",
                    "ckpt_glob": ckpt_glob,
                    "cmd_template": [
                        sys.executable, "-m", "st_dssm.cli.baseline",
                        "--model", "st_gcn",
                        "--config", config,
                        "--output-dir", OUTPUT_DIR,
                        "--split", "test",
                        "--predict-from", "{ckpt}",
                    ],
                }
            )

            # --- st_dssm ---
            ckpt_glob = os.path.join(
                CHECKPOINT_DIR, f"st_dssm-test-seed{seed}-*_best.pt"
            )
            runs.append(
                {
                    "label": f"st_dssm/{label}",
                    "done_pattern": f"st_dssm-test-seed{seed}-{label}-predict-*_manifest.json",
                    "ckpt_glob": ckpt_glob,
                    "cmd_template": [
                        sys.executable, "-m", "st_dssm.cli.train",
                        "--config", config,
                        "--output-dir", OUTPUT_DIR,
                        "--split", "test",
                        "--predict-from", "{ckpt}",
                    ],
                }
            )

    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 condition matrix runner.")
    parser.add_argument(
        "--ratio",
        type=int,
        choices=[10, 20, 30],
        default=None,
        help="Run only conditions at this masking ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=[2026, 2027, 2028],
        default=None,
        help="Run only conditions paired with this seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    args = parser.parse_args(argv)

    ratios = [args.ratio] if args.ratio else RATIOS
    seeds = [args.seed] if args.seed else SEEDS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = build_run_list(ratios, seeds)
    total = len(runs)
    skipped = 0
    executed = 0
    failed = 0

    for run in runs:
        label = run["label"]
        if _manifest_exists(run["done_pattern"]):
            print(f"[skip]  {label} — manifest already present")
            skipped += 1
            continue

        # Resolve the checkpoint path for learned models.
        cmd = list(run.get("cmd") or [])
        if not cmd:
            try:
                ckpt = _find_checkpoint(run["ckpt_glob"])
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[error] {label}: {exc}", file=sys.stderr)
                failed += 1
                continue
            cmd = [c.replace("{ckpt}", ckpt) for c in run["cmd_template"]]

        if args.dry_run:
            print(f"[dry]   {label}")
            print("        " + " ".join(cmd))
            executed += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"Running: {label}")
        print("=" * 60)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\n[error] {label} failed with exit code {result.returncode}.",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"\n[done]  {label}")
            executed += 1

    print(
        f"\nPhase 12 run summary: {total} conditions, "
        f"{skipped} skipped, {executed} {'would run' if args.dry_run else 'executed'}, "
        f"{failed} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
