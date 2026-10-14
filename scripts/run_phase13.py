"""Automation script for the Phase 13 calibration and robustness review.

Phase 13 needs two things the earlier phases never produced:

1. **Validation predictions** for each canonical seed. ADR-0008 allows
   temperature scaling only as a validation-fitted secondary analysis, so the
   temperature must be selected on the validation split and frozen before any
   test number is computed. Each run is a forward pass of a locked Phase 10
   checkpoint over the 5,190 validation windows (``--split val
   --predict-from``), so nothing is retrained and no model changes.
2. **Latency records**, measured rather than estimated, for the accelerator and
   for the CPU fallback ADR-0008 permits.

Both steps are idempotent: a validation run whose manifest already exists is
skipped, so re-running continues an interrupted sweep.

Usage:
    python scripts/run_phase13.py                 # validation runs + latency
    python scripts/run_phase13.py --dry-run
    python scripts/run_phase13.py --skip-latency  # validation runs only
    python scripts/run_phase13.py --seed 2026     # one seed

Then lock the deliverable:
    python scripts/aggregate_phase13.py
    python scripts/aggregate_phase13.py --check
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

SEEDS = [2026, 2027, 2028]

CONFIG = "configs/calibration/normal-val.yaml"
CHECKPOINT_DIR = "experiments/normal/checkpoints"
OUTPUT_DIR = "experiments/calibration"


def _find_checkpoint(seed: int) -> str:
    """Return the unique locked Phase 10 ST-DSSM checkpoint for a seed."""
    pattern = os.path.join(CHECKPOINT_DIR, f"st_dssm-test-seed{seed}-*_best.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found matching: {pattern}\n"
            "Phase 13 calibrates the locked Phase 10 model; run "
            "scripts/run_phase10.py first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoints matched {pattern}: {matches}\nDisambiguate manually."
        )
    return matches[0]


def _manifest_exists(seed: int) -> bool:
    return bool(
        glob.glob(os.path.join(OUTPUT_DIR, f"st_dssm-val-seed{seed}-*_manifest.json"))
    )


def build_run_list(seeds: list[int]) -> list[dict]:
    """One validation forward pass per seed, against that seed's checkpoint.

    A seed's temperature may only be fitted on that seed's own validation
    predictions, so the checkpoint and the run seed are always paired.
    """
    return [
        {
            "label": f"st_dssm/val/seed{seed}",
            "seed": seed,
            "ckpt_glob": os.path.join(
                CHECKPOINT_DIR, f"st_dssm-test-seed{seed}-*_best.pt"
            ),
            "cmd_template": [
                sys.executable, "-m", "st_dssm.cli.train",
                "--config", CONFIG,
                "--seed", str(seed),
                "--split", "val",
                "--output-dir", OUTPUT_DIR,
                "--predict-from", "{ckpt}",
            ],
        }
        for seed in seeds
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 13 calibration runner.")
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=None,
        help="Run only this seed's validation pass.",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip the latency measurement step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    args = parser.parse_args(argv)

    seeds = [args.seed] if args.seed else SEEDS
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = build_run_list(seeds)
    skipped = executed = failed = 0

    print(
        f"Phase 13 validation passes: {len(runs)} forward pass(es) over the "
        "validation split. No training happens; the locked Phase 10 checkpoints "
        "are the models under test."
    )

    for run in runs:
        label = run["label"]
        if _manifest_exists(run["seed"]):
            print(f"[skip]  {label} — manifest already present")
            skipped += 1
            continue

        try:
            ckpt = _find_checkpoint(run["seed"])
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

        print(f"\n{'=' * 60}\nRunning: {label}\n{'=' * 60}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"\n[error] {label} failed with exit code {result.returncode}.",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"\n[done]  {label}")
            executed += 1

    # --- Latency records ---
    if not args.skip_latency:
        latency_runs = [
            ("accelerator", [sys.executable, "scripts/measure_latency.py"]),
            (
                "cpu fallback",
                [
                    sys.executable, "scripts/measure_latency.py",
                    "--device", "cpu",
                    "--output", os.path.join(OUTPUT_DIR, "latency_record_cpu.json"),
                ],
            ),
        ]
        for label, cmd in latency_runs:
            if args.dry_run:
                print(f"[dry]   latency/{label}")
                print("        " + " ".join(cmd))
                continue
            print(f"\n{'=' * 60}\nMeasuring latency: {label}\n{'=' * 60}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(
                    f"\n[error] latency/{label} failed with exit code "
                    f"{result.returncode}.",
                    file=sys.stderr,
                )
                failed += 1

    print(
        f"\nPhase 13 run summary: {len(runs)} validation pass(es), "
        f"{skipped} skipped, {executed} {'would run' if args.dry_run else 'executed'}, "
        f"{failed} failed."
    )
    if not failed and not args.dry_run:
        print(
            "\nNext: python scripts/aggregate_phase13.py   "
            "(then --check to verify the lock)"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
