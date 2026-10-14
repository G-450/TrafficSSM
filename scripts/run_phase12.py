"""Automation script for Phase 12 Robustness condition matrix.

Executes the nine missingness conditions (10/20/30% x mask seeds 2026/2027/2028)
for all three models (ST-DSSM, deterministic ST-GCN, historical persistence),
reusing the locked Phase 10 checkpoints via ``--predict-from`` (ADR-0011).

All primary runs are forward passes over the test split; no training happens and
no GPU is required. Each run writes a manifest to ``experiments/missingness/``.
Re-running skips conditions that already have a manifest.

``--ablation`` runs the masked-training arm instead (ADR-0011): the model is
retrained with the same sensors withheld during training and validation, so each
run is a full training job. These runs carry a ``-masktrain`` suffix in their run
identity and are passed ``--resume auto``, so an interrupted run continues from
its last completed epoch rather than starting over.

Usage:
    python scripts/run_phase12.py
    python scripts/run_phase12.py --dry-run          # show which runs would execute
    python scripts/run_phase12.py --ratio 20         # only the 20% conditions
    python scripts/run_phase12.py --seed 2026        # only conditions paired with seed 2026

    # Masked-training ablation, scoped to the primary condition (3 training runs):
    python scripts/run_phase12.py --ablation --ratio 20 --models st_dssm
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
# The ablation is the only Phase 12 arm that fits a model, so it is the only one
# that writes checkpoints. They belong with the study, not in the CLI's default
# artifacts/checkpoints, which is ignored scratch: a 60-epoch run's resume state
# and best checkpoint must not sit somewhere a cleanup would sweep.
ABLATION_CHECKPOINT_DIR = "experiments/missingness/checkpoints"


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


MODEL_CHOICES = ("persistence", "st_gcn", "st_dssm")

# Measured on an RTX 4060 Laptop (8 GB, micro_batch_size 32) at 60 epochs.
ABLATION_HOURS_PER_RUN = {"st_dssm": 4.5, "st_gcn": 3.25}


def build_ablation_run_list(
    ratios: list[int],
    seeds: list[int],
    models: list[str],
) -> list[dict]:
    """Build the masked-training ablation run list (ADR-0011).

    Each entry retrains a model with the mask applied to training and validation
    as well as inference, so these are training jobs, not forward passes. The
    run identity carries ``-masktrain``, which keeps them from resuming, or being
    aggregated alongside, the primary condition runs at the same condition.

    Persistence is never included: it fits nothing, so there is no training
    representation for the mask to change.
    """
    runs = []
    for seed in seeds:
        for ratio in ratios:
            label = f"node{ratio}-seed{seed}"
            config = os.path.join(CONFIG_DIR, f"{label}.yaml")

            if "st_gcn" in models:
                runs.append(
                    {
                        "label": f"st_gcn/{label}/masktrain",
                        "done_pattern": (
                            f"baseline-st_gcn-test-seed{seed}-{label}-masktrain-*_manifest.json"
                        ),
                        "cmd": [
                            sys.executable, "-m", "st_dssm.cli.baseline",
                            "--model", "st_gcn",
                            "--config", config,
                            "--output-dir", OUTPUT_DIR,
                            "--checkpoint-dir", ABLATION_CHECKPOINT_DIR,
                            "--split", "test",
                            "--mask-training",
                            "--resume", "auto",
                        ],
                    }
                )

            if "st_dssm" in models:
                runs.append(
                    {
                        "label": f"st_dssm/{label}/masktrain",
                        "done_pattern": (
                            f"st_dssm-test-seed{seed}-{label}-masktrain-*_manifest.json"
                        ),
                        "cmd": [
                            sys.executable, "-m", "st_dssm.cli.train",
                            "--config", config,
                            "--output-dir", OUTPUT_DIR,
                            "--checkpoint-dir", ABLATION_CHECKPOINT_DIR,
                            "--split", "test",
                            "--mask-training",
                            "--resume", "auto",
                        ],
                    }
                )

    return runs


def build_run_list(
    ratios: list[int],
    seeds: list[int],
    models: list[str] | None = None,
) -> list[dict]:
    """Build the full ordered run list for the condition matrix."""
    models = list(models) if models else list(MODEL_CHOICES)
    runs = []
    for seed in seeds:
        for ratio in ratios:
            label = f"node{ratio}-seed{seed}"
            config = os.path.join(CONFIG_DIR, f"{label}.yaml")

            # --- persistence ---
            if "persistence" in models:
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
            if "st_gcn" in models:
                runs.append(
                    {
                        "label": f"st_gcn/{label}",
                        "done_pattern": f"baseline-st_gcn-test-seed{seed}-{label}-predict-*_manifest.json",
                        "ckpt_glob": os.path.join(
                            CHECKPOINT_DIR, f"baseline-st_gcn-test-seed{seed}-*_best.pt"
                        ),
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
            if "st_dssm" in models:
                runs.append(
                    {
                        "label": f"st_dssm/{label}",
                        "done_pattern": f"st_dssm-test-seed{seed}-{label}-predict-*_manifest.json",
                        "ckpt_glob": os.path.join(
                            CHECKPOINT_DIR, f"st_dssm-test-seed{seed}-*_best.pt"
                        ),
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
        "--ablation",
        action="store_true",
        help="Run the masked-training ablation arm (ADR-0011) instead of the "
        "primary condition matrix. These are full training runs, not forward "
        "passes; each is launched with --resume auto so it can be interrupted.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of models to run "
        f"({', '.join(MODEL_CHOICES)}). Defaults to all three for the primary "
        "matrix and to st_dssm for --ablation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    args = parser.parse_args(argv)

    ratios = [args.ratio] if args.ratio else RATIOS
    seeds = [args.seed] if args.seed else SEEDS

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in MODEL_CHOICES]
        if unknown:
            print(
                f"[error] unknown model(s) {unknown}; choose from {list(MODEL_CHOICES)}",
                file=sys.stderr,
            )
            return 2
    elif args.ablation:
        # Persistence fits nothing, and the ST-GCN arm doubles the budget; the
        # ablation question is about ST-DSSM's learned representation.
        models = ["st_dssm"]
    else:
        models = list(MODEL_CHOICES)

    if args.ablation and "persistence" in models:
        print(
            "[skip]  persistence has no training stage; dropping it from the "
            "ablation arm.",
        )
        models = [m for m in models if m != "persistence"]
        if not models:
            print("[error] nothing left to run.", file=sys.stderr)
            return 2

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.ablation:
        os.makedirs(ABLATION_CHECKPOINT_DIR, exist_ok=True)

    if args.ablation:
        runs = build_ablation_run_list(ratios, seeds, models)
        budget = sum(ABLATION_HOURS_PER_RUN.get(m, 0.0) for m in models) * len(
            ratios
        ) * len(seeds)
        print(
            f"Masked-training ablation: {len(runs)} training runs "
            f"({', '.join(models)} x {len(ratios)} ratio(s) x {len(seeds)} seed(s)).\n"
            f"Estimated budget: ~{budget:.1f} GPU-hours at 60 epochs. "
            "Runs are resumable: re-run this command to continue.\n"
        )
    else:
        runs = build_run_list(ratios, seeds, models)
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
        # The exit code is handled below, so a failure is reported and the rest
        # of the matrix still runs.
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

    print(
        f"\nPhase 12 run summary: {total} conditions, "
        f"{skipped} skipped, {executed} {'would run' if args.dry_run else 'executed'}, "
        f"{failed} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
