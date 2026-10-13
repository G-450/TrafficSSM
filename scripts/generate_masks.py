"""Publish the canonical Phase 11 experimental masks.

Writes one artifact per (missingness level, seed) pair under ``experiments/masks/``
recording the withheld sensor indices, their PEMS-BAY sensor IDs and a checksum,
so every condition run can be audited against a published mask instead of a mask
regenerated from a remembered seed.

The masks are a deterministic function of the sensor count and the seed
(ADR-0010), so regenerating them is a check rather than a new draw:

    python scripts/generate_masks.py            # write the artifacts
    python scripts/generate_masks.py --check    # verify them, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from st_dssm.mask_generator import MaskGenerator, generate_canonical_masks

RATIOS = (0.1, 0.2, 0.3)
SEEDS = (2026, 2027, 2028)


def _sensor_ids(artifact_dir: str) -> list[str]:
    """Read the canonical sensor IDs from the Phase 3 artifact metadata.

    Only the metadata is read: loading the arrays to list 325 sensor IDs would
    pull in about a gigabyte of windows.
    """
    metadata_path = Path(artifact_dir) / "processed_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(
            f"Data artifact metadata not found at {metadata_path}. "
            "Generate the artifact first with 'st-dssm-preprocess generate'."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sensor_ids = [str(sid) for sid in metadata.get("sensor_ids", [])]
    if not sensor_ids:
        raise SystemExit(
            f"No 'sensor_ids' in the data artifact metadata at {artifact_dir}. "
            "Generate the artifact first with 'st-dssm-preprocess generate'."
        )
    return sensor_ids


def check(out_dir: Path, sensor_ids: list[str]) -> int:
    """Re-derive every published mask and report any that do not match."""
    failures: list[str] = []
    for ratio in RATIOS:
        for seed in SEEDS:
            expected = MaskGenerator(
                num_nodes=len(sensor_ids), missing_ratio=ratio, seed=seed
            )
            path = out_dir / f"{expected.mask_id}.json"
            if not path.exists():
                failures.append(f"{path}: missing")
                continue
            try:
                # load() already verifies the checksum and the stored indices
                # against the mask regenerated from the seed.
                MaskGenerator.load(path)
            except ValueError as exc:
                failures.append(f"{path}: {exc}")
                continue

            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("masked_sensor_ids") != expected.get_masked_sensor_ids(sensor_ids):
                failures.append(f"{path}: masked_sensor_ids do not match the data artifact")

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} mask artifact(s) failed verification.", file=sys.stderr)
        return 1

    print(f"All {len(RATIOS) * len(SEEDS)} canonical mask artifacts verified in {out_dir}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the canonical Phase 11 masks.")
    parser.add_argument(
        "--artifact-dir",
        default="data/processed",
        help="Phase 3 processed artifact directory holding the canonical sensor IDs.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/masks",
        help="Directory the mask artifacts are written to.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the published masks instead of writing them.",
    )
    args = parser.parse_args(argv)

    sensor_ids = _sensor_ids(args.artifact_dir)
    out_dir = Path(args.out_dir)

    if args.check:
        return check(out_dir, sensor_ids)

    generate_canonical_masks(sensor_ids, out_dir=str(out_dir))
    for ratio in RATIOS:
        for seed in SEEDS:
            generator = MaskGenerator(
                num_nodes=len(sensor_ids), missing_ratio=ratio, seed=seed
            )
            withheld = int((~generator.node_mask).sum())
            print(
                f"{generator.mask_id}: {withheld} of {generator.num_nodes} sensors "
                f"withheld (checksum {generator.checksum[:12]})"
            )
    print(f"Wrote {len(RATIOS) * len(SEEDS)} mask artifacts to {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
