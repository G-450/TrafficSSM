"""Phase 13 inference-latency records for the three evaluated models.

Phase 13 requires latency records that the project's claims can be reconciled
against. Latency here is strictly *inference* cost: the forward pass that
produces a 12-horizon forecast for all 325 sensors from one window of history,
which is the operation a deployment would run. Training cost is recorded
separately in the Phase 10/12 run manifests.

Protocol
--------
- Inputs are real PEMS-BAY test windows, in the batch layout the evaluation
  runners use, so the numbers describe the measured pipeline and not a synthetic
  best case.
- Each model is warmed up before timing, which matters on CUDA: the first
  forward pass pays for kernel autotuning and lazy module initialisation.
- CUDA is synchronized around every timed region, otherwise the host returns
  before the device has finished and the measurement is meaningless.
- Timing uses ``time.perf_counter`` over whole batches, repeated, and the
  report carries the median and the p95 rather than a mean: latency
  distributions are right-skewed and a mean hides the tail a deployment feels.
- ST-DSSM is timed through ``forward_predict`` (prior-only sampling, no target
  access), which is exactly what the condition runs call.

Usage:
    python scripts/measure_latency.py
    python scripts/measure_latency.py --device cpu       # CPU fallback record
    python scripts/measure_latency.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from st_dssm.baselines.persistence import HistoricalPersistence
from st_dssm.baselines.st_gcn import DeterministicSTGCN
from st_dssm.dssm import GaussianDSSM
from st_dssm.graph import (
    calculate_normalized_laplacian,
    calculate_scaled_laplacian,
    compute_chebyshev_polynomials,
    symmetrize_adjacency,
)
from st_dssm.io import load_and_validate_artifact

CONFIG_PATH = "configs/tuned.yaml"
ARTIFACT_DIR = "data/processed"
ADJ_PATH = "data/raw/adj_mx_bay.pkl"
CHECKPOINT_DIR = "experiments/normal/checkpoints"
OUTPUT_PATH = "experiments/calibration/latency_record.json"

WARMUP_BATCHES = 10
TIMED_BATCHES = 30


def _load_cheb(cheb_k: int, device: torch.device) -> torch.Tensor:
    """Load the canonical graph and return its Chebyshev polynomial basis."""
    with open(ADJ_PATH, "rb") as f:
        _ids, _id_map, adj_mx = pickle.load(f, encoding="latin1")
    if not np.allclose(adj_mx, adj_mx.T, atol=1e-5):
        adj_mx = symmetrize_adjacency(adj_mx, method="average")
    scaled_lap, _ = calculate_scaled_laplacian(calculate_normalized_laplacian(adj_mx))
    cheb = compute_chebyshev_polynomials(scaled_lap, k=cheb_k)
    return torch.tensor(cheb, dtype=torch.float32, device=device)


def _find_checkpoint(prefix: str) -> str:
    """Return the newest checkpoint for a run prefix."""
    import glob

    matches = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, f"{prefix}*_best.pt")))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching {prefix}*_best.pt under {CHECKPOINT_DIR}. "
            "Run scripts/run_phase10.py first."
        )
    return matches[-1]


def _sync(device: torch.device) -> None:
    """Block until the device has actually finished its queued work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_batches(
    fn,
    batches: list[Any],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Time ``fn`` over batches, returning per-window latencies in milliseconds."""
    for i in range(warmup):
        fn(batches[i % len(batches)])
    _sync(device)

    per_window_ms: list[float] = []
    for i in range(repeats):
        batch = batches[i % len(batches)]
        n_windows = len(batch) if not isinstance(batch, torch.Tensor) else batch.shape[0]
        _sync(device)
        start = time.perf_counter()
        fn(batch)
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_window_ms.append(elapsed_ms / n_windows)
    return per_window_ms


def _summarize(
    model_name: str,
    per_window_ms: list[float],
    *,
    batch_size: int,
    parameters: int,
    device: torch.device,
) -> dict[str, Any]:
    """Reduce a latency sample to the recorded statistics."""
    ordered = sorted(per_window_ms)
    median = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    batch_median_ms = median * batch_size
    return {
        "model_name": model_name,
        "trainable_parameters": parameters,
        "device": str(device),
        "batch_size": batch_size,
        "timed_batches": len(per_window_ms),
        "per_window_median_ms": median,
        "per_window_p95_ms": p95,
        "per_window_min_ms": ordered[0],
        "per_window_max_ms": ordered[-1],
        "per_batch_median_ms": batch_median_ms,
        "windows_per_second": 1000.0 / median if median > 0 else float("inf"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 13 inference-latency records.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to measure ('cuda', 'cpu'). Defaults to cuda when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Which locked Phase 10 checkpoint to load. [default: 2026]",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=TIMED_BATCHES,
        help=f"Timed batches per model. [default: {TIMED_BATCHES}]",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Where to write the latency record. [default: {OUTPUT_PATH}]",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        print("[error] --repeats must be at least 1.", file=sys.stderr)
        return 2

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[error] CUDA requested but torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    cheb_k = int(model_cfg.get("cheb_k", 3))

    print(f"Measuring inference latency on {device} (batch size {batch_size}).")

    arrays, _ = load_and_validate_artifact(ARTIFACT_DIR)
    test_x = arrays["test_X"]
    test_x_mask = arrays["test_X_mask"]

    # Build the batches once, in the evaluation layout: values concatenated with
    # the binary observation mask along the channel axis.
    n_batches = max(WARMUP_BATCHES, args.repeats)
    torch_batches: list[torch.Tensor] = []
    numpy_batches: list[np.ndarray] = []
    for i in range(n_batches):
        lo = (i * batch_size) % (len(test_x) - batch_size)
        bx = test_x[lo : lo + batch_size]
        bxm = test_x_mask[lo : lo + batch_size]
        numpy_batches.append(bx)
        torch_batches.append(
            torch.cat(
                [torch.from_numpy(bx), torch.from_numpy(bxm).float()], dim=-1
            ).to(device)
        )

    cheb_poly = _load_cheb(cheb_k, device)
    records: list[dict[str, Any]] = []

    # --- Historical persistence (no parameters, numpy path) ---
    persistence = HistoricalPersistence(forecast_horizon=int(model_cfg.get("horizon", 12)))
    samples = _time_batches(
        lambda b: persistence.predict(b),
        numpy_batches,
        torch.device("cpu"),
        warmup=WARMUP_BATCHES,
        repeats=args.repeats,
    )
    records.append(
        _summarize(
            "Historical Persistence",
            samples,
            batch_size=batch_size,
            parameters=0,
            # Persistence is an array slice with no learned parameters, so it is
            # recorded on CPU regardless of the requested device: moving it to
            # an accelerator would measure the transfer, not the method.
            device=torch.device("cpu"),
        )
    )

    # --- Deterministic ST-GCN ---
    st_gcn = DeterministicSTGCN(
        num_nodes=int(model_cfg.get("num_nodes", 325)),
        in_channels=int(model_cfg.get("in_channels", 2)),
        hidden_channels=64,
        input_length=int(model_cfg.get("input_length", 12)),
        forecast_horizon=int(model_cfg.get("horizon", 12)),
        cheb_k=cheb_k,
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    st_gcn_ckpt = _find_checkpoint(f"baseline-st_gcn-test-seed{args.seed}-")
    state = torch.load(st_gcn_ckpt, map_location=device, weights_only=False)
    st_gcn.load_state_dict(state.get("model_state_dict", state))
    st_gcn.eval()

    with torch.no_grad():
        samples = _time_batches(
            lambda b: st_gcn(b, cheb_poly),
            torch_batches,
            device,
            warmup=WARMUP_BATCHES,
            repeats=args.repeats,
        )
    records.append(
        _summarize(
            "Deterministic ST-GCN",
            samples,
            batch_size=batch_size,
            parameters=sum(p.numel() for p in st_gcn.parameters() if p.requires_grad),
            device=device,
        )
    )

    # --- ST-DSSM (prior-only predictive path) ---
    dssm = GaussianDSSM(
        num_nodes=int(model_cfg.get("num_nodes", 325)),
        in_channels=int(model_cfg.get("in_channels", 2)),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        context_dim=int(model_cfg.get("context_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        horizon=int(model_cfg.get("horizon", 12)),
        input_length=int(model_cfg.get("input_length", 12)),
        cheb_k=cheb_k,
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    dssm_ckpt = _find_checkpoint(f"st_dssm-test-seed{args.seed}-")
    state = torch.load(dssm_ckpt, map_location=device, weights_only=False)
    dssm.load_state_dict(state.get("model_state_dict", state))
    dssm.eval()

    with torch.no_grad():
        samples = _time_batches(
            lambda b: dssm.forward_predict(b, cheb_poly),
            torch_batches,
            device,
            warmup=WARMUP_BATCHES,
            repeats=args.repeats,
        )
    records.append(
        _summarize(
            "ST-DSSM (ours)",
            samples,
            batch_size=batch_size,
            parameters=dssm.get_capacity_report()["total_trainable_parameters"],
            device=device,
        )
    )

    record = {
        "study": "Phase 13 inference latency",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "operation": "one forward pass: 12 input steps -> 12 horizons x 325 sensors",
            "split": "test",
            "batch_size": batch_size,
            "warmup_batches": WARMUP_BATCHES,
            "timed_batches": args.repeats,
            "cuda_synchronized": device.type == "cuda",
            "checkpoint_seed": args.seed,
            "st_dssm_path": "forward_predict (prior-only, no target access)",
        },
        "environment": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu"
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "checkpoints": {
            "Deterministic ST-GCN": st_gcn_ckpt.replace("\\", "/"),
            "ST-DSSM (ours)": dssm_ckpt.replace("\\", "/"),
        },
        "models": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\n{'Model':<24} {'median ms/window':>18} {'p95':>8} {'windows/s':>12}")
    for rec in records:
        print(
            f"{rec['model_name']:<24} {rec['per_window_median_ms']:>18.4f} "
            f"{rec['per_window_p95_ms']:>8.4f} {rec['windows_per_second']:>12.1f}"
        )
    print(f"\nLatency record: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
