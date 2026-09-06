"""Rerun the primary MS-GRAN setting after masking empty signed channels.

The final Cascadia Top-K graph contains no selected negative edge.  This audit
therefore tests the corrected implementation in which an unsupported signed
channel contributes an exact zero contrast instead of the spurious ``-u_i``
term present in the legacy M74 implementation.  The frozen NLinear backbone,
split, seeds, optimizer, and evaluation protocol are unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m58_multiscale_highpass_screen.py"
EVALUATE = ROOT / "src" / "m58_evaluate_multiscale_highpass.py"
REPORTS = ROOT / "reports"

BASE_RUNS = {
    1: "M28_nlinear_1day_90stations_seed1",
    2: "M28_nlinear_1day_90stations_seed2",
    3: "M28_nlinear_1day_90stations_seed3",
}
SEEDS = {1: 20260803, 2: 20260804, 3: 20260805}


def run(command: list[str]) -> None:
    print("[M80]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed-indices", default="1,2,3")
    args = parser.parse_args()

    selected = [int(x) for x in args.seed_indices.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for seed_index in selected:
        run_name = f"M80_msgran_empty_channel_fixed_1day_90stations_seed{seed_index}"
        base_run = BASE_RUNS[seed_index]
        run([
            sys.executable,
            str(TRAIN),
            "--base-checkpoint", str(ROOT / "experiments" / base_run / "best.pt"),
            "--run-name", run_name,
            "--aggregation-mode", "signed_prior",
            "--gate-mode", "global",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(SEEDS[seed_index]),
            "--device", args.device,
        ])
        run([
            sys.executable,
            str(EVALUATE),
            "--checkpoint", str(ROOT / "experiments" / run_name / "best.pt"),
            "--device", args.device,
            "--save-forecasts",
        ])
        metrics = json.loads(
            (ROOT / "experiments" / run_name / "real_test_forecast_metrics.json").read_text(encoding="utf-8")
        )
        rows.append({
            "seed_index": seed_index,
            "seed": SEEDS[seed_index],
            "base_run": base_run,
            "run_name": run_name,
            "nlinear_rmse_mm": metrics["base_nlinear"]["overall"]["rmse_mm"],
            "corrected_model_rmse_mm": metrics["model"]["overall"]["rmse_mm"],
            "gain_vs_nlinear_percent": 100.0 * metrics["gain_vs_nlinear"],
            "mae_mm": metrics["model"]["overall"]["mae_mm"],
            "p95_abs_error_mm": metrics["model"]["overall"]["p95_abs_error_mm"],
            "graph_gate": metrics.get("graph_gate"),
            "parameter_count": metrics["efficiency"]["parameter_count"],
            "trainable_parameter_count": metrics["efficiency"]["trainable_parameter_count"],
            "milliseconds_per_window": metrics["efficiency"]["milliseconds_per_window"],
        })

    rmses = np.asarray([row["corrected_model_rmse_mm"] for row in rows], dtype=float)
    gains = np.asarray([row["gain_vs_nlinear_percent"] for row in rows], dtype=float)
    payload = {
        "experiment": "M80_empty_signed_channel_audit",
        "implementation_fix": "mask unsupported signed contrasts so empty channels contribute exactly zero",
        "interpretation": "The Cascadia negative channel has no selected edge; this run cannot establish a negative-edge benefit.",
        "rows": rows,
        "summary": {
            "seeds": len(rows),
            "corrected_model_rmse_mm_mean": float(rmses.mean()),
            "corrected_model_rmse_mm_sd": float(rmses.std(ddof=1)) if len(rows) > 1 else 0.0,
            "gain_vs_nlinear_percent_mean": float(gains.mean()),
            "gain_vs_nlinear_percent_sd": float(gains.std(ddof=1)) if len(rows) > 1 else 0.0,
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "M80_empty_signed_channel_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
