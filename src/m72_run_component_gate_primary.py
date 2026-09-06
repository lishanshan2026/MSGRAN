"""Run the minimal component-gate MS-GRAN upgrade on the primary benchmark.

The experiment changes only the scalar global graph gate into three E/N/U gates.
The NLinear checkpoint, graph construction, multi-scale residual encoder, loss,
chronological split, and optimizer protocol are kept matched to M68.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def run_command(command: list[str]) -> None:
    print("[M72]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--seed-indices", default="1,2,3")
    args = parser.parse_args()

    selected = [int(item) for item in args.seed_indices.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for seed_index in selected:
        run_name = f"M72_msgran_component_gate_1day_90stations_seed{seed_index}"
        base_run = BASE_RUNS[seed_index]
        command = [
            sys.executable,
            str(TRAIN),
            "--base-checkpoint",
            str(ROOT / "experiments" / base_run / "best.pt"),
            "--run-name",
            run_name,
            "--gate-mode",
            "component",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(SEEDS[seed_index]),
            "--device",
            args.device,
        ]
        if args.max_windows:
            command += ["--max-windows", str(args.max_windows)]
        run_command(command)
        run_command([
            sys.executable,
            str(EVALUATE),
            "--checkpoint",
            str(ROOT / "experiments" / run_name / "best.pt"),
            "--device",
            args.device,
            "--save-forecasts",
        ])
        metrics = json.loads((ROOT / "experiments" / run_name / "real_test_forecast_metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "seed_index": seed_index,
            "seed": SEEDS[seed_index],
            "base_run": base_run,
            "run_name": run_name,
            "nlinear_rmse_mm": metrics["base_nlinear"]["overall"]["rmse_mm"],
            "component_gate_rmse_mm": metrics["model"]["overall"]["rmse_mm"],
            "gain_vs_nlinear_percent": 100.0 * metrics["gain_vs_nlinear"],
            "mae_mm": metrics["model"]["overall"]["mae_mm"],
            "p95_abs_error_mm": metrics["model"]["overall"]["p95_abs_error_mm"],
            "graph_gate_enu": metrics.get("graph_gate"),
            "parameter_count": metrics["efficiency"]["parameter_count"],
            "trainable_parameter_count": metrics["efficiency"]["trainable_parameter_count"],
            "milliseconds_per_window": metrics["efficiency"]["milliseconds_per_window"],
        })

    import numpy as np

    gains = np.array([row["gain_vs_nlinear_percent"] for row in rows], dtype=float)
    rmses = np.array([row["component_gate_rmse_mm"] for row in rows], dtype=float)
    payload = {
        "experiment": "M72_component_gate_primary",
        "change": "global graph gate -> E/N/U component-wise graph gate",
        "rows": rows,
        "summary": {
            "seeds": len(rows),
            "component_gate_rmse_mm_mean": float(rmses.mean()),
            "component_gate_rmse_mm_sd": float(rmses.std(ddof=1)) if len(rows) > 1 else 0.0,
            "gain_vs_nlinear_percent_mean": float(gains.mean()),
            "gain_vs_nlinear_percent_sd": float(gains.std(ddof=1)) if len(rows) > 1 else 0.0,
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "M72_component_gate_primary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
