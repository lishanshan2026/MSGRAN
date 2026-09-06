"""Run the missing NLinear + graph-adapter density/horizon robustness matrix.

The 90-station cells are reused from M28/M29/M31. This runner creates the
remaining 27 paired configurations: 3 horizons x 3 nested subnetworks x 3 seeds.
It is idempotent with --resume and saves test forecasts for station-wise maps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "src" / "m22_train_strong_baseline.py"
BASE_EVAL = ROOT / "src" / "m22_evaluate_strong_baseline.py"
GRAPH_TRAIN = ROOT / "src" / "m26_train_dlinear_graph_adapter.py"
GRAPH_EVAL = ROOT / "src" / "m26_evaluate_dlinear_graph_adapter.py"
SUBNETWORKS = ROOT / "reports" / "M9_nested_subnetworks_v1.json"
SEEDS = (20260803, 20260804, 20260805)


def invoke(command: list[str], log) -> None:
    message = "[M52] " + " ".join(command)
    print(message, flush=True)
    print(message, file=log, flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def evaluate(run_dir: Path, script: Path, device: str, log, resume: bool) -> dict:
    metrics = run_dir / "real_test_forecast_metrics.json"
    forecasts = run_dir / "real_test_forecasts.npz"
    if not (resume and metrics.exists() and forecasts.exists()):
        invoke([
            sys.executable, str(script), "--checkpoint", str(run_dir / "best.pt"),
            "--device", device, "--save-forecasts",
        ], log)
    return json.loads(metrics.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sizes", default="70,50,30")
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--seed-indices", default="1,2,3")
    args = parser.parse_args()

    subnetworks = json.loads(SUBNETWORKS.read_text(encoding="utf-8"))
    sizes = [int(v) for v in args.sizes.split(",") if v]
    horizons = [int(v) for v in args.horizons.split(",") if v]
    seed_indices = [int(v) for v in args.seed_indices.split(",") if v]
    rows: list[dict[str, object]] = []
    log_path = ROOT / "reports" / "M52_density_matrix_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for horizon in horizons:
            for size in sizes:
                indices = subnetworks[str(size)]["indices"]
                station_arg = ",".join(str(value) for value in indices)
                for seed_index in seed_indices:
                    seed = SEEDS[seed_index - 1]
                    base_name = f"M52_nlinear_{horizon}day_{size}stations_seed{seed_index}"
                    base_dir = ROOT / "experiments" / base_name
                    if not (args.resume and (base_dir / "best.pt").exists()):
                        invoke([
                            sys.executable, str(BASE_TRAIN), "--model", "nlinear",
                            "--horizon", str(horizon), "--epochs", str(args.epochs),
                            "--seed", str(seed), "--device", args.device,
                            "--station-indices", station_arg, "--run-name", base_name,
                        ], log)
                    base_metrics = evaluate(base_dir, BASE_EVAL, args.device, log, args.resume)

                    graph_name = f"M52_nlinear_graph_{horizon}day_{size}stations_seed{seed_index}"
                    graph_dir = ROOT / "experiments" / graph_name
                    if not (args.resume and (graph_dir / "best.pt").exists()):
                        invoke([
                            sys.executable, str(GRAPH_TRAIN), "--base-checkpoint", str(base_dir / "best.pt"),
                            "--run-name", graph_name, "--epochs", str(args.epochs),
                            "--seed", str(seed), "--device", args.device,
                        ], log)
                    graph_metrics = evaluate(graph_dir, GRAPH_EVAL, args.device, log, args.resume)
                    base_rmse = float(base_metrics["model"]["overall"]["rmse_mm"])
                    graph_rmse = float(graph_metrics["model"]["overall"]["rmse_mm"])
                    rows.append({
                        "horizon_days": horizon,
                        "stations": size,
                        "seed_index": seed_index,
                        "seed": seed,
                        "base_run": base_name,
                        "graph_run": graph_name,
                        "nlinear_rmse_mm": base_rmse,
                        "graph_rmse_mm": graph_rmse,
                        "graph_gain_percent": 100.0 * (1.0 - graph_rmse / base_rmse),
                    })
                    partial = ROOT / "reports" / "M52_density_matrix_partial.json"
                    partial.write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")

    summary: list[dict[str, object]] = []
    for horizon in horizons:
        for size in sizes:
            selected = [r for r in rows if r["horizon_days"] == horizon and r["stations"] == size]
            item: dict[str, object] = {"horizon_days": horizon, "stations": size, "seeds": len(selected)}
            for field in ("nlinear_rmse_mm", "graph_rmse_mm", "graph_gain_percent"):
                values = np.asarray([float(row[field]) for row in selected])
                item[field + "_mean"] = float(values.mean())
                item[field + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary.append(item)
    destination = ROOT / "reports" / "M52_nlinear_density_horizon_3seed.json"
    destination.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
