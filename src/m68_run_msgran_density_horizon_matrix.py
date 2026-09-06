"""Run the final MS-GRAN horizon-density matrix.

This runner retrains only the MS-GRAN residual adapter for every
forecast-horizon/network-density/seed cell, using the already trained matched
NLinear checkpoints as frozen local backbones. It produces the forecast files
required by the updated Figure 6.
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
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
SEEDS = (20260803, 20260804, 20260805)
HORIZONS = (1, 3, 7)
STATION_COUNTS = (90, 70, 50, 30)


def base_run_name(horizon: int, stations: int, seed_index: int) -> str:
    if stations == 90 and horizon == 1:
        return f"M28_nlinear_1day_90stations_seed{seed_index}"
    if stations == 90:
        return f"M31_nlinear_{horizon}day_seed{seed_index}"
    return f"M52_nlinear_{horizon}day_{stations}stations_seed{seed_index}"


def run_name(horizon: int, stations: int, seed_index: int) -> str:
    return f"M68_msgran_{horizon}day_{stations}stations_seed{seed_index}"


def invoke(command: list[str], log) -> None:
    message = "[M68] " + " ".join(command)
    print(message, flush=True)
    print(message, file=log, flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def evaluate(run_dir: Path, device: str, log, resume: bool, max_windows: int) -> dict:
    metric = run_dir / "real_test_forecast_metrics.json"
    forecast = run_dir / "real_test_forecasts.npz"
    if not (resume and metric.exists() and forecast.exists()):
        cmd = [
            sys.executable,
            str(EVALUATE),
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--device",
            device,
            "--save-forecasts",
        ]
        if max_windows:
            cmd += ["--max-windows", str(max_windows)]
        invoke(cmd, log)
    return json.loads(metric.read_text(encoding="utf-8"))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for stations in STATION_COUNTS:
            selected = [row for row in rows if row["horizon_days"] == horizon and row["network_stations"] == stations]
            if not selected:
                continue
            item: dict[str, object] = {
                "horizon_days": horizon,
                "network_stations": stations,
                "seeds": len(selected),
            }
            for key in ("nlinear_rmse_mm", "msgran_rmse_mm", "gain_vs_nlinear_percent", "skill_vs_persistence_percent"):
                values = np.asarray([float(row[key]) for row in selected], dtype=float)
                item[key + "_mean"] = float(values.mean())
                item[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--sizes", default="90,70,50,30")
    parser.add_argument("--seed-indices", default="1,2,3")
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()

    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    seed_indices = [int(value) for value in args.seed_indices.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS / "M68_msgran_density_horizon_matrix.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for horizon in horizons:
            for stations in sizes:
                for seed_index in seed_indices:
                    seed = SEEDS[seed_index - 1]
                    base_name = base_run_name(horizon, stations, seed_index)
                    base_checkpoint = EXPERIMENTS / base_name / "best.pt"
                    if not base_checkpoint.exists():
                        raise FileNotFoundError(base_checkpoint)
                    name = run_name(horizon, stations, seed_index)
                    run_dir = EXPERIMENTS / name
                    if not (args.resume and (run_dir / "best.pt").exists()):
                        cmd = [
                            sys.executable,
                            str(TRAIN),
                            "--base-checkpoint",
                            str(base_checkpoint),
                            "--run-name",
                            name,
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                        if args.max_windows:
                            cmd += ["--max-windows", str(args.max_windows)]
                        invoke(cmd, log)
                    payload = evaluate(run_dir, args.device, log, args.resume, args.max_windows)
                    base_rmse = float(payload["base_nlinear"]["overall"]["rmse_mm"])
                    model_rmse = float(payload["model"]["overall"]["rmse_mm"])
                    rows.append(
                        {
                            "horizon_days": horizon,
                            "network_stations": stations,
                            "seed_index": seed_index,
                            "seed": seed,
                            "base_run": base_name,
                            "msgran_run": name,
                            "nlinear_rmse_mm": base_rmse,
                            "msgran_rmse_mm": model_rmse,
                            "gain_vs_nlinear_percent": 100.0 * float(payload["gain_vs_nlinear"]),
                            "skill_vs_persistence_percent": 100.0 * float(payload["overall_rmse_skill_vs_persistence"]),
                            "graph_gate": payload.get("graph_gate"),
                            "parameter_count": payload["efficiency"]["parameter_count"],
                            "trainable_parameter_count": payload["efficiency"]["trainable_parameter_count"],
                            "milliseconds_per_window": payload["efficiency"]["milliseconds_per_window"],
                        }
                    )
                    partial = REPORTS / "M68_msgran_density_horizon_matrix_partial.json"
                    partial.write_text(json.dumps({"runs": rows, "summary": summarize(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")

    destination = REPORTS / "M68_msgran_density_horizon_matrix.json"
    destination.write_text(json.dumps({"runs": rows, "summary": summarize(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
