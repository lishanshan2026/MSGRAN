"""Run the empty-channel-corrected MS-GRAN horizon-density matrix.

M80 one-day/90-station runs are reused. Every remaining cell retrains only the
residual adapter from the matched frozen NLinear checkpoint. Unsupported signed
channels are masked by the corrected implementation in m58.
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


def base_run_name(horizon: int, stations: int, seed_index: int) -> str:
    if stations == 90 and horizon == 1:
        return f"M28_nlinear_1day_90stations_seed{seed_index}"
    if stations == 90:
        return f"M31_nlinear_{horizon}day_seed{seed_index}"
    return f"M52_nlinear_{horizon}day_{stations}stations_seed{seed_index}"


def run_name(horizon: int, stations: int, seed_index: int) -> str:
    if horizon == 1 and stations == 90:
        return f"M80_msgran_empty_channel_fixed_1day_90stations_seed{seed_index}"
    return f"M81_msgran_corrected_{horizon}day_{stations}stations_seed{seed_index}"


def invoke(command: list[str], log) -> None:
    message = "[M81] " + " ".join(command)
    print(message, flush=True)
    print(message, file=log, flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def summarize(rows: list[dict[str, object]], horizons: list[int], sizes: list[int]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for horizon in horizons:
        for stations in sizes:
            selected = [r for r in rows if r["horizon_days"] == horizon and r["network_stations"] == stations]
            if not selected:
                continue
            item: dict[str, object] = {"horizon_days": horizon, "network_stations": stations, "seeds": len(selected)}
            for key in ("nlinear_rmse_mm", "corrected_msgran_rmse_mm", "gain_vs_nlinear_percent", "skill_vs_persistence_percent"):
                values = np.asarray([float(r[key]) for r in selected], dtype=float)
                item[key + "_mean"] = float(values.mean())
                item[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizons", default="1,3,7")
    parser.add_argument("--sizes", default="90,70,50,30")
    parser.add_argument("--seed-indices", default="1,2,3")
    args = parser.parse_args()

    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    sizes = [int(v) for v in args.sizes.split(",") if v.strip()]
    seed_indices = [int(v) for v in args.seed_indices.split(",") if v.strip()]
    rows: list[dict[str, object]] = []
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS / "M81_corrected_density_horizon_matrix.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for horizon in horizons:
            for stations in sizes:
                for seed_index in seed_indices:
                    seed = SEEDS[seed_index - 1]
                    base_name = base_run_name(horizon, stations, seed_index)
                    checkpoint = EXPERIMENTS / base_name / "best.pt"
                    if not checkpoint.exists():
                        raise FileNotFoundError(checkpoint)
                    name = run_name(horizon, stations, seed_index)
                    run_dir = EXPERIMENTS / name
                    metric_path = run_dir / "real_test_forecast_metrics.json"
                    forecast_path = run_dir / "real_test_forecasts.npz"
                    if not (args.resume and (run_dir / "best.pt").exists()):
                        invoke([
                            sys.executable, str(TRAIN), "--base-checkpoint", str(checkpoint),
                            "--run-name", name, "--aggregation-mode", "signed_prior",
                            "--gate-mode", "global", "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size), "--seed", str(seed),
                            "--device", args.device,
                        ], log)
                    if not (args.resume and metric_path.exists() and forecast_path.exists()):
                        invoke([
                            sys.executable, str(EVALUATE), "--checkpoint", str(run_dir / "best.pt"),
                            "--device", args.device, "--save-forecasts",
                        ], log)
                    payload = json.loads(metric_path.read_text(encoding="utf-8"))
                    base_rmse = float(payload["base_nlinear"]["overall"]["rmse_mm"])
                    model_rmse = float(payload["model"]["overall"]["rmse_mm"])
                    rows.append({
                        "horizon_days": horizon, "network_stations": stations,
                        "seed_index": seed_index, "seed": seed, "base_run": base_name,
                        "corrected_msgran_run": name, "nlinear_rmse_mm": base_rmse,
                        "corrected_msgran_rmse_mm": model_rmse,
                        "gain_vs_nlinear_percent": 100.0 * float(payload["gain_vs_nlinear"]),
                        "skill_vs_persistence_percent": 100.0 * float(payload["overall_rmse_skill_vs_persistence"]),
                        "graph_gate": payload.get("graph_gate"),
                        "parameter_count": payload["efficiency"]["parameter_count"],
                        "trainable_parameter_count": payload["efficiency"]["trainable_parameter_count"],
                        "milliseconds_per_window": payload["efficiency"]["milliseconds_per_window"],
                    })
                    partial = {"runs": rows, "summary": summarize(rows, horizons, sizes)}
                    (REPORTS / "M81_corrected_density_horizon_matrix_partial.json").write_text(
                        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
    final = {"runs": rows, "summary": summarize(rows, horizons, sizes)}
    destination = REPORTS / "M81_corrected_density_horizon_matrix.json"
    destination.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
