"""Run the DLinear graph adapter for three matched random seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m26_train_dlinear_graph_adapter.py"
EVALUATE = ROOT / "src" / "m26_evaluate_dlinear_graph_adapter.py"
SEEDS = (20260803, 20260804, 20260805)


def invoke(command: list[str]) -> None:
    print("[M26]", " ".join(command), flush=True); subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--resume", action="store_true"); parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args(); rows = []
    for seed_index, seed in enumerate(SEEDS, start=1):
        base = ROOT / "experiments" / f"M22_dlinear_1day_90stations_seed{seed_index}" / "best.pt"
        suffix = f"_smoke{args.max_windows}" if args.max_windows else ""
        run_name = f"M26_dlinear_graph_adapter_1day_90stations_seed{seed_index}{suffix}"
        run_dir = ROOT / "experiments" / run_name; metric = run_dir / "real_test_forecast_metrics.json"
        if not (args.resume and metric.exists()):
            command = [sys.executable, str(TRAIN), "--base-checkpoint", str(base), "--run-name", run_name, "--device", args.device, "--epochs", str(args.epochs), "--seed", str(seed)]
            if args.max_windows: command += ["--max-windows", str(args.max_windows), "--hidden", "16", "--batch-size", "8"]
            invoke(command)
            evaluate_command = [sys.executable, str(EVALUATE), "--checkpoint", str(run_dir / "best.pt"), "--device", args.device, "--save-forecasts"]
            if args.max_windows: evaluate_command += ["--max-windows", str(args.max_windows)]
            invoke(evaluate_command)
        payload = json.loads(metric.read_text(encoding="utf-8"))
        base_payload = json.loads((base.parent / "real_test_forecast_metrics.json").read_text(encoding="utf-8"))
        rows.append({"seed": seed, "run_name": run_name, "rmse_mm": payload["model"]["overall"]["rmse_mm"], "mae_mm": payload["model"]["overall"]["mae_mm"], "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"], "gain_vs_dlinear": 1 - payload["model"]["overall"]["rmse_mm"] / base_payload["model"]["overall"]["rmse_mm"], "graph_gate": payload["graph_gate"], **payload["efficiency"]})
    summary = {"seeds": len(rows)}
    for field in ("rmse_mm", "mae_mm", "skill_vs_persistence", "gain_vs_dlinear", "graph_gate", "milliseconds_per_window"):
        values = np.array([float(row[field]) for row in rows]); summary[f"{field}_mean"] = float(values.mean()); summary[f"{field}_std"] = float(values.std(ddof=1))
    summary["parameter_count"] = int(rows[0]["parameter_count"]); summary["trainable_parameter_count"] = int(rows[0]["trainable_parameter_count"])
    destination = ROOT / "reports" / f"M26_dlinear_graph_adapter_3seed{suffix}.json"; destination.write_text(json.dumps({"runs": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8"); print(destination)


if __name__ == "__main__":
    main()
