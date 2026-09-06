"""Run the gated graph residual adapter over the selected NLinear backbone."""

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
    print("[M29]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS, start=1):
        base = ROOT / "experiments" / f"M28_nlinear_1day_90stations_seed{index}" / "best.pt"
        run_name = f"M29_nlinear_graph_adapter_1day_90stations_seed{index}"
        run_dir = ROOT / "experiments" / run_name
        metric = run_dir / "real_test_forecast_metrics.json"
        if not (args.resume and metric.exists()):
            invoke([sys.executable, str(TRAIN), "--base-checkpoint", str(base), "--run-name", run_name, "--device", args.device, "--epochs", str(args.epochs), "--seed", str(seed)])
            invoke([sys.executable, str(EVALUATE), "--checkpoint", str(run_dir / "best.pt"), "--device", args.device, "--save-forecasts"])
        payload = json.loads(metric.read_text(encoding="utf-8"))
        base_payload = json.loads((base.parent / "real_test_forecast_metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "seed": seed, "run_name": run_name, "rmse_mm": payload["model"]["overall"]["rmse_mm"], "mae_mm": payload["model"]["overall"]["mae_mm"],
            "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"], "gain_vs_nlinear": 1 - payload["model"]["overall"]["rmse_mm"] / base_payload["model"]["overall"]["rmse_mm"],
            "graph_gate": payload["graph_gate"], **payload["efficiency"],
        })
    summary: dict[str, object] = {"seeds": len(rows)}
    for field in ("rmse_mm", "mae_mm", "skill_vs_persistence", "gain_vs_nlinear", "graph_gate", "milliseconds_per_window"):
        values = np.asarray([float(row[field]) for row in rows])
        summary[f"{field}_mean"] = float(values.mean())
        summary[f"{field}_std"] = float(values.std(ddof=1))
    summary["parameter_count"] = int(rows[0]["parameter_count"])
    summary["trainable_parameter_count"] = int(rows[0]["trainable_parameter_count"])
    destination = ROOT / "reports" / "M29_nlinear_graph_adapter_3seed.json"
    destination.write_text(json.dumps({"runs": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
