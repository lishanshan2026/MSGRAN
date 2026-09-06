"""Run NLinear and closed-form RidgeAR under the frozen three-seed protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m22_train_strong_baseline.py"
EVALUATE = ROOT / "src" / "m22_evaluate_strong_baseline.py"
MODELS = ("nlinear", "ridge_ar")


def invoke(command: list[str]) -> None:
    print("[M28]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="20260803,20260804,20260805")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for index, seed in enumerate(seeds, start=1):
            run_name = f"M28_{model}_1day_90stations_seed{index}"
            run_dir = ROOT / "experiments" / run_name
            metric = run_dir / "real_test_forecast_metrics.json"
            if not (args.resume and metric.exists()):
                invoke([
                    sys.executable, str(TRAIN), "--model", model, "--epochs", str(args.epochs),
                    "--seed", str(seed), "--device", args.device, "--run-name", run_name,
                ])
                invoke([sys.executable, str(EVALUATE), "--checkpoint", str(run_dir / "best.pt"), "--device", args.device, "--save-forecasts"])
            payload = json.loads(metric.read_text(encoding="utf-8"))
            rows.append({
                "model": model, "seed": seed, "run_name": run_name,
                "rmse_mm": payload["model"]["overall"]["rmse_mm"],
                "mae_mm": payload["model"]["overall"]["mae_mm"],
                "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
                **payload["efficiency"],
            })
    summary: list[dict[str, object]] = []
    for model in MODELS:
        group = [item for item in rows if item["model"] == model]
        record: dict[str, object] = {"model": model, "seeds": len(group)}
        for field in ("rmse_mm", "mae_mm", "skill_vs_persistence", "milliseconds_per_window"):
            values = np.asarray([float(item[field]) for item in group])
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        record["parameter_count"] = int(group[0]["parameter_count"])
        summary.append(record)
    destination = ROOT / "reports" / "M28_additional_baselines_3seed.json"
    destination.write_text(json.dumps({"protocol": "frozen 2022-2024 test; identical split, normalization, mask, and validation selection", "runs": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
