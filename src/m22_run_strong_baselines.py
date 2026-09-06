"""Run and aggregate three-seed DLinear/TCN/PatchTST/STGCN baselines."""

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
MODELS = ("dlinear", "nlinear", "ridge_ar", "lstm", "tcn", "patchtst", "stgcn")


def invoke(command: list[str]) -> None:
    print("[M22]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="20260803,20260804,20260805")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    models = [value.strip().lower() for value in args.models.split(",") if value.strip()]

    rows: list[dict[str, object]] = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"unsupported baseline={model}")
        for seed_index, seed in enumerate(seeds, start=1):
            suffix = f"_smoke{args.max_windows}" if args.max_windows else ""
            run_name = f"M22_{model}_1day_90stations_seed{seed_index}{suffix}"
            run_dir = ROOT / "experiments" / run_name
            metric_path = run_dir / "real_test_forecast_metrics.json"
            if not (args.resume and metric_path.exists()):
                command = [sys.executable, str(TRAIN), "--model", model, "--device", args.device, "--epochs", str(args.epochs), "--seed", str(seed), "--run-name", run_name]
                if args.max_windows:
                    command += ["--max-windows", str(args.max_windows), "--hidden", "16", "--d-model", "32", "--layers", "1", "--batch-size", "8"]
                invoke(command)
                evaluate_command = [sys.executable, str(EVALUATE), "--device", args.device, "--checkpoint", str(run_dir / "best.pt"), "--save-forecasts"]
                if args.max_windows:
                    evaluate_command += ["--max-windows", str(args.max_windows)]
                invoke(evaluate_command)
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            rows.append({
                "model": model, "seed": seed, "run_name": run_name,
                "rmse_mm": payload["model"]["overall"]["rmse_mm"],
                "mae_mm": payload["model"]["overall"]["mae_mm"],
                "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
                **payload["efficiency"],
            })

    summary: list[dict[str, object]] = []
    for model in models:
        group = [row for row in rows if row["model"] == model]
        record: dict[str, object] = {"model": model, "seeds": len(group)}
        for field in ("rmse_mm", "mae_mm", "skill_vs_persistence", "milliseconds_per_window"):
            values = np.array([float(row[field]) for row in group])
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        record["parameter_count"] = int(group[0]["parameter_count"])
        summary.append(record)
    output = {"protocol": "frozen 2022-2024 test; identical split, normalization, mask, Huber loss, and validation selection", "runs": rows, "summary": summary}
    suffix = f"_smoke{args.max_windows}" if args.max_windows else ""
    destination = ROOT / "reports" / f"M22_strong_baselines_3seed{suffix}.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M22] complete: {destination}")


if __name__ == "__main__":
    main()
