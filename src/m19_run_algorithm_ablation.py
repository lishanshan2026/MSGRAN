"""Run graph-construction and gate ablation experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m7_train_multiscale_residual_graph.py"
EVALUATE = ROOT / "src" / "m7_evaluate_multiscale_residual_graph.py"
NEW_VARIANTS = (
    ("correlation_learned", "correlation", "learned"),
    ("distance_learned", "distance", "learned"),
    ("hybrid_fixed_one", "hybrid", "fixed_one"),
    ("hybrid_fixed_initial", "hybrid", "fixed_initial"),
)
EXISTING_VARIANTS = (
    ("proposed_hybrid_learned", "M7_gpu_full_1day_v1"),
    ("no_graph", "M7_gpu_ablation_no_graph_v1"),
)


def invoke(command: list[str]) -> None:
    print("[M19]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_metric(run_name: str) -> dict:
    path = ROOT / "experiments" / run_name / "real_test_forecast_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for label, graph_mode, gate_mode in NEW_VARIANTS:
        run_name = f"M19_{label}_1day_90stations_seed1"
        run_dir = ROOT / "experiments" / run_name
        metric_path = run_dir / "real_test_forecast_metrics.json"
        if not (args.resume and metric_path.exists()):
            invoke([
                sys.executable,
                str(TRAIN),
                "--device", args.device,
                "--epochs", str(args.epochs),
                "--horizon", "1",
                "--seed", str(args.seed),
                "--graph-mode", graph_mode,
                "--gate-mode", gate_mode,
                "--run-name", run_name,
            ])
            invoke([
                sys.executable,
                str(EVALUATE),
                "--device", args.device,
                "--checkpoint", str(run_dir / "best.pt"),
                "--save-forecasts",
            ])
        payload = load_metric(run_name)
        records.append({
            "variant": label,
            "run_name": run_name,
            "graph_mode": graph_mode,
            "gate_mode": gate_mode,
            "rmse_mm": payload["model"]["overall"]["rmse_mm"],
            "mae_mm": payload["model"]["overall"]["mae_mm"],
            "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
        })

    for label, run_name in EXISTING_VARIANTS:
        metric_path = ROOT / "experiments" / run_name / "real_test_forecast_metrics.json"
        if metric_path.exists():
            payload = load_metric(run_name)
            records.append({
                "variant": label,
                "run_name": run_name,
                "rmse_mm": payload["model"]["overall"]["rmse_mm"],
                "mae_mm": payload["model"]["overall"]["mae_mm"],
                "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
            })

    no_graph = next((float(row["rmse_mm"]) for row in records if row["variant"] == "no_graph"), None)
    if no_graph is not None:
        for row in records:
            row["gain_vs_no_graph"] = 1.0 - float(row["rmse_mm"]) / no_graph

    destination = ROOT / "reports" / "M19_algorithm_ablation_ledger.json"
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M19] complete: {destination}")


if __name__ == "__main__":
    main()
