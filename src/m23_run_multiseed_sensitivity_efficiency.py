"""Complete three-seed core/ablation evidence and one-seed sensitivity analyses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m7_train_multiscale_residual_graph.py"
EVALUATE = ROOT / "src" / "m7_evaluate_multiscale_residual_graph.py"
SEEDS = (20260803, 20260804, 20260805)

VARIANTS = {
    "proposed": [],
    "no_graph": ["--disable-graph-correction"],
    "correlation_learned": ["--graph-mode", "correlation"],
    "distance_learned": ["--graph-mode", "distance"],
    "hybrid_fixed_initial": ["--gate-mode", "fixed_initial"],
    "hybrid_fixed_one": ["--gate-mode", "fixed_one"],
}

EXISTING = {
    ("proposed", 1): "M7_gpu_full_1day_v1",
    ("proposed", 2): "M7_gpu_full_1day_seed2",
    ("no_graph", 1): "M7_gpu_ablation_no_graph_v1",
    ("no_graph", 2): "M7_gpu_ablation_no_graph_seed2",
    ("correlation_learned", 1): "M19_correlation_learned_1day_90stations_seed1",
    ("distance_learned", 1): "M19_distance_learned_1day_90stations_seed1",
    ("hybrid_fixed_initial", 1): "M19_hybrid_fixed_initial_1day_90stations_seed1",
    ("hybrid_fixed_one", 1): "M19_hybrid_fixed_one_1day_90stations_seed1",
}


def invoke(command: list[str]) -> None:
    print("[M23]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate(run_name: str, device: str, force: bool = False) -> dict:
    run_dir = ROOT / "experiments" / run_name
    metric = run_dir / "real_test_forecast_metrics.json"
    if force or not metric.exists() or "efficiency" not in json.loads(metric.read_text(encoding="utf-8")):
        invoke([sys.executable, str(EVALUATE), "--device", device, "--checkpoint", str(run_dir / "best.pt"), "--save-forecasts"])
    return json.loads(metric.read_text(encoding="utf-8"))


def aggregate(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    records = []
    for value in sorted({str(row[key]) for row in rows}):
        group = [row for row in rows if str(row[key]) == value]
        record: dict[str, object] = {key: value, "seeds": len(group)}
        for field in ("rmse_mm", "mae_mm", "skill_vs_persistence", "milliseconds_per_window"):
            values = np.array([float(row[field]) for row in group])
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        record["parameter_count"] = int(group[0]["parameter_count"])
        records.append(record)
    return records


def run_multiseed(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for variant, extra in VARIANTS.items():
        for seed_index, seed in enumerate(SEEDS, start=1):
            run_name = EXISTING.get((variant, seed_index), f"M23_{variant}_1day_90stations_seed{seed_index}")
            run_dir = ROOT / "experiments" / run_name
            metric = run_dir / "real_test_forecast_metrics.json"
            if not (args.resume and metric.exists()):
                if not (run_dir / "best.pt").exists():
                    command = [sys.executable, str(TRAIN), "--device", args.device, "--epochs", str(args.epochs), "--horizon", "1", "--seed", str(seed), "--run-name", run_name] + extra
                    invoke(command)
                payload = evaluate(run_name, args.device, force=True)
            else:
                payload = evaluate(run_name, args.device)
            rows.append({
                "variant": variant, "seed": seed, "run_name": run_name,
                "rmse_mm": payload["model"]["overall"]["rmse_mm"],
                "mae_mm": payload["model"]["overall"]["mae_mm"],
                "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
                **payload["efficiency"],
            })
    output = {"runs": rows, "summary": aggregate(rows, "variant")}
    destination = ROOT / "reports" / "M23_core_algorithm_ablation_3seed.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M23] multiseed complete: {destination}")


def sensitivity_configs() -> list[tuple[str, list[str], dict[str, object]]]:
    configs: list[tuple[str, list[str], dict[str, object]]] = [("default", [], {"context": 60, "ewma_span": 31, "neighbors": 8, "correlation_weight": 0.75})]
    configs += [(f"context_{value}", ["--context", str(value)], {"context": value, "ewma_span": 31, "neighbors": 8, "correlation_weight": 0.75}) for value in (30, 90, 120)]
    configs += [(f"ewma_{value}", ["--ewma-span", str(value)], {"context": 60, "ewma_span": value, "neighbors": 8, "correlation_weight": 0.75}) for value in (15, 61, 91)]
    configs += [(f"k_{value}", ["--neighbors", str(value)], {"context": 60, "ewma_span": 31, "neighbors": value, "correlation_weight": 0.75}) for value in (4, 12, 16)]
    configs += [(f"weight_{value:.2f}".replace(".", "p"), ["--correlation-weight", str(value)], {"context": 60, "ewma_span": 31, "neighbors": 8, "correlation_weight": value}) for value in (0.0, 0.25, 0.5, 1.0)]
    return configs


def run_sensitivity(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for label, extra, settings in sensitivity_configs():
        if label == "default":
            run_name = "M7_gpu_full_1day_v1"
        else:
            run_name = f"M23_sensitivity_{label}_seed1"
        run_dir = ROOT / "experiments" / run_name
        metric = run_dir / "real_test_forecast_metrics.json"
        if not (args.resume and metric.exists()):
            if not (run_dir / "best.pt").exists():
                invoke([sys.executable, str(TRAIN), "--device", args.device, "--epochs", str(args.epochs), "--horizon", "1", "--seed", str(SEEDS[0]), "--run-name", run_name] + extra)
            payload = evaluate(run_name, args.device, force=True)
        else:
            payload = evaluate(run_name, args.device)
        rows.append({
            "setting": label, "run_name": run_name, **settings,
            "rmse_mm": payload["model"]["overall"]["rmse_mm"],
            "mae_mm": payload["model"]["overall"]["mae_mm"],
            "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"],
            **payload["efficiency"],
        })
    destination = ROOT / "reports" / "M23_parameter_sensitivity_efficiency.json"
    destination.write_text(json.dumps({"seed": SEEDS[0], "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M23] sensitivity complete: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "multiseed", "sensitivity"), default="all")
    args = parser.parse_args()
    if args.stage in {"all", "multiseed"}:
        run_multiseed(args)
    if args.stage in {"all", "sensitivity"}:
        run_sensitivity(args)


if __name__ == "__main__":
    main()
