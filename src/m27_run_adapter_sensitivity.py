"""Run one-factor sensitivity tests for the final NLinear graph adapter.

All tests use seed 20260803 and the frozen 2022--2024 test split.  Context
tests retrain the matched NLinear backbone because changing the input length
changes its two linear projections.  Graph-parameter tests reuse the frozen
60-day DLinear backbone and vary exactly one adapter setting at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "src" / "m22_train_strong_baseline.py"
BASE_EVALUATE = ROOT / "src" / "m22_evaluate_strong_baseline.py"
ADAPTER_TRAIN = ROOT / "src" / "m26_train_dlinear_graph_adapter.py"
ADAPTER_EVALUATE = ROOT / "src" / "m26_evaluate_dlinear_graph_adapter.py"
SEED = 20260803


def invoke(command: list[str]) -> None:
    print("[M27]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_if_needed(run_dir: Path, evaluator: Path, device: str, resume: bool) -> dict:
    metric = run_dir / "real_test_forecast_metrics.json"
    if not (resume and metric.exists()):
        invoke([sys.executable, str(evaluator), "--checkpoint", str(run_dir / "best.pt"), "--device", device])
    return json.loads(metric.read_text(encoding="utf-8"))


def ensure_nlinear(context: int, epochs: int, device: str, resume: bool) -> tuple[Path, dict]:
    if context == 60:
        run_dir = ROOT / "experiments" / "M28_nlinear_1day_90stations_seed1"
    else:
        run_dir = ROOT / "experiments" / f"M27_nlinear_context{context}_seed1"
        metric = run_dir / "real_test_forecast_metrics.json"
        if not (resume and metric.exists()):
            invoke([
                sys.executable, str(BASE_TRAIN), "--model", "nlinear", "--context", str(context),
                "--horizon", "1", "--epochs", str(epochs), "--seed", str(SEED),
                "--device", device, "--run-name", run_dir.name,
            ])
    return run_dir / "best.pt", evaluate_if_needed(run_dir, BASE_EVALUATE, device, resume)


def ensure_adapter(
    run_name: str,
    base_checkpoint: Path,
    epochs: int,
    device: str,
    resume: bool,
    *,
    ewma_span: int = 31,
    neighbors: int = 8,
    correlation_weight: float = 0.75,
) -> dict:
    run_dir = ROOT / "experiments" / run_name
    metric = run_dir / "real_test_forecast_metrics.json"
    if not (resume and metric.exists()):
        invoke([
            sys.executable, str(ADAPTER_TRAIN), "--base-checkpoint", str(base_checkpoint),
            "--run-name", run_name, "--epochs", str(epochs), "--seed", str(SEED),
            "--device", device, "--ewma-span", str(ewma_span), "--neighbors", str(neighbors),
            "--correlation-weight", str(correlation_weight),
        ])
    return evaluate_if_needed(run_dir, ADAPTER_EVALUATE, device, resume)


def row(category: str, value: float, adapter: dict, base: dict) -> dict:
    adapter_rmse = float(adapter["model"]["overall"]["rmse_mm"])
    base_rmse = float(base["model"]["overall"]["rmse_mm"])
    return {
        "category": category,
        "value": value,
        "adapter_rmse_mm": adapter_rmse,
        "nlinear_rmse_mm": base_rmse,
        "gain_vs_nlinear_percent": 100.0 * (1.0 - adapter_rmse / base_rmse),
        "skill_vs_persistence_percent": 100.0 * float(adapter["overall_rmse_skill_vs_persistence"]),
        "graph_gate": float(adapter["graph_gate"]),
        "parameter_count": int(adapter["efficiency"]["parameter_count"]),
        "trainable_parameter_count": int(adapter["efficiency"]["trainable_parameter_count"]),
        "milliseconds_per_window": float(adapter["efficiency"]["milliseconds_per_window"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    default_base_checkpoint, default_base = ensure_nlinear(60, args.epochs, args.device, args.resume)
    default_adapter = ensure_adapter(
        "M29_nlinear_graph_adapter_1day_90stations_seed1", default_base_checkpoint,
        args.epochs, args.device, args.resume,
    )

    for context in (30, 60, 90, 120):
        base_checkpoint, base = ensure_nlinear(context, args.epochs, args.device, args.resume)
        adapter = default_adapter if context == 60 else ensure_adapter(
            f"M27_adapter_context{context}_seed1", base_checkpoint, args.epochs, args.device, args.resume,
        )
        rows.append(row("context_days", float(context), adapter, base))

    for span in (15, 31, 61, 91):
        adapter = default_adapter if span == 31 else ensure_adapter(
            f"M27_adapter_ewma{span}_seed1", default_base_checkpoint, args.epochs, args.device, args.resume,
            ewma_span=span,
        )
        rows.append(row("ewma_span_days", float(span), adapter, default_base))

    for neighbors in (4, 8, 12, 16):
        adapter = default_adapter if neighbors == 8 else ensure_adapter(
            f"M27_adapter_k{neighbors}_seed1", default_base_checkpoint, args.epochs, args.device, args.resume,
            neighbors=neighbors,
        )
        rows.append(row("neighbors_k", float(neighbors), adapter, default_base))

    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        adapter = default_adapter if weight == 0.75 else ensure_adapter(
            f"M27_adapter_alpha{str(weight).replace('.', 'p')}_seed1", default_base_checkpoint,
            args.epochs, args.device, args.resume, correlation_weight=weight,
        )
        rows.append(row("correlation_weight", weight, adapter, default_base))

    best = {}
    for category in ("context_days", "ewma_span_days", "neighbors_k", "correlation_weight"):
        candidates = [item for item in rows if item["category"] == category]
        best[category] = min(candidates, key=lambda item: item["adapter_rmse_mm"])
    payload = {
        "protocol": "one-factor-at-a-time; seed 20260803; frozen 2022-2024 test split",
        "default": {"context_days": 60, "ewma_span_days": 31, "neighbors_k": 8, "correlation_weight": 0.75},
        "rows": rows,
        "best_by_category": best,
    }
    destination = ROOT / "reports" / "M27_final_adapter_parameter_sensitivity.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
