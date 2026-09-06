"""Report frozen component-wise 1-day test metrics from archived GPU forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT = ROOT / "reports" / "M12_component_metrics_1day_v1.json"


def forecast_path(run_name: str) -> Path:
    matches = list(EXPERIMENTS.rglob(f"{run_name}/real_test_forecasts.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archived forecast for {run_name}, found {matches}")
    return matches[0]


def metrics(prediction: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    component_names = ("east", "north", "up")
    result: dict[str, object] = {}
    for index, name in enumerate(component_names):
        error = (prediction[..., index] - truth[..., index])[observed]
        result[name] = {
            "rmse_mm": float(np.sqrt(np.mean(error**2))),
            "mae_mm": float(np.mean(np.abs(error))),
        }
    expanded_mask = np.repeat(observed[..., None], 3, axis=-1)
    pooled_error = (prediction - truth)[expanded_mask]
    result["pooled_enu"] = {
        "rmse_mm": float(np.sqrt(np.mean(pooled_error**2))),
        "mae_mm": float(np.mean(np.abs(pooled_error))),
    }
    return result


def main() -> None:
    full = np.load(forecast_path("M7_gpu_full_1day_seed2"))
    no_graph = np.load(forecast_path("M7_gpu_ablation_no_graph_seed2"))
    truth = full["truth_enu_mm"]
    observed = full["observed_mask"]
    report = {
        "protocol": "Archived seed-2 one-day forecasts; values use observed held-out E/N/U components only.",
        "full": metrics(full["prediction_enu_mm"], truth, observed),
        "no_graph": metrics(no_graph["prediction_enu_mm"], truth, observed),
        "persistence": metrics(full["persistence_enu_mm"], truth, observed),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
