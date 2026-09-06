"""Station-bootstrap test for the paired graph-correction ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("M7_gpu_full_1day_v1", "M7_gpu_ablation_no_graph_v1"),
    ("M7_gpu_full_1day_seed2", "M7_gpu_ablation_no_graph_seed2"),
    ("M23_proposed_1day_90stations_seed3", "M23_no_graph_1day_90stations_seed3"),
)


def station_mse(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    error = archive["prediction_enu_mm"] - archive["truth_enu_mm"]
    mask = archive["observed_mask"]
    return np.stack([np.mean(error[:, :, station, :][mask[:, :, station]] ** 2) for station in range(error.shape[2])])


def main() -> None:
    rng = np.random.default_rng(20260803); records = []
    for graph_run, no_graph_run in PAIRS:
        graph = np.load(ROOT / "experiments" / graph_run / "real_test_forecasts.npz")
        no_graph = np.load(ROOT / "experiments" / no_graph_run / "real_test_forecasts.npz")
        graph_mse, no_graph_mse = station_mse(graph), station_mse(no_graph)
        gain = 1 - np.sqrt(graph_mse) / np.sqrt(no_graph_mse)
        # Resample stations, preserving all time samples within each station.
        draws = rng.integers(0, len(gain), size=(10000, len(gain)))
        bootstrap = gain[draws].mean(axis=1)
        records.append({
            "graph_run": graph_run, "no_graph_run": no_graph_run,
            "stations_improved": int((gain > 0).sum()), "stations": int(len(gain)),
            "mean_station_rmse_gain_fraction": float(gain.mean()),
            "bootstrap_95_ci_fraction": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "bootstrap_probability_gain_positive": float(np.mean(bootstrap > 0)),
        })
    result = {"method": "10,000 paired station bootstrap resamples; each station retains its full frozen-test time series.", "pairs": records}
    destination = ROOT / "reports" / "M7_station_bootstrap_spatial_gain_v2_3seed.json"; destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
