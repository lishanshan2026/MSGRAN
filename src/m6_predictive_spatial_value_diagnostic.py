"""Test whether a train-only correlation neighborhood adds 1-day predictive value.

This is deliberately a linear diagnostic, not the final neural model.  If spatial
features cannot improve a simple held-out forecast here, GPU graph training is not
scientifically justified.  All neighbor identities and normalization use train only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma, component_correlations


OUT = ROOT / "reports" / "M6_predictive_spatial_value_diagnostic_v1.json"


def normalized(values: np.ndarray, observed: np.ndarray, train: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float32)
    for station in range(values.shape[1]):
        good = train & observed[:, station]
        mean = np.nanmean(values[good, station], axis=0)
        scale = np.nanstd(values[good, station], axis=0)
        result[:, station] = (values[:, station] - mean) / np.maximum(scale, 1e-4)
    return result


def fit_mse(x: np.ndarray, y: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray) -> tuple[float, int]:
    design = np.column_stack((np.ones(int(train_rows.sum())), x[train_rows]))
    beta = np.linalg.solve(design.T @ design + np.diag([1e-8] + [1e-3] * x.shape[1]), design.T @ y[train_rows])
    error = np.column_stack((np.ones(int(test_rows.sum())), x[test_rows])) @ beta - y[test_rows]
    return float(np.mean(error**2)), int(test_rows.sum())


def evaluate_band(values: np.ndarray, observed: np.ndarray, split: np.ndarray, horizon_days: int) -> dict[str, float | int]:
    train_days = split == 0
    test_days = split == 2
    z = normalized(values, observed, train_days)
    corr = component_correlations(values[train_days], observed[train_days])
    own_mse, graph_mse, samples, improved = [], [], 0, 0
    # Predict t+horizon using information at t.  The target date determines
    # whether the example belongs to the chronological train or test split.
    for station in range(values.shape[1]):
        neighbors = np.argsort(np.nan_to_num(np.abs(corr[station]), nan=-1.0))[::-1]
        neighbors = [int(x) for x in neighbors if x != station][:8]
        for component in range(3):
            neighbor_value = np.nanmean(z[:, neighbors, component], axis=1)
            own = z[:-horizon_days, station, component]
            target = z[horizon_days:, station, component]
            valid_base = np.isfinite(own) & np.isfinite(target)
            valid_graph = valid_base & np.isfinite(neighbor_value[:-horizon_days])
            tr_base, te_base = valid_base & train_days[horizon_days:], valid_base & test_days[horizon_days:]
            tr_graph, te_graph = valid_graph & train_days[horizon_days:], valid_graph & test_days[horizon_days:]
            if tr_graph.sum() < 300 or te_graph.sum() < 100:
                continue
            base, _ = fit_mse(own[:, None], target, tr_base & valid_graph, te_graph)
            graph, count = fit_mse(np.column_stack((own, neighbor_value[:-horizon_days])), target, tr_graph, te_graph)
            own_mse.append(base); graph_mse.append(graph); samples += count
            improved += int(graph < base)
    own_value, graph_value = float(np.mean(own_mse)), float(np.mean(graph_mse))
    return {
        "horizon_days": horizon_days, "targets": int(len(own_mse)), "test_samples_across_targets": samples,
        "own_history_mse_standardized": own_value,
        "train_correlation_neighbor_mse_standardized": graph_value,
        "relative_mse_gain": float(1.0 - graph_value / own_value),
        "fraction_of_station_component_targets_improved": float(improved / len(own_mse)),
    }


def main() -> None:
    source = np.load(DATA)
    values = source["residual_enu_mm"].astype(np.float32)
    observed, split = source["observed_mask"].astype(bool), source["split"]
    low, high = causal_ewma(values, observed, span_days=31)
    result = {
        "purpose": "Held-out 1-day and 7-day linear test of whether train-only correlation-neighbor information adds predictive value beyond own history.",
        "data_split": "train 2010-2019; test 2022-2024; neighbor graph and normalization fitted on train only",
        "neighborhood": "eight largest absolute median E/N/U Pearson correlations, excluding self",
        "one_day": {"low": evaluate_band(low, observed, split, 1), "high": evaluate_band(high, observed, split, 1)},
        "seven_day": {"low": evaluate_band(low, observed, split, 7), "high": evaluate_band(high, observed, split, 7)},
    }
    result["decision"] = {
        "one_day_slow_band_spatial_features_add_value": bool(result["one_day"]["low"]["relative_mse_gain"] > 0 and result["one_day"]["low"]["fraction_of_station_component_targets_improved"] > 0.5),
        "seven_day_slow_band_spatial_features_add_value": bool(result["seven_day"]["low"]["relative_mse_gain"] > 0 and result["seven_day"]["low"]["fraction_of_station_component_targets_improved"] > 0.5),
        "interpretation": "A positive held-out result is required before graph training is made a central model claim. This screen does not itself establish neural-model performance.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

