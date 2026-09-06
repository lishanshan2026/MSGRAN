"""Leakage-safe CPU diagnostic for the multiscale spatial-dependence hypothesis.

The diagnostic uses a causal exponential low-pass filter.  At day t its low-frequency
value only depends on observations at or before t.  Pairwise correlations are computed
separately per chronological split; only train correlations may be used by a later graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_residuals_v1.npz"
OUT = ROOT / "reports" / "M6_multiscale_correlation_diagnostic_v1.json"


def causal_ewma(values: np.ndarray, observed: np.ndarray, span_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Return causal low-pass and high-pass values without imputing missing samples."""
    alpha = 2.0 / (span_days + 1.0)
    low = np.full_like(values, np.nan, dtype=np.float32)
    for station in range(values.shape[1]):
        state = np.full(values.shape[2], np.nan, dtype=np.float64)
        for day in range(values.shape[0]):
            if not observed[day, station]:
                continue
            current = values[day, station].astype(np.float64)
            state = np.where(np.isfinite(state), (1.0 - alpha) * state + alpha * current, current)
            low[day, station] = state
    high = values - low
    high[~observed] = np.nan
    return low, high


def component_correlations(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Median signed Pearson correlation across E/N/U for every station pair."""
    stations = values.shape[1]
    result = np.eye(stations, dtype=np.float32)
    for i in range(stations):
        for j in range(i + 1, stations):
            good = observed[:, i] & observed[:, j]
            correlations = []
            if int(good.sum()) >= 60:
                for component in range(3):
                    a, b = values[good, i, component], values[good, j, component]
                    if np.nanstd(a) > 1e-7 and np.nanstd(b) > 1e-7:
                        correlations.append(float(np.corrcoef(a, b)[0, 1]))
            result[i, j] = result[j, i] = float(np.median(correlations)) if correlations else np.nan
    return result


def pair_summary(corr: np.ndarray) -> dict[str, float | int]:
    values = corr[np.triu_indices_from(corr, k=1)]
    values = values[np.isfinite(values)]
    return {
        "pairs": int(values.size),
        "median_signed_r": float(np.median(values)),
        "median_absolute_r": float(np.median(np.abs(values))),
        "mean_absolute_r": float(np.mean(np.abs(values))),
        "strong_abs_r_ge_0_5_fraction": float(np.mean(np.abs(values) >= 0.5)),
    }


def correlation_stability(train: np.ndarray, later: np.ndarray) -> float:
    a, b = train[np.triu_indices_from(train, k=1)], later[np.triu_indices_from(later, k=1)]
    good = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[good], b[good])[0, 1]) if int(good.sum()) > 3 else float("nan")


def main() -> None:
    source = np.load(DATA)
    values = source["residual_enu_mm"].astype(np.float32)
    observed, split = source["observed_mask"].astype(bool), source["split"]
    low, high = causal_ewma(values, observed, span_days=31)
    result: dict[str, object] = {
        "purpose": "CPU decision gate for frequency-specific spatial fusion; no test data used to construct a future graph.",
        "filter": {"type": "causal_ewma", "span_days": 31, "definition": "low[t] depends only on values up to t; high[t]=value[t]-low[t]"},
        "splits": {"train": "2010-2019", "validation": "2020-2021", "test": "2022-2024"},
        "components": {},
    }
    all_correlations: dict[str, dict[int, np.ndarray]] = {"low": {}, "high": {}}
    names = {0: "train", 1: "validation", 2: "test"}
    for band, array in (("low", low), ("high", high)):
        result["components"][band] = {}
        for index, name in names.items():
            use = split == index
            corr = component_correlations(array[use], observed[use])
            all_correlations[band][index] = corr
            result["components"][band][name] = pair_summary(corr)
        result["components"][band]["train_to_validation_pair_correlation"] = correlation_stability(all_correlations[band][0], all_correlations[band][1])
        result["components"][band]["train_to_test_pair_correlation"] = correlation_stability(all_correlations[band][0], all_correlations[band][2])
    low_train = result["components"]["low"]["train"]["median_absolute_r"]
    high_train = result["components"]["high"]["train"]["median_absolute_r"]
    stable_low = result["components"]["low"]["train_to_test_pair_correlation"]
    stable_high = result["components"]["high"]["train_to_test_pair_correlation"]
    result["decision"] = {
        "slow_component_has_stronger_spatial_signal": bool(low_train > high_train),
        "slow_component_graph_is_preliminarily_justified": bool(low_train > high_train and stable_low > stable_high),
        "interpretation": "Proceed to a slow-component correlation graph only if both diagnostics are true; otherwise retain graph as a baseline rather than a confirmed mechanism.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

