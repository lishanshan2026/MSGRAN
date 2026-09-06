"""Traditional GNSS forecasting baselines for the frozen protocol.

Baselines:
- seasonal naive: last year's same-day observation, falling back to persistence.
- harmonic trend: linear trend plus annual and semiannual sin/cos terms.
- harmonic plus ridge-AR residual: harmonic trend with a station/component AR
  correction fit only from training-period residual windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from m6_multiscale_correlation_diagnostic import DATA, ROOT
from m56_component_gate_screen import metric_block


COMPONENTS = ("east", "north", "up")


def design_matrix(mjd: np.ndarray, origin: float) -> np.ndarray:
    days = mjd.astype(np.float64) - origin
    year = 365.25
    return np.column_stack(
        (
            np.ones_like(days),
            days,
            np.sin(2.0 * np.pi * days / year),
            np.cos(2.0 * np.pi * days / year),
            np.sin(4.0 * np.pi * days / year),
            np.cos(4.0 * np.pi * days / year),
        )
    )


def fit_harmonic(raw: np.ndarray, mask: np.ndarray, split: np.ndarray, mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    origin = float(mjd[split == 0][0])
    x_all = design_matrix(mjd, origin)
    train = split == 0
    coefficients = np.zeros((raw.shape[1], raw.shape[2], x_all.shape[1]), dtype=np.float64)
    prediction = np.full_like(raw, np.nan, dtype=np.float32)
    for station in range(raw.shape[1]):
        for component in range(raw.shape[2]):
            valid = train & mask[:, station] & np.isfinite(raw[:, station, component])
            if int(valid.sum()) >= x_all.shape[1] + 10:
                beta, *_ = np.linalg.lstsq(x_all[valid], raw[valid, station, component].astype(np.float64), rcond=None)
            else:
                beta = np.zeros(x_all.shape[1], dtype=np.float64)
            coefficients[station, component] = beta
            prediction[:, station, component] = (x_all @ beta).astype(np.float32)
    return coefficients, prediction, origin


def iter_window_targets(split: np.ndarray, context: int, horizon: int, split_id: int) -> list[int]:
    total = context + horizon
    return [start + context for start in range(len(split) - total + 1) if np.all(split[start:start + total] == split_id)]


def persistence_prediction(raw: np.ndarray, mask: np.ndarray, target_days: list[int]) -> np.ndarray:
    out = np.zeros((len(target_days), 1, raw.shape[1], raw.shape[2]), dtype=np.float32)
    for row, target in enumerate(target_days):
        previous = target - 1
        value = np.nan_to_num(raw[previous], nan=0.0)
        out[row, 0] = value
    return out


def seasonal_naive_prediction(raw: np.ndarray, mask: np.ndarray, mjd: np.ndarray, target_days: list[int], context: int) -> np.ndarray:
    by_mjd = {int(day): index for index, day in enumerate(mjd.tolist())}
    out = np.zeros((len(target_days), 1, raw.shape[1], raw.shape[2]), dtype=np.float32)
    for row, target in enumerate(target_days):
        seasonal_index = by_mjd.get(int(mjd[target]) - 365)
        fallback = np.nan_to_num(raw[target - 1], nan=0.0)
        pred = fallback.copy()
        if seasonal_index is not None:
            same_day = raw[seasonal_index]
            same_mask = mask[seasonal_index]
            pred[same_mask] = np.nan_to_num(same_day[same_mask], nan=0.0)
        out[row, 0] = pred
    return out


def direct_prediction(series: np.ndarray, target_days: list[int]) -> np.ndarray:
    out = np.zeros((len(target_days), 1, series.shape[1], series.shape[2]), dtype=np.float32)
    for row, target in enumerate(target_days):
        out[row, 0] = series[target]
    return out


def fit_ar_residual(raw: np.ndarray, mask: np.ndarray, split: np.ndarray, harmonic: np.ndarray, context: int, order: int, alpha: float) -> np.ndarray:
    residual = raw - harmonic
    residual[~mask] = np.nan
    train_targets = iter_window_targets(split, context, 1, 0)
    coefficients = np.zeros((raw.shape[1], raw.shape[2], 2 * order + 1), dtype=np.float64)
    identity = np.eye(2 * order + 1, dtype=np.float64)
    identity[-1, -1] = 0.0
    for station in range(raw.shape[1]):
        for component in range(raw.shape[2]):
            features, targets = [], []
            for target in train_targets:
                start = target - order
                if start < 0 or not mask[target, station] or not np.isfinite(residual[target, station, component]):
                    continue
                r = residual[start:target, station, component]
                m = mask[start:target, station].astype(np.float64)
                features.append(np.concatenate((np.nan_to_num(r, nan=0.0), m, [1.0])))
                targets.append(float(residual[target, station, component]))
            if len(targets) >= 2 * order + 5:
                x = np.asarray(features, dtype=np.float64)
                y = np.asarray(targets, dtype=np.float64)
                coefficients[station, component] = np.linalg.solve(x.T @ x + alpha * identity, x.T @ y)
    return coefficients.astype(np.float32)


def harmonic_ar_prediction(raw: np.ndarray, mask: np.ndarray, harmonic: np.ndarray, coefficients: np.ndarray, target_days: list[int], order: int) -> np.ndarray:
    residual = raw - harmonic
    residual[~mask] = np.nan
    out = np.zeros((len(target_days), 1, raw.shape[1], raw.shape[2]), dtype=np.float32)
    for row, target in enumerate(target_days):
        pred = harmonic[target].copy()
        start = target - order
        for station in range(raw.shape[1]):
            m = mask[start:target, station].astype(np.float32) if start >= 0 else np.zeros(order, dtype=np.float32)
            for component in range(raw.shape[2]):
                if start >= 0:
                    r = residual[start:target, station, component]
                    feature = np.concatenate((np.nan_to_num(r, nan=0.0), m, [1.0])).astype(np.float32)
                    pred[station, component] += float(feature @ coefficients[station, component])
        out[row, 0] = pred
    return out


def truth_and_mask(raw: np.ndarray, mask: np.ndarray, target_days: list[int]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.zeros((len(target_days), 1, raw.shape[1], raw.shape[2]), dtype=np.float32)
    valid = np.zeros((len(target_days), 1, raw.shape[1]), dtype=bool)
    for row, target in enumerate(target_days):
        truth[row, 0] = raw[target]
        valid[row, 0] = mask[target]
    return truth, valid


def evaluate(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, float]]:
    return metric_block(prediction, truth, valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ar-order", type=int, default=30)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--station-indices", default="")
    parser.add_argument("--run-name", default="M59_traditional_gnss_baselines_1day_90stations")
    args = parser.parse_args()
    if args.horizon != 1:
        raise ValueError("M59 traditional baselines currently support one-day forecasts only")

    source = np.load(DATA)
    raw = source["residual_enu_mm"].astype(np.float32)
    mask = source["observed_mask"].astype(bool)
    split = source["split"]
    mjd = source["mjd"].astype(np.int64)
    station_indices = np.array([int(value) for value in args.station_indices.split(",") if value.strip()], dtype=int) if args.station_indices else np.arange(raw.shape[1])
    raw, mask = raw[:, station_indices], mask[:, station_indices]

    _, harmonic_all, origin = fit_harmonic(raw, mask, split, mjd)
    ar_coefficients = fit_ar_residual(raw, mask, split, harmonic_all, args.context, args.ar_order, args.ridge_alpha)

    results: dict[str, object] = {
        "protocol": {
            "dataset": str(DATA),
            "context": args.context,
            "horizon": args.horizon,
            "ar_order": args.ar_order,
            "ridge_alpha": args.ridge_alpha,
            "harmonic_terms": "intercept + linear trend + annual sin/cos + semiannual sin/cos",
            "fit_split": "train only, 2010-2019",
            "station_count": int(len(station_indices)),
        },
        "splits": {},
    }
    for split_id, split_name in ((1, "validation_2020_2021"), (2, "frozen_test_2022_2024")):
        target_days = iter_window_targets(split, args.context, args.horizon, split_id)
        truth, valid = truth_and_mask(raw, mask, target_days)
        predictions = {
            "persistence": persistence_prediction(raw, mask, target_days),
            "seasonal_naive": seasonal_naive_prediction(raw, mask, mjd, target_days, args.context),
            "harmonic_trend": direct_prediction(harmonic_all, target_days),
            "harmonic_ar_residual": harmonic_ar_prediction(raw, mask, harmonic_all, ar_coefficients, target_days, args.ar_order),
        }
        split_result = {
            "forecast_windows": int(len(target_days)),
            "observed_samples": int(valid.sum() * 3),
            "models": {},
        }
        for name, prediction in predictions.items():
            split_result["models"][name] = evaluate(prediction, truth, valid)
        results["splits"][split_name] = split_result

    out = ROOT / "experiments" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
