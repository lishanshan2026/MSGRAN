"""Create leakage-safe GNSS residuals and fixed chronological splits.

Trend coefficients are fitted only on 2010--2019 observations. They are then
applied unchanged to validation and test epochs. No missing values are filled.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_igs20_tenv3_raw.npz"
OUTPUT = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_residuals_v1.npz"


def mjd(value: str) -> int:
    return (date.fromisoformat(value) - date(1858, 11, 17)).days


def main() -> None:
    source = np.load(SOURCE)
    days = source["mjd"]
    positions_m = source["position_enu_m"].astype(np.float64)
    sigma_m = source["sigma_enu_m"].astype(np.float32)
    observed = source["observed_mask"].astype(bool)
    train = days <= mjd("2019-12-31")
    validation = (days >= mjd("2020-01-01")) & (days <= mjd("2021-12-31"))
    test = days >= mjd("2022-01-01")
    split = np.full(len(days), -1, dtype=np.int8)
    split[train], split[validation], split[test] = 0, 1, 2
    relative_days = (days - days[0]).astype(np.float64)
    slopes = np.full((positions_m.shape[1], 3), np.nan, dtype=np.float64)
    intercepts = np.full_like(slopes, np.nan)
    for station in range(positions_m.shape[1]):
        for component in range(3):
            usable = train & observed[:, station] & np.isfinite(positions_m[:, station, component])
            slopes[station, component], intercepts[station, component] = np.polyfit(relative_days[usable], positions_m[usable, station, component], 1)
    trend = relative_days[:, None, None] * slopes[None, :, :] + intercepts[None, :, :]
    residual_mm = ((positions_m - trend) * 1000).astype(np.float32)
    residual_mm[~observed, :] = np.nan
    # Train-only robust normalisation constants for later neural-network input.
    train_values = np.where(train[:, None, None] & observed[:, :, None], residual_mm, np.nan)
    center = np.nanmedian(train_values, axis=0).astype(np.float32)
    scale = np.nanmedian(np.abs(train_values - center[None, :, :]), axis=0).astype(np.float32)
    scale = np.maximum(scale * 1.4826, 0.1)  # robust sigma, at least 0.1 mm
    normalized = ((residual_mm - center[None, :, :]) / scale[None, :, :]).astype(np.float32)
    normalized[~observed, :] = np.nan
    np.savez_compressed(
        OUTPUT,
        mjd=days,
        split=split,
        station_codes=source["station_codes"],
        latitude=source["latitude"],
        longitude=source["longitude"],
        observed_mask=observed,
        sigma_enu_mm=(sigma_m * 1000).astype(np.float32),
        residual_enu_mm=residual_mm,
        normalized_residual_enu=normalized,
        train_center_mm=center,
        train_robust_scale_mm=scale,
        trend_slope_m_per_day=slopes.astype(np.float32),
        trend_intercept_m=intercepts.astype(np.float32),
        split_note=np.asarray("0=train:2010-01-01..2019-12-31; 1=validation:2020-01-01..2021-12-31; 2=test:2022-01-01..2024-12-31"),
        processing_note=np.asarray("Linear E/N/U trend fitted on train split only; no imputation; residuals in mm."),
    )
    print(f"output={OUTPUT} train={train.sum()} validation={validation.sum()} test={test.sum()} observed={observed.mean():.6f}")


if __name__ == "__main__":
    main()
