"""Build a fixed-calendar GNSS archive from the frozen M1 station network.

The output preserves raw local E/N/U coordinates in metres, their formal
uncertainties, and a Boolean observation mask. It does not demean, detrend,
interpolate, or otherwise alter observations.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data_raw" / "ngl_tenv3_igs20_200stations"
SELECTION = PROJECT_ROOT / "reports" / "M1_最终候选站_2010-01-01_至_2024-12-31.csv"
OUTPUT = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_igs20_tenv3_raw.npz"
METADATA = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_metadata.csv"


def mjd(value: str) -> int:
    return (date.fromisoformat(value) - date(1858, 11, 17)).days


def main() -> None:
    start, end = mjd("2010-01-01"), mjd("2024-12-31")
    calendar = np.arange(start, end + 1, dtype=np.int32)
    with SELECTION.open(encoding="utf-8-sig", newline="") as handle:
        stations = [row for row in csv.DictReader(handle) if row["eligible"] == "1"]
    codes = [row["station"] for row in stations]
    n_stations, n_days = len(codes), len(calendar)
    positions = np.full((n_days, n_stations, 3), np.nan, dtype=np.float32)
    sigma = np.full((n_days, n_stations, 3), np.nan, dtype=np.float32)
    mask = np.zeros((n_days, n_stations), dtype=bool)
    for station_index, code in enumerate(codes):
        # NGL tenv3 stores an integer reference component next to a fractional
        # local ENU component.  The latter is the deformation time series to
        # plot and analyse; adding the integer reference produces artificial
        # metre-scale offsets when the reference representation changes.
        # Source columns: MJD, E/N/U fractional local position, sigma E/N/U.
        data = np.loadtxt(RAW_DIRECTORY / f"{code}.tenv3", skiprows=1, usecols=[3, 8, 10, 12, 14, 15, 16])
        day = data[:, 0].astype(np.int32)
        keep = (day >= start) & (day <= end)
        data, day = data[keep], day[keep]
        index = day - start
        positions[index, station_index, :] = data[:, 1:4]
        sigma[index, station_index, :] = data[:, 4:7]
        mask[index, station_index] = True
    np.savez_compressed(
        OUTPUT,
        mjd=calendar,
        station_codes=np.asarray(codes),
        latitude=np.asarray([float(row["latitude"]) for row in stations], dtype=np.float32),
        longitude=np.asarray([float(row["longitude"]) for row in stations], dtype=np.float32),
        position_enu_m=positions,
        sigma_enu_m=sigma,
        observed_mask=mask,
        source_product=np.asarray("NGL IGS20 24-hour final tenv3"),
        processing_note=np.asarray("NGL tenv3 fractional local ENU components only; no interpolation, demeaning, detrending, or outlier removal."),
    )
    with METADATA.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station", "latitude", "longitude", "observed_days", "coverage_fraction"])
        writer.writeheader()
        for station_index, station in enumerate(stations):
            observed = int(mask[:, station_index].sum())
            writer.writerow({**{key: station[key] for key in ("station", "latitude", "longitude")}, "observed_days": observed, "coverage_fraction": round(observed / n_days, 6)})
    print(f"output={OUTPUT} shape={positions.shape} observed_fraction={mask.mean():.6f}")


if __name__ == "__main__":
    main()
