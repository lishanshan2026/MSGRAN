"""M1: quality audit for downloaded NGL IGS20 tenv3 daily series."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data_raw" / "ngl_tenv3_igs20_200stations"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "M1_NGL_200站时间覆盖与缺测审计.csv"


def audit(path: Path) -> dict[str, object]:
    # Columns 3--23 are numeric; MJD is source column 4 => index 1 here.
    values = np.loadtxt(path, skiprows=1, usecols=range(2, 23))
    mjd = values[:, 1].astype(np.int64)
    gaps = np.diff(mjd)
    duration = int(mjd[-1] - mjd[0] + 1)
    return {
        "station": path.stem,
        "rows": len(values),
        "mjd_start": int(mjd[0]),
        "mjd_end": int(mjd[-1]),
        "duration_days": duration,
        "coverage_fraction": round(len(values) / duration, 6),
        "gap_count": int(np.sum(gaps > 1)),
        "missing_days": int(np.maximum(gaps - 1, 0).sum()),
        "max_gap_days": int(gaps.max(initial=1) - 1),
        "mean_sigma_e_mm": round(float(values[:, 12].mean() * 1000), 4),
        "mean_sigma_n_mm": round(float(values[:, 13].mean() * 1000), 4),
        "mean_sigma_u_mm": round(float(values[:, 14].mean() * 1000), 4),
    }


def main() -> None:
    paths = sorted(RAW_DIRECTORY.glob("*.tenv3"))
    if len(paths) != 200:
        raise SystemExit(f"Expected 200 complete files, found {len(paths)}. Finish download before audit.")
    records = [audit(path) for path in paths]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    eligible = [record for record in records if record["coverage_fraction"] >= 0.90 and record["max_gap_days"] <= 30]
    print(f"audited={len(records)} eligible_per_station_rule={len(eligible)} output={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
