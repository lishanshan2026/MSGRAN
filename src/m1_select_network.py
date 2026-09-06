"""Select an auditable common-time-window GNSS station network."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data_raw" / "ngl_tenv3_igs20_200stations"
COORDINATES = PROJECT_ROOT / "code_reference" / "sse-cascadia" / "INPUT_FILES" / "stations_cascadia_200.txt"


def mjd(value: str) -> int:
    return (date.fromisoformat(value) - date(1858, 11, 17)).days


def coordinates() -> dict[str, tuple[float, float]]:
    return {parts[0]: (float(parts[1]), float(parts[2])) for parts in (line.split() for line in COORDINATES.read_text().splitlines()) if parts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-gap", type=int, default=30)
    args = parser.parse_args()
    start, end = mjd(args.start), mjd(args.end)
    duration = end - start + 1
    coords = coordinates()
    records = []
    for path in sorted(RAW_DIRECTORY.glob("*.tenv3")):
        values = np.loadtxt(path, skiprows=1, usecols=3, dtype=np.int64)
        days = values[(values >= start) & (values <= end)]
        gaps = np.diff(days)
        max_gap = int(gaps.max(initial=1) - 1)
        coverage = len(days) / duration
        lat, lon = coords[path.stem]
        records.append({"station": path.stem, "latitude": lat, "longitude": lon, "rows_in_window": len(days), "coverage_fraction": round(coverage, 6), "max_gap_days": max_gap, "eligible": int(coverage >= args.min_coverage and max_gap <= args.max_gap)})
    output = PROJECT_ROOT / "reports" / f"M1_最终候选站_{args.start}_至_{args.end}.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"window={args.start}..{args.end} eligible={sum(x['eligible'] for x in records)}/{len(records)} output={output}")


if __name__ == "__main__":
    main()
