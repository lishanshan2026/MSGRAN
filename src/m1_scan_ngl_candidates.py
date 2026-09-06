"""M1: read-only availability scan for NGL IGS20 tenv3 candidate stations.

No time-series files are downloaded. The script writes a timestamped CSV and
JSON summary below reports/ so that the eventual station list is auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIONS = (
    PROJECT_ROOT / "code_reference" / "sse-cascadia" / "INPUT_FILES" / "stations_cascadia_200.txt"
)
URL_TEMPLATE = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{station}.tenv3"


def load_stations(path: Path) -> list[str]:
    stations: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            stations.append(fields[0].upper())
    return stations


def probe(station: str, timeout: int) -> dict[str, object]:
    url = URL_TEMPLATE.format(station=station)
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return {
            "station": station,
            "status_code": response.status_code,
            "content_length_bytes": int(response.headers.get("Content-Length", 0) or 0),
            "url": url,
            "error": "",
        }
    except requests.RequestException as exc:
        return {"station": station, "status_code": 0, "content_length_bytes": 0, "url": url, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    stations = load_stations(args.stations)
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(probe, station, args.timeout) for station in stations]
        records = [future.result() for future in as_completed(futures)]
    records.sort(key=lambda item: str(item["station"]))

    reports = PROJECT_ROOT / "reports"
    csv_path = reports / "M1_NGL_200站端点可用性.csv"
    json_path = reports / "M1_NGL_200站端点可用性_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station", "status_code", "content_length_bytes", "url", "error"])
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "checked_at_utc": checked_at,
        "product": "NGL IGS20 24-hour final tenv3",
        "candidate_count": len(stations),
        "available_count": sum(record["status_code"] == 200 for record in records),
        "failed_count": sum(record["status_code"] != 200 for record in records),
        "content_length_total_bytes": sum(int(record["content_length_bytes"]) for record in records if record["status_code"] == 200),
        "csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
