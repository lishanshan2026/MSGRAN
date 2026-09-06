"""M1: resumable downloader for the NGL candidate-station raw archive.

It preserves the downloaded files exactly as supplied by NGL and records an
SHA-256 manifest. Existing verified files are not re-downloaded.
"""

from __future__ import annotations

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVAILABILITY_CSV = PROJECT_ROOT / "reports" / "M1_NGL_200站端点可用性.csv"
RAW_DIRECTORY = PROJECT_ROOT / "data_raw" / "ngl_tenv3_igs20_200stations"
MANIFEST_CSV = PROJECT_ROOT / "data_raw" / "M1_NGL_200站下载清单.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def download(record: dict[str, str], timeout: int) -> dict[str, object]:
    station = record["station"]
    expected_size = int(record["content_length_bytes"])
    output = RAW_DIRECTORY / f"{station}.tenv3"
    partial = output.with_suffix(".tenv3.part")
    if output.exists() and output.stat().st_size == expected_size:
        return {"station": station, "status": "existing", "bytes": output.stat().st_size, "sha256": sha256(output), "error": ""}
    try:
        with requests.get(record["url"], timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch: got {partial.stat().st_size}, expected {expected_size}")
        partial.replace(output)
        return {"station": station, "status": "downloaded", "bytes": output.stat().st_size, "sha256": sha256(output), "error": ""}
    except Exception as exc:  # Keep .part for diagnosis; never treat it as raw data.
        return {"station": station, "status": "failed", "bytes": partial.stat().st_size if partial.exists() else 0, "sha256": "", "error": str(exc)}


def main() -> None:
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with AVAILABILITY_CSV.open(encoding="utf-8-sig", newline="") as handle:
        sources = [row for row in csv.DictReader(handle) if row["status_code"] == "200"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download, source, 120) for source in sources]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: str(item["station"]))
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station", "status", "bytes", "sha256", "error"])
        writer.writeheader()
        writer.writerows(results)
    failures = [item for item in results if item["status"] == "failed"]
    print(f"stations={len(results)} ok={len(results) - len(failures)} failed={len(failures)} raw_dir={RAW_DIRECTORY}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
