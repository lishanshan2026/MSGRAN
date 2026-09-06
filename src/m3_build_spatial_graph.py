"""Construct a fixed, distance-weighted kNN graph for the frozen GNSS network."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_residuals_v1.npz"
OUTPUT = PROJECT_ROOT / "data_processed" / "ngl_cascadia_90stations_knn8_graph_v1.npz"


def main() -> None:
    data = np.load(SOURCE)
    lat, lon = np.radians(data["latitude"]), np.radians(data["longitude"])
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2))
    np.fill_diagonal(distance, np.inf)
    n, k = len(lat), 8
    neighbours = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = neighbours.reshape(-1)
    distances = distance[rows, cols]
    # 100 km radial basis is fixed before model fitting and is interpretable.
    weights = np.exp(-0.5 * (distances / 100.0) ** 2).astype(np.float32)
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    np.savez_compressed(OUTPUT, edge_index=edge_index, edge_weight=weights, edge_distance_km=distances.astype(np.float32), station_codes=data["station_codes"], graph_note=np.asarray("Directed kNN graph: k=8; edge weight=exp(-0.5*(distance_km/100)^2)."))
    print(f"nodes={n} directed_edges={edge_index.shape[1]} distance_km_min_median_max={distances.min():.3f},{np.median(distances):.3f},{distances.max():.3f}")


if __name__ == "__main__":
    main()
