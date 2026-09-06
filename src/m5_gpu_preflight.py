"""Local preflight for GPU memory and data-shape checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_graph_recon_predict import GraphReconPredict  # noqa: E402
from m4_train_graph_recon_predict import WindowDataset


def main() -> None:
    source = np.load(ROOT / "data_processed" / "ngl_cascadia_90stations_2010-2024_residuals_v1.npz")
    values, mask, split = source["normalized_residual_enu"].astype(np.float32), source["observed_mask"].astype(bool), source["split"]
    context, horizon, batch = 60, 1, 64
    windows = {name: len(WindowDataset(values, mask, split, index, context, horizon)) for index, name in enumerate(("train", "validation", "test"))}
    model = GraphReconPredict(values.shape[1], hidden=64, context=context, horizon=horizon, adaptive_graph=True)
    with torch.no_grad():
        x = torch.from_numpy(np.nan_to_num(values[None, :context], nan=0.0))
        observed = torch.from_numpy(mask[None, :context])
        reconstruction, forecast = model(x, observed, torch.eye(values.shape[1]))
    # Float32 input and output tensors only. Optimizer states and activations are
    # represented by a conservative multiplier for feasibility checks.
    tensor_bytes = batch * values.shape[1] * (context + horizon) * values.shape[2] * 4 * 3
    report = {
        "status": "READY_FOR_ONE_GPU_BATCH",
        "task": "past 60 daily observations -> next 1 day E/N/U residual forecast",
        "data_shape": list(values.shape),
        "observed_fraction": float(mask.mean()),
        "windows": windows,
        "adaptive_model_parameters": int(sum(p.numel() for p in model.parameters())),
        "cpu_smoke_output_shapes": {"reconstruction": list(reconstruction.shape), "forecast": list(forecast.shape)},
        "batch_size": batch,
        "conservative_model_tensor_memory_mb": round(tensor_bytes * 12 / 1024**2, 1),
        "gpu_requirement": "RTX 4090 24 GB is more than sufficient; expected training time is about 1-3 minutes per 50-epoch configuration based on completed fixed-graph runs.",
        "required_paid_batch": ["evaluate completed single-station baseline", "train/evaluate fixed graph final seed", "train/evaluate adaptive graph", "train/evaluate 7-day adaptive graph", "run ablation with adaptive graph disabled"],
        "stop_rule": "Do not continue to additional seeds or variants unless adaptive graph improves held-out RMSE over persistence and fixed graph.",
    }
    destination = ROOT / "reports" / "M5_GPU_preflight_v1.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
