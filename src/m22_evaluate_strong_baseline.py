"""Evaluate an M22 strong baseline on the frozen 2022--2024 test period."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from m6_multiscale_correlation_diagnostic import DATA, causal_ewma
from m7_evaluate_multiscale_residual_graph import metrics
from m7_train_multiscale_residual_graph import correlation_distance_graph
from m22_train_strong_baseline import DirectWindows, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-forecasts", action="store_true")
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    source = np.load(DATA)
    raw = source["residual_enu_mm"].astype(np.float32)
    mask, split = source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array(config.get("station_indices", list(range(raw.shape[1]))), dtype=int)
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    target = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    target[~mask] = np.nan
    context, horizon = int(config["context"]), int(config["horizon"])
    dataset = DirectWindows(target, mask, split, 2, context, horizon, args.max_windows)

    graph = None
    if str(config["model"]).lower() == "stgcn":
        _, high_raw = causal_ewma(raw, mask, span_days=int(config.get("ewma_span", 31)))
        graph = correlation_distance_graph(high_raw, mask, split, int(config.get("neighbors", 8)), station_indices, "hybrid", float(config.get("correlation_weight", 0.75))).to(device)
    model = build_model(config, raw.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions, truths, target_masks, persistence = [], [], [], []
    inference_seconds, inference_windows, cursor = 0.0, 0, 0
    warmed_up = False
    with torch.no_grad():
        for values, context_mask, _, future_mask in DataLoader(dataset, batch_size=64):
            values_device, mask_device = values.to(device), context_mask.to(device)
            if not warmed_up:
                model(values_device, mask_device, graph)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmed_up = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = model(values_device, mask_device, graph)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            inference_windows += int(values.shape[0])
            starts = dataset.starts[cursor:cursor + values.shape[0]]
            cursor += values.shape[0]
            predictions.append(prediction.cpu().numpy())
            truths.append(np.stack([raw[start + context:start + context + horizon] for start in starts]))
            target_masks.append(future_mask.numpy())
            persistence.append(np.stack([np.nan_to_num(raw[start + context - 1], nan=0.0) for start in starts])[:, None].repeat(horizon, axis=1))

    pred_norm = np.concatenate(predictions)
    truth, valid, persist = np.concatenate(truths), np.concatenate(target_masks), np.concatenate(persistence)
    center = source["train_center_mm"][station_indices]
    scale = source["train_robust_scale_mm"][station_indices]
    prediction_mm = pred_norm * scale[None, None] + center[None, None]
    model_metrics, persistence_metrics = metrics(prediction_mm, truth, valid), metrics(persist, truth, valid)
    payload = {
        "checkpoint": str(args.checkpoint),
        "model_name": str(config["model"]),
        "seed": int(config["seed"]),
        "frozen_test": "2022-2024",
        "model": model_metrics,
        "persistence": persistence_metrics,
        "overall_rmse_skill_vs_persistence": float(1.0 - model_metrics["overall"]["rmse_mm"] / persistence_metrics["overall"]["rmse_mm"]),
        "efficiency": {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters()) + sum(buffer.numel() for buffer in model.buffers())),
            "inference_seconds": inference_seconds,
            "forecast_windows": inference_windows,
            "milliseconds_per_window": 1000.0 * inference_seconds / max(1, inference_windows),
            "timed_batch_size": 64,
            "device": str(device),
        },
    }
    destination = args.checkpoint.parent
    (destination / "real_test_forecast_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_forecasts:
        np.savez_compressed(
            destination / "real_test_forecasts.npz",
            prediction_enu_mm=prediction_mm.astype(np.float32), truth_enu_mm=truth.astype(np.float32),
            observed_mask=valid, persistence_enu_mm=persist.astype(np.float32),
            station_codes=source["station_codes"][station_indices], latitude=source["latitude"][station_indices], longitude=source["longitude"][station_indices],
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
