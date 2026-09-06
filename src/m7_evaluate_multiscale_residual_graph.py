"""Evaluate an M7 checkpoint on the frozen 2022-2024 GNSS test period."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma
from m7_train_multiscale_residual_graph import MultiScaleResidualGraph, Windows, correlation_distance_graph


def metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for component, name in enumerate(("east", "north", "up")):
        error = prediction[..., component][mask] - truth[..., component][mask]
        output[name] = {"mae_mm": float(np.mean(np.abs(error))), "rmse_mm": float(np.sqrt(np.mean(error**2))), "samples": int(error.size)}
    full_mask = np.broadcast_to(mask[..., None], prediction.shape)
    error = prediction[full_mask] - truth[full_mask]
    output["overall"] = {"mae_mm": float(np.mean(np.abs(error))), "rmse_mm": float(np.sqrt(np.mean(error**2))), "samples": int(error.size)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--device", default="auto"); parser.add_argument("--save-forecasts", action="store_true")
    args = parser.parse_args(); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); config = checkpoint["config"]
    source = np.load(DATA); raw, mask, split = source["residual_enu_mm"].astype(np.float32), source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array(config.get("station_indices", list(range(raw.shape[1]))), dtype=int)
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    low_raw, high_raw = causal_ewma(raw, mask, span_days=int(config.get("ewma_span", 31))); center, scale = source["train_center_mm"][station_indices], source["train_robust_scale_mm"][station_indices]
    low, high = ((low_raw - center) / scale).astype(np.float32), (high_raw / scale).astype(np.float32); target = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    low[~mask], high[~mask] = np.nan, np.nan
    dataset = Windows(low, high, target, mask, split, 2, int(config["context"]), int(config["horizon"]))
    graph = correlation_distance_graph(high_raw, mask, split, int(config["neighbors"]), station_indices, str(config.get("graph_mode", "hybrid")), float(config.get("correlation_weight", 0.75))).to(device)
    model = MultiScaleResidualGraph(raw.shape[1], int(config["hidden"]), int(config["context"]), int(config["horizon"]), use_graph_correction=not bool(config.get("disable_graph_correction", False)), gate_mode=str(config.get("gate_mode", "learned"))).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    predictions, truths, target_masks, persistence = [], [], [], []
    inference_seconds = 0.0
    inference_windows = 0
    warmed_up = False
    with torch.no_grad():
        for low_x, high_x, context_mask, _, _, _, future_mask in DataLoader(dataset, batch_size=64):
            low_device, high_device, mask_device = low_x.to(device), high_x.to(device), context_mask.to(device)
            if not warmed_up:
                model(low_device, high_device, mask_device, graph)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmed_up = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            low_pred, high_pred, _ = model(low_device, high_device, mask_device, graph)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            inference_windows += int(low_x.shape[0])
            starts = dataset.starts[len(predictions) * 64 : len(predictions) * 64 + low_x.shape[0]]
            predictions.append((low_pred + high_pred).cpu().numpy())
            truths.append(np.stack([raw[s + int(config["context"]) : s + int(config["context"]) + int(config["horizon"])] for s in starts]))
            target_masks.append(future_mask.numpy())
            persistence.append(np.stack([np.nan_to_num(raw[s + int(config["context"]) - 1], nan=0.0) for s in starts])[:, None].repeat(int(config["horizon"]), axis=1))
    pred_norm, truth, valid, persist = np.concatenate(predictions), np.concatenate(truths), np.concatenate(target_masks), np.concatenate(persistence)
    prediction_mm = pred_norm * scale[None, None] + center[None, None]
    proposed, baseline = metrics(prediction_mm, truth, valid), metrics(persist, truth, valid)
    output = {"checkpoint": str(args.checkpoint), "frozen_test": "2022-2024", "model": proposed, "persistence": baseline, "overall_rmse_skill_vs_persistence": float(1 - proposed["overall"]["rmse_mm"] / baseline["overall"]["rmse_mm"]), "efficiency": {"parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "inference_seconds": inference_seconds, "forecast_windows": inference_windows, "milliseconds_per_window": 1000.0 * inference_seconds / max(1, inference_windows), "timed_batch_size": 64, "device": str(device)}}
    destination = args.checkpoint.parent; (destination / "real_test_forecast_metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_forecasts:
        np.savez_compressed(destination / "real_test_forecasts.npz", prediction_enu_mm=prediction_mm.astype(np.float32), truth_enu_mm=truth.astype(np.float32), observed_mask=valid, persistence_enu_mm=persist.astype(np.float32), station_codes=source["station_codes"][station_indices], latitude=source["latitude"][station_indices], longitude=source["longitude"][station_indices])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
