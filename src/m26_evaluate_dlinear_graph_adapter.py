"""Evaluate a frozen linear-backbone graph-adapter checkpoint."""

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
from m22_train_strong_baseline import build_model
from m26_train_dlinear_graph_adapter import AdapterWindows, DLinearGraphAdapter


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--device", default="auto"); parser.add_argument("--save-forecasts", action="store_true"); parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); config = checkpoint["config"]; base_config = config["base_config"]
    source = np.load(DATA); raw, mask, split = source["residual_enu_mm"].astype(np.float32), source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array(config["station_indices"], dtype=int); raw, mask = raw[:, station_indices], mask[:, station_indices]
    values = source["normalized_residual_enu"][:, station_indices].astype(np.float32); values[~mask] = np.nan
    _, high_raw = causal_ewma(raw, mask, span_days=int(config["ewma_span"])); scale = source["train_robust_scale_mm"][station_indices]; center = source["train_center_mm"][station_indices]
    high = (high_raw / scale).astype(np.float32); high[~mask] = np.nan
    context, horizon = int(config["context"]), int(config["horizon"])
    dataset = AdapterWindows(values, high, mask, split, 2, context, horizon, args.max_windows)
    graph = correlation_distance_graph(high_raw, mask, split, int(config["neighbors"]), station_indices, "hybrid", float(config["correlation_weight"])).to(device)
    base = build_model(base_config, raw.shape[1]).to(device)
    model = DLinearGraphAdapter(base, raw.shape[1], int(config["hidden"]), horizon).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()

    predictions, truths, masks, persistence = [], [], [], []
    cursor, elapsed, windows, warmed = 0, 0.0, 0, False
    with torch.no_grad():
        for value_x, high_x, context_mask, _, future_mask in DataLoader(dataset, batch_size=64):
            value_device, high_device, mask_device = value_x.to(device), high_x.to(device), context_mask.to(device)
            if not warmed:
                model(value_device, high_device, mask_device, graph)
                if device.type == "cuda": torch.cuda.synchronize(device)
                warmed = True
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter(); prediction, gate = model(value_device, high_device, mask_device, graph)
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started; windows += int(value_x.shape[0])
            starts = dataset.starts[cursor:cursor + value_x.shape[0]]; cursor += value_x.shape[0]
            predictions.append(prediction.cpu().numpy()); truths.append(np.stack([raw[start + context:start + context + horizon] for start in starts])); masks.append(future_mask.numpy())
            persistence.append(np.stack([np.nan_to_num(raw[start + context - 1], nan=0.0) for start in starts])[:, None].repeat(horizon, axis=1))
    pred_norm, truth, valid, persist = np.concatenate(predictions), np.concatenate(truths), np.concatenate(masks), np.concatenate(persistence)
    prediction_mm = pred_norm * scale[None, None] + center[None, None]
    proposed, baseline = metrics(prediction_mm, truth, valid), metrics(persist, truth, valid)
    payload = {"checkpoint": str(args.checkpoint), "model_name": f"{base_config['model']}_graph_adapter", "seed": int(config["seed"]), "frozen_test": "2022-2024", "model": proposed, "persistence": baseline, "overall_rmse_skill_vs_persistence": float(1 - proposed["overall"]["rmse_mm"] / baseline["overall"]["rmse_mm"]), "graph_gate": float(gate.cpu()), "efficiency": {"parameter_count": int(sum(p.numel() for p in model.parameters())), "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "inference_seconds": elapsed, "forecast_windows": windows, "milliseconds_per_window": 1000 * elapsed / max(1, windows), "timed_batch_size": 64, "device": str(device)}}
    destination = args.checkpoint.parent; (destination / "real_test_forecast_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_forecasts:
        np.savez_compressed(destination / "real_test_forecasts.npz", prediction_enu_mm=prediction_mm.astype(np.float32), truth_enu_mm=truth.astype(np.float32), observed_mask=valid, persistence_enu_mm=persist.astype(np.float32), station_codes=source["station_codes"][station_indices], latitude=source["latitude"][station_indices], longitude=source["longitude"][station_indices])
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
