"""Evaluate a multi-scale high-pass graph adapter checkpoint on the frozen test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from m6_multiscale_correlation_diagnostic import DATA, causal_ewma
from m7_train_multiscale_residual_graph import correlation_distance_graph
from m22_train_strong_baseline import build_model
from m56_component_gate_screen import metric_block
from m58_multiscale_highpass_screen import MultiScaleAdapterWindows, MultiScaleHighPassAdapter, gate_to_jsonable, signed_correlation_distance_graph


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
    base_config = config["base_config"]
    spans = [int(value) for value in config.get("spans", [7, 15, 31, 61])]
    graph_span = int(config.get("graph_span", 31))

    source = np.load(DATA)
    raw = source["residual_enu_mm"].astype(np.float32)
    mask = source["observed_mask"].astype(bool)
    split = source["split"]
    station_indices = np.asarray(config["station_indices"], dtype=int)
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    values = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    values[~mask] = np.nan
    scale = source["train_robust_scale_mm"][station_indices]
    center = source["train_center_mm"][station_indices]

    high_arrays = []
    graph_high_raw = None
    for span in spans:
        _, high_raw = causal_ewma(raw, mask, span_days=span)
        if span == graph_span:
            graph_high_raw = high_raw
        high = (high_raw / scale).astype(np.float32)
        high[~mask] = np.nan
        high_arrays.append(high)
    if graph_high_raw is None:
        _, graph_high_raw = causal_ewma(raw, mask, span_days=graph_span)
    high_stack = np.stack(high_arrays, axis=2).astype(np.float32)

    context, horizon = int(config["context"]), int(config["horizon"])
    dataset = MultiScaleAdapterWindows(values, high_stack, mask, split, 2, context, horizon, args.max_windows)
    aggregation_mode = str(config.get("aggregation_mode", "fixed_prior"))
    if aggregation_mode == "signed_prior":
        graph = signed_correlation_distance_graph(
            graph_high_raw,
            mask,
            split,
            int(config["neighbors"]),
            station_indices,
            float(config["correlation_weight"]),
        ).to(device)
    else:
        graph = correlation_distance_graph(
            graph_high_raw,
            mask,
            split,
            int(config["neighbors"]),
            station_indices,
            "hybrid",
            float(config["correlation_weight"]),
        ).to(device)

    base = build_model(base_config, raw.shape[1]).to(device)
    base.load_state_dict(torch.load(Path(config["base_checkpoint"]), map_location=device, weights_only=False)["state_dict"])
    gate_mode = str(config.get("gate_mode", "global"))
    model = MultiScaleHighPassAdapter(base, raw.shape[1], int(config["hidden"]), horizon, len(spans), gate_mode, aggregation_mode).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    base.eval()

    predictions, base_predictions, truths, masks, persistence = [], [], [], [], []
    cursor, elapsed, windows, warmed, last_gate = 0, 0.0, 0, False, None
    with torch.no_grad():
        for value_x, high_x, context_mask, _, future_mask in DataLoader(dataset, batch_size=64):
            value_device, high_device, mask_device = value_x.to(device), high_x.to(device), context_mask.to(device)
            if not warmed:
                model(value_device, high_device, mask_device, graph)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmed = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction_norm, last_gate = model(value_device, high_device, mask_device, graph)
            base_norm = base(value_device, mask_device, None)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            windows += int(value_x.shape[0])
            starts = dataset.starts[cursor:cursor + value_x.shape[0]]
            cursor += value_x.shape[0]
            predictions.append(prediction_norm.cpu().numpy())
            base_predictions.append(base_norm.cpu().numpy())
            truths.append(np.stack([raw[start + context:start + context + horizon] for start in starts]))
            masks.append(future_mask.numpy())
            persistence.append(np.stack([np.nan_to_num(raw[start + context - 1], nan=0.0) for start in starts])[:, None].repeat(horizon, axis=1))

    pred_norm = np.concatenate(predictions)
    base_norm = np.concatenate(base_predictions)
    truth = np.concatenate(truths)
    valid = np.concatenate(masks)
    persist = np.concatenate(persistence)
    prediction_mm = pred_norm * scale[None, None] + center[None, None]
    base_mm = base_norm * scale[None, None] + center[None, None]
    proposed = metric_block(prediction_mm, truth, valid)
    base_metrics = metric_block(base_mm, truth, valid)
    persistence_metrics = metric_block(persist, truth, valid)
    payload = {
        "checkpoint": str(args.checkpoint),
        "model_name": "nlinear_multiscale_highpass_graph_adapter",
        "gate_mode": gate_mode,
        "aggregation_mode": aggregation_mode,
        "seed": int(config["seed"]),
        "frozen_test": "2022-2024",
        "spans": spans,
        "graph_span": graph_span,
        "model": proposed,
        "base_nlinear": base_metrics,
        "persistence": persistence_metrics,
        "gain_vs_nlinear": float(1 - proposed["overall"]["rmse_mm"] / base_metrics["overall"]["rmse_mm"]),
        "overall_rmse_skill_vs_persistence": float(1 - proposed["overall"]["rmse_mm"] / persistence_metrics["overall"]["rmse_mm"]),
        "graph_gate": gate_to_jsonable(last_gate),
        "efficiency": {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "inference_seconds": elapsed,
            "forecast_windows": windows,
            "milliseconds_per_window": 1000 * elapsed / max(1, windows),
            "timed_batch_size": 64,
            "device": str(device),
        },
    }
    destination = args.checkpoint.parent
    (destination / "real_test_forecast_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_forecasts:
        np.savez_compressed(
            destination / "real_test_forecasts.npz",
            prediction_enu_mm=prediction_mm.astype(np.float32),
            base_nlinear_enu_mm=base_mm.astype(np.float32),
            truth_enu_mm=truth.astype(np.float32),
            observed_mask=valid,
            persistence_enu_mm=persist.astype(np.float32),
            station_codes=source["station_codes"][station_indices],
            latitude=source["latitude"][station_indices],
            longitude=source["longitude"][station_indices],
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
