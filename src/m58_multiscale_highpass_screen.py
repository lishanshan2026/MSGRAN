"""Validation-only screen for multi-scale high-pass residual graph fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma, component_correlations
from m7_train_multiscale_residual_graph import correlation_distance_graph, masked_huber
from m22_train_strong_baseline import build_model
from m56_component_gate_screen import COMPONENTS, metric_block


def signed_correlation_distance_graph(high_raw: np.ndarray, observed: np.ndarray, split: np.ndarray, neighbors: int, station_indices: np.ndarray | None = None, correlation_weight: float = 0.75) -> torch.Tensor:
    """Build train-only positive and negative correlation-prior graphs.

    The original hybrid graph ranks neighbours by absolute training correlation.
    This variant keeps the same candidate rule but separates positively and
    negatively correlated edges into two independently normalized channels.
    """
    if not 0.0 <= correlation_weight <= 1.0:
        raise ValueError("correlation_weight must lie in [0, 1]")
    corr = component_correlations(high_raw[split == 0], observed[split == 0])
    distance = np.load(ROOT / "data_processed" / "ngl_cascadia_90stations_knn8_graph_v1.npz")
    total_stations = int(distance["edge_index"].max()) + 1
    distance_graph = np.zeros((total_stations, total_stations), dtype=np.float32)
    distance_graph[distance["edge_index"][0], distance["edge_index"][1]] = distance["edge_weight"]
    distance_graph = np.maximum(distance_graph, distance_graph.T)
    if station_indices is not None:
        distance_graph = distance_graph[np.ix_(station_indices, station_indices)]

    positive = np.zeros((corr.shape[0], corr.shape[0]), dtype=np.float32)
    negative = np.zeros((corr.shape[0], corr.shape[0]), dtype=np.float32)
    for station in range(corr.shape[0]):
        scores = np.nan_to_num(np.abs(corr[station]), nan=-1.0)
        choices = np.argsort(scores)[::-1]
        choices = [int(index) for index in choices if index != station][:neighbors]
        for other in choices:
            signed_correlation = float(np.nan_to_num(corr[station, other], nan=0.0))
            edge_weight = correlation_weight * abs(signed_correlation) + (1.0 - correlation_weight) * float(distance_graph[station, other])
            if signed_correlation >= 0:
                positive[station, other] = edge_weight
            else:
                negative[station, other] = edge_weight

    positive = np.maximum(positive, positive.T)
    negative = np.maximum(negative, negative.T)
    for graph in (positive, negative):
        row_sum = graph.sum(axis=1, keepdims=True)
        np.divide(graph, row_sum, out=graph, where=row_sum > 0)
    return torch.from_numpy(np.stack((positive, negative), axis=0).astype(np.float32))


class MultiScaleAdapterWindows(Dataset):
    def __init__(self, values: np.ndarray, high: np.ndarray, mask: np.ndarray, split: np.ndarray, split_id: int, context: int, horizon: int, max_windows: int = 0):
        self.values, self.high, self.mask = values, high, mask
        self.context, self.horizon = context, horizon
        total = context + horizon
        self.starts = [index for index in range(len(split) - total + 1) if np.all(split[index:index + total] == split_id)]
        if max_windows:
            self.starts = self.starts[:max_windows]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int):
        start, pivot = self.starts[item], self.starts[item] + self.context
        arrays = (
            self.values[start:pivot],
            self.high[start:pivot],
            self.mask[start:pivot],
            self.values[pivot:pivot + self.horizon],
            self.mask[pivot:pivot + self.horizon],
        )
        return tuple(torch.from_numpy(array) for array in arrays)


class MultiScaleHighPassAdapter(nn.Module):
    def __init__(self, base: nn.Module, nodes: int, hidden: int, horizon: int, scales: int, gate_mode: str = "global", aggregation_mode: str = "fixed_prior"):
        super().__init__()
        self.base, self.nodes, self.horizon, self.scales = base, nodes, horizon, scales
        if gate_mode not in {"global", "component", "horizon_component", "dynamic_component"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if aggregation_mode not in {"fixed_prior", "signed_prior"}:
            raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
        self.gate_mode = gate_mode
        self.aggregation_mode = aggregation_mode
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.high_encoder = nn.GRU(3 * scales + 3, hidden, batch_first=True)
        correction_input = hidden * 2 if aggregation_mode == "signed_prior" else hidden
        self.graph_correction = nn.Sequential(
            nn.Linear(correction_input, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, horizon * 3, bias=False),
        )
        if gate_mode == "global":
            gate_shape: tuple[int, ...] = ()
        elif gate_mode == "component":
            gate_shape = (1, 1, 1, 3)
        elif gate_mode == "horizon_component":
            gate_shape = (1, horizon, 1, 3)
        else:
            gate_shape = (1, 1, 1, 3)
            gate_hidden = max(8, hidden // 2)
            self.dynamic_gate = nn.Sequential(
                nn.Linear(hidden * 3, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, horizon * 3),
            )
            nn.init.zeros_(self.dynamic_gate[-1].weight)
            nn.init.zeros_(self.dynamic_gate[-1].bias)
        self.graph_gate_logit = nn.Parameter(torch.full(gate_shape, -2.0))
        nn.init.zeros_(self.graph_correction[-1].weight)

    def encode_high(self, high: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, scales, components = high.shape
        high_features = torch.nan_to_num(high, nan=0.0).clamp(-10, 10).reshape(batch, steps, nodes, scales * components)
        mask_features = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        x = torch.cat((high_features, mask_features), dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 3 * scales + 3)
        return self.high_encoder(x)[1][-1].reshape(batch, nodes, -1)

    def forward(self, values: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor):
        with torch.no_grad():
            base_prediction = self.base(values, mask, None)
        own = self.encode_high(high, mask)
        if self.aggregation_mode == "signed_prior":
            positive_graph, negative_graph = graph[0], graph[1]
            positive_neighbor = torch.bmm(positive_graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
            negative_neighbor = torch.bmm(negative_graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
            # A missing signed neighbourhood must contribute an exact zero
            # contrast.  Without this mask, an empty channel produces
            # 0 - own = -own and silently becomes an extra station-local
            # pathway rather than evidence from signed neighbours.
            positive_support = (positive_graph.abs().sum(dim=-1, keepdim=True) > 0).to(own.dtype)
            negative_support = (negative_graph.abs().sum(dim=-1, keepdim=True) > 0).to(own.dtype)
            positive_contrast = (positive_neighbor - own) * positive_support.unsqueeze(0)
            negative_contrast = (negative_neighbor - own) * negative_support.unsqueeze(0)
            graph_features = torch.cat((positive_contrast, negative_contrast), dim=-1)
            neighbor = positive_neighbor - negative_neighbor
        else:
            neighbor = torch.bmm(graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
            graph_features = neighbor - own
        correction = self.graph_correction(graph_features).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        if self.gate_mode == "dynamic_component":
            gate_features = torch.cat((own, neighbor, neighbor - own), dim=-1)
            dynamic_logit = self.dynamic_gate(gate_features).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
            gate = torch.sigmoid(self.graph_gate_logit + dynamic_logit)
        else:
            gate = torch.sigmoid(self.graph_gate_logit)
        return base_prediction + gate * correction, gate


def gate_to_jsonable(gate: torch.Tensor | None) -> float | list[float] | None:
    if gate is None:
        return None
    array = gate.detach().cpu().numpy()
    if array.shape == ():
        return float(array)
    return array.reshape(-1).astype(float).tolist()


def run_epoch(model: nn.Module, loader: DataLoader, graph: torch.Tensor, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    model.base.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for values, high, mask, future, future_mask in loader:
            values, high, mask, future, future_mask = (item.to(device) for item in (values, high, mask, future, future_mask))
            prediction, _ = model(values, high, mask, graph)
            loss = masked_huber(prediction, future, future_mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def evaluate_validation(model: MultiScaleHighPassAdapter, base: nn.Module, dataset: MultiScaleAdapterWindows, graph: torch.Tensor, raw: np.ndarray, scale: np.ndarray, center: np.ndarray, device: torch.device) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], float | list[float] | None]:
    model.eval()
    base.eval()
    predictions, base_predictions, truths, masks = [], [], [], []
    cursor, last_gate = 0, None
    with torch.no_grad():
        for value_x, high_x, context_mask, _, future_mask in DataLoader(dataset, batch_size=64):
            values_device, high_device, mask_device = value_x.to(device), high_x.to(device), context_mask.to(device)
            prediction_norm, last_gate = model(values_device, high_device, mask_device, graph)
            base_norm = base(values_device, mask_device, None)
            starts = dataset.starts[cursor:cursor + value_x.shape[0]]
            cursor += value_x.shape[0]
            predictions.append(prediction_norm.cpu().numpy())
            base_predictions.append(base_norm.cpu().numpy())
            truths.append(np.stack([raw[start + dataset.context:start + dataset.context + dataset.horizon] for start in starts]))
            masks.append(future_mask.numpy())
    pred_norm = np.concatenate(predictions)
    base_norm = np.concatenate(base_predictions)
    truth = np.concatenate(truths)
    valid = np.concatenate(masks)
    prediction_mm = pred_norm * scale[None, None] + center[None, None]
    base_mm = base_norm * scale[None, None] + center[None, None]
    return metric_block(prediction_mm, truth, valid), metric_block(base_mm, truth, valid), gate_to_jsonable(last_gate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--spans", default="7,15,31,61")
    parser.add_argument("--graph-span", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--gate-mode", choices=("global", "component", "horizon_component", "dynamic_component"), default="global")
    parser.add_argument("--aggregation-mode", choices=("fixed_prior", "signed_prior"), default="fixed_prior")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--correlation-weight", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    spans = [int(item) for item in args.spans.split(",") if item.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    base_checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base_config = base_checkpoint["config"]
    context, horizon = int(base_config["context"]), int(base_config["horizon"])
    source = np.load(DATA)
    raw = source["residual_enu_mm"].astype(np.float32)
    mask = source["observed_mask"].astype(bool)
    split = source["split"]
    station_indices = np.array(base_config.get("station_indices", list(range(raw.shape[1]))), dtype=int)
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    values = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    values[~mask] = np.nan
    scale = source["train_robust_scale_mm"][station_indices]
    center = source["train_center_mm"][station_indices]
    high_arrays = []
    graph_high_raw = None
    for span in spans:
        _, high_raw = causal_ewma(raw, mask, span_days=span)
        if span == args.graph_span:
            graph_high_raw = high_raw
        high = (high_raw / scale).astype(np.float32)
        high[~mask] = np.nan
        high_arrays.append(high)
    if graph_high_raw is None:
        _, graph_high_raw = causal_ewma(raw, mask, span_days=args.graph_span)
    high_stack = np.stack(high_arrays, axis=2).astype(np.float32)

    train = MultiScaleAdapterWindows(values, high_stack, mask, split, 0, context, horizon, args.max_windows)
    validation = MultiScaleAdapterWindows(values, high_stack, mask, split, 1, context, horizon, args.max_windows)
    if args.aggregation_mode == "signed_prior":
        graph = signed_correlation_distance_graph(graph_high_raw, mask, split, args.neighbors, station_indices, args.correlation_weight).to(device)
    else:
        graph = correlation_distance_graph(graph_high_raw, mask, split, args.neighbors, station_indices, "hybrid", args.correlation_weight).to(device)
    base = build_model(base_config, raw.shape[1]).to(device)
    base.load_state_dict(base_checkpoint["state_dict"])
    model = MultiScaleHighPassAdapter(base, raw.shape[1], args.hidden, horizon, len(spans), args.gate_mode, args.aggregation_mode).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3, weight_decay=1e-4)

    out = ROOT / "experiments" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    best_loss = run_epoch(model, validation_loader, graph, None, device)
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, graph, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, graph, None, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gate = gate_to_jsonable(torch.sigmoid(model.graph_gate_logit))
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "graph_gate": gate, "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    model_metrics, base_metrics, final_gate = evaluate_validation(model, base, validation, graph, raw, scale, center, device)
    gain = 1.0 - model_metrics["overall"]["rmse_mm"] / base_metrics["overall"]["rmse_mm"]
    payload = {
        "device": str(device),
        "model": "multiscale_highpass_graph_adapter",
        "gate_mode": args.gate_mode,
        "aggregation_mode": args.aggregation_mode,
        "seed": args.seed,
        "spans": spans,
        "graph_span": args.graph_span,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "best_validation_loss": float(best_loss),
        "validation_model": model_metrics,
        "validation_base_nlinear": base_metrics,
        "validation_gain_vs_nlinear": float(gain),
        "graph_gate": final_gate,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "config": vars(args) | {"base_config": base_config, "station_indices": station_indices.tolist(), "context": context, "horizon": horizon, "spans": spans, "gate_mode": args.gate_mode, "aggregation_mode": args.aggregation_mode}, "best_validation_loss": best_loss}, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
