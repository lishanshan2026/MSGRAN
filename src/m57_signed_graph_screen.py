"""Validation-only screen for a signed positive/negative correlation graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma, component_correlations
from m7_train_multiscale_residual_graph import masked_huber
from m22_train_strong_baseline import build_model
from m56_component_gate_screen import AdapterWindows, COMPONENTS, metric_block


def signed_hybrid_graphs(high_raw: np.ndarray, observed: np.ndarray, split: np.ndarray, neighbors: int, station_indices: np.ndarray, correlation_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    corr = component_correlations(high_raw[split == 0], observed[split == 0])
    distance = np.load(ROOT / "data_processed" / "ngl_cascadia_90stations_knn8_graph_v1.npz")
    total_stations = int(distance["edge_index"].max()) + 1
    distance_graph = np.zeros((total_stations, total_stations), dtype=np.float32)
    distance_graph[distance["edge_index"][0], distance["edge_index"][1]] = distance["edge_weight"]
    distance_graph = np.maximum(distance_graph, distance_graph.T)
    distance_graph = distance_graph[np.ix_(station_indices, station_indices)]

    nodes = corr.shape[0]
    pos = np.zeros((nodes, nodes), dtype=np.float32)
    neg = np.zeros((nodes, nodes), dtype=np.float32)
    for station in range(nodes):
        order = np.argsort(np.nan_to_num(np.abs(corr[station]), nan=-1.0))[::-1]
        choices = [int(index) for index in order if index != station][:neighbors]
        for other in choices:
            signed_r = float(np.nan_to_num(corr[station, other], nan=0.0))
            weight = correlation_weight * abs(signed_r) + (1.0 - correlation_weight) * float(distance_graph[station, other])
            if signed_r >= 0:
                pos[station, other] = weight
            else:
                neg[station, other] = weight
    pos = np.maximum(pos, pos.T)
    neg = np.maximum(neg, neg.T)
    pos_sum = pos.sum(axis=1, keepdims=True)
    neg_sum = neg.sum(axis=1, keepdims=True)
    pos_mask = (pos_sum[:, 0] > 0).astype(np.float32)
    neg_mask = (neg_sum[:, 0] > 0).astype(np.float32)
    pos = np.divide(pos, pos_sum, out=np.zeros_like(pos), where=pos_sum > 0)
    neg = np.divide(neg, neg_sum, out=np.zeros_like(neg), where=neg_sum > 0)
    return (
        torch.from_numpy(pos.astype(np.float32)),
        torch.from_numpy(neg.astype(np.float32)),
        torch.from_numpy(pos_mask),
        torch.from_numpy(neg_mask),
    )


class SignedGraphAdapter(nn.Module):
    def __init__(self, base: nn.Module, nodes: int, hidden: int, horizon: int):
        super().__init__()
        self.base, self.nodes, self.horizon = base, nodes, horizon
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.high_encoder = nn.GRU(6, hidden, batch_first=True)
        self.graph_correction = nn.Sequential(
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, horizon * 3, bias=False),
        )
        self.graph_gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.graph_correction[-1].weight)

    def encode_high(self, high: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = high.shape
        x = torch.cat(
            (
                torch.nan_to_num(high, nan=0.0).clamp(-10, 10),
                mask.unsqueeze(-1).expand(-1, -1, -1, 3),
            ),
            dim=-1,
        )
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 6)
        return self.high_encoder(x)[1][-1].reshape(batch, nodes, -1)

    def forward(self, values: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, pos_graph: torch.Tensor, neg_graph: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
        with torch.no_grad():
            base_prediction = self.base(values, mask, None)
        own = self.encode_high(high, mask)
        pos_neighbor = torch.bmm(pos_graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
        neg_neighbor = torch.bmm(neg_graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
        pos_delta = (pos_neighbor - own) * pos_mask.view(1, -1, 1)
        neg_delta = (neg_neighbor - own) * neg_mask.view(1, -1, 1)
        correction = self.graph_correction(torch.cat((pos_delta, neg_delta), dim=-1)).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        gate = torch.sigmoid(self.graph_gate_logit)
        return base_prediction + gate * correction, gate


def run_epoch(model: nn.Module, loader: DataLoader, graph_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    model.base.eval()
    total = 0.0
    pos_graph, neg_graph, pos_mask, neg_mask = graph_pack
    with torch.set_grad_enabled(training):
        for values, high, mask, future, future_mask in loader:
            values, high, mask, future, future_mask = (item.to(device) for item in (values, high, mask, future, future_mask))
            prediction, _ = model(values, high, mask, pos_graph, neg_graph, pos_mask, neg_mask)
            loss = masked_huber(prediction, future, future_mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def evaluate_validation(model: SignedGraphAdapter, base: nn.Module, dataset: AdapterWindows, graph_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], raw: np.ndarray, scale: np.ndarray, center: np.ndarray, device: torch.device) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], float]:
    model.eval()
    base.eval()
    pos_graph, neg_graph, pos_mask, neg_mask = graph_pack
    predictions, base_predictions, truths, masks = [], [], [], []
    cursor, last_gate = 0, None
    with torch.no_grad():
        for value_x, high_x, context_mask, _, future_mask in DataLoader(dataset, batch_size=64):
            values_device, high_device, mask_device = value_x.to(device), high_x.to(device), context_mask.to(device)
            prediction_norm, last_gate = model(values_device, high_device, mask_device, pos_graph, neg_graph, pos_mask, neg_mask)
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
    return metric_block(prediction_mm, truth, valid), metric_block(base_mm, truth, valid), float(last_gate.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--ewma-span", type=int, default=31)
    parser.add_argument("--correlation-weight", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
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
    _, high_raw = causal_ewma(raw, mask, span_days=args.ewma_span)
    scale = source["train_robust_scale_mm"][station_indices]
    center = source["train_center_mm"][station_indices]
    high = (high_raw / scale).astype(np.float32)
    high[~mask] = np.nan

    train = AdapterWindows(values, high, mask, split, 0, context, horizon, args.max_windows)
    validation = AdapterWindows(values, high, mask, split, 1, context, horizon, args.max_windows)
    graph_pack = tuple(item.to(device) for item in signed_hybrid_graphs(high_raw, mask, split, args.neighbors, station_indices, args.correlation_weight))
    base = build_model(base_config, raw.shape[1]).to(device)
    base.load_state_dict(base_checkpoint["state_dict"])
    model = SignedGraphAdapter(base, raw.shape[1], args.hidden, horizon).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3, weight_decay=1e-4)

    out = ROOT / "experiments" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    best_loss = run_epoch(model, validation_loader, graph_pack, None, device)
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, graph_pack, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, graph_pack, None, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        gate = float(torch.sigmoid(model.graph_gate_logit).detach().cpu())
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "graph_gate": gate, "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    model_metrics, base_metrics, final_gate = evaluate_validation(model, base, validation, graph_pack, raw, scale, center, device)
    gain = 1.0 - model_metrics["overall"]["rmse_mm"] / base_metrics["overall"]["rmse_mm"]
    pos_graph, neg_graph, pos_mask, neg_mask = graph_pack
    payload = {
        "device": str(device),
        "model": "signed_positive_negative_graph_adapter",
        "seed": args.seed,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "best_validation_loss": float(best_loss),
        "validation_model": model_metrics,
        "validation_base_nlinear": base_metrics,
        "validation_gain_vs_nlinear": float(gain),
        "graph_gate": final_gate,
        "positive_neighbor_stations": int(pos_mask.detach().cpu().sum()),
        "negative_neighbor_stations": int(neg_mask.detach().cpu().sum()),
        "positive_edges": int((pos_graph.detach().cpu().numpy() > 0).sum()),
        "negative_edges": int((neg_graph.detach().cpu().numpy() > 0).sum()),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "config": vars(args) | {"base_config": base_config, "station_indices": station_indices.tolist(), "context": context, "horizon": horizon}, "best_validation_loss": best_loss}, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
