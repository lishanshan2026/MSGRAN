"""M7 CPU/GPU trainable multiscale GNSS forecaster.

The station's own temporal state is always retained.  A train-only, correlation plus
distance graph supplies a *residual correction* to the fast component instead of
replacing the own-station forecast by a spatial average.
"""

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


class Windows(Dataset):
    def __init__(self, low: np.ndarray, high: np.ndarray, target: np.ndarray, mask: np.ndarray, split: np.ndarray, split_id: int, context: int, horizon: int, max_windows: int = 0):
        self.low, self.high, self.target, self.mask = low, high, target, mask
        total = context + horizon
        self.context, self.horizon = context, horizon
        self.starts = [i for i in range(len(split) - total + 1) if np.all(split[i : i + total] == split_id)]
        if max_windows:
            self.starts = self.starts[:max_windows]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start, pivot = self.starts[index], self.starts[index] + self.context
        return tuple(torch.from_numpy(x) for x in (
            self.low[start:pivot], self.high[start:pivot], self.mask[start:pivot],
            self.low[pivot:pivot + self.horizon], self.high[pivot:pivot + self.horizon],
            self.target[pivot:pivot + self.horizon], self.mask[pivot:pivot + self.horizon],
        ))


def correlation_distance_graph(high_raw: np.ndarray, observed: np.ndarray, split: np.ndarray, neighbors: int, station_indices: np.ndarray | None = None, graph_mode: str = "hybrid", correlation_weight: float = 0.75) -> torch.Tensor:
    """Build a train-only hybrid, correlation-only, or distance-only graph."""
    if graph_mode not in {"hybrid", "correlation", "distance"}:
        raise ValueError(f"unsupported graph_mode={graph_mode}")
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
    graph = np.eye(corr.shape[0], dtype=np.float32)
    for station in range(corr.shape[0]):
        if graph_mode == "distance":
            choices = np.argsort(distance_graph[station])[::-1]
            choices = [int(i) for i in choices if i != station and distance_graph[station, i] > 0][:neighbors]
        else:
            choices = np.argsort(np.nan_to_num(np.abs(corr[station]), nan=-1.0))[::-1]
            choices = [int(i) for i in choices if i != station][:neighbors]
        for other in choices:
            edge_correlation = float(np.nan_to_num(abs(corr[station, other]), nan=0.0))
            if graph_mode == "hybrid":
                # Correlation supplies the principal edge weight; distance only regularizes it.
                graph[station, other] = correlation_weight * edge_correlation + (1.0 - correlation_weight) * distance_graph[station, other]
            elif graph_mode == "correlation":
                graph[station, other] = edge_correlation
            else:
                graph[station, other] = distance_graph[station, other]
    graph = np.maximum(graph, graph.T)
    graph /= graph.sum(axis=1, keepdims=True)
    return torch.from_numpy(graph.astype(np.float32))


class MultiScaleResidualGraph(nn.Module):
    def __init__(self, nodes: int, hidden: int, context: int, horizon: int, use_graph_correction: bool = True, gate_mode: str = "learned"):
        super().__init__()
        if gate_mode not in {"learned", "fixed_one", "fixed_initial"}:
            raise ValueError(f"unsupported gate_mode={gate_mode}")
        self.nodes, self.context, self.horizon, self.use_graph_correction, self.gate_mode = nodes, context, horizon, use_graph_correction, gate_mode
        self.low_encoder = nn.GRU(6, hidden, batch_first=True)
        self.high_encoder = nn.GRU(6, hidden, batch_first=True)
        self.low_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * 3))
        self.high_own_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * 3))
        self.graph_correction = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, horizon * 3))
        # Initialize with a small spatial-correction weight.
        self.graph_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def encode(self, values: torch.Tensor, mask: torch.Tensor, encoder: nn.GRU) -> torch.Tensor:
        batch, steps, nodes, _ = values.shape
        x = torch.cat((torch.nan_to_num(values, nan=0.0).clamp(-10, 10), mask.unsqueeze(-1).expand(-1, -1, -1, 3)), dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 6)
        return encoder(x)[1][-1].reshape(batch, nodes, -1)

    def forward(self, low: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor):
        low_state = self.encode(low, mask, self.low_encoder)
        high_state = self.encode(high, mask, self.high_encoder)
        low_delta = self.low_head(low_state).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        own_delta = self.high_own_head(high_state).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        neighbor_state = torch.bmm(graph.unsqueeze(0).expand(low.shape[0], -1, -1), high_state)
        correction = self.graph_correction(torch.cat((high_state, neighbor_state), dim=-1)).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        if not self.use_graph_correction:
            gate = self.graph_gate_logit * 0.0
        elif self.gate_mode == "fixed_one":
            gate = self.graph_gate_logit * 0.0 + 1.0
        elif self.gate_mode == "fixed_initial":
            gate = self.graph_gate_logit * 0.0 + float(torch.sigmoid(torch.tensor(-2.0)))
        else:
            gate = torch.sigmoid(self.graph_gate_logit)
        low_prediction = torch.nan_to_num(low[:, -1], nan=0.0)[:, None] + low_delta
        high_prediction = torch.nan_to_num(high[:, -1], nan=0.0)[:, None] + own_delta + gate * correction
        return low_prediction, high_prediction, gate


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    use = mask.unsqueeze(-1).expand_as(prediction)
    loss = torch.nn.functional.huber_loss(prediction, torch.nan_to_num(target, nan=0.0).clamp(-10, 10), reduction="none", delta=1.5)
    return loss[use].mean()


def run_epoch(model, loader, graph, optimizer, device):
    training = optimizer is not None
    model.train(training); total = 0.0
    with torch.set_grad_enabled(training):
        for low, high, context_mask, low_future, high_future, target, future_mask in loader:
            low, high, context_mask, low_future, high_future, target, future_mask = (x.to(device) for x in (low, high, context_mask, low_future, high_future, target, future_mask))
            low_pred, high_pred, _ = model(low, high, context_mask, graph)
            # Component losses make the intended split identifiable; final loss keeps output physically consistent.
            loss = masked_huber(low_pred, low_future, future_mask) + masked_huber(high_pred, high_future, future_mask) + masked_huber(low_pred + high_pred, target, future_mask)
            if training:
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48); parser.add_argument("--context", type=int, default=60); parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=8); parser.add_argument("--device", default="auto"); parser.add_argument("--run-name", default="M7_multiscale_residual_graph_v1")
    parser.add_argument("--ewma-span", type=int, default=31, help="Causal EWMA span in days.")
    parser.add_argument("--correlation-weight", type=float, default=0.75, help="Hybrid-graph correlation weight; distance receives 1-weight.")
    parser.add_argument("--graph-mode", choices=("hybrid", "correlation", "distance"), default="hybrid")
    parser.add_argument("--gate-mode", choices=("learned", "fixed_one", "fixed_initial"), default="learned")
    parser.add_argument("--max-windows", type=int, default=0, help="Use only this many chronological windows per split for a smoke test.")
    parser.add_argument("--disable-graph-correction", action="store_true", help="Ablation: retain the multiscale temporal model but set spatial residual correction to zero.")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--station-indices", default="", help="Comma-separated original station indices for a fixed subnet experiment.")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source = np.load(DATA); raw, mask, split = source["residual_enu_mm"].astype(np.float32), source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array([int(value) for value in args.station_indices.split(",") if value.strip()], dtype=int) if args.station_indices else np.arange(raw.shape[1])
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    args.station_indices = station_indices.tolist()
    low_raw, high_raw = causal_ewma(raw, mask, span_days=args.ewma_span)
    center, scale = source["train_center_mm"][station_indices], source["train_robust_scale_mm"][station_indices]
    low = ((low_raw - center) / scale).astype(np.float32); high = (high_raw / scale).astype(np.float32); target = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    low[~mask], high[~mask] = np.nan, np.nan
    train = Windows(low, high, target, mask, split, 0, args.context, args.horizon, args.max_windows); validation = Windows(low, high, target, mask, split, 1, args.context, args.horizon, args.max_windows)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    graph = correlation_distance_graph(high_raw, mask, split, args.neighbors, station_indices, args.graph_mode, args.correlation_weight).to(device)
    model = MultiScaleResidualGraph(raw.shape[1], args.hidden, args.context, args.horizon, use_graph_correction=not args.disable_graph_correction, gate_mode=args.gate_mode).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    out = ROOT / "experiments" / args.run_name; out.mkdir(parents=True, exist_ok=True); history, best = [], float("inf")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = run_epoch(model, DataLoader(train, batch_size=args.batch_size, shuffle=True), graph, optimizer, device)
        validation_loss = run_epoch(model, DataLoader(validation, batch_size=args.batch_size), graph, None, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "graph_gate": float(torch.sigmoid(model.graph_gate_logit).detach().cpu()), "epoch_seconds": time.perf_counter() - epoch_started}; history.append(record)
        if validation_loss < best:
            best = validation_loss; torch.save({"state_dict": model.state_dict(), "config": vars(args), "best_validation_loss": best}, out / "best.pt")
        print(json.dumps(record))
    (out / "metrics.json").write_text(json.dumps({"device": str(device), "parameter_count": parameter_count, "train_windows": len(train), "validation_windows": len(validation), "best_validation_loss": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
