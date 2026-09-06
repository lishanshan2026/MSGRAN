"""Train a high-frequency graph residual adapter on a frozen linear backbone."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma
from m7_train_multiscale_residual_graph import correlation_distance_graph, masked_huber
from m22_train_strong_baseline import build_model


class AdapterWindows(Dataset):
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
            self.values[start:pivot], self.high[start:pivot], self.mask[start:pivot],
            self.values[pivot:pivot + self.horizon], self.mask[pivot:pivot + self.horizon],
        )
        return tuple(torch.from_numpy(array) for array in arrays)


class DLinearGraphAdapter(nn.Module):
    def __init__(self, base: nn.Module, nodes: int, hidden: int, horizon: int):
        super().__init__()
        self.base, self.nodes, self.horizon = base, nodes, horizon
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.high_encoder = nn.GRU(6, hidden, batch_first=True)
        # Bias-free neighbor-minus-own input makes the adapter identically zero for
        # an identity graph; any gain is therefore attributable to cross-station signal.
        self.graph_correction = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.GELU(), nn.Linear(hidden, horizon * 3, bias=False))
        self.graph_gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.graph_correction[-1].weight)

    def encode_high(self, high: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = high.shape
        x = torch.cat((torch.nan_to_num(high, nan=0.0).clamp(-10, 10), mask.unsqueeze(-1).expand(-1, -1, -1, 3)), dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 6)
        return self.high_encoder(x)[1][-1].reshape(batch, nodes, -1)

    def forward(self, values: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor):
        with torch.no_grad():
            base_prediction = self.base(values, mask, None)
        own = self.encode_high(high, mask)
        neighbor = torch.bmm(graph.unsqueeze(0).expand(values.shape[0], -1, -1), own)
        correction = self.graph_correction(neighbor - own).view(-1, self.nodes, self.horizon, 3).permute(0, 2, 1, 3)
        gate = torch.sigmoid(self.graph_gate_logit)
        return base_prediction + gate * correction, gate


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
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=48); parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--ewma-span", type=int, default=31); parser.add_argument("--correlation-weight", type=float, default=0.75)
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=20260803); parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    base_checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base_config = base_checkpoint["config"]
    if str(base_config["model"]).lower() not in ("dlinear", "nlinear"):
        raise ValueError("base checkpoint must be DLinear or NLinear")
    context, horizon = int(base_config["context"]), int(base_config["horizon"])
    source = np.load(DATA)
    raw, mask, split = source["residual_enu_mm"].astype(np.float32), source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array(base_config.get("station_indices", list(range(raw.shape[1]))), dtype=int)
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    values = source["normalized_residual_enu"][:, station_indices].astype(np.float32); values[~mask] = np.nan
    _, high_raw = causal_ewma(raw, mask, span_days=args.ewma_span)
    scale = source["train_robust_scale_mm"][station_indices]
    high = (high_raw / scale).astype(np.float32); high[~mask] = np.nan
    train = AdapterWindows(values, high, mask, split, 0, context, horizon, args.max_windows)
    validation = AdapterWindows(values, high, mask, split, 1, context, horizon, args.max_windows)
    graph = correlation_distance_graph(high_raw, mask, split, args.neighbors, station_indices, "hybrid", args.correlation_weight).to(device)

    base = build_model(base_config, raw.shape[1]).to(device); base.load_state_dict(base_checkpoint["state_dict"])
    model = DLinearGraphAdapter(base, raw.shape[1], args.hidden, horizon).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3, weight_decay=1e-4)
    out = ROOT / "experiments" / args.run_name; out.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    best = run_epoch(model, validation_loader, graph, None, device)
    config = vars(args) | {"base_config": base_config, "station_indices": station_indices.tolist(), "context": context, "horizon": horizon}
    torch.save({"state_dict": model.state_dict(), "config": config, "best_validation_loss": best}, out / "best.pt")
    history = [{"epoch": 0, "train_loss": None, "validation_loss": best, "graph_gate": float(torch.sigmoid(model.graph_gate_logit).detach().cpu()), "epoch_seconds": 0.0}]
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, graph, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, graph, None, device)
        if device.type == "cuda": torch.cuda.synchronize(device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "graph_gate": float(torch.sigmoid(model.graph_gate_logit).detach().cpu()), "epoch_seconds": time.perf_counter() - started}
        history.append(record)
        if validation_loss < best:
            best = validation_loss
            torch.save({"state_dict": model.state_dict(), "config": config, "best_validation_loss": best}, out / "best.pt")
        print(json.dumps(record), flush=True)
    payload = {"device": str(device), "parameter_count": int(sum(p.numel() for p in model.parameters())), "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "train_windows": len(train), "validation_windows": len(validation), "best_validation_loss": best, "history": history}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
