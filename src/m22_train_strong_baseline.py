"""Train strong forecasting baselines under the frozen M7 GNSS protocol.

The four baselines share the exact chronological partitions, normalization, context,
forecast horizon, observation masks, validation checkpoint selection, and Huber loss.
STGCN also uses the same training-only graph as the proposed model so that its result
tests the spatial architecture rather than a different data split or graph source.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from m6_multiscale_correlation_diagnostic import DATA, ROOT, causal_ewma
from m7_train_multiscale_residual_graph import correlation_distance_graph, masked_huber


class DirectWindows(Dataset):
    def __init__(self, values: np.ndarray, mask: np.ndarray, split: np.ndarray, split_id: int, context: int, horizon: int, max_windows: int = 0):
        self.values, self.mask = values, mask
        self.context, self.horizon = context, horizon
        total = context + horizon
        self.starts = [index for index in range(len(split) - total + 1) if np.all(split[index:index + total] == split_id)]
        if max_windows:
            self.starts = self.starts[:max_windows]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int):
        start = self.starts[item]
        pivot = start + self.context
        arrays = (
            self.values[start:pivot], self.mask[start:pivot],
            self.values[pivot:pivot + self.horizon], self.mask[pivot:pivot + self.horizon],
        )
        return tuple(torch.from_numpy(array) for array in arrays)


class DLinear(nn.Module):
    """Channel-independent DLinear with moving-average decomposition."""

    def __init__(self, context: int, horizon: int, moving_average: int = 25):
        super().__init__()
        self.context, self.horizon, self.moving_average = context, horizon, moving_average
        self.seasonal = nn.Linear(context, horizon)
        self.trend = nn.Linear(context, horizon)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, components = values.shape
        series = torch.nan_to_num(values, nan=0.0).permute(0, 2, 3, 1).reshape(batch * nodes * components, 1, steps)
        kernel = min(self.moving_average, steps if steps % 2 else steps - 1)
        kernel = max(1, kernel)
        pad = (kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(series, (pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
        seasonal = series - trend
        forecast = self.seasonal(seasonal[:, 0]) + self.trend(trend[:, 0])
        return forecast.view(batch, nodes, components, self.horizon).permute(0, 3, 1, 2)


class NLinear(nn.Module):
    """Channel-independent NLinear baseline with last-value normalization."""

    def __init__(self, context: int, horizon: int):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.linear = nn.Linear(context, horizon)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, components = values.shape
        series = torch.nan_to_num(values, nan=0.0).permute(0, 2, 3, 1).reshape(batch * nodes * components, steps)
        last = series[:, -1:].detach()
        forecast = self.linear(series - last) + last
        return forecast.view(batch, nodes, components, self.horizon).permute(0, 3, 1, 2)


class RidgeAR(nn.Module):
    """Station-wise ridge autoregression fitted analytically on training windows."""

    def __init__(self, nodes: int, context: int, horizon: int):
        if horizon != 1:
            raise ValueError("RidgeAR currently supports one-day forecasting only")
        super().__init__()
        self.context, self.horizon = context, horizon
        self.register_buffer("coefficients", torch.zeros(nodes, 3, context + 1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        series = torch.nan_to_num(values, nan=0.0).permute(0, 2, 3, 1)
        intercept = torch.ones_like(series[..., :1])
        design = torch.cat((series, intercept), dim=-1)
        forecast = torch.einsum("bncf,ncf->bnc", design, self.coefficients)
        return forecast.unsqueeze(1)


def fit_ridge_ar(model: RidgeAR, windows: "DirectWindows", alpha: float) -> None:
    """Fit each station/component AR coefficient vector using training data only."""
    starts = np.asarray(windows.starts, dtype=int)
    offsets = np.arange(model.context)
    identity = np.eye(model.context + 1, dtype=np.float64)
    identity[-1, -1] = 0.0  # Do not penalize the intercept.
    coefficients = np.zeros((windows.values.shape[1], 3, model.context + 1), dtype=np.float32)
    for station in range(windows.values.shape[1]):
        x_all = windows.values[starts[:, None] + offsets[None, :], station]
        y_all = windows.values[starts + model.context, station]
        observed = windows.mask[starts[:, None] + offsets[None, :], station].all(axis=1) & windows.mask[starts + model.context, station]
        for component in range(3):
            valid = observed & np.isfinite(x_all[:, :, component]).all(axis=1) & np.isfinite(y_all[:, component])
            design = np.column_stack((x_all[valid, :, component], np.ones(valid.sum(), dtype=np.float32))).astype(np.float64)
            target = y_all[valid, component].astype(np.float64)
            if len(target) >= model.context + 2:
                coefficients[station, component] = np.linalg.solve(design.T @ design + alpha * identity, design.T @ target)
    model.coefficients.copy_(torch.from_numpy(coefficients))


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=self.padding)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def causal(self, layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return layer(x)[..., :-self.padding] if self.padding else layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.causal(self.conv1, x))))
        x = self.dropout(F.gelu(self.norm2(self.causal(self.conv2, x))))
        return x + residual


class TCN(nn.Module):
    def __init__(self, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.input_projection = nn.Conv1d(6, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*(CausalResidualBlock(hidden, dilation, dropout) for dilation in (1, 2, 4, 8)))
        self.head = nn.Linear(hidden, horizon * 3)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, _ = values.shape
        observed = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        x = torch.cat((torch.nan_to_num(values, nan=0.0).clamp(-10, 10), observed), dim=-1)
        x = x.permute(0, 2, 3, 1).reshape(batch * nodes, 6, steps)
        state = self.blocks(self.input_projection(x))[..., -1]
        return self.head(state).view(batch, nodes, self.horizon, 3).permute(0, 2, 1, 3)


class LSTM(nn.Module):
    """Mask-aware station-independent recurrent baseline."""

    def __init__(self, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(6, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon * 3)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, _ = values.shape
        observed = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        x = torch.cat((torch.nan_to_num(values, nan=0.0).clamp(-10, 10), observed), dim=-1)
        x = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 6)
        state = self.encoder(x)[1][0][-1]
        return self.head(state).view(batch, nodes, self.horizon, 3).permute(0, 2, 1, 3)


class PatchTST(nn.Module):
    """Mask-aware, channel-independent PatchTST baseline."""

    def __init__(self, context: int, horizon: int, patch_length: int, stride: int, d_model: int, heads: int, layers: int, dropout: float):
        super().__init__()
        if context < patch_length:
            raise ValueError("context must be at least patch_length")
        self.context, self.horizon = context, horizon
        self.patch_length, self.stride = patch_length, stride
        self.patch_count = 1 + (context - patch_length) // stride
        self.embedding = nn.Linear(2 * patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.patch_count * d_model, horizon)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, nodes, components = values.shape
        value_series = torch.nan_to_num(values, nan=0.0).clamp(-10, 10).permute(0, 2, 3, 1).reshape(batch * nodes * components, self.context)
        mask_series = mask.unsqueeze(2).expand(-1, -1, components, -1).permute(0, 3, 2, 1).reshape(batch * nodes * components, self.context)
        value_patches = value_series.unfold(-1, self.patch_length, self.stride)
        mask_patches = mask_series.unfold(-1, self.patch_length, self.stride)
        tokens = self.embedding(torch.cat((value_patches, mask_patches), dim=-1)) + self.position
        encoded = self.norm(self.encoder(tokens)).reshape(tokens.shape[0], -1)
        forecast = self.head(encoded)
        return forecast.view(batch, nodes, components, self.horizon).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    """Temporal-graph-temporal convolution using the frozen training-only adjacency."""

    def __init__(self, hidden: int, horizon: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.temporal_in = nn.Conv2d(6, hidden, kernel_size=(1, 3), padding=(0, 2))
        self.graph_projection = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.temporal_out = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 2))
        self.norm1 = nn.GroupNorm(1, hidden)
        self.norm2 = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon * 3)

    @staticmethod
    def crop_causal(x: torch.Tensor) -> torch.Tensor:
        return x[..., :-2]

    def forward(self, values: torch.Tensor, mask: torch.Tensor, graph: torch.Tensor | None = None) -> torch.Tensor:
        if graph is None:
            raise ValueError("STGCN requires a graph")
        observed = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        x = torch.cat((torch.nan_to_num(values, nan=0.0).clamp(-10, 10), observed), dim=-1).permute(0, 3, 2, 1)
        x = self.dropout(F.gelu(self.norm1(self.crop_causal(self.temporal_in(x)))))
        neighbor = torch.einsum("ij,bcjt->bcit", graph, x)
        x = x + self.graph_projection(neighbor)
        x = self.dropout(F.gelu(self.norm2(self.crop_causal(self.temporal_out(x)))))
        state = x[..., -1].permute(0, 2, 1)
        return self.head(state).view(values.shape[0], values.shape[2], self.horizon, 3).permute(0, 2, 1, 3)


def build_model(config: dict, nodes: int) -> nn.Module:
    name = str(config["model"]).lower()
    if name == "dlinear":
        return DLinear(int(config["context"]), int(config["horizon"]), int(config.get("moving_average", 25)))
    if name == "nlinear":
        return NLinear(int(config["context"]), int(config["horizon"]))
    if name == "ridge_ar":
        return RidgeAR(nodes, int(config["context"]), int(config["horizon"]))
    if name == "tcn":
        return TCN(int(config["hidden"]), int(config["horizon"]), float(config.get("dropout", 0.1)))
    if name == "lstm":
        return LSTM(int(config["hidden"]), int(config["horizon"]), float(config.get("dropout", 0.1)))
    if name == "patchtst":
        return PatchTST(int(config["context"]), int(config["horizon"]), int(config.get("patch_length", 12)), int(config.get("patch_stride", 6)), int(config.get("d_model", 64)), int(config.get("heads", 4)), int(config.get("layers", 2)), float(config.get("dropout", 0.1)))
    if name == "stgcn":
        return STGCN(int(config["hidden"]), int(config["horizon"]), float(config.get("dropout", 0.1)))
    raise ValueError(f"unsupported model={name}")


def run_epoch(model: nn.Module, loader: DataLoader, graph: torch.Tensor | None, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for values, mask, future, future_mask in loader:
            values, mask, future, future_mask = (item.to(device) for item in (values, mask, future, future_mask))
            prediction = model(values, mask, graph)
            loss = masked_huber(prediction, future, future_mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dlinear", "nlinear", "ridge_ar", "lstm", "tcn", "patchtst", "stgcn"), required=True)
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=60); parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=48); parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--moving-average", type=int, default=25)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--patch-length", type=int, default=12); parser.add_argument("--patch-stride", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64); parser.add_argument("--heads", type=int, default=4); parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8); parser.add_argument("--ewma-span", type=int, default=31); parser.add_argument("--correlation-weight", type=float, default=0.75)
    parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--station-indices", default=""); parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source = np.load(DATA)
    raw = source["residual_enu_mm"].astype(np.float32)
    mask, split = source["observed_mask"].astype(bool), source["split"]
    station_indices = np.array([int(value) for value in args.station_indices.split(",") if value.strip()], dtype=int) if args.station_indices else np.arange(raw.shape[1])
    raw, mask = raw[:, station_indices], mask[:, station_indices]
    target = source["normalized_residual_enu"][:, station_indices].astype(np.float32)
    target[~mask] = np.nan
    args.station_indices = station_indices.tolist()
    train = DirectWindows(target, mask, split, 0, args.context, args.horizon, args.max_windows)
    validation = DirectWindows(target, mask, split, 1, args.context, args.horizon, args.max_windows)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    graph = None
    if args.model == "stgcn":
        _, high_raw = causal_ewma(raw, mask, span_days=args.ewma_span)
        graph = correlation_distance_graph(high_raw, mask, split, args.neighbors, station_indices, "hybrid", args.correlation_weight).to(device)
    model = build_model(vars(args), raw.shape[1]).to(device)
    out = ROOT / "experiments" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    best, history = math.inf, []
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()) + sum(buffer.numel() for buffer in model.buffers()))
    if args.model == "ridge_ar":
        fit_ridge_ar(model, train, args.ridge_alpha)
        validation_loss = run_epoch(model, DataLoader(validation, batch_size=args.batch_size), graph, None, device)
        best = validation_loss
        history.append({"epoch": 0, "train_loss": None, "validation_loss": validation_loss, "epoch_seconds": 0.0, "fit": "closed_form_ridge"})
        torch.save({"state_dict": model.state_dict(), "config": vars(args), "best_validation_loss": best}, out / "best.pt")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = run_epoch(model, DataLoader(train, batch_size=args.batch_size, shuffle=True), graph, optimizer, device)
            validation_loss = run_epoch(model, DataLoader(validation, batch_size=args.batch_size), graph, None, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "epoch_seconds": time.perf_counter() - started}
            history.append(record)
            if validation_loss < best:
                best = validation_loss
                torch.save({"state_dict": model.state_dict(), "config": vars(args), "best_validation_loss": best}, out / "best.pt")
            print(json.dumps(record), flush=True)
    payload = {"device": str(device), "model": args.model, "parameter_count": parameter_count, "train_windows": len(train), "validation_windows": len(validation), "best_validation_loss": best, "history": history}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
